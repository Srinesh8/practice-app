"""
Shared RAG pipeline: transcript ingestion, chunking, vector store, and the
question-answering chain. Used by both the notebook and app.py so there is a
single implementation of the pipeline.
"""

import json
import os

import requests
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

import model_config as cfg

SYSTEM_PROMPT = """You are a helpful assistant answering questions ONLY using \
the context below, which comes from YouTube video transcripts.

Rules:
- Answer using only the information in the context below.
- If the context does not contain the answer, respond exactly: "I don't know."
- Do not use outside knowledge, even if you know the answer.
- If asked what topics/videos you can answer questions about, answer using the
  "Indexed videos" list below — that list is itself valid context.
- You are summarizing what the transcripts say, not describing yourself. If a
  transcript speaker talks in first person (e.g. a video about "what I, an AI
  model, can do"), you must still report it in third person, attributed to the
  video (e.g. "The video explains that language models can..."). Never say
  "I can..." about any ability, tool, or capability unless it is this chatbot's
  own documented behavior (answering only from indexed transcripts).
- Write the answer as clear prose or a short bullet list, whichever fits the
  question. Do not include inline [source] tags in the answer text itself;
  sources are shown separately.

Indexed videos:
{video_list}

Retrieved transcript excerpts:
{context}
"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


def _titles_path() -> str:
    return os.path.join(cfg.TRANSCRIPTS_DIR, "_titles.json")


def _load_titles() -> dict:
    path = _titles_path()
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_title(video_id: str, title: str) -> None:
    titles = _load_titles()
    titles[video_id] = title
    with open(_titles_path(), "w", encoding="utf-8") as f:
        json.dump(titles, f, indent=2)


def fetch_video_title(video_id: str) -> str:
    """Best-effort video title lookup via YouTube's public oEmbed endpoint (no API key)."""
    try:
        resp = requests.get(
            "https://www.youtube.com/oembed",
            params={"url": f"https://www.youtube.com/watch?v={video_id}", "format": "json"},
            timeout=5,
        )
        resp.raise_for_status()
        return resp.json().get("title", video_id)
    except requests.RequestException:
        return video_id


def _video_ids_in_vectorstore(vectorstore: Chroma) -> set:
    existing = vectorstore.get()
    return {meta["video_id"] for meta in existing["metadatas"] if meta}


def get_indexed_videos(vectorstore: Chroma | None = None) -> dict:
    """Return {video_id: title} for videos actually searchable in the vector store.

    If vectorstore is omitted, falls back to whatever's in the title cache file.
    Any video found in the vector store but missing a cached title (e.g. it was
    indexed before title tracking existed) gets backfilled on the spot, so the
    list the LLM sees always matches what's actually retrievable.
    """
    titles = _load_titles()

    if vectorstore is None:
        return titles

    video_ids = _video_ids_in_vectorstore(vectorstore)
    for video_id in video_ids:
        if video_id not in titles:
            titles[video_id] = fetch_video_title(video_id)
            _save_title(video_id, titles[video_id])

    return {vid: titles[vid] for vid in video_ids}


def fetch_transcripts(video_ids: list[str]) -> list[str]:
    """Fetch and save transcripts for video_ids that aren't already saved.

    Returns the list of video_ids whose .txt transcript is available locally
    (either just-fetched or already present), skipping ones that fail.
    """
    os.makedirs(cfg.TRANSCRIPTS_DIR, exist_ok=True)
    available = []
    api = cfg.get_transcript_api()

    for video_id in video_ids:
        path = os.path.join(cfg.TRANSCRIPTS_DIR, f"{video_id}.txt")
        if os.path.exists(path):
            if video_id not in _load_titles():
                _save_title(video_id, fetch_video_title(video_id))
            available.append(video_id)
            continue

        try:
            transcript = api.fetch(video_id)
        except Exception as exc:
            print(f"[skip] could not fetch transcript for {video_id}: {exc}")
            continue

        text = " ".join(snippet.text for snippet in transcript)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

        _save_title(video_id, fetch_video_title(video_id))
        available.append(video_id)
        print(f"[ok] saved transcript for {video_id} ({len(text)} chars)")

    return available


def load_and_split_documents(video_ids: list[str]) -> list[Document]:
    """Load saved transcripts for video_ids and split them into chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.CHUNK_SIZE,
        chunk_overlap=cfg.CHUNK_OVERLAP,
    )

    documents = []
    for video_id in video_ids:
        path = os.path.join(cfg.TRANSCRIPTS_DIR, f"{video_id}.txt")
        with open(path, encoding="utf-8") as f:
            text = f.read()

        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "video_id": video_id,
                        "chunk_index": i,
                        "source": f"{video_id}#chunk{i}",
                    },
                )
            )

    return documents


def get_or_build_vectorstore(video_ids: list[str] | None = None) -> Chroma:
    """Return the persisted Chroma vectorstore, building/adding to it if needed.

    If the persisted collection already has data, it is reused as-is. Pass
    video_ids the first time (or whenever new videos should be indexed).
    """
    embeddings = cfg.get_embeddings()
    vectorstore = Chroma(
        collection_name=cfg.CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=cfg.CHROMA_PERSIST_DIR,
    )

    already_indexed = _video_ids_in_vectorstore(vectorstore)

    if video_ids:
        new_video_ids = [v for v in video_ids if v not in already_indexed]
        if new_video_ids:
            documents = load_and_split_documents(new_video_ids)
            vectorstore.add_documents(documents)
            print(f"[index] added {len(documents)} chunks for {new_video_ids}")

    return vectorstore


def build_qa_chain(vectorstore: Chroma):
    """Build a retriever + prompt + LLM chain that answers only from context."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": cfg.RETRIEVER_TOP_K})
    llm = cfg.get_llm()

    def answer(question: str) -> dict:
        docs = retriever.invoke(question)
        context = "\n\n".join(
            f"[{d.metadata['source']}] {d.page_content}" for d in docs
        )
        titles = get_indexed_videos(vectorstore)
        video_list = "\n".join(f"- {vid}: {title}" for vid, title in titles.items()) or "(none indexed yet)"

        messages = PROMPT.format_messages(
            video_list=video_list, context=context, question=question
        )
        response = llm.invoke(messages)
        return {
            "answer": response.content,
            "sources": [d.metadata["source"] for d in docs],
        }

    return answer


def answer_question(qa_chain, question: str) -> dict:
    """Convenience wrapper mirroring the function signature used in the notebook/app."""
    return qa_chain(question)
