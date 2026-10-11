"""Integration tests: real Chroma + text splitter + chain wiring, with fake
embeddings/LLM standing in for the network-dependent Ollama calls."""

import pytest

import rag


@pytest.mark.integration
def test_full_pipeline_answers_from_indexed_transcript(tmp_path, monkeypatch, fake_embeddings, fake_llm):
    monkeypatch.setattr(rag.cfg, "TRANSCRIPTS_DIR", str(tmp_path / "transcripts"))
    monkeypatch.setattr(rag.cfg, "CHROMA_PERSIST_DIR", str(tmp_path / "chroma_db"))
    monkeypatch.setattr(rag.cfg, "CHROMA_COLLECTION_NAME", "test_collection")
    monkeypatch.setattr(rag.cfg, "CHUNK_SIZE", 200)
    monkeypatch.setattr(rag.cfg, "CHUNK_OVERLAP", 0)
    monkeypatch.setattr(rag.cfg, "RETRIEVER_TOP_K", 2)
    monkeypatch.setattr(rag.cfg, "get_embeddings", lambda: fake_embeddings)
    monkeypatch.setattr(rag.cfg, "get_llm", lambda: fake_llm)

    (tmp_path / "transcripts").mkdir()
    (tmp_path / "transcripts" / "vid1.txt").write_text(
        "The sky is blue because of Rayleigh scattering of sunlight in the atmosphere."
    )

    vectorstore = rag.get_or_build_vectorstore(["vid1"])
    qa_chain = rag.build_qa_chain(vectorstore)

    result = rag.answer_question(qa_chain, "Why is the sky blue?")

    assert "Rayleigh scattering" in result["answer"]
    assert result["sources"] == ["vid1#chunk0"]


@pytest.mark.integration
def test_pipeline_answers_i_dont_know_when_out_of_context(tmp_path, monkeypatch, fake_embeddings, fake_llm):
    monkeypatch.setattr(rag.cfg, "TRANSCRIPTS_DIR", str(tmp_path / "transcripts"))
    monkeypatch.setattr(rag.cfg, "CHROMA_PERSIST_DIR", str(tmp_path / "chroma_db"))
    monkeypatch.setattr(rag.cfg, "CHROMA_COLLECTION_NAME", "test_collection_2")
    monkeypatch.setattr(rag.cfg, "CHUNK_SIZE", 200)
    monkeypatch.setattr(rag.cfg, "CHUNK_OVERLAP", 0)
    monkeypatch.setattr(rag.cfg, "RETRIEVER_TOP_K", 2)
    monkeypatch.setattr(rag.cfg, "get_embeddings", lambda: fake_embeddings)
    monkeypatch.setattr(rag.cfg, "get_llm", lambda: fake_llm)

    (tmp_path / "transcripts").mkdir()
    (tmp_path / "transcripts" / "vid1.txt").write_text(
        "This video is about baking sourdough bread at home."
    )

    vectorstore = rag.get_or_build_vectorstore(["vid1"])
    qa_chain = rag.build_qa_chain(vectorstore)

    result = rag.answer_question(qa_chain, "What is the capital of France?")

    answer_lower = result["answer"].lower()
    assert answer_lower.startswith("i don't know")
    assert "i can only answer questions about" in answer_lower
    assert "vid1" in result["answer"]  # names what it *can* help with, not just a bare refusal


@pytest.mark.integration
def test_pipeline_refuses_capability_requests_with_helpful_message(
    tmp_path, monkeypatch, fake_embeddings, fake_llm
):
    """E.g. 'can you sing a song?' must not get a bare 'I don't know.' — it
    should say it can't help with that and name what it *can* answer about."""
    monkeypatch.setattr(rag.cfg, "TRANSCRIPTS_DIR", str(tmp_path / "transcripts"))
    monkeypatch.setattr(rag.cfg, "CHROMA_PERSIST_DIR", str(tmp_path / "chroma_db"))
    monkeypatch.setattr(rag.cfg, "CHROMA_COLLECTION_NAME", "test_collection_5")
    monkeypatch.setattr(rag.cfg, "CHUNK_SIZE", 200)
    monkeypatch.setattr(rag.cfg, "CHUNK_OVERLAP", 0)
    monkeypatch.setattr(rag.cfg, "RETRIEVER_TOP_K", 2)
    monkeypatch.setattr(rag.cfg, "get_embeddings", lambda: fake_embeddings)
    monkeypatch.setattr(rag.cfg, "get_llm", lambda: fake_llm)
    monkeypatch.setattr(rag, "fetch_video_title", lambda video_id: "Intro to Large Language Models")

    (tmp_path / "transcripts").mkdir()
    (tmp_path / "transcripts" / "vid1.txt").write_text(
        "This video is about large language models and how they are trained."
    )

    vectorstore = rag.get_or_build_vectorstore(["vid1"])
    qa_chain = rag.build_qa_chain(vectorstore)

    for question in ["can you sing a song?", "write me a poem about the ocean"]:
        result = rag.answer_question(qa_chain, question)
        answer_lower = result["answer"].lower()
        assert answer_lower.startswith("i don't know")
        assert "i can only answer questions about" in answer_lower
        assert "large language models" in answer_lower


@pytest.mark.integration
def test_get_indexed_videos_backfills_title_for_videos_missing_from_cache(
    tmp_path, monkeypatch, fake_embeddings
):
    """Regression test: a video indexed before title-tracking existed (so it's
    missing from transcripts/_titles.json) must still show up with its title,
    derived from the vector store itself rather than the title cache file."""
    monkeypatch.setattr(rag.cfg, "TRANSCRIPTS_DIR", str(tmp_path / "transcripts"))
    monkeypatch.setattr(rag.cfg, "CHROMA_PERSIST_DIR", str(tmp_path / "chroma_db"))
    monkeypatch.setattr(rag.cfg, "CHROMA_COLLECTION_NAME", "test_collection_4")
    monkeypatch.setattr(rag.cfg, "CHUNK_SIZE", 200)
    monkeypatch.setattr(rag.cfg, "CHUNK_OVERLAP", 0)
    monkeypatch.setattr(rag.cfg, "get_embeddings", lambda: fake_embeddings)
    monkeypatch.setattr(rag, "fetch_video_title", lambda video_id: f"Title for {video_id}")

    (tmp_path / "transcripts").mkdir()
    (tmp_path / "transcripts" / "vid1.txt").write_text("some transcript text")

    # Index directly without ever calling fetch_transcripts, so no title gets
    # cached the "normal" way - simulates data indexed before this feature existed.
    vectorstore = rag.get_or_build_vectorstore(["vid1"])

    assert rag.get_indexed_videos(vectorstore) == {"vid1": "Title for vid1"}
    # And it's now cached for next time.
    assert rag.get_indexed_videos() == {"vid1": "Title for vid1"}


@pytest.mark.integration
def test_get_or_build_vectorstore_skips_reindexing_existing_video(tmp_path, monkeypatch, fake_embeddings):
    monkeypatch.setattr(rag.cfg, "TRANSCRIPTS_DIR", str(tmp_path / "transcripts"))
    monkeypatch.setattr(rag.cfg, "CHROMA_PERSIST_DIR", str(tmp_path / "chroma_db"))
    monkeypatch.setattr(rag.cfg, "CHROMA_COLLECTION_NAME", "test_collection_3")
    monkeypatch.setattr(rag.cfg, "CHUNK_SIZE", 200)
    monkeypatch.setattr(rag.cfg, "CHUNK_OVERLAP", 0)
    monkeypatch.setattr(rag.cfg, "get_embeddings", lambda: fake_embeddings)

    (tmp_path / "transcripts").mkdir()
    (tmp_path / "transcripts" / "vid1.txt").write_text("some transcript text about topic A")

    vectorstore1 = rag.get_or_build_vectorstore(["vid1"])
    count_after_first = len(vectorstore1.get()["ids"])

    vectorstore2 = rag.get_or_build_vectorstore(["vid1"])
    count_after_second = len(vectorstore2.get()["ids"])

    assert count_after_first == count_after_second
    assert count_after_first > 0
