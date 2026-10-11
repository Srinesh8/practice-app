"""Streamlit chat UI for the YouTube transcript RAG chatbot."""

import os

import streamlit as st

import model_config as cfg
from rag import (
    answer_question,
    build_qa_chain,
    extract_video_id,
    fetch_transcripts,
    get_or_build_vectorstore,
)

st.set_page_config(page_title="YouTube RAG Chatbot", page_icon="🎥")
st.title("🎥 YouTube Transcript RAG Chatbot")
st.caption("Answers only from the indexed video transcripts. Says \"I don't know\" otherwise.")

with st.sidebar:
    with st.expander("⚙️ Model & credentials settings"):
        settings = cfg.load_settings()

        st.caption("Model selection")
        llm_provider_options = list(cfg.LLM_PROVIDERS)
        llm_provider = st.selectbox(
            "LLM provider",
            options=llm_provider_options,
            index=llm_provider_options.index(settings["llm_provider"]),
        )
        llm_model = st.text_input("LLM model", value=settings["llm_model"])
        llm_temperature = st.slider(
            "LLM temperature", min_value=0.0, max_value=1.0, value=settings["llm_temperature"]
        )

        embedding_provider_options = list(cfg.EMBEDDING_PROVIDERS)
        embedding_provider = st.selectbox(
            "Embedding provider",
            options=embedding_provider_options,
            index=embedding_provider_options.index(settings["embedding_provider"]),
        )
        embedding_model = st.text_input("Embedding model", value=settings["embedding_model"])

        ollama_base_url = st.text_input("Ollama base URL", value=settings["ollama_base_url"])

        st.caption("Credentials (masked, stored locally in `secrets/`, never committed)")
        openai_required = llm_provider == "openai" or embedding_provider == "openai"
        gemini_required = llm_provider == "gemini"

        openai_api_key = st.text_input(
            "OpenAI API key" + (" *(required)*" if openai_required else ""),
            value="",
            type="password",
            placeholder=cfg.mask(settings["openai_api_key"]) or "not set",
        )
        google_api_key = st.text_input(
            "Google (Gemini) API key" + (" *(required)*" if gemini_required else ""),
            value="",
            type="password",
            placeholder=cfg.mask(settings["google_api_key"]) or "not set",
        )
        webshare_username = st.text_input(
            "Webshare proxy username", value=settings["webshare_proxy_username"]
        )
        webshare_password = st.text_input(
            "Webshare proxy password",
            value="",
            type="password",
            placeholder=cfg.mask(settings["webshare_proxy_password"]) or "not set",
        )

        st.caption("YouTube cookies (fixes transcript downloads being blocked)")
        cookies_file = st.file_uploader("Upload cookies.txt", type=["txt"])

        if st.button("Save settings"):
            has_openai_key = bool(openai_api_key or settings["openai_api_key"])
            has_google_key = bool(google_api_key or settings["google_api_key"])

            missing = []
            if openai_required and not has_openai_key:
                missing.append("OpenAI API key (required for the OpenAI provider you selected)")
            if gemini_required and not has_google_key:
                missing.append("Google (Gemini) API key (required for the Gemini provider you selected)")

            if missing:
                for msg in missing:
                    st.error(msg)
            else:
                updates = {
                    "llm_provider": llm_provider,
                    "llm_model": llm_model,
                    "llm_temperature": llm_temperature,
                    "embedding_provider": embedding_provider,
                    "embedding_model": embedding_model,
                    "ollama_base_url": ollama_base_url,
                    "webshare_proxy_username": webshare_username,
                }
                # Only overwrite a secret if the user actually typed a new value;
                # otherwise keep whatever was already saved.
                if openai_api_key:
                    updates["openai_api_key"] = openai_api_key
                if google_api_key:
                    updates["google_api_key"] = google_api_key
                if webshare_password:
                    updates["webshare_proxy_password"] = webshare_password

                if cookies_file is not None:
                    os.makedirs(os.path.dirname(cfg.COOKIES_UPLOAD_PATH), exist_ok=True)
                    with open(cfg.COOKIES_UPLOAD_PATH, "wb") as f:
                        f.write(cookies_file.getbuffer())
                    os.chmod(cfg.COOKIES_UPLOAD_PATH, 0o600)
                    updates["youtube_cookies_path"] = cfg.COOKIES_UPLOAD_PATH

                cfg.save_settings(updates)

                # Force the QA chain/embeddings to be rebuilt with the new settings.
                st.session_state.pop("qa_chain", None)
                st.session_state.pop("vectorstore", None)
                st.success("Settings saved.")
                st.rerun()

    st.header("Indexed videos")
    video_ids_input = st.text_area(
        "YouTube video IDs or URLs (one per line)",
        value="\n".join(cfg.DEFAULT_VIDEO_IDS),
        height=100,
    )
    if st.button("Fetch + Index"):
        video_ids = [
            extract_video_id(v) for v in video_ids_input.splitlines() if v.strip()
        ]
        with st.spinner("Fetching transcripts and updating vector store..."):
            available = fetch_transcripts(video_ids)
            st.session_state.vectorstore = get_or_build_vectorstore(available)
            st.session_state.qa_chain = build_qa_chain(st.session_state.vectorstore)
        st.success(f"Indexed: {available}")

def _friendly_provider_error(exc: Exception) -> str:
    settings = cfg.load_settings()
    return (
        f"Could not reach the configured model ({settings['llm_provider']} / "
        f"{settings['llm_model']}): **{exc}**\n\n"
        "If you switched to OpenAI or Gemini, make sure you entered and saved "
        "the matching API key in **⚙️ Model & credentials settings** first."
    )


if "vectorstore" not in st.session_state:
    video_ids = [extract_video_id(v) for v in cfg.DEFAULT_VIDEO_IDS]
    fetch_transcripts(video_ids)
    try:
        st.session_state.vectorstore = get_or_build_vectorstore(video_ids)
    except Exception as exc:
        st.error(_friendly_provider_error(exc))
        st.stop()

if "qa_chain" not in st.session_state:
    try:
        st.session_state.qa_chain = build_qa_chain(st.session_state.vectorstore)
    except Exception as exc:
        st.error(_friendly_provider_error(exc))
        st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("Ask a question about the indexed videos..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = answer_question(st.session_state.qa_chain, question)
                answer = result["answer"]
                if result["sources"] and not result["refused"]:
                    answer += "\n\n*Sources: " + ", ".join(result["sources"]) + "*"
            except Exception as exc:
                answer = _friendly_provider_error(exc)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
