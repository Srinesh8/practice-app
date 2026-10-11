"""
Central place to configure which embedding model and LLM the RAG pipeline uses.

Settings resolve in this order (highest precedence first):
  1. `secrets/settings.json` (written by the Streamlit UI via save_settings())
  2. Environment variables / `.env`
  3. Hardcoded defaults below

This lets the app be configured either by editing `.env` once, or live from
the Streamlit sidebar (credentials there are masked and persisted to the
gitignored `secrets/` folder, never committed).
"""

import json
import os

from dotenv import load_dotenv

load_dotenv()

SETTINGS_PATH = os.getenv("SETTINGS_PATH", "secrets/settings.json")

DEFAULTS = {
    "embedding_provider": "ollama",
    "embedding_model": "nomic-embed-text",
    "llm_provider": "ollama",
    "llm_model": "gpt-oss:120b-cloud",
    "llm_temperature": 0.0,
    "ollama_base_url": "http://localhost:11434",
    "openai_api_key": "",
    "google_api_key": "",
    "webshare_proxy_username": "",
    "webshare_proxy_password": "",
    "youtube_cookies_path": "",
}

# Maps each settings key to the env var that can override its default.
ENV_VARS = {
    "embedding_provider": "EMBEDDING_PROVIDER",
    "embedding_model": "EMBEDDING_MODEL",
    "llm_provider": "LLM_PROVIDER",
    "llm_model": "LLM_MODEL",
    "llm_temperature": "LLM_TEMPERATURE",
    "ollama_base_url": "OLLAMA_BASE_URL",
    "openai_api_key": "OPENAI_API_KEY",
    "google_api_key": "GOOGLE_API_KEY",
    "webshare_proxy_username": "WEBSHARE_PROXY_USERNAME",
    "webshare_proxy_password": "WEBSHARE_PROXY_PASSWORD",
    "youtube_cookies_path": "YOUTUBE_COOKIES_PATH",
}

# Keys whose values should be masked when displayed back (e.g. in the UI).
SECRET_KEYS = {"openai_api_key", "google_api_key", "webshare_proxy_password"}


def load_settings() -> dict:
    """Return the current effective settings (file > env > defaults)."""
    settings = dict(DEFAULTS)

    for key, env_name in ENV_VARS.items():
        value = os.getenv(env_name)
        if value:
            settings[key] = value

    if os.path.exists(SETTINGS_PATH):
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            file_settings = json.load(f)
        settings.update({k: v for k, v in file_settings.items() if v not in (None, "")})

    settings["llm_temperature"] = float(settings["llm_temperature"])
    return settings


def save_settings(updates: dict) -> None:
    """Persist updates to SETTINGS_PATH, merged with whatever is already there."""
    os.makedirs(os.path.dirname(SETTINGS_PATH) or ".", exist_ok=True)

    current = {}
    if os.path.exists(SETTINGS_PATH):
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            current = json.load(f)

    current.update(updates)

    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(current, f, indent=2)
    os.chmod(SETTINGS_PATH, 0o600)


def mask(value: str) -> str:
    """Mask a secret for display, keeping only the last 4 characters visible."""
    if not value:
        return ""
    if len(value) <= 4:
        return "*" * len(value)
    return "*" * (len(value) - 4) + value[-4:]


# --- Chunking / retrieval / storage: static pipeline config, not UI-editable ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
RETRIEVER_TOP_K = int(os.getenv("RETRIEVER_TOP_K", "4"))
TRANSCRIPTS_DIR = os.getenv("TRANSCRIPTS_DIR", "transcripts")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "youtube_transcripts")
COOKIES_UPLOAD_PATH = os.getenv("COOKIES_UPLOAD_PATH", "secrets/cookies.txt")
DEFAULT_VIDEO_IDS = [v for v in os.getenv("DEFAULT_VIDEO_IDS", "zjkBMFhNj_g").split(",") if v]


def _ollama_embeddings(s: dict):
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(model=s["embedding_model"], base_url=s["ollama_base_url"])


def _openai_embeddings(s: dict):
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=s["embedding_model"], api_key=s["openai_api_key"] or None)


def _ollama_llm(s: dict):
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=s["llm_model"], base_url=s["ollama_base_url"], temperature=s["llm_temperature"]
    )


def _openai_llm(s: dict):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=s["llm_model"], temperature=s["llm_temperature"], api_key=s["openai_api_key"] or None
    )


def _gemini_llm(s: dict):
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=s["llm_model"],
        temperature=s["llm_temperature"],
        google_api_key=s["google_api_key"] or None,
    )


# Single source of truth for which providers exist. Add a new provider by
# adding one entry here — get_llm()/get_embeddings() and the Streamlit
# dropdowns (built from list(...)) both pick it up automatically.
EMBEDDING_PROVIDERS = {
    "ollama": _ollama_embeddings,
    "openai": _openai_embeddings,
}

LLM_PROVIDERS = {
    "ollama": _ollama_llm,
    "openai": _openai_llm,
    "gemini": _gemini_llm,
}


def get_embeddings():
    """Return a LangChain embeddings instance for the configured provider."""
    settings = load_settings()
    provider = settings["embedding_provider"]
    if provider not in EMBEDDING_PROVIDERS:
        raise ValueError(f"Unknown embedding_provider: {provider!r}")
    return EMBEDDING_PROVIDERS[provider](settings)


def get_llm():
    """Return a LangChain chat model instance for the configured provider."""
    settings = load_settings()
    provider = settings["llm_provider"]
    if provider not in LLM_PROVIDERS:
        raise ValueError(f"Unknown llm_provider: {provider!r}")
    return LLM_PROVIDERS[provider](settings)


def get_transcript_api():
    """Return a YouTubeTranscriptApi instance, using cookies/proxy if configured."""
    from youtube_transcript_api import YouTubeTranscriptApi

    settings = load_settings()
    kwargs = {}

    cookies_path = settings["youtube_cookies_path"]
    if cookies_path and os.path.exists(cookies_path):
        import http.cookiejar

        import requests

        jar = http.cookiejar.MozillaCookieJar(cookies_path)
        jar.load(ignore_discard=True, ignore_expires=True)
        session = requests.Session()
        session.cookies = jar
        kwargs["http_client"] = session

    if settings["webshare_proxy_username"] and settings["webshare_proxy_password"]:
        from youtube_transcript_api.proxies import WebshareProxyConfig

        kwargs["proxy_config"] = WebshareProxyConfig(
            proxy_username=settings["webshare_proxy_username"],
            proxy_password=settings["webshare_proxy_password"],
        )

    return YouTubeTranscriptApi(**kwargs)
