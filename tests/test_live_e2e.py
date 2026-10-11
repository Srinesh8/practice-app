"""Live end-to-end tests: real YouTube Transcript API + real Ollama.

Skipped automatically when Ollama isn't reachable or the required models
aren't pulled, since these hit real external services and are not expected
to run in CI or on a machine without Ollama set up.
"""

import shutil

import pytest
import requests

import model_config as cfg
import rag

LIVE_VIDEO_ID = "zjkBMFhNj_g"  # Andrej Karpathy - "Intro to Large Language Models"


def _ollama_ready() -> bool:
    settings = cfg.load_settings()
    if settings["llm_provider"] != "ollama" or settings["embedding_provider"] != "ollama":
        return False
    if shutil.which("ollama") is None:
        return False
    try:
        resp = requests.get(f"{settings['ollama_base_url']}/api/tags", timeout=2)
        resp.raise_for_status()
    except requests.RequestException:
        return False

    # Ollama's /api/tags returns tags with an implicit ":latest" suffix when
    # none was pulled explicitly, so compare by base name (before the colon).
    models = {m["name"].split(":")[0] for m in resp.json().get("models", [])}
    return (
        settings["embedding_model"].split(":")[0] in models
        and settings["llm_model"].split(":")[0] in models
    )


skip_reason = "Ollama not installed/running, or required models not pulled"


@pytest.mark.live
@pytest.mark.skipif(not _ollama_ready(), reason=skip_reason)
def test_live_fetch_index_and_answer(tmp_path, monkeypatch):
    monkeypatch.setattr(rag.cfg, "TRANSCRIPTS_DIR", str(tmp_path / "transcripts"))
    monkeypatch.setattr(rag.cfg, "CHROMA_PERSIST_DIR", str(tmp_path / "chroma_db"))
    monkeypatch.setattr(rag.cfg, "CHROMA_COLLECTION_NAME", "live_test_collection")

    available = rag.fetch_transcripts([LIVE_VIDEO_ID])
    assert available == [LIVE_VIDEO_ID], "transcript fetch failed - video may lack captions"

    vectorstore = rag.get_or_build_vectorstore(available)
    qa_chain = rag.build_qa_chain(vectorstore)

    result = rag.answer_question(qa_chain, "What is this video about?")
    assert result["answer"].strip() != ""
    assert result["sources"]

    off_topic = rag.answer_question(qa_chain, "What is the boiling point of mercury in Fahrenheit?")
    assert "don't know" in off_topic["answer"].lower()

    capability_request = rag.answer_question(qa_chain, "can you sing a song?")
    capability_answer = capability_request["answer"].lower()
    assert capability_answer.startswith("i don't know")
    assert "i can only answer questions about" in capability_answer

    scope_question = rag.answer_question(qa_chain, "what can you do?")
    scope_answer = scope_question["answer"].lower()
    assert not scope_answer.startswith("i don't know")
