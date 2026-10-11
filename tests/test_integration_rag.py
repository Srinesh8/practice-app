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

    assert result["answer"] == "I don't know."


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
