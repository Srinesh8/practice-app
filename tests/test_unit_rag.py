"""Unit tests: each rag.py function in isolation, all external calls mocked."""

from unittest.mock import MagicMock

import rag


class _FakeSnippet:
    def __init__(self, text):
        self.text = text


def test_fetch_transcripts_skips_already_saved(tmp_path, monkeypatch):
    monkeypatch.setattr(rag.cfg, "TRANSCRIPTS_DIR", str(tmp_path))
    monkeypatch.setattr(rag, "fetch_video_title", lambda video_id: "Fake Title")
    existing = tmp_path / "abc123.txt"
    existing.write_text("already here")

    fake_api = MagicMock()
    monkeypatch.setattr(rag.cfg, "get_transcript_api", lambda: fake_api)

    result = rag.fetch_transcripts(["abc123"])

    fake_api.fetch.assert_not_called()
    assert result == ["abc123"]


def test_fetch_transcripts_saves_new_video(tmp_path, monkeypatch):
    monkeypatch.setattr(rag.cfg, "TRANSCRIPTS_DIR", str(tmp_path))
    monkeypatch.setattr(rag, "fetch_video_title", lambda video_id: "Fake Title")

    fake_api = MagicMock()
    fake_api.fetch.return_value = [_FakeSnippet("hello"), _FakeSnippet("world")]
    monkeypatch.setattr(rag.cfg, "get_transcript_api", lambda: fake_api)

    result = rag.fetch_transcripts(["vid1"])

    assert result == ["vid1"]
    saved = (tmp_path / "vid1.txt").read_text()
    assert saved == "hello world"


def test_fetch_transcripts_persists_title_for_indexed_videos(tmp_path, monkeypatch):
    monkeypatch.setattr(rag.cfg, "TRANSCRIPTS_DIR", str(tmp_path))
    monkeypatch.setattr(rag, "fetch_video_title", lambda video_id: f"Title for {video_id}")

    fake_api = MagicMock()
    fake_api.fetch.return_value = [_FakeSnippet("hello")]
    monkeypatch.setattr(rag.cfg, "get_transcript_api", lambda: fake_api)

    rag.fetch_transcripts(["vid1"])

    assert rag.get_indexed_videos() == {"vid1": "Title for vid1"}


def test_fetch_transcripts_skips_on_api_error(tmp_path, monkeypatch):
    monkeypatch.setattr(rag.cfg, "TRANSCRIPTS_DIR", str(tmp_path))

    fake_api = MagicMock()
    fake_api.fetch.side_effect = Exception("transcripts disabled")
    monkeypatch.setattr(rag.cfg, "get_transcript_api", lambda: fake_api)

    result = rag.fetch_transcripts(["bad_vid"])

    assert result == []
    assert not (tmp_path / "bad_vid.txt").exists()


def test_load_and_split_documents_sets_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(rag.cfg, "TRANSCRIPTS_DIR", str(tmp_path))
    monkeypatch.setattr(rag.cfg, "CHUNK_SIZE", 20)
    monkeypatch.setattr(rag.cfg, "CHUNK_OVERLAP", 0)
    (tmp_path / "vid1.txt").write_text("word " * 50)

    documents = rag.load_and_split_documents(["vid1"])

    assert len(documents) > 1
    for i, doc in enumerate(documents):
        assert doc.metadata["video_id"] == "vid1"
        assert doc.metadata["chunk_index"] == i
        assert doc.metadata["source"] == f"vid1#chunk{i}"


def test_build_qa_chain_returns_answer_and_sources(monkeypatch):
    doc = MagicMock()
    doc.page_content = "some context"
    doc.metadata = {"source": "vid1#chunk0"}

    retriever = MagicMock()
    retriever.invoke.return_value = [doc]

    vectorstore = MagicMock()
    vectorstore.as_retriever.return_value = retriever

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = MagicMock(content="the answer")
    monkeypatch.setattr(rag.cfg, "get_llm", lambda: fake_llm)

    qa_chain = rag.build_qa_chain(vectorstore)
    result = rag.answer_question(qa_chain, "a question?")

    assert result == {"answer": "the answer", "sources": ["vid1#chunk0"]}
    retriever.invoke.assert_called_once_with("a question?")
