import hashlib

import pytest


class FakeEmbeddings:
    """Deterministic, offline stand-in for a real embeddings model.

    Hashes text into a fixed-size vector so that same/similar text yields
    close vectors, without ever calling out to Ollama/OpenAI/etc.
    """

    def _embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255.0 for b in digest[:16]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class FakeLLMResponse:
    def __init__(self, content: str):
        self.content = content


class FakeLLM:
    """Offline stand-in for ChatOllama/ChatOpenAI/etc.

    Returns a canned answer when the question text is found verbatim in the
    prompt's context, otherwise returns the "I don't know" fallback, mimicking
    the behavior the real prompt is designed to elicit from a real LLM.
    """

    def __init__(self, answer_for_known="The sky is blue because of Rayleigh scattering."):
        self.answer_for_known = answer_for_known
        self.last_messages = None

    def invoke(self, messages):
        self.last_messages = messages
        context = messages[0].content if messages else ""
        if "Rayleigh scattering" in context:
            return FakeLLMResponse(self.answer_for_known)
        return FakeLLMResponse("I don't know.")


@pytest.fixture
def fake_embeddings():
    return FakeEmbeddings()


@pytest.fixture
def fake_llm():
    return FakeLLM()
