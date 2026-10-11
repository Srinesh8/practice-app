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


_SCOPE_QUESTION_PHRASES = (
    "what can you do",
    "what can you help",
    "your capabilities",
    "what do you know",
    "what topics",
    "what videos",
    "podcasts you know",
)


class FakeLLM:
    """Offline stand-in for ChatOllama/ChatOpenAI/etc.

    Returns a canned answer when the question text is found verbatim in the
    prompt's context. Otherwise mimics the real prompt's two distinct shapes:
    a direct, confident answer for general scope questions ("what can you
    do?"), or an "I don't know, here's what I *can* help with" refusal for
    everything else not covered by the context.
    """

    def __init__(self, answer_for_known="The sky is blue because of Rayleigh scattering."):
        self.answer_for_known = answer_for_known
        self.last_messages = None

    def invoke(self, messages):
        self.last_messages = messages
        system_content = messages[0].content if messages else ""
        question = messages[1].content if len(messages) > 1 else ""

        if "Rayleigh scattering" in system_content:
            return FakeLLMResponse(self.answer_for_known)

        titles = []
        if "Indexed videos:" in system_content:
            block = system_content.split("Indexed videos:", 1)[1]
            block = block.split("Retrieved transcript excerpts:", 1)[0]
            for line in block.strip().splitlines():
                if line.strip().startswith("-"):
                    titles.append(line.split(":", 1)[-1].strip())

        titles_text = "; ".join(titles) if titles else "nothing yet"

        is_scope_question = any(phrase in question.lower() for phrase in _SCOPE_QUESTION_PHRASES)
        if is_scope_question:
            return FakeLLMResponse(f"I can answer questions about: {titles_text}.")

        return FakeLLMResponse(
            f"I don't know. I'm not able to help with that — I can only answer "
            f"questions about: {titles_text}."
        )


@pytest.fixture
def fake_embeddings():
    return FakeEmbeddings()


@pytest.fixture
def fake_llm():
    return FakeLLM()
