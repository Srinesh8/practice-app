# YouTube Transcript RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions **only** from the
transcripts of YouTube videos you index. If the answer isn't in the transcripts, it says
"I don't know" instead of guessing or using outside knowledge.

## What's in here

| File | Purpose |
|---|---|
| `rag.py` | Core pipeline: fetch transcripts, chunk, embed, store in Chroma, build the QA chain |
| `model_config.py` | All configurable settings (which LLM/embeddings to use, credentials, cookies) |
| `app.py` | Streamlit chat UI, including a settings panel to configure models/credentials from the browser |
| `rag_youtube_chatbot.ipynb` | Same pipeline, step by step, as a runnable notebook |
| `tests/` | Unit, integration, and live end-to-end tests |
| `requirements.txt` | Pinned Python dependencies |

## Why these tools

- **Ollama** (`nomic-embed-text` for embeddings, `gpt-oss:120b-cloud` for answering) — runs
  locally and free for embeddings; the cloud model needs an Ollama account but no separate
  API key. Other providers (OpenAI, Gemini) are supported too — see **Configuration** below.
- **Chroma** — a simple, file-persisted vector database with a clean Python API; good fit
  for a single-user/local project like this (vs. a server-based DB).
- **LangChain** (`langchain-ollama`, `langchain-openai`, `langchain-google-genai`,
  `langchain-chroma`, `langchain-text-splitters`) — gives one consistent interface across
  embedding/LLM providers and a ready-made recursive text splitter, so swapping providers
  doesn't mean swapping code.
- **youtube-transcript-api** — fetches YouTube captions directly without needing OAuth or
  the full YouTube Data API (we only need the transcript text, not video metadata).
- **Streamlit** — fastest way to get a chat UI with session state and a sidebar for
  settings, without writing any frontend code.
- **pytest** — unit tests mock all external calls; integration tests exercise the real
  Chroma + text-splitter wiring with fake embeddings/LLM; live tests hit the real YouTube
  and Ollama services and are skipped automatically if those aren't available.

## Setup

### 1. Install Python dependencies

**Linux / macOS / WSL:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Install Ollama (needed for the default local/cloud models)

**Linux / WSL:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows / macOS:** download the installer from [ollama.com/download](https://ollama.com/download)
and run it.

Then, from any terminal:
```bash
ollama pull nomic-embed-text      # local embedding model, free
ollama signin                     # opens a browser login for your Ollama account
ollama pull gpt-oss:120b-cloud    # cloud LLM, needs the signed-in account
ollama list                       # confirm both models show up
```

> If you'd rather use OpenAI or Gemini instead of Ollama, skip this step entirely — see
> **Configuration** below.

### 3. (Optional, but usually necessary) Set up YouTube cookies

YouTube blocks transcript downloads from many cloud/VPN/WSL IP addresses. If
`fetch_transcripts` fails with an XML parse error, you need to supply cookies from a
logged-in browser session:

1. Install the **"Get cookies.txt LOCALLY"** browser extension (Chrome Web Store / Firefox
   Add-ons).
2. Go to `youtube.com` while logged in, click the extension icon, and export `cookies.txt`.
3. Either:
   - Upload it directly in the Streamlit app's **⚙️ Model & credentials settings** panel, or
   - Save it to `secrets/cookies.txt` and set `YOUTUBE_COOKIES_PATH=secrets/cookies.txt` in
     a `.env` file.

**Never commit or paste cookie file contents anywhere** — they're live session credentials.
`secrets/` and `.env` are both gitignored.

### 4. Run it

**Notebook** (step-by-step, matches the original assignment's 7 steps):
```bash
jupyter notebook rag_youtube_chatbot.ipynb
```

**Streamlit app:**
```bash
streamlit run app.py
```
Then open the printed local URL (typically `http://localhost:8501`) in your browser.

### 5. Run the tests

```bash
pytest -m "not live"      # unit + integration (fully mocked, no network needed)
pytest -m live             # hits real YouTube + Ollama; auto-skips if not set up
pytest                     # everything
```

## Configuration

Everything is resolved in this order (highest priority first): the Streamlit UI's saved
settings (`secrets/settings.json`) → environment variables / `.env` → hardcoded defaults in
`model_config.py`.

| Setting | Env var | Default | Notes |
|---|---|---|---|
| LLM provider | `LLM_PROVIDER` | `ollama` | `ollama`, `openai`, or `gemini` |
| LLM model | `LLM_MODEL` | `gpt-oss:120b-cloud` | e.g. `gpt-4o-mini`, `gemini-1.5-flash` |
| LLM temperature | `LLM_TEMPERATURE` | `0.0` | |
| Embedding provider | `EMBEDDING_PROVIDER` | `ollama` | `ollama` or `openai` |
| Embedding model | `EMBEDDING_MODEL` | `nomic-embed-text` | |
| Ollama base URL | `OLLAMA_BASE_URL` | `http://localhost:11434` | point this at a remote Ollama host if needed |
| OpenAI API key | `OPENAI_API_KEY` | — | only needed if provider = `openai` |
| Google API key | `GOOGLE_API_KEY` | — | only needed if provider = `gemini` |
| Webshare proxy user/pass | `WEBSHARE_PROXY_USERNAME` / `_PASSWORD` | — | alternative to cookies for YouTube blocking |
| YouTube cookies path | `YOUTUBE_COOKIES_PATH` | — | see step 3 above |
| Chunk size / overlap | `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | not exposed in the UI |
| Retriever top-k | `RETRIEVER_TOP_K` | `4` | chunks retrieved per question |

**Switching providers**: in the UI, picking `openai`/`gemini` marks the matching API key
field **"(required)"**, and **Save settings** refuses to save (with a clear error) until
that key is filled in — so a misconfiguration is caught immediately, not discovered later
mid-chat. If a provider somehow still fails at answer time (e.g. an invalid key), the chat
shows a friendly in-app error instead of crashing.

## How the Streamlit UI works

- **⚙️ Model & credentials settings** (sidebar, collapsed by default) — change LLM/embedding
  provider and model, Ollama URL, temperature, and paste API keys or proxy credentials.
  Credential fields are masked (`type="password"`); previously saved secrets show only a
  `****1234`-style hint, never the full value. You can also upload `cookies.txt` directly
  here instead of editing `.env`. Clicking **Save settings** persists everything to
  `secrets/settings.json` (owner-only file permissions, gitignored) and rebuilds the chatbot
  with the new config.
- **Indexed videos** — paste one or more YouTube video IDs (just the ID, e.g. `NYOkZObSjoM`
  from `youtube.com/watch?v=NYOkZObSjoM` — not the whole URL) and click **Fetch + Index** to
  pull their transcripts and add them to the vector store.
- **Chat** — ask questions at the bottom. Answers are grounded only in retrieved transcript
  chunks, plus a live list of indexed video titles (derived straight from the vector store,
  so it can never drift out of sync — see **Answer behavior** below) so "what can you tell
  me about?" works too. Every answer lists which transcript chunks it came from, except
  "I don't know" answers, which skip the source list since there's nothing to cite.

## Answer behavior

The system prompt (in `rag.py`) enforces a few guarantees beyond "only answer from
context":

- **Strict grounding**: if the retrieved chunks don't contain the answer, it says exactly
  "I don't know" — it won't fall back to the model's general knowledge.
- **No self-impersonation**: some transcripts talk in first person about what "I, an AI
  model, can do" (e.g. a talk about LLM capabilities). Without guarding against it, the bot
  would parrot that framing and sound like it was claiming those abilities for itself. The
  prompt forces third-person attribution instead — it answers "The video explains that
  language models can..." rather than "I can...". This was caught and fixed after testing
  with a video that discusses AI capabilities.
- **Meta-questions**: "what topics/videos can you answer about?" is answered from a live
  "indexed videos" list injected into every prompt (backed directly by the vector store's
  contents, not a side cache), so it always matches what's actually searchable.

## Example questions to try

Using the defaults (Andrej Karpathy's "Intro to Large Language Models" + any podcast you
index):

- **In-context**: "What is a large language model?", "What does the video say about
  tokenization?"
- **Meta / "what do you know"**: "What topics or videos can you answer questions about?"
- **Third-person attribution check**: "What are your capabilities?" (if the video discusses
  AI capabilities, the answer should say "The video explains that..." — never "I can...")
- **Out-of-context** (should answer "I don't know"): "What's the weather today?", "Who won
  the last cricket world cup?"

## Local deployment (what we actually did)

This app is designed to run **locally**, since the default LLM path depends on Ollama
running on the same machine (or one reachable via `OLLAMA_BASE_URL`). What we set up:

1. Installed Ollama locally, pulled `nomic-embed-text` and signed in for the cloud
   `gpt-oss:120b-cloud` model.
2. Ran `streamlit run app.py` directly on the dev machine — Streamlit serves the app on
   `localhost:8501`.
3. To let someone test it from outside the machine without deploying to a cloud platform,
   we exposed the local Streamlit port through a **Cloudflare quick tunnel**:
   ```bash
   cloudflared tunnel --url http://localhost:8501
   ```
   This prints a temporary public `https://*.trycloudflare.com` URL that proxies straight
   to the local app — no account needed, no code changes, and it still uses the local
   Ollama install, local Chroma index, and local cookies file. The tunnel is throwaway: close
   the `cloudflared` process and the URL stops working.

If you wanted a permanently-hosted version instead (e.g. Streamlit Community Cloud), you'd
need to switch `LLM_PROVIDER`/`EMBEDDING_PROVIDER` to `openai` or `gemini`, since those
managed platforms can't run the Ollama server process — a straightforward settings change,
no code change, thanks to the provider abstraction in `model_config.py`.
