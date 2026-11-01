# MedBot: AI-Powered Medical Chatbot with Doctor Recommendation ⚕️

[![CI](https://github.com/Swapnil26-art/Doctor-Recommendation-System-RAG-chat-bot-/actions/workflows/ci.yml/badge.svg)](https://github.com/Swapnil26-art/Doctor-Recommendation-System-RAG-chat-bot-/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Flask-green.svg)](https://flask.palletsprojects.com/)
[![AI-Platform](https://img.shields.io/badge/AI_Platform-Google_Gemini-orange.svg)](https://ai.google.dev/)
[![VectorDB](https://img.shields.io/badge/Vector_DB-Pinecone-4C5C6F.svg)](https://www.pinecone.io/)
[![Maps-API](https://img.shields.io/badge/Maps_API-Google_Places-red.svg)](https://developers.google.com/maps/documentation/places/web-service/overview)

## ✨ Overview

MedBot answers medical questions from a curated medical reference book using
Retrieval-Augmented Generation, and — when you ask for a doctor — finds real
clinics and hospitals near you using your browser's location.

Both capabilities live behind a single chat box: the app decides which route a
message needs, so the user never has to switch modes.

## 🚀 Features

- **Medical Q&A (RAG)** — the PDF is chunked and stored in Pinecone with the
  hosted `llama-text-embed-v2` model; Gemini answers using only the retrieved
  context and is instructed to admit when the book does not cover something.
- **Page citations** — every answer shows which page of the book it came from.
- **Relevance floor** — weak matches are discarded, so unrelated questions get
  "I couldn't find that" instead of a confident hallucination.
- **Doctor recommendation** — Google Places lookups ranked by true geodesic
  distance, permanently-closed businesses filtered out, results returned as
  structured data and rendered as safe cards.
- **Symptom → specialty routing** — "my tooth hurts" searches dentists, "chest
  pain" searches cardiologists, "nearest hospital" searches hospitals.
- **Safe rendering** — no `innerHTML` on untrusted data. Every API/LLM string is
  escaped before display, and map links are validated to `https://*.google.com`.
- **No local ML stack** — embeddings run inside Pinecone. No torch, no 2 GB
  download, no Python 3.13 ceiling.
- **Configurable, testable, container-ready** — settings in one module, Flask
  application factory, 98 tests, `Dockerfile`, `docker compose`, and CI.

## 🏗️ Architecture

```
Browser (templates/index.html)
   ├── geolocation ─────────────────┐
   └── POST /get { msg, location }  │
                                    ▼
                        Flask (app.py)
                         ├── places.is_doctor_request(msg)
                         │      ├── yes → places.find_nearby_places()
                         │      │          └── Google Places API + geopy
                         │      └── no  → rag.answer_question()
                         │                 ├── Pinecone search(inputs={"text": q})
                         │                 │     └── llama-text-embed-v2 (server-side)
                         │                 ├── relevance floor + page citations
                         │                 └── Gemini 3.8 Flash
                         └── JSON: { type: "text" | "places", ... }
```

### Data ingestion (one-off)

```
data/*.pdf → pypdf → per-page chunks (1000 chars / 150 overlap)
           → Pinecone upsert_records (text only)
           → Pinecone embeds server-side with llama-text-embed-v2
           → namespace "knowledge-base"
```

Because Pinecone computes the embeddings, `ingest.py` runs in seconds and needs
no model download.

## 🛠️ Tech stack

| Layer | Choice |
| --- | --- |
| Backend | Flask 3.1 |
| LLM | Google Gemini 3.8 Flash (`google-genai`) |
| Vector DB | Pinecone serverless, **integrated index** |
| Embeddings | `llama-text-embed-v2`, hosted by Pinecone (384 dims, cosine) |
| Retrieval | Pinecone text search (no local model) |
| Maps | Google Places API (Nearby Search) + `geopy` |
| PDF parsing | `pypdf` |
| Frontend | Vanilla HTML/CSS/JS |

## 📁 Project structure

```
app.py                 Flask factory + routes (/  /get  /healthz)
rag.py                 Pinecone text search + Gemini prompt/answer + citations
knowledge.py           PDF -> chunks (pure Python, testable)
places.py              Google Places search, specialty mapping, distance sort
config.py              Environment settings, validation
ingest.py              One-off PDF -> Pinecone ingestion (batched, retryable)
templates/index.html   Chat UI (all rendering escaped/validated)
static/style.css       Styling
tests/                 Pytest suite (98 tests, no network required)
Dockerfile             Production image (non-root, healthcheck)
docker-compose.yml     Local container setup
SECURITY.md            Key rotation + deployment checklist
```

## ⚙️ Getting started

### Prerequisites

- **Python 3.12 or newer**
- A Google AI Studio key (Gemini)
- A Google Maps **Places API** key (Places API enabled, billing on)
- A Pinecone **integrated index** using `llama-text-embed-v2`:

  | Setting | Value |
  | --- | --- |
  | Name | `medicalbot` |
  | Dimension | `384` |
  | Metric | `cosine` |
  | Embedding model | `llama-text-embed-v2` |
  | Text field | `text` |

  Create it in the Pinecone console (**Create index → Integrated model**) or:

  ```python
  from pinecone import Pinecone
  Pinecone(api_key="...").create_index_for_model(
      name="medicalbot",
      cloud="aws",
      region="us-east-1",
      embed={"model": "llama-text-embed-v2", "field_map": {"text": "text"}},
  )
  ```

  > A *standard* index (one where you supply your own vectors) will not work
  > with this app: retrieval is done with `search(inputs={"text": ...})`.

### 1. Clone and install

```bash
git clone https://github.com/Swapnil26-art/Doctor-Recommendation-System-RAG-chat-bot-.git
cd Doctor-Recommendation-System-RAG-chat-bot-

python -m venv .venv
.\.venv\Scripts\activate          # Windows
# source .venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
```

### 2. Configure secrets

```powershell
Copy-Item .env.example .env      # Windows
cp .env.example .env             # macOS / Linux
```

Fill in `.env`:

```dotenv
GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
GOOGLE_MAPS_API_KEY="YOUR_GOOGLE_MAPS_PLACES_API_KEY"
PINECONE_API_KEY="YOUR_PINECONE_API_KEY"
```

`.env` is gitignored and **takes precedence over system environment variables**
(so a stale `PINECONE_API_KEY` in your OS cannot silently break the app).
Never commit it — see [SECURITY.md](SECURITY.md).

### 3. Ingest the knowledge base

```bash
python ingest.py --dry-run   # count chunks, touch nothing (3522 for this PDF)
python ingest.py             # upsert into PINECONE_NAMESPACE
python ingest.py --reset     # delete the namespace first, then re-ingest
```

### 4. Run

```bash
python app.py
```

Open <http://127.0.0.1:8080>. Allow location access when prompted (the browser
only shares it with this app's server).

## 🐳 Docker

```bash
docker compose up --build
```

Or without Compose:

```bash
docker build -t medbot .
docker run --rm -p 8080:8080 --env-file .env medbot
```

The image is now ~180 MB (no torch), runs as a non-root user, and serves
through gunicorn with a `/healthz` healthcheck.

## 🔌 API

| Method | Path | Body | Response |
| --- | --- | --- | --- |
| `GET` | `/` | – | Chat page |
| `GET` | `/healthz` | – | `{"status":"ok","configured":{"google":true,"maps":true,"pinecone":true}}` |
| `POST` | `/get` | `{"msg": "...", "location": "lat,lon"}` | see below |

```jsonc
// RAG answer (with citations)
{
  "type": "text",
  "response": "Asthma is ...",
  "sources": [{ "source": "Medical_book.pdf", "page": 395, "score": 0.54 }]
}

// Doctor search
{
  "type": "places",
  "specialty": "cardiologist",
  "intro": "Here are the closest cardiologists and hospitals I found near you:",
  "places": [
    { "name": "...", "address": "...", "distance_km": 1.24,
      "map_url": "https://www.google.com/maps?q=...", "rating": 4.5, "user_ratings_total": 120 }
  ]
}
```

`location` is only required for doctor requests. Messages are capped at 1000
characters. Upstream failures return a generic message with HTTP 502 — internal
details are logged, never returned.

## 🧪 Tests

```bash
pip install -r requirements-dev.txt
pytest            # 98 tests, ~1 second
ruff check .
```

The suite needs no API keys and no network: Pinecone, Gemini, and Google Places
are replaced with fakes at the module boundary.

## 🩺 Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `401 Unauthorized` from Pinecone | A stale `PINECONE_API_KEY` in your OS environment. `.env` now wins; you can also remove it: `[Environment]::SetEnvironmentVariable("PINECONE_API_KEY",$null,"User")` |
| Answers are irrelevant / score ~0.15 | The index contains vectors from a different model than `llama-text-embed-v2`. Run `python ingest.py --reset` |
| No sources returned at all | Wrong namespace — check `PINECONE_NAMESPACE` and `index.describe_index_stats()` |
| `Invalid API key` from Gemini | `GOOGLE_API_KEY` is a placeholder or expired |
| `REQUEST_DENIED` from Places | Places API not enabled, billing off, or key restricted to another API |
| `No module named torch` | You are on the old commit; this version has no local ML stack |

## 🔐 Security

- Keys come from the environment only; `.env` is ignored by git and Docker.
- API keys never appear in logs (Places key travels in request params).
- All chat output is escaped before rendering; map links are host-validated.
- Error responses never echo exception text to the browser.
- Rotate keys and purge history if anything was ever committed — see
  [SECURITY.md](SECURITY.md).

## 🗺️ Roadmap

- Conversation memory so follow-up questions keep context.
- Streaming responses.
- Migrate to Places API (New) — the legacy Nearby Search endpoint is deprecated.
- LLM-based intent routing instead of keyword matching.
- Rate limiting and per-session quotas.

## 🤝 Contributing

Issues and pull requests are welcome. Please run `pytest` and `ruff check .`
before opening a PR.

## 📄 License

MIT — see [LICENSE](LICENSE).

## 📞 Contact

Swapnil Panda — <https://www.linkedin.com/in/swapnil-panda-1207992ba>
