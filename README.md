# MedBot: AI-Powered Medical Chatbot with Doctor Recommendation ⚕️

[![CI](https://github.com/Swapnil26-art/Doctor-Recommendation-System-RAG-chat-bot-/actions/workflows/ci.yml/badge.svg)](https://github.com/Swapnil26-art/Doctor-Recommendation-System-RAG-chat-bot-/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.13-blue.svg)](https://www.python.org/)
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

- **Medical Q&A (RAG)** — PDF is chunked, embedded with `all-MiniLM-L6-v2`, stored
  in Pinecone, and answered by Gemini using only the retrieved context. The model
  is instructed to say "I don't have that information" rather than guess.
- **Doctor recommendation** — Google Places lookups ranked by true geodesic
  distance, permanently-closed businesses filtered out, results returned as
  structured data and rendered as safe cards.
- **Symptom → specialty routing** — "my tooth hurts" searches dentists, "chest
  pain" searches cardiologists, "nearest hospital" searches hospitals.
- **Safe rendering** — no `innerHTML` on untrusted data. Every API/LLM string is
  escaped before display, and map links are validated to `https://*.google.com`.
- **Configurable, testable, container-ready** — settings in one module, Flask
  application factory, lazy model loading, `Dockerfile`, `docker compose`, and CI.

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
                         │                 ├── MiniLM embeddings (local)
                         │                 ├── Pinecone similarity search (k=3)
                         │                 └── Gemini 2.5 Flash
                         └── JSON: { type: "text" | "places", ... }
```

### Data ingestion (one-off)

```
data/*.pdf → PyPDFLoader → RecursiveCharacterTextSplitter (500/20)
           → all-MiniLM-L6-v2 (384 dims) → Pinecone index "medicalbot"
```

## 🛠️ Tech stack

| Layer | Choice |
| --- | --- |
| Backend | Flask 3.1 |
| LLM | Google Gemini 2.5 Flash (`langchain-google-genai`) |
| Vector DB | Pinecone (384 dimensions, cosine) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, no API cost) |
| RAG orchestration | LangChain 0.3 |
| Maps | Google Places API (Nearby Search) + `geopy` |
| PDF parsing | `pypdf` via `PyPDFLoader` |
| Frontend | Vanilla HTML/CSS/JS |

> **Python 3.12 or 3.13 is required.** `torch` and `langchain-pinecone` do not
> publish wheels for Python 3.14 yet. On 3.14 `pip install` fails with
> "no matching distribution found".

## 📁 Project structure

```
app.py                 Flask factory + routes (/  /get  /healthz)
rag.py                 Lazy RAG chain (torch imported only on first question)
places.py              Google Places search, specialty mapping, distance sort
config.py              Environment settings, validation
ingest.py              One-off PDF → Pinecone ingestion
templates/index.html   Chat UI (all rendering escaped/validated)
static/style.css       Styling
tests/                 Pytest suite (65 tests, no network, no ML stack needed)
Dockerfile             Production image (CPU torch, non-root, healthcheck)
docker-compose.yml     Local container setup with model cache volume
SECURITY.md            Key rotation + deployment checklist
```

## ⚙️ Getting started

### Prerequisites

- **Python 3.12 or 3.13**
- A Google AI Studio key (Gemini)
- A Google Maps **Places API** key (Places API enabled in Google Cloud)
- A Pinecone account with an index named `medicalbot`, **dimension 384**, **cosine**

### 1. Clone and install

```bash
git clone https://github.com/Swapnil26-art/Doctor-Recommendation-System-RAG-chat-bot-.git
cd Doctor-Recommendation-System-RAG-chat-bot-

py -3.13 -m venv .venv            # Windows
# python3.12 -m venv .venv        # macOS / Linux

.\.venv\Scripts\activate          # Windows
# source .venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
```

### 2. Configure secrets

```bash
Copy-Item .env.example .env      # Windows
cp .env.example .env             # macOS / Linux
```

Fill in `.env`:

```dotenv
GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
GOOGLE_MAPS_API_KEY="YOUR_GOOGLE_MAPS_PLACES_API_KEY"
PINECONE_API_KEY="YOUR_PINECONE_API_KEY"
```

`.env` is gitignored. Never commit it — see [SECURITY.md](SECURITY.md).

### 3. Ingest the knowledge base

```bash
python ingest.py
```

Wait for `Successfully ingested N chunks`. Re-running will add duplicate
vectors, so only run it when the source PDF or chunking changes.

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

The image installs CPU-only torch, runs as a non-root user, and serves through
gunicorn with a `/healthz` healthcheck.

## 🔌 API

| Method | Path | Body | Response |
| --- | --- | --- | --- |
| `GET` | `/` | – | Chat page |
| `GET` | `/healthz` | – | `{"status":"ok","configured":{"google":true,"maps":true,"pinecone":true}}` |
| `POST` | `/get` | `{"msg": "...", "location": "lat,lon"}` | see below |

```jsonc
// RAG answer
{ "type": "text", "response": "Asthma is ..." }

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
pytest            # 65 tests, ~1 second
ruff check .
```

The suite needs no API keys, no network, and no ML stack: `rag.py` imports
torch lazily, so the web layer is tested in isolation with mocked boundaries.
CI runs lint + tests on every push and builds the Docker image on `main`.

## 🔐 Security

- Keys come from the environment only; `.env` is ignored by git and Docker.
- API keys never appear in logs (Places key travels in request params).
- All chat output is escaped before rendering; map links are host-validated.
- Error responses never echo exception text to the browser.
- Rotate keys and purge history if anything was ever committed — see
  [SECURITY.md](SECURITY.md).

## 🗺️ Roadmap

- Conversation memory so follow-up questions keep context.
- Source citations (page numbers) alongside RAG answers.
- Streaming responses.
- Migrate from `RetrievalQA` to LangChain LCEL.
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
