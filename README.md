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
  application factory, 115 tests, `Dockerfile`, `docker compose`, and CI.

## 🏗️ Architecture

```
Browser (templates/index.html)
   ├── geolocation ─────────────────┐
   └── POST /get { msg, location }  │
                                    ▼
                        Flask (app.py)
                         ├── places.is_doctor_request(msg)
                         │      ├── yes → places.find_nearby_places()
                         │      │          └── Places API (New) Text Search + geopy
                         │      └── no  → rag.answer_question()
                         │                 ├── Pinecone search(inputs={"text": q})
                         │                 │     └── llama-text-embed-v2 (server-side)
                         │                 ├── relevance floor + page citations
                         │                 └── Gemini 3.8 Flash (3.6/3.5 fallback)
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
| LLM | Google Gemini 3.8 Flash (`google-genai`), falling back to 3.6/3.5 when overloaded |
| Vector DB | Pinecone serverless, **integrated index** |
| Embeddings | `llama-text-embed-v2`, hosted by Pinecone (384 dims, cosine) |
| Retrieval | Pinecone text search (no local model) |
| Maps | Google Places API (New) Text Search + `geopy` |
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
tests/                 Pytest suite (115 tests, no network required)
Dockerfile             Production image (non-root, healthcheck)
docker-compose.yml     Local container setup
vercel.json            Vercel function timeout (Flask entrypoint: app.py)
.vercelignore          Files excluded from the Vercel upload
.python-version        Python 3.12 for Vercel
SECURITY.md            Key rotation + deployment checklist
```

## ⚙️ Getting started

### Prerequisites

- **Python 3.12 or newer**
- A Google AI Studio key (Gemini)
- A Google Maps **Places API (New)** key (enabled in the project, billing on)
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

## ▲ Vercel

`app.py` exposes a top-level `app`, which is exactly what Vercel's Python
runtime looks for, so no adapter (`api/index.py`, `Procfile`, custom start
command) is needed. `vercel.json` only sets the function timeout, and
`.vercelignore` keeps `.venv`, `data/` and the tests out of the bundle.

### 1. Import the repository

On <https://vercel.com/new> import the GitHub repo and keep the defaults:

| Setting | Value |
| --- | --- |
| Framework preset | **Flask** (pinned in `vercel.json`; `app.py` is the entrypoint) |
| Root directory | `.` |
| Build command | *(empty — Vercel installs `requirements.txt` itself)* |
| Output directory | *(empty)* |

`.python-version` pins Python **3.12**. Do **not** copy the Docker `CMD` or a
`gunicorn` command into a start-command field; Vercel calls the WSGI app
directly.

### 2. Add environment variables

Project → **Settings** → **Environment Variables** (tick Production, Preview
and Development):

| Variable | Required | Without it |
| --- | --- | --- |
| `GOOGLE_API_KEY` | yes | Chat answers return HTTP 502 |
| `PINECONE_API_KEY` | yes | Retrieval fails, chat returns HTTP 502 |
| `GOOGLE_MAPS_API_KEY` | for doctor search | Chat still works; “find a doctor” reports the service is unavailable |

Optional overrides (`LLM_MODEL`, `LLM_FALLBACK_MODELS`, `RETRIEVAL_TOP_K`,
`PLACES_RADIUS_KM`, …) are documented in [`.env.example`](.env.example).
Never commit a real `.env` — Vercel deployments read these variables only.

Three things that make keys fail on Vercel even though they work locally:

- **Redeploy after adding or editing a variable.** Vercel bakes the values into
  a deployment; an existing deployment keeps the old ones.
- **Paste the raw value**, without quotes and without a trailing newline. The
  dashboard stores exactly what you type (`"abc"` becomes a 5-character key).
- **Key restrictions must allow server-side use.** A Maps key restricted to
  *HTTP referrers* will be rejected (the call comes from the function, not the
  browser), and Vercel's egress IPs change, so IP restrictions are fragile.
  Keep the application restriction open and restrict by **API** instead:
  *Generative Language API* for `GOOGLE_API_KEY`, *Places API (New)* for
  `GOOGLE_MAPS_API_KEY`.

A real `.env` is excluded by both `.gitignore` and `.vercelignore`; keep it
that way, because `config.py` loads `.env` with `override=True`, so an uploaded
`.env` would silently win over the dashboard values.

### 3. Deploy and verify

```bash
# after the first Git deploy
curl https://<project>.vercel.app/healthz
# {"status":"ok","configured":{"google":true,"maps":true,"pinecone":true}}
```

Or from the CLI:

```bash
npm i -g vercel
vercel link
vercel env add GOOGLE_API_KEY production
vercel env add PINECONE_API_KEY production
vercel env add GOOGLE_MAPS_API_KEY production
vercel --prod
```

### Notes specific to Vercel

- **No `gunicorn`, no persistent disk, read-only filesystem.** The app never
  writes files, so this is fine. But `python ingest.py` cannot run there —
  ingest locally (or in CI) and let the function query Pinecone.
- **Cold starts.** The first request after idle imports `pinecone` and
  `google-genai`; expect a few seconds before the first answer.
- **Timeout.** `vercel.json` sets `maxDuration` to 120 s so a slow Gemini
  call (primary + 2 fallbacks, 20 s each) is not cut off. Hobby allows up to
  300 s.
- **Region.** Hobby functions run in `iad1` (Washington, D.C.); a Pro plan can
  move them closer to your users.
- **Geolocation works on the live site.** Vercel serves HTTPS, and browsers
  only allow `navigator.geolocation` on `https://` or `localhost`. The badge
  turns green without extra configuration; if it stays red, check the browser
  / OS location permission rather than the deployment.
- Doctor search still costs money per call (Places API) and Gemini's free tier
  is ~20 requests/day/model — see [SECURITY.md](SECURITY.md) before sharing the
  URL publicly.

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
pytest            # 115 tests, ~1 second
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
| Answers take 20–50 s, log shows `503 UNAVAILABLE` / `504 DEADLINE_EXCEEDED` | Gemini is overloaded. The app retries briefly, then falls back to `LLM_FALLBACK_MODELS`; tune `LLM_MODEL` / `LLM_FALLBACK_MODELS` in `.env` |
| `REQUEST_DENIED` from Places | Places API **(New)** not enabled, billing off, or key restricted to another API. Legacy Nearby Search keys do not work — the app calls `places.googleapis.com/v1/places:searchText` |
| `No module named torch` | You are on the old commit; this version has no local ML stack |
| Vercel build: “No entrypoint found” | `app.py` must export a top-level `app` (it does) and stay in the repo root; remove any custom start command |
| Vercel: `FUNCTION_INVOCATION_TIMEOUT` / 504 | Gemini was slow past `maxDuration`; raise it in `vercel.json` or set a faster `LLM_MODEL` |
| Vercel: page loads unstyled | `static/` or `templates/` was excluded from the upload — check `.vercelignore` |
| Vercel: `/healthz` still says `false` after adding keys | The variables were added after the deployment — run **Redeploy** |
| Maps works locally but returns 403 on Vercel | The key is restricted by HTTP referrer/IP; restrict by API instead (see the Vercel section) |
| Live badge says “Location blocked” | Browsers need HTTPS (Vercel gives you that) plus browser/OS location permission |

## 🔐 Security

- Keys come from the environment only; `.env` is ignored by git and Docker.
- API keys never appear in logs or URLs (the Places key travels in the `X-Goog-Api-Key` header).
- All chat output is escaped before rendering; map links are host-validated.
- Error responses never echo exception text to the browser.
- Rotate keys and purge history if anything was ever committed — see
  [SECURITY.md](SECURITY.md).

## 🗺️ Roadmap

- Conversation memory so follow-up questions keep context.
- Streaming responses.
- Cache Places results briefly to cut API cost on repeat searches.
- LLM-based intent routing instead of keyword matching.
- Rate limiting and per-session quotas.

## 🤝 Contributing

Issues and pull requests are welcome. Please run `pytest` and `ruff check .`
before opening a PR.

## 📄 License

MIT — see [LICENSE](LICENSE).

## 📞 Contact

Swapnil Panda — <https://www.linkedin.com/in/swapnil-panda-1207992ba>
