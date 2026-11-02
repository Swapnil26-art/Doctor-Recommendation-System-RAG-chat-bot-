# Security Policy

## Reporting a vulnerability

Please open a private security advisory on GitHub, or email the maintainer
directly. Do not open a public issue for anything that exposes API keys, user
location data, or allows code execution.

## Secrets

API keys are read **only** from environment variables (via `.env` locally).
They must never be committed.

- `.env` is listed in `.gitignore`.
- `.env.example` documents the required variable names with placeholder values.
- The Google Places request key is sent in the `X-Goog-Api-Key` header, never in
  the URL or query parameters, so it cannot leak through access logs.
- Error responses return generic messages; stack traces stay server-side.

### ⚠️ If a key has ever been committed

Replacing the value in a later commit is **not enough** — the old value stays
in git history and remains fetchable. Do all three steps:

**1. Revoke and rotate the key at the provider.**

| Key | Where to rotate |
| --- | --- |
| `GOOGLE_API_KEY` | <https://aistudio.google.com/apikey> |
| `GOOGLE_MAPS_API_KEY` | Google Cloud Console → APIs & Services → Credentials |
| `PINECONE_API_KEY` | <https://app.pinecone.io> → API Keys |

For the Maps key, also restrict it: *Application restrictions → HTTP referrers
or IP addresses*, and *API restrictions → Places API (New) only*. An unrestricted
Maps key can be used by anyone who finds it, and billing is on you.

**2. Purge the file from history** (on a fresh clone):

```bash
python -m pip install git-filter-repo
git clone https://github.com/<you>/Doctor-Recommendation-System-RAG-chat-bot-.git medbot-clean
cd medbot-clean
git filter-repo --path .env --invert-paths --force
git remote add origin https://github.com/<you>/Doctor-Recommendation-System-RAG-chat-bot-.git
git push --force --all
git push --force --tags
```

Then ask collaborators to re-clone — their old clones still contain the key.

**3. Rotate again if the repository was public for any length of time.**
Assume the key was scraped. Check the provider dashboards for unexpected usage
and set billing alerts.

## Deployment checklist

- [ ] `.env` is not in the image (`docker build` respects `.dockerignore`).
- [ ] The app runs as a non-root user (the provided image does).
- [ ] HTTPS terminates in front of the app; geolocation requires a secure origin.
- [ ] API keys are restricted by API and by referrer/IP at the provider.
- [ ] Rate limiting is enabled at the reverse proxy (Gemini and Places calls cost money).
- [ ] `LOG_LEVEL=INFO` — debug logs may include retrieved document text.

## Supported versions

Only the latest commit on `main` is supported.

## Medical disclaimer

MedBot returns general information from a medical reference book. It is not a
medical device and must not be used for diagnosis or treatment decisions.
