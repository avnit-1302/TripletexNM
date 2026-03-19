# Tripletex Competition Agent

This repo contains a first competition-oriented `/solve` implementation for the Tripletex AI Accounting Agent task.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Optional environment variables

- `COMPETITION_API_KEY`: protects the inbound `/solve` endpoint with `Authorization: Bearer <key>`
- `OPENAI_API_KEY`: enables model-assisted multilingual extraction on top of the rule-based parser
- `OPENAI_MODEL`: overrides the default extraction model
- `OPENAI_BASE_URL`: overrides the OpenAI base URL
- `LOG_LEVEL`: logging level, defaults to `INFO`

## Notes

- Tripletex API calls use the incoming `tripletex_credentials.base_url` and `session_token`.
- PDF extraction uses `pypdf` when available.
- The parser falls back to deterministic extraction if the OpenAI integration is not configured.
