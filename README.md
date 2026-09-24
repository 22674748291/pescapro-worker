# pescapro-worker

Isolated marine-data worker for PescaPro.

This first version is intentionally a deployment-validation build. It checks FastAPI, non-root execution, port 8080, private bearer authentication and installation of the stable Copernicus Marine package.

It does **not** call production PescaPro, Supabase, Vercel or Copernicus yet.

## Endpoints
- `GET /health` — public health check.
- `POST /marine` — requires `Authorization: Bearer <WORKER_API_KEY>`.

## Required environment variable
`WORKER_API_KEY`

Set the real value only in the hosting provider's private Environment settings. Never commit it to GitHub.

There are no credentials, API keys, Supabase secrets or Copernicus credentials in this repository.
