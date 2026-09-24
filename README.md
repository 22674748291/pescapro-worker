# pescapro-worker v0.2

Isolated Copernicus Marine validation worker for PescaPro.

Required Render environment variables:
- WORKER_API_KEY
- COPERNICUSMARINE_SERVICE_USERNAME
- COPERNICUSMARINE_SERVICE_PASSWORD

Endpoints:
- GET /health (public)
- POST /marine (Bearer-protected)

This validation build does not connect to Supabase or Vercel production.
`days` is intentionally limited to 1–2 days while validating the 512 MB Render instance.
