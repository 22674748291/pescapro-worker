# pescapro-worker v0.3

Isolated Copernicus Marine validation worker for PescaPro.

Required Render environment variables:
- WORKER_API_KEY
- COPERNICUSMARINE_SERVICE_USERNAME
- COPERNICUSMARINE_SERVICE_PASSWORD

Endpoints:
- GET /health (public)
- POST /marine (Bearer-protected)

v0.3 adds protection for the 512 MB free instance:
- in-memory 15-minute cache, grouped into ~0.05-degree zones;
- single-flight deduplication for simultaneous identical zone/kind/horizon requests;
- maximum one heavy Copernicus query at a time, so different requests queue instead of multiplying RAM use;
- 90-second waiter timeout and bounded cache (64 entries).

This validation build does not connect to Supabase or Vercel production.
`days` remains intentionally limited to 1–2 days while concurrency is validated.
