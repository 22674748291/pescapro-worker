import os, hmac, time
from typing import Optional
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="PescaPro Marine Worker", version="0.1.0")
STARTED_AT = time.time()

def require_worker_key(authorization: Optional[str]) -> None:
    expected = os.getenv("WORKER_API_KEY", "")
    if not expected:
        raise HTTPException(status_code=503, detail="WORKER_API_KEY is not configured")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not hmac.compare_digest(authorization[7:], expected):
        raise HTTPException(status_code=401, detail="Unauthorized")

class MarineRequest(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    days: int = Field(default=7, ge=1, le=14)
    kind: str = Field(default="physics", pattern="^(physics|waves)$")

@app.get("/health")
def health():
    return {"ok": True, "service": "pescapro-worker", "version": "0.1.0",
            "uptime_seconds": int(time.time() - STARTED_AT)}

@app.post("/marine")
def marine(req: MarineRequest, authorization: Optional[str] = Header(default=None)):
    require_worker_key(authorization)
    return {"ok": True, "mode": "validation", "request": req.model_dump(),
            "message": "Worker is healthy; Copernicus production query is not enabled yet."}
