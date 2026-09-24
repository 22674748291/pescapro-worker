import os, hmac, math, time, resource
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Literal

import numpy as np
import copernicusmarine
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="PescaPro Marine Worker", version="0.2.0")
STARTED_AT = time.time()

WAVE_DATASET = "cmems_mod_glo_wav_anfc_0.083deg_PT3H-i"
PHY_DATASET = "cmems_mod_glo_phy_anfc_0.083deg_PT1H-m"

class MarineRequest(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    days: int = Field(default=1, ge=1, le=2)
    kind: Literal["waves", "physics"] = "waves"

def _rss_mb():
    # Linux ru_maxrss is KiB.
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)

def _require_worker_key(authorization: str | None):
    expected = os.getenv("WORKER_API_KEY", "").strip()
    if not expected:
        raise HTTPException(status_code=503, detail="WORKER_API_KEY is not configured")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not hmac.compare_digest(authorization[7:], expected):
        raise HTTPException(status_code=401, detail="Unauthorized")

def _credentials():
    u = (os.getenv("COPERNICUSMARINE_SERVICE_USERNAME") or "").strip()
    p = (os.getenv("COPERNICUSMARINE_SERVICE_PASSWORD") or "").strip()
    if not u or not p:
        raise HTTPException(status_code=503, detail="Copernicus credentials are not configured")
    return u, p

def _allowed_pescapro_area(lat, lon):
    mainland = 35.0 <= lat <= 44.8 and -10.8 <= lon <= 5.0
    canary = 27.0 <= lat <= 30.2 and -19.0 <= lon <= -12.0
    return mainland or canary

def _finite(v):
    try:
        x = float(np.asarray(v).reshape(-1)[0])
        return x if math.isfinite(x) else None
    except Exception:
        return None

def _tz_for_lon(lon):
    return ZoneInfo("Atlantic/Canary" if float(lon) < -12 else "Europe/Madrid")

def _coord_name(ds, options):
    for name in options:
        if name in ds.coords or name in ds.dims:
            return name
    raise RuntimeError("Missing coordinate: " + "/".join(options))

def _local_iso(v, tz):
    s = str(v).replace(" ", "T")
    if s.endswith("Z"):
        s = s[:-1]
    s = s[:19]
    dt = datetime.fromisoformat(s).replace(tzinfo=timezone.utc).astimezone(tz)
    return dt.replace(tzinfo=None).isoformat(timespec="minutes")

def _nearest_valid_series(ds, var_names, lat, lon, surface=False):
    tz = _tz_for_lon(lon)
    tname = _coord_name(ds, ["time"])
    latname = _coord_name(ds, ["latitude", "lat"])
    lonname = _coord_name(ds, ["longitude", "lon"])
    work = ds

    if surface:
        for dname in ("depth", "elevation"):
            if dname in work.dims:
                coord = np.asarray(work[dname].values, dtype=float).reshape(-1)
                finite_idx = np.where(np.isfinite(coord))[0]
                surface_i = int(finite_idx[np.argmin(np.abs(coord[finite_idx]))]) if finite_idx.size else 0
                work = work.isel({dname: surface_i})
                break

    lats = np.asarray(work[latname].values, dtype=float).reshape(-1)
    lons = np.asarray(work[lonname].values, dtype=float).reshape(-1)
    candidates = []

    for yi, la in enumerate(lats):
        for xi, lo in enumerate(lons):
            valid = True
            for vn in var_names:
                if vn not in work:
                    raise RuntimeError(f"Missing variable {vn}")
                da = work[vn]
                idx = {}
                if latname in da.dims:
                    idx[latname] = yi
                if lonname in da.dims:
                    idx[lonname] = xi
                vals = np.asarray(da.isel(idx).values).reshape(-1)
                if not any(_finite(x) is not None for x in vals):
                    valid = False
                    break
            if valid:
                dx = (float(lo) - lon) * math.cos(math.radians(lat))
                dy = float(la) - lat
                candidates.append((dx * dx + dy * dy, yi, xi, float(la), float(lo)))

    if not candidates:
        raise RuntimeError("No valid marine cell found")

    candidates.sort(key=lambda x: x[0])
    _, yi, xi, sla, slo = candidates[0]
    result = {
        "time": [_local_iso(t, tz) for t in work[tname].values],
        "sample_latitude": sla,
        "sample_longitude": slo,
    }
    for vn in var_names:
        da = work[vn]
        idx = {}
        if latname in da.dims:
            idx[latname] = yi
        if lonname in da.dims:
            idx[lonname] = xi
        result[vn] = [_finite(x) for x in np.asarray(da.isel(idx).values).reshape(-1)]
    return result

def _nearest_index(times, tz):
    now = datetime.now(tz).replace(tzinfo=None)
    best = None
    for i, t in enumerate(times or []):
        try:
            d = abs((datetime.fromisoformat(t) - now).total_seconds())
            if best is None or d < best[0]:
                best = (d, i)
        except Exception:
            pass
    return None if best is None else best[1]

def _current_direction(u, v):
    if u is None or v is None:
        return None
    return (math.degrees(math.atan2(u, v)) + 360.0) % 360.0

def _open_wave(lat, lon, start, end, username, password):
    return copernicusmarine.open_dataset(
        dataset_id=WAVE_DATASET,
        variables=["VHM0", "VTM10", "VMDR"],
        minimum_longitude=lon - 0.10,
        maximum_longitude=lon + 0.10,
        minimum_latitude=lat - 0.10,
        maximum_latitude=lat + 0.10,
        start_datetime=start,
        end_datetime=end,
        coordinates_selection_method="outside",
        username=username,
        password=password,
    )

def _open_physics(lat, lon, start, end, username, password):
    return copernicusmarine.open_dataset(
        dataset_id=PHY_DATASET,
        variables=["thetao", "uo", "vo"],
        minimum_longitude=lon - 0.10,
        maximum_longitude=lon + 0.10,
        minimum_latitude=lat - 0.10,
        maximum_latitude=lat + 0.10,
        minimum_depth=0,
        maximum_depth=1,
        start_datetime=start,
        end_datetime=end,
        coordinates_selection_method="outside",
        username=username,
        password=password,
    )

@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "pescapro-worker",
        "version": "0.2.0",
        "uptime_seconds": int(time.time() - STARTED_AT),
        "max_rss_mb": _rss_mb(),
        "copernicus_credentials_configured": bool(
            (os.getenv("COPERNICUSMARINE_SERVICE_USERNAME") or "").strip()
            and (os.getenv("COPERNICUSMARINE_SERVICE_PASSWORD") or "").strip()
        ),
    }

@app.post("/marine")
def marine(req: MarineRequest, authorization: str | None = Header(default=None)):
    _require_worker_key(authorization)
    if not _allowed_pescapro_area(req.latitude, req.longitude):
        raise HTTPException(status_code=422, detail="Coordinates outside PescaPro area")

    username, password = _credentials()
    now = datetime.now(timezone.utc)
    start = (now - timedelta(hours=2)).isoformat(timespec="seconds")
    end = (now + timedelta(days=req.days, hours=2)).isoformat(timespec="seconds")
    t0 = time.monotonic()
    rss_before = _rss_mb()

    try:
        if req.kind == "waves":
            ds = _open_wave(req.latitude, req.longitude, start, end, username, password)
            row = _nearest_valid_series(ds, ["VHM0", "VTM10", "VMDR"], req.latitude, req.longitude)
            hourly = {
                "time": row["time"],
                "wave_height": row["VHM0"],
                "wave_period": row["VTM10"],
                "wave_direction": row["VMDR"],
            }
            i = _nearest_index(hourly["time"], _tz_for_lon(req.longitude))
            current = {
                "time": hourly["time"][i] if i is not None else None,
                "wave_height": hourly["wave_height"][i] if i is not None else None,
                "wave_period": hourly["wave_period"][i] if i is not None else None,
                "wave_direction": hourly["wave_direction"][i] if i is not None else None,
            }
            dataset = WAVE_DATASET

        else:
            ds = _open_physics(req.latitude, req.longitude, start, end, username, password)
            row = _nearest_valid_series(ds, ["thetao", "uo", "vo"], req.latitude, req.longitude, surface=True)
            speeds, directions = [], []
            for u, v in zip(row["uo"], row["vo"]):
                speeds.append(None if u is None or v is None else math.hypot(u, v) * 3.6)
                directions.append(_current_direction(u, v))
            hourly = {
                "time": row["time"],
                "ocean_current_velocity": speeds,
                "ocean_current_direction": directions,
                "sea_surface_temperature": row["thetao"],
            }
            i = _nearest_index(hourly["time"], _tz_for_lon(req.longitude))
            current = {
                "time": hourly["time"][i] if i is not None else None,
                "ocean_current_velocity": hourly["ocean_current_velocity"][i] if i is not None else None,
                "ocean_current_direction": hourly["ocean_current_direction"][i] if i is not None else None,
                "sea_surface_temperature": hourly["sea_surface_temperature"][i] if i is not None else None,
            }
            dataset = PHY_DATASET

        return {
            "ok": True,
            "mode": req.kind,
            "source": "Copernicus Marine Service",
            "worker_version": "0.2.0",
            "dataset": dataset,
            "latitude": req.latitude,
            "longitude": req.longitude,
            "sample_latitude": row["sample_latitude"],
            "sample_longitude": row["sample_longitude"],
            "current": current,
            "hourly": hourly,
            "elapsed_seconds": round(time.monotonic() - t0, 2),
            "memory": {
                "max_rss_before_mb": rss_before,
                "max_rss_after_mb": _rss_mb(),
            },
        }
    except HTTPException:
        raise
    except Exception as exc:
        # Do not expose credentials or provider internals to the caller.
        print("Copernicus worker error:", type(exc).__name__, str(exc)[:300])
        raise HTTPException(status_code=502, detail="Copernicus Marine unavailable")
