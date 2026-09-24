import os, hmac, math, time, resource, threading
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Literal

import numpy as np
import copernicusmarine
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="PescaPro Marine Worker", version="0.4.0")
STARTED_AT = time.time()

WAVE_DATASET = "cmems_mod_glo_wav_anfc_0.083deg_PT3H-i"
PHY_DATASET = "cmems_mod_glo_phy_anfc_0.083deg_PT1H-m"

# Free-instance protection: only one heavy Copernicus query runs at a time.
HEAVY_QUERY_SLOTS = threading.Semaphore(1)
STATE_LOCK = threading.Lock()
INFLIGHT = {}
CACHE = {}
CACHE_TTL_SECONDS = 15 * 60
CACHE_MAX_ENTRIES = 64
ZONE_STEP = 0.05
ADAPTIVE_WINDOWS = (0.03, 0.06, 0.10)
PHYSICS_SURFACE_DEPTH_M = 0.494

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

    # Check cells in distance order and stop at the first cell that is valid
    # for every requested variable. This avoids scanning the whole grid.
    candidates = []
    for yi, la in enumerate(lats):
        for xi, lo in enumerate(lons):
            dx = (float(lo) - lon) * math.cos(math.radians(lat))
            dy = float(la) - lat
            candidates.append((dx * dx + dy * dy, yi, xi, float(la), float(lo)))
    candidates.sort(key=lambda x: x[0])

    chosen = None
    for _, yi, xi, sla, slo in candidates:
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
            if not np.any(np.isfinite(vals.astype(float, copy=False))):
                valid = False
                break
        if valid:
            chosen = (yi, xi, sla, slo)
            break

    if chosen is None:
        raise RuntimeError("No valid marine cell found")

    yi, xi, sla, slo = chosen
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

def _open_wave(lat, lon, start, end, username, password, window):
    return copernicusmarine.open_dataset(
        dataset_id=WAVE_DATASET,
        variables=["VHM0", "VTM10", "VMDR"],
        minimum_longitude=lon - window,
        maximum_longitude=lon + window,
        minimum_latitude=lat - window,
        maximum_latitude=lat + window,
        start_datetime=start,
        end_datetime=end,
        coordinates_selection_method="outside",
        username=username,
        password=password,
    )

def _open_physics(lat, lon, start, end, username, password, window):
    return copernicusmarine.open_dataset(
        dataset_id=PHY_DATASET,
        variables=["thetao", "uo", "vo"],
        minimum_longitude=lon - window,
        maximum_longitude=lon + window,
        minimum_latitude=lat - window,
        maximum_latitude=lat + window,
        # Request only the model's surface layer (~0.494 m), rather than 0-1 m.
        minimum_depth=PHYSICS_SURFACE_DEPTH_M,
        maximum_depth=PHYSICS_SURFACE_DEPTH_M,
        start_datetime=start,
        end_datetime=end,
        coordinates_selection_method="outside",
        username=username,
        password=password,
    )

def _fetch_adaptive(req, start, end, username, password):
    errors = []
    for window in ADAPTIVE_WINDOWS:
        ds = None
        t_provider = time.monotonic()
        try:
            if req.kind == "waves":
                ds = _open_wave(req.latitude, req.longitude, start, end, username, password, window)
                variables = ["VHM0", "VTM10", "VMDR"]
                surface = False
            else:
                ds = _open_physics(req.latitude, req.longitude, start, end, username, password, window)
                variables = ["thetao", "uo", "vo"]
                surface = True

            # Materialize only this small subset so provider/network time and
            # local processing time can be measured separately.
            ds.load()
            provider_seconds = time.monotonic() - t_provider
            t_process = time.monotonic()
            row = _nearest_valid_series(ds, variables, req.latitude, req.longitude, surface=surface)
            processing_seconds = time.monotonic() - t_process
            return row, window, provider_seconds, processing_seconds
        except Exception as exc:
            errors.append(f"{window:.2f}:{type(exc).__name__}")
        finally:
            if ds is not None:
                try:
                    ds.close()
                except Exception:
                    pass
    raise RuntimeError("No valid marine data after adaptive windows: " + ",".join(errors))

@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "pescapro-worker",
        "version": "0.4.0",
        "uptime_seconds": int(time.time() - STARTED_AT),
        "max_rss_mb": _rss_mb(),
        "heavy_query_concurrency": 1,
        "adaptive_windows_degrees": list(ADAPTIVE_WINDOWS),
        "physics_surface_depth_m": PHYSICS_SURFACE_DEPTH_M,
        "cache_ttl_seconds": CACHE_TTL_SECONDS,
        "cache_entries": len(CACHE),
        "inflight": len(INFLIGHT),
        "copernicus_credentials_configured": bool(
            (os.getenv("COPERNICUSMARINE_SERVICE_USERNAME") or "").strip()
            and (os.getenv("COPERNICUSMARINE_SERVICE_PASSWORD") or "").strip()
        ),
    }

def _compute_marine(req: MarineRequest, authorization: str | None):
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
            row, query_window, provider_seconds, processing_seconds = _fetch_adaptive(req, start, end, username, password)
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
            row, query_window, provider_seconds, processing_seconds = _fetch_adaptive(req, start, end, username, password)
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
            "worker_version": "0.4.0",
            "dataset": dataset,
            "latitude": req.latitude,
            "longitude": req.longitude,
            "sample_latitude": row["sample_latitude"],
            "sample_longitude": row["sample_longitude"],
            "current": current,
            "hourly": hourly,
            "query_window_degrees": query_window,
            "timing": {
                "provider_seconds": round(provider_seconds, 2),
                "processing_seconds": round(processing_seconds, 3),
            },
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


def _zone_key(req: MarineRequest):
    zlat = round(req.latitude / ZONE_STEP) * ZONE_STEP
    zlon = round(req.longitude / ZONE_STEP) * ZONE_STEP
    return (round(zlat, 4), round(zlon, 4), req.days, req.kind)

def _cache_get(key):
    now = time.monotonic()
    with STATE_LOCK:
        item = CACHE.get(key)
        if not item:
            return None
        if now - item["stored_at"] > CACHE_TTL_SECONDS:
            CACHE.pop(key, None)
            return None
        return item

def _cache_put(key, value):
    with STATE_LOCK:
        now = time.monotonic()
        expired = [k for k, v in CACHE.items() if now - v["stored_at"] > CACHE_TTL_SECONDS]
        for k in expired:
            CACHE.pop(k, None)
        if len(CACHE) >= CACHE_MAX_ENTRIES:
            oldest = min(CACHE, key=lambda k: CACHE[k]["stored_at"])
            CACHE.pop(oldest, None)
        CACHE[key] = {"stored_at": now, "value": value}

def _with_cache_meta(value, status, key, waited=False):
    out = dict(value)
    out["cache"] = {
        "status": status,
        "zone_latitude": key[0],
        "zone_longitude": key[1],
        "ttl_seconds": CACHE_TTL_SECONDS,
        "waited_for_inflight": waited,
    }
    return out

@app.post("/marine")
def marine(req: MarineRequest, authorization: str | None = Header(default=None)):
    _require_worker_key(authorization)
    if not _allowed_pescapro_area(req.latitude, req.longitude):
        raise HTTPException(status_code=422, detail="Coordinates outside PescaPro area")

    key = _zone_key(req)
    cached = _cache_get(key)
    if cached:
        return _with_cache_meta(cached["value"], "hit", key)

    # Single-flight: one leader computes a given zone/kind/horizon; followers wait
    # for that result instead of launching duplicate Copernicus downloads.
    while True:
        with STATE_LOCK:
            event = INFLIGHT.get(key)
            if event is None:
                event = threading.Event()
                INFLIGHT[key] = event
                leader = True
            else:
                leader = False

        if leader:
            try:
                # Different keys are also serialized to protect the 512 MB instance.
                with HEAVY_QUERY_SLOTS:
                    result = _compute_marine(req, authorization)
                _cache_put(key, result)
                return _with_cache_meta(result, "miss", key)
            finally:
                with STATE_LOCK:
                    done = INFLIGHT.pop(key, None)
                    if done:
                        done.set()

        if not event.wait(timeout=90):
            raise HTTPException(status_code=503, detail="Marine worker queue timeout")
        cached = _cache_get(key)
        if cached:
            return _with_cache_meta(cached["value"], "hit", key, waited=True)
        # The leader may have failed. Loop and allow one waiter to retry.
