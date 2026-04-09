import os
import requests
from datetime import datetime, timezone
from extensions import db
from models import AQIHistory

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# minimal city -> coordinates (you can expand)
CITY_COORDS = {
    "Chennai":   (13.0827, 80.2707),
    "Delhi":     (28.6139, 77.2090),
    "Bengaluru": (12.9716, 77.5946),
}

def get_city_coords(city: str):
    if city not in CITY_COORDS:
        raise ValueError(f"Unknown city: {city}. Add it to CITY_COORDS.")
    return CITY_COORDS[city]

def fetch_openweather_aqi(city: str) -> dict:
    """Returns {'aqi': 1..5, 'components': {...}} for current hour."""
    if not OPENWEATHER_API_KEY:
        raise RuntimeError("OPENWEATHER_API_KEY is not set")

    lat, lon = get_city_coords(city)
    url = "https://api.openweathermap.org/data/2.5/air_pollution"
    params = {"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    item = data["list"][0]
    aqi_index = item["main"]["aqi"]           # 1..5
    comp = item.get("components", {})         # μg/m3 where available

    return {
        "aqi": float(aqi_index),
        "components": {
            "pm25": comp.get("pm2_5"),
            "pm10": comp.get("pm10"),
            "no2":  comp.get("no2"),
            "co":   comp.get("co"),
            "so2":  comp.get("so2"),
            "o3":   comp.get("o3"),
        }
    }

def save_current_aqi(city: str) -> AQIHistory:
    """Fetch current AQI and save one row (UTC, rounded to hour)."""
    payload = fetch_openweather_aqi(city)

    # round current UTC to hour boundary so hourly uniqueness works
    now_utc = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    row = AQIHistory(
        city=city,
        ts=now_utc,
        aqi=payload["aqi"],
        pm25=payload["components"]["pm25"],
        pm10=payload["components"]["pm10"],
        no2=payload["components"]["no2"],
        co=payload["components"]["co"],
        so2=payload["components"]["so2"],
        o3=payload["components"]["o3"],
    )
    try:
        db.session.add(row)
        db.session.commit()
    except Exception:
        db.session.rollback()
        # if a row for (city, ts) already exists, just ignore – this makes the job idempotent
        existing = AQIHistory.query.filter_by(city=city, ts=now_utc).first()
        if existing:
            row = existing
        else:
            raise
    return row
