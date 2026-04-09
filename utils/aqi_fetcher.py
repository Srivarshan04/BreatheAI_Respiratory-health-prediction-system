import requests
from datetime import datetime
from extensions import db
from models import AQIHistory

API_KEY = os.getenv("OPENWEATHER_API_KEY")

def get_city_coords(city):
    """Fetch latitude/longitude for a city in India."""
    url = f"http://api.openweathermap.org/geo/1.0/direct?q={city},IN&limit=1&appid={OPENWEATHER_API_KEY}"
    resp = requests.get(url).json()
    if resp and len(resp) > 0:
        return resp[0]["lat"], resp[0]["lon"]
    return None, None

def fetch_and_store_aqi(city):
    """Fetch AQI for a given city and save to DB."""
    lat, lon = get_city_coords(city)
    if not lat or not lon:
        print(f"⚠️ Could not find coords for {city}")
        return None

    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}"
    response = requests.get(url).json()

    if "list" in response and len(response["list"]) > 0:
        record = response["list"][0]
        main = record["main"]
        components = record["components"]

        row = AQIHistory(
            city=city,
            ts=datetime.utcfromtimestamp(record["dt"]),
            aqi=main["aqi"],
            pm25=components.get("pm2_5"),
            pm10=components.get("pm10"),
            no2=components.get("no2"),
            co=components.get("co"),
            so2=components.get("so2"),
            o3=components.get("o3"),
        )
        db.session.add(row)
        db.session.commit()
        return row

    return None
