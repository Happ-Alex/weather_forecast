"""
data_loader.py — завантаження даних з Open-Meteo API
Підтримує пошук координат за назвою міста через Geocoding API Open-Meteo
"""

import requests
import pandas as pd
from datetime import date, datetime
from typing import Tuple, Optional


GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

DAILY_VARIABLES = [
    "precipitation_sum",
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "windspeed_10m_max",
    "windgusts_10m_max",
    "sunshine_duration",
    "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",
    "precipitation_hours",
    "rain_sum",
    "snowfall_sum",
    "weathercode",
    "pressure_msl_mean",
    "dewpoint_2m_mean",
]

FORECAST_DAILY_VARIABLES = [
    "precipitation_sum",
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "windspeed_10m_max",
    "windgusts_10m_max",
    "sunshine_duration",
    "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",
    "precipitation_hours",
    "rain_sum",
    "snowfall_sum",
    "weathercode",
    "precipitation_probability_mean",
]


def geocode_city(city_name: str) -> Tuple[float, float, str]:
    """
    Знаходить координати міста за назвою.
    Повертає (latitude, longitude, display_name).
    """
    params = {"name": city_name, "count": 1, "language": "uk", "format": "json"}
    resp = requests.get(GEOCODING_URL, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if not data.get("results"):
        raise ValueError(f"Місто '{city_name}' не знайдено. Спробуйте інший варіант назви або введіть координати вручну.")

    result = data["results"][0]
    lat = result["latitude"]
    lon = result["longitude"]
    name = result.get("name", city_name)
    country = result.get("country", "")
    display = f"{name}, {country}" if country else name
    return lat, lon, display


def fetch_historical_data(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Завантажує щоденні історичні дані з Open-Meteo Archive API.
    start_date, end_date: рядки формату 'YYYY-MM-DD'
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": "auto",
    }
    resp = requests.get(HISTORICAL_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    daily = data.get("daily", {})
    if not daily or "time" not in daily:
        raise ValueError("Не вдалося отримати дані від Open-Meteo. Перевірте координати або дати.")

    df = pd.DataFrame(daily)
    df["date"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def fetch_forecast_data(lat: float, lon: float, days: int = 7) -> pd.DataFrame:
    """
    Завантажує прогнозні дані на наступні days днів.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join(FORECAST_DAILY_VARIABLES),
        "timezone": "auto",
        "forecast_days": min(days, 16),
    }
    resp = requests.get(FORECAST_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    daily = data.get("daily", {})
    if not daily or "time" not in daily:
        raise ValueError("Не вдалося отримати прогноз від Open-Meteo.")

    df = pd.DataFrame(daily)
    df["date"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def save_dataset(df: pd.DataFrame, path: str = "weather_daily.csv") -> None:
    df.to_csv(path, index=False)


def load_dataset(path: str = "weather_daily.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df
