#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_weather_data.py

Descarga datos de precipitación diaria desde Open-Meteo para cada lote.
Guarda un CSV por lote en app/ai/datasets/weather/.
"""

import os
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Directorio de salida
OUTPUT_DIR = "app/ai/datasets/water_stress/weather"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Coordenadas de los lotes (latitud, longitud)
LOTS = {
    "lote_1": {"lat": -32.775, "lon": -62.275},
    "lote_2": {"lat": -31.475, "lon": -62.775},
    "lote_3": {"lat": -31.975, "lon": -63.175},
    "lote_4": {"lat": -32.975, "lon": -60.975},
    "lote_5": {"lat": -30.775, "lon": -64.075},
    "lote_6": {"lat": -34.975, "lon": -63.475},
}

# Rango de fechas de tus imágenes NDVI
START_DATE = '2024-01-01'
END_DATE   = '2024-12-12'

DAILY_VARS = [
    "precipitation_sum",
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "shortwave_radiation_sum",
    "windspeed_10m_mean",
    "relative_humidity_2m_mean",
    "et0_fao_evapotranspiration"
]

TIMEZONE = "America/Argentina/Buenos_Aires"

def fetch_data(lat, lon, start_date, end_date):
    """Descarga datos de precipitación diaria desde Open-Meteo"""

    daily_params = ",".join(DAILY_VARS)

    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&daily={daily_params}"
        f"&timezone={TIMEZONE}"
    )

    r = requests.get(url)
    r.raise_for_status()
    data = r.json()

    if "daily" not in data or "precipitation_sum" not in data["daily"]:
        print("⚠️ No se encontraron datos meteorológicos.")
        return None

    df = pd.DataFrame({
        "fecha": pd.to_datetime(data["daily"]["time"]),
        "precipitacion_mm": data["daily"]["precipitation_sum"],
        "temp_mean_c": data["daily"]["temperature_2m_mean"],
        "temp_max_c": data["daily"]["temperature_2m_max"],
        "temp_min_c": data["daily"]["temperature_2m_min"],
        "radiacion_sw_mj_m2": data["daily"]["shortwave_radiation_sum"],
        "vel_viento_m_s": data["daily"]["windspeed_10m_mean"],
        "humedad_relativa_pct": data["daily"]["relative_humidity_2m_mean"],
        "et0_mm": data["daily"]["et0_fao_evapotranspiration"],
    })
    return df


def main():
    for name, coords in LOTS.items():
        print(f"🌦️ Descargando datos de {name}...")
        df = fetch_data(coords["lat"], coords["lon"], START_DATE, END_DATE)
        if df is not None:
            output_path = os.path.join(OUTPUT_DIR, f"{name}.csv")
            df.to_csv(output_path, index=False)
            print(f"✅ Guardado: {output_path} ({len(df)} registros)")
        else:
            print(f"⚠️ Falló descarga de {name}.")


if __name__ == "__main__":
    main()


# "precipitacion_mm",
# "temp_mean_c",
# "temp_max_c",
# "temp_min_c",
# "radiacion_sw_mj_m2",
# "vel_viento_m_s",
# "humedad_relativa_pct",
# "et0_mm",