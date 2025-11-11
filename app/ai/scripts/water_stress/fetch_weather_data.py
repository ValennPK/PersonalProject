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
OUTPUT_DIR = "app/ai/datasets/weather"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Coordenadas de los lotes (latitud, longitud)
LOTS = {
    "lote_1": {"lat": -33.123, "lon": -60.456},
    "lote_2": {"lat": -32.987, "lon": -61.234},
    "lote_3": {"lat": -33.876, "lon": -59.876},
}

# Rango de fechas de tus imágenes NDVI
START_DATE = "2024-04-01"
END_DATE = "2024-09-30"

def fetch_precipitation(lat, lon, start_date, end_date):
    """Descarga datos de precipitación diaria desde Open-Meteo"""
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        "&daily=precipitation_sum&timezone=America%2FSao_Paulo"
    )

    r = requests.get(url)
    r.raise_for_status()
    data = r.json()

    if "daily" not in data or "precipitation_sum" not in data["daily"]:
        print("⚠️ No se encontraron datos meteorológicos.")
        return None

    df = pd.DataFrame({
        "fecha": pd.to_datetime(data["daily"]["time"]),
        "precipitacion_mm": data["daily"]["precipitation_sum"]
    })
    return df


def main():
    for name, coords in LOTS.items():
        print(f"🌦️ Descargando datos de {name}...")
        df = fetch_precipitation(coords["lat"], coords["lon"], START_DATE, END_DATE)
        if df is not None:
            output_path = os.path.join(OUTPUT_DIR, f"{name}.csv")
            df.to_csv(output_path, index=False)
            print(f"✅ Guardado: {output_path} ({len(df)} registros)")
        else:
            print(f"⚠️ Falló descarga de {name}.")


if __name__ == "__main__":
    main()
