import os
import re
import csv
import rasterio
import numpy as np
from datetime import datetime, timedelta

# --- CONFIGURACIÓN ---
BASE_DIR = "app/ai/datasets/water_stress"
OUTPUT_CSV = os.path.join(BASE_DIR, "ndvi_dataset.csv")

# Patrón para nombres: lote_1_NDVI_20240418.tif o NDVI_20240424_interp.tif
DATE_PATTERN = re.compile(r"(\d{8})")
SPLITS = ["train", "val", "test"]

# Factor de muestreo (1 de cada N píxeles)
DOWNSAMPLE_FACTOR = 5

# --- CARGAR DATOS DE PRECIPITACIÓN ---
def load_precip_data(lote):
    """Carga el CSV de precipitaciones del lote."""
    weather_path = os.path.join(BASE_DIR, "weather", f"{lote}.csv")
    if not os.path.exists(weather_path):
        print(f"⚠️ No se encontró archivo de precipitación para {lote}")
        return {}
    data = {}
    with open(weather_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                date = datetime.strptime(row["fecha"], "%Y-%m-%d").date()
                data[date] = float(row["precipitacion_mm"])
            except Exception:
                continue
    return data

# --- CREAR CSV ---
with open(OUTPUT_CSV, "w", newline="") as csvfile:
    fieldnames = [
        "split", "lote", "fecha", "interpolada",
        "lat", "lon", "ndvi",
        "precip_t", "precip_t_1", "precip_t_2", "precip_t_3"
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # --- RECORRER CADA SPLIT ---
    for split in SPLITS:
        split_path = os.path.join(BASE_DIR, split)
        if not os.path.exists(split_path):
            print(f"⚠️ Carpeta {split} no encontrada, se omite.")
            continue

        # --- RECORRER LOTES ---
        for lote in os.listdir(split_path):
            lote_path = os.path.join(split_path, lote)
            if not os.path.isdir(lote_path):
                continue

            # Cargar precipitaciones del lote
            precip_data = load_precip_data(lote)

            for filename in os.listdir(lote_path):
                if not filename.endswith(".tif"):
                    continue

                # Extraer fecha
                match = DATE_PATTERN.search(filename)
                if not match:
                    print(f"⚠️ No se encontró fecha en {filename}")
                    continue

                date_part = match.group(1)
                interpolada = "_interp" in filename
                file_path = os.path.join(lote_path, filename)
                date_obj = datetime.strptime(date_part, "%Y%m%d").date()

                # Obtener precipitaciones del día y 3 previos
                precip_t   = precip_data.get(date_obj, 0.0)
                precip_t_1 = precip_data.get(date_obj - timedelta(days=1), 0.0)
                precip_t_2 = precip_data.get(date_obj - timedelta(days=2), 0.0)
                precip_t_3 = precip_data.get(date_obj - timedelta(days=3), 0.0)

                # --- LECTURA DE RASTER ---
                with rasterio.open(file_path) as src:
                    ndvi = src.read(1).astype(np.float32)
                    transform = src.transform

                    # Crear rejilla de índices
                    rows, cols = ndvi.shape
                    for y in range(0, rows, DOWNSAMPLE_FACTOR):
                        for x in range(0, cols, DOWNSAMPLE_FACTOR):
                            value = ndvi[y, x]
                            if np.isnan(value) or value < -1 or value > 1:
                                continue  # filtro de valores inválidos

                            # Convertir (x, y) a (lon, lat)
                            lon, lat = transform * (x, y)

                            writer.writerow({
                                "split": split,
                                "lote": lote,
                                "fecha": date_part,
                                "interpolada": int(interpolada),
                                "lat": lat,
                                "lon": lon,
                                "ndvi": float(value),
                                "precip_t": precip_t,
                                "precip_t_1": precip_t_1,
                                "precip_t_2": precip_t_2,
                                "precip_t_3": precip_t_3
                            })

                print(f"✅ Procesado: {split}/{lote}/{filename}")

print(f"\n📁 CSV final generado en: {OUTPUT_CSV}")
