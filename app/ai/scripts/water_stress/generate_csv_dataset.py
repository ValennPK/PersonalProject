import os
import re
import csv
from datetime import datetime
import rasterio
from rasterio.transform import xy
import numpy as np

# --- CONFIGURACIÓN ---
BASE_DIR = "app/ai/datasets/water_stress"
OUTPUT_CSV = os.path.join(BASE_DIR, "ndvi_dataset.csv")

# Patrones posibles
PATTERN_REAL = re.compile(r"(lote_\d+)_NDVI_(\d{8})")
PATTERN_INTERP = re.compile(r"NDVI_(\d{8})_interp")

def get_image_metadata(filename, lote_fallback=None):
    """Detecta si una imagen es real o interpolada, y extrae lote + fecha."""
    match_real = PATTERN_REAL.search(filename)
    match_interp = PATTERN_INTERP.search(filename)
    
    if match_real:
        lote = match_real.group(1)
        fecha = datetime.strptime(match_real.group(2), "%Y%m%d").date()
        tipo = "real"
    elif match_interp:
        lote = lote_fallback or "unknown"
        fecha = datetime.strptime(match_interp.group(1), "%Y%m%d").date()
        tipo = "interpolada"
    else:
        return None
    return lote, fecha, tipo

# --- GENERACIÓN DEL CSV ---
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "split", "lote", "fecha", "tipo", "lat", "lon", "ndvi"
    ])
    writer.writeheader()

    for split in ["train", "val", "test"]:
        split_dir = os.path.join(BASE_DIR, split)
        if not os.path.isdir(split_dir):
            print(f"⚠️ Carpeta no encontrada: {split_dir}")
            continue

        for lote_name in os.listdir(split_dir):
            lote_path = os.path.join(split_dir, lote_name)
            if not os.path.isdir(lote_path):
                continue

            for filename in os.listdir(lote_path):
                if not filename.lower().endswith(".tif"):
                    continue

                meta = get_image_metadata(filename, lote_name)
                if not meta:
                    print(f"❌ No se pudo parsear: {filename}")
                    continue
                lote, fecha, tipo = meta

                filepath = os.path.join(lote_path, filename)
                try:
                    with rasterio.open(filepath) as src:
                        ndvi = src.read(1).astype(np.float32)
                        ndvi[ndvi == src.nodata] = np.nan  # eliminamos valores nulos
                        rows, cols = np.where(~np.isnan(ndvi))
                        for r, c in zip(rows, cols):
                            lat, lon = xy(src.transform, r, c)
                            writer.writerow({
                                "split": split,
                                "lote": lote,
                                "fecha": fecha,
                                "tipo": tipo,
                                "lat": lat,
                                "lon": lon,
                                "ndvi": float(ndvi[r, c])
                            })
                except Exception as e:
                    print(f"⚠️ Error procesando {filename}: {e}")

print(f"✅ Dataset final generado en: {OUTPUT_CSV}")
