import os
import re
from datetime import datetime, timedelta
import numpy as np
import rasterio

# --- CONFIGURACIÓN ---
DATASET_DIR = "app/ai/datasets/water_stress/lote_6"
DATE_PATTERN = re.compile(r"NDVI_(\d{8})\.tif")  # Nombres tipo NDVI_20231125.tif
MIN_GAP_DAYS = 4   # Mínimo para considerar interpolación
MAX_GAP_DAYS = 10   # Máximo permitido para interpolar

# --- FUNCIONES AUXILIARES ---

def list_images_sorted():
    """Lista y ordena las imágenes por fecha extraída del nombre."""
    files = []
    for f in os.listdir(DATASET_DIR):
        if f.endswith(".tif") and "_interp" not in f:
            match = DATE_PATTERN.search(f)
            if match:
                date = datetime.strptime(match.group(1), "%Y%m%d")
                files.append((date, os.path.join(DATASET_DIR, f)))
    files.sort(key=lambda x: x[0])
    return files


def interpolate_images(img1_path, img2_path, date1, date2, output_dir):
    with rasterio.open(img1_path) as src1, rasterio.open(img2_path) as src2:
        ndvi1 = src1.read(1).astype(np.float32)
        ndvi2 = src2.read(1).astype(np.float32)
        meta = src1.meta.copy()

    days_gap = (date2 - date1).days
    if days_gap < MIN_GAP_DAYS or days_gap > MAX_GAP_DAYS:
        return None

    mid_date = date1 + timedelta(days=days_gap // 2)

    # Interpolación ponderada (por distancia temporal)
    weight = 0.5  # punto medio
    ndvi_interp = ndvi1 * (1 - weight) + ndvi2 * weight

    output_path = os.path.join(output_dir, f"NDVI_{mid_date.strftime('%Y%m%d')}_interp.tif")
    meta.update(dtype=rasterio.float32, compress='lzw')
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(ndvi_interp, 1)

    print(f"🟢 Interpolación creada: {output_path}")
    return output_path



# --- SCRIPT PRINCIPAL ---
if __name__ == "__main__":
    images = list_images_sorted()
    print(f"🔎 Se encontraron {len(images)} imágenes NDVI ordenadas por fecha.")

    if len(images) < 2:
        print("⚠️ No hay suficientes imágenes para interpolar.")
        exit()

    interpolated = []
    for (date1, path1), (date2, path2) in zip(images[:-1], images[1:]):
        gap = (date2 - date1).days
        print(f"⏳ Intervalo entre {date1.strftime('%Y-%m-%d')} y {date2.strftime('%Y-%m-%d')}: {gap} días")

        if MIN_GAP_DAYS <= gap <= MAX_GAP_DAYS:
            out = interpolate_images(path1, path2, date1, date2, DATASET_DIR)
            if out:
                interpolated.append(out)

    print(f"✅ Total de interpolaciones generadas: {len(interpolated)}")
