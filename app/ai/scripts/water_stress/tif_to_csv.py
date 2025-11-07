import os
import glob
import rasterio
import numpy as np
import pandas as pd
from datetime import datetime

# --- CONFIGURACIÓN ---
tif_dir = "ndvi_series"  # Carpeta con tus NDVI_YYYY-MM-DD.tif
output_csv = "dataset_ndvi.csv"

# --- CONVERSIÓN DE TIFF A DATAFRAME ---
def extract_ndvi_from_tif(tif_path):
    """Convierte un archivo NDVI .tif en un DataFrame con lat, lon, ndvi y fecha."""
    with rasterio.open(tif_path) as src:
        ndvi = src.read(1)
        ndvi = np.where((ndvi < -1) | (ndvi > 1), np.nan, ndvi)
        rows, cols = np.where(~np.isnan(ndvi))
        xs, ys = src.transform * (cols, rows)

        fecha_str = os.path.basename(tif_path).split("_")[1].split(".")[0]
        fecha = datetime.strptime(fecha_str, "%Y-%m-%d").date()

        df = pd.DataFrame({
            "lat": ys,
            "lon": xs,
            "ndvi": ndvi[rows, cols],
            "fecha": fecha
        })
        return df

# Procesar todos los archivos .tif
all_dfs = [extract_ndvi_from_tif(p) for p in sorted(glob.glob(os.path.join(tif_dir, "NDVI_*.tif")))]
df = pd.concat(all_dfs, ignore_index=True)

# --- GENERAR COLUMNAS TEMPORALES (lags y futuro) ---
df = df.sort_values(["lat", "lon", "fecha"])
df["ndvi_t"] = df["ndvi"]

# Para cada píxel, agregamos NDVI de 3 y 6 días atrás y 3 días adelante (futuro)
df["ndvi_t-3"] = df.groupby(["lat", "lon"])["ndvi_t"].shift(1)
df["ndvi_t-6"] = df.groupby(["lat", "lon"])["ndvi_t"].shift(2)
df["ndvi_futuro"] = df.groupby(["lat", "lon"])["ndvi_t"].shift(-1)

# Limpiamos filas con NaN (sin histórico suficiente)
df = df.dropna(subset=["ndvi_t-3", "ndvi_t-6", "ndvi_futuro"])

# --- GUARDAR ---
df.to_csv(output_csv, index=False)
print(f"✅ Dataset generado: {output_csv}")
print(df.head())
