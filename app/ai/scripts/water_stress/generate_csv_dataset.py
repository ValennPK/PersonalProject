import os
import re
import csv
import rasterio
import numpy as np
from datetime import datetime, timedelta

# --- CONFIGURACIÓN ---
BASE_DIR = "app/ai/datasets/water_stress"
OUTPUT_CSV = os.path.join(BASE_DIR, "ndvi_dataset.csv")

DATE_PATTERN = re.compile(r"(\d{8})")
SPLITS = ["train", "val", "test"]
DOWNSAMPLE_FACTOR = 20  # muestreo espacial

# --- FUNCIONES AUXILIARES ---

# def load_data(lote):
#     """Carga los datos meteorológicos del CSV del lote."""
#     weather_path = os.path.join(BASE_DIR, "weather", f"{lote}.csv")
#     if not os.path.exists(weather_path):
#         print(f"⚠️ No se encontró archivo de clima para {lote}")
#         return {}

#     data = {}
#     with open(weather_path, "r", encoding="utf-8") as f:
#         reader = csv.DictReader(f)
#         for row in reader:
#             try:
#                 date = datetime.strptime(row["fecha"], "%Y-%m-%d").date()
#                 data[date] = {
#                     "precipitation_mm": float(row.get("precipitacion_mm", 0.0)),
#                     "temp_mean_c": float(row.get("temp_mean_c", 0.0)),
#                     "temp_max_c": float(row.get("temp_max_c", 0.0)),
#                     "temp_min_c": float(row.get("temp_min_c", 0.0)),
#                     "radiacion_sw_mj_m2": float(row.get("radiacion_sw_mj_m2", 0.0)),
#                     "vel_viento_m_s": float(row.get("vel_viento_m_s", 0.0)),
#                     "humedad_relativa_pct": float(row.get("humedad_relativa_pct", 0.0)),
#                     "et0_mm": float(row.get("et0_mm", 0.0)),
#                 }
#             except Exception as e:
#                 print(f"Error leyendo fila en {weather_path}: {e}")
#     return data


# def get_weather_value(weather_data, date, key):
#     """Obtiene un valor numérico de una fecha; si no existe, devuelve 0.0."""
#     if date in weather_data:
#         return weather_data[date].get(key, 0.0)
#     return 0.0


# --- CREAR CSV ---
with open(OUTPUT_CSV, "w", newline="") as csvfile:
    fieldnames = [
        "split", "lote", "fecha", "interpolada",
        "lat", "lon", "ndvi", 
        # "ndvi_t_1",
        # "precip_t", "precip_t_1", "precip_t_2", "precip_t_3", "precip_t_4", "precip_t_5",
        # "temp_t", "temp_t_1", "temp_t_2", "temp_t_3", "temp_t_4", "temp_t_5",
        # "hum_t", "hum_t_1", "hum_t_2", "hum_t_3", "hum_t_4", "hum_t_5",
        # "et0_t", "et0_t_1", "et0_t_2", "et0_t_3", "et0_t_4", "et0_t_5",
        # "rad_t", "rad_t_1", "rad_t_2", "rad_t_3", "rad_t_4", "rad_t_5",
        # "viento_t", "viento_t_1", "viento_t_2", "viento_t_3", "viento_t_4", "viento_t_5",
    ]

    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for split in SPLITS:
        split_path = os.path.join(BASE_DIR, split)
        if not os.path.exists(split_path):
            print(f"⚠️ Carpeta {split} no encontrada.")
            continue

        for lote in os.listdir(split_path):
            lote_path = os.path.join(split_path, lote)
            if not os.path.isdir(lote_path):
                continue

            # weather_data = load_data(lote)

            # 🔹 Guardar NDVI previos por coordenada
            # ndvi_prev = {}

            # Procesar archivos ordenados por fecha
            tif_files = sorted(
                [f for f in os.listdir(lote_path) if f.endswith(".tif")],
                key=lambda n: datetime.strptime(DATE_PATTERN.search(n).group(1), "%Y%m%d")
            )

            for filename in tif_files:
                match = DATE_PATTERN.search(filename)
                if not match:
                    print(f"⚠️ No se encontró fecha en {filename}")
                    continue

                date_obj = datetime.strptime(match.group(1), "%Y%m%d").date()
                interpolada = int("_interp" in filename)
                file_path = os.path.join(lote_path, filename)

                # --- Obtener variables climáticas ---
                # def seq(var):
                #     return [get_weather_value(weather_data, date_obj - timedelta(days=i), var) for i in range(6)]

                # precip_vals = seq("precipitation_mm")
                # temp_vals = seq("temp_mean_c")
                # hum_vals = seq("humedad_relativa_pct")
                # et0_vals = seq("et0_mm")
                # rad_vals = seq("radiacion_sw_mj_m2")
                # viento_vals = seq("vel_viento_m_s")

                # --- LECTURA DE NDVI ---
                with rasterio.open(file_path) as src:
                    ndvi = src.read(1).astype(np.float32)
                    transform = src.transform

                    rows, cols = ndvi.shape
                    for y in range(0, rows, DOWNSAMPLE_FACTOR):
                        for x in range(0, cols, DOWNSAMPLE_FACTOR):
                            val = ndvi[y, x]
                            if np.isnan(val) or val < -1 or val > 1:
                                continue
                            lon, lat = transform * (x, y)

                            # 🔸 Obtener NDVI del día anterior si existe
                            # prev_key = (round(lat, 5), round(lon, 5))
                            # ndvi_t_1 = ndvi_prev.get(prev_key, np.nan)

                            # Guardar NDVI actual para uso futuro
                            # ndvi_prev[prev_key] = val

                            writer.writerow({
                                "split": split,
                                "lote": lote,
                                "fecha": date_obj.strftime("%Y%m%d"),
                                "interpolada": interpolada,
                                "lat": lat,
                                "lon": lon,
                                "ndvi": val,
                                # "ndvi_t_1": ndvi_t_1 if not np.isnan(ndvi_t_1) else "",
                                # **{f"precip_t_{i if i > 0 else ''}".rstrip('_'): precip_vals[i] for i in range(6)},
                                # **{f"temp_t_{i if i > 0 else ''}".rstrip('_'): temp_vals[i] for i in range(6)},
                                # **{f"hum_t_{i if i > 0 else ''}".rstrip('_'): hum_vals[i] for i in range(6)},
                                # **{f"et0_t_{i if i > 0 else ''}".rstrip('_'): et0_vals[i] for i in range(6)},
                                # **{f"rad_t_{i if i > 0 else ''}".rstrip('_'): rad_vals[i] for i in range(6)},
                                # **{f"viento_t_{i if i > 0 else ''}".rstrip('_'): viento_vals[i] for i in range(6)},
                            })

                print(f"✅ Procesado: {split}/{lote}/{filename}")

print(f"\n📁 CSV final generado en: {OUTPUT_CSV}")
