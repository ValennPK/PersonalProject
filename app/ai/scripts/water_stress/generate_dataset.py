import os
import csv
from datetime import datetime
import ee
import requests
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from app.ai.scripts.water_stress.utilities import image_to_url, fetch_NDVI_ee_image
import matplotlib.pyplot as plt
import geemap


load_dotenv()
try:
    ee.Initialize(project=os.getenv("PROJECT_ID"))
except Exception:
    ee.Authenticate()
    ee.Initialize(project=os.getenv("PROJECT_ID"))

# --- Configuración de lotes ---
LOCALES = [
    # {
    #     "id": "lote_1",
    #     "lat1": -32.729761460992094,
    #     "lon1": -62.340202331542976,
    #     "lat2": -32.858479118815666,
    #     "lon2": -62.18982696533204
    # },
    # {
    #     "id": "lote_2",
    #     "lat1": -31.508781018208474,
    #     "lon1": -62.85621643066407,
    #     "lat2": -31.707607067034466,
    #     "lon2": -62.530746459960945
    # },
    {
        "id": "lote_3",
        "lat1": -31.925299894838293,
        "lon1": -63.10340881347657,
        # "lat2": -32.310290728069184,
        "lat2": -32.000000000000000,
        # "lon2": -62.65571594238282
        "lon2": -63.200000000000000
    }
]

GRID_STEP = 0.005  # tamaño del paso en grados (~1 km)
START_DATE = '2025-10-01'
END_DATE = '2025-11-01'
OUTPUT_PATH = 'app/ai/datasets/water_stress/dataset_wsi.csv'


def get_ndvi_grid(lat1, lon1, lat2, lon2, start_date, end_date, grid_step=0.01):
    """
    Obtiene valores de NDVI sobre una grilla de puntos dentro de una región rectangular.
    Retorna una lista de tuplas (lat, lon, ndvi).
    """

    # Crear el rectángulo
    region = ee.Geometry.Rectangle([lon1, lat1, lon2, lat2])

    # Cargar la colección MODIS NDVI
    dataset = (
        ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.listContains("system:band_names", "B4"))
        .filter(ee.Filter.listContains("system:band_names", "B8"))
    )

    def add_ndvi(img):
        ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
        return img.addBands(ndvi)
    
    dataset = dataset.map(add_ndvi)

    # Promedio temporal de todo el período
    ndvi_image = dataset.mean()

    # Crear las listas de coordenadas para el grid
    lats = np.arange(min(lat1, lat2), max(lat1, lat2), grid_step)
    lons = np.arange(min(lon1, lon2), max(lon1, lon2), grid_step)

    results = []

    # Iterar sobre la grilla de puntos
    for lat in lats:
        for lon in lons:
            point = ee.Geometry.Point([lon, lat])
            ndvi_value = ndvi_image.select("NDVI").reduceRegion(
                reducer=ee.Reducer.first(),
                geometry=point,
                scale=30
            ).get("NDVI")
            try:
                value = ndvi_value.getInfo()
                if value is not None:
                    results.append((lat, lon, value))
            except Exception:
                continue

    return results, ndvi_image

def show_ndvi_map(results):
    """
    Muestra dos mapas NDVI:
    1. Interpolado a partir de los puntos muestreados (results)
    2. NDVI promedio real de la imagen satelital (ndvi_image)
    """
    if not results:
        print("No hay datos NDVI para mostrar.")
        return

    import matplotlib.pyplot as plt
    import numpy as np
    import geemap
    import ee

    # --- 1. Convertir puntos muestreados a grilla ---
    sample_points = results[0]  # lista de tuplas (lat, lon, ndvi)

    lats = np.array([r[0] for r in sample_points])
    lons = np.array([r[1] for r in sample_points])
    ndvi_vals = np.array([r[2] for r in sample_points])

    grid_lat = np.unique(lats)
    grid_lon = np.unique(lons)
    ndvi_grid = np.zeros((len(grid_lat), len(grid_lon)))

    for (lat, lon, val) in sample_points:
        i = np.where(grid_lat == lat)[0][0]
        j = np.where(grid_lon == lon)[0][0]
        ndvi_grid[i, j] = val


    # --- 3. Mostrar ambos mapas lado a lado ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Mapa interpolado
    im1 = axes[0].imshow(
        ndvi_grid,
        cmap="RdYlGn",
        origin="lower",
        extent=[min(grid_lon), max(grid_lon), min(grid_lat), max(grid_lat)]
    )
    axes[0].set_title("NDVI interpolado (muestreo)")
    axes[0].set_xlabel("Longitud")
    axes[0].set_ylabel("Latitud")
    fig.colorbar(im1, ax=axes[0], label="NDVI")

    plt.tight_layout()
    plt.show()




# --- Función: obtener ET₀ y precipitación de Open-Meteo ---
def get_meteo(lat, lon, date):
    url = (
        f"https://archive-api.open-meteo.com/v1/era5?"
        f"latitude={lat}&longitude={lon}&start_date={date}&end_date={date}"
        f"&daily=et0_fao_evapotranspiration,precipitation_sum"
        f"&timezone=America/Argentina/Buenos_Aires"
    )
    r = requests.get(url)
    if r.status_code != 200:
        return None, None
    data = r.json()
    try:
        et0 = data['daily']['et0_fao_evapotranspiration'][0]
        precip = data['daily']['precipitation_sum'][0]
        return et0, precip
    except (KeyError, IndexError):
        return None, None


# --- Generar dataset completo ---
def build_dataset():
    for lote in LOCALES:
        lote_id = lote["id"]
        lat1 = lote["lat1"]
        lon1 = lote["lon1"]
        lat2 = lote["lat2"]
        lon2 = lote["lon2"]

        ndvi_points = get_ndvi_grid(lat1, lon1, lat2, lon2, START_DATE, END_DATE, GRID_STEP)

        print(f"Lote {lote_id}: obtenido {len(ndvi_points[0])} puntos NDVI.")

        # start_date = datetime.strptime(START_DATE, "%Y-%m-%d")
        # start_date = start_date.strftime("%Y%m%d")
        # end_date = datetime.strptime(END_DATE, "%Y-%m-%d")
        # end_date = end_date.strftime("%Y%m%d")

        # ndvi_result = fetch_NDVI_ee_image(lat1, lon1, lat2, lon2, start_date, end_date)

        # if not ndvi_result["success"]:
        #     print(f"Error fetching NDVI image for {lote_id}: {ndvi_result['error']}")
        #     continue
        # else:
        #     ndvi_img, region, img_date = ndvi_result["data"]
        #     ndvi_url = image_to_url(ndvi_img, region)
        #     print(f"NDVI image URL for {lote_id}: {ndvi_url}")

        # print("Ejemplo de results:", ndvi_points[:5])

        show_ndvi_map(ndvi_points)
        

#     os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

#     with open(OUTPUT_PATH, 'w', newline='') as csvfile:
#         fieldnames = ['lote_id', 'lat', 'lon', 'date', 'ndvi', 'et0', 'precip', 'wsi']
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writeheader()

#         print("Generando dataset con cálculo de WSI...\n")

#         for lote in LOCALES:
#             lote_id = lote["id"]
#             lats = np.arange(min(lote["lat1"], lote["lat2"]), max(lote["lat1"], lote["lat2"]), GRID_STEP)
#             lons = np.arange(min(lote["lon1"], lote["lon2"]), max(lote["lon1"], lote["lon2"]), GRID_STEP)
#             # print (f"grilla: {len(grid)} puntos para {lote_id}")
#             print (f"grilla: {len(lats)*len(lons)} puntos para {lote_id}")

#             for lat in lats:
#                 for lon in lons:
#                     try:
#                         ndvi = get_ndvi(lat, lon, START_DATE, END_DATE)
#                         et0, precip = get_meteo(lat, lon, END_DATE)
#                         if ndvi is None or et0 is None:
#                             continue

#                         # --- Calcular WSI ---
#                         wsi = calc_water_stress_scalar(ndvi, et0)

#                         writer.writerow({
#                             'lote_id': lote_id,
#                             'lat': lat,
#                             'lon': lon,
#                             'date': END_DATE,
#                             'ndvi': ndvi,
#                             'et0': et0,
#                             'precip': precip,
#                             'wsi': wsi
#                         })

#                         print (f"Guardado: Lote {lote_id}, Punto ({lat},{lon}), NDVI: {ndvi}, ET0: {et0}, Precip: {precip}, WSI: {wsi}")
#                     except Exception as e:
#                         print(f"Error en punto ({lat},{lon}): {e}")
#                         continue


if __name__ == '__main__':
    build_dataset()
