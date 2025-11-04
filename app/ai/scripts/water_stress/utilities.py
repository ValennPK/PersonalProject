import math
import requests
import ee
from datetime import datetime
from dotenv import load_dotenv
import os

def fetch_NASA_data(lat, lon, start, end):
    """
    Obtiene datos diarios de la NASA POWER API.

    Parámetros:
        lat, lon (float): Coordenadas en grados decimales
        start, end (str): Fechas en formato 'YYYYMMDD'

    Retorna:
        dict: Datos en formato JSON
    """
    if not (-90 <= lat <= 90):
        raise ValueError("La latitud debe estar entre -90 y 90 grados.")
    if not (-180 <= lon <= 180):
        raise ValueError("La longitud debe estar entre -180 y 180 grados.")
    
    base = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": "ALLSKY_SFC_SW_DWN,T2M,T2M_MAX,T2M_MIN,RH2M,PS,WS10M",
        "community": "AG",
        "latitude": lat,
        "longitude": lon,
        "start": start,
        "end": end,
        "format": "JSON"
    }

    resp = requests.get(base, params=params)

    if resp.status_code != 200:
        raise Exception(f"Error en la solicitud: {resp.status_code}")

    return resp.json()

def calc_et0_fao56(data):
    # Extraer datos
    params = data["properties"]["parameter"]
    dates = list(params["T2M"].keys())

    results = {}
    for date in dates:
        # Variables meteorológicas
        t_max = params["T2M_MAX"][date]
        t_min = params["T2M_MIN"][date]
        t_mean = params["T2M"][date]
        rh_mean = params["RH2M"][date]
        ws_10m = params["WS10M"][date]
        rs = params["ALLSKY_SFC_SW_DWN"][date]  # MJ/m²/day
        p = params["PS"][date]                  # kPa

        # Convertir viento a 2 m (FAO recomienda: u2 = u10 * 0.748)
        u2 = ws_10m * 0.748

        # Saturation vapor pressure
        es_tmax = 0.6108 * math.exp((17.27 * t_max) / (t_max + 237.3))
        es_tmin = 0.6108 * math.exp((17.27 * t_min) / (t_min + 237.3))
        es = (es_tmax + es_tmin) / 2

        # Actual vapor pressure
        ea = es * (rh_mean / 100.0)

        # Slope of vapor pressure curve (kPa/°C)
        delta = 4098 * (0.6108 * math.exp((17.27 * t_mean) / (t_mean + 237.3))) / ((t_mean + 237.3) ** 2)

        # Psychrometric constant (kPa/°C)
        gamma = 0.000665 * p

        # Radiación neta simplificada (suponemos 0.77 como coeficiente de albedo medio)
        rn = 0.77 * rs  # radiación neta MJ/m²/día
        g = 0  # flujo de calor al suelo despreciable (diario)

        # Penman–Monteith (FAO 56)
        et0 = (0.408 * delta * (rn - g) + gamma * (900 / (t_mean + 273)) * u2 * (es - ea)) / (
            delta + gamma * (1 + 0.34 * u2)
        )

        results[date] = round(et0, 3)

    return results


def fetch_NDVI_ee_image(lat1, lon1, lat2, lon2, start, end, cloud_thresh=30):
    """
    Busca la imagen NDVI más reciente disponible desde end_date hacia atrás,
    dentro del rango definido por start_date.
    """
    from datetime import datetime, timedelta
    import ee, os
    from dotenv import load_dotenv

    load_dotenv()
    try:
        ee.Initialize(project=os.getenv("PROJECT_ID"))
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=os.getenv("PROJECT_ID"))

    # Convertir fechas a formato datetime y luego a string ISO
    start_dt = datetime.strptime(start, "%Y%m%d")
    end_dt = datetime.strptime(end, "%Y%m%d")

    region = ee.Geometry.Rectangle([lon1, lat1, lon2, lat2])

    ndvi_mean = None
    ndvi_date = None

    # 🔁 Buscar desde la fecha más reciente hacia atrás hasta el inicio
    days_back = (end_dt - start_dt).days
    for delta in range(days_back + 1):
        day_end = end_dt - timedelta(days=delta)
        day_start = day_end - timedelta(days=1)

        # Convertir a string en formato YYYY-MM-DD
        start_str = day_start.strftime("%Y-%m-%d")
        end_str = day_end.strftime("%Y-%m-%d")

        # Buscar imágenes Sentinel-2 disponibles en ese día
        collection = (
            ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
            .filterBounds(region)
            .filterDate(start_str, end_str)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_thresh))
            .filter(ee.Filter.listContains("system:band_names", "B4"))
            .filter(ee.Filter.listContains("system:band_names", "B8"))
        )

        size = collection.size().getInfo()

        if size > 0:
            # Encontramos una colección con imágenes disponibles
            def add_ndvi(img):
                ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
                return img.addBands(ndvi)

            ndvi_collection = collection.map(add_ndvi)
            ndvi_mean = ndvi_collection.select("NDVI").mean()

            # Calcular la fecha media
            times = ndvi_collection.aggregate_array('system:time_start')
            if times.size().getInfo() > 0:
                times_list = ee.List(times)
                mean_time = ee.Number(times_list.reduce(ee.Reducer.mean()))
                ndvi_date = ee.Date(mean_time).format('YYYY-MM-dd')
            break  

    if ndvi_mean is None:
        return {"success": False, "error": f"No available images between {start} and {end} in the selected region."}

    return {"success": True, "data": (ndvi_mean, region, ndvi_date)}


def calc_water_stress(ndvi_image, et0_value, region):
    """
    Calcula el Water Stress Index (WSI) a partir de una imagen NDVI y un valor diario de ET0.
    
    Parámetros:
        ndvi_image: ee.Image con NDVI (0-1)
        et0_value: float o ee.Number (evapotranspiración de referencia diaria)
        region: ee.Geometry (región de interés)

    Retorna:
        ee.Image: WSI (0 = sin estrés, 1 = estrés máximo)
    """
    # Recortar la imagen a la región
    ndvi = ndvi_image.clip(region)

    # Coeficiente de cultivo aproximado a partir de NDVI
    kc = ndvi.multiply(1.25).subtract(0.2).clamp(0.1, 1.2)  # evitar kc = 0

    # Evapotranspiración del cultivo
    etc = kc.multiply(et0_value)

    # Fracción de agua real transpirada basada en NDVI
    # Se normaliza NDVI a [0.1, 1] para que siempre haya algo de ETa
    frac = ndvi.clamp(0.1, 1)
    eta = etc.multiply(frac)

    # WSI = (ETc - ETa) / ETc
    wsi = etc.subtract(eta).divide(etc).clamp(0, 1)

    return wsi


def image_to_url(image, region, dimensions=512):
    """
    Convierte un ee.Image NDVI en una URL PNG para mostrar en un <img>.
    """
    thumb_params = {
        "min": 0,
        "max": 1,
        "dimensions": dimensions,
        "region": region.getInfo()["coordinates"],
        "palette": ["red", "yellow", "green"],
    }
    return image.getThumbURL(thumb_params)

def calc_water_stress_scalar(ndvi_mean, et0_value):
    """
    Calcula WSI a partir de NDVI promedio (float) y ET0 (float)
    """
    kc = max(0.1, min(1.2, 1.25*ndvi_mean - 0.2))
    etc = kc * et0_value
    frac = max(0.1, ndvi_mean)
    eta = etc * frac
    wsi = max(0, min(1, (etc - eta)/etc))
    return wsi



# def fetch_NDVI_stac(
#     lat1, lon1, lat2, lon2,
#     start, end,
#     max_cloud=50,
#     out_dir=None,
#     collection_name='sentinel-2-l2a'
# ):
#     """
#     Descarga una escena satelital vía STAC (Planetary Computer o Sentinel Harmonized),
#     calcula NDVI localmente, guarda un PNG en /static/ai/ y devuelve la URL relativa
#     junto con la fecha de adquisición (YYYY-MM-DD).

#     Compatible con:
#         - sentinel-2-l2a (Planetary Computer)
#         - COPERNICUS/S2_HARMONIZED (Earth Engine STAC proxy)
#         - MOD13Q1 / modis-061-mod13q1 (MODIS NDVI)

#     Requiere: pystac-client, planetary_computer, rasterio, numpy, imageio
#     """
#     from pystac_client import Client
#     import planetary_computer as pc
#     import rasterio
#     import numpy as np
#     from rasterio.enums import Resampling
#     import imageio
#     from datetime import datetime

#     # --- Preparar bbox (minx, miny, maxx, maxy)
#     bbox = [min(lon1, lon2), min(lat1, lat2), max(lon1, lon2), max(lat1, lat2)]

#     # --- Función auxiliar: convierte fecha a formato ISO
#     def _to_iso(s):
#         if s is None:
#             return None
#         if isinstance(s, datetime):
#             return s.strftime('%Y-%m-%d')
#         if isinstance(s, str):
#             if len(s) == 8 and s.isdigit():
#                 return datetime.strptime(s, "%Y%m%d").strftime('%Y-%m-%d')
#             return s
#         return str(s)

#     start_iso = _to_iso(start)
#     end_iso = _to_iso(end)
#     datetime_range = f"{start_iso}/{end_iso}"

#     # --- Inicializar cliente STAC de Planetary Computer
#     client = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

#     # --- Colecciones candidatas en orden de prioridad
#     candidate_collections = [
#         collection_name,
#         'COPERNICUS/S2_HARMONIZED'
#         # 'sentinel-2-l2a',
#         # 'modis-061-mod13q1',
#         # 'MOD13Q1',
#         # 'MOD13Q1.061',
#         # 'MODIS/061/MOD13Q1'
#     ]

#     items = []
#     tried = []

#     for coll in candidate_collections:
#         if coll in tried:
#             continue
#         tried.append(coll)
#         try:
#             search = client.search(
#                 collections=[coll],
#                 bbox=bbox,
#                 datetime=datetime_range,
#                 query={"eo:cloud_cover": {"lt": max_cloud}},
#                 limit=10,
#             )
#             items = list(search.get_items())
#             if items:
#                 collection_name = coll
#                 break
#         except Exception:
#             continue

#     if not items:
#         raise RuntimeError(f"No items found for bbox/date range (collections tried: {tried})")

#     # --- Obtener la fecha más reciente y el item con menor nubosidad
#     item_dates = [it.datetime for it in items if getattr(it, 'datetime', None) is not None]
#     newest_date = max(item_dates).strftime('%Y-%m-%d') if item_dates else None
#     item = sorted(items, key=lambda it: it.properties.get('eo:cloud_cover', 100))[0]

#     # --- Buscar asset NDVI (si ya está calculado, como en MODIS)
#     ndvi_asset = item.assets.get('NDVI') or item.assets.get('ndvi')
#     if ndvi_asset is not None:
#         ndvi_href = pc.sign(ndvi_asset.href)
#         with rasterio.Env(), rasterio.open(ndvi_href) as r_ndvi:
#             ndvi = r_ndvi.read(1).astype('float32')
#             meta = r_ndvi.meta.copy()
#         if meta.get('dtype', '').startswith('int') or ndvi.max() > 1:
#             ndvi *= 0.0001  # Escalado MODIS típico
#         ndvi = np.clip(ndvi, -1, 1)
#     else:
#         # --- Leer bandas Sentinel (B04 y B08)
#         red_asset = item.assets.get('B04') or item.assets.get('B04.jp2')
#         nir_asset = item.assets.get('B08') or item.assets.get('B08.jp2')

#         if not red_asset or not nir_asset:
#             raise RuntimeError(f"Required assets (B04/B08) not found in item for {collection_name}")

#         red_href = pc.sign(red_asset.href)
#         nir_href = pc.sign(nir_asset.href)

#         with rasterio.Env(), rasterio.open(red_href) as r_red, rasterio.open(nir_href) as r_nir:
#             red = r_red.read(1).astype('float32')
#             nir = r_nir.read(1).astype('float32')

#             if red.shape != nir.shape:
#                 nir = r_nir.read(1, out_shape=red.shape, resampling=Resampling.bilinear)

#             np.seterr(divide='ignore', invalid='ignore')
#             ndvi = (nir - red) / (nir + red)
#             ndvi = np.nan_to_num(ndvi, nan=0.0)
#             ndvi = np.clip(ndvi, -1, 1)

#     # --- Normalizar NDVI para visualización (escala 0–255)
#     norm = ((ndvi + 1) / 2.0 * 255).astype('uint8')
#     rgb = np.dstack([norm] * 3)

#     # --- Guardar PNG en carpeta estática
#     if out_dir is None:
#         out_dir = os.path.join(os.getcwd(), 'app', 'static', 'ai')
#     os.makedirs(out_dir, exist_ok=True)
#     filename = f"ndvi_{collection_name.replace('/', '_')}_{newest_date or 'unknown'}.png"
#     out_path = os.path.join(out_dir, filename)
#     imageio.imwrite(out_path, rgb)

#     rel_url = f"/static/ai/{filename}"
#     return rel_url, newest_date

