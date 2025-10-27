import math
import requests
import pandas as pd
import ee
from datetime import datetime
from dotenv import load_dotenv
import os

import requests

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
    if len(start) != 8 or len(end) != 8 or not (start.isdigit() and end.isdigit()):
        raise ValueError("Las fechas deben estar en formato 'YYYYMMDD'.")
    if start > end:
        raise ValueError("La fecha de inicio debe ser anterior a la fecha de fin.")

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

def fetch_NDVI_ee_image(lat1, lon1, lat2, lon2, start, end, cloud_thresh=50):
    """
    Devuelve un objeto ee.Image con el NDVI promedio
    de un rectángulo definido por dos puntos.
    """
    load_dotenv()
    try:
        ee.Initialize(project=os.getenv("PROJECT_ID"))
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=os.getenv("PROJECT_ID"))

    start_dt = datetime.strptime(start, "%Y%m%d").strftime("%Y-%m-%d")
    end_dt   = datetime.strptime(end, "%Y%m%d").strftime("%Y-%m-%d")

    region = ee.Geometry.Rectangle([lon1, lat1, lon2, lat2])

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR")
        .filterBounds(region)
        .filterDate(start_dt, end_dt)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_thresh))
    )

    def add_ndvi(img):
        ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
        return img.addBands(ndvi)

    ndvi_collection = collection.map(add_ndvi)
    ndvi_mean = ndvi_collection.select("NDVI").mean()

    # Obtener una fecha representativa para la colección (por ejemplo, la fecha media de las imágenes)
    # Calculamos el promedio de las fechas de adquisición como timestamp medio.
    def img_time(img):
        return ee.Image(img).get('system:time_start')

    times = ndvi_collection.aggregate_array('system:time_start')
    # Si no hay imágenes, devolvemos None para la fecha
    ndvi_date = None
    if times.size().getInfo() > 0:
        # Convertir lista de millis a ee.List de números y tomar el promedio
        times_list = ee.List(times)
        mean_time = ee.Number(times_list.reduce(ee.Reducer.mean()))
        # Formatear la fecha como string 'YYYY-MM-dd'
        ndvi_date = ee.Date(mean_time).format('YYYY-MM-dd')

    return ndvi_mean, region, ndvi_date  # devolvemos imagen, región y la fecha representativa


def fetch_NDVI_stac(lat1, lon1, lat2, lon2, start, end, max_cloud=50, out_dir=None, collection_name='sentinel-2-l2a'):
    """
    Descarga una escena Sentinel-2 vía STAC (Planetary Computer), calcula NDVI localmente,
    guarda un PNG en el filesystem (por defecto en app/static/ai/) y devuelve la URL relativa
    junto con la fecha de adquisición (YYYY-MM-DD).

    Nota: importa paquetes opcionales (pystac-client, planetary_computer, rasterio, numpy).
    """
    # Imports locales para que el resto del módulo no requiera estas librerías
    from pystac_client import Client
    import planetary_computer as pc
    import rasterio
    import numpy as np
    from rasterio.enums import Resampling
    from rasterio.plot import reshape_as_image

    # Prepare bbox (minx, miny, maxx, maxy) and datetime window
    min_lon = min(lon1, lon2)
    max_lon = max(lon1, lon2)
    min_lat = min(lat1, lat2)
    max_lat = max(lat1, lat2)
    bbox = [min_lon, min_lat, max_lon, max_lat]

    # Accept start/end in either 'YYYYMMDD' or 'YYYY-MM-DD' (or datetime)
    def _to_iso(s):
        if s is None:
            return None
        if isinstance(s, datetime):
            return s.strftime('%Y-%m-%d')
        if isinstance(s, str):
            if len(s) == 8 and s.isdigit():
                return datetime.strptime(s, "%Y%m%d").strftime('%Y-%m-%d')
            return s
        return str(s)

    start_iso = _to_iso(start)
    end_iso = _to_iso(end)
    datetime_range = f"{start_iso}/{end_iso}"

    # STAC search on Planetary Computer
    client = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    # Try search with requested collection first; if that fails (invalid collection id or API error)
    # fall back to a broader search without collections.
    items = []
    tried_collections = []
    if collection_name:
        # Try a few common MODIS/STAC collection ids if the provided one yields no results
        candidate_collections = [
            collection_name,
            'modis-061-mod13q1',
            'MOD13Q1',
            'MOD13Q1.061',
            'MODIS/061/MOD13Q1'
        ]
        for coll in candidate_collections:
            if coll in tried_collections:
                continue
            tried_collections.append(coll)
            try:
                search = client.search(
                    collections=[coll],
                    bbox=bbox,
                    datetime=datetime_range,
                    query={"eo:cloud_cover": {"lt": max_cloud}},
                    limit=10,
                )
                items = list(search.get_items())
                if items:
                    collection_name = coll
                    break
            except Exception:
                # ignore and try next candidate
                continue

    # If no items found via collection-specific searches, try a generic search
    if not items:
        try:
            search = client.search(
                bbox=bbox,
                datetime=datetime_range,
                query={"eo:cloud_cover": {"lt": max_cloud}},
                limit=10,
            )
            items = list(search.get_items())
        except Exception as e:
            raise RuntimeError(f"STAC search failed for collection candidates {tried_collections}: {e}")
    if not items:
        raise RuntimeError(f"No items found for bbox/date range (collection={collection_name})")

    # Compute the newest acquisition date among the found items
    item_dates = [it.datetime for it in items if getattr(it, 'datetime', None) is not None]
    newest_date = None
    if item_dates:
        newest_date_dt = max(item_dates)
        newest_date = newest_date_dt.strftime('%Y-%m-%d')

    # Choose best item (lowest cloud cover) as a default for actual asset download
    items_sorted = sorted(items, key=lambda it: (it.properties.get('eo:cloud_cover', 100)))
    item = items_sorted[0]

    # Detect if the item already contains an NDVI asset (e.g., MOD13Q1)
    ndvi_asset = item.assets.get('NDVI') or item.assets.get('ndvi')
    if ndvi_asset is not None:
        # MODIS-like product: NDVI asset exists and usually needs scaling (e.g., 0.0001)
        ndvi_href = pc.sign(ndvi_asset.href)
        with rasterio.Env():
            with rasterio.open(ndvi_href) as r_ndvi:
                ndvi_arr = r_ndvi.read(1).astype('float32')
                meta = r_ndvi.meta.copy()
        # Apply MODIS scaling if values are integer (heuristic)
        if meta.get('dtype', '').startswith('int') or ndvi_arr.max() > 1:
            scale = 0.0001
            ndvi = ndvi_arr * scale
        else:
            ndvi = ndvi_arr

    else:
        # Sentinel-like product: read red/nir and compute NDVI
        red_asset = item.assets.get('B04') or item.assets.get('B04.jp2')
        nir_asset = item.assets.get('B08') or item.assets.get('B08.jp2')
        if red_asset is None or nir_asset is None:
            raise RuntimeError('Required assets (B04/B08 or NDVI) not found in item')

        # Sign URLs using planetary_computer
        red_href = pc.sign(red_asset.href)
        nir_href = pc.sign(nir_asset.href)

        # Read bands with rasterio (resample to match if needed)
        with rasterio.Env():
            with rasterio.open(red_href) as r_red, rasterio.open(nir_href) as r_nir:
                # Read first band
                red = r_red.read(1).astype('float32')
                nir = r_nir.read(1).astype('float32')
                meta = r_red.meta.copy()

                # If shapes differ, resample nir to red's shape
                if red.shape != nir.shape:
                    nir = r_nir.read(1, out_shape=red.shape, resampling=Resampling.bilinear).astype('float32')
                # Compute NDVI
                np.seterr(divide='ignore', invalid='ignore')
                ndvi = (nir - red) / (nir + red)
                ndvi = np.nan_to_num(ndvi, nan=0.0)
                ndvi = np.clip(ndvi, -1, 1)

    # Compute NDVI
    np.seterr(divide='ignore', invalid='ignore')
    ndvi = (nir - red) / (nir + red)
    ndvi = np.nan_to_num(ndvi, nan=0.0)
    ndvi = np.clip(ndvi, -1, 1)

    # Render as RGB-like PNG for quick display (map NDVI -1..1 to 0..255 palette)
    norm = ((ndvi + 1) / 2.0 * 255).astype('uint8')
    rgb = np.dstack([norm, norm, norm])

    # Save PNG to static folder
    if out_dir is None:
        out_dir = os.path.join(os.getcwd(), 'app', 'static', 'ai')
    os.makedirs(out_dir, exist_ok=True)
    filename = f"ndvi_{newest_date if newest_date else 'unknown'}.png"
    out_path = os.path.join(out_dir, filename)

    # Use rasterio to save PNG with geo metadata removed (simple view)
    import imageio
    imageio.imwrite(out_path, rgb)

    # Return relative URL for Flask (static path) and acquisition date
    rel_url = f"/static/ai/{filename}"
    return rel_url, newest_date


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






