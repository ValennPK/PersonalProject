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

    return ndvi_mean, region  # devolvemos imagen y región para generar thumbnail luego


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






