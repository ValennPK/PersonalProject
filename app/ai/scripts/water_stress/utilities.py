import math
import requests
import pandas as pd
import ee
from datetime import datetime
from dotenv import load_dotenv
import os

import requests

def fetch_NASA_data(lat, lon, date):
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
        "start": date,
        "end": date,
        "format": "JSON"
    }

    resp = requests.get(base, params=params)

    if resp.status_code != 200:
        raise Exception(f"Error en la solicitud: {resp.status_code}")

    return resp.json()

import math

def calc_et0_fao56_day(data, date):
    """
    Calcula la evapotranspiración de referencia ET0 (FAO 56)
    para un solo día, usando los datos del diccionario 'data' y la fecha 'date' en formato YYYYMMDD.
    
    data: diccionario con la estructura de NASA POWER (Feature -> properties -> parameter)
    date: string, por ejemplo '20251001'
    """
    params = data["properties"]["parameter"]

    # Extraer variables meteorológicas
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

    return round(et0, 3)


def fetch_NDVI_ee_image(lat1, lon1, lat2, lon2, target_date, cloud_thresh=50, days_window=3):
    """
    Devuelve un objeto ee.Image con el NDVI promedio de la imagen más cercana
    a la fecha objetivo dentro de un rango de ±days_window días, junto con la fecha de la imagen.
    """
    import ee
    from datetime import datetime, timedelta
    import os
    from dotenv import load_dotenv

    load_dotenv()
    try:
        ee.Initialize(project=os.getenv("PROJECT_ID"))
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=os.getenv("PROJECT_ID"))

    # Convertir la fecha a datetime
    date_dt = datetime.strptime(target_date, "%Y%m%d")
    start = (date_dt - timedelta(days=days_window)).strftime("%Y-%m-%d")
    end = (date_dt + timedelta(days=days_window)).strftime("%Y-%m-%d")

    region = ee.Geometry.Rectangle([lon1, lat1, lon2, lat2])

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR")
        .filterBounds(region)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_thresh))
    )

    if collection.size().getInfo() == 0:
        raise ValueError(f"No hay imágenes disponibles entre {start} y {end}.")

    def add_ndvi(img):
        ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
        return img.addBands(ndvi)

    ndvi_collection = collection.map(add_ndvi)

    # Seleccionar la imagen más cercana a la fecha objetivo
    def diff_from_target(img):
        return img.set('date_diff', ee.Number(img.date().difference(ee.Date(date_dt), 'day')).abs())

    closest_img = ndvi_collection.map(diff_from_target).sort('date_diff').first()
    ndvi_img = closest_img.select("NDVI")

    # Obtener la fecha de la imagen seleccionada en formato YYYY-MM-DD
    image_date = ee.Date(closest_img.get('system:time_start')).format('YYYY-MM-dd').getInfo()

    return ndvi_img, region, image_date




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



def fetch_forecast_data(lat1, lon1, lat2, lon2):
    """
    Obtiene pronósticos climáticos diarios (5 días) del servicio Open-Meteo
    para el punto central del rectángulo definido por (lat1, lon1) y (lat2, lon2).

    Retorna un diccionario con los datos de ET0, temperatura, humedad, etc.
    """
    # Calcular el punto central del área
    lat_c = (lat1 + lat2) / 2
    lon_c = (lon1 + lon2) / 2

    base_url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": lat_c,
        "longitude": lon_c,
        "daily": (
            "temperature_2m_max,temperature_2m_min,"
            "relative_humidity_2m_mean,et0_fao_evapotranspiration,"
            "precipitation_sum,wind_speed_10m_max,shortwave_radiation_sum"
        ),
        "timezone": "America/Argentina/Cordoba",
        "forecast_days": 5
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        # Validar formato esperado
        if "daily" not in data:
            raise ValueError("Respuesta inesperada del servicio meteorológico.")

        return data["daily"]

    except Exception as e:
        print(f"[ERROR] No se pudieron obtener los datos: {e}")
        return None


def generate_wsi_timeseries(ndvi_img, forecast_data, region):
    """Genera una serie temporal de imágenes WSI pronosticadas."""
    
    wsi_series = []

    et0_values = forecast_data["et0_fao_evapotranspiration"]
    dates = forecast_data["time"]

    for date, et0 in zip(dates, et0_values):
        # Calcular WSI para ese día
        wsi_img = calc_water_stress(ndvi_img, et0, region)
        wsi_url = image_to_url(wsi_img, region, dimensions=512)

        wsi_series.append({
            "date": date,
            "et0": et0,
            "wsi_url": wsi_url
        })

    return wsi_series


def estimate_etc_series(et0_series, kc=0.85):
    """Calcula la evapotranspiración del cultivo (ETc)."""
    return [round(et0 * kc, 2) for et0 in et0_series]


def estimate_eta_series(et0_series, ndvi_mean):
    """Modelo calibrable para ETa (ET real)."""
    return [round(et0 * (0.2 + 0.8 * ndvi_mean), 2) for et0 in et0_series]


def compute_wsi_series(eta_series, etc_series, time_series):
    """Calcula la serie temporal del Water Stress Index."""
    wsi = []
    for eta, etc in zip(eta_series, etc_series):
        if etc == 0:
            wsi.append(None)
        else:
            wsi.append(round(1 - (eta / etc), 3))

    return {
        "time": time_series,
        "values": wsi
    }