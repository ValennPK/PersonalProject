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

def ndvi_image_to_pixels(image, region, scale=10):
    import ee
    
    # Reducir la imagen a una lista de píxeles
    sampled = image.sample(
        region=region,
        scale=scale,
        geometries=True  # necesario para obtener lat/lon
    ).getInfo()

    pixels = []
    for f in sampled['features']:
        ndvi = f['properties']['NDVI']
        coords = f['geometry']['coordinates']  # lon, lat
        pixels.append({
            "lat": coords[1],
            "lon": coords[0],
            "ndvi": ndvi
        })

    return pixels

def build_features_for_pixel(ndvi_prev, clima_df):
    features = {}

    # NDVI del día anterior
    features["ndvi_t_1"] = ndvi_prev

    # Variables climáticas
    for var_base, col in [
        ("precip", "precipitacion_mm"),
        ("temp",   "temp_mean_c"),
        ("hum",    "humedad_relativa_pct"),
        ("et0",    "et0_mm"),
        ("rad",    "radiacion_sw_mj_m2"),
        ("viento", "vel_viento_m_s")
    ]:
        last_6_days = clima_df[col].tail(6).values[::-1]  # t0..t5

        for i in range(6):
            key = f"{var_base}_t_{i if i>0 else ''}".rstrip("_")
            features[key] = last_6_days[i]

    return features

def fetch_data(lat, lon, start_date, end_date, daily_vars=None, timezone=None):
    """Descarga datos de precipitación diaria desde Open-Meteo"""
    import requests
    import pandas as pd

    if daily_vars is None:
        daily_vars = [
            "precipitation_sum",
            "temperature_2m_mean",
            "temperature_2m_max",
            "temperature_2m_min",
            "shortwave_radiation_sum",
            "windspeed_10m_mean",
            "relative_humidity_2m_mean",
            "et0_fao_evapotranspiration"
        ]   

    if timezone is None:
        timezone = "America/Argentina/Buenos_Aires"
    
    daily_params = ",".join(daily_vars)

    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&daily={daily_params}"
        f"&timezone={timezone}"
    )

    r = requests.get(url)
    r.raise_for_status()
    data = r.json()

    if "daily" not in data or "precipitation_sum" not in data["daily"]:
        print("⚠️ No se encontraron datos meteorológicos.")
        return None

    df = pd.DataFrame({
        "fecha": pd.to_datetime(data["daily"]["time"]),
        "precipitacion_mm": data["daily"]["precipitation_sum"],
        "temp_mean_c": data["daily"]["temperature_2m_mean"],
        "temp_max_c": data["daily"]["temperature_2m_max"],
        "temp_min_c": data["daily"]["temperature_2m_min"],
        "radiacion_sw_mj_m2": data["daily"]["shortwave_radiation_sum"],
        "vel_viento_m_s": data["daily"]["windspeed_10m_mean"],
        "humedad_relativa_pct": data["daily"]["relative_humidity_2m_mean"],
        "et0_mm": data["daily"]["et0_fao_evapotranspiration"],
    })
    return df