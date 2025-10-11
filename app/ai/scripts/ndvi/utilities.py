import requests
import pandas as pd

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


def parse_NASA_data(resp):
    """
    Convierte la respuesta JSON de la NASA en un DataFrame diario.
    """
    data = resp.json()
    params = data["properties"]["parameter"]
    
    # Convertir cada parámetro en un DataFrame temporal
    df_dict = {}
    for key, val in params.items():
        df_dict[key] = pd.Series(val, name=key)
    
    # Combinar en un solo DataFrame
    df = pd.concat(df_dict.values(), axis=1)
    df.index = pd.to_datetime(df_dict[list(df_dict.keys())[0]].index)  # index = fechas
    return df


