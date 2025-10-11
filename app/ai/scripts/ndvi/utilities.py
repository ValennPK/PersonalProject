import requests
import pandas as pd

def fetch_power_solar(lat, lon, start, end):
    """
    lat, lon en grados decimales
    start, end como cadenas 'YYYYMMDD'
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
        "parameters": "ALLSKY_SFC_SW_DWN",  # radiación solar incidente (onda corta)
        "community": "AG",  # comunidad agrícola, por ejemplo
        "latitude": lat,
        "longitude": lon,
        "start": start,
        "end": end,
        "format": "JSON"
    }
    resp = requests.get(base, params=params)
    
    if resp.status_code != 200:
        raise Exception(f"Error en la solicitud: {resp.status_code}")

    return resp

