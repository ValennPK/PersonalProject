import numpy as np
import cv2
import rasterio
from rasterio.plot import reshape_as_raster, reshape_as_image
from pathlib import Path


def load_image(path: str):
    """
    Carga una imagen satelital desde un archivo.
    Soporta formatos TIFF (multibanda) y PNG/JPG (RGB).

    Parámetros:
        path (str): Ruta de la imagen.

    Retorna:
        img (np.ndarray): Imagen cargada (array NumPy).
        bands (int): Número de bandas detectadas.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"La imagen no existe: {path}")

    if path.suffix.lower() in [".tif", ".tiff"]:
        with rasterio.open(path) as src:
            img = src.read()  # (bandas, alto, ancho)
            img = reshape_as_image(img)  # (alto, ancho, bandas)
        print(f"[INFO] Imagen TIFF cargada con {img.shape[2]} bandas.")
        return img, img.shape[2]

    elif path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(f"[INFO] Imagen RGB cargada con forma {img.shape}.")
        return img, 3

    else:
        raise ValueError(f"Formato no soportado: {path.suffix}")


def normalize_image(img: np.ndarray):
    """
    Normaliza la imagen a valores entre 0 y 1.
    """
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0
    return img


def extract_bands(img: np.ndarray, red_band=0, nir_band=1):
    """
    Extrae las bandas necesarias para el cálculo del NDVI.

    Parámetros:
        img (np.ndarray): Imagen multibanda.
        red_band (int): Índice de la banda roja.
        nir_band (int): Índice de la banda NIR.

    Retorna:
        red (np.ndarray): Banda roja.
        nir (np.ndarray): Banda NIR.
    """
    if img.ndim < 3 or img.shape[2] < 2:
        raise ValueError("La imagen no tiene suficientes bandas para NDVI.")

    red = img[:, :, red_band]
    nir = img[:, :, nir_band]
    return red, nir


def preprocess_image(path: str, red_band=0, nir_band=1):
    """
    Carga, normaliza y extrae las bandas necesarias para NDVI.

    Parámetros:
        path (str): Ruta de la imagen.
        red_band (int): Índice de la banda roja.
        nir_band (int): Índice de la banda NIR.

    Retorna:
        red, nir (np.ndarray, np.ndarray): Bandas listas para NDVI.
    """
    img, bands = load_image(path)
    img = normalize_image(img)
    red, nir = extract_bands(img, red_band, nir_band)
    return red, nir
