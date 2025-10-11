import numpy as np
import matplotlib.pyplot as plt
import rasterio
from utilities import fetch_power_solar


# # rutas a tus bandas
# b4_path = "T20HQJ_20251007T135659_B04_10m.jp2"  # banda roja
# b8_path = "T20HQJ_20251007T135659_B08_10m.jp2"  # banda NIR

# # abrir ambas
# with rasterio.open(b4_path) as red_src, rasterio.open(b8_path) as nir_src:
#     red = red_src.read(1).astype('float32')
#     nir = nir_src.read(1).astype('float32')

# # NDVI = (NIR - RED) / (NIR + RED)
# ndvi = (nir - red) / (nir + red + 1e-6)

# # recortar valores fuera de rango (por seguridad)
# ndvi = np.clip(ndvi, -1, 1)

# plt.imshow(ndvi, cmap='RdYlGn')
# plt.colorbar(label='NDVI')
# plt.title("NDVI Sentinel-2")
# plt.show()

# Ejemplo de fetch_power_solar
# Ejemplo de uso:
lat = -33.0
lon = -60.0
start = "20250101"
end = "20250131"
resp = fetch_power_solar(lat, lon, start, end)
print(resp.json())  # o resp.text para ver la respuesta cruda

