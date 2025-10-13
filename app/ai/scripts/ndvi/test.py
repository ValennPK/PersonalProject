import numpy as np
import matplotlib.pyplot as plt
import rasterio
from utilities import fetch_NASA_data, calc_et0_fao56


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


latitude = 19.4326
longitude = -99.1332
start = "20251001"
end = "20251002"


data = fetch_NASA_data(latitude, longitude, start, end)
print("Datos obtenidos de la NASA POWER API:")
print (data)

# Calcular ET0 usando la función
data_et0 = calc_et0_fao56(data)
print("Datos con ET0 calculado:")
print(data_et0)
