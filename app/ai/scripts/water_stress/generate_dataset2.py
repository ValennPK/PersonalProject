import ee
import os
from dotenv import load_dotenv
import requests
import geemap

# --- Inicialización de Earth Engine ---
load_dotenv()
try:
    ee.Initialize(project=os.getenv("PROJECT_ID"))
except Exception:
    ee.Authenticate()
    ee.Initialize(project=os.getenv("PROJECT_ID"))

# --- FUNCIONES AUXILIARES ---
def mask_clouds_SCL(image):
    scl = image.select('SCL')
    mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
    return image.updateMask(mask)

def add_ndvi(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return image.addBands(ndvi)

# --- PARÁMETROS ---
start_date = '2023-11-01'
end_date   = '2024-03-01'

lon1, lat1, lon2, lat2 = -63.20, -32.00, -63.10, -31.92
region = ee.Geometry.Rectangle([lon1, lat1, lon2, lat2])

output_dir = "app/ai/datasets/water_stress"
os.makedirs(output_dir, exist_ok=True)

# --- CARGA DE COLECCIÓN ---
collection_raw = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
    .filterBounds(region) \
    .filterDate(start_date, end_date)

print("Total imágenes brutas:", collection_raw.size().getInfo())

# Filtro de nubes
collection_filtered = collection_raw.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 70))
print("Después del filtro de nubes:", collection_filtered.size().getInfo())

# NDVI + máscara SCL
collection_ndvi = collection_filtered.map(mask_clouds_SCL).map(add_ndvi)

# --- VALIDACIÓN ---
count = collection_ndvi.size().getInfo()
print("Imágenes finales disponibles:", count)

if count == 0:
    raise Exception("No se encontraron imágenes NDVI válidas en el rango y zona especificados.")

# --- EXPORTACIÓN ---
image_list = collection_ndvi.toList(count)

for i in range(count):
    image = ee.Image(image_list.get(i))
    date = image.date().format('YYYYMMdd').getInfo()
    file_name = f"NDVI_{date}.tif"
    file_path = os.path.join(output_dir, file_name)
    
    print(f"Descargando {file_name} ...")
    
    # Generar URL de descarga
    url = image.select('NDVI').getDownloadURL({
        'scale': 10,
        'region': region,
        'crs': 'EPSG:4326',
        'format': 'GEO_TIFF'
    })
    
    # Descargar y guardar localmente
    response = requests.get(url)
    with open(file_path, 'wb') as f:
        f.write(response.content)

    print(f"✅ Guardado en {file_path}")

print("Todas las imágenes NDVI fueron descargadas correctamente.")