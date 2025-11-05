import ee
import os
from dotenv import load_dotenv
import geemap

# --- Inicialización de Earth Engine ---
load_dotenv()
try:
    ee.Initialize(project=os.getenv("PROJECT_ID"))
except Exception:
    ee.Authenticate()
    ee.Initialize(project=os.getenv("PROJECT_ID"))

# --- CONFIGURACIÓN ---
aoi = ee.Geometry.Point([-63.0, -32.0]).buffer(1000)  # región circular de 1 km
start_date = '2024-11-01'
end_date = '2024-12-31'
output_dir = "ndvi_series"

# Crear carpeta de salida si no existe
os.makedirs(output_dir, exist_ok=True)

# --- FUNCIONES AUXILIARES ---

def mask_clouds_SCL(image):
    """Aplica una máscara usando la banda SCL (Scene Classification Layer)."""
    scl = image.select('SCL')
    mask = scl.remap(
        [3, 8, 9, 10, 11],  # sombras, nubes medias/altas, cirros, nieve
        [0, 0, 0, 0, 0],
        1
    )
    return image.updateMask(mask)

def add_ndvi(image):
    """Calcula y agrega la banda NDVI."""
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return image.addBands(ndvi)

# --- CARGA DE COLECCIÓN ---
collection = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(aoi)
    .filterDate(start_date, end_date)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 70))
    .map(mask_clouds_SCL)
    .map(add_ndvi)
)

# --- EXPORTACIÓN LOCAL DE TODAS LAS IMÁGENES NDVI ---
image_list = collection.toList(collection.size())
n_images = image_list.size().getInfo()

print(f"Se encontraron {n_images} imágenes en el rango de fechas.")

for i in range(n_images):
    img = ee.Image(image_list.get(i))
    date_str = ee.Date(img.get('system:time_start')).format('YYYY-MM-dd').getInfo()
    filename = f"NDVI_{date_str}.tif"
    filepath = os.path.join(output_dir, filename)

    print(f"Descargando {filename}...")

    try:
        geemap.ee_export_image(
            img.select('NDVI'),
            filename=filepath,
            scale=10,
            region=aoi,
            file_per_band=False,
        )
        print(f"✅ {filename} exportado correctamente.")
    except Exception as e:
        print(f"⚠️ Error al exportar {filename}: {e}")

print("🚀 Exportación de serie NDVI finalizada.")
