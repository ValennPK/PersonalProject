import ee
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import requests

# --- Inicialización de Earth Engine ---
load_dotenv()
try:
    ee.Initialize(project=os.getenv("PROJECT_ID"))
except Exception:
    ee.Authenticate()
    ee.Initialize(project=os.getenv("PROJECT_ID"))

# --- FUNCIONES AUXILIARES ---
def mask_clouds_SCL(image):
    """Máscara de nubes más flexible usando la banda SCL."""
    scl = image.select('SCL')
    # Mantiene píxeles que no son agua (6), nubes densas (9,10) ni sombras (3)
    mask = scl.neq(3).And(scl.neq(9)).And(scl.neq(10))
    return image.updateMask(mask)

def add_ndvi(image):
    """Calcula NDVI y lo agrega como banda."""
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return image.addBands(ndvi)

def count_valid_pixels(image, region):
    """Cuenta cuántos píxeles válidos hay en la banda NDVI."""
    stats = image.select('NDVI').mask().reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=region,
        scale=10,
        maxPixels=1e9
    )
    return ee.Number(stats.get('NDVI'))

# --- PARÁMETROS ---
start_date = '2023-11-01'
end_date   = '2024-03-01'
window_days = 3  # ventana temporal de 3 días

lon1, lat1, lon2, lat2 = -63.20, -32.00, -63.10, -31.92
region = ee.Geometry.Rectangle([lon1, lat1, lon2, lat2])

output_dir = "app/ai/datasets/water_stress"
os.makedirs(output_dir, exist_ok=True)

# --- COLECCIÓN BASE ---
collection_raw = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(region)
    .filterDate(start_date, end_date)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 70))
    .map(mask_clouds_SCL)
    .map(add_ndvi)
)

print("Imágenes brutas después del filtro:", collection_raw.size().getInfo())

# --- GENERACIÓN DE COMPOSITES TEMPORALES ---
start = datetime.strptime(start_date, "%Y-%m-%d")
end   = datetime.strptime(end_date, "%Y-%m-%d")

current = start
composite_count = 0

while current < end:
    next_date = current + timedelta(days=window_days)
    
    # Filtrar por rango temporal
    subset = collection_raw.filterDate(current.strftime("%Y-%m-%d"), next_date.strftime("%Y-%m-%d"))
    
    if subset.size().getInfo() == 0:
        current = next_date
        continue
    
    # Crear composite NDVI (mediana)
    composite = subset.median().clip(region).select('NDVI')
    
    # Verificar píxeles válidos
    valid_pixels = count_valid_pixels(composite, region).getInfo()
    total_pixels = region.area().getInfo() / (10 * 10)  # píxeles aprox. (10 m resolución)
    valid_ratio = (valid_pixels / total_pixels) if valid_pixels else 0

    if valid_ratio < 0.1:  # menos de 10 % válidos
        print(f"⏭️  {current.strftime('%Y-%m-%d')} - Omitido (solo {valid_ratio*100:.1f}% válidos)")
    else:
        date_str = current.strftime("%Y%m%d")
        file_name = f"NDVI_{date_str}.tif"
        file_path = os.path.join(output_dir, file_name)

        print(f"⬇️  Exportando {file_name} ({valid_ratio*100:.1f}% válidos)...")

        try:
            # Generar URL y descargar
            url = composite.getDownloadURL({
                'scale': 10,
                'region': region,
                'crs': 'EPSG:4326',
                'format': 'GEO_TIFF'
            })
            response = requests.get(url)
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            print(f"✅ Guardado en {file_path}")
            composite_count += 1
        except Exception as e:
            print(f"❌ Error exportando {file_name}: {e}")

    current = next_date

print(f"\nProceso completado. {composite_count} composites NDVI exportados correctamente.")


# LOCALES = [
#     {
#         "id": "lote_1",
#         "lat1": -32.80,
#         "lon1": -62.30,
#         "lat2": -32.75,
#         "lon2": -62.25
#     },
#     {
#         "id": "lote_2",
#         "lat1": -31.50,
#         "lon1": -62.80,
#         "lat2": -31.45,
#         "lon2": -62.75
#     },
#     {
#         "id": "lote_3",
#         "lat1": -32.00,
#         "lon1": -63.20,
#         "lat2": -31.95,
#         "lon2": -63.15
#     },
#     # --- Nuevos lotes ---
#     {
#         "id": "lote_4",
#         "lat1": -33.00,
#         "lon1": -61.00,
#         "lat2": -32.95,
#         "lon2": -60.95
#     },  # Región sur de Santa Fe
#     {
#         "id": "lote_5",
#         "lat1": -30.80,
#         "lon1": -64.10,
#         "lat2": -30.75,
#         "lon2": -64.05
#     },  # Centro de Córdoba
#     {
#         "id": "lote_6",
#         "lat1": -35.00,
#         "lon1": -63.50,
#         "lat2": -34.95,
#         "lon2": -63.45
#     }  # Norte de La Pampa
# ]
