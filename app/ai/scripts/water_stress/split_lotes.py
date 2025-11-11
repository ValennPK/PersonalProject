import os
import shutil

# --- CONFIGURACIÓN ---
BASE_DIR = "app/ai/datasets/water_stress"
TRAIN_LOTES = ["lote_4", "lote_3", "lote_6"]
VAL_LOTES   = ["lote_2"]
TEST_LOTES  = ["lote_1", "lote_5"]

SPLITS = {
    "train": TRAIN_LOTES,
    "val": VAL_LOTES,
    "test": TEST_LOTES
}

# --- CREAR CARPETAS DE SALIDA ---
for split in SPLITS:
    split_dir = os.path.join(BASE_DIR, split)
    os.makedirs(split_dir, exist_ok=True)

# --- MOVER LOTES A SUS CARPETAS ---
for split, lotes in SPLITS.items():
    for lote in lotes:
        src = os.path.join(BASE_DIR, lote)
        dst = os.path.join(BASE_DIR, split, lote)
        if os.path.exists(src):
            print(f"📦 Moviendo {lote} → {split}/")
            shutil.move(src, dst)
        else:
            print(f"⚠️ Lote {lote} no encontrado en {BASE_DIR}")

print("✅ División de lotes completada correctamente.")
