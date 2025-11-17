import pandas as pd
import numpy as np
from datetime import datetime
from tensorflow.keras import models, layers, callbacks
import glob
import os

# Paths
NDVI_PATH = "app/ai/datasets/water_stress/ndvi_dataset.csv"
WEATHER_DIR = "app/ai/datasets/water_stress/weather"

print("[INFO] Loading NDVI dataset...")
# 1. Load NDVI dataset
ndvi_df = pd.read_csv(NDVI_PATH, parse_dates=["fecha"])

print("[INFO] Loading weather datasets...")
# 2. Load weather datasets indexed by lote
weather_dfs = {}
for path in glob.glob(os.path.join(WEATHER_DIR, "lote_*.csv")):
    # Extract 'lote_3' from filename
    basename = os.path.basename(path).split(".")[0]  # → 'lote_3'
    lote_number = basename  # keep exact key to match NDVI CSV

    df = pd.read_csv(path, parse_dates=["fecha"])
    df = df.sort_values("fecha")
    df.set_index("fecha", inplace=True)
    weather_dfs[lote_number] = df

# print("[INFO] Weather datasets loaded. Preparing helper functions...")[lote_number] = df

print("[INFO] Weather datasets loaded. Preparing helper functions...")
# 3. Function: extract climatological window of 30 days before NDVI date
def extract_weather_window(lote, fecha, window=30):
    df = weather_dfs[lote]
    start = fecha - pd.Timedelta(days=window)
    segment = df.loc[start:fecha]

    if len(segment) < window:
        return None

    return segment.tail(window)

print("[INFO] Computing delta days for NDVI…")
# 4. Add delta days since previous NDVI for each pixel
ndvi_df = ndvi_df.sort_values(["lote", "lat", "lon", "fecha"])
ndvi_df["fecha_prev"] = ndvi_df.groupby(["lote","lat","lon"])['fecha'].shift(1)
ndvi_df["delta_days"] = (ndvi_df["fecha"] - ndvi_df["fecha_prev"]).dt.days

print("[INFO] Removing first NDVI entries (cannot compute delta)…")
# Remove first NDVI of each pixel (cannot compute delta)
ndvi_df = ndvi_df.dropna(subset=["delta_days"])

print("[INFO] Building temporal sequences (30-day windows)…")
# 5. Build temporal sequences (weather 30 days window)
X_list = []
X_static = []
y_list = []

# Ensure NDVI lote column is string
dvi_lote = ndvi_df["lote"].astype(str)
# NDVI lote column keeps values like 'lote_3'
ndvi_df["lote"] = ndvi_df["lote"].astype(str)

for idx, row in ndvi_df.iterrows():
    w = extract_weather_window(row.lote, row.fecha)
    if w is None:
        continue

    # Sequence input (30, features)
    X_list.append(w.values)

    # Static features
    X_static.append([
        row.lat,
        row.lon,
        row.delta_days
    ])

    y_list.append(row.ndvi)

X_seq = np.array(X_list)
X_static = np.array(X_static)
y = np.array(y_list)

print("[INFO] Creating GRU dual-input model…")
# 6. Build GRU dual-input model
seq_input = layers.Input(shape=(30, X_seq.shape[2]))
x = layers.GRU(128, return_sequences=False)(seq_input)
x = layers.Dropout(0.2)(x)

static_input = layers.Input(shape=(X_static.shape[1],))
s = layers.Dense(32, activation="relu")(static_input)

combined = layers.concatenate([x, s])
z = layers.Dense(64, activation="relu")(combined)
z = layers.Dense(1)(z)

model = models.Model(inputs=[seq_input, static_input], outputs=z)
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

print("[INFO] Starting training…")
# 7. Train model
early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
hist = model.fit(
    [X_seq, X_static], y,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop]
)

# 8. Evaluation
print("[INFO] Evaluating model…")
eval_results = model.evaluate([X_seq, X_static], y)
print(f"Evaluation results: {eval_results}")


# Save model
model.save("app/ai/models/ndvi_predictor_gru2.h5")
