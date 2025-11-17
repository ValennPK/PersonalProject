#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_ndvi_gru.py
Entrena un modelo GRU para predecir NDVI usando historial climático y NDVI previo.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# --- CONFIGURACIÓN ---
DATASET_PATH = "app/ai/datasets/water_stress/ndvi_dataset.csv"
MODEL_PATH = "app/ai/models/ndvi_predictor_gru.h5"
SEQ_LEN = 6  # t, t-1, ..., t-5

print("📂 Cargando dataset...")
df = pd.read_csv(DATASET_PATH)

# Eliminamos filas sin NDVI
df = df.dropna(subset=["ndvi"])

# --- FEATURES POR T ----
# Cada variable tiene 6 columnas: var_t, var_t_1 ... var_t_5
vars_prefix = {
    "precip": "precip_t",
    "temp": "temp_t",
    "hum": "hum_t",
    "et0": "et0_t",
    "rad": "rad_t",
    "viento": "viento_t",
}

feature_columns = []

for var_name, prefix in vars_prefix.items():
    for i in range(6):
        suffix = f"_{i}" if i > 0 else ""
        feature_columns.append(f"{prefix}{suffix}")

# NDVI previo (solo se usa en t-1)
feature_columns.append("ndvi_t_1")

# --- 1. NORMALIZACIÓN (solo de features numéricos) ---
scaler = StandardScaler()
df[feature_columns] = df[feature_columns].fillna(0)
scaled_values = scaler.fit_transform(df[feature_columns])
df_scaled = pd.DataFrame(scaled_values, columns=feature_columns)

# --- 2. ARMAR SECUENCIAS ---
def build_sequence(row):
    """
    Convierte una fila en una secuencia temporal 6xF para una GRU.
    Secuencia:
        paso 0 -> t-5
        paso 1 -> t-4
        ...
        paso 5 -> t
    """
    seq = []
    for i in reversed(range(6)):  # de t-5 a t
        step_features = []
        for var_name, prefix in vars_prefix.items():
            col = f"{prefix}_{i}" if i > 0 else prefix
            step_features.append(row[col])

        # NDVI previo solo se usa en t-1
        if i == 1:
            step_features.append(row["ndvi_t_1"])
        else:
            step_features.append(0.0)

        seq.append(step_features)

    return np.array(seq, dtype=np.float32)


print("📐 Construyendo tensores de secuencias...")

X_list = []
y_list = []

for idx, row in df_scaled.iterrows():
    X_list.append(build_sequence(row))
    y_list.append(df["ndvi"].iloc[idx])

X = np.stack(X_list)  # (N, 6, F)
y = np.array(y_list)

print(f"✔ X shape: {X.shape}   (N, 6, features)")
print(f"✔ y shape: {y.shape}")

# --- 3. DIVISIÓN USANDO split ---
train_mask = df["split"] == "train"
val_mask = df["split"] == "val"
test_mask = df["split"] == "test"

X_train = X[train_mask]
y_train = y[train_mask]

X_val = X[val_mask]
y_val = y[val_mask]

X_test = X[test_mask]
y_test = y[test_mask]

# --- 4. MODELO GRU ---
print("\n🧠 Construyendo modelo GRU...")

model = models.Sequential([
    layers.Input(shape=(SEQ_LEN, len(vars_prefix) + 1)),  # F features
    layers.GRU(128, return_sequences=False),
    layers.Dropout(0.2),
    layers.Dense(64, activation="relu"),
    layers.Dense(1)
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

early_stop = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

print("🚀 Entrenando...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)

# --- 5. EVALUACIÓN ---
print("\n📊 Evaluando...")
y_pred = model.predict(X_test).flatten()

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\n📊 Resultados en test:")
print(f" RMSE = {rmse:.4f}")
print(f" R²   = {r2:.4f}")

# --- 6. GUARDADO ---
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
model.save(MODEL_PATH)

print(f"\n💾 Modelo GRU guardado en: {MODEL_PATH}")
