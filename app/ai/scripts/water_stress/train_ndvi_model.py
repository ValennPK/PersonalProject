#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_ndvi_model.py
Entrena una red neuronal para predecir NDVI a partir de variables climáticas y NDVI previo.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# --- CONFIGURACIÓN ---
DATASET_PATH = "app/ai/datasets/water_stress/ndvi_dataset.csv"
MODEL_PATH = "app/ai/models/ndvi_predictor.h5"

# --- 1. CARGA DE DATOS ---
print("📂 Cargando dataset...")
df = pd.read_csv(DATASET_PATH)

# Eliminamos filas con NDVI faltante
df = df.dropna(subset=["ndvi"])

# --- 2. SELECCIÓN DE FEATURES ---
feature_cols = [
    "ndvi_t_1",
    *[f"precip_t_{i if i>0 else ''}".rstrip('_') for i in range(6)],
    *[f"temp_t_{i if i>0 else ''}".rstrip('_') for i in range(6)],
    *[f"hum_t_{i if i>0 else ''}".rstrip('_') for i in range(6)],
    *[f"et0_t_{i if i>0 else ''}".rstrip('_') for i in range(6)],
    *[f"rad_t_{i if i>0 else ''}".rstrip('_') for i in range(6)],
    *[f"viento_t_{i if i>0 else ''}".rstrip('_') for i in range(6)],
]

X = df[feature_cols]
y = df["ndvi"]

# --- 3. LIMPIEZA DE NaN ---
X = X.fillna(X.mean())

# --- 4. NORMALIZACIÓN ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 5. DIVISIÓN DE DATOS ---
train_df = df[df["split"] == "train"]
val_df   = df[df["split"] == "val"]
test_df  = df[df["split"] == "test"]

# Usamos las mismas columnas ya escaladas
X_train = scaler.fit_transform(train_df[feature_cols].fillna(train_df[feature_cols].mean()))
y_train = train_df["ndvi"].values

X_val = scaler.transform(val_df[feature_cols].fillna(val_df[feature_cols].mean()))
y_val = val_df["ndvi"].values

X_test = scaler.transform(test_df[feature_cols].fillna(test_df[feature_cols].mean()))
y_test = test_df["ndvi"].values

# --- 6. DEFINICIÓN DEL MODELO ---
model = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)  # salida NDVI (regresión)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# --- 7. ENTRENAMIENTO ---
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print("🚀 Entrenando modelo...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)

# --- 8. EVALUACIÓN ---
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\n📊 Evaluación en test:")
print(f"   RMSE: {rmse:.4f}")
print(f"   R²:   {r2:.4f}")

# --- 9. GUARDAR MODELO ---
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
model.save(MODEL_PATH)
print(f"✅ Modelo guardado en: {MODEL_PATH}")
