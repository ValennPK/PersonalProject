# train_model.py
import joblib
from sklearn.linear_model import LogisticRegression
import numpy as np

# Datos ficticios: [edad, años_estudio]
X = np.array([
    [18, 10],
    [25, 12],
    [30, 14],
    [40, 16],
    [50, 18],
    [22, 8],
    [35, 10],
    [45, 12],
])

# Etiquetas: 0 = ingreso bajo, 1 = ingreso alto
y = np.array([0, 0, 1, 1, 1, 0, 0, 1])

# Entrenar modelo
model = LogisticRegression()
model.fit(X, y)

# Guardar el model entrenado
joblib.dump(model, "model.pkl")
print("model guardado en model.pkl")
