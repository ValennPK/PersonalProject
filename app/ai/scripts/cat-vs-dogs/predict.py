import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# --- Cargar el modelo ---
model = tf.keras.models.load_model("app/ai/models/cat_vs_dog_model.h5")

# --- Configuración ---
IMG_SIZE = 128  # Debe coincidir con el tamaño de entrada del modelo

def predict_image(img_path):
    # Cargar la imagen desde el path
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    
    # Convertir a array
    img_array = image.img_to_array(img)
    
    # Normalizar al rango [0,1]
    img_array = img_array / 255.0
    
    # Expandir dimensiones -> (1, IMG_SIZE, IMG_SIZE, 3)
    img_array = np.expand_dims(img_array, axis=0)

    # Hacer predicción
    prediction = model.predict(img_array)[0][0]

    # Interpretar resultado
    if prediction > 0.5:
        print(f"{img_path} → Es un **Perro** 🐶 ({prediction:.4f})")
    else:
        print(f"{img_path} → Es un **Gato** 🐱 ({1 - prediction:.4f})")

# --- Ejemplo de uso ---
predict_image("app/ai/datasets/dogs-vs-cats-resized/test/cat/cat.110.jpg")
predict_image("app/ai/datasets/dogs-vs-cats-resized/test/dog/dog.44.jpg")
