import os
import cv2
import numpy as np

IMG_SIZE = 128

BASE_DIR = 'app/ai/datasets/dogs-vs-cats'
OUTPUT_DIR = 'app/ai/datasets/dogs-vs-cats-resized'

splits = ['train', 'val', 'test']

for split in splits:
    for label in ['dog', 'cat']:
        os.makedirs(os.path.join(OUTPUT_DIR, split, label), exist_ok=True)

def resize_images(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        img_path = os.path.join(input_dir, filename)

    
        if filename.startswith("dog"):
            label = "dog"
        elif filename.startswith("cat"):
            label = "cat"
        else:
            print(f"Archivo desconocido: {filename}")
            continue

        output_folder = os.path.join(output_dir, label)

        img = cv2.imread(img_path)
        if img is None:
            print(f"Error leyendo {img_path}")
            continue
    
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_normalized = img_resized / 255.0
        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, (img_normalized * 255).astype(np.uint8))

    print(f"Procesadas imágenes de {input_dir}")

for split in splits:
    input_dir = os.path.join(BASE_DIR, split)
    output_dir = os.path.join(OUTPUT_DIR, split)
    resize_images(input_dir, output_dir)

print("Todas las imágenes han sido procesadas y guardadas en:", OUTPUT_DIR)
