# === usar_modelos.py ===
import tensorflow as tf
import numpy as np
import random
import os
from datos import plot_training, evaluate_model, get_random_val_image, predict_single_image

# --- Definir rutas y parámetros ---
VAL_DIR = r"C:\Users\ll529\OneDrive\Documentos\modelado_datos\dataset\dataset\val"
IMG_SIZE = (180, 180)
class_names = ["Damaged", "Old", "Ripe", "Unripe"]

# --- Generador de validación ---
from tensorflow.keras.preprocessing.image import ImageDataGenerator
val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# --- Modelos ya entrenados ---
model_files = {
    "ResNet50": "modelo_tomates_resnet50.h5",
    "EfficientNetB0": "modelo_tomates_efficientnetb0.h5",
    "DenseNet121": "modelo_tomates_densenet121.h5"
}

# --- Evaluación y predicción para cada modelo ---
for name, file in model_files.items():
    print(f"\n================ {name} ================\n")
    
    # Cargar modelo
    model = tf.keras.models.load_model(file)
    model.summary()
    
    # Evaluar en conjunto de validación
    evaluate_model(model, name=name)
    
    # Predicción de imagen aleatoria
    random_img = get_random_val_image()
    predict_single_image(model, random_img, title_model=name)
