from ultralytics import YOLO
import os

# Rutas base del proyecto
# Este archivo está en: .../proyecto_tomate/training/usar_modelo_yolo11.py
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../proyecto_tomate
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
SEG_MODEL_PATH = os.path.join(MODELS_DIR, "modelos_entrenados", "SegmentacionYolo.pt")

# Cargar el modelo guardado
model = YOLO(SEG_MODEL_PATH)

if __name__ == "__main__":
    # Ejemplo de uso: cambia esta ruta por una imagen tuya
    image_path = r"C:\Users\ll529\Downloads\uu.jpg"
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"No se encontró la imagen de ejemplo: {image_path}")
    results = model(image_path)
    results[0].show()