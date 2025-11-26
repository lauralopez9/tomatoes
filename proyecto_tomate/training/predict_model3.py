import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
import numpy as np

# --- Configuración del modelo y datos ---
MODEL_PATH = "modelo_tomates_densenet121.h5"
IMG_SIZE = (180, 180)
class_names = ["Damaged", "Old", "Ripe", "Unripe"]  # mismo orden que en entrenamiento

# --- Cargar modelo ---
model = tf.keras.models.load_model(MODEL_PATH)
print("Modelo DenseNet121 cargado correctamente.")

# --- Seleccionar imagen ---
Tk().withdraw()  # ocultar ventana principal de Tkinter
img_path = askopenfilename(title="Selecciona una imagen de tomate",
                           filetypes=[("Imagenes", "*.jpg *.jpeg *.png")])

if not img_path:
    print("No se seleccionó ninguna imagen.")
    exit()

# --- Preprocesar imagen ---
img = load_img(img_path, target_size=IMG_SIZE)
x = img_to_array(img) / 255.0
x = np.expand_dims(x, axis=0)

# --- Predicción ---
pred = model.predict(x, verbose=0)[0]
pred_idx = np.argmax(pred)
class_name = class_names[pred_idx]
confidence = pred[pred_idx]

# --- Mostrar resultado ---
plt.imshow(img)
plt.axis('off')
plt.title(f"DenseNet121 -> {class_name} ({confidence:.2f})")
plt.show()

print(f"Imagen: {img_path}")
print(f"Predicción: {class_name} (confianza: {confidence:.2f})")
