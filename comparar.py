# comparar_modelos.py

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import itertools

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# === CONFIGURACIÓN ===
VAL_DIR = r"C:/Users/ll529/OneDrive/Documentos/modelado_datos/dataset/dataset/val"
IMG_SIZE = (180, 180)
BATCH_SIZE = 32
CLASS_NAMES = ["Damaged", "Old", "Ripe", "Unripe"]

model_paths = {
    "ResNet50": "modelo_tomates_resnet50.h5",
    "EfficientNetB0": "modelo_tomates_efficientnetb0.h5",
    "DenseNet121": "modelo_tomates_densenet121.h5"
}

# === GENERADOR DE VALIDACIÓN ===
val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# === FUNCIÓN: MATRIZ DE CONFUSIÓN ===
def plot_confusion_matrix(cm, classes, title):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('Etiqueta verdadera')
    plt.xlabel('Etiqueta predicha')
    plt.tight_layout()
    plt.show()

# === ALMACENAMIENTO DE RESULTADOS ===
resultados = {
    "Modelo": [],
    "Accuracy": [],
    "Loss": []
}

# === EVALUACIÓN DE LOS 3 MODELOS ===
for nombre, ruta_modelo in model_paths.items():
    print(f"\n================== {nombre} ==================\n")

    # Cargar modelo
    model = tf.keras.models.load_model(ruta_modelo)

    # Evaluación
    loss, acc = model.evaluate(val_generator, verbose=0)
    print(f"Accuracy: {acc:.4f} | Loss: {loss:.4f}")

    resultados["Modelo"].append(nombre)
    resultados["Accuracy"].append(acc)
    resultados["Loss"].append(loss)

    # Predicciones completas
    val_generator.reset()
    preds = model.predict(val_generator, verbose=0)
    y_pred = np.argmax(preds, axis=1)
    y_true = val_generator.classes

    # Reporte de clasificación
    print("\nReporte de clasificación:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, CLASS_NAMES, f"Matriz de Confusión - {nombre}")


# ===============================================
# === GRÁFICO COMPARATIVO DE ACCURACY Y LOSS ====
# ===============================================

plt.figure(figsize=(10, 4))

# Accuracy
plt.subplot(1, 2, 1)
plt.bar(resultados["Modelo"], resultados["Accuracy"])
plt.title("Comparación de Accuracy")
plt.ylim(0, 1)
plt.ylabel("Accuracy")

# Loss
plt.subplot(1, 2, 2)
plt.bar(resultados["Modelo"], resultados["Loss"])
plt.title("Comparación de Loss")
plt.ylabel("Loss")

plt.tight_layout()
plt.show()

# === TABLA COMPARATIVA ===
print("\n========== TABLA COMPARATIVA ==========\n")
for i in range(len(resultados["Modelo"])):
    print(f"{resultados['Modelo'][i]} -> Accuracy: {resultados['Accuracy'][i]:.4f} | Loss: {resultados['Loss'][i]:.4f}")

# === MEJOR MODELO ===
best_idx = np.argmax(resultados["Accuracy"])
mejor_modelo = resultados["Modelo"][best_idx]

print("\n================ RESULTADO FINAL =================")
print(f"📌 El modelo más preciso es: **{mejor_modelo}**")
print(f"✔ Accuracy: {resultados['Accuracy'][best_idx]:.4f}")
print("=================================================")
