import os
import random
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# NOTA: VAL_DIR, IMG_SIZE y class_names se importan desde entrenamiento.py cuando se necesiten
# from entrenamiento import VAL_DIR, IMG_SIZE, class_names, val_generator


def plot_training(history, title="Modelo"):
    """Grafica accuracy y loss de entrenamiento/validación."""
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12,4))

    # Accuracy
    plt.subplot(1,2,1)
    plt.plot(epochs_range, acc, label='Entrenamiento')
    plt.plot(epochs_range, val_acc, label='Validación')
    plt.title(f'Accuracy - {title}')
    plt.xlabel('Épocas')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1,2,2)
    plt.plot(epochs_range, loss, label='Entrenamiento')
    plt.plot(epochs_range, val_loss, label='Validación')
    plt.title(f'Loss - {title}')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


def evaluate_model(model, val_generator, class_names, name="Modelo"):
    """Evalúa el modelo en el conjunto de validación y muestra métricas."""
    print(f"\n========== Evaluación de {name} ==========")
    val_loss, val_acc = model.evaluate(val_generator, verbose=0)
    print(f"Loss de validación: {val_loss:.4f}")
    print(f"Accuracy de validación: {val_acc:.4f}")

    # Predicciones sobre todo el conjunto de validación
    val_generator.reset()
    preds = model.predict(val_generator, verbose=0)
    y_pred = np.argmax(preds, axis=1)
    y_true = val_generator.classes

    print("\nReporte de clasificación:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Matriz de confusión - {name}")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Verdadera')
    plt.xlabel('Predicha')
    plt.tight_layout()
    plt.show()


def get_random_val_image(VAL_DIR):
    """Devuelve la ruta de una imagen aleatoria del conjunto de validación."""
    all_files = []
    for root, dirs, files in os.walk(VAL_DIR):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_files.append(os.path.join(root, f))
    if not all_files:
        raise ValueError("No se encontraron imágenes en VAL_DIR.")
    return random.choice(all_files)


def predict_single_image(model, img_path, class_names, IMG_SIZE, title_model="Modelo"):
    """Cargar imagen, preprocesar y predecir con el modelo."""
    img = load_img(img_path, target_size=IMG_SIZE)
    x = img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    pred = model.predict(x, verbose=0)[0]
    pred_idx = np.argmax(pred)
    class_name = class_names[pred_idx]
    confidence = pred[pred_idx]

    plt.imshow(img)
    plt.axis('off')
    plt.title(f"{title_model} -> {class_name} ({confidence:.2f})")
    plt.show()

    print(f"Imagen: {img_path}")
    print(f"Predicción {title_model}: {class_name} (confianza: {confidence:.2f})")
