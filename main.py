import tensorflow as tf
from datos import evaluate_model, get_random_val_image, predict_single_image
from entrenamiento import val_generator, class_names, VAL_DIR, IMG_SIZE

# Lista de modelos y sus nombres
model_files = [
    ("modelo_tomates_resnet50.h5", "ResNet50"),
    ("modelo_tomates_efficientnetb0.h5", "EfficientNetB0"),
    ("modelo_tomates_densenet121.h5", "DenseNet121")
]

for file_path, model_name in model_files:
    print(f"\n=== CARGANDO MODELO: {model_name} ===")
    model = tf.keras.models.load_model(file_path)
    
    # Evaluación
    evaluate_model(model, val_generator, class_names, name=model_name)
    
    # Predicción aleatoria
    img = get_random_val_image(VAL_DIR)
    predict_single_image(model, img, class_names, IMG_SIZE, title_model=model_name)
