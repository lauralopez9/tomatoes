"""
Script para entrenar YOLO11 con segmentación de tomates usando dataset local
Dataset con 4 categorías: damaged, old, ripe, unripe
"""

# === PASO 1: INSTALAR DEPENDENCIAS ===
print("Verificando dependencias necesarias...")
import subprocess
import sys

def install_package(package):
    """Instala un paquete usando pip"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    from ultralytics import YOLO
    print("✓ ultralytics ya está instalado")
except ImportError:
    print("Instalando ultralytics (YOLO11)...")
    install_package("ultralytics")

try:
    import yaml
    print("✓ pyyaml ya está instalado")
except ImportError:
    print("Instalando pyyaml...")
    install_package("pyyaml")

# === PASO 2: CONFIGURAR DATASET LOCAL ===
print("\n" + "="*60)
print("CONFIGURANDO DATASET LOCAL")
print("="*60)

from ultralytics import YOLO
import os
import torch
import yaml
from datetime import datetime

# Ruta al dataset local (con 4 categorías: damaged, old, ripe, unripe)
DATASET_DIR = "Tomates.v2-tomates-v2.yolov11"
data_yaml_path = os.path.join(DATASET_DIR, "data.yaml")

# Verificar que existe el dataset
if not os.path.exists(data_yaml_path):
    raise FileNotFoundError(
        f"No se encontró el dataset en: {DATASET_DIR}\n"
        f"Asegúrate de que el directorio existe y contiene el archivo data.yaml"
    )

print(f"✓ Dataset encontrado en: {os.path.abspath(DATASET_DIR)}")
print(f"✓ Archivo de configuración: {data_yaml_path}")

# === PASO 3: CORREGIR RUTAS EN DATA.YAML ===
print("\nCorrigiendo rutas en data.yaml...")
with open(data_yaml_path, 'r', encoding='utf-8') as f:
    data_config = yaml.safe_load(f)

# Obtener el directorio base del dataset
dataset_dir = os.path.dirname(os.path.abspath(data_yaml_path))

# Verificar y corregir rutas (usar rutas absolutas)
train_path = os.path.abspath(os.path.join(dataset_dir, "train", "images"))
val_path = os.path.abspath(os.path.join(dataset_dir, "valid", "images"))
test_path = os.path.abspath(os.path.join(dataset_dir, "test", "images"))

# Verificar que existen los directorios
if not os.path.exists(train_path):
    raise FileNotFoundError(f"No se encontró el directorio de entrenamiento: {train_path}")

if not os.path.exists(val_path):
    print("⚠ Directorio 'valid' no encontrado. YOLO dividirá automáticamente el dataset.")
    val_path = train_path

# Corregir las rutas en el config (usar rutas absolutas)
data_config['train'] = train_path
data_config['val'] = val_path
if 'test' in data_config:
    data_config['test'] = test_path if os.path.exists(test_path) else train_path

# Guardar el data.yaml corregido
with open(data_yaml_path, 'w', encoding='utf-8') as f:
    yaml.dump(data_config, f, default_flow_style=False, allow_unicode=True)

print(f"✓ Rutas corregidas:")
print(f"  - Train: {data_config['train']}")
print(f"  - Val: {data_config['val']}")
if 'test' in data_config:
    print(f"  - Test: {data_config['test']}")

# Mostrar información del dataset
print(f"\n📊 Información del dataset:")
print(f"  - Número de clases: {data_config.get('nc', 'N/A')}")
print(f"  - Clases: {data_config.get('names', 'N/A')}")
if data_config.get('nc') == 4:
    print(f"  ✓ Dataset con 4 categorías: damaged, old, ripe, unripe")

# Contar imágenes
train_count = len([f for f in os.listdir(train_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
val_count = len([f for f in os.listdir(val_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists(val_path) else 0
print(f"  - Imágenes de entrenamiento: {train_count}")
if val_count > 0:
    print(f"  - Imágenes de validación: {val_count}")

# === PASO 4: CONFIGURAR ENTRENAMIENTO ===
print("\n" + "="*60)
print("CONFIGURANDO ENTRENAMIENTO YOLO11")
print("="*60)

# Cargar modelo YOLO11 para segmentación
# 'yolo11n-seg.pt' = nano (más rápido, menos preciso)
# 'yolo11s-seg.pt' = small (balanceado)
# 'yolo11m-seg.pt' = medium
# 'yolo11l-seg.pt' = large
# 'yolo11x-seg.pt' = extra large (más preciso, más lento)

print("\nCargando modelo YOLO11 para segmentación...")
model = YOLO('yolo11n-seg.pt')  # Puedes cambiar a 'yolo11s-seg.pt' o 'yolo11m-seg.pt' para mejor precisión
print("✓ Modelo cargado")

# Detectar dispositivo (GPU o CPU)
if torch.cuda.is_available():
    device = 0  # Usar GPU
    BATCH_SIZE = 16
    print("✓ GPU detectada - usando GPU para entrenamiento")
else:
    device = 'cpu'  # Usar CPU
    BATCH_SIZE = 8  # Batch más pequeño para CPU
    print("⚠ GPU no detectada - usando CPU (será más lento)")

# Parámetros de entrenamiento
EPOCHS = 60
IMG_SIZE = 640  # Tamaño de imagen para YOLO (640 es estándar)

# Generar nombre único para el experimento (con timestamp)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_name = f"SegmentacionYolo_{timestamp}"

print(f"\n✓ Nombre del experimento: {experiment_name}")

# === PASO 5: ENTRENAR EL MODELO ===
print("\n" + "="*60)
print("INICIANDO ENTRENAMIENTO")
print("="*60)
print(f"Épocas: {EPOCHS}")
print(f"Tamaño de imagen: {IMG_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Dataset: {data_yaml_path}")
print(f"Experimento: {experiment_name}")
print("="*60 + "\n")

# Entrenar el modelo
results = model.train(
    data=data_yaml_path,      # Ruta al archivo de configuración
    epochs=EPOCHS,            # Número de épocas
    imgsz=IMG_SIZE,           # Tamaño de imagen
    batch=BATCH_SIZE,         # Tamaño de batch
    name=experiment_name,     # Nombre único del experimento
    project='runs/segment',   # Directorio del proyecto
    device=device,            # GPU o CPU (detectado automáticamente)
    patience=50,              # Early stopping patience
    save=True,                # Guardar checkpoints
    save_period=5,            # Guardar cada N épocas
    val=True,                 # Validar durante entrenamiento
    plots=True,               # Generar gráficas
    verbose=True              # Mostrar información detallada
)

print("\n" + "="*60)
print("ENTRENAMIENTO COMPLETADO")
print("="*60)

# === PASO 6: GUARDAR MODELO CON NOMBRE ESPECÍFICO ===
print("\nGuardando modelo con nombre 'SegmentacionYolo'...")

# Rutas de los modelos entrenados
best_model_path = f"runs/segment/{experiment_name}/weights/best.pt"
last_model_path = f"runs/segment/{experiment_name}/weights/last.pt"

# Crear directorio para modelos guardados
saved_models_dir = "modelos_entrenados"
os.makedirs(saved_models_dir, exist_ok=True)

# Copiar el mejor modelo con el nombre solicitado
import shutil
if os.path.exists(best_model_path):
    saved_model_path = os.path.join(saved_models_dir, "SegmentacionYolo.pt")
    shutil.copy2(best_model_path, saved_model_path)
    print(f"✓ Modelo guardado como: {saved_model_path}")
else:
    print(f"⚠ No se encontró best.pt, usando last.pt")
    if os.path.exists(last_model_path):
        saved_model_path = os.path.join(saved_models_dir, "SegmentacionYolo.pt")
        shutil.copy2(last_model_path, saved_model_path)
        print(f"✓ Modelo guardado como: {saved_model_path}")

# === PASO 7: INFORMACIÓN FINAL ===
print(f"\n📁 Ubicaciones de los modelos:")
print(f"  - Mejor modelo: {best_model_path}")
print(f"  - Modelo final: {last_model_path}")
if os.path.exists(os.path.join(saved_models_dir, "SegmentacionYolo.pt")):
    print(f"  - Modelo guardado: {os.path.abspath(os.path.join(saved_models_dir, 'SegmentacionYolo.pt'))}")

# Mostrar métricas finales
print("\n" + "="*60)
print("MÉTRICAS FINALES")
print("="*60)
if hasattr(results, 'results_dict'):
    print(f"Mejor mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"Mejor mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")

print("\n✓ ¡Entrenamiento completado exitosamente!")
print("\nPara usar el modelo entrenado:")
print("  from ultralytics import YOLO")
print("  model = YOLO('modelos_entrenados/SegmentacionYolo.pt')")
print("  results = model('ruta/a/imagen.jpg')")
