"""
Script combinado: YOLO11 para detectar/segmentar tomates + TensorFlow para clasificar el tipo
Combina detección con clasificación (Ripe, Unripe, Damaged, Old)
"""

from ultralytics import YOLO
import tensorflow as tf
import cv2
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import glob

# === CONFIGURACIÓN ===
# Rutas base del proyecto
# Este archivo está en: .../proyecto_tomate/training/detectar_y_clasificar_tomates.py
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../proyecto_tomate
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RUNS_DIR = os.path.join(MODELS_DIR, "runs", "segment")

# Modelo YOLO para detección y segmentación
YOLO_MODEL_PATH = None  # Se detectará automáticamente

# Modelo TensorFlow para clasificación (dentro de models/)
# Usamos DenseNet121 como modelo principal por defecto
TF_MODEL_PATH = os.path.join(MODELS_DIR, "modelo_tomates_densenet121.h5")
TF_IMG_SIZE = (180, 180)
TF_CLASS_NAMES = ["Damaged", "Old", "Ripe", "Unripe"]

# === CARGAR MODELOS ===
def cargar_modelo_yolo():
    """Carga el modelo YOLO11 entrenado"""
    import glob
    
    posibles_rutas = [
        os.path.join(RUNS_DIR, "yolo11_tomates_seg*", "weights", "best.pt"),
        os.path.join(RUNS_DIR, "yolo11_tomates_seg*", "weights", "last.pt"),
    ]
    
    modelos_encontrados = []
    for patron in posibles_rutas:
        modelos_encontrados.extend(glob.glob(patron))
    
    modelos_encontrados.sort(key=os.path.getmtime, reverse=True)
    best_models = [m for m in modelos_encontrados if m.endswith('best.pt')]
    
    if best_models:
        modelo_path = best_models[0]
        print(f"✓ YOLO11 cargado desde: {modelo_path}")
        return YOLO(modelo_path)
    elif modelos_encontrados:
        modelo_path = modelos_encontrados[0]
        print(f"✓ YOLO11 cargado desde: {modelo_path}")
        return YOLO(modelo_path)
    else:
        raise FileNotFoundError("No se encontró el modelo YOLO11 entrenado")

def cargar_modelo_tensorflow():
    """Carga el modelo de TensorFlow para clasificación"""
    if not os.path.exists(TF_MODEL_PATH):
        # Intentar con otros modelos dentro de models/
        modelos_alternativos = [
            "modelo_tomates_densenet121.h5",
            "modelo_tomates_efficientnetb0.h5",
            "modelo_tomates_resnet50.h5",
        ]
        for nombre in modelos_alternativos:
            modelo = os.path.join(MODELS_DIR, nombre)
            if os.path.exists(modelo):
                print(f"✓ TensorFlow cargado desde: {modelo}")
                return tf.keras.models.load_model(modelo), modelo
        raise FileNotFoundError(f"No se encontró ningún modelo de TensorFlow en 'models/'")
    
    print(f"✓ TensorFlow cargado desde: {TF_MODEL_PATH}")
    return tf.keras.models.load_model(TF_MODEL_PATH), TF_MODEL_PATH

print("="*70)
print("DETECCIÓN Y CLASIFICACIÓN DE TOMATES")
print("="*70)
print("Cargando modelos...\n")

modelo_yolo = cargar_modelo_yolo()
modelo_tf, tf_model_name = cargar_modelo_tensorflow()

print(f"\n✓ Modelos cargados exitosamente")
print(f"  - YOLO11: Detección y segmentación")
print(f"  - TensorFlow ({Path(tf_model_name).stem}): Clasificación")
print("="*70 + "\n")

# === FUNCIÓN PRINCIPAL ===
def detectar_y_clasificar(ruta_imagen, mostrar=True, guardar=True, confianza_min=0.25):
    """
    Detecta tomates con YOLO y clasifica cada uno con TensorFlow
    
    Args:
        ruta_imagen: Ruta a la imagen
        mostrar: Mostrar resultado
        guardar: Guardar imagen con resultados
        confianza_min: Confianza mínima para YOLO
    """
    if not os.path.exists(ruta_imagen):
        print(f"❌ Error: No se encontró la imagen en {ruta_imagen}")
        return None
    
    print(f"\n{'='*70}")
    print(f"Procesando: {os.path.basename(ruta_imagen)}")
    print(f"{'='*70}")
    
    # 1. Cargar imagen original
    img_original = cv2.imread(ruta_imagen)
    img_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    
    # 2. Detectar tomates con YOLO
    print("\n🔍 Detectando tomates con YOLO11...")
    resultados_yolo = modelo_yolo(ruta_imagen, conf=confianza_min)
    resultado = resultados_yolo[0]
    
    num_tomates = len(resultado.boxes)
    print(f"   ✓ {num_tomates} tomate(s) detectado(s)")
    
    if num_tomates == 0:
        print("   ⚠ No se detectaron tomates en esta imagen")
        return None
    
    # 3. Clasificar cada tomate detectado con TensorFlow
    print(f"\n📊 Clasificando cada tomate con TensorFlow...")
    
    clasificaciones = []
    img_resultado = img_original.copy()
    
    for i, box in enumerate(resultado.boxes, 1):
        # Obtener coordenadas del bounding box
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf_yolo = float(box.conf[0])
        
        # Extraer región del tomate (con margen)
        margin = 10
        x1_crop = max(0, x1 - margin)
        y1_crop = max(0, y1 - margin)
        x2_crop = min(img_original.shape[1], x2 + margin)
        y2_crop = min(img_original.shape[0], y2 + margin)
        
        tomate_crop = img_rgb[y1_crop:y2_crop, x1_crop:x2_crop]
        
        if tomate_crop.size == 0:
            continue
        
        # Preprocesar para TensorFlow
        tomate_pil = Image.fromarray(tomate_crop)
        tomate_resized = tomate_pil.resize(TF_IMG_SIZE)
        tomate_array = np.array(tomate_resized) / 255.0
        tomate_array = np.expand_dims(tomate_array, axis=0)
        
        # Clasificar con TensorFlow
        pred = modelo_tf.predict(tomate_array, verbose=0)[0]
        pred_idx = np.argmax(pred)
        clase = TF_CLASS_NAMES[pred_idx]
        confianza_tf = pred[pred_idx]
        
        clasificaciones.append({
            'numero': i,
            'clase': clase,
            'confianza_yolo': conf_yolo,
            'confianza_tf': confianza_tf,
            'bbox': (x1, y1, x2, y2)
        })
        
        # Dibujar en la imagen
        # Color según la clase
        colores = {
            'Ripe': (0, 255, 0),      # Verde
            'Unripe': (255, 255, 0),   # Amarillo
            'Damaged': (0, 0, 255),    # Rojo
            'Old': (255, 165, 0)       # Naranja
        }
        color = colores.get(clase, (255, 255, 255))
        
        # Dibujar bounding box
        cv2.rectangle(img_resultado, (x1, y1), (x2, y2), color, 2)
        
        # Texto con clasificación
        texto = f"{clase} ({confianza_tf:.1%})"
        texto_size = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(img_resultado, (x1, y1 - texto_size[1] - 10), 
                     (x1 + texto_size[0], y1), color, -1)
        cv2.putText(img_resultado, texto, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        print(f"   {i}. {clase}: {confianza_tf:.1%} confianza")
    
    # 4. Mostrar resultados
    print(f"\n📋 RESUMEN:")
    print(f"   Total de tomates: {num_tomates}")
    for clf in clasificaciones:
        print(f"   Tomate {clf['numero']}: {clf['clase']} "
              f"(YOLO: {clf['confianza_yolo']:.1%}, TF: {clf['confianza_tf']:.1%})")
    
    # 5. Visualizar
    if mostrar:
        img_mostrar = cv2.cvtColor(img_resultado, cv2.COLOR_BGR2RGB)
        h, w = img_mostrar.shape[:2]
        if h > 800 or w > 1200:
            scale = min(800/h, 1200/w)
            new_h, new_w = int(h*scale), int(w*scale)
            img_mostrar = cv2.resize(img_mostrar, (new_w, new_h))
        
        plt.figure(figsize=(12, 8))
        plt.imshow(img_mostrar)
        plt.axis('off')
        plt.title(f"Detectados: {num_tomates} tomate(s)\n"
                 f"Clases: {', '.join(set([c['clase'] for c in clasificaciones]))}")
        plt.tight_layout()
        plt.show()
    
    # 6. Guardar
    if guardar:
        output_dir = "predicciones_completas"
        os.makedirs(output_dir, exist_ok=True)
        
        nombre_base = Path(ruta_imagen).stem
        extension = Path(ruta_imagen).suffix
        output_path = os.path.join(output_dir, f"{nombre_base}_detectado_clasificado{extension}")
        
        cv2.imwrite(output_path, img_resultado)
        print(f"\n✓ Imagen guardada en: {output_path}")
    
    return clasificaciones


def procesar_directorio(ruta_directorio, guardar=True, confianza_min=0.25):
    """Procesa todas las imágenes de un directorio"""
    if not os.path.exists(ruta_directorio):
        print(f"❌ Error: No se encontró el directorio {ruta_directorio}")
        return
    
    extensiones = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG']
    imagenes = []
    for root, dirs, files in os.walk(ruta_directorio):
        for archivo in files:
            if any(archivo.lower().endswith(ext.lower()) for ext in extensiones):
                imagenes.append(os.path.join(root, archivo))
    
    if len(imagenes) == 0:
        print(f"❌ No se encontraron imágenes en {ruta_directorio}")
        return
    
    print(f"\n{'='*70}")
    print(f"Procesando directorio: {ruta_directorio}")
    print(f"Imágenes encontradas: {len(imagenes)}")
    print(f"{'='*70}")
    
    estadisticas = {
        'total_imagenes': len(imagenes),
        'total_tomates': 0,
        'por_clase': {'Ripe': 0, 'Unripe': 0, 'Damaged': 0, 'Old': 0}
    }
    
    for i, imagen in enumerate(imagenes, 1):
        print(f"\n[{i}/{len(imagenes)}] {os.path.basename(imagen)}")
        try:
            clasificaciones = detectar_y_clasificar(imagen, mostrar=False, guardar=guardar, confianza_min=confianza_min)
            if clasificaciones:
                estadisticas['total_tomates'] += len(clasificaciones)
                for clf in clasificaciones:
                    estadisticas['por_clase'][clf['clase']] += 1
        except Exception as e:
            print(f"   ❌ Error procesando: {e}")
    
    # Resumen final
    print(f"\n{'='*70}")
    print(f"RESUMEN FINAL")
    print(f"{'='*70}")
    print(f"Imágenes procesadas: {estadisticas['total_imagenes']}")
    print(f"Total de tomates detectados: {estadisticas['total_tomates']}")
    print(f"\nDistribución por clase:")
    for clase, cantidad in estadisticas['por_clase'].items():
        porcentaje = (cantidad / estadisticas['total_tomates'] * 100) if estadisticas['total_tomates'] > 0 else 0
        print(f"  - {clase}: {cantidad} ({porcentaje:.1f}%)")
    print(f"{'='*70}")


# === MENÚ INTERACTIVO ===
def menu():
    while True:
        print("\n" + "="*70)
        print("MENÚ - DETECCIÓN Y CLASIFICACIÓN DE TOMATES")
        print("="*70)
        print("1. Procesar una imagen")
        print("2. Procesar un directorio")
        print("3. Salir")
        print("="*70)
        
        opcion = input("\nSelecciona una opción (1-3): ").strip()
        
        if opcion == "1":
            ruta = input("\nIngresa la ruta de la imagen: ").strip().strip('"')
            if ruta:
                try:
                    detectar_y_clasificar(ruta)
                except Exception as e:
                    print(f"❌ Error: {e}")
        
        elif opcion == "2":
            ruta = input("\nIngresa la ruta del directorio: ").strip().strip('"')
            if ruta:
                try:
                    guardar = input("¿Guardar imágenes? (s/n): ").strip().lower() == 's'
                    procesar_directorio(ruta, guardar=guardar)
                except Exception as e:
                    print(f"❌ Error: {e}")
        
        elif opcion == "3":
            print("\n¡Hasta luego!")
            break
        
        else:
            print("❌ Opción no válida")


# === EJECUCIÓN ===
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        ruta = sys.argv[1]
        if os.path.isfile(ruta):
            detectar_y_clasificar(ruta)
        elif os.path.isdir(ruta):
            procesar_directorio(ruta)
        else:
            print(f"❌ Error: {ruta} no es válido")
    else:
        menu()

