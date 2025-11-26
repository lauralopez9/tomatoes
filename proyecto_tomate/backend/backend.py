"""
Backend Flask para procesar imágenes con modelos de segmentación y clasificación
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import io
import base64

# Modelos
from ultralytics import YOLO
import tensorflow as tf

# Rutas base del proyecto
# Este archivo está en: .../proyecto_tomate/backend/backend.py
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../proyecto_tomate
MODELS_DIR = os.path.join(BASE_DIR, "models")
UPLOAD_FOLDER = os.path.join(MODELS_DIR, "uploads")

app = Flask(__name__)
CORS(app)  # Permitir CORS para el frontend

# Configuración
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Cargar modelos al iniciar
print("Cargando modelos...")
modelo_segmentacion = None
modelo_clasificacion = None

try:
    # Permitir sobreescribir la ruta del modelo por variable de entorno
    seg_path = os.getenv(
        "YOLO_SEG_MODEL_PATH",
        os.path.join(MODELS_DIR, "modelos_entrenados", "SegmentacionYolo.pt"),
    )
    modelo_segmentacion = YOLO(seg_path)
    print(f"✓ Modelo de segmentación cargado desde: {seg_path}")
except Exception as e:
    print(f"⚠ Error cargando modelo de segmentación: {e}")

try:
    # Modelo principal de clasificación (DenseNet por defecto)
    tf_model_path = os.getenv(
        "TF_CLASS_MODEL_PATH",
        os.path.join(MODELS_DIR, "modelo_tomates_densenet121.h5"),
    )
    if os.path.exists(tf_model_path):
        modelo_clasificacion = tf.keras.models.load_model(tf_model_path)
        print(f"✓ Modelo de clasificación cargado desde: {tf_model_path}")
    else:
        # Intentar con otros modelos dentro de models/
        modelos_tf = ["modelo_tomates_densenet121.h5",
                      "modelo_tomates_efficientnetb0.h5",
                      "modelo_tomates_resnet50.h5"]
        for nombre in modelos_tf:
            modelo_path = os.path.join(MODELS_DIR, nombre)
            if os.path.exists(modelo_path):
                modelo_clasificacion = tf.keras.models.load_model(modelo_path)
                print(f"✓ Modelo de clasificación cargado desde: {modelo_path}")
                break
    if modelo_clasificacion is None:
        print("⚠ No se encontró ningún modelo de clasificación en la carpeta 'models'")
except Exception as e:
    print(f"⚠ Error cargando modelo de clasificación: {e}")

CLASES_CLASIFICACION = ["Damaged", "Old", "Ripe", "Unripe"]
IMG_SIZE_CLASIFICACION = (180, 180)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def procesar_imagen_base64(imagen_base64):
    """Convierte imagen base64 a PIL Image"""
    # Remover el prefijo data:image/...;base64,
    if ',' in imagen_base64:
        imagen_base64 = imagen_base64.split(',')[1]
    
    imagen_bytes = base64.b64decode(imagen_base64)
    imagen = Image.open(io.BytesIO(imagen_bytes))
    
    # Convertir a RGB si es necesario
    if imagen.mode != 'RGB':
        imagen = imagen.convert('RGB')
    
    return imagen


@app.route('/')
def index():
    return jsonify({"message": "Backend funcionando correctamente"})


@app.route('/api/segmentacion', methods=['POST'])
def segmentacion():
    """Endpoint para segmentación con YOLO11"""
    if modelo_segmentacion is None:
        return jsonify({"error": "Modelo de segmentación no disponible"}), 500
    
    try:
        data = request.get_json()
        
        if 'imagen' not in data:
            return jsonify({"error": "No se proporcionó imagen"}), 400
        
        # Procesar imagen base64
        imagen = procesar_imagen_base64(data['imagen'])
        
        # Guardar temporalmente
        temp_path = os.path.join(UPLOAD_FOLDER, 'temp_seg.jpg')
        imagen.save(temp_path)
        
        # Realizar predicción
        results = modelo_segmentacion(temp_path, conf=0.25)
        result = results[0]
        
        # Obtener información de las detecciones
        detecciones = []
        for i, box in enumerate(result.boxes):
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = modelo_segmentacion.names[cls]
            
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            
            detecciones.append({
                "numero": i + 1,
                "clase": class_name,
                "confianza": round(conf, 3),
                "bbox": {
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2)
                }
            })
        
        # Guardar imagen con anotaciones
        annotated_path = os.path.join(UPLOAD_FOLDER, 'resultado_seg.jpg')
        result.save(annotated_path)
        
        # Convertir a base64
        with open(annotated_path, 'rb') as f:
            imagen_resultado = base64.b64encode(f.read()).decode('utf-8')
        
        return jsonify({
            "success": True,
            "num_detecciones": len(detecciones),
            "detecciones": detecciones,
            "imagen_resultado": f"data:image/jpeg;base64,{imagen_resultado}"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/clasificacion', methods=['POST'])
def clasificacion():
    """Endpoint para clasificación con TensorFlow"""
    if modelo_clasificacion is None:
        return jsonify({"error": "Modelo de clasificación no disponible"}), 500
    
    try:
        data = request.get_json()
        
        if 'imagen' not in data:
            return jsonify({"error": "No se proporcionó imagen"}), 400
        
        # Procesar imagen base64
        imagen = procesar_imagen_base64(data['imagen'])
        
        # Preprocesar para TensorFlow
        imagen_resized = imagen.resize(IMG_SIZE_CLASIFICACION)
        imagen_array = np.array(imagen_resized) / 255.0
        imagen_array = np.expand_dims(imagen_array, axis=0)
        
        # Realizar predicción
        pred = modelo_clasificacion.predict(imagen_array, verbose=0)[0]
        pred_idx = np.argmax(pred)
        clase = CLASES_CLASIFICACION[pred_idx]
        confianza = float(pred[pred_idx])
        
        # Obtener todas las probabilidades
        probabilidades = {}
        for i, clase_nombre in enumerate(CLASES_CLASIFICACION):
            probabilidades[clase_nombre] = round(float(pred[i]), 3)
        
        return jsonify({
            "success": True,
            "clase_predicha": clase,
            "confianza": round(confianza, 3),
            "probabilidades": probabilidades
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Endpoint para verificar estado de los modelos"""
    return jsonify({
        "segmentacion": modelo_segmentacion is not None,
        "clasificacion": modelo_clasificacion is not None
    })


if __name__ == '__main__':
    print("\n" + "="*50)
    print("Backend iniciado")
    print("="*50)
    print("Modelos disponibles:")
    print(f"  - Segmentación: {'✓' if modelo_segmentacion else '✗'}")
    print(f"  - Clasificación: {'✓' if modelo_clasificacion else '✗'}")
    print("="*50)
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    print(f"\nServidor corriendo en http://0.0.0.0:{port} (debug={debug})")
    app.run(host='0.0.0.0', port=port, debug=debug)

