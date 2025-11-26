# 🍅 Clasificador de Tomates con IA

Sistema completo de detección y clasificación de tomates usando modelos de Deep Learning.

## 📋 Características

- **Segmentación con YOLO11**: Detecta y segmenta múltiples tomates en imágenes con 4 categorías (damaged, old, ripe, unripe)
- **Clasificación con TensorFlow**: Clasifica el tipo de tomate usando modelos pre-entrenados (ResNet50, EfficientNet, DenseNet)
- **Aplicación Web**: Frontend moderno con diseño en morado pastel y backend Flask
- **Cámara móvil**: Toma fotos directamente desde tu celular
- **Múltiples modelos**: Comparación entre diferentes arquitecturas de clasificación

## 🚀 Inicio Rápido

### Requisitos

- Python 3.8+
- Modelos entrenados:
  - `modelos_entrenados/SegmentacionYolo.pt` (YOLO11)
  - `modelo_tomates_efficientnetb0.h5` (o ResNet50/DenseNet121)

### Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/lauralopez9/tomatoes.git
cd tomatoes
```

2. Instalar dependencias:
```bash
pip install -r requirements_backend.txt
```

3. Iniciar el backend:
```bash
python backend.py
```

4. Abrir el frontend:
   - Abre `frontend/index.html` en tu navegador
   - O usa un servidor local: `python -m http.server 8000` en la carpeta frontend

## 📁 Estructura del Proyecto

```
tomatoes/
├── backend.py                          # Servidor Flask
├── frontend/                           # Interfaz web
│   ├── index.html
│   ├── styles.css
│   └── script.js
├── entrenar_yolo11_segmentacion.py     # Script de entrenamiento YOLO11
├── detectar_y_clasificar_tomates.py   # Script combinado
├── modelo1.py, modelo2.py, modelo3.py # Modelos de clasificación
├── Tomates.v2-tomates-v2.yolov11/     # Dataset con 4 categorías
└── requirements_backend.txt           # Dependencias
```

## 🎯 Uso

### Entrenar Modelo de Segmentación

```bash
python entrenar_yolo11_segmentacion.py
```

### Usar Modelo Entrenado

```python
from ultralytics import YOLO
model = YOLO('modelos_entrenados/SegmentacionYolo.pt')
results = model('imagen.jpg')
results[0].show()
```

### Aplicación Web

1. Inicia el backend: `python backend.py`
2. Abre `frontend/index.html` en tu navegador
3. Selecciona el modelo (Segmentación o Clasificación)
4. Carga una imagen o usa la cámara
5. ¡Ve los resultados!

## 📊 Modelos Incluidos

### Segmentación (YOLO11)
- Detecta y segmenta tomates
- 4 categorías: damaged, old, ripe, unripe
- Dataset: Tomates.v2-tomates-v2.yolov11

### Clasificación (TensorFlow)
- ResNet50
- EfficientNetB0
- DenseNet121
- 4 clases: Damaged, Old, Ripe, Unripe

## 🔧 Tecnologías

- **Backend**: Flask, Flask-CORS
- **IA**: Ultralytics (YOLO11), TensorFlow/Keras
- **Frontend**: HTML5, CSS3, JavaScript
- **Procesamiento**: PIL, NumPy, OpenCV

## 📝 Notas

- Los modelos grandes (`.pt`, `.h5`) no se suben a git (ver `.gitignore`)
- El dataset incluye imágenes de entrenamiento, validación y test
- La aplicación web funciona en móvil y desktop

## 📄 Licencia

Este proyecto está bajo la licencia CC BY 4.0 (según el dataset de Roboflow).

## 👤 Autor

Laura López

## 🔗 Enlaces

- Dataset: [Roboflow - Tomates](https://universe.roboflow.com/nathy/tomates-mi456/dataset/2)
- Repositorio: [GitHub](https://github.com/lauralopez9/tomatoes)

