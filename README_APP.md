# 🍅 Aplicación Web - Clasificador de Tomates

Aplicación web completa para detectar y clasificar tomates usando modelos de IA.

## 🚀 Características

- **Segmentación con YOLO11**: Detecta y segmenta múltiples tomates en una imagen
- **Clasificación con TensorFlow**: Clasifica el tipo de tomate (Ripe, Unripe, Damaged, Old)
- **Cámara móvil**: Toma fotos directamente desde tu celular
- **Carga de archivos**: Arrastra y suelta o selecciona imágenes
- **Diseño moderno**: Interfaz en colores morado pastel

## 📋 Requisitos

- Python 3.8+
- Modelos entrenados:
  - `modelos_entrenados/SegmentacionYolo.pt` (YOLO11)
  - `modelo_tomates_efficientnetb0.h5` (o ResNet50/DenseNet121)

## 🔧 Instalación

1. **Instalar dependencias del backend:**
```bash
pip install -r requirements_backend.txt
```

2. **Verificar que los modelos estén en su lugar:**
   - `modelos_entrenados/SegmentacionYolo.pt`
   - `modelo_tomates_efficientnetb0.h5` (o similar)

## ▶️ Uso

1. **Iniciar el backend:**
```bash
python backend.py
```

El servidor se iniciará en `http://localhost:5000`

2. **Abrir el frontend:**
   - Abre el archivo `frontend/index.html` en tu navegador
   - O usa un servidor local:
   ```bash
   cd frontend
   python -m http.server 8000
   ```
   Luego abre `http://localhost:8000`

3. **Usar la aplicación:**
   - Selecciona el modelo (Segmentación o Clasificación)
   - Carga una imagen o usa la cámara
   - Haz clic en "Procesar imagen"
   - ¡Ve los resultados!

## 📱 Uso en móvil

1. Asegúrate de que tu celular y computadora estén en la misma red WiFi
2. Encuentra la IP de tu computadora (ej: `192.168.1.100`)
3. En `frontend/script.js`, cambia:
   ```javascript
   const API_URL = 'http://TU_IP:5000';
   ```
4. Inicia el backend con:
   ```bash
   python backend.py
   ```
5. Abre el frontend desde tu celular usando la IP de tu computadora

## 🎨 Características del Frontend

- Diseño responsive (funciona en móvil y desktop)
- Colores morado pastel
- Animaciones suaves
- Drag & drop de imágenes
- Acceso a cámara del dispositivo
- Visualización de resultados en tiempo real

## 🔌 Endpoints del Backend

- `GET /` - Estado del servidor
- `GET /api/health` - Estado de los modelos
- `POST /api/segmentacion` - Procesar imagen con YOLO11
- `POST /api/clasificacion` - Procesar imagen con TensorFlow

## 📝 Notas

- El backend procesa imágenes en formato base64
- Las imágenes se guardan temporalmente en la carpeta `uploads/`
- Los resultados incluyen confianza y detalles de las detecciones

## 🐛 Solución de problemas

**Error: "Modelo no disponible"**
- Verifica que los archivos de modelo estén en las rutas correctas
- Revisa los logs del backend al iniciar

**Error: "No se pudo conectar con el backend"**
- Asegúrate de que el backend esté corriendo
- Verifica que el puerto 5000 no esté en uso
- Revisa la URL en `script.js`

**Cámara no funciona:**
- Asegúrate de dar permisos de cámara al navegador
- Usa HTTPS o localhost (algunos navegadores requieren HTTPS para la cámara)

