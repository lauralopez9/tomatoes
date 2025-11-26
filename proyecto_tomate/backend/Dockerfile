FROM python:3.10

WORKDIR /app

# Copiar backend y requirements
COPY backend.py ./backend.py
COPY requirements.txt ./requirements.txt

# Copiar modelos (YOLO y DenseNet) y estructura de carpetas
COPY ../models ./models

# Instalar dependencias
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Variables de entorno por defecto (puedes sobrescribir en Azure)
ENV YOLO_SEG_MODEL_PATH=/app/models/modelos_entrenados/SegmentacionYolo.pt
ENV TF_CLASS_MODEL_PATH=/app/models/modelo_tomates_densenet121.h5
ENV FLASK_DEBUG=0

# Puerto estándar para App Service / Containers
ENV PORT=8000

# Comando de inicio con gunicorn
CMD ["gunicorn", "backend:app", "--bind", "0.0.0.0:8000", "--timeout", "600", "--workers", "1"]


