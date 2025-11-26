#!/bin/bash

set -e

echo ">>> Instalando dependencias del sistema para OpenCV (libGL)..."
apt-get update -y
apt-get install -y libgl1 libglib2.0-0

echo "=== Tomates backend startup v3 (auto-detect APP_DIR) ==="

# 1) Intentar usar /home/site/wwwroot como por defecto
APP_DIR="/home/site/wwwroot"

# Buscar backend.py en diferentes ubicaciones posibles
BACKEND_PATH=""

# Opción 1: En la raíz del proyecto
if [ -f "$APP_DIR/backend.py" ]; then
  BACKEND_PATH="$APP_DIR/backend.py"
  echo "✓ Encontrado backend.py en: $BACKEND_PATH"
# Opción 2: En proyecto_tomate/backend/backend.py
elif [ -f "$APP_DIR/proyecto_tomate/backend/backend.py" ]; then
  BACKEND_PATH="$APP_DIR/proyecto_tomate/backend/backend.py"
  echo "✓ Encontrado backend.py en: $BACKEND_PATH"
# Opción 3: Buscar en /tmp (donde Oryx suele descomprimir la app)
else
  echo "⚠ backend.py no está en $APP_DIR. Buscando en /tmp..."
  # Buscar en /tmp con mayor profundidad para encontrar archivos descomprimidos por Oryx
  FOUND_BACKEND=$(find /tmp -maxdepth 8 -type f -name "backend.py" 2>/dev/null | grep -E "(proyecto_tomate|backend)" | head -n 1 || true)
  
  # Si no se encuentra con el filtro, buscar sin filtro
  if [ -z "$FOUND_BACKEND" ]; then
    FOUND_BACKEND=$(find /tmp -maxdepth 8 -type f -name "backend.py" 2>/dev/null | head -n 1 || true)
  fi
  
  if [ -n "$FOUND_BACKEND" ]; then
    BACKEND_PATH="$FOUND_BACKEND"
    echo "✓ Encontrado backend.py en: $BACKEND_PATH"
  else
    echo "❌ ERROR: no se encontró backend.py"
    echo "--- Contenido de /home/site/wwwroot ---"
    ls -R /home/site/wwwroot 2>/dev/null | head -50 || true
    echo "--- Buscando en /tmp ---"
    find /tmp -maxdepth 4 -type d 2>/dev/null | head -20 || true
    exit 1
  fi
fi

# Determinar el directorio del backend y el nombre del módulo
BACKEND_DIR=$(dirname "$BACKEND_PATH")
BACKEND_MODULE=$(basename "$BACKEND_PATH" .py)

echo ">>> Directorio del backend: $BACKEND_DIR"
echo ">>> Módulo: $BACKEND_MODULE"

# Cambiar al directorio del backend
cd "$BACKEND_DIR"

# Verificar que requirements.txt esté disponible
if [ -f "requirements.txt" ]; then
  echo "✓ requirements.txt encontrado en: $BACKEND_DIR/requirements.txt"
elif [ -f "$APP_DIR/requirements.txt" ]; then
  echo "✓ requirements.txt encontrado en: $APP_DIR/requirements.txt"
  echo ">>> Copiando requirements.txt al directorio del backend..."
  cp "$APP_DIR/requirements.txt" "$BACKEND_DIR/requirements.txt"
else
  echo "⚠ requirements.txt no encontrado, asumiendo que las dependencias ya están instaladas"
fi

# Obtener el puerto desde la variable de entorno (Azure lo proporciona)
PORT=${PORT:-8000}
echo ">>> Puerto: $PORT"

# Activar el entorno virtual si existe
if [ -d "antenv" ]; then
  echo ">>> Activando entorno virtual antenv..."
  source antenv/bin/activate
elif [ -d "$APP_DIR/antenv" ]; then
  echo ">>> Activando entorno virtual desde $APP_DIR/antenv..."
  source "$APP_DIR/antenv/bin/activate"
fi

# Iniciar Gunicorn
echo ">>> Iniciando Gunicorn..."
echo ">>> Comando: gunicorn $BACKEND_MODULE:app --bind 0.0.0.0:$PORT --timeout 600 --workers 1"

exec gunicorn "$BACKEND_MODULE:app" \
  --bind "0.0.0.0:$PORT" \
  --timeout 600 \
  --workers 1 \
  --access-logfile - \
  --error-logfile -

