#!/bin/bash

echo "========================================"
echo "Iniciando Aplicación de Clasificación de Tomates"
echo "========================================"
echo ""

# Iniciar backend en segundo plano
echo "Iniciando backend..."
python3 backend.py &
BACKEND_PID=$!

# Esperar un poco para que el backend inicie
sleep 3

# Abrir frontend
echo "Abriendo frontend..."
if command -v xdg-open &> /dev/null; then
    xdg-open http://localhost:8000
elif command -v open &> /dev/null; then
    open http://localhost:8000
fi

echo ""
echo "Backend corriendo en http://localhost:5000"
echo "Frontend disponible en frontend/index.html"
echo ""
echo "Presiona Ctrl+C para detener el servidor"
echo ""

# Esperar a que el usuario presione Ctrl+C
wait $BACKEND_PID

