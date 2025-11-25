@echo off
echo Instalando dependencias del backend...
echo.

echo Instalando Flask y Flask-CORS...
pip install flask flask-cors

echo.
echo Instalando Ultralytics...
pip install ultralytics

echo.
echo Instalando TensorFlow...
pip install tensorflow

echo.
echo Instalando Pillow...
pip install pillow

echo.
echo Instalando Werkzeug...
pip install werkzeug

echo.
echo ========================================
echo Instalacion completada!
echo ========================================
echo.
echo Nota: numpy ya esta instalado (version 2.2.6)
echo.
pause

