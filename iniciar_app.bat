@echo off
echo ========================================
echo Iniciando Aplicacion de Clasificacion de Tomates
echo ========================================
echo.
echo Iniciando backend...
start cmd /k "python backend.py"
timeout /t 3 /nobreak >nul
echo.
echo Abriendo frontend en el navegador...
start http://localhost:8000
echo.
echo Si el navegador no se abre automaticamente, abre manualmente:
echo frontend/index.html
echo.
echo Presiona cualquier tecla para cerrar esta ventana...
pause >nul

