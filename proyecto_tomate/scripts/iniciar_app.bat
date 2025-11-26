@echo off
REM Script para iniciar backend y frontend del proyecto_tomate

echo ========================================
echo Iniciando Aplicacion de Clasificacion de Tomates
echo ========================================
echo.

REM Ir a la carpeta backend y lanzar el servidor
echo Iniciando backend...
cd /d "%~dp0..\backend"
start cmd /k "python backend.py"

REM Volver a la raiz del proyecto
cd /d "%~dp0.."

timeout /t 3 /nobreak >nul
echo.
echo Abriendo frontend en el navegador...
start "" "proyecto_tomate/frontend/index.html"
echo.
echo Si el navegador no se abre automaticamente, abre manualmente:
echo   proyecto_tomate/frontend/index.html
echo.
echo Presiona cualquier tecla para cerrar esta ventana...
pause >nul

