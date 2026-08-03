@echo off
setlocal
cd /d "%~dp0"

git add app.py outputs assets requirements.txt .streamlit README.md
git commit -m "Actualizar forecast"
git push

if errorlevel 1 (
    echo.
    echo No se pudo publicar. Comprueba Git y la conexion con GitHub.
) else (
    echo.
    echo Forecast publicado correctamente.
)
pause
