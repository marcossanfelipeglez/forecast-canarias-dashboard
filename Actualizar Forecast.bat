@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo ACTUALIZAR FORECAST CANARIAS
echo ============================================================
set /p ORIGEN=Ruta de la carpeta funcional con data y pipeline.py: 

python actualizar_forecast.py --proyecto-origen "%ORIGEN%" --forzar
if errorlevel 1 (
    echo.
    echo ERROR: no se pudieron generar los resultados.
    pause
    exit /b 1
)

echo.
echo Resultados generados correctamente en outputs.
echo Ahora puedes publicarlos con GitHub Desktop o ejecutar publicar_github.bat.
pause
