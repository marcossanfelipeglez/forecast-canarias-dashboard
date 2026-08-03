# Forecast Canarias · publicación

Esta carpeta separa el cálculo local del dashboard remoto.

## Actualización diaria

1. Sustituye los archivos descargados dentro de la carpeta `data` de tu aplicación funcional.
2. Ejecuta `Actualizar Forecast.bat` y pega la ruta de esa carpeta funcional.
3. Comprueba que se hayan creado:
   - `outputs/dashboard_bundle.joblib`
   - `outputs/metadata.json`
4. Publica los cambios mediante GitHub Desktop o `Publicar en GitHub.bat`.

## Dashboard local

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Streamlit Community Cloud

Conecta este repositorio y selecciona `app.py` como archivo principal.
La nube solo recibe resultados agregados y predicciones; no recibe las bases internas.

## Seguridad

No añadas la carpeta `data`, modelos, archivos SQLite ni Excel al repositorio. El `.gitignore` ya los excluye.
