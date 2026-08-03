from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

PUBLIC_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = PUBLIC_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


def resolver_proyecto_origen(ruta: str | None) -> Path:
    if ruta:
        proyecto = Path(ruta).expanduser().resolve()
    else:
        # Por defecto usa la carpeta funcional situada junto a esta carpeta.
        candidatos = [
            PUBLIC_DIR.parent / "forecast_canarias_literal_notebook",
            PUBLIC_DIR.parent,
        ]
        proyecto = next((p for p in candidatos if (p / "pipeline.py").exists()), candidatos[0])
    if not (proyecto / "pipeline.py").exists():
        raise FileNotFoundError(
            f"No se encontró pipeline.py en {proyecto}. "
            "Usa --proyecto-origen con la ruta de la app funcional."
        )
    return proyecto


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--proyecto-origen",
        help="Ruta de la carpeta local que contiene pipeline.py, data, models y outputs.",
    )
    parser.add_argument(
        "--forzar",
        action="store_true",
        help="Reconstruye los datasets aunque exista caché vigente.",
    )
    args = parser.parse_args()

    proyecto = resolver_proyecto_origen(args.proyecto_origen)
    sys.path.insert(0, str(proyecto))

    from pipeline import (  # noqa: E402
        _execute_exact_notebook_pipeline,
        generar_forecast_inicial_mes_actual,
        generar_prevision_anticipada_mes_siguiente,
        generar_reforecast_mes_actual,
        generar_validaciones_historicas,
    )

    print(f"Proyecto origen: {proyecto}")
    print("1/4 Construyendo dataset y validaciones históricas...")
    validaciones = generar_validaciones_historicas(force_rebuild=args.forzar)

    print("2/4 Generando forecast inicial congelado...")
    forecast_inicial = generar_forecast_inicial_mes_actual(force_rebuild=args.forzar)

    print("3/4 Generando seguimiento del mes actual...")
    seguimiento = generar_reforecast_mes_actual(force_rebuild=args.forzar)

    print("4/4 Generando previsión anticipada cuando corresponda...")
    anticipado = generar_prevision_anticipada_mes_siguiente(force_rebuild=args.forzar)

    dataset = _execute_exact_notebook_pipeline(force=args.forzar)
    ultima_fecha = pd.to_datetime(dataset["Fecha"], errors="coerce").max()
    ahora = datetime.now().astimezone()

    metadata = {
        "generado_en": ahora.isoformat(timespec="seconds"),
        "fecha_referencia": pd.Timestamp.today().normalize().date().isoformat(),
        "ultima_fecha_datos": ultima_fecha.date().isoformat() if pd.notna(ultima_fecha) else None,
        "proyecto_origen": str(proyecto),
    }

    bundle = {
        "metadata": metadata,
        "validaciones": validaciones,
        "forecast_inicial": forecast_inicial,
        "seguimiento": seguimiento,
        "anticipado": anticipado,
    }

    destino = OUTPUTS_DIR / "dashboard_bundle.joblib"
    temporal = OUTPUTS_DIR / "dashboard_bundle.tmp"
    joblib.dump(bundle, temporal, compress=3)
    temporal.replace(destino)

    (OUTPUTS_DIR / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nActualización completada.")
    print(f"Bundle publicado: {destino}")
    print(f"Última fecha de datos: {metadata['ultima_fecha_datos']}")


if __name__ == "__main__":
    main()
