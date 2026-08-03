from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import joblib

PUBLIC_DIR = Path(__file__).resolve().parent
OUTPUTS_PUBLIC_DIR = PUBLIC_DIR / "outputs"
BUNDLE_PATH = OUTPUTS_PUBLIC_DIR / "dashboard_bundle.joblib"
METADATA_PATH = OUTPUTS_PUBLIC_DIR / "metadata.json"

APP_DIR = Path(__file__).resolve().parent
ASSETS_DIR = APP_DIR / "assets"
LOGO_CANDIDATOS = [
    ASSETS_DIR / "logo_empresa.png",
    ASSETS_DIR / "logo_empresa.jpg",
    ASSETS_DIR / "logo_empresa.jpeg",
]

# Forecast corporativo de Welysis expresado en cisternas.
# Valores transcritos de los gráficos facilitados para abril-julio de 2026.
FORECAST_WELYSIS_CISTERNAS = {
    "2026-04": {"GC": 44.0, "TF": 32.0, "CAN": 77.0},
    "2026-05": {"GC": 49.0, "TF": 40.0, "CAN": 89.0},
    "2026-06": {"GC": 49.0, "TF": 35.0, "CAN": 84.0},
    "2026-07": {"GC": 64.0, "TF": 45.0, "CAN": 108.0},
}


def aplicar_tema(modo):
    """Aplica un tema visual completo a Streamlit y devuelve colores Plotly."""
    oscuro = modo == "Oscuro"

    colores = {
        "fondo": "#0E1117" if oscuro else "#FFFFFF",
        "fondo_sec": "#161B22" if oscuro else "#F3F6F9",
        "fondo_card": "#1B222C" if oscuro else "#FFFFFF",
        "fondo_input": "#212936" if oscuro else "#FFFFFF",
        "texto": "#F3F4F6" if oscuro else "#1F2937",
        "texto_sec": "#C5CED8" if oscuro else "#667085",
        "borde": "#343D4A" if oscuro else "#D8DEE6",
        "acento": "#2F80ED",
        "hover": "#293241" if oscuro else "#EAF2FF",
        "plot_grid": "#303946" if oscuro else "#E5E7EB",
    }

    st.markdown(
        f"""
        <style>
        :root {{
            --app-bg: {colores['fondo']};
            --secondary-bg: {colores['fondo_sec']};
            --card-bg: {colores['fondo_card']};
            --input-bg: {colores['fondo_input']};
            --app-text: {colores['texto']};
            --muted-text: {colores['texto_sec']};
            --app-border: {colores['borde']};
            --app-hover: {colores['hover']};
        }}

        html, body, [data-testid="stAppViewContainer"], .stApp {{
            background: var(--app-bg) !important;
            color: var(--app-text) !important;
        }}

        /* Barra superior de Streamlit */
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stDecoration"] {{
            background: var(--app-bg) !important;
            color: var(--app-text) !important;
        }}
        [data-testid="stHeader"] {{
            border-bottom: 1px solid var(--app-border) !important;
        }}

        /* Barra lateral y botón para contraer/expandir */
        [data-testid="stSidebar"] {{
            background: var(--secondary-bg) !important;
            border-right: 1px solid var(--app-border) !important;
        }}
        [data-testid="stSidebar"] * {{
            color: var(--app-text);
        }}
        [data-testid="stSidebarCollapseButton"] button,
        [data-testid="stSidebarCollapsedControl"] button,
        button[kind="headerNoPadding"] {{
            background: var(--secondary-bg) !important;
            border: 1px solid var(--app-border) !important;
            color: var(--app-text) !important;
            opacity: 1 !important;
        }}
        [data-testid="stSidebarCollapseButton"] svg,
        [data-testid="stSidebarCollapsedControl"] svg,
        button[kind="headerNoPadding"] svg {{
            fill: var(--app-text) !important;
            stroke: var(--app-text) !important;
            color: var(--app-text) !important;
            opacity: 1 !important;
        }}

        /* Texto general */
        h1, h2, h3, h4, h5, h6, p, label,
        .stMarkdown, [data-testid="stCaptionContainer"] {{
            color: var(--app-text) !important;
        }}
        [data-testid="stCaptionContainer"] p {{
            color: var(--muted-text) !important;
        }}

        /* Métricas y contenedores */
        [data-testid="stMetric"] {{
            background: var(--card-bg) !important;
            border: 1px solid var(--app-border) !important;
            padding: 0.85rem !important;
            border-radius: 0.7rem !important;
        }}
        [data-testid="stMetric"] * {{
            color: var(--app-text) !important;
        }}

        /* Botones normales y de descarga */
        .stButton > button,
        .stDownloadButton > button {{
            background: var(--input-bg) !important;
            color: var(--app-text) !important;
            border: 1px solid var(--app-border) !important;
        }}
        .stButton > button:hover,
        .stDownloadButton > button:hover {{
            background: var(--app-hover) !important;
            color: var(--app-text) !important;
            border-color: {colores['acento']} !important;
        }}
        .stButton > button:disabled,
        .stDownloadButton > button:disabled {{
            background: var(--secondary-bg) !important;
            color: var(--muted-text) !important;
            opacity: 0.75 !important;
        }}

        /* Selectores, radios y desplegables */
        [data-baseweb="select"] > div,
        [data-baseweb="input"] > div,
        [data-baseweb="popover"] > div,
        [role="listbox"] {{
            background: var(--input-bg) !important;
            color: var(--app-text) !important;
            border-color: var(--app-border) !important;
        }}
        [role="option"] {{
            background: var(--input-bg) !important;
            color: var(--app-text) !important;
        }}
        [role="option"]:hover {{
            background: var(--app-hover) !important;
        }}

        /* Expander: elimina los rectángulos blancos */
        [data-testid="stExpander"] {{
            background: var(--card-bg) !important;
            border: 1px solid var(--app-border) !important;
            border-radius: 0.65rem !important;
            overflow: hidden !important;
        }}
        [data-testid="stExpander"] details,
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] div {{
            background: transparent !important;
            color: var(--app-text) !important;
        }}
        [data-testid="stExpander"] svg {{
            fill: var(--app-text) !important;
            stroke: var(--app-text) !important;
        }}

        /* Pestañas */
        [data-baseweb="tab-list"] {{
            background: var(--app-bg) !important;
        }}
        [data-baseweb="tab"] {{
            color: var(--muted-text) !important;
        }}
        [aria-selected="true"][data-baseweb="tab"] {{
            color: var(--app-text) !important;
        }}

        /* Gráficos y tablas */
        [data-testid="stPlotlyChart"],
        [data-testid="stDataFrame"],
        [data-testid="stTable"] {{
            background: var(--card-bg) !important;
            border: 1px solid var(--app-border) !important;
            border-radius: 0.6rem !important;
            overflow: hidden !important;
        }}
        [data-testid="stPlotlyChart"] > div,
        [data-testid="stPlotlyChart"] iframe {{
            background: var(--card-bg) !important;
        }}

        /* Alertas y mensajes */
        [data-testid="stAlert"] {{
            color: var(--app-text) !important;
            border: 1px solid var(--app-border) !important;
        }}

        /* Menú superior y elementos SVG */
        [data-testid="stMainMenu"] button,
        [data-testid="stToolbar"] button {{
            color: var(--app-text) !important;
            background: transparent !important;
        }}
        [data-testid="stMainMenu"] svg,
        [data-testid="stToolbar"] svg {{
            fill: var(--app-text) !important;
            stroke: var(--app-text) !important;
        }}

        /* Ajustes generales de densidad */
        .block-container {{
            max-width: 1500px;
            padding-top: 2.2rem;
            padding-bottom: 2rem;
        }}
        [data-testid="stMetricValue"] {{
            font-size: clamp(1.45rem, 2.2vw, 2.25rem) !important;
            line-height: 1.1 !important;
        }}
        [data-testid="stMetricLabel"] {{
            font-size: 0.92rem !important;
        }}
        [data-testid="stMetricDelta"] {{
            font-size: 0.82rem !important;
        }}

        /* Móvil: una sola columna, menos márgenes y controles manejables */
        @media (max-width: 768px) {{
            .block-container {{
                padding: 0.75rem 0.72rem 1.5rem 0.72rem !important;
                max-width: 100% !important;
            }}

            h1 {{
                font-size: 1.65rem !important;
                line-height: 1.15 !important;
                margin-bottom: 0.35rem !important;
            }}
            h2 {{
                font-size: 1.35rem !important;
                line-height: 1.2 !important;
            }}
            h3 {{
                font-size: 1.12rem !important;
            }}
            p, label, .stMarkdown {{
                font-size: 0.92rem !important;
            }}

            /* Apilar todas las columnas de Streamlit */
            [data-testid="stHorizontalBlock"] {{
                flex-wrap: wrap !important;
                gap: 0.55rem !important;
            }}
            [data-testid="column"] {{
                width: 100% !important;
                flex: 1 1 100% !important;
                min-width: 100% !important;
            }}

            [data-testid="stMetric"] {{
                padding: 0.72rem 0.8rem !important;
                border-radius: 0.65rem !important;
                min-height: auto !important;
            }}
            [data-testid="stMetricValue"] {{
                font-size: 1.58rem !important;
            }}
            [data-testid="stMetricLabel"] {{
                font-size: 0.84rem !important;
            }}
            [data-testid="stMetricDelta"] {{
                font-size: 0.76rem !important;
            }}

            /* Cabecera y logo más compactos */
            [data-testid="stImage"] img {{
                max-height: 58px !important;
                width: auto !important;
                object-fit: contain !important;
            }}

            /* Pestañas desplazables, sin comprimir el texto */
            [data-baseweb="tab-list"] {{
                overflow-x: auto !important;
                overflow-y: hidden !important;
                white-space: nowrap !important;
                scrollbar-width: thin;
                gap: 0.15rem !important;
            }}
            [data-baseweb="tab"] {{
                flex: 0 0 auto !important;
                padding: 0.55rem 0.65rem !important;
                font-size: 0.82rem !important;
            }}

            /* Controles ocupan todo el ancho */
            .stButton > button,
            .stDownloadButton > button {{
                width: 100% !important;
                min-height: 2.65rem !important;
            }}
            [data-baseweb="select"] {{
                width: 100% !important;
            }}

            /* Gráficos sin marco excesivo y con scroll evitado */
            [data-testid="stPlotlyChart"] {{
                border-radius: 0.5rem !important;
                width: 100% !important;
            }}
            [data-testid="stPlotlyChart"] > div {{
                width: 100% !important;
            }}

            /* Tablas: altura limitada y desplazamiento horizontal interno */
            [data-testid="stDataFrame"] {{
                max-height: 420px !important;
                overflow: auto !important;
            }}

            /* Sidebar usable en pantallas pequeñas */
            [data-testid="stSidebar"] {{
                width: min(86vw, 310px) !important;
            }}
            [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{
                gap: 0.65rem !important;
            }}

            /* Expander más compacto */
            [data-testid="stExpander"] summary {{
                padding: 0.65rem 0.75rem !important;
                font-size: 0.88rem !important;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    return colores


def estilizar_figura(fig, colores):
    """
    Aplica el tema de la aplicación y bloquea los ejes del gráfico.

    Los ejes quedan fijos para impedir zoom, desplazamientos o cambios
    accidentales cuando el usuario hace scroll desde un móvil.
    """
    fig.update_layout(
        template="plotly_dark" if colores["fondo"] == "#0E1117" else "plotly_white",
        paper_bgcolor=colores["fondo_card"],
        plot_bgcolor=colores["fondo_card"],
        font={"color": colores["texto"]},
        dragmode=False,
        hovermode=False,
        legend={
            "font": {"color": colores["texto"]},
            "bgcolor": "rgba(0,0,0,0)",
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "center",
            "x": 0.5,
        },
        margin={"l": 18, "r": 10, "t": 78, "b": 35},
        height=360,
        autosize=True,
    )
    fig.update_xaxes(
        color=colores["texto"],
        gridcolor=colores["plot_grid"],
        zerolinecolor=colores["plot_grid"],
        linecolor=colores["borde"],
        fixedrange=True,
        automargin=True,
        nticks=7,
    )
    fig.update_yaxes(
        color=colores["texto"],
        gridcolor=colores["plot_grid"],
        zerolinecolor=colores["plot_grid"],
        linecolor=colores["borde"],
        fixedrange=True,
        automargin=True,
        title=None,
    )
    return fig


def mostrar_grafico(fig, key=None):
    """
    Muestra un gráfico completamente estático y responsive.

    staticPlot=True hace que Plotly trate el gráfico como una imagen:
    - no captura gestos táctiles;
    - no permite zoom;
    - no permite arrastrar;
    - no modifica los ejes;
    - facilita el scroll vertical en móvil.
    """
    st.plotly_chart(
        fig,
        use_container_width=True,
        key=key,
        config={
            "responsive": True,
            "displayModeBar": False,
            "scrollZoom": False,
            "doubleClick": False,
            "showTips": False,
            "staticPlot": True,
        },
    )


def encontrar_logo():
    return next((ruta for ruta in LOGO_CANDIDATOS if ruta.exists()), None)


def comparativa_welysis(periodo, res):
    valores = FORECAST_WELYSIS_CISTERNAS.get(periodo)
    if valores is None:
        return pd.DataFrame()

    filas = []
    nombres = {"GC": "Gran Canaria", "TF": "Tenerife", "CAN": "Canarias"}
    for isla in ["GC", "TF", "CAN"]:
        fila = res[res["isla"] == isla]
        if fila.empty:
            continue
        fila = fila.iloc[0]
        real_cis = float(fila["toneladas_reales"]) / 24.0
        modelo_cis = float(fila["toneladas_previstas"]) / 24.0
        welysis_cis = float(valores[isla])
        error_modelo = abs(modelo_cis - real_cis)
        error_welysis = abs(welysis_cis - real_cis)
        ventaja = error_welysis - error_modelo
        if abs(ventaja) < 1e-9:
            ganador = "Empate"
        elif ventaja > 0:
            ganador = "Modelo ML"
        else:
            ganador = "Welysis"
        filas.append({
            "Zona": nombres[isla],
            "Real (cisternas)": real_cis,
            "Modelo ML (cisternas)": modelo_cis,
            "Forecast Welysis (cisternas)": welysis_cis,
            "Error modelo": error_modelo,
            "Error Welysis": error_welysis,
            "Ganador": ganador,
            "Ventaja del ganador (cisternas)": abs(ventaja),
        })
    return pd.DataFrame(filas)

st.set_page_config(
    page_title="Forecast Hipoclorito Canarias",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="auto",
)

MESES_ES = {
    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
    5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
    9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre",
}


def nombre_mes(fecha):
    fecha = pd.Timestamp(fecha)
    return f"{MESES_ES[fecha.month]} de {fecha.year}"


@st.cache_resource(show_spinner=False)
def cargar_bundle(_firma):
    if not BUNDLE_PATH.exists():
        raise FileNotFoundError(
            "No existe outputs/dashboard_bundle.joblib. "
            "Ejecuta primero actualizar_forecast.py en tu ordenador."
        )
    return joblib.load(BUNDLE_PATH)


def cargar_forecast_inicial(_firma):
    return cargar_bundle(_firma)["forecast_inicial"]


def cargar_backtests(_firma):
    return cargar_bundle(_firma)["validaciones"]


def cargar_reforecast(_firma):
    return cargar_bundle(_firma)["seguimiento"]


def cargar_anticipado(_firma):
    return cargar_bundle(_firma).get("anticipado")


modo_visual = st.sidebar.radio(
    "Apariencia",
    ["Claro", "Oscuro"],
    horizontal=True,
    key="modo_visual",
)
COLORES_TEMA = aplicar_tema(modo_visual)

logo_app = encontrar_logo()
if logo_app is not None:
    col_logo, col_titulo = st.columns([1, 5], vertical_alignment="center")
    with col_logo:
        st.image(str(logo_app), use_container_width=True)
    with col_titulo:
        st.title("Forecast de hipoclorito · Canarias")
        st.caption("Forecast oficial congelado, seguimiento vivo y previsión anticipada")
else:
    st.title("Forecast de hipoclorito · Canarias")
    st.caption("Forecast oficial congelado, seguimiento vivo y previsión anticipada")
    st.sidebar.info(
        "Logo: guarda el archivo como assets/logo_empresa.png "
        "y reinicia la aplicación."
    )

firma_datos = BUNDLE_PATH.stat().st_mtime_ns if BUNDLE_PATH.exists() else 0
try:
    bundle_meta = cargar_bundle(firma_datos).get("metadata", {})
except Exception:
    bundle_meta = {}

fecha_ref_texto = bundle_meta.get("fecha_referencia")
hoy_app = (
    pd.Timestamp(fecha_ref_texto).normalize()
    if fecha_ref_texto
    else pd.Timestamp.today().normalize()
)
mostrar_anticipado = cargar_bundle(firma_datos).get("anticipado") is not None if BUNDLE_PATH.exists() else False

if st.sidebar.button("🔄 Recargar resultados", use_container_width=True):
    st.cache_resource.clear()
    st.rerun()

st.sidebar.caption(f"Datos publicados con corte: {bundle_meta.get('ultima_fecha_datos', 'no disponible')}")
st.sidebar.caption(f"Actualización: {bundle_meta.get('generado_en', 'no disponible')}")
st.sidebar.caption("En móvil, usa la flecha superior para ocultar este panel.")


_etiquetas_tabs = [
    "Validación histórica",
    "Forecast inicial del mes",
    "Seguimiento del mes actual",
]
if mostrar_anticipado:
    _etiquetas_tabs.append("Previsión del mes siguiente")

_tabs = st.tabs(_etiquetas_tabs)
tab_validacion, tab_inicial, tab_seguimiento = _tabs[:3]
tab_anticipado = _tabs[3] if mostrar_anticipado else None

with tab_validacion:
    st.header("Validación histórica del forecast recursivo")
    st.caption(
        "Cada mes se simula utilizando únicamente los datos disponibles hasta "
        "el último día del mes anterior. Los datos reales del mes evaluado no "
        "entran en los lags de la predicción."
    )

    try:
        with st.spinner("Calculando simulaciones históricas..."):
            comparaciones, resumen_backtests = cargar_backtests(firma_datos)
    except Exception as exc:
        st.error("No se pudieron calcular las validaciones históricas.")
        with st.expander("Detalle técnico"):
            st.exception(exc)
    else:
        periodos = sorted(comparaciones["periodo"].unique().tolist())
        etiquetas = {
            p: nombre_mes(pd.Timestamp(f"{p}-01")) for p in periodos
        }
        periodo = st.selectbox(
            "Mes evaluado",
            periodos,
            format_func=lambda p: etiquetas[p],
            index=0,
        )

        comp = comparaciones[comparaciones["periodo"] == periodo].copy()
        res = resumen_backtests[
            resumen_backtests["periodo"] == periodo
        ].copy()
        total = res[res["isla"] == "CAN"].iloc[0]

        toneladas_reales = float(total["toneladas_reales"])
        toneladas_previstas = float(total["toneladas_previstas"])
        diferencia_abs_t = abs(toneladas_previstas - toneladas_reales)
        cisternas_reales = toneladas_reales / 24.0
        cisternas_previstas = toneladas_previstas / 24.0
        diferencia_abs_cisternas = abs(cisternas_previstas - cisternas_reales)

        st.subheader("Resultado mensual · Canarias")
        c1, c2, c3 = st.columns(3)
        c1.metric("Toneladas reales", f"{toneladas_reales:,.1f} t")
        c2.metric("Toneladas previstas", f"{toneladas_previstas:,.1f} t")
        c3.metric("Diferencia absoluta", f"{diferencia_abs_t:,.1f} t")

        c4, c5, c6 = st.columns(3)
        c4.metric("Cisternas reales", f"{cisternas_reales:,.1f}")
        c5.metric("Cisternas previstas", f"{cisternas_previstas:,.1f}")
        c6.metric(
            "Diferencia absoluta de cisternas",
            f"{diferencia_abs_cisternas:,.1f}",
        )

        st.subheader("Comparación con forecast Welysis")
        comparativa_w = comparativa_welysis(periodo, res)
        if comparativa_w.empty:
            st.info("No hay forecast Welysis configurado para este periodo.")
        else:
            fila_can = comparativa_w[comparativa_w["Zona"] == "Canarias"].iloc[0]
            w1, w2, w3, w4 = st.columns(4)
            w1.metric(
                "Forecast Welysis · Canarias",
                f'{fila_can["Forecast Welysis (cisternas)"]:,.1f} cisternas',
            )
            w2.metric(
                "Error modelo ML",
                f'{fila_can["Error modelo"]:,.2f} cisternas',
            )
            w3.metric(
                "Error Welysis",
                f'{fila_can["Error Welysis"]:,.2f} cisternas',
            )
            w4.metric(
                "Ganador",
                fila_can["Ganador"],
                f'{fila_can["Ventaja del ganador (cisternas)"]:,.2f} cisternas',
            )
            with st.expander("Ver comparación completa por zona"):
                st.dataframe(
                    comparativa_w.style.format({
                    "Real (cisternas)": "{:,.2f}",
                    "Modelo ML (cisternas)": "{:,.2f}",
                    "Forecast Welysis (cisternas)": "{:,.2f}",
                    "Error modelo": "{:,.2f}",
                    "Error Welysis": "{:,.2f}",
                    "Ventaja del ganador (cisternas)": "{:,.2f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

        vista = st.radio(
            "Visualización",
            ["Diaria", "Acumulada"],
            horizontal=True,
        )

        def grafico_zona(datos, titulo):
            datos = datos.sort_values("Fecha").copy()
            if vista == "Diaria":
                real_col = "toneladas_real"
                pred_col = "toneladas_pred"
                y_label = "Toneladas"
                real_label = "Real"
                pred_label = "Previsto"
            else:
                real_col = "real_acumulado"
                pred_col = "pred_acumulado"
                y_label = "Toneladas acumuladas"
                real_label = "Real acumulado"
                pred_label = "Previsto acumulado"

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=datos["Fecha"],
                y=datos[real_col],
                mode="lines+markers",
                name=real_label,
            ))
            fig.add_trace(go.Scatter(
                x=datos["Fecha"],
                y=datos[pred_col],
                mode="lines+markers",
                name=pred_label,
                line={"dash": "dot"},
            ))
            fig.update_layout(
                title=titulo,
                xaxis_title="Fecha",
                yaxis_title=y_label,
                legend_title_text="",
                hovermode="x unified",
                
            )
            return fig

        gc_comp = comp[comp["isla"] == "GC"].copy()
        tf_comp = comp[comp["isla"] == "TF"].copy()
        can_comp = (
            comp.groupby("Fecha", as_index=False)
            .agg(
                toneladas_real=("toneladas_real", "sum"),
                toneladas_pred=("toneladas_pred", "sum"),
            )
            .sort_values("Fecha")
        )
        can_comp["real_acumulado"] = can_comp["toneladas_real"].cumsum()
        can_comp["pred_acumulado"] = can_comp["toneladas_pred"].cumsum()

        st.subheader("Gran Canaria")
        mostrar_grafico(
            estilizar_figura(
                grafico_zona(gc_comp, "Real frente a previsto · Gran Canaria"),
                COLORES_TEMA,
            ),
            key=f"validacion_gc_{periodo}_{vista}",
        )

        st.subheader("Tenerife")
        mostrar_grafico(
            estilizar_figura(
                grafico_zona(tf_comp, "Real frente a previsto · Tenerife"),
                COLORES_TEMA,
            ),
            key=f"validacion_tf_{periodo}_{vista}",
        )

        st.subheader("Canarias")
        mostrar_grafico(
            estilizar_figura(
                grafico_zona(can_comp, "Real frente a previsto · Total Canarias"),
                COLORES_TEMA,
            ),
            key=f"validacion_can_{periodo}_{vista}",
        )

        with st.expander("Ver resultado detallado por isla"):
            tabla_res = res.copy()
            tabla_res["Cisternas reales"] = tabla_res["toneladas_reales"] / 24.0
            tabla_res["Cisternas previstas"] = tabla_res["toneladas_previstas"] / 24.0
            tabla_res["Diferencia absoluta (t)"] = (
                tabla_res["toneladas_previstas"] - tabla_res["toneladas_reales"]
            ).abs()
            tabla_res["Diferencia absoluta (cisternas)"] = (
                tabla_res["Cisternas previstas"] - tabla_res["Cisternas reales"]
            ).abs()
            tabla_res = tabla_res.rename(columns={
                "nombre_isla": "Zona",
                "dias_evaluados": "Días evaluados",
                "toneladas_reales": "Toneladas reales",
                "toneladas_previstas": "Toneladas previstas",
            })[[
                "Zona", "Días evaluados", "Toneladas reales",
                "Toneladas previstas", "Diferencia absoluta (t)",
                "Cisternas reales", "Cisternas previstas",
                "Diferencia absoluta (cisternas)",
            ]]
            st.dataframe(
                tabla_res.style.format({
                    "Toneladas reales": "{:,.1f}",
                    "Toneladas previstas": "{:,.1f}",
                    "Diferencia absoluta (t)": "{:,.1f}",
                    "Cisternas reales": "{:,.1f}",
                    "Cisternas previstas": "{:,.1f}",
                    "Diferencia absoluta (cisternas)": "{:,.1f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

        with st.expander("Ver comparativa mensual abril–julio"):
            historico_total = resumen_backtests[
                resumen_backtests["isla"] == "CAN"
            ].copy()
            historico_total["Mes"] = historico_total["periodo"].map(etiquetas)
            historico_total["Diferencia absoluta (t)"] = (
                historico_total["toneladas_previstas"]
                - historico_total["toneladas_reales"]
            ).abs()
            historico_total["Cisternas reales"] = (
                historico_total["toneladas_reales"] / 24.0
            )
            historico_total["Cisternas previstas"] = (
                historico_total["toneladas_previstas"] / 24.0
            )
            historico_total["Diferencia absoluta (cisternas)"] = (
                historico_total["Cisternas previstas"]
                - historico_total["Cisternas reales"]
            ).abs()
            historico_total["Forecast Welysis (cisternas)"] = historico_total["periodo"].map(
                lambda p: FORECAST_WELYSIS_CISTERNAS.get(p, {}).get("CAN", np.nan)
            )
            historico_total["Error Welysis (cisternas)"] = (
                historico_total["Forecast Welysis (cisternas)"]
                - historico_total["Cisternas reales"]
            ).abs()
            historico_total["Ganador"] = np.where(
                historico_total["Diferencia absoluta (cisternas)"]
                < historico_total["Error Welysis (cisternas)"],
                "Modelo ML",
                np.where(
                    historico_total["Diferencia absoluta (cisternas)"]
                    > historico_total["Error Welysis (cisternas)"],
                    "Welysis",
                    "Empate",
                ),
            )
            historico_total["Ventaja del ganador (cisternas)"] = (
                historico_total["Diferencia absoluta (cisternas)"]
                - historico_total["Error Welysis (cisternas)"]
            ).abs()
            historico_total = historico_total.rename(columns={
                "toneladas_reales": "Real (t)",
                "toneladas_previstas": "Previsto (t)",
            })[[
                "Mes", "Real (t)", "Previsto (t)",
                "Diferencia absoluta (t)", "Cisternas reales",
                "Cisternas previstas", "Diferencia absoluta (cisternas)",
                "Forecast Welysis (cisternas)", "Error Welysis (cisternas)",
                "Ganador", "Ventaja del ganador (cisternas)",
            ]]
            st.dataframe(
                historico_total.style.format({
                    "Real (t)": "{:,.1f}",
                    "Previsto (t)": "{:,.1f}",
                    "Diferencia absoluta (t)": "{:,.1f}",
                    "Cisternas reales": "{:,.1f}",
                    "Cisternas previstas": "{:,.1f}",
                    "Diferencia absoluta (cisternas)": "{:,.1f}",
                    "Forecast Welysis (cisternas)": "{:,.1f}",
                    "Error Welysis (cisternas)": "{:,.1f}",
                    "Ventaja del ganador (cisternas)": "{:,.1f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

        with st.expander("Ver detalle diario del mes"):
            detalle = comp.copy()
            detalle["Diferencia absoluta diaria"] = (
                detalle["toneladas_pred"] - detalle["toneladas_real"]
            ).abs()
            detalle["Cisternas reales"] = detalle["toneladas_real"] / 24.0
            detalle["Cisternas previstas"] = detalle["toneladas_pred"] / 24.0
            detalle = detalle.rename(columns={
                "nombre_isla": "Isla",
                "toneladas_real": "Real diario",
                "toneladas_pred": "Previsto diario",
                "real_acumulado": "Real acumulado",
                "pred_acumulado": "Previsto acumulado",
            })[[
                "Fecha", "Isla", "Real diario", "Previsto diario",
                "Diferencia absoluta diaria", "Cisternas reales",
                "Cisternas previstas", "Real acumulado", "Previsto acumulado",
            ]]
            st.dataframe(
                detalle.style.format({
                    "Real diario": "{:,.1f}",
                    "Previsto diario": "{:,.1f}",
                    "Diferencia absoluta diaria": "{:,.1f}",
                    "Cisternas reales": "{:,.2f}",
                    "Cisternas previstas": "{:,.2f}",
                    "Real acumulado": "{:,.1f}",
                    "Previsto acumulado": "{:,.1f}",
                }),
                use_container_width=True,
                hide_index=True,
            )


with tab_inicial:
    try:
        with st.spinner("Cargando el forecast oficial del mes..."):
            forecast, resumen, inicio, fin, corte = cargar_forecast_inicial(firma_datos)
    except Exception as exc:
        st.error("No se pudo cargar el forecast inicial del mes actual.")
        with st.expander("Detalle técnico"):
            st.exception(exc)
    else:
        st.header(f"Forecast inicial · {nombre_mes(inicio)}")
        st.caption(
            f"Predicción oficial calculada con datos disponibles hasta el "
            f"{corte:%d/%m/%Y}. Permanece congelada durante todo el mes."
        )

        total = resumen[resumen["isla"] == "CAN"].iloc[0]
        gc = resumen[resumen["isla"] == "GC"].iloc[0]
        tf = resumen[resumen["isla"] == "TF"].iloc[0]

        c1, c2, c3 = st.columns(3)
        c1.metric("Gran Canaria", f'{gc["toneladas_predichas"]:,.1f} t', f'{gc["cisternas_equivalentes_predichas"]:,.1f} cisternas')
        c2.metric("Tenerife", f'{tf["toneladas_predichas"]:,.1f} t', f'{tf["cisternas_equivalentes_predichas"]:,.1f} cisternas')
        c3.metric("Total Canarias", f'{total["toneladas_predichas"]:,.1f} t', f'{total["cisternas_equivalentes_predichas"]:,.1f} cisternas')

        st.subheader("Previsión diaria oficial")
        fig = px.line(
            forecast,
            x="Fecha",
            y="toneladas_pred",
            color="nombre_isla",
            markers=True,
            labels={"toneladas_pred": "Toneladas previstas", "nombre_isla": "Isla"},
        )
        fig.update_layout(legend_title_text="")
        mostrar_grafico(estilizar_figura(fig, COLORES_TEMA))

        with st.expander("Ver resumen por isla"):
            tabla = resumen.rename(columns={
                "nombre_isla": "Zona",
                "dias_previstos": "Días previstos",
                "toneladas_predichas": "Toneladas previstas",
                "cisternas_equivalentes_predichas": "Cisternas equivalentes",
            })[["Zona", "Días previstos", "Toneladas previstas", "Cisternas equivalentes"]]
            st.dataframe(
                tabla.style.format({
                    "Toneladas previstas": "{:,.1f}",
                    "Cisternas equivalentes": "{:,.1f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

        with st.expander("Ver detalle diario"):
            st.dataframe(forecast, use_container_width=True, hide_index=True)

with tab_seguimiento:
    try:
        with st.spinner("Calculando el seguimiento del mes actual..."):
            seguimiento, resumen_seg, inicio_seg, fin_seg, corte_seg = cargar_reforecast(firma_datos)
    except Exception as exc:
        st.error("No se pudo calcular el seguimiento mensual.")
        with st.expander("Detalle técnico"):
            st.exception(exc)
    else:
        st.header(f"Seguimiento de {nombre_mes(inicio_seg)}")
        st.caption(
            f"Datos reales incorporados hasta el {corte_seg:%d/%m/%Y}. "
            f"El resto del mes se pronostica hasta el {fin_seg:%d/%m/%Y}."
        )

        total_seg = resumen_seg[resumen_seg["isla"] == "CAN"].iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Real acumulado", f'{total_seg["real_acumulado"]:,.1f} t')
        c2.metric("Forecast pendiente", f'{total_seg["forecast_pendiente"]:,.1f} t')
        c3.metric("Cierre mensual estimado", f'{total_seg["cierre_estimado"]:,.1f} t', f'{total_seg["cisternas_cierre_estimado"]:,.1f} cisternas')
        c4.metric(
            "Variación vs. forecast inicial",
            f'{total_seg["desviacion_vs_inicial_t"]:+,.1f} t',
            delta_color="inverse",
        )

        st.subheader("Real, forecast actualizado y forecast inicial congelado")
        fig = go.Figure()
        for isla_nombre in seguimiento["nombre_isla"].dropna().unique():
            d = seguimiento[seguimiento["nombre_isla"] == isla_nombre]
            fig.add_trace(go.Scatter(
                x=d["Fecha"], y=d["forecast_inicial"],
                mode="lines", name=f"{isla_nombre} · forecast inicial",
                line={"dash": "dot"},
            ))
            reales = d[d["toneladas_real"].notna()]
            fig.add_trace(go.Scatter(
                x=reales["Fecha"], y=reales["toneladas_real"],
                mode="lines+markers", name=f"{isla_nombre} · real",
            ))
            futuro = d[d["forecast_actualizado"].notna()]
            fig.add_trace(go.Scatter(
                x=futuro["Fecha"], y=futuro["forecast_actualizado"],
                mode="lines+markers", name=f"{isla_nombre} · forecast actualizado",
            ))
        fig.update_layout(
            xaxis_title="Fecha",
            yaxis_title="Toneladas",
            legend_title_text="",
            
        )
        mostrar_grafico(estilizar_figura(fig, COLORES_TEMA))

        with st.expander("Ver cierre estimado por isla"):
            tabla_seg = resumen_seg.rename(columns={
                "nombre_isla": "Zona",
                "real_acumulado": "Real acumulado",
                "forecast_pendiente": "Forecast pendiente",
                "cierre_estimado": "Cierre estimado",
                "forecast_inicial_mes": "Forecast inicial",
                "desviacion_vs_inicial_t": "Variación vs. inicial",
                "cisternas_cierre_estimado": "Cisternas estimadas",
            })[[
                "Zona", "Real acumulado", "Forecast pendiente", "Cierre estimado",
                "Forecast inicial", "Variación vs. inicial", "Cisternas estimadas",
            ]]
            st.dataframe(
                tabla_seg.style.format({
                    "Real acumulado": "{:,.1f}",
                    "Forecast pendiente": "{:,.1f}",
                    "Cierre estimado": "{:,.1f}",
                    "Forecast inicial": "{:,.1f}",
                    "Variación vs. inicial": "{:+,.1f}",
                    "Cisternas estimadas": "{:,.1f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

        with st.expander("Ver calendario diario completo"):
            st.dataframe(seguimiento, use_container_width=True, hide_index=True)

if mostrar_anticipado and tab_anticipado is not None:
    with tab_anticipado:
        try:
            with st.spinner("Actualizando la previsión anticipada..."):
                anticipado = cargar_anticipado(firma_datos)
        except Exception as exc:
            st.error("No se pudo calcular la previsión del mes siguiente.")
            with st.expander("Detalle técnico"):
                st.exception(exc)
        else:
            if anticipado is None:
                st.info("Esta pestaña se activa a partir del día 25 de cada mes.")
            else:
                forecast_ant, resumen_ant, inicio_ant, fin_ant, corte_ant = anticipado
                st.header(f"Previsión anticipada · {nombre_mes(inicio_ant)}")
                st.caption(
                    f"Versión viva calculada con datos disponibles hasta el "
                    f"{corte_ant:%d/%m/%Y}. Se actualizará con cada nueva carga "
                    f"de datos hasta el cierre del mes actual."
                )

                total_ant = resumen_ant[resumen_ant["isla"] == "CAN"].iloc[0]
                gc_ant = resumen_ant[resumen_ant["isla"] == "GC"].iloc[0]
                tf_ant = resumen_ant[resumen_ant["isla"] == "TF"].iloc[0]
                c1, c2, c3 = st.columns(3)
                c1.metric("Gran Canaria", f'{gc_ant["toneladas_predichas"]:,.1f} t', f'{gc_ant["cisternas_equivalentes_predichas"]:,.1f} cisternas')
                c2.metric("Tenerife", f'{tf_ant["toneladas_predichas"]:,.1f} t', f'{tf_ant["cisternas_equivalentes_predichas"]:,.1f} cisternas')
                c3.metric("Total Canarias", f'{total_ant["toneladas_predichas"]:,.1f} t', f'{total_ant["cisternas_equivalentes_predichas"]:,.1f} cisternas')

                fig = px.line(
                    forecast_ant,
                    x="Fecha",
                    y="toneladas_pred",
                    color="nombre_isla",
                    markers=True,
                    labels={"toneladas_pred": "Toneladas previstas", "nombre_isla": "Isla"},
                )
                fig.update_layout(legend_title_text="")
                mostrar_grafico(estilizar_figura(fig, COLORES_TEMA))

                with st.expander("Ver detalle diario"):
                    st.dataframe(forecast_ant, use_container_width=True, hide_index=True)

# Informe único con las áreas disponibles.
try:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        if "resumen_backtests" in locals():
            resumen_backtests.to_excel(writer, sheet_name="Backtests resumen", index=False)
            comparaciones.to_excel(writer, sheet_name="Backtests detalle", index=False)
        if "forecast" in locals():
            resumen.to_excel(writer, sheet_name="Forecast inicial resumen", index=False)
            forecast.to_excel(writer, sheet_name="Forecast inicial diario", index=False)
        if "seguimiento" in locals():
            resumen_seg.to_excel(writer, sheet_name="Seguimiento resumen", index=False)
            seguimiento.to_excel(writer, sheet_name="Seguimiento diario", index=False)
        if "forecast_ant" in locals():
            resumen_ant.to_excel(writer, sheet_name="Anticipado resumen", index=False)
            forecast_ant.to_excel(writer, sheet_name="Anticipado diario", index=False)
    st.download_button(
        "Descargar informe completo Excel",
        buffer.getvalue(),
        file_name="informe_forecast_canarias.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
except Exception:
    pass
