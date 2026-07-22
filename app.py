# =============================================================================
# app.py — Dashboard Streamlit sobre el data lake (capas gold y silver).
#
# Lee las tablas Delta ya procesadas por el pipeline y las presenta de forma
# interactiva: líder, tabla de posiciones, métricas, bubble chart y goleadas.
#
# Ejecutar:  streamlit run app.py
# (requiere haber corrido antes el pipeline: python JoaquinRubinstein_TP2.py)
# =============================================================================

import pandas as pd
import plotly.express as px
import streamlit as st

from config import DIR_ESTADISTICAS_GOLD, DIR_PARTIDOS_SILVER
from utils.delta import leer_tabla_delta, tabla_delta_existe

st.set_page_config(page_title="TheSportsDB — Data Lake", page_icon="⚽", layout="wide")


@st.cache_data(show_spinner=False)
def cargar_gold() -> pd.DataFrame:
    """Tabla de posiciones (capa gold). Cacheada para no releer Delta en cada rerun."""
    if not tabla_delta_existe(DIR_ESTADISTICAS_GOLD):
        return pd.DataFrame()
    return leer_tabla_delta(DIR_ESTADISTICAS_GOLD)


@st.cache_data(show_spinner=False)
def cargar_silver() -> pd.DataFrame:
    """Partidos procesados (capa silver)."""
    if not tabla_delta_existe(DIR_PARTIDOS_SILVER):
        return pd.DataFrame()
    return leer_tabla_delta(DIR_PARTIDOS_SILVER)


def _medalla(pos: int) -> str:
    """Emoji de podio para el top 3; el número para el resto."""
    return {1: "🥇", 2: "🥈", 3: "🥉"}.get(pos, str(pos))


@st.cache_resource(show_spinner="Primer arranque: generando el data lake desde la API (unos segundos)...")
def asegurar_datos_generados() -> bool:
    """Genera el data lake si está vacío (ej. primer arranque en la nube).

    Corre el pipeline batch directo (sin Prefect, para no depender de la
    orquestación en el deploy). cache_resource garantiza una sola ejecución
    por proceso, aunque varios usuarios abran el dashboard a la vez.
    """
    if not tabla_delta_existe(DIR_ESTADISTICAS_GOLD):
        from pipelines import bronze_pipeline, silver_pipeline, gold_pipeline
        bronze_pipeline.ejecutar()
        silver_pipeline.ejecutar()
        gold_pipeline.ejecutar()
    return True


st.title("⚽ TheSportsDB — Data Lake")
st.caption("Arquitectura medallion: bronze → silver → gold. Dashboard sobre las capas procesadas.")

asegurar_datos_generados()
gold = cargar_gold()
silver = cargar_silver()

if gold.empty:
    st.error(
        "No se pudieron generar los datos. Verificá la conexión con la API de "
        "TheSportsDB y volvé a intentar."
    )
    st.stop()

# --- Filtro por liga -------------------------------------------------------
ligas = sorted(gold["liga"].dropna().unique())
liga_sel = st.sidebar.selectbox("Liga", ligas)
st.sidebar.caption(f"{len(ligas)} ligas en el data lake")

tabla = gold[gold["liga"] == liga_sel].copy()
partidos = silver[silver["liga"] == liga_sel].copy() if not silver.empty else pd.DataFrame()

posiciones = tabla.sort_values(["Pts", "DG"], ascending=False).reset_index(drop=True)
posiciones.insert(0, "Pos", range(1, len(posiciones) + 1))

# --- Líder destacado -------------------------------------------------------
lider = posiciones.iloc[0]
st.success(
    f"🏆 **Líder de {liga_sel}: {lider['equipo']}** — "
    f"{int(lider['Pts'])} pts · {int(lider['PG'])} PG · DG {int(lider['DG']):+d}"
)

# --- KPIs ------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Equipos", len(tabla))
col2.metric("Partidos", len(partidos))
if not partidos.empty:
    goles = int(partidos["goles_local"].sum() + partidos["goles_visitante"].sum())
    prom = goles / len(partidos) if len(partidos) else 0
    col3.metric("Goles totales", goles, f"{prom:.1f} por partido")
    col4.metric("Goleadas", int(partidos.get("es_goleada", pd.Series(dtype=bool)).sum()))

st.divider()

# --- Tabla de posiciones ---------------------------------------------------
st.subheader(f"Tabla de posiciones — {liga_sel}")
max_pts = max(int(posiciones["Pts"].max()), 1)
tabla_display = posiciones.copy()
tabla_display["Pos"] = tabla_display["Pos"].map(_medalla)
st.dataframe(
    tabla_display[["Pos", "equipo", "PJ", "PG", "PE", "PP", "GF", "GC", "DG", "Pts"]],
    hide_index=True,
    use_container_width=True,
    column_config={
        # Barra nativa (no usa altair): los puntos se ven de un vistazo.
        "Pts": st.column_config.ProgressColumn(
            "Pts", format="%d", min_value=0, max_value=max_pts,
        ),
    },
)

# --- Bubble chart: ataque vs. defensa --------------------------------------
st.subheader("Ataque vs. defensa")
st.caption(
    "Cada burbuja es un equipo. Derecha = más goles a favor; arriba = menos goles "
    "en contra (mejor defensa). Tamaño y color = puntos. Pasá el mouse por una "
    "burbuja para ver el detalle del equipo."
)
bubble = posiciones.copy()
bubble["_size"] = bubble["Pts"] + 1  # +1 para que 0 pts siga siendo visible
fig = px.scatter(
    bubble,
    x="GF", y="GC",
    size="_size", color="Pts",
    hover_name="equipo",
    hover_data={
        "_size": False, "Pts": True, "PJ": True, "PG": True,
        "DG": True, "GF": True, "GC": True,
    },
    color_continuous_scale="Viridis",   # secuencial, perceptualmente uniforme y CVD-safe
    size_max=48,
    labels={"GF": "Goles a favor", "GC": "Goles en contra"},
)
# Anillo sutil en cada burbuja para separarlas cuando se solapan.
fig.update_traces(marker=dict(line=dict(width=1.5, color="rgba(255,255,255,0.35)")))
fig.update_yaxes(autorange="reversed")  # menos goles en contra hacia arriba
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    height=480,
    margin=dict(l=10, r=10, t=10, b=10),
    coloraxis_colorbar=dict(title="Pts"),
)
st.plotly_chart(fig, use_container_width=True)

# --- Resultados ------------------------------------------------------------
st.subheader("Resultados de los partidos")
if not partidos.empty and "resultado" in partidos:
    conteo = partidos["resultado"].value_counts()
    r1, r2, r3 = st.columns(3)
    r1.metric("Victorias local", int(conteo.get("Local", 0)))
    r2.metric("Empates", int(conteo.get("Empate", 0)))
    r3.metric("Victorias visitante", int(conteo.get("Visitante", 0)))
else:
    st.info("Sin partidos procesados para esta liga.")

# --- Goleadas --------------------------------------------------------------
if not partidos.empty and "es_goleada" in partidos:
    st.subheader("Goleadas")
    goleadas = partidos[partidos["es_goleada"]].copy()
    if goleadas.empty:
        st.info("No hubo goleadas registradas en esta liga.")
    else:
        cols = ["fecha_partido", "equipo_local", "goles_local", "goles_visitante", "equipo_visitante"]
        st.dataframe(
            goleadas.sort_values("fecha_partido", ascending=False)[cols],
            hide_index=True,
            use_container_width=True,
        )
