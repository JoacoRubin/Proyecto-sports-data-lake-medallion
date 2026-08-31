# =============================================================================
# ml/features_goles.py — Features para predecir OVER 2.5 GOLES.
#
# POR QUE UN SEGUNDO MODELO
# -------------------------
# El de expulsiones es honesto pero esta contra el techo del fenomeno: las
# rojas ocurren 0,20 veces por partido y el 83% de los partidos no tiene
# ninguna. Los goles ocurren 2,78 veces por partido y el target queda
# balanceado en 52,8%. Hay muchisima mas senal para extraer.
#
# Y sobre todo: existe un BASELINE ADVERSARIO.
#
# EL MERCADO COMO VARA
# --------------------
# Hasta ahora nos mediamos contra "predecir la prevalencia": una vara pasiva,
# que no se defiende. El mercado de apuestas es otra cosa — es el consenso de
# gente que se juega plata y lo ajusta en tiempo real. En el dataset asigna al
# over 2.5 una probabilidad media de 0.5225 contra una tasa real de 0.5280.
# Esta bien calibrado y es dificil de batir.
#
# Por eso el mercado NO entra como feature por defecto. Si entrara, el modelo
# aprenderia a copiarlo y no se podria afirmar nada. El diseno experimental es:
#
#   A. Mercado solo              -> la vara
#   B. Nuestras features         -> le ganamos al mercado con estadisticas?
#   C. Nuestras features + A     -> aportamos algo que el mercado NO ve?
#
# C vs A es la pregunta que importa: si C le gana a A, el modelo contiene
# informacion que el mercado no esta incorporando.
# =============================================================================

import numpy as np
import pandas as pd

from ml.leakage import COLUMNAS_POST_PARTIDO
from ml.features import (
    features_de_cuotas,
    medias_moviles_por_equipo,
    tasa_historica_por_equipo,
    tasa_historica_suavizada,
)

TARGET_GOLES = "over_2_5"

UMBRAL_GOLES = 2.5

# Para goles importan ataque y defensa, no la agresividad.
#
# Cada tupla es (columna_para_el_local, columna_para_el_visitante). Fijate en
# `goles_contra`: los goles en contra del local son los goles DEL VISITANTE,
# asi que la tupla va invertida. Es el bug clasico de este feature engineering
# y por eso tiene su propio test.
_ESTADISTICAS_GOLES = {
    "goles_favor":  ("goles_local",      "goles_visitante"),
    "goles_contra": ("goles_visitante",  "goles_local"),
    "tiros":        ("tiros_local",      "tiros_visitante"),
    "tiros_arco":   ("tiros_arco_local", "tiros_arco_visitante"),
    "corners":      ("corners_local",    "corners_visitante"),
}

_COLUMNAS_A_DESCARTAR = [
    *COLUMNAS_POST_PARTIDO,
    "arbitro",
    # Cuotas crudas: entran normalizadas o no entran.
    "cuota_local", "cuota_empate", "cuota_visitante",
    "cuota_over25", "cuota_under25",
    "cuota_local_mercado", "cuota_empate_mercado", "cuota_visitante_mercado",
]


def agregar_target_goles(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega el target binario: hubo mas de 2.5 goles en el partido?

    2.5 no es arbitrario: es la linea estandar del mercado de apuestas, lo que
    permite comparar contra la probabilidad implicita de las cuotas. Y al ser
    un umbral con decimal, no existe el empate.
    """
    df = df.copy()
    df["goles_totales"] = (
        pd.to_numeric(df["goles_local"], errors="coerce")
        + pd.to_numeric(df["goles_visitante"], errors="coerce")
    )
    df[TARGET_GOLES] = (df["goles_totales"] > UMBRAL_GOLES).astype(int)
    return df


def probabilidad_mercado_over25(df: pd.DataFrame) -> pd.Series:
    """Probabilidad implicita del mercado, sin el margen de la casa.

    1/cuota es la probabilidad implicita, pero over y under suman mas de 1: ese
    exceso es el overround (5,31% promedio en este dataset). Normalizar lo
    elimina y deja una probabilidad comparable con la que emite el modelo.
    """
    inv_over  = 1.0 / pd.to_numeric(df["cuota_over25"],  errors="coerce")
    inv_under = 1.0 / pd.to_numeric(df["cuota_under25"], errors="coerce")
    return inv_over / (inv_over + inv_under)


def construir_features_goles(
    df: pd.DataFrame,
    prior: float,
    ventanas: tuple[int, ...] = (5, 10),
    k_suavizado: float = 20.0,
    incluir_mercado: bool = False,
) -> pd.DataFrame:
    """Construye la matriz de features para predecir over 2.5 goles.

    Args:
        df: partidos de bronze con estadisticas y cuotas.
        prior: tasa base de over 2.5. DEBE venir del split de entrenamiento.
        ventanas: tamanos de las medias moviles por equipo.
        k_suavizado: fuerza del suavizado de las tasas historicas.
        incluir_mercado: agrega `prob_mercado_over25` como feature. Por defecto
            False — con el mercado adentro el modelo lo copia y deja de poder
            compararse contra el. Se activa solo para el experimento C.

    Returns:
        Una fila por partido: features + TARGET_GOLES, sin ninguna columna que
        describa lo ocurrido en el propio partido.
    """
    df = agregar_target_goles(df).copy()
    df["_fecha"] = pd.to_datetime(df["fecha_partido"], errors="coerce")
    df = df.sort_values(["_fecha"], kind="stable").reset_index(drop=True)
    df["_i"] = np.arange(len(df))

    # --- Tasa historica de over 2.5 de cada equipo ---
    tasa_local, tasa_visitante = tasa_historica_por_equipo(
        df, prior=prior, k=k_suavizado, columna_evento=TARGET_GOLES
    )
    df["equipo_tasa_over_local"]     = tasa_local
    df["equipo_tasa_over_visitante"] = tasa_visitante

    # --- Historia de ataque y defensa ---
    df = medias_moviles_por_equipo(df, ventanas=ventanas, estadisticas=_ESTADISTICAS_GOLES)

    # --- Contexto 1X2, con el CONSENSO del mercado, no una sola casa ---
    # El promedio de todas las casas es mejor estimador que Bet365 en solitario.
    df = features_de_cuotas(
        df, columnas=("cuota_local_mercado", "cuota_empate_mercado", "cuota_visitante_mercado")
    )

    # --- Historial del cruce ---
    a = df["equipo_local"].astype(str)
    b = df["equipo_visitante"].astype(str)
    df["_par"] = np.where(a < b, a + "|" + b, b + "|" + a)
    df["h2h_tasa_over"] = tasa_historica_suavizada(
        df, "_par", TARGET_GOLES, k=k_suavizado, prior=prior
    )

    # --- Temporales ---
    df["mes"] = df["_fecha"].dt.month.astype("Int64")
    df["fecha_num"] = (
        df.groupby("temporada", dropna=False)["_fecha"].rank(method="dense").astype("Int64")
    )

    # --- El mercado: baseline por defecto, feature solo si se pide ---
    if incluir_mercado:
        df["prob_mercado_over25"] = probabilidad_mercado_over25(df)

    descartar = [c for c in _COLUMNAS_A_DESCARTAR if c in df.columns]
    descartar += [c for c in df.columns if c.startswith("_")]
    return df.drop(columns=descartar)
