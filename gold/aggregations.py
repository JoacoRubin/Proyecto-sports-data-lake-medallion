# =============================================================================
# gold/aggregations.py — Agregaciones T7: tabla de posiciones por equipo
# =============================================================================
#
# Lee los datos de la capa silver (partidos procesados) y calcula estadísticas
# por equipo (GROUP BY): PJ, PG, PE, PP, GF, GC, DG, Pts.
# El resultado se guarda en la capa gold.
#
# La capa gold contiene datos listos para consumo analítico y reportes.
# =============================================================================

import pandas as pd


def _stats_por_perspectiva(
    df: pd.DataFrame,
    col_equipo: str,
    col_gf: str,
    col_gc: str,
    victoria: str,
) -> pd.DataFrame:
    """Calcula PJ, GF, GC, PG y PE desde la perspectiva de un rol (local o visitante)."""
    return (
        df.groupby(["liga", col_equipo])
        .agg(
            PJ=(col_equipo,  "count"),
            GF=(col_gf,      "sum"),
            GC=(col_gc,      "sum"),
            PG=("resultado", lambda x: (x == victoria).sum()),
            PE=("resultado", lambda x: (x == "Empate").sum()),
        )
        .rename_axis(["liga", "equipo"])
    )


def calcular_tabla_posiciones(df_silver: pd.DataFrame) -> pd.DataFrame:
    """T7 — Agregaciones por equipo (GROUP BY).

    Combina estadísticas de local y visitante para obtener la tabla de
    posiciones: PJ, PG, PE, PP, GF, GC, DG (diferencia de goles), Pts.

    Args:
        df_silver: DataFrame de partidos procesados (capa silver).

    Returns:
        DataFrame con la tabla de posiciones ordenada por puntos.
    """
    stats_local = _stats_por_perspectiva(
        df_silver, "equipo_local", "goles_local", "goles_visitante", "Local"
    )
    stats_visitante = _stats_por_perspectiva(
        df_silver, "equipo_visitante", "goles_visitante", "goles_local", "Visitante"
    )

    stats = pd.concat([stats_local, stats_visitante]).groupby(["liga", "equipo"]).sum()
    stats["PP"]  = stats["PJ"] - stats["PG"] - stats["PE"]
    stats["DG"]  = stats["GF"] - stats["GC"]
    stats["Pts"] = stats["PG"] * 3 + stats["PE"]

    return (
        stats[["PJ", "PG", "PE", "PP", "GF", "GC", "DG", "Pts"]]
        .sort_values(["liga", "Pts"], ascending=[True, False])
        .reset_index()
        .astype({
            "PJ": "int64", "PG": "int64", "PE": "int64", "PP": "int64",
            "GF": "int64", "GC": "int64", "DG": "int64", "Pts": "int64",
        })
    )
