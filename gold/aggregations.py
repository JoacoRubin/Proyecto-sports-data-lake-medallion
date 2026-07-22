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
    col_id: str,
    col_gf: str,
    col_gc: str,
    victoria: str,
) -> pd.DataFrame:
    """Calcula PJ, GF, GC, PG y PE desde la perspectiva de un rol (local o visitante).

    Agrupa por `col_id` (id del equipo) —clave estable— y no por el nombre,
    que puede variar entre registros (ej. 'River' vs 'River Plate').
    """
    return (
        df.groupby(["liga", col_id])
        .agg(
            PJ=(col_id,      "count"),
            GF=(col_gf,      "sum"),
            GC=(col_gc,      "sum"),
            PG=("resultado", lambda x: (x == victoria).sum()),
            PE=("resultado", lambda x: (x == "Empate").sum()),
        )
        .rename_axis(["liga", "id_equipo"])
    )


def _mapa_id_a_nombre(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve un DataFrame id_equipo -> equipo (nombre para mostrar).

    Toma los nombres tanto de la columna local como visitante; ante nombres
    distintos para un mismo id, conserva el primero.
    """
    locales = df[["id_equipo_local", "equipo_local"]].rename(
        columns={"id_equipo_local": "id_equipo", "equipo_local": "equipo"}
    )
    visitantes = df[["id_equipo_visitante", "equipo_visitante"]].rename(
        columns={"id_equipo_visitante": "id_equipo", "equipo_visitante": "equipo"}
    )
    return (
        pd.concat([locales, visitantes], ignore_index=True)
        .dropna(subset=["id_equipo"])
        .drop_duplicates(subset=["id_equipo"], keep="first")
    )


def calcular_tabla_posiciones(df_silver: pd.DataFrame) -> pd.DataFrame:
    """T7 — Agregaciones por equipo (GROUP BY).

    Combina estadísticas de local y visitante para obtener la tabla de
    posiciones: PJ, PG, PE, PP, GF, GC, DG (diferencia de goles), Pts.
    Agrupa por id_equipo (clave estable) y agrega el nombre para mostrar.

    Args:
        df_silver: DataFrame de partidos procesados (capa silver).

    Returns:
        DataFrame con la tabla de posiciones ordenada por puntos.
    """
    stats_local = _stats_por_perspectiva(
        df_silver, "id_equipo_local", "goles_local", "goles_visitante", "Local"
    )
    stats_visitante = _stats_por_perspectiva(
        df_silver, "id_equipo_visitante", "goles_visitante", "goles_local", "Visitante"
    )

    stats = pd.concat([stats_local, stats_visitante]).groupby(["liga", "id_equipo"]).sum()
    stats["PP"]  = stats["PJ"] - stats["PG"] - stats["PE"]
    stats["DG"]  = stats["GF"] - stats["GC"]
    stats["Pts"] = stats["PG"] * 3 + stats["PE"]

    tabla = (
        stats[["PJ", "PG", "PE", "PP", "GF", "GC", "DG", "Pts"]]
        .sort_values(["liga", "Pts"], ascending=[True, False])
        .reset_index()
        .astype({
            "PJ": "int64", "PG": "int64", "PE": "int64", "PP": "int64",
            "GF": "int64", "GC": "int64", "DG": "int64", "Pts": "int64",
        })
    )

    # Nombre para mostrar, sin perder el id como clave de la fila.
    nombres = _mapa_id_a_nombre(df_silver)
    tabla = tabla.merge(nombres, on="id_equipo", how="left")

    columnas = ["liga", "id_equipo", "equipo", "PJ", "PG", "PE", "PP", "GF", "GC", "DG", "Pts"]
    return tabla[columnas]
