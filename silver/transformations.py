# =============================================================================
# silver/transformations.py — Transformaciones T1–T6 sobre datos bronze
# =============================================================================
#
#   T1 | Deduplicacion       — conserva el registro más reciente de cada id_evento
#   T2 | Manejo de nulos     — elimina filas inútiles, rellena campos opcionales
#   T3 | Conversión de tipos — fecha a datetime64, goles a int64
#   T4 | Columnas derivadas  — resultado, diferencia_goles, es_goleada, partes de fecha
#   T5 | Renombrado          — normaliza nombres ambiguos
#   T6 | JOIN / enriquec.    — agrega ciudad y capacidad del estadio del local
# =============================================================================

import numpy as np
import pandas as pd


def deduplicar_partidos(df: pd.DataFrame) -> pd.DataFrame:
    """T1 — Deduplicación.

    Un mismo partido puede aparecer en múltiples particiones (una por día
    de extracción). Se conserva el registro con la fecha de extracción
    más reciente.
    """
    return (
        df
        .sort_values("fecha_extraccion", ascending=False)
        .drop_duplicates(subset=["id_evento"], keep="first")
        .reset_index(drop=True)
    )


def manejar_nulos_partidos(df: pd.DataFrame) -> pd.DataFrame:
    """T2 — Manejo de nulos en partidos.

    Elimina filas sin ID o sin resultado (no analizables) y filtra solo
    partidos finalizados. Rellena campos opcionales para no descartar
    partidos válidos.
    """
    df = df.dropna(subset=["id_evento", "goles_local", "goles_visitante"])
    df = df[df["estado"].str.contains("Finished", case=False, na=False)].copy()
    df["estadio"]   = df["estadio"].fillna("Estadio desconocido")
    df["temporada"] = df["temporada"].fillna("Sin temporada")
    return df.reset_index(drop=True)


def manejar_nulos_equipos(df: pd.DataFrame) -> pd.DataFrame:
    """T2 — Manejo de nulos en equipos.

    Elimina registros sin ID o nombre, reemplaza nulos informativos.
    """
    df = df.dropna(subset=["id_equipo", "nombre"]).copy()
    df["ciudad"]    = df["ciudad"].fillna("No informado")
    df["estadio"]   = df["estadio"].fillna("No informado")
    df["sitio_web"] = df["sitio_web"].fillna("Sin sitio web")
    return df.reset_index(drop=True)


def convertir_tipos_y_fechas(df: pd.DataFrame) -> pd.DataFrame:
    """T3 — Conversión de tipos y formateo de fechas.

    Convierte fecha_partido a datetime64 para operar sobre ella.
    Convierte goles a int64 estándar (no nullable) para cálculos.
    """
    df = df.copy()
    df["fecha_partido"]   = pd.to_datetime(df["fecha_partido"], errors="coerce")
    df["goles_local"]     = df["goles_local"].astype("int64")
    df["goles_visitante"] = df["goles_visitante"].astype("int64")
    return df


def agregar_columnas_derivadas(df: pd.DataFrame) -> pd.DataFrame:
    """T4 — Nuevas columnas derivadas.

    Evitan recalcular lógica repetida en análisis posteriores:
    - anio_partido, mes_partido, dia_semana: partes de la fecha.
    - resultado: quién ganó (Local / Visitante / Empate).
    - diferencia_goles: valor absoluto de la diferencia de goles.
    - es_goleada: True si la diferencia supera 2 goles.
    """
    df = df.copy()
    df["anio_partido"] = df["fecha_partido"].dt.year.astype("Int64")
    df["mes_partido"]  = df["fecha_partido"].dt.month.astype("Int64")
    df["dia_semana"]   = df["fecha_partido"].dt.day_name()

    conditions = [
        df["goles_local"] > df["goles_visitante"],
        df["goles_local"] < df["goles_visitante"],
    ]
    df["resultado"] = np.select(conditions, ["Local", "Visitante"], default="Empate")
    df["diferencia_goles"] = (df["goles_local"] - df["goles_visitante"]).abs()
    df["es_goleada"]       = df["diferencia_goles"] > 2
    return df


def renombrar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """T5 — Renombrado de columnas ambiguas."""
    return df.rename(columns={
        "nombre_evento": "descripcion_partido",
        "estado":        "estado_partido",
        "estadio":       "estadio_sede",
    })


def enriquecer_con_equipos(df_partidos: pd.DataFrame, df_equipos: pd.DataFrame) -> pd.DataFrame:
    """T6 — JOIN entre partidos y equipos.

    Agrega ciudad y capacidad del estadio del equipo local a cada partido.
    Usa LEFT JOIN para conservar todos los partidos aunque falte info del equipo.
    """
    info_local = (
        df_equipos[["id_equipo", "ciudad", "capacidad_estadio"]]
        .rename(columns={
            "ciudad":            "ciudad_local",
            "capacidad_estadio": "capacidad_estadio_local",
        })
    )
    merged = pd.merge(
        df_partidos, info_local,
        how="left", left_on="id_equipo_local", right_on="id_equipo",
    ).drop(columns=["id_equipo"])
    merged["ciudad_local"] = merged["ciudad_local"].fillna("No informado")
    return merged


def procesar_partidos(df_bronze: pd.DataFrame, df_equipos: pd.DataFrame) -> pd.DataFrame:
    """Aplica el pipeline completo de transformaciones T1–T6 sobre partidos bronze."""
    df = (
        df_bronze
        .pipe(deduplicar_partidos)        # T1
        .pipe(manejar_nulos_partidos)      # T2
        .pipe(convertir_tipos_y_fechas)    # T3
        .pipe(agregar_columnas_derivadas)  # T4
        .pipe(renombrar_columnas)          # T5
    )
    return enriquecer_con_equipos(df, df_equipos)  # T6
