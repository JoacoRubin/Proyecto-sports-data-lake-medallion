# =============================================================================
# bronze/loaders.py — Persistencia de datos en la capa bronze
# =============================================================================

import logging

import pandas as pd

from utils.delta import guardar_en_delta

logger = logging.getLogger(__name__)


def guardar_equipos_bronze(df: pd.DataFrame, ruta: str) -> None:
    """Persiste equipos en bronze — estrategia FULL (overwrite total).

    Los metadatos de equipos son datos estáticos; se sobreescribe la tabla
    completa para garantizar que siempre refleje el estado actual.
    """
    guardar_en_delta(df, ruta, modo="overwrite", modo_esquema="merge")
    logger.info(
        "[3/4] Equipos en '%s' | %d registros | Estrategia: FULL",
        ruta, len(df),
    )


def guardar_partidos_bronze(df: pd.DataFrame, ruta: str) -> None:
    """Persiste partidos en bronze — estrategia INCREMENTAL (INSERT-OVERWRITE por fecha).

    Particiona por fecha_extraccion y sobreescribe solo la partición del día
    actual, preservando el historial sin duplicar datos (idempotente).
    """
    fecha     = df["fecha_extraccion"].iloc[0]
    predicado = f"fecha_extraccion = '{fecha}'"
    guardar_en_delta(
        df, ruta,
        modo="overwrite",
        columnas_particion=["fecha_extraccion"],
        predicado=predicado,
        modo_esquema="merge",
    )
    logger.info(
        "[4/4] Partidos en '%s' | Particion: %s | %d registros | Estrategia: INCREMENTAL",
        ruta, fecha, len(df),
    )


def guardar_partidos_footballdata_bronze(df: pd.DataFrame, ruta: str) -> None:
    """Persiste partidos de football-data.co.uk — INSERT-OVERWRITE por liga-temporada.

    Sobreescribe únicamente la partición (id_liga, temporada) que se está
    cargando. Como los id_evento son deterministas, re-ingerir la misma
    temporada es idempotente: reemplaza los mismos registros, no los duplica.
    """
    id_liga   = df["id_liga"].iloc[0]
    temporada = df["temporada"].iloc[0]
    predicado = f"id_liga = '{id_liga}' AND temporada = '{temporada}'"
    guardar_en_delta(
        df, ruta,
        modo="overwrite",
        columnas_particion=["id_liga", "temporada"],
        predicado=predicado,
        modo_esquema="merge",
    )
    logger.info(
        "football-data en '%s' | Particion: %s / %s | %d registros",
        ruta, id_liga, temporada, len(df),
    )
