# =============================================================================
# pipelines/silver_pipeline.py — Pipeline TP2: procesamiento bronze → silver
# =============================================================================

import logging

from config import (
    DIR_EQUIPOS_BRONZE, DIR_PARTIDOS_BRONZE,
    DIR_PARTIDOS_SILVER,
)
from silver.transformations import manejar_nulos_equipos, procesar_partidos
from utils.delta import leer_tabla_delta, merge_en_delta, mostrar_resumen_tabla

logger = logging.getLogger(__name__)


def ejecutar() -> None:
    """TP2 — Lee bronze, aplica transformaciones T1–T6 y guarda en silver."""
    logger.info("=" * 62)
    logger.info("  TP2 - Procesamiento de Datos (bronze -> silver)")
    logger.info("=" * 62)

    logger.info("[1/3] Leyendo datos bronze...")
    df_equipos_bronze  = leer_tabla_delta(DIR_EQUIPOS_BRONZE)
    df_partidos_bronze = leer_tabla_delta(DIR_PARTIDOS_BRONZE)
    logger.info(
        "      Equipos: %d filas | Partidos: %d filas",
        len(df_equipos_bronze), len(df_partidos_bronze),
    )

    logger.info("[2/3] Aplicando transformaciones T1-T6...")
    df_equipos_silver  = manejar_nulos_equipos(df_equipos_bronze)
    df_partidos_silver = procesar_partidos(df_partidos_bronze, df_equipos_silver)
    logger.info("      Partidos procesados: %d", len(df_partidos_silver))

    logger.info("[3/3] Guardando en silver (MERGE por id_evento, partición por temporada)...")
    merge_en_delta(
        df_partidos_silver,
        DIR_PARTIDOS_SILVER,
        predicado_merge="src.id_evento = tgt.id_evento",
        columnas_particion=["temporada"],
    )
    logger.info("      Silver guardado correctamente.")

    logger.info("--- Verificacion silver ---")
    mostrar_resumen_tabla(DIR_PARTIDOS_SILVER, "silver/partidos_procesados")
