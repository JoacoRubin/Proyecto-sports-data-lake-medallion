# =============================================================================
# pipelines/silver_pipeline.py — Pipeline TP2: procesamiento bronze → silver
# =============================================================================

import logging

from config import (
    DIR_EQUIPOS_BRONZE, DIR_PARTIDOS_BRONZE,
    DIR_PARTIDOS_FOOTBALLDATA_BRONZE, DIR_PARTIDOS_SILVER,
)
from quality.contracts import (
    validar_silver_partidos, verificar_silver_no_vacio_si_habia_finalizados,
)
import pandas as pd

from silver.transformations import (
    manejar_nulos_equipos, procesar_partidos, procesar_partidos_footballdata,
)
from utils.delta import tabla_delta_existe
from utils.delta import leer_tabla_delta, merge_en_delta, mostrar_resumen_tabla

logger = logging.getLogger(__name__)


def ejecutar() -> None:
    """TP2 — Lee bronze, aplica transformaciones T1–T6 y guarda en silver."""
    logger.info("=" * 62)
    logger.info("  TP2 - Procesamiento de Datos (bronze -> silver)")
    logger.info("=" * 62)

    # En un data lake multi-fuente ninguna fuente es obligatoria: se procesa la
    # que este ingerida. Exigir las dos hacia que un clon fresco que solo corrio
    # la ingesta historica no pudiera construir silver.
    logger.info("[1/3] Leyendo datos bronze...")
    partes: list[pd.DataFrame] = []
    df_partidos_bronze = pd.DataFrame()

    if tabla_delta_existe(DIR_PARTIDOS_BRONZE) and tabla_delta_existe(DIR_EQUIPOS_BRONZE):
        df_equipos_bronze  = leer_tabla_delta(DIR_EQUIPOS_BRONZE)
        df_partidos_bronze = leer_tabla_delta(DIR_PARTIDOS_BRONZE)
        logger.info(
            "      TheSportsDB — Equipos: %d | Partidos: %d",
            len(df_equipos_bronze), len(df_partidos_bronze),
        )
        logger.info("[2/3] Aplicando transformaciones T1-T6...")
        df_equipos_silver = manejar_nulos_equipos(df_equipos_bronze)
        partes.append(procesar_partidos(df_partidos_bronze, df_equipos_silver))
        logger.info("      Partidos procesados: %d", len(partes[-1]))
    else:
        logger.info("      TheSportsDB no ingerido todavia; se omite.")

    # --- Segunda fuente: football-data.co.uk ---
    #
    # Se procesa aparte y se concatena. NO se hace matching de nombres entre
    # fuentes: `fuente` es una dimension mas y gold agrupa por ella. Ver
    # silver/transformations.py.
    if tabla_delta_existe(DIR_PARTIDOS_FOOTBALLDATA_BRONZE):
        df_fd_bronze = leer_tabla_delta(DIR_PARTIDOS_FOOTBALLDATA_BRONZE)
        partes.append(procesar_partidos_footballdata(df_fd_bronze))
        logger.info(
            "      football-data: %d filas bronze -> %d procesadas",
            len(df_fd_bronze), len(partes[-1]),
        )
    else:
        logger.info("      football-data no ingerido todavia; se omite.")

    if not partes:
        raise RuntimeError(
            "Ninguna fuente bronze esta ingerida. Corre bronze_pipeline o "
            "footballdata_pipeline antes de silver."
        )

    df_partidos_silver = pd.concat(partes, ignore_index=True)
    logger.info("      Total silver (%d fuente/s): %d", len(partes), len(df_partidos_silver))

    logger.info("[2.5/3] Validando contrato de calidad de silver...")
    if not df_partidos_bronze.empty:
        verificar_silver_no_vacio_si_habia_finalizados(df_partidos_bronze, df_partidos_silver)
    validar_silver_partidos(df_partidos_silver)
    logger.info("      Contrato silver OK.")

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
