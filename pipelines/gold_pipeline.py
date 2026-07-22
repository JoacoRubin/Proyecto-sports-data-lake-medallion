# =============================================================================
# pipelines/gold_pipeline.py — Pipeline TP2: agregaciones silver → gold
# =============================================================================
#
# Lee los partidos procesados desde silver, aplica la transformación T7
# (GROUP BY por equipo) y guarda la tabla de posiciones en la capa gold.
# =============================================================================

import logging

from config import DIR_PARTIDOS_SILVER, DIR_ESTADISTICAS_GOLD
from gold.aggregations import calcular_tabla_posiciones
from quality.contracts import validar_gold_posiciones
from utils.delta import leer_tabla_delta, merge_en_delta, mostrar_resumen_tabla

logger = logging.getLogger(__name__)


def ejecutar() -> None:
    """Lee silver, calcula la tabla de posiciones (T7) y guarda en gold."""
    logger.info("=" * 62)
    logger.info("  TP2 - Agregaciones (silver -> gold)")
    logger.info("=" * 62)

    logger.info("[1/3] Leyendo partidos desde silver...")
    df_silver = leer_tabla_delta(DIR_PARTIDOS_SILVER)
    logger.info("      Partidos leídos: %d", len(df_silver))

    logger.info("[2/3] Calculando tabla de posiciones (T7 - GROUP BY)...")
    df_estadisticas = calcular_tabla_posiciones(df_silver)
    logger.info("      Equipos con estadísticas: %d", len(df_estadisticas))

    logger.info("[2.5/3] Validando contrato de calidad de gold (invariantes)...")
    validar_gold_posiciones(df_estadisticas)
    logger.info("      Contrato gold OK.")

    logger.info("[3/3] Guardando en gold (MERGE por liga+id_equipo, partición por liga)...")
    merge_en_delta(
        df_estadisticas,
        DIR_ESTADISTICAS_GOLD,
        predicado_merge="src.id_equipo = tgt.id_equipo AND src.liga = tgt.liga",
        columnas_particion=["liga"],
    )
    logger.info("      Gold guardado correctamente.")

    logger.info("--- Verificacion gold ---")
    mostrar_resumen_tabla(DIR_ESTADISTICAS_GOLD, "gold/estadisticas_equipos")
