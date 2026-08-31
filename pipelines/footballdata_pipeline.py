# =============================================================================
# pipelines/footballdata_pipeline.py — Ingesta bronze desde football-data.co.uk
# =============================================================================
#
# Carga histórica: una pasada por cada combinación (división × temporada).
# Cada CSV es una unidad independiente — si una falla, se registra y se sigue,
# igual que en el pipeline de TheSportsDB.
#
# El contrato bronze se valida ANTES de escribir: datos que no cumplen no
# entran al data lake. Esa es la diferencia entre un data lake y un pantano.
# =============================================================================

import logging

from config import (
    DIR_PARTIDOS_FOOTBALLDATA_BRONZE,
    FOOTBALLDATA_DIVISIONES,
    FOOTBALLDATA_TEMPORADAS,
)
from bronze.extractors_footballdata import extraer_partidos_footballdata
from bronze.loaders import guardar_partidos_footballdata_bronze
from bronze.schemas import inicializar_tabla_partidos_footballdata
from quality.contracts import validar_bronze_partidos
from utils.delta import mostrar_resumen_tabla

logger = logging.getLogger(__name__)


def ingerir_temporada(div: str, codigo_temporada: str, ruta: str) -> int:
    """Extrae, valida y persiste una liga-temporada. Devuelve la cantidad de filas."""
    df = extraer_partidos_footballdata(div, codigo_temporada)
    validar_bronze_partidos(df)
    guardar_partidos_footballdata_bronze(df, ruta)
    return len(df)


def ejecutar(
    divisiones: list[str] | None = None,
    temporadas: list[str] | None = None,
) -> None:
    """Ingesta histórica de football-data.co.uk hacia bronze."""
    divisiones = divisiones or FOOTBALLDATA_DIVISIONES
    temporadas = temporadas or FOOTBALLDATA_TEMPORADAS
    ruta = DIR_PARTIDOS_FOOTBALLDATA_BRONZE

    logger.info("=" * 62)
    logger.info("  Ingesta bronze - football-data.co.uk")
    logger.info("  %d divisiones x %d temporadas", len(divisiones), len(temporadas))
    logger.info("=" * 62)

    inicializar_tabla_partidos_footballdata(ruta)

    total, fallidas = 0, []
    for div in divisiones:
        for temporada in temporadas:
            try:
                total += ingerir_temporada(div, temporada, ruta)
            except Exception as e:  # noqa: BLE001 — un CSV caído no frena la carga
                logger.warning("Omitida %s/%s: %s", div, temporada, e)
                fallidas.append(f"{div}/{temporada}")

    if total == 0:
        raise RuntimeError("Ninguna liga-temporada devolvió datos. Abortando.")

    logger.info("Ingesta completa: %d partidos.", total)
    if fallidas:
        logger.warning("Combinaciones omitidas (%d): %s", len(fallidas), fallidas)

    mostrar_resumen_tabla(ruta, "bronze/footballdata (HISTORICO)")


if __name__ == "__main__":
    ejecutar()
