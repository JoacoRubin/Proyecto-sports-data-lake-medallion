# =============================================================================
# bronze/extractors_footballdata.py — Extracción desde football-data.co.uk
# =============================================================================
#
# Fuente de CSVs históricos, sin API key y sin registro. Una URL por
# (división, temporada):
#
#     https://www.football-data.co.uk/mmz4281/{temporada}/{division}.csv
#
# El descargador se recibe por parámetro para poder testear la extracción sin
# tocar la red. La firma por defecto usa el cliente HTTP real.
# =============================================================================

import io
import logging
from datetime import datetime, timezone
from typing import Callable

import pandas as pd
import requests

from bronze.mappers_footballdata import (
    LIGAS_FOOTBALLDATA,
    mapear_partido_footballdata,
    tipar_partidos_footballdata,
)

logger = logging.getLogger(__name__)

URL_BASE_FOOTBALLDATA = "https://www.football-data.co.uk/mmz4281"

# Sin estas columnas el CSV no describe un partido: si faltan, football-data
# cambió el formato y queremos enterarnos ahora, no en gold.
_COLUMNAS_OBLIGATORIAS = ("Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG")


def url_csv(div: str, codigo_temporada: str) -> str:
    """Arma la URL del CSV de una división y temporada."""
    return f"{URL_BASE_FOOTBALLDATA}/{codigo_temporada}/{div}.csv"


def _descargar_csv(url: str) -> str:
    """Descarga el CSV como texto, con reintentos implícitos de requests.

    football-data publica en cp1252 (nombres de árbitros con acentos), pero
    algunos archivos vienen en UTF-8 con BOM. Se intenta UTF-8 y se cae a
    latin-1, que nunca falla al decodificar.
    """
    respuesta = requests.get(url, timeout=60)
    respuesta.raise_for_status()
    crudo = respuesta.content
    try:
        return crudo.decode("utf-8-sig")
    except UnicodeDecodeError:
        return crudo.decode("latin-1")


def _fecha_extraccion_actual() -> tuple[str, str]:
    """Retorna (fecha_YYYY-MM-DD, timestamp_ISO) en UTC."""
    ahora = datetime.now(timezone.utc)
    return ahora.strftime("%Y-%m-%d"), ahora.isoformat()


def _leer_csv(texto: str) -> pd.DataFrame:
    """Parsea el CSV a DataFrame de strings crudos.

    dtype=str preserva la fidelidad de bronze: el tipado se aplica después,
    en el mapper, en un solo lugar.
    """
    df = pd.read_csv(io.StringIO(texto), dtype=str, encoding_errors="replace")
    df.columns = [str(c).replace("﻿", "").strip() for c in df.columns]
    return df


def _descartar_filas_basura(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina las filas vacías o parciales con las que terminan estos CSVs.

    Una fila sin equipos o sin fecha no es un partido: es ruido del archivo.
    """
    return df.dropna(subset=["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)


def extraer_partidos_footballdata(
    div: str,
    codigo_temporada: str,
    descargador: Callable[[str], str] = _descargar_csv,
) -> pd.DataFrame:
    """Extrae los partidos de una división y temporada de football-data.co.uk.

    Args:
        div: código de división ('E0', 'SP1', ...). Debe estar en LIGAS_FOOTBALLDATA.
        codigo_temporada: código de la URL ('2425' = temporada 2024-2025).
        descargador: función que baja el CSV. Inyectable para tests.

    Returns:
        DataFrame con el esquema bronze canónico + estadísticas de partido.

    Raises:
        ValueError: división desconocida, CSV sin datos o sin columnas obligatorias.
    """
    if div not in LIGAS_FOOTBALLDATA:
        raise ValueError(
            f"División '{div}' desconocida. Válidas: {sorted(LIGAS_FOOTBALLDATA)}"
        )

    logger.info(
        "Extrayendo football-data — %s (%s) | temporada %s",
        LIGAS_FOOTBALLDATA[div], div, codigo_temporada,
    )

    df_crudo = _leer_csv(descargador(url_csv(div, codigo_temporada)))

    faltantes = [c for c in _COLUMNAS_OBLIGATORIAS if c not in df_crudo.columns]
    if faltantes:
        raise ValueError(
            f"Columnas obligatorias ausentes en el CSV de {div}/{codigo_temporada}: "
            f"{faltantes}. ¿Cambió el formato de football-data.co.uk?"
        )

    df_crudo = _descartar_filas_basura(df_crudo)
    if df_crudo.empty:
        raise ValueError(f"CSV de {div}/{codigo_temporada} sin datos utilizables.")

    fecha_extraccion, timestamp = _fecha_extraccion_actual()

    registros = [
        mapear_partido_footballdata(fila, div, codigo_temporada, timestamp, fecha_extraccion)
        for fila in df_crudo.to_dict("records")
    ]

    df = pd.DataFrame(registros).drop_duplicates(subset=["id_evento"], keep="first")
    df = tipar_partidos_footballdata(df).reset_index(drop=True)

    logger.info("  > %d partidos extraídos.", len(df))
    return df
