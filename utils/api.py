# =============================================================================
# utils/api.py — Cliente HTTP genérico con reintentos y backoff exponencial
# =============================================================================

import logging
import time

import requests

logger = logging.getLogger(__name__)


def obtener_datos_api(
    url_base: str,
    endpoint: str,
    parametros: dict = None,
    campo_datos: str = None,
    max_reintentos: int = 3,
) -> dict | list:
    """Realiza una solicitud GET a una API REST con reintentos y backoff exponencial.

    Args:
        url_base: URL base de la API.
        endpoint: ruta relativa (ej. '/search_all_teams.php').
        parametros: query params como diccionario.
        campo_datos: clave del JSON donde están los datos.
        max_reintentos: intentos máximos ante fallos transitorios (1s → 2s → 4s).

    Returns:
        JSON deserializado (dict o list).

    Raises:
        requests.RequestException: si todos los intentos fallan.
    """
    url = f"{url_base}{endpoint}"
    ultimo_error: Exception | None = None

    for intento in range(1, max_reintentos + 1):
        try:
            respuesta = requests.get(url, params=parametros or {}, timeout=30)
            respuesta.raise_for_status()
            datos = respuesta.json()
            return datos.get(campo_datos) if campo_datos else datos
        except requests.RequestException as error:
            ultimo_error = error
            if intento < max_reintentos:
                espera = 2 ** (intento - 1)
                logger.warning(
                    "Intento %d/%d fallido. Reintentando en %ds...",
                    intento, max_reintentos, espera,
                )
                time.sleep(espera)

    raise ultimo_error  # type: ignore[misc]
