# =============================================================================
# streaming/sink.py — Sink de streaming: Kafka → Delta Lake (capa bronze).
#
# Cierra el loop del pipeline de streaming. En vez de imprimir los mensajes,
# el consumer los entrega a este sink, que los persiste en las MISMAS tablas
# bronze que alimenta el batch. Así bronze recibe de dos fuentes (API batch +
# stream Kafka) y silver/gold no necesitan enterarse del origen.
#
# Estrategia de escritura:
#   - Micro-batches: acumula mensajes y escribe de a bloques (no mensaje a
#     mensaje, que generaría una versión Delta por cada registro).
#   - MERGE (upsert) por id: idempotente. Reprocesar el mismo mensaje no
#     duplica filas; la última versión de un id gana (coherente con la
#     deduplicación que hace silver).
# =============================================================================

import logging

import pandas as pd

from bronze.mappers import tipar_equipos, tipar_partidos
from bronze.schemas import inicializar_tabla_equipos, inicializar_tabla_partidos
from utils.delta import merge_en_delta

logger = logging.getLogger(__name__)


class BronzeSink:
    """Acumula mensajes de Kafka por tópico y los escribe a bronze en micro-batches."""

    def __init__(
        self,
        ruta_equipos: str,
        ruta_partidos: str,
        topico_equipos: str,
        topico_partidos: str,
        tam_microbatch: int = 10,
    ) -> None:
        self.ruta_equipos    = ruta_equipos
        self.ruta_partidos   = ruta_partidos
        self.topico_equipos  = topico_equipos
        self.topico_partidos = topico_partidos
        self.tam_microbatch  = tam_microbatch

        self._buffers: dict[str, list[dict]] = {topico_equipos: [], topico_partidos: []}
        self._escritos: dict[str, int] = {topico_equipos: 0, topico_partidos: 0}

        # Garantiza que las tablas existan con schema y constraints.
        inicializar_tabla_equipos(ruta_equipos)
        inicializar_tabla_partidos(ruta_partidos)

    def agregar(self, topico: str, datos: dict) -> None:
        """Encola un mensaje; si el buffer del tópico se llena, hace flush."""
        if topico not in self._buffers:
            logger.warning("[SINK] Tópico desconocido, mensaje ignorado: '%s'", topico)
            return
        self._buffers[topico].append(datos)
        if len(self._buffers[topico]) >= self.tam_microbatch:
            self._flush(topico)

    def cerrar(self) -> None:
        """Vacía todos los buffers pendientes. Llamar al terminar el consumo."""
        for topico in self._buffers:
            self._flush(topico)
        logger.info(
            "[SINK] Cerrado. Persistidos → equipos: %d | partidos: %d",
            self._escritos[self.topico_equipos], self._escritos[self.topico_partidos],
        )

    def _flush(self, topico: str) -> None:
        """Escribe el micro-batch acumulado de un tópico a bronze vía MERGE."""
        registros = self._buffers[topico]
        if not registros:
            return

        df = pd.DataFrame(registros)
        if topico == self.topico_equipos:
            df = tipar_equipos(df).drop_duplicates(subset=["id_equipo"], keep="last")
            merge_en_delta(
                df, self.ruta_equipos,
                predicado_merge="src.id_equipo = tgt.id_equipo",
            )
        elif topico == self.topico_partidos:
            df = tipar_partidos(df).drop_duplicates(subset=["id_evento"], keep="last")
            merge_en_delta(
                df, self.ruta_partidos,
                predicado_merge="src.id_evento = tgt.id_evento",
                columnas_particion=["fecha_extraccion"],
            )

        self._escritos[topico] += len(df)
        self._buffers[topico] = []
        logger.info("[SINK] Micro-batch persistido → '%s' | %d registros", topico, len(df))
