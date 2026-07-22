# =============================================================================
# EXTRA — Streaming con Apache Kafka + TheSportsDB
# Alumno: Joaquin Rubinstein
# =============================================================================
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │                         PAPER EXPLICATIVO                               │
# └─────────────────────────────────────────────────────────────────────────┘
#
# CONTEXTO
# --------
# El TP principal cubre un pipeline batch: extracción desde una API,
# almacenamiento en Delta Lake (capa bronze) y transformaciones hacia
# la capa silver. Todo el procesamiento ocurre en un momento puntual
# y sobre datos históricos.
#
# Este módulo extiende el trabajo incorporando procesamiento en tiempo
# real mediante Apache Kafka, un sistema de mensajería distribuida
# diseñado para flujos de datos de alta disponibilidad.
#
# ¿QUÉ ES APACHE KAFKA?
# ----------------------
# Kafka es una plataforma de streaming distribuida que funciona como
# un bus de mensajes tolerante a fallos. Sus conceptos clave son:
#
#   - Tópico (Topic): canal con nombre donde se publican mensajes.
#     Similar a una tabla, pero ordenada en el tiempo.
#
#   - Producer: proceso que escribe mensajes en un tópico.
#
#   - Consumer: proceso que lee mensajes de un tópico.
#
#   - Broker: servidor que almacena y distribuye los mensajes.
#
# En este caso el broker es gestionado por Aiven (servicio cloud),
# lo que elimina la necesidad de instalar Kafka localmente.
#
# ARQUITECTURA DE ESTE SCRIPT
# ----------------------------
#
#   ┌──────────────────────┐
#   │   TheSportsDB API    │  ← fuente de datos (equipos + partidos)
#   └──────────┬───────────┘
#              │ HTTP GET
#              ▼
#   ┌──────────────────────┐
#   │  Producer (Thread)   │  ← extrae y publica en Kafka con delay
#   └──────────┬───────────┘
#              │ Produce (JSON, esquema bronze completo)
#       ┌──────┴──────┐
#       ▼             ▼
#   [equipos]    [partidos]   ← tópicos en Kafka (Aiven cloud)
#       │             │
#       └──────┬──────┘
#              ▼
#   ┌──────────────────────┐
#   │  Consumer (Thread)   │  ← lee los mensajes en tiempo real
#   └──────────┬───────────┘
#              │ micro-batch + MERGE por id
#              ▼
#   ┌──────────────────────┐
#   │  Delta Lake (bronze) │  ← MISMA tabla que alimenta el batch
#   └──────────────────────┘
#
# FLUJO DE DATOS
# --------------
# 1. El Producer consulta los endpoints de TheSportsDB (equipos y partidos).
#    NOTA: la API entrega la temporada completa de una sola vez, así que el
#    "tiempo real" se SIMULA reproduciendo los registros mensaje a mensaje
#    con una pequeña pausa. No es un stream de eventos genuino (la fuente no
#    lo ofrece), pero el mecanismo de ingesta streaming sí es real.
# 2. Cada registro se serializa como JSON con el esquema completo de bronze
#    (bronze/mappers.py) y se publica en Kafka.
# 3. El Consumer corre en paralelo (hilo separado), lee los mensajes a medida
#    que llegan, los muestra y los entrega al sink (streaming/sink.py).
# 4. El sink los persiste en la MISMA capa bronze que el batch, en micro-batches
#    y con MERGE por id (idempotente). Así bronze recibe de dos fuentes —API
#    batch + stream Kafka— y silver/gold no cambian.
# 5. Ambos hilos se sincronizan: el Consumer se detiene automáticamente
#    cuando el Producer termina de enviar todos los mensajes.
#
# AUTENTICACIÓN CON AIVEN
# ------------------------
# Aiven utiliza TLS mutuo (mTLS): tanto el servidor como el cliente
# se autentican con certificados. Por eso se necesitan 3 archivos:
#   - ca.pem        : certificado de la autoridad certificante (CA)
#   - service.cert  : certificado público del cliente
#   - service.key   : clave privada del cliente
#
# Las rutas y credenciales se leen desde un archivo .env para no
# exponer datos sensibles en el código fuente.
#
# =============================================================================

# Dependencias declaradas en requirements.txt.
# Instalar antes de correr:  python -m pip install -r requirements.txt

import json
import os
import time
import logging
import threading
from datetime import datetime, timezone

from confluent_kafka import Producer, Consumer, KafkaError
from confluent_kafka.admin import AdminClient, NewTopic
from dotenv import load_dotenv

load_dotenv()

# Validar que las variables de entorno críticas estén definidas
_vars_requeridas = [
    "KAFKA_BOOTSTRAP_SERVERS", "KAFKA_SSL_CA",
    "KAFKA_SSL_CERT", "KAFKA_SSL_KEY",
    "THESPORTSDB_API_KEY", "THESPORTSDB_LIGA_ID",
]
_faltantes = [v for v in _vars_requeridas if not os.getenv(v)]
if _faltantes:
    raise EnvironmentError(
        f"Faltan variables en el archivo .env: {', '.join(_faltantes)}"
    )

from config import (
    URL_BASE, LIGA_NOMBRE as LIGA, LIGA_ID, TEMPORADA,
    DIR_EQUIPOS_BRONZE, DIR_PARTIDOS_BRONZE,
)
from bronze.mappers import mapear_equipo, mapear_partido
from streaming.sink import BronzeSink
from utils.api import obtener_datos_api

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# CONFIGURACIÓN KAFKA — leída desde .env

_SSL_BASE = {
    "bootstrap.servers":        os.getenv("KAFKA_BOOTSTRAP_SERVERS"),
    "security.protocol":        "SSL",
    "ssl.ca.location":          os.getenv("KAFKA_SSL_CA"),
    "ssl.certificate.location": os.getenv("KAFKA_SSL_CERT"),
    "ssl.key.location":         os.getenv("KAFKA_SSL_KEY"),
}

KAFKA_PRODUCER_CONFIG = _SSL_BASE.copy()

KAFKA_CONSUMER_CONFIG = {
    **_SSL_BASE,
    "group.id":          "consumer_thesportsdb",
    "auto.offset.reset": "earliest",
}

TOPICO_EQUIPOS  = "equipos"
TOPICO_PARTIDOS = "partidos"
PAUSA_ENTRE_MENSAJES = 0.3  # segundos entre mensajes para simular streaming


def _log_banner(*lineas: str) -> None:
    """Imprime un separador con título centrado en el log."""
    sep = "=" * 62
    logger.info(sep)
    for linea in lineas:
        logger.info("  " + linea)
    logger.info(sep)


# CREACIÓN DE TÓPICOS

def crear_topicos_si_no_existen(topicos: list[str]) -> None:
    """Crea los tópicos en Aiven si todavía no existen."""
    admin = AdminClient(_SSL_BASE)
    existentes = admin.list_topics(timeout=10).topics.keys()
    nuevos = [
        NewTopic(t, num_partitions=1, replication_factor=1)
        for t in topicos if t not in existentes
    ]
    if not nuevos:
        logger.info("Tópicos ya existentes: %s", topicos)
        return
    resultados = admin.create_topics(nuevos)
    for topico, futuro in resultados.items():
        try:
            futuro.result()
            logger.info("Tópico '%s' creado correctamente.", topico)
        except Exception as e:
            logger.error("Error al crear tópico '%s': %s", topico, e)

# EXTRACCIÓN DESDE LA API

def _obtener_registros(endpoint: str, params: dict, clave: str, descripcion: str) -> list:
    """Llama a la API y devuelve la lista bajo `clave`, registrando cuántos se obtuvieron."""
    registros = obtener_datos_api(URL_BASE, endpoint, params, clave)
    logger.info("%s obtenidos de la API: %d", descripcion, len(registros or []))
    return registros or []


def obtener_equipos(liga: str) -> list:
    return _obtener_registros("/search_all_teams.php", {"l": liga}, "teams", "Equipos")


def obtener_partidos(liga_id: str, temporada: str) -> list:
    return _obtener_registros("/eventsseason.php", {"id": liga_id, "s": temporada}, "events", "Partidos")

# SERIALIZACIÓN
#
# Se usa el mapeo canónico de bronze (bronze/mappers.py): los mensajes viajan
# con el esquema COMPLETO de la tabla bronze, de modo que el consumer pueda
# persistirlos en la misma tabla que el batch sin divergencias de esquema.

def _timestamp_actual() -> tuple[str, str]:
    """Retorna (timestamp_ISO, fecha_YYYY-MM-DD) en UTC para el envío."""
    ahora = datetime.now(timezone.utc)
    return ahora.isoformat(), ahora.strftime("%Y-%m-%d")

# 
# PRODUCER

def _callback_entrega(err, msg):
    """Informa el resultado de cada mensaje producido a Kafka."""
    if err:
        logger.error("Error al entregar mensaje: %s", err)
    else:
        logger.info(
            "[PRODUCER] Entregado → tópico '%s' | offset %d",
            msg.topic(), msg.offset()
        )


def producir_mensajes(producer: Producer, topico: str, registros: list,
                      serializador, clave_id: str) -> None:
    """Envía registros a un tópico de Kafka uno por uno con pausa entre cada uno.

    Args:
        producer   : instancia del Producer de Kafka
        topico     : nombre del tópico destino
        registros  : lista de dicts crudos de la API
        serializador: función que transforma el dict crudo al formato deseado
        clave_id   : campo usado como key del mensaje en Kafka
    """
    logger.info("[PRODUCER] Enviando %d registros al tópico '%s'...", len(registros), topico)
    for registro in registros:
        datos = serializador(registro)
        producer.produce(
            topic=topico,
            key=str(datos.get(clave_id, "")).encode("utf-8"),
            value=json.dumps(datos, ensure_ascii=False).encode("utf-8"),
            callback=_callback_entrega,
        )
        producer.poll(0)
        time.sleep(PAUSA_ENTRE_MENSAJES)

    producer.flush()
    logger.info("[PRODUCER] Tópico '%s' completado.", topico)


def ejecutar_producer(evento_fin: threading.Event) -> None:
    """Extrae datos de TheSportsDB y los produce a Kafka. Señala fin al terminar."""
    _log_banner("PRODUCER — TheSportsDB → Kafka (Aiven)", f"Liga: {LIGA} | Temporada: {TEMPORADA}")

    producer = Producer(KAFKA_PRODUCER_CONFIG)

    equipos  = obtener_equipos(LIGA)
    partidos = obtener_partidos(LIGA_ID, TEMPORADA)

    timestamp, fecha_extraccion = _timestamp_actual()
    ser_equipo  = lambda raw: mapear_equipo(raw, timestamp)
    ser_partido = lambda raw: mapear_partido(raw, timestamp, fecha_extraccion)

    producir_mensajes(producer, TOPICO_EQUIPOS,  equipos,  ser_equipo,  "id_equipo")
    producir_mensajes(producer, TOPICO_PARTIDOS, partidos, ser_partido, "id_evento")

    logger.info("[PRODUCER] Todos los mensajes enviados correctamente.")
    evento_fin.set()  # señal para que el consumer se detenga

# CONSUMER

def mostrar_mensaje(topico: str, datos: dict) -> None:
    """Formatea e imprime un mensaje recibido según su tópico."""
    if topico == TOPICO_EQUIPOS:
        logger.info(
            "[CONSUMER] EQUIPO | %s | Ciudad: %s | Estadio: %s",
            datos.get("nombre"), datos.get("ciudad"), datos.get("estadio"),
        )
    elif topico == TOPICO_PARTIDOS:
        logger.info(
            "[CONSUMER] PARTIDO | %s vs %s | %s-%s | Fecha: %s | Estado: %s",
            datos.get("equipo_local"), datos.get("equipo_visitante"),
            datos.get("goles_local"),  datos.get("goles_visitante"),
            datos.get("fecha_partido"), datos.get("estado"),
        )
    else:
        logger.warning("[CONSUMER] Mensaje de tópico desconocido: '%s'", topico)


def ejecutar_consumer(
    evento_fin: threading.Event,
    evento_suscripto: threading.Event,
    sink: BronzeSink,
) -> None:
    """Lee mensajes de Kafka y los persiste en bronze vía el sink.

    Corre hasta que el Producer señale que terminó y no queden mensajes.
    Cada mensaje se muestra por consola (feedback) y se entrega al sink,
    que lo escribe a la capa bronze en micro-batches.
    """
    consumer = Consumer(KAFKA_CONSUMER_CONFIG)
    consumer.subscribe([TOPICO_EQUIPOS, TOPICO_PARTIDOS])
    evento_suscripto.set()  # avisa al pipeline que ya está listo para recibir

    logger.info("[CONSUMER] Escuchando tópicos: %s, %s", TOPICO_EQUIPOS, TOPICO_PARTIDOS)

    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                # El producer terminó y no hay mensajes pendientes: salimos
                if evento_fin.is_set():
                    break
                continue
            if msg.error():
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    logger.error("[CONSUMER] Error: %s", msg.error())
                continue
            try:
                datos = json.loads(msg.value().decode("utf-8"))
                mostrar_mensaje(msg.topic(), datos)
                sink.agregar(msg.topic(), datos)  # persiste a bronze (micro-batch)
            except json.JSONDecodeError as e:
                logger.warning("[CONSUMER] No se pudo parsear mensaje: %s", e)
    finally:
        sink.cerrar()      # vacía los buffers pendientes antes de salir
        consumer.close()
        logger.info("[CONSUMER] Cerrado correctamente.")

# PIPELINE PRINCIPAL

def ejecutar_pipeline() -> None:
    """Corre Producer y Consumer en paralelo usando hilos."""
    _log_banner("PIPELINE KAFKA — TheSportsDB (Aiven)")

    crear_topicos_si_no_existen([TOPICO_EQUIPOS, TOPICO_PARTIDOS])

    # Sink que persiste el stream en la MISMA capa bronze que el batch.
    sink = BronzeSink(
        ruta_equipos=DIR_EQUIPOS_BRONZE,
        ruta_partidos=DIR_PARTIDOS_BRONZE,
        topico_equipos=TOPICO_EQUIPOS,
        topico_partidos=TOPICO_PARTIDOS,
    )

    evento_fin       = threading.Event()
    evento_suscripto = threading.Event()

    hilo_consumer = threading.Thread(
        target=ejecutar_consumer, args=(evento_fin, evento_suscripto, sink), name="Consumer"
    )
    hilo_producer = threading.Thread(
        target=ejecutar_producer, args=(evento_fin,), name="Producer"
    )

    hilo_consumer.start()
    evento_suscripto.wait()  # espera hasta que el consumer confirme la suscripción
    hilo_producer.start()

    hilo_producer.join()
    hilo_consumer.join()     # espera a que el consumer drene todos los mensajes
    _log_banner("Pipeline finalizado.")


if __name__ == "__main__":
    ejecutar_pipeline()
