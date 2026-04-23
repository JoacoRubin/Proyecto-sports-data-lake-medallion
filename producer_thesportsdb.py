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
#              │ Produce (JSON)
#       ┌──────┴──────┐
#       ▼             ▼
#   [equipos]    [partidos]   ← tópicos en Kafka (Aiven cloud)
#       │             │
#       └──────┬──────┘
#              ▼
#   ┌──────────────────────┐
#   │  Consumer (Thread)   │  ← lee y muestra los mensajes en tiempo real
#   └──────────────────────┘
#
# FLUJO DE DATOS
# --------------
# 1. El Producer consulta los endpoints de TheSportsDB (equipos y partidos).
# 2. Cada registro se serializa como JSON y se publica en Kafka mensaje a
#    mensaje con una pequeña pausa, simulando un flujo en tiempo real.
# 3. El Consumer corre en paralelo (hilo separado) y lee los mensajes
#    a medida que llegan, imprimiéndolos por consola.
# 4. Ambos hilos se sincronizan: el Consumer se detiene automáticamente
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

import subprocess
import sys

def instalar_dependencias():
    paquetes = ["confluent-kafka", "requests", "python-dotenv"]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet"] + paquetes)

instalar_dependencias()

# IMPORTACIONES (después de instalar dependencias)

import json
import os
import time
import logging
import threading
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

from config import URL_BASE, LIGA_NOMBRE as LIGA, LIGA_ID, TEMPORADA
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

def serializar_equipo(equipo: dict) -> dict:
    """Extrae los campos relevantes de un equipo y los estructura como dict."""
    return {
        "id_equipo":         equipo.get("idTeam"),
        "nombre":            equipo.get("strTeam"),
        "pais":              equipo.get("strCountry"),
        "ciudad":            equipo.get("strCity") or equipo.get("strStadiumLocation"),
        "estadio":           equipo.get("strStadium"),
        "capacidad_estadio": equipo.get("intStadiumCapacity"),
        "anio_fundacion":    equipo.get("intFormedYear"),
        "liga":              equipo.get("strLeague"),
    }


def serializar_partido(partido: dict) -> dict:
    """Extrae los campos relevantes de un partido y los estructura como dict."""
    return {
        "id_evento":        partido.get("idEvent"),
        "equipo_local":     partido.get("strHomeTeam"),
        "equipo_visitante": partido.get("strAwayTeam"),
        "goles_local":      partido.get("intHomeScore"),
        "goles_visitante":  partido.get("intAwayScore"),
        "fecha_partido":    partido.get("dateEvent"),
        "estado":           partido.get("strStatus"),
        "estadio":          partido.get("strVenue"),
        "temporada":        partido.get("strSeason"),
    }

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

    producir_mensajes(producer, TOPICO_EQUIPOS,  equipos,  serializar_equipo,  "id_equipo")
    producir_mensajes(producer, TOPICO_PARTIDOS, partidos, serializar_partido, "id_evento")

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


def ejecutar_consumer(evento_fin: threading.Event, evento_suscripto: threading.Event) -> None:
    """Lee mensajes de Kafka hasta que el Producer señale que terminó y no queden mensajes."""
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
            except json.JSONDecodeError as e:
                logger.warning("[CONSUMER] No se pudo parsear mensaje: %s", e)
    finally:
        consumer.close()
        logger.info("[CONSUMER] Cerrado correctamente.")

# PIPELINE PRINCIPAL

def ejecutar_pipeline() -> None:
    """Corre Producer y Consumer en paralelo usando hilos."""
    _log_banner("PIPELINE KAFKA — TheSportsDB (Aiven)")

    crear_topicos_si_no_existen([TOPICO_EQUIPOS, TOPICO_PARTIDOS])

    evento_fin       = threading.Event()
    evento_suscripto = threading.Event()

    hilo_consumer = threading.Thread(
        target=ejecutar_consumer, args=(evento_fin, evento_suscripto), name="Consumer"
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
