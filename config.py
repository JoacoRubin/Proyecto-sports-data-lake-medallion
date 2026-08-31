# =============================================================================
# config.py — Configuración global del proyecto
# =============================================================================

import os
import logging

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# API
# La key "3" es la clave pública de test de TheSportsDB: sirve como default para
# que el proyecto funcione sin configuración (por ej. en un deploy). Para datos
# completos se puede definir THESPORTSDB_API_KEY con una key premium.
API_KEY  = os.getenv("THESPORTSDB_API_KEY", "3")
URL_BASE = f"https://www.thesportsdb.com/api/v1/json/{API_KEY}"

LIGA_NOMBRE = os.getenv("THESPORTSDB_LIGA")
LIGA_ID     = os.getenv("THESPORTSDB_LIGA_ID")
TEMPORADA   = os.getenv("THESPORTSDB_TEMPORADA")

# -----------------------------------------------------------------------------
# Ligas del pipeline batch (multi-liga).
#
# La API key pública "3" limita eventsseason a 5 partidos y search_all_teams a
# 10 equipos POR liga. Para enriquecer el data lake se ingiere un conjunto de
# ligas: cada una aporta sus 5 partidos y sus 10 equipos.
#
# Ojo con la temporada: las ligas sudamericanas / MLS son de año calendario
# ("2024"); las europeas cruzan año ("2024-2025").
# -----------------------------------------------------------------------------
LIGAS = [
    {"nombre": "Argentinian Primera Division", "id": "4406", "temporada": "2024"},
    {"nombre": "Brazilian Serie A",            "id": "4351", "temporada": "2024"},
    {"nombre": "American Major League Soccer",  "id": "4346", "temporada": "2024"},
    {"nombre": "English Premier League",        "id": "4328", "temporada": "2024-2025"},
    {"nombre": "Spanish La Liga",               "id": "4335", "temporada": "2024-2025"},
    {"nombre": "Italian Serie A",               "id": "4332", "temporada": "2024-2025"},
    {"nombre": "German Bundesliga",             "id": "4331", "temporada": "2024-2025"},
    {"nombre": "French Ligue 1",                "id": "4334", "temporada": "2024-2025"},
    {"nombre": "Portuguese Primeira Liga",      "id": "4344", "temporada": "2024-2025"},
]

# -----------------------------------------------------------------------------
# Fuente secundaria: football-data.co.uk (CSV historico, sin API key).
#
# La key publica de TheSportsDB devuelve 5 partidos por liga: alcanza para
# demostrar el pipeline, no para entrenar un modelo. Esta fuente publica
# temporadas COMPLETAS desde 1993 y ademas trae estadisticas de partido
# (tiros, corners, tarjetas) y cuotas de apuestas.
#
# Codigos de division: E0=Premier, SP1=La Liga, I1=Serie A, D1=Bundesliga,
# F1=Ligue 1. Ver bronze/mappers_footballdata.LIGAS_FOOTBALLDATA.
#
# Codigo de temporada = los dos anios en dos digitos: "2425" -> 2024-2025.
# -----------------------------------------------------------------------------
FOOTBALLDATA_DIVISIONES = os.getenv(
    "FOOTBALLDATA_DIVISIONES", "E0,SP1,I1,D1,F1"
).split(",")

FOOTBALLDATA_TEMPORADAS = os.getenv(
    "FOOTBALLDATA_TEMPORADAS",
    "1516,1617,1718,1819,1920,2021,2122,2223,2324,2425",
).split(",")

DIR_PARTIDOS_FOOTBALLDATA_BRONZE = "data/bronze/footballdata/partidos"


# Rutas bronze
DIR_EQUIPOS_BRONZE  = "data/bronze/thesportsdb/equipos"
DIR_PARTIDOS_BRONZE = "data/bronze/thesportsdb/partidos"

# -----------------------------------------------------------------------------
# Capa ML: modelo de expulsiones.
#
# El modelo de produccion es el LOGISTICO, no el gradient boosting. Medido
# sobre 7 folds de origen movil, la logistica da PR-AUC 0.2134 contra 0.2042
# del HistGradientBoosting, con intervalos solapados: son indistinguibles.
# Cuando dos modelos empatan, gana el simple y explicable.
#
# Los experimentos se registran en una tabla Delta de gold en vez de en MLflow:
# cero infraestructura nueva y se consultan como cualquier otra tabla del lake.
# -----------------------------------------------------------------------------
ML_MODELO_PRODUCCION = os.getenv("ML_MODELO_PRODUCCION", "logistico")
ML_MIN_TEMPORADAS_TRAIN = int(os.getenv("ML_MIN_TEMPORADAS_TRAIN", "3"))

DIR_MODELOS            = "data/models"
DIR_EXPERIMENTOS_GOLD  = "data/gold/ml/experimentos"
DIR_IMPORTANCIAS_GOLD  = "data/gold/ml/importancias"
NOMBRE_MODELO_EXPULSIONES = "expulsiones"
NOMBRE_MODELO_GOLES       = "goles"


# Rutas silver
DIR_PARTIDOS_SILVER = "data/silver/thesportsdb/partidos_procesados"

# Rutas gold
DIR_ESTADISTICAS_GOLD = "data/gold/thesportsdb/estadisticas_equipos"
