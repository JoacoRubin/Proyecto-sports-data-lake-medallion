# =============================================================================
# bronze/mappers_footballdata.py — Mapeo de football-data.co.uk al esquema
# canónico bronze.
#
# POR QUÉ EXISTE ESTA SEGUNDA FUENTE
# ----------------------------------
# La API pública de TheSportsDB devuelve 5 partidos por liga (verificado en
# todos sus endpoints). Alcanza para demostrar el pipeline, no para entrenar
# nada. football-data.co.uk publica temporadas completas desde 1993, sin key
# y sin registro, y además trae estadísticas de partido (tiros, corners,
# tarjetas) y cuotas de apuestas: features reales, no derivadas del resultado.
#
# EL PROBLEMA QUE RESUELVE ESTE MÓDULO
# ------------------------------------
# El CSV no trae ids. Hay que sintetizarlos, y tienen que ser DETERMINISTAS:
# si el id cambiara entre descargas, la deduplicación por id_evento dejaría de
# funcionar y cada re-ingesta duplicaría el data lake en silencio.
# Se derivan por hash del contenido identificatorio del partido.
#
# Los ids van prefijados con 'FD-' para no colisionar jamás con los ids
# numéricos de TheSportsDB dentro de la misma tabla.
# =============================================================================

import hashlib
import re
import unicodedata
from datetime import datetime

import pandas as pd

# Divisiones de football-data.co.uk -> nombre de liga.
# Los nombres coinciden con los de config.LIGAS para que ambas fuentes hablen
# el mismo idioma en la columna 'liga'.
LIGAS_FOOTBALLDATA = {
    "E0":  "English Premier League",
    "E1":  "English Championship",
    "SP1": "Spanish La Liga",
    "I1":  "Italian Serie A",
    "D1":  "German Bundesliga",
    "F1":  "French Ligue 1",
    "N1":  "Dutch Eredivisie",
    "P1":  "Portuguese Primeira Liga",
    "B1":  "Belgian First Division A",
    "SC0": "Scottish Premiership",
    "T1":  "Turkish Super Lig",
    "G1":  "Greek Super League",
}

# El CSV publica únicamente partidos ya jugados.
ESTADO_FINALIZADO = "FT"

_PREFIJO_FUENTE = "FD"

# Columnas del CSV que NO son canónicas pero son la razón de traer esta fuente:
# estadísticas de partido y cuotas. Son las que alimentan el feature
# engineering (promedios móviles de fechas anteriores, nunca del propio
# partido — eso sería leakage).
_ESTADISTICAS = {
    "goles_local_entretiempo":     "HTHG",
    "goles_visitante_entretiempo": "HTAG",
    "tiros_local":                 "HS",
    "tiros_visitante":             "AS",
    "tiros_arco_local":            "HST",
    "tiros_arco_visitante":        "AST",
    "faltas_local":                "HF",
    "faltas_visitante":            "AF",
    "corners_local":               "HC",
    "corners_visitante":           "AC",
    "amarillas_local":             "HY",
    "amarillas_visitante":         "AY",
    "rojas_local":                 "HR",
    "rojas_visitante":             "AR",
}

_CUOTAS = {
    "cuota_local":     "B365H",
    "cuota_empate":    "B365D",
    "cuota_visitante": "B365A",
}

# Cuotas del MERCADO: promedio de todas las casas, no de una sola.
#
# Son mejor estimador que Bet365 en solitario, y habilitan el baseline mas duro
# que existe en prediccion deportiva: la prediccion agregada de gente que se
# juega plata. Le ganamos a "predecir la prevalencia" es una vara pasiva; el
# mercado es una vara adversaria.
#
# GOTCHA: football-data renombro estas columnas en la temporada 2019-20.
# Antes se llamaban BbAv* (Betbrain average). Es la misma magnitud, y sin el
# fallback se pierden cuatro temporadas de historia.
_CUOTAS_MERCADO = {
    "cuota_over25":           ("Avg>2.5", "BbAv>2.5"),
    "cuota_under25":          ("Avg<2.5", "BbAv<2.5"),
    "cuota_local_mercado":    ("AvgH",    "BbAvH"),
    "cuota_empate_mercado":   ("AvgD",    "BbAvD"),
    "cuota_visitante_mercado":("AvgA",    "BbAvA"),
}

_COLUMNAS_ENTERAS = [
    "goles_local", "goles_visitante", *_ESTADISTICAS.keys(),
]
_COLUMNAS_DECIMALES = list(_CUOTAS.keys()) + list(_CUOTAS_MERCADO.keys())


def _valor(raw: dict, clave: str) -> str | None:
    """Lee una clave del CSV crudo devolviendo None ante ausencia, NaN o vacío.

    Las temporadas viejas no traen tiros ni cuotas, y el CSV real termina con
    filas parciales: no se puede asumir que la columna exista ni tenga valor.
    """
    valor = raw.get(clave)
    if valor is None or valor == "" or (isinstance(valor, float) and pd.isna(valor)):
        return None
    return str(valor).strip()


def parsear_fecha(fecha: str | None) -> str | None:
    """Convierte la fecha del CSV (DD/MM/YYYY o DD/MM/YY) a ISO YYYY-MM-DD.

    Las temporadas anteriores a ~2003 usan año de 2 dígitos. Devolver ISO no
    es cosmético: es lo que exige el contrato bronze, porque TheSportsDB ya
    entrega ISO y ambas fuentes tienen que ser comparables.
    """
    if not fecha:
        return None
    texto = str(fecha).strip()
    for formato in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(texto, formato).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def codigo_temporada_a_nombre(codigo: str) -> str:
    """Traduce el código de temporada de la URL ('2425') a '2024-2025'.

    football-data arranca en la temporada 93/94, así que un código que empieza
    con 90-99 es del siglo XX y el resto del XXI.
    """
    inicio_corto = int(codigo[:2])
    inicio = 1900 + inicio_corto if inicio_corto >= 90 else 2000 + inicio_corto
    return f"{inicio}-{inicio + 1}"


def _slug(texto: str) -> str:
    """Normaliza un nombre a slug ascii: 'Atlético Madrid' -> 'atletico-madrid'."""
    sin_acentos = "".join(
        c for c in unicodedata.normalize("NFKD", str(texto))
        if not unicodedata.combining(c)
    )
    limpio = re.sub(r"[^a-z0-9]+", " ", sin_acentos.lower()).strip()
    return re.sub(r"\s+", "-", limpio)


def generar_id_equipo(nombre: str) -> str:
    """Id sintético y legible para un equipo: 'Man United' -> 'FD-man-united'.

    Legible a propósito: facilita depurar el cruce de nombres entre fuentes,
    que es el punto donde este tipo de integración suele romperse.
    """
    return f"{_PREFIJO_FUENTE}-{_slug(nombre)}"


def generar_id_evento(
    div: str, codigo_temporada: str, fecha_iso: str, local: str, visitante: str
) -> str:
    """Id sintético DETERMINISTA de un partido.

    Se hashea la tupla que identifica unívocamente un partido. Determinista
    significa que re-descargar el mismo CSV produce exactamente los mismos
    ids: es lo que vuelve idempotente la ingesta y hace que la deduplicación
    por id_evento realmente deduplique.

    Los nombres se normalizan a slug para que un espacio de más en el CSV no
    genere un partido "nuevo".
    """
    clave = "|".join([
        div.strip().upper(), codigo_temporada.strip(),
        str(fecha_iso), _slug(local), _slug(visitante),
    ])
    digest = hashlib.sha1(clave.encode("utf-8")).hexdigest()[:12]
    return f"{_PREFIJO_FUENTE}-{digest}"


def mapear_partido_footballdata(
    raw: dict, div: str, codigo_temporada: str, timestamp: str, fecha_extraccion: str
) -> dict:
    """Convierte una fila cruda del CSV al esquema bronze canónico + estadísticas.

    Las 17 columnas canónicas son las mismas que produce el mapper de
    TheSportsDB: así ambas fuentes escriben el mismo esquema y silver las
    procesa sin saber de dónde vinieron. Las columnas extra viajan de más.
    """
    local     = _valor(raw, "HomeTeam")
    visitante = _valor(raw, "AwayTeam")
    fecha_iso = parsear_fecha(_valor(raw, "Date"))

    registro = {
        # --- 17 columnas canónicas (esquema bronze compartido) ---
        "id_evento":            generar_id_evento(
                                    div, codigo_temporada, fecha_iso, local or "", visitante or ""
                                ),
        "nombre_evento":        f"{local} vs {visitante}",
        "temporada":            codigo_temporada_a_nombre(codigo_temporada),
        "liga":                 LIGAS_FOOTBALLDATA.get(div, div),
        "id_liga":              f"{_PREFIJO_FUENTE}-{div}",
        "equipo_local":         local,
        "id_equipo_local":      generar_id_equipo(local) if local else None,
        "goles_local":          _valor(raw, "FTHG"),
        "equipo_visitante":     visitante,
        "id_equipo_visitante":  generar_id_equipo(visitante) if visitante else None,
        "goles_visitante":      _valor(raw, "FTAG"),
        "fecha_partido":        fecha_iso,
        "hora_partido":         _valor(raw, "Time"),
        "estadio":              None,          # el CSV no publica sede
        "estado":               ESTADO_FINALIZADO,
        "timestamp_extraccion": timestamp,
        "fecha_extraccion":     fecha_extraccion,
        # --- extras: el motivo de traer esta fuente ---
        "arbitro":              _valor(raw, "Referee"),
    }
    registro.update({
        destino: _valor(raw, origen)
        for destino, origen in {**_ESTADISTICAS, **_CUOTAS}.items()
    })
    # Cuotas de mercado: nombre nuevo primero, nombre viejo como fallback.
    registro.update({
        destino: next(
            (v for v in (_valor(raw, o) for o in origenes) if v is not None), None
        )
        for destino, origenes in _CUOTAS_MERCADO.items()
    })
    return registro


def tipar_partidos_footballdata(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica los tipos canónicos + los de las columnas de estadísticas.

    Enteros y decimales van a tipos NULLABLE (Int64/Float64): las temporadas
    viejas no traen estadísticas y un NaN no puede tumbar la ingesta.
    """
    df = df.copy()

    for col in _COLUMNAS_ENTERAS:
        if col in df.columns:
            numerico = pd.to_numeric(df[col], errors="coerce")
            df[col] = numerico.round().astype("Int64")

    for col in _COLUMNAS_DECIMALES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Float64")

    columnas_texto = [
        c for c in df.columns
        if c not in _COLUMNAS_ENTERAS and c not in _COLUMNAS_DECIMALES
    ]
    return df.astype({col: "string" for col in columnas_texto})
