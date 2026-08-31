# =============================================================================
# quality/contracts.py — Contratos de datos entre capas (Pandera).
#
# Convierten fallas SILENCIOSAS en fallas RUIDOSAS. Si una transformación
# produce datos que violan lo esperado (tipos, rangos, invariantes de negocio),
# el pipeline se detiene EN EL MOMENTO, en la capa donde se rompió, en vez de
# propagar basura hacia gold y el dashboard.
#
# Ejemplo real que motivó esto: un cambio de la API ('Match Finished' -> 'FT')
# dejaba silver vacío y el pipeline "corría OK". El check cross-layer de abajo
# lo habría detectado en el acto.
# =============================================================================

import pandera.pandas as pa

from bronze.mappers import ESTADOS_FINALIZADOS
from ml.leakage import COLUMNAS_POST_PARTIDO, detectar_leakage

_RESULTADOS_VALIDOS = ["Local", "Visitante", "Empate"]


# --- Contrato de la capa bronze (partidos crudos, MULTI-FUENTE) ------------
#
# Es la frontera que hace posible ingerir de TheSportsDB y de
# football-data.co.uk hacia la misma tabla. Cada fuente tiene su extractor y
# su mapper, pero ambas deben cumplir ESTE contrato antes de tocar silver.
#
# Si mañana una fuente cambia su formato (una fecha que deja de venir en ISO,
# ids que se repiten), revienta acá: en la capa donde se rompió y con el
# nombre de la fuente todavía a la vista.
#
# Bronze es más permisivo que silver a propósito: acepta partidos pendientes
# sin goles, porque filtrarlos es trabajo de silver, no de la ingesta.
_FECHA_ISO = r"^\d{4}-\d{2}-\d{2}$"

BRONZE_PARTIDOS = pa.DataFrameSchema(
    columns={
        "id_evento":           pa.Column(str, nullable=False, unique=True, coerce=True),
        "liga":                pa.Column(str, nullable=False, coerce=True),
        "equipo_local":        pa.Column(str, nullable=False, coerce=True),
        "equipo_visitante":    pa.Column(str, nullable=False, coerce=True),
        "id_equipo_local":     pa.Column(str, nullable=False, coerce=True),
        "id_equipo_visitante": pa.Column(str, nullable=False, coerce=True),
        "goles_local":         pa.Column("Int64", pa.Check.ge(0), nullable=True, coerce=True),
        "goles_visitante":     pa.Column("Int64", pa.Check.ge(0), nullable=True, coerce=True),
        "fecha_partido":       pa.Column(
            str,
            pa.Check.str_matches(_FECHA_ISO),
            nullable=False,
            coerce=True,
        ),
        "estado":              pa.Column(str, nullable=False, coerce=True),
    },
    checks=[
        # Invariante de negocio: nadie juega contra sí mismo. Atrapa errores
        # de mapeo de columnas local/visitante en una fuente nueva.
        pa.Check(
            lambda df: df["equipo_local"] != df["equipo_visitante"],
            error="Invariante violado: un equipo no puede jugar contra sí mismo",
        ),
    ],
    strict=False,
    name="bronze_partidos",
)


# --- Contrato de la capa silver (partidos procesados) ----------------------
SILVER_PARTIDOS = pa.DataFrameSchema(
    columns={
        "id_evento":        pa.Column(str,  nullable=False, unique=True, coerce=True),
        "liga":             pa.Column(str,  nullable=False, coerce=True),
        "goles_local":      pa.Column(int,  pa.Check.ge(0), coerce=True),
        "goles_visitante":  pa.Column(int,  pa.Check.ge(0), coerce=True),
        "resultado":        pa.Column(str,  pa.Check.isin(_RESULTADOS_VALIDOS), coerce=True),
        "diferencia_goles": pa.Column(int,  pa.Check.ge(0), coerce=True),
        "es_goleada":       pa.Column(bool, coerce=True),
    },
    strict=False,   # se permiten columnas adicionales
    name="silver_partidos",
)


# --- Contrato de la capa gold (tabla de posiciones) ------------------------
GOLD_POSICIONES = pa.DataFrameSchema(
    columns={
        "liga":      pa.Column(str, nullable=False, coerce=True),
        "id_equipo": pa.Column(str, nullable=False, coerce=True),
        "equipo":    pa.Column(str, nullable=True,  coerce=True),
        "PJ":  pa.Column(int, pa.Check.ge(0), coerce=True),
        "PG":  pa.Column(int, pa.Check.ge(0), coerce=True),
        "PE":  pa.Column(int, pa.Check.ge(0), coerce=True),
        "PP":  pa.Column(int, pa.Check.ge(0), coerce=True),
        "GF":  pa.Column(int, pa.Check.ge(0), coerce=True),
        "GC":  pa.Column(int, pa.Check.ge(0), coerce=True),
        "DG":  pa.Column(int, coerce=True),
        "Pts": pa.Column(int, pa.Check.ge(0), coerce=True),
    },
    checks=[
        # Invariantes de negocio: si estos no se cumplen, la agregación tiene un bug.
        pa.Check(
            lambda df: df["Pts"] == df["PG"] * 3 + df["PE"],
            error="Invariante violado: Pts debe ser PG*3 + PE",
        ),
        pa.Check(
            lambda df: df["PJ"] == df["PG"] + df["PE"] + df["PP"],
            error="Invariante violado: PJ debe ser PG + PE + PP",
        ),
    ],
    # Un equipo aparece una sola vez por liga Y POR FUENTE. Los nombres no
    # estan reconciliados entre fuentes, asi que "Man United" (football-data) y
    # "Manchester United" (TheSportsDB) son dos filas legitimas.
    unique=["fuente", "liga", "id_equipo"],
    strict=False,
    name="gold_posiciones",
)


# --- Contrato de la matriz de FEATURES (frontera de ML) ---------------------
#
# El proyecto ya valida bronze, silver y gold. Esta es la frontera que faltaba,
# y es donde un error silencioso sale mas barato: una columna con leakage, una
# probabilidad fuera de rango o una tasa historica corrupta no rompen nada.
# Simplemente entrenan un modelo malo, y nadie se entera hasta produccion.
#
# El check mas importante NO es de tipos: es que no sobreviva ninguna columna
# que describa lo ocurrido EN el partido que se quiere predecir.

# La regla de que es leakage se importa, no se redefine: estaba escrita aca y
# en ml/features.py, y las dos copias ya habian divergido. Ver ml/leakage.py.
_COLUMNAS_PROHIBIDAS_ML = COLUMNAS_POST_PARTIDO



def _probabilidades_suman_uno(df):
    """Las probabilidades implicitas se normalizan al construirlas.

    `skipna=False` es imprescindible: con el default de pandas, una fila con
    las tres cuotas nulas suma 0.0 en vez de NaN y el check la marcaria como
    violacion. Y esas filas existen —6 partidos del dataset no tienen cuotas
    publicadas, probablemente suspendidos—, asi que son un caso valido, no un
    error de mapeo.
    """
    columnas = ["prob_local", "prob_empate", "prob_visitante"]
    if not all(c in df.columns for c in columnas):
        return True
    total = df[columnas].sum(axis=1, skipna=False)
    return total.isna() | ((total - 1.0).abs() < 1e-6)


def contrato_matriz_ml(target: str) -> pa.DataFrameSchema:
    """Arma el contrato de la matriz de features para un target dado.

    Se parametriza en vez de copiarse porque el proyecto tiene dos modelos
    sobre la misma maquinaria. Las probabilidades y las tasas historicas se
    validan POR PATRON y no por lista de nombres: se llaman `tasa_exp` en
    expulsiones y `tasa_over` en goles, pero la invariante es la misma —una
    probabilidad vive en [0, 1]— y un patron no hay que actualizarlo cada vez
    que nace un modelo.
    """
    return pa.DataFrameSchema(
        columns={
            target: pa.Column(int, pa.Check.isin([0, 1]), coerce=True),
            "liga":  pa.Column(str, nullable=False, coerce=True),
            # Probabilidades implicitas (incluida la del mercado) y tasas
            # historicas suavizadas: por construccion viven en [0, 1].
            # OJO: pandera ancla los regex al INICIO del nombre (re.match, no
            # re.search), asi que `tasa_` no matchearia `equipo_tasa_exp_local`.
            # Y `required=False` es necesario porque una matriz puede no tener
            # ninguna columna de ese patron sin estar mal.
            r"^prob_.*": pa.Column(
                float, pa.Check.in_range(0.0, 1.0),
                nullable=True, coerce=True, regex=True, required=False,
            ),
            r".*tasa_.*": pa.Column(
                float, pa.Check.in_range(0.0, 1.0),
                nullable=True, coerce=True, regex=True, required=False,
            ),
            "paridad": pa.Column(
                float, pa.Check.in_range(0.0, 1.0),
                nullable=True, coerce=True, required=False,
            ),
        },
        checks=[
            pa.Check(
                _probabilidades_suman_uno,
                error="Invariante violado: prob_local + prob_empate + prob_visitante debe ser 1",
            ),
        ],
        strict=False,   # las medias moviles varian segun las ventanas configuradas
        name=f"matriz_ml[{target}]",
    )



def validar_matriz_ml(df, target: str = "hubo_expulsion"):
    """Valida la matriz de features antes de entrenar. Sirve a ambos modelos.

    Primero el check de leakage —que no es de esquema y por eso va aparte— y
    despues los tipos, rangos e invariantes.

    Args:
        df: matriz de features.
        target: columna objetivo. Por defecto el modelo de expulsiones.

    Raises:
        ValueError: si sobrevivio una columna del propio partido.
        SchemaErrors: si algun tipo, rango o invariante no se cumple.
    """
    filtradas = detectar_leakage(df)
    if filtradas:
        raise ValueError(
            f"Leakage en la matriz de features: {filtradas}. Son estadisticas "
            f"del propio partido, que al predecir todavia no se jugo."
        )
    return contrato_matriz_ml(target).validate(df, lazy=True)


def validar_bronze_partidos(df):
    """Valida el contrato de bronze. Lanza SchemaError si algo no cumple.

    Se aplica a la salida de CADA extractor, sin importar la fuente.
    """
    return BRONZE_PARTIDOS.validate(df, lazy=True)


def validar_silver_partidos(df):
    """Valida el contrato de silver. Lanza SchemaError si algo no cumple."""
    return SILVER_PARTIDOS.validate(df, lazy=True)


def validar_gold_posiciones(df):
    """Valida el contrato de gold. Lanza SchemaError si algo no cumple."""
    return GOLD_POSICIONES.validate(df, lazy=True)


def verificar_silver_no_vacio_si_habia_finalizados(df_bronze, df_silver) -> None:
    """Check cross-layer: silver no puede quedar vacío si bronze traía finalizados.

    Es la red que atrapa fallas silenciosas de filtrado (como 'FT' vs 'Finished'):
    datos que entran, no salen, y nadie se entera.
    """
    finalizados = (
        df_bronze["estado"].str.strip().str.upper().isin(ESTADOS_FINALIZADOS).sum()
    )
    if finalizados > 0 and len(df_silver) == 0:
        raise ValueError(
            f"Calidad de datos: bronze tenía {finalizados} partidos finalizados "
            f"pero silver quedó vacío. Revisar el filtrado en silver/transformations.py."
        )
