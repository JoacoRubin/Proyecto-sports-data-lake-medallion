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

_RESULTADOS_VALIDOS = ["Local", "Visitante", "Empate"]


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
    unique=["liga", "id_equipo"],   # un equipo aparece una sola vez por liga
    strict=False,
    name="gold_posiciones",
)


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
