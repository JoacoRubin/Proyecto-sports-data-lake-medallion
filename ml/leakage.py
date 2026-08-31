# =============================================================================
# ml/leakage.py — Que es leakage en este problema. UNA sola definicion.
#
# POR QUE EXISTE ESTE MODULO
# --------------------------
# "Estas columnas describen el partido ya jugado" es una regla de DOMINIO, y
# la necesitan dos lugares distintos:
#
#   - ml/features.py y ml/features_goles.py   -> para descartarlas al construir
#   - quality/contracts.py                    -> para rechazarlas al validar
#
# Estaba escrita dos veces, y las dos copias ya habian divergido. El riesgo es
# concreto: se agrega una estadistica nueva a bronze, se actualiza una lista y
# no la otra, y queda un agujero por donde entra leakage sin que nada lo note.
#
# Una regla de negocio se define una vez, en el lugar que la nombra, y los
# demas la importan. Este modulo no depende de nada: ni pandas hace falta para
# declararla.
#
# QUE NO VA ACA
# -------------
# - `arbitro`: se descarta por no tener senal predictiva (fiabilidad split-half
#   -0.04), que es una DECISION DE MODELADO, no leakage. Vive en
#   ml/dataset.METADATA. Mezclarlas hacia que el contrato rechazara una columna
#   perfectamente legitima.
# - `_i`, `_par`, `_fecha`: andamiaje interno del calculo. Se borran por
#   prefijo, no por lista.
# =============================================================================

# Columnas que describen lo que PASO en el partido.
#
# Cuando se predice, ese partido todavia no se jugo: ninguna de estas existe
# aun. Usarlas da metricas espectaculares en validacion y un modelo inservible
# en produccion.
COLUMNAS_POST_PARTIDO: tuple[str, ...] = (
    # Marcador
    "goles_local", "goles_visitante",
    "goles_local_entretiempo", "goles_visitante_entretiempo",
    # Disciplina
    "rojas_local", "rojas_visitante",
    "amarillas_local", "amarillas_visitante",
    "faltas_local", "faltas_visitante",
    # Juego
    "tiros_local", "tiros_visitante",
    "tiros_arco_local", "tiros_arco_visitante",
    "corners_local", "corners_visitante",
    # Derivados del marcador: salen de las anteriores, asi que arrastran el
    # mismo problema.
    "rojas_total", "goles_totales",
)


def detectar_leakage(df) -> list[str]:
    """Devuelve las columnas post-partido presentes en el DataFrame.

    Lista vacia significa que la matriz esta limpia. Se devuelven TODAS las
    encontradas y no la primera: si hay varias, conviene verlas juntas.
    """
    return [c for c in COLUMNAS_POST_PARTIDO if c in df.columns]
