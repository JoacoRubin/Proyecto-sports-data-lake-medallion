# =============================================================================
# ml/features.py — Feature engineering para predecir expulsiones.
#
# REGLA ÚNICA DE ESTE MÓDULO
# --------------------------
# Toda feature se calcula EXCLUSIVAMENTE con partidos anteriores al que se
# está prediciendo. Ni el propio partido, ni ninguno posterior.
#
# Suena obvio y es el error más común del ML aplicado a deportes: usar las
# faltas o las amarillas DEL PARTIDO para predecir la expulsión DE ESE
# PARTIDO. En validación da métricas espectaculares. En producción no sirve
# para nada, porque cuando querés predecir, ese partido todavía no se jugó.
#
# El test `test_ninguna_feature_depende_del_resultado_del_propio_partido`
# verifica esta regla de forma mecánica: cambia el resultado de un partido y
# exige que ninguna feature se mueva.
#
# CONVENCIÓN: todas las funciones asumen el DataFrame ORDENADO por fecha.
# `construir_features` se encarga de ordenarlo.
# =============================================================================

import numpy as np
import pandas as pd

from ml.leakage import COLUMNAS_POST_PARTIDO

TARGET = "hubo_expulsion"

# Estadísticas de partido que tienen versión local y visitante, y de las que
# se construye el historial de agresividad de cada equipo.
_ESTADISTICAS_EQUIPO = {
    "faltas":    ("faltas_local",    "faltas_visitante"),
    "amarillas": ("amarillas_local", "amarillas_visitante"),
    "rojas":     ("rojas_local",     "rojas_visitante"),
}

# La regla de que es leakage vive en ml/leakage.py, definida UNA vez y
# compartida con quality/contracts.py y con el modelo de goles. Aca se le suma
# solo lo que es especifico de este modulo: andamiaje interno y la columna que
# decidimos no usar por falta de senal (ver `arbitro` en dataset.METADATA).
_COLUMNAS_CON_LEAKAGE = [*COLUMNAS_POST_PARTIDO, "_i", "_par", "arbitro"]


def agregar_target(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega el target binario: ¿hubo al menos una expulsión en el partido?

    Los nulos se tratan como cero: en football-data.co.uk la ausencia de dato
    de tarjetas corresponde a partidos sin registro de tarjetas, no a un
    partido con expulsiones desconocidas.
    """
    df = df.copy()
    df["rojas_total"] = (
        pd.to_numeric(df["rojas_local"], errors="coerce").fillna(0)
        + pd.to_numeric(df["rojas_visitante"], errors="coerce").fillna(0)
    )
    df[TARGET] = (df["rojas_total"] > 0).astype(int)
    return df


def tasa_historica_suavizada(
    df: pd.DataFrame,
    columna_grupo: str,
    columna_evento: str,
    k: float,
    prior: float,
) -> pd.Series:
    """Tasa histórica de un evento por grupo, con ventana expansiva y suavizado.

    Para cada fila calcula la tasa del evento en las filas ANTERIORES del
    mismo grupo (un árbitro, un cruce entre dos equipos):

        tasa = (eventos_previos + k * prior) / (partidos_previos + k)

    Las dos mitades de la fórmula resuelven dos problemas distintos:

    - La ventana expansiva ("previos") evita el leakage: el partido en curso
      jamás entra en su propia feature.
    - El suavizado hacia `prior` evita el sobreajuste de grupos chicos: un
      árbitro con 2 partidos dirigidos y 1 expulsión no "tiene" una tasa del
      50%; con k=20 queda pegado a la media global, que es lo honesto.

    `k` es la cantidad de partidos ficticios en el prior: cuanto más alto,
    más historia hace falta para que el grupo se despegue de la media.

    Args:
        df: DataFrame ORDENADO POR FECHA.
        columna_grupo: columna que define el grupo (ej. 'arbitro').
        columna_evento: columna binaria del evento (ej. el target).
        k: fuerza del suavizado. k=0 devuelve la tasa cruda.
        prior: valor hacia el que se suaviza. DEBE calcularse solo con datos
            de entrenamiento, o el suavizado mismo introduce leakage.

    Returns:
        Serie alineada al índice de `df`.
    """
    grupo  = df.groupby(columna_grupo, dropna=False)[columna_evento]
    evento = pd.to_numeric(df[columna_evento], errors="coerce").fillna(0)

    # cumsum incluye la fila actual: restarla deja solo las anteriores.
    eventos_previos  = grupo.cumsum() - evento
    partidos_previos = df.groupby(columna_grupo, dropna=False).cumcount()

    numerador   = eventos_previos + k * prior
    denominador = partidos_previos + k

    tasa = numerador / denominador
    # Sin historia y sin suavizado (k=0) no hay nada que estimar: queda el prior.
    return tasa.where(denominador > 0, prior)


def media_historica(df: pd.DataFrame, columna_grupo: str, columna: str) -> pd.Series:
    """Media de una columna en los partidos ANTERIORES del mismo grupo.

    Misma lógica de ventana expansiva que `tasa_historica_suavizada`, sin
    suavizado: se usa para magnitudes continuas (faltas y amarillas que suele
    pitar un árbitro), donde el sobreajuste de grupos chicos es menos severo.
    """
    valores = pd.to_numeric(df[columna], errors="coerce")
    suma_previa = (
        valores.groupby(df[columna_grupo], dropna=False).cumsum() - valores
    )
    n_previos = df.groupby(columna_grupo, dropna=False).cumcount()
    return (suma_previa / n_previos).where(n_previos > 0, np.nan)


def _formato_largo(df: pd.DataFrame, estadisticas: dict | None = None) -> pd.DataFrame:
    """Convierte el DataFrame a una fila por (partido, equipo).

    Un equipo juega de local y de visitante, pero su historia de agresividad
    es UNA SOLA. Sin este paso habría que llevar dos historias paralelas por
    equipo, que es justamente el bug clásico de este feature engineering.
    """
    partes = []
    for es_local, (columna_equipo, indice) in {
        True:  ("equipo_local", 0),
        False: ("equipo_visitante", 1),
    }.items():
        parte = pd.DataFrame({
            "_i":       df["_i"].values,
            "fecha":    df["_fecha"].values,
            "equipo":   df[columna_equipo].values,
            "es_local": es_local,
        })
        for stat, columnas in (estadisticas or _ESTADISTICAS_EQUIPO).items():
            parte[stat] = pd.to_numeric(df[columnas[indice]], errors="coerce").values
        partes.append(parte)

    return (
        pd.concat(partes, ignore_index=True)
        .sort_values(["fecha", "_i"], kind="stable")
        .reset_index(drop=True)
    )


def medias_moviles_por_equipo(
    df: pd.DataFrame,
    ventanas: tuple[int, ...] = (5, 10),
    estadisticas: dict | None = None,
) -> pd.DataFrame:
    """Medias móviles de faltas, amarillas y rojas de los últimos N partidos.

    El `shift(1)` antes del `rolling` es lo que garantiza que el partido en
    curso quede afuera de su propia media. Sin historia devuelve NaN, y eso
    está bien: HistGradientBoostingClassifier maneja NaN nativamente, así que
    no hay que inventar imputaciones que ensucien la señal.

    Returns:
        DataFrame con columnas '{stat}_prom{N}_local' y '{stat}_prom{N}_visitante'.
    """
    df = df.copy()
    if "_i" not in df.columns:
        df["_i"] = np.arange(len(df))
    if "_fecha" not in df.columns:
        df["_fecha"] = pd.to_datetime(df["fecha_partido"], errors="coerce")

    estadisticas = estadisticas or _ESTADISTICAS_EQUIPO
    largo = _formato_largo(df, estadisticas)
    por_equipo = largo.groupby("equipo", dropna=False)

    for stat in estadisticas:
        for ventana in ventanas:
            largo[f"{stat}_prom{ventana}"] = por_equipo[stat].transform(
                lambda s, v=ventana: s.shift(1).rolling(v, min_periods=1).mean()
            )

    columnas_calculadas = [
        f"{stat}_prom{v}" for stat in estadisticas for v in ventanas
    ]

    resultado = df.copy()
    for es_local, sufijo in [(True, "local"), (False, "visitante")]:
        lado = (
            largo[largo["es_local"] == es_local]
            .set_index("_i")[columnas_calculadas]
            .add_suffix(f"_{sufijo}")
        )
        resultado = resultado.join(lado, on="_i")

    return resultado


def tasa_historica_por_equipo(
    df: pd.DataFrame, prior: float, k: float = 20.0, columna_evento: str | None = None
) -> tuple[pd.Series, pd.Series]:
    """Tasa histórica de expulsión de cada equipo, para local y visitante.

    Reemplaza a los dummies de equipo, que eran una fuente de memorización:
    con ~160 equipos y muchos de ellos presentes solo un par de temporadas
    (ascensos y descensos), la logística les asignaba coeficientes enormes
    ajustados sobre 30-60 partidos. Es la misma trampa que tenía el árbitro.

    Igual que en `medias_moviles_por_equipo`, la historia de un equipo cruza
    sus partidos de local Y de visitante: es una sola trayectoria.

    Returns:
        (tasa_local, tasa_visitante), alineadas al índice de `df`.
    """
    df = df.copy()
    if "_i" not in df.columns:
        df["_i"] = np.arange(len(df))
    if "_fecha" not in df.columns:
        df["_fecha"] = pd.to_datetime(df["fecha_partido"], errors="coerce")

    evento = columna_evento or TARGET
    largo = pd.concat([
        pd.DataFrame({
            "_i":     df["_i"].values,
            "fecha":  df["_fecha"].values,
            "equipo": df[columna].values,
            "es_local": es_local,
            evento:   df[evento].values,
        })
        for columna, es_local in [("equipo_local", True), ("equipo_visitante", False)]
    ], ignore_index=True).sort_values(["fecha", "_i"], kind="stable").reset_index(drop=True)

    largo["tasa"] = tasa_historica_suavizada(largo, "equipo", evento, k=k, prior=prior)

    salida = []
    for es_local in (True, False):
        lado = largo[largo["es_local"] == es_local].set_index("_i")["tasa"]
        salida.append(df["_i"].map(lado))
    return salida[0], salida[1]


_CUOTAS_1X2_BET365 = ("cuota_local", "cuota_empate", "cuota_visitante")


def features_de_cuotas(
    df: pd.DataFrame, columnas: tuple[str, str, str] = _CUOTAS_1X2_BET365
) -> pd.DataFrame:
    """Convierte las cuotas de Bet365 en probabilidades implícitas normalizadas.

    La inversa de la cuota es la probabilidad implícita, pero las tres suman
    más de 1: ese exceso es el margen de la casa de apuestas. Normalizar lo
    elimina y deja probabilidades comparables entre partidos.

    Estas features no son solo contexto: las cuotas son la predicción del
    mercado, y sirven además como baseline contra el cual medirse.

    `paridad` mide qué tan parejo es el partido. La hipótesis es que un
    partido parejo genera más fricción — y por lo tanto más expulsiones — que
    una goleada anunciada.
    """
    df = df.copy()
    inversas = {
        destino: 1.0 / pd.to_numeric(df[origen], errors="coerce")
        for destino, origen in zip(
            ("prob_local", "prob_empate", "prob_visitante"), columnas
        )
    }
    total = sum(inversas.values())
    for nombre, valor in inversas.items():
        df[nombre] = valor / total

    df["paridad"] = 1.0 - (df["prob_local"] - df["prob_visitante"]).abs()
    return df


def _clave_cruce(df: pd.DataFrame) -> pd.Series:
    """Clave simétrica del enfrentamiento: A-vs-B y B-vs-A son el mismo cruce."""
    a = df["equipo_local"].astype(str)
    b = df["equipo_visitante"].astype(str)
    return np.where(a < b, a + "|" + b, b + "|" + a)


def _agregar_features_de_arbitro(
    df: pd.DataFrame, prior: float, k_suavizado: float
) -> pd.DataFrame:
    """Features históricas del árbitro. NO se construyen por defecto.

    Se conservan para poder re-testear la hipótesis, pero la evidencia dice
    que no sirven:

    - Fiabilidad split-half de la tasa de expulsión por árbitro: -0.04. La
      tendencia de un árbitro en la primera mitad de los datos NO predice la
      segunda (P Tierney 6.3% -> 12.6%, M Dean 19.4% -> 13.7%).
    - Una simulación binomial con la tasa base real (11%) y ~130 partidos por
      árbitro produce por PURO AZAR un spread de 8.4 puntos entre el que más
      y el que menos expulsa. El spread observado, 13.1, apenas lo supera.
    - La feature honesta correlaciona +0.003 con el target y sus terciles no
      son monótonos.

    Lo que parecía el efecto del árbitro era regresión a la media. Además el
    dato solo existe en la Premier League (0% en las otras cuatro ligas).
    """
    df = df.copy()
    df["arbitro_tasa_exp"] = tasa_historica_suavizada(
        df, "arbitro", TARGET, k=k_suavizado, prior=prior
    )
    df["_faltas_partido"] = (
        pd.to_numeric(df["faltas_local"], errors="coerce")
        + pd.to_numeric(df["faltas_visitante"], errors="coerce")
    )
    df["_amarillas_partido"] = (
        pd.to_numeric(df["amarillas_local"], errors="coerce")
        + pd.to_numeric(df["amarillas_visitante"], errors="coerce")
    )
    df["arbitro_prom_faltas"]    = media_historica(df, "arbitro", "_faltas_partido")
    df["arbitro_prom_amarillas"] = media_historica(df, "arbitro", "_amarillas_partido")
    return df


def construir_features(
    df: pd.DataFrame,
    prior: float,
    ventanas: tuple[int, ...] = (5, 10),
    k_suavizado: float = 20.0,
    incluir_arbitro: bool = False,
) -> pd.DataFrame:
    """Construye la matriz de features para predecir expulsiones.

    La columna `liga` sobrevive como categórica y es una de las features con
    señal real: la tasa de expulsión por liga es un rasgo estable en el tiempo
    (fiabilidad split-half +0.88) y va de 11% en la Premier a 22% en Ligue 1.

    Args:
        df: partidos de bronze, con estadísticas y cuotas.
        prior: tasa base de expulsión. DEBE venir del split de entrenamiento.
        ventanas: tamaños de las medias móviles por equipo.
        k_suavizado: fuerza del suavizado de las tasas históricas.
        incluir_arbitro: construir las features del árbitro. Por defecto False;
            ver `_agregar_features_de_arbitro` para la evidencia.

    Returns:
        DataFrame con una fila por partido: features + TARGET, sin ninguna
        columna que describa lo ocurrido en el propio partido.
    """
    df = agregar_target(df).copy()
    df["_fecha"] = pd.to_datetime(df["fecha_partido"], errors="coerce")
    df = df.sort_values(["_fecha"], kind="stable").reset_index(drop=True)
    df["_i"] = np.arange(len(df))

    if incluir_arbitro and "arbitro" in df.columns:
        df = _agregar_features_de_arbitro(df, prior, k_suavizado)

    # --- Tasa histórica de cada equipo (reemplaza a los dummies) ---
    tasa_local, tasa_visitante = tasa_historica_por_equipo(df, prior=prior, k=k_suavizado)
    df["equipo_tasa_exp_local"]     = tasa_local
    df["equipo_tasa_exp_visitante"] = tasa_visitante

    # --- Historia de agresividad de cada equipo ---
    df = medias_moviles_por_equipo(df, ventanas=ventanas)

    # --- Contexto del partido según el mercado ---
    df = features_de_cuotas(df)

    # --- Historial del cruce ---
    df["_par"] = _clave_cruce(df)
    df["h2h_tasa_exp"] = tasa_historica_suavizada(
        df, "_par", TARGET, k=k_suavizado, prior=prior
    )

    # --- Temporales ---
    df["mes"] = df["_fecha"].dt.month.astype("Int64")
    df["fecha_num"] = (
        df.groupby("temporada", dropna=False)["_fecha"].rank(method="dense").astype("Int64")
    )

    descartar = [c for c in _COLUMNAS_CON_LEAKAGE if c in df.columns]
    descartar += [c for c in df.columns if c.startswith("_")]
    return df.drop(columns=descartar)
