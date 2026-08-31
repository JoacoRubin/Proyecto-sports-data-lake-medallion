# =============================================================================
# tests/test_ml_features.py — Tests del feature engineering para predecir
# expulsiones.
#
# EL TEST QUE IMPORTA ES EL DE LEAKAGE.
#
# Toda feature de este módulo se calcula sobre partidos ANTERIORES. Si una
# sola mira el presente o el futuro, el modelo va a dar métricas hermosas en
# validación y no va a servir para nada en producción — y nadie se va a
# enterar hasta que sea tarde. Por eso el leakage se testea explícitamente,
# no se confía en que "lo hicimos bien".
# =============================================================================

import pandas as pd
import pytest

from ml.features import (
    TARGET,
    agregar_target,
    construir_features,
    features_de_cuotas,
    medias_moviles_por_equipo,
    tasa_historica_suavizada,
)


def _partido(fecha, local, visitante, rojas_l=0, rojas_v=0, arbitro="R1",
             faltas_l=10, faltas_v=10, amarillas_l=2, amarillas_v=2,
             cuota_l=2.0, cuota_e=3.0, cuota_v=4.0, temporada="2020-2021"):
    return {
        "fecha_partido": fecha, "temporada": temporada, "arbitro": arbitro,
        "equipo_local": local, "equipo_visitante": visitante,
        "rojas_local": rojas_l, "rojas_visitante": rojas_v,
        "faltas_local": faltas_l, "faltas_visitante": faltas_v,
        "amarillas_local": amarillas_l, "amarillas_visitante": amarillas_v,
        "cuota_local": cuota_l, "cuota_empate": cuota_e, "cuota_visitante": cuota_v,
    }


def _df(filas):
    return pd.DataFrame(filas)


# --- Target ----------------------------------------------------------------
def test_target_es_binario_hubo_expulsion():
    df = agregar_target(_df([
        _partido("2020-01-01", "A", "B", rojas_l=0, rojas_v=0),
        _partido("2020-01-02", "A", "B", rojas_l=1, rojas_v=0),
        _partido("2020-01-03", "A", "B", rojas_l=1, rojas_v=2),
    ]))
    assert list(df[TARGET]) == [0, 1, 1]


def test_target_trata_nulos_como_sin_expulsion():
    df = agregar_target(_df([_partido("2020-01-01", "A", "B", rojas_l=None, rojas_v=None)]))
    assert df[TARGET].iloc[0] == 0


# --- Tasa histórica suavizada (la feature del árbitro) ---------------------
def test_tasa_historica_del_primer_partido_es_el_prior():
    """Sin historia previa, la única información honesta es el prior."""
    df = agregar_target(_df([_partido("2020-01-01", "A", "B", arbitro="R1")]))
    tasa = tasa_historica_suavizada(df, "arbitro", TARGET, k=10, prior=0.11)
    assert tasa.iloc[0] == pytest.approx(0.11)


def test_tasa_historica_usa_solo_partidos_anteriores():
    """El partido en curso NUNCA entra en su propia feature."""
    df = agregar_target(_df([
        _partido("2020-01-01", "A", "B", arbitro="R1", rojas_l=1),   # expulsión
        _partido("2020-01-02", "C", "D", arbitro="R1", rojas_l=1),   # expulsión
        _partido("2020-01-03", "E", "F", arbitro="R1", rojas_l=0),
    ]))
    tasa = tasa_historica_suavizada(df, "arbitro", TARGET, k=0, prior=0.11)
    # k=0 -> sin suavizado, tasa cruda de los partidos previos.
    assert pd.isna(tasa.iloc[0]) or tasa.iloc[0] == pytest.approx(0.11)
    assert tasa.iloc[1] == pytest.approx(1.0)   # 1 de 1 previo
    assert tasa.iloc[2] == pytest.approx(1.0)   # 2 de 2 previos


def test_suavizado_tira_hacia_el_prior_con_poca_historia():
    """Un árbitro con 1 partido no 'tiene' 100% de expulsiones."""
    df = agregar_target(_df([
        _partido("2020-01-01", "A", "B", arbitro="R1", rojas_l=1),
        _partido("2020-01-02", "C", "D", arbitro="R1"),
    ]))
    tasa = tasa_historica_suavizada(df, "arbitro", TARGET, k=20, prior=0.10)
    # (1 evento + 20*0.10) / (1 partido + 20) = 3.0/21
    assert tasa.iloc[1] == pytest.approx(3.0 / 21.0)
    assert tasa.iloc[1] < 0.20, "con k=20 y 1 partido debe quedar cerca del prior"


def test_cada_arbitro_tiene_su_propia_historia():
    df = agregar_target(_df([
        _partido("2020-01-01", "A", "B", arbitro="R1", rojas_l=1),
        _partido("2020-01-02", "C", "D", arbitro="R2", rojas_l=0),
        _partido("2020-01-03", "E", "F", arbitro="R2", rojas_l=0),
    ]))
    tasa = tasa_historica_suavizada(df, "arbitro", TARGET, k=0, prior=0.11)
    assert tasa.iloc[2] == pytest.approx(0.0), "R2 no hereda la expulsión de R1"


# --- Medias móviles por equipo ---------------------------------------------
def test_media_movil_toma_los_partidos_previos_del_equipo():
    """El equipo A juega de local y de visitante: su historia es una sola."""
    df = agregar_target(_df([
        _partido("2020-01-01", "A", "B", faltas_l=10, faltas_v=20),
        _partido("2020-01-02", "C", "A", faltas_l=99, faltas_v=30),   # A de visitante
        _partido("2020-01-03", "A", "D", faltas_l=0,  faltas_v=0),
    ]))
    out = medias_moviles_por_equipo(df, ventanas=(2,))
    # Partido 3: A viene de cometer 10 y 30 faltas -> promedio 20.
    assert out["faltas_prom2_local"].iloc[2] == pytest.approx(20.0)


def test_media_movil_no_incluye_el_partido_en_curso():
    df = agregar_target(_df([
        _partido("2020-01-01", "A", "B", faltas_l=10, faltas_v=10),
        _partido("2020-01-02", "A", "C", faltas_l=99, faltas_v=10),
    ]))
    out = medias_moviles_por_equipo(df, ventanas=(2,))
    # Si el 99 del partido 2 entrara en su propia feature, el promedio no seria 10.
    assert out["faltas_prom2_local"].iloc[1] == pytest.approx(10.0)


def test_media_movil_del_primer_partido_es_nula():
    """Sin historia no se inventa un valor: NaN, que HistGB maneja nativamente."""
    df = agregar_target(_df([_partido("2020-01-01", "A", "B")]))
    out = medias_moviles_por_equipo(df, ventanas=(5,))
    assert pd.isna(out["faltas_prom5_local"].iloc[0])
    assert pd.isna(out["faltas_prom5_visitante"].iloc[0])


# --- Cuotas ----------------------------------------------------------------
def test_cuotas_se_convierten_en_probabilidades_normalizadas():
    """1/cuota es la probabilidad implícita; suman >1 por el margen de la casa."""
    df = _df([_partido("2020-01-01", "A", "B", cuota_l=2.0, cuota_e=4.0, cuota_v=4.0)])
    out = features_de_cuotas(df)
    total = out["prob_local"].iloc[0] + out["prob_empate"].iloc[0] + out["prob_visitante"].iloc[0]
    assert total == pytest.approx(1.0), "deben normalizarse a 1"
    assert out["prob_local"].iloc[0] > out["prob_visitante"].iloc[0]


def test_paridad_es_mayor_cuando_el_partido_es_parejo():
    parejo   = features_de_cuotas(_df([_partido("2020-01-01","A","B",cuota_l=3.0,cuota_e=3.0,cuota_v=3.0)]))
    desparejo = features_de_cuotas(_df([_partido("2020-01-01","A","B",cuota_l=1.1,cuota_e=8.0,cuota_v=20.0)]))
    assert parejo["paridad"].iloc[0] > desparejo["paridad"].iloc[0]


def test_cuotas_faltantes_no_revientan():
    df = _df([_partido("2020-01-01", "A", "B", cuota_l=None, cuota_e=None, cuota_v=None)])
    out = features_de_cuotas(df)
    assert pd.isna(out["prob_local"].iloc[0])


# =============================================================================
# EL TEST CENTRAL: ninguna feature puede depender del presente ni del futuro.
# =============================================================================
def _dataset_largo(resultado_ultimo=0):
    """12 partidos con el mismo árbitro y equipos repetidos."""
    filas, equipos = [], ["A", "B", "C", "D"]
    for i in range(12):
        local     = equipos[i % 4]
        visitante = equipos[(i + 1) % 4]
        rojas = 1 if i in (2, 5, 7) else 0
        if i == 11:
            rojas = resultado_ultimo
        filas.append(_partido(
            f"2020-01-{i+1:02d}", local, visitante,
            rojas_l=rojas, arbitro="R1", faltas_l=10 + i, faltas_v=12,
        ))
    return _df(filas)


def test_ninguna_feature_depende_del_resultado_del_propio_partido():
    """LA prueba de leakage.

    Se calculan las features dos veces cambiando SOLO el resultado del último
    partido. Si alguna feature de cualquier fila cambia, esa feature está
    mirando información que en producción todavía no existe.
    """
    sin_roja  = construir_features(_dataset_largo(resultado_ultimo=0), prior=0.11)
    con_roja  = construir_features(_dataset_largo(resultado_ultimo=1), prior=0.11)

    columnas_features = [c for c in sin_roja.columns if c != TARGET]
    pd.testing.assert_frame_equal(
        sin_roja[columnas_features],
        con_roja[columnas_features],
        check_dtype=False,
        obj="cambiar el resultado del ultimo partido alteró alguna feature -> LEAKAGE",
    )


def test_cambiar_un_partido_pasado_si_afecta_las_features_posteriores():
    """Contraprueba: si NADA cambiara, las features no estarían usando la historia."""
    base = _dataset_largo()
    alterado = base.copy()
    alterado.loc[0, "rojas_local"] = 1     # cambia el PRIMER partido

    f_base     = construir_features(base, prior=0.11)
    f_alterado = construir_features(alterado, prior=0.11)

    assert not f_base["h2h_tasa_exp"].equals(f_alterado["h2h_tasa_exp"]), \
        "la tasa historica del cruce deberia reflejar la historia previa"


def test_construir_features_no_deja_columnas_del_propio_partido():
    """Las estadísticas crudas del partido no pueden sobrevivir en la matriz."""
    out = construir_features(_dataset_largo(), prior=0.11)
    prohibidas = {
        "faltas_local", "faltas_visitante", "amarillas_local", "amarillas_visitante",
        "rojas_local", "rojas_visitante", "goles_local", "goles_visitante",
    }
    filtradas = prohibidas & set(out.columns)
    assert not filtradas, f"features con leakage directo: {filtradas}"


def test_construir_features_conserva_una_fila_por_partido():
    df = _dataset_largo()
    assert len(construir_features(df, prior=0.11)) == len(df)


def test_construir_features_incluye_el_target():
    out = construir_features(_dataset_largo(), prior=0.11)
    assert TARGET in out.columns


# =============================================================================
# Árbitro: excluido por defecto.
#
# El test de fiabilidad split-half mostró que la tendencia de un árbitro no es
# un rasgo estable (correlación entre mitades: -0.04). El spread que se ve
# in-sample es el que produce el azar con ~130 partidos por árbitro sobre una
# tasa base del 11%. Además el dato solo existe en la Premier.
# =============================================================================
def test_arbitro_excluido_por_defecto():
    out = construir_features(_dataset_largo(), prior=0.11)
    assert not [c for c in out.columns if c.startswith("arbitro_")], \
        "las features de arbitro no deben construirse por defecto"


def test_arbitro_se_puede_pedir_explicitamente():
    """Se conserva la capacidad de construirlas, para poder re-testear la hipótesis."""
    out = construir_features(_dataset_largo(), prior=0.11, incluir_arbitro=True)
    assert "arbitro_tasa_exp" in out.columns


def test_liga_sobrevive_como_feature_categorica():
    """La liga SÍ es un rasgo estable (split-half +0.88): tiene que llegar al modelo."""
    df = _dataset_largo()
    df["liga"] = "Spanish La Liga"
    out = construir_features(df, prior=0.11)
    assert "liga" in out.columns


def test_features_de_arbitro_no_rompen_si_la_columna_no_existe():
    """Fuera de la Premier no hay columna 'arbitro'. No debe explotar."""
    df = _dataset_largo().drop(columns=["arbitro"])
    out = construir_features(df, prior=0.11)
    assert len(out) == len(df)


# =============================================================================
# Equipos: tasa histórica suavizada en lugar de dummies.
#
# Los coeficientes de la logística estaban dominados por equipos con pocas
# temporadas en el dataset (Hannover, Bastia, Nimes, Parma: ascendidos y
# descendidos). Con 30-60 partidos cada uno, el modelo les asignaba pesos
# enormes: memorización, no aprendizaje. Es exactamente la misma enfermedad
# que tenía el árbitro.
#
# La cura es la que ya usamos para el h2h: target encoding con ventana
# expansiva y suavizado hacia el prior.
# =============================================================================
from ml.features import tasa_historica_por_equipo


def test_tasa_del_equipo_arranca_en_el_prior():
    df = agregar_target(_df([_partido("2020-01-01", "A", "B")]))
    local, visitante = tasa_historica_por_equipo(df, prior=0.17, k=20)
    assert local.iloc[0] == pytest.approx(0.17)
    assert visitante.iloc[0] == pytest.approx(0.17)


def test_la_historia_del_equipo_cruza_local_y_visitante():
    """El equipo A juega de local y de visitante: su historia es UNA SOLA.

    Es el bug clásico de este encoding: llevar dos historias paralelas por
    equipo y perder la mitad de la información.
    """
    df = agregar_target(_df([
        _partido("2020-01-01", "A", "B", rojas_l=1),   # A local, hubo expulsión
        _partido("2020-01-02", "C", "A", rojas_l=1),   # A visitante, hubo expulsión
        _partido("2020-01-03", "A", "D"),              # A local otra vez
    ]))
    local, _ = tasa_historica_por_equipo(df, prior=0.17, k=0)
    # A viene de 2 partidos, los dos con expulsión -> tasa cruda 1.0
    assert local.iloc[2] == pytest.approx(1.0)


def test_la_tasa_del_equipo_no_incluye_el_partido_en_curso():
    df = agregar_target(_df([
        _partido("2020-01-01", "A", "B", rojas_l=0),
        _partido("2020-01-02", "A", "C", rojas_l=1),
    ]))
    local, _ = tasa_historica_por_equipo(df, prior=0.17, k=0)
    assert local.iloc[1] == pytest.approx(0.0), "el partido 2 no puede verse a si mismo"


def test_el_suavizado_protege_a_los_equipos_con_pocos_partidos():
    """Un equipo recien ascendido no 'tiene' 100% de expulsiones."""
    df = agregar_target(_df([
        _partido("2020-01-01", "A", "B", rojas_l=1),
        _partido("2020-01-02", "A", "C"),
    ]))
    local, _ = tasa_historica_por_equipo(df, prior=0.17, k=20)
    assert local.iloc[1] == pytest.approx((1 + 20 * 0.17) / 21)
    assert local.iloc[1] < 0.25


def test_construir_features_expone_las_tasas_de_equipo():
    out = construir_features(_dataset_largo(), prior=0.17)
    assert "equipo_tasa_exp_local" in out.columns
    assert "equipo_tasa_exp_visitante" in out.columns
