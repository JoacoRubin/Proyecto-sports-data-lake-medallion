# =============================================================================
# tests/test_ml_dataset.py — Tests de la preparación de la matriz y del split.
#
# El split es la otra mitad del problema de leakage. Ya garantizamos que
# ninguna FEATURE mira el futuro; acá garantizamos que ningún FOLD lo haga.
#
# Con 46-350 positivos por temporada, un único test set da métricas con un
# intervalo de confianza enorme. Por eso se usa validación de ORIGEN MÓVIL:
# entrenar con todo lo anterior, evaluar en la temporada siguiente, avanzar.
# Da 7 evaluaciones en vez de 1, y sigue siendo estrictamente temporal.
# =============================================================================

import numpy as np
import pandas as pd
import pytest

from ml.dataset import (
    CATEGORICAS,
    columnas_features,
    preparar_matriz,
    splits_origen_movil,
)
from ml.features import TARGET


def _df_features(temporadas=("2018-2019", "2019-2020", "2020-2021", "2021-2022"), por_temporada=10):
    filas = []
    for t, temporada in enumerate(temporadas):
        for i in range(por_temporada):
            filas.append({
                "id_evento": f"FD-{t}{i}", "nombre_evento": "A vs B",
                "temporada": temporada, "liga": "Spanish La Liga", "id_liga": "FD-SP1",
                "equipo_local": "A", "id_equipo_local": "FD-a",
                "equipo_visitante": "B", "id_equipo_visitante": "FD-b",
                "fecha_partido": f"20{18+t}-01-{i+1:02d}",
                "hora_partido": "20:00", "estadio": None, "estado": "FT",
                "timestamp_extraccion": "2026-01-01T00:00:00+00:00",
                "fecha_extraccion": "2026-01-01",
                "cuota_local": 2.0, "cuota_empate": 3.0, "cuota_visitante": 4.0,
                "faltas_prom5_local": 10.0 + i, "faltas_prom10_local": 11.0,
                "amarillas_prom5_local": 2.0, "amarillas_prom10_local": 2.1,
                "rojas_prom5_local": 0.1, "rojas_prom10_local": 0.2,
                "faltas_prom5_visitante": 12.0, "faltas_prom10_visitante": 12.5,
                "amarillas_prom5_visitante": 2.2, "amarillas_prom10_visitante": 2.3,
                "rojas_prom5_visitante": 0.1, "rojas_prom10_visitante": 0.15,
                "prob_local": 0.45, "prob_empate": 0.30, "prob_visitante": 0.25,
                "paridad": 0.8, "h2h_tasa_exp": 0.17, "mes": 1, "fecha_num": i + 1,
                TARGET: 1 if i % 5 == 0 else 0,
            })
    return pd.DataFrame(filas)


# --- Selección de columnas -------------------------------------------------
def test_las_features_excluyen_el_target():
    assert TARGET not in columnas_features(_df_features())


def test_las_features_excluyen_metadata_e_identificadores():
    """Un id no es una feature: memorizarlo es sobreajustar."""
    cols = set(columnas_features(_df_features()))
    prohibidas = {
        "id_evento", "nombre_evento", "id_liga", "id_equipo_local",
        "id_equipo_visitante", "fecha_partido", "estado", "estadio",
        "timestamp_extraccion", "fecha_extraccion", "temporada",
    }
    assert not (cols & prohibidas), f"metadata colada como feature: {cols & prohibidas}"


def test_las_features_excluyen_las_cuotas_crudas():
    """Ya están como probabilidades normalizadas: la cuota cruda es redundante."""
    cols = set(columnas_features(_df_features()))
    assert not (cols & {"cuota_local", "cuota_empate", "cuota_visitante"})


def test_las_features_incluyen_las_construidas():
    cols = set(columnas_features(_df_features()))
    for esperada in ("faltas_prom5_local", "paridad", "h2h_tasa_exp", "liga", "mes"):
        assert esperada in cols, f"falta la feature '{esperada}'"


# --- Matriz ----------------------------------------------------------------
def test_preparar_matriz_devuelve_X_e_y_alineados():
    df = _df_features()
    X, y = preparar_matriz(df)
    assert len(X) == len(y) == len(df)
    assert list(X.index) == list(y.index)


def test_las_categoricas_quedan_con_dtype_category():
    """HistGradientBoostingClassifier las detecta por dtype, sin one-hot."""
    X, _ = preparar_matriz(_df_features())
    for col in CATEGORICAS:
        if col in X.columns:
            assert isinstance(X[col].dtype, pd.CategoricalDtype), f"'{col}' no es category"


def test_el_target_es_entero_binario():
    _, y = preparar_matriz(_df_features())
    assert set(y.unique()).issubset({0, 1})


# --- Split de origen móvil -------------------------------------------------
def test_genera_un_fold_por_temporada_evaluable():
    """4 temporadas y mínimo 2 de entrenamiento -> 2 folds (la 3ª y la 4ª)."""
    folds = splits_origen_movil(_df_features(), min_temporadas_train=2)
    assert len(folds) == 2


def test_cada_fold_entrena_solo_con_el_pasado():
    """LA garantía del split: ninguna fecha de train puede ser >= a una de test."""
    df = _df_features()
    fechas = pd.to_datetime(df["fecha_partido"])
    for train_idx, test_idx in splits_origen_movil(df, min_temporadas_train=2):
        assert fechas.iloc[train_idx].max() < fechas.iloc[test_idx].min()


def test_train_y_test_no_se_solapan():
    df = _df_features()
    for train_idx, test_idx in splits_origen_movil(df, min_temporadas_train=2):
        assert not (set(train_idx) & set(test_idx))


def test_el_test_de_cada_fold_es_una_sola_temporada():
    df = _df_features()
    for _, test_idx in splits_origen_movil(df, min_temporadas_train=2):
        assert df.iloc[test_idx]["temporada"].nunique() == 1


def test_el_train_crece_en_cada_fold():
    """Origen móvil expansivo: cada fold entrena con más historia que el anterior."""
    df = _df_features()
    tamanios = [len(tr) for tr, _ in splits_origen_movil(df, min_temporadas_train=2)]
    assert tamanios == sorted(tamanios) and len(set(tamanios)) == len(tamanios)


def test_falla_si_no_hay_temporadas_suficientes():
    df = _df_features(temporadas=("2020-2021",))
    with pytest.raises(ValueError, match="[Tt]emporadas"):
        splits_origen_movil(df, min_temporadas_train=3)


# =============================================================================
# Las cuotas CRUDAS se excluyen por PATRON, no por lista.
#
# Regresion real: al agregar las cuotas de mercado a bronze para el modelo de
# goles, se colaron como features del modelo de EXPULSIONES, porque METADATA
# nombraba una por una las viejas de Bet365 y nadie actualizo la lista.
#
# Una lista que hay que acordarse de mantener es una lista que va a quedar
# desactualizada. El prefijo `cuota_` es la regla: las cuotas entran
# normalizadas como probabilidad, o no entran.
# =============================================================================

def test_ninguna_cuota_cruda_llega_a_las_features():
    df = _df_features()
    df["cuota_over25"] = 1.85
    df["cuota_under25"] = 1.95
    df["cuota_local_mercado"] = 1.70
    df["cuota_inventada_del_futuro"] = 2.0   # una que todavia no existe

    coladas = [c for c in columnas_features(df) if c.startswith("cuota_")]
    assert not coladas, f"cuotas crudas coladas como features: {coladas}"


def test_las_probabilidades_normalizadas_SI_pasan():
    """Se excluye la cuota cruda, no la probabilidad derivada de ella."""
    cols = columnas_features(_df_features())
    assert "prob_local" in cols
    assert "paridad" in cols


def test_preparar_matriz_tampoco_las_deja_pasar():
    df = _df_features()
    df["cuota_over25"] = 1.85
    X, _ = preparar_matriz(df)
    assert not [c for c in X.columns if c.startswith("cuota_")]
