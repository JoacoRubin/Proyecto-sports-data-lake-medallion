# =============================================================================
# tests/test_ml_tuning.py — Tests de la busqueda ANIDADA de hiperparametros.
#
# POR QUE ANIDADA
#
# Si buscas hiperparametros mirando el mismo set con el que despues reportas
# las metricas, esas metricas quedan optimistas: elegiste la configuracion que
# mejor le va A ESE SET. Es una forma sutil de leakage y es de las mas comunes
# en proyectos que "dieron bien".
#
# La busqueda anidada separa las dos preguntas:
#   - Bucle INTERNO (solo sobre train): que hiperparametros elijo?
#   - Bucle EXTERNO: cuanto rinde ese procedimiento en datos no vistos?
#
# Y los dos bucles son TEMPORALES: nunca se entrena con posterior a lo que se
# evalua, ni adentro ni afuera.
# =============================================================================

import numpy as np
import pandas as pd
import pytest

from ml.tuning import buscar_hiperparametros, evaluar_con_tuning_anidado
from tests.test_ml_dataset import _df_features


def _fabrica(**params):
    from sklearn.tree import DecisionTreeClassifier
    return DecisionTreeClassifier(random_state=0, **params)


_GRILLA = {"max_depth": [2, 4], "min_samples_leaf": [5, 20]}


def _matriz(por_temporada=80):
    """6 temporadas para que haya lugar a bucle interno y externo.

    Se descarta `liga`: estos tests ejercitan el ANIDAMIENTO, no el manejo de
    categoricas (que es responsabilidad de ml/dataset.py). Con un arbol pelado
    como estimador de prueba, dejarla solo agregaria ruido al fixture.
    """
    temporadas = tuple(f"20{15+i}-20{16+i}" for i in range(6))
    df = _df_features(temporadas=temporadas, por_temporada=por_temporada).drop(columns=["liga"])
    rng = np.random.default_rng(0)
    df["paridad"] = rng.random(len(df))
    return df


# --- Busqueda interna ------------------------------------------------------
def test_devuelve_una_combinacion_de_la_grilla():
    mejor = buscar_hiperparametros(_matriz(), _fabrica, _GRILLA, min_temporadas_train=2)
    assert mejor["max_depth"] in _GRILLA["max_depth"]
    assert mejor["min_samples_leaf"] in _GRILLA["min_samples_leaf"]


def test_la_busqueda_es_determinista():
    a = buscar_hiperparametros(_matriz(), _fabrica, _GRILLA, min_temporadas_train=2)
    b = buscar_hiperparametros(_matriz(), _fabrica, _GRILLA, min_temporadas_train=2)
    assert a == b


def test_una_grilla_de_un_solo_punto_devuelve_ese_punto():
    grilla = {"max_depth": [3]}
    assert buscar_hiperparametros(_matriz(), _fabrica, grilla, min_temporadas_train=2) == {"max_depth": 3}


def test_falla_si_no_hay_temporadas_para_el_bucle_interno():
    df = _df_features(temporadas=("2020-2021", "2021-2022"), por_temporada=30)
    with pytest.raises(ValueError, match="[Tt]emporadas"):
        buscar_hiperparametros(df, _fabrica, _GRILLA, min_temporadas_train=3)


# --- Anidamiento -----------------------------------------------------------
def test_devuelve_una_fila_por_fold_externo():
    res = evaluar_con_tuning_anidado(
        _matriz(), _fabrica, _GRILLA, min_temporadas_train=3, min_temporadas_interno=2
    )
    assert len(res) == 3          # 6 temporadas, 3 de train minimo -> 3 folds


def test_registra_los_hiperparametros_elegidos_en_cada_fold():
    """Que la eleccion cambie entre folds es informacion, no ruido."""
    res = evaluar_con_tuning_anidado(
        _matriz(), _fabrica, _GRILLA, min_temporadas_train=3, min_temporadas_interno=2
    )
    assert "hiperparametros" in res.columns
    assert res["hiperparametros"].notna().all()


def test_trae_las_metricas_del_fold_externo():
    res = evaluar_con_tuning_anidado(
        _matriz(), _fabrica, _GRILLA, min_temporadas_train=3, min_temporadas_interno=2
    )
    for col in ("pr_auc", "lift_pr_auc", "log_loss", "brier", "temporada_test"):
        assert col in res.columns


def test_la_busqueda_interna_NUNCA_ve_la_temporada_de_test(monkeypatch):
    """LA garantia del anidamiento, verificada de forma mecanica.

    Se intercepta la busqueda y se registra que temporadas recibio. Ninguna
    puede coincidir con la temporada evaluada en ese fold externo.
    """
    vistas = []
    import ml.tuning as tuning
    original = tuning.buscar_hiperparametros

    def espia(matriz, fabrica, grilla, min_temporadas_train, **kw):
        vistas.append((set(matriz["temporada"].unique()), None))
        return original(matriz, fabrica, grilla, min_temporadas_train, **kw)

    monkeypatch.setattr(tuning, "buscar_hiperparametros", espia)

    res = evaluar_con_tuning_anidado(
        _matriz(), _fabrica, _GRILLA, min_temporadas_train=3, min_temporadas_interno=2
    )

    for (temporadas_busqueda, _), temporada_test in zip(vistas, res["temporada_test"]):
        assert temporada_test not in temporadas_busqueda, (
            f"la busqueda interna vio '{temporada_test}', que es el test de ese fold"
        )
