# =============================================================================
# ml/calibration.py — Calibración con holdout TEMPORAL.
#
# ⚠️  RESULTADO NEGATIVO: NO SE USA EN PRODUCCIÓN. Medido, no supuesto.
# ---------------------------------------------------------------------
# Se implementó para corregir la sobreconfianza en la cola del HGB. Medido
# sobre los 7 folds de origen movil, empeora TODO en los dos modelos:
#
#   logistica sin calibrar   PR-AUC 0.2134 | log-loss 0.4462 | Brier 0.13810
#   logistica + isotonica    PR-AUC 0.2069 | log-loss 0.4687 | Brier 0.13892
#   logistica + sigmoide     PR-AUC 0.2037 | log-loss 0.4486 | Brier 0.13867
#
#   HGB sin calibrar         PR-AUC 0.2042 | log-loss 0.4481 | Brier 0.13855
#   HGB + isotonica          PR-AUC 0.1939 | log-loss 0.4719 | Brier 0.13967
#   HGB + sigmoide           PR-AUC 0.1902 | log-loss 0.4505 | Brier 0.13922
#
# Dos causas:
#
#   1. La regresion logistica optimiza log-loss directamente: sus
#      probabilidades ya estan calibradas por construccion. No habia defecto
#      que corregir en el modelo de produccion. El problema medido estaba en
#      el HGB, que no es el que va a produccion.
#   2. El calibrador se lleva el 20% del entrenamiento como holdout. Con ~2100
#      positivos, ese costo domina cualquier ganancia de calibracion.
#
# El modulo queda: esta testeado, es correcto, y la tecnica es la adecuada
# cuando el modelo base SI esta descalibrado (un SVM, un naive bayes, un
# random forest sin ajustar). Aca no lo estaba. Borrarlo seria perder la
# evidencia de que se probo.
#
# QUE HACE, CUANDO SE USA
# -----------------------
# El modelo calibra bien en el grueso pero se sobreconfía en la cola:
#
#     bin con n=8021 -> predice 14,9% | ocurre 15,5%   (bien)
#     bin con n=3710 -> predice 23,6% | ocurre 20,0%
#     bin con n=253  -> predice 32,1% | ocurre 21,7%   (sobreconfiado)
#
# Para un modelo cuyo producto ES la probabilidad, eso importa más que el
# ranking. La regresión isotónica aprende el mapeo monótono entre lo que el
# modelo dice y lo que realmente pasa.
#
# POR QUÉ NO `CalibratedClassifierCV`
# -----------------------------------
# Porque su cv por defecto es un KFold estratificado: mezcla el tiempo. El
# calibrador terminaría ajustado con partidos posteriores a los que predice —
# leakage por la puerta de atrás, en el último paso del pipeline.
#
# Acá el holdout de calibración es siempre el TRAMO FINAL del entrenamiento:
# el modelo base aprende del pasado, el calibrador se ajusta sobre el tramo
# más reciente, y ninguno de los dos ve el futuro.
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.isotonic import IsotonicRegression


class _CalibradorSigmoide:
    """Platt scaling: una logística sobre la probabilidad del modelo base.

    Estrictamente monótona, así que preserva el orden exacto de las
    predicciones y el ranking (PR-AUC, ROC-AUC) queda intacto.
    """

    def fit(self, prob, y):
        from sklearn.linear_model import LogisticRegression
        self._lr = LogisticRegression(max_iter=1000).fit(
            np.asarray(prob).reshape(-1, 1), y
        )
        return self

    def transform(self, prob):
        return self._lr.predict_proba(np.asarray(prob).reshape(-1, 1))[:, 1]


def _ajustar_calibrador(metodo: str, prob, y):
    """Devuelve el calibrador ajustado según el método pedido."""
    if metodo == "isotonic":
        return IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(prob, y)
    if metodo == "sigmoid":
        return _CalibradorSigmoide().fit(prob, y)
    raise ValueError(f"Metodo de calibracion desconocido: '{metodo}'. Use 'isotonic' o 'sigmoid'.")


class CalibradorTemporal(BaseEstimator, ClassifierMixin):
    """Envuelve un clasificador y calibra su probabilidad con isotónica temporal.

    Se comporta como cualquier estimador de sklearn (`fit` / `predict_proba`),
    así que entra sin cambios en `evaluar_por_folds` y en el registro.

    ISOTONICA vs SIGMOIDE — es un trade-off real, no una preferencia:

    - `isotonic`: función ESCALONADA. Más flexible, corrige distorsiones de
      cualquier forma. Pero mapea muchas probabilidades distintas al mismo
      valor, y esos empates DEGRADAN el ranking. Medido en el test sintético:
      PR-AUC 0.401 -> 0.370.
    - `sigmoid` (Platt): estrictamente monótona y continua. Preserva el orden
      EXACTO, así que el PR-AUC no se mueve ni un decimal. A cambio solo puede
      corregir distorsiones con forma de sigmoide.

    Si el modelo ya rankea bien y solo está mal escalado, sigmoide. Si la
    distorsión es irregular, isotónica y se paga el ranking.

    Args:
        estimador_base: clasificador a calibrar.
        fraccion_calibracion: porción FINAL del entrenamiento reservada para
            ajustar el calibrador. 0.2 deja historia suficiente para el modelo
            base sin quedarse sin positivos en el holdout.
        metodo: "isotonic" o "sigmoid".
    """

    def __init__(
        self,
        estimador_base=None,
        fraccion_calibracion: float = 0.2,
        metodo: str = "isotonic",
    ):
        self.estimador_base = estimador_base
        self.fraccion_calibracion = fraccion_calibracion
        self.metodo = metodo

    def fit(self, X, y):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        y = pd.Series(y) if not isinstance(y, pd.Series) else y

        corte = int(len(X) * (1 - self.fraccion_calibracion))
        if corte < 1 or corte >= len(X):
            raise ValueError(
                f"fraccion_calibracion={self.fraccion_calibracion} no deja un "
                f"corte valido para {len(X)} filas."
            )

        X_base, y_base = X.iloc[:corte], y.iloc[:corte]
        X_cal,  y_cal  = X.iloc[corte:], y.iloc[corte:]

        if y_cal.nunique() < 2:
            raise ValueError(
                "El holdout de calibracion no contiene ambas clases: la "
                "isotonica no puede ajustarse. Subi fraccion_calibracion o "
                "revisa el orden temporal de los datos."
            )

        self.base_ = clone(self.estimador_base).fit(X_base, y_base)

        prob_cal = self.base_.predict_proba(X_cal)[:, 1]
        self.calibrador_ = _ajustar_calibrador(self.metodo, prob_cal, y_cal.values)

        self.classes_        = np.array([0, 1])
        self.indice_corte_   = corte
        self.n_base_         = len(X_base)
        self.n_calibracion_  = len(X_cal)
        return self

    def predict_proba(self, X) -> np.ndarray:
        prob = self.calibrador_.transform(self.base_.predict_proba(X)[:, 1])
        prob = np.clip(prob, 0.0, 1.0)
        return np.column_stack([1.0 - prob, prob])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
