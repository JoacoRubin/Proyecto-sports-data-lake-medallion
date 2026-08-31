# ⚽ TheSportsDB — Data Lake (arquitectura medallion)

Pipeline de ingeniería de datos sobre la API pública de **TheSportsDB**, con
arquitectura **medallion** (bronze → silver → gold), calidad de datos, orquestación
y un dashboard interactivo.

- **Bronze**: ingesta multi-liga desde la API → Delta Lake (crudo).
- **Silver**: limpieza, tipado, columnas derivadas y JOIN con equipos.
- **Gold**: agregaciones → tabla de posiciones por liga.
- **Calidad**: contratos Pandera en las fronteras (fallan ruidosamente ante datos corruptos).
- **Orquestación**: flow de Prefect (tareas con reintentos y dependencias).
- **Dashboard**: Streamlit + Plotly (tabla, KPIs, bubble chart).
- **Extra**: streaming con Kafka (Aiven) que ingesta a la misma capa bronze.
- **Multi-fuente**: segunda ingesta desde football-data.co.uk (histórico + estadísticas).
- **ML**: dos modelos sobre 18.011 partidos —expulsiones y over 2.5 goles—, este último medido contra el mercado de apuestas.

---

## Capa ML — dos modelos

Predice la probabilidad de que haya **al menos una expulsión** en un partido.

ML no es una capa nueva del medallion: es un **consumidor** del data lake, igual
que el dashboard. Vive en `ml/` y se orquesta con el flow `ml-expulsiones`.

```bash
python -m pipelines.footballdata_pipeline   # ingesta historica -> bronze
python -m pipelines.ml_pipeline             # entrena, evalua y registra
```

### Resultado

| modelo | PR-AUC | lift | ROC-AUC | log-loss | Brier |
|---|---|---|---|---|---|
| tasa base | 0.1681 | 1.000 | 0.500 | 0.4533 | 0.1400 |
| **logístico** | **0.2171** | **1.291** | 0.5909 | **0.4463** | **0.1381** |
| HistGradientBoosting | 0.2103 | 1.251 | 0.5797 | 0.4485 | 0.1386 |

PR-AUC pooled: **0.2134**, IC95 `[0.2014, 0.2266]`, contra una tasa base de
`0.1681`. El intervalo no incluye la tasa base: la señal es real, no ruido.

El modelo de producción es el **logístico**. Le gana o empata al gradient
boosting con intervalos solapados, y cuando dos modelos empatan gana el simple
y explicable.

### Decisiones que sostienen el resultado

**Sin leakage, y testeado.** Toda feature usa solo partidos anteriores.
`test_ninguna_feature_depende_del_resultado_del_propio_partido` lo verifica
mecánicamente: cambia el resultado de un partido y exige que ninguna feature se
mueva. Las faltas y tarjetas del propio partido nunca entran — cuando querés
predecir, ese partido todavía no se jugó.

**Validación de origen móvil**, no un test set fijo. Se entrena con todas las
temporadas anteriores y se evalúa en la siguiente: 7 evaluaciones en vez de 1.
Un `train_test_split(shuffle=True)` sobre datos de fútbol es entrenar con el
futuro.

**No se reporta accuracy.** Con una tasa base del 17%, predecir siempre «no»
acierta el 83% sin haber aprendido nada. Se usa PR-AUC (cuyo baseline ES la
prevalencia), log-loss, Brier y bootstrap para el intervalo.

**Sin `class_weight="balanced"`.** Es el reflejo automático ante un target
desbalanceado y acá rompía la calibración: probabilidad media 0.466 contra una
tasa real de 0.168, log-loss 0.6683 vs 0.4563. El PR-AUC quedaba igual. Solo
sirve si vas a decidir con un umbral duro.

### Dos features descartadas por evidencia

**Árbitro.** In-sample parecía la señal más fuerte (M Dean 17.9% de partidos con
expulsión vs S Hooper 5.6%). El test de fiabilidad split-half la desmintió:
correlación **-0.04** entre la tasa de un árbitro en la primera mitad temporal y
la segunda. Una simulación binomial muestra que el azar ya produce ese spread
con ~130 partidos por árbitro. Era regresión a la media.

**Nombres de equipo como dummies.** Los coeficientes quedaban dominados por
equipos con pocas temporadas (ascendidos y descendidos, 30-60 partidos):
memorización. Reemplazados por tasa histórica suavizada, el lift subió de 1.21 a
1.29 y los coeficientes se volvieron legibles.

La **liga** sí sobrevivió: fiabilidad split-half **+0.88**, y va de 11% en la
Premier a 22% en Ligue 1.

### Contrato en la frontera de ML

`quality/contracts.MATRIZ_ML` valida la matriz de features antes de entrenar,
igual que bronze, silver y gold validan las suyas. El check principal no es de
tipos: **rechaza cualquier columna que describa el propio partido**. Además
verifica que las probabilidades implícitas sumen 1 y que las tasas históricas
vivan en `[0, 1]`.

En su primera corrida marcó 6 partidos sin cuotas publicadas. Resultó ser un
falso positivo —`sum()` de pandas devuelve `0.0` y no `NaN` cuando toda la fila
es nula— pero forzó a mirar y a documentar que esos 6 partidos existen.

### Tres módulos de metodología

| Módulo | Qué hace | Resultado medido |
|---|---|---|
| `ml/inspection.py` | Importancia por permutación sobre PR-AUC, con desvíos | Solo **1 de 22** features supera 2 desvíos: `paridad` |
| `ml/tuning.py` | Búsqueda anidada y temporal de hiperparámetros | **+0.0017**, dentro del ruido. Eligió `max_depth=2` en 6 de 7 folds |
| `ml/calibration.py` | Calibración isotónica y sigmoide con holdout temporal | ⚠️ **Resultado negativo**: empeora las tres métricas. No se usa |

Las importancias se registran por corrida en `data/gold/ml/importancias`.

`calibration.py` se conserva pese a no usarse: está testeado, la técnica es la
correcta cuando el modelo base *sí* está descalibrado, y borrarlo sería perder
la evidencia de que se probó y se midió.

**Sobre reducir el modelo**: con 4 features se alcanza el mismo PR-AUC que con
22 (`0.2140` vs `0.2134`). No se adoptó porque esa selección se hizo mirando el
fold de test — sería *leakage de selección*. Adoptarlo exige anidar también la
selección de features.

---

### Tracking de experimentos

En una tabla Delta de gold, `data/gold/ml/experimentos` — no en MLflow. Se
consulta con el mismo `leer_tabla_delta` que el resto del proyecto, se ve en el
mismo Streamlit, y Delta aporta versionado y time travel. Cero infraestructura
nueva.

Los modelos se versionan en `data/models/` con joblib más una ficha JSON
(features, prior, métricas, fecha). Un `.joblib` suelto es un binario que nadie
puede auditar.

---

## Modelo de goles (over 2.5) — contra el mercado

```bash
python -m pipelines.ml_pipeline     # entrena los dos modelos
```

Predice si el partido termina con más de 2.5 goles. Target **balanceado** (53%)
y evento frecuente (2,78 goles por partido contra 0,20 rojas): mucha más señal
que el modelo de expulsiones.

Pero lo que lo hace distinto es la **vara adversaria**. Hasta acá nos medíamos
contra "predecir la prevalencia", que es una vara pasiva. El mercado de
apuestas es el consenso de gente que se juega plata y lo ajusta en tiempo real.

### El experimento de tres brazos

| | ROC-AUC | log-loss | Brier |
|---|---|---|---|
| **A. Mercado solo** | **0.6157** `[0.6057, 0.6250]` | **0.6708** | **0.23906** |
| B. Nuestras features (sin cuotas de over/under) | 0.6045 `[0.5949, 0.6151]` | 0.6744 | 0.24084 |
| C. Nuestras features + mercado | 0.6119 `[0.6012, 0.6215]` | 0.6720 | 0.23962 |
| tasa base | 0.5000 | 0.6912 | 0.24912 |

**El mercado gana.** Pero:

1. **B queda muy cerca usando solo estadísticas de partido.** El mercado le saca
   a la tasa base `0.0204` de log-loss; nosotros `0.0166`. Capturamos ~85% de su
   ventaja sin mirar una sola cuota de over/under.
2. **C no le gana a A** (intervalos solapados). Nuestro modelo **no contiene
   información que el mercado no tenga ya**. Eso no es un fracaso: es eficiencia
   de mercado, medida. Un modelo tabular con 18k partidos que le ganara
   consistentemente al mercado tendría leakage.

El mercado viaja en la matriz para poder evaluarlo sobre los **mismos folds**,
pero está en `ml/dataset.METADATA`: el modelo nunca lo ve. Si lo viera,
aprendería a copiarlo y la comparación no significaría nada. Hay un test que lo
verifica.

### Lo que aporta señal

Importancia por permutación: **4 de 30** features superan dos desvíos (contra
1 de 22 en expulsiones).

```
prob_empate               0.08926 ±0.01166   ← empate probable = partido trabado
paridad                   0.01337 ±0.00570
tiros_arco_prom10_local   0.00756 ±0.00304   ← fútbol puro
prob_local                0.00738 ±0.00325
```

### El bug que casi arruina la comparación

La primera corrida dio el mercado en **ROC-AUC 0.5261** — casi azar, imposible
para un mercado real. La causa: la probabilidad del mercado se calculaba sobre
un DataFrame ordenado con `sort_values()` por defecto (quicksort, **no
estable**) mientras la matriz usaba `kind="stable"`. Mismo largo, orden
distinto, **filas desalineadas**.

Regla: nunca alinear dos DataFrames por posición si se ordenaron por separado.
Y cuando un baseline conocido da un número imposible, el error es propio.

---

## Fuentes de datos

El data lake es **multi-fuente**. Cada fuente tiene su extractor y su mapper,
pero todas cumplen el mismo contrato bronze (`quality/contracts.BRONZE_PARTIDOS`)
antes de escribir. Esa frontera es lo que permite sumar fuentes sin tocar
silver, gold ni el dashboard.

| Fuente | Ingesta | Volumen | Aporta |
|---|---|---|---|
| **TheSportsDB** (API) | batch + streaming Kafka | 5 partidos/liga con la key pública | metadatos de equipos, estadios, tiempo real |
| **football-data.co.uk** (CSV) | batch histórico | temporadas completas desde 1993 | estadísticas de partido y cuotas |

### Por qué una segunda fuente

La key pública de TheSportsDB (`3`) limita la respuesta a **5 partidos por
liga**, y el tope está en la key, no en el endpoint: `eventsseason`,
`eventsround`, `eventspastleague` y `eventsday` devuelven todos lo mismo.
Alcanza para demostrar el pipeline; no alcanza para entrenar nada.

Además, TheSportsDB solo publica **goles**. Cualquier variable que se derive de
ahí (resultado, diferencia de goles) sale del propio marcador: usarla para
predecir el partido sería *target leakage*. No hay features honestas que sacar.

football-data.co.uk no requiere API key ni registro, y publica por partido:
tiros, tiros al arco, corners, faltas, tarjetas, marcador del entretiempo y
cuotas de apuestas. Son datos con los que sí se puede construir feature
engineering legítimo — promedios móviles de fechas **anteriores**, nunca del
partido a predecir.

### Ingesta histórica

```bash
python -m pipelines.footballdata_pipeline
```

Configurable por entorno (valores por defecto en `config.py`):

```bash
FOOTBALLDATA_DIVISIONES=E0,SP1,I1,D1,F1
FOOTBALLDATA_TEMPORADAS=1516,1617,1718,1819,1920,2021,2122,2223,2324,2425
```

Resultado de la carga por defecto: **18.011 partidos**, 10 temporadas
(2015-2025), 160 equipos, 5 ligas → `data/bronze/footballdata/partidos`.

**Idempotencia**: el CSV no trae ids, así que se sintetizan por hash del
partido (`FD-` + sha1 de división, temporada, fecha y equipos normalizados).
Al ser deterministas, re-ingerir una temporada reemplaza los mismos registros
en lugar de duplicarlos. La tabla particiona por `(id_liga, temporada)`, que es
la unidad natural de carga.

**Nota sobre `arbitro`**: solo la Premier League lo publica (100% de cobertura);
el resto de las ligas viene vacío. No es una feature utilizable cross-liga.

---

## Datos en producción — por qué `data/` está commiteado

El dashboard corre en Streamlit Community Cloud, que clona el repo y **no
persiste el filesystem** entre reinicios del contenedor (se duerme tras
inactividad y arranca de cero). Si `data/` estuviera en `.gitignore`, cada
cold start dejaría al primer usuario que entra pagando el costo de rehacer
bronze → silver → gold en vivo contra la API — y a la capa ML directamente no
la regeneraría nadie, porque `app.py` solo la LEE, nunca la entrena.

Por eso `data/` se trackea, y dos GitHub Actions la mantienen fresca con la
cadencia que ya tenía sentido en `pipelines/flow.py` (datos de API seguido,
histórico de football-data casi nunca):

| Workflow | Cadencia | Corre |
|---|---|---|
| `.github/workflows/refresh-thesportsdb.yml` | diaria | `pipeline_medallion` (bronze → silver → gold de TheSportsDB) |
| `.github/workflows/refresh-ml.yml` | semanal | `pipeline_ml` (ingesta football-data + reentrena los dos modelos) |

Cada uno commitea solo lo que le corresponde y pushea a `main` — Streamlit
Cloud redeploya con el commit y sirve datos ya generados. El pipeline dejó de
correr en el request del usuario.

---

