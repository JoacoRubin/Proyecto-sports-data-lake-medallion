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
- **ML**: dos modelos sobre 20.013 partidos —expulsiones y over 2.5 goles—, este último medido contra el mercado de apuestas.

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

### Quality gate — cuándo un reentrenamiento reemplaza al modelo en producción

Antes de esto, `pipelines/ml_pipeline.py` guardaba una versión nueva en
**cada** corrida — reentrenar y promover eran la misma acción. `ml/promotion.py`
los separa: el candidato solo reemplaza al modelo en producción si

1. su métrica principal mejora sobre la del modelo actual (`pr_auc` en
   expulsiones, `roc_auc` en goles — mismo criterio que
   `app.py::_seccion_modelo`), **y**
2. el Brier no empeora más de una tolerancia (`config.ML_TOLERANCIA_CALIBRACION`,
   default `0.01`).

La condición (2) existe por un caso ya medido en este mismo proyecto: el
`class_weight="balanced"` documentado en `ml/training.py` mejoraba el PR-AUC
pero rompía la calibración (log-loss 0.6683 vs 0.4563). Sin el chequeo de
Brier, ese modelo hubiera pasado el gate igual.

Cada corrida — se promueva o no — queda registrada en
`data/gold/ml/promociones`, visible en el Streamlit del modelo
correspondiente ("Historial de promociones"). Un rechazo no borra nada: el
`.joblib` candidato ni se guarda, así que `cargar_modelo` sigue devolviendo
la versión anterior sin que nada más se entere.

**Probado contra el pipeline real, no solo con mocks**: correr
`python -m pipelines.ml_pipeline` con los mismos datos que ya entrenaron el
modelo en producción da exactamente la misma métrica (el entrenamiento es
determinístico) — el gate lo rechazó por empate, `pr_auc 0.2171 no supera al
actual 0.2171`, y no se tocó ningún archivo en `data/models/`. Es el
comportamiento correcto: un empate no es una mejora.

**Por qué no es el Model Registry de MLflow**: mismo argumento que la sección
anterior. MLflow resuelve "cuál versión es la de producción" con estados
(Staging/Production) en un servidor propio; acá esa respuesta ya existe sin
infraestructura nueva — es la versión más reciente en `data/models/` — así
que promover es, literalmente, decidir si se llama a `guardar_modelo`.

### Tracking de experimentos

En una tabla Delta de gold, `data/gold/ml/experimentos` — no en MLflow. Se
consulta con el mismo `leer_tabla_delta` que el resto del proyecto, se ve en el
mismo Streamlit, y Delta aporta versionado y time travel. Cero infraestructura
nueva.

Los modelos se versionan en `data/models/` con joblib más una ficha JSON
(features, prior, métricas, fecha). Un `.joblib` suelto es un binario que nadie
puede auditar.

---

## Servir los modelos — API de inferencia

`ml/inference.py` existía desde antes, testeado y todo, pero no lo llamaba
nadie: `app.py` solo mostraba la ficha y el historial de cada modelo, nunca
predecía un partido nuevo. `services/inference_api/` es la puerta de entrada
que faltaba — una capa HTTP delgada encima de `ml/inference.py` y
`ml/registry.py`, sin reimplementar carga de modelos ni feature engineering.

```bash
python -m pip install -r requirements-api.txt
uvicorn services.inference_api.main:app --reload
```

| Endpoint | Qué hace |
|---|---|
| `GET /health` | Liveness. No toca modelos ni Delta. |
| `GET /models/{modelo}` | La ficha registrada (`expulsiones` o `goles`): features, prior, métricas, versión. |
| `POST /predictions/{modelo}` | Predice un partido futuro. Ver `PartidoRequest` en `services/inference_api/schemas.py`. |

```json
POST /predictions/expulsiones
{
  "liga": "Spanish La Liga",
  "temporada": "2024-2025",
  "equipo_local": "Real Madrid",
  "equipo_visitante": "Barcelona",
  "fecha_partido": "2025-06-01"
}
```

**El endpoint jamás entrena.** Carga el modelo aprobado que dejó
`pipelines/ml_pipeline.py` en `data/models/` y solo infiere — entrenamiento e
inferencia quedan en procesos y momentos distintos, así el modelo se puede
versionar y desplegar sin depender del pipeline de entrenamiento.

**Por qué una ruta genérica y no cuatro endpoints copiados**: el resto del
proyecto ya resolvió "dos modelos, una sola maquinaria" con un runner
genérico y dos configuraciones (`pipelines/ml_pipeline.py`,
`app.py::_seccion_modelo`). Esta API sigue el mismo patrón en vez de duplicar
rutas por modelo.

**Por qué no se tocó la decisión de MLflow**: `ml/registry.py` ya la
justifica con evidencia (tracking en Delta, cero infraestructura nueva). Este
servicio consume esa misma capa de registro tal cual está — no había ninguna
razón nueva para revisarla.

### Dashboard + API juntos, con Docker

`app.py` tiene una sección "Predecir un partido nuevo" que le pega a esta API
por HTTP. Localmente, sin nada más corriendo, esa sección no tiene a quién
llamar. `docker-compose.yml` levanta los dos servicios en la misma red para
que se encuentren:

```bash
docker compose up --build
```

| Servicio | Puerto en el host | Cómo lo encuentra el otro |
|---|---|---|
| `api` (inference API) | `localhost:8010` | — |
| `dashboard` (Streamlit) | `localhost:8501` | `INFERENCE_API_URL=http://api:8000` (nombre del servicio de compose, no `localhost`) |

El `8010` en vez de `8000` es simplemente para no pisar otro proceso que ya
podés tener escuchando en ese puerto en tu máquina; puertos del host, no del
contenedor — adentro de la red de compose los dos servicios se siguen
hablando por el `8000` interno.

**Por qué la imagen base es `python:3.14-slim`, no `3.11`**: `requirements.txt`
fija `numpy==2.5.1` y otras versiones que solo tienen wheels para cp314 (ver
el comentario de ese archivo — es la misma versión que corre en Streamlit
Community Cloud). Usar `3.11-slim` ahí rompe el build; lo dice el propio pip
cuando no encuentra la versión.

`Dockerfile.api` y `Dockerfile.dashboard` son imágenes separadas a propósito
(mismo criterio que separar `requirements-api.txt` de `requirements.txt`):
cada servicio instala solo lo que necesita, y uno puede reconstruirse sin
invalidar la cache del otro.

Esto resuelve "correr los dos servicios juntos localmente". Streamlit
Community Cloud (donde el dashboard está deployado hoy) sigue sirviendo un
solo proceso — por eso la inference API tiene su propio deploy en AWS Lambda,
ver la sección siguiente.

### Deploy en AWS Lambda — la inference API en producción

**Por qué Lambda y no ECS/App Runner/EC2**: el tráfico de esta API es
esporádico (alguien abre el dashboard y pide una predicción), no continuo.
Lambda escala a cero entre invocaciones: costo ≈ 0 cuando nadie la usa,
contra un piso fijo mensual de las alternativas administradas. El costo es
un cold start de unos segundos la primera vez que se invoca tras un rato de
inactividad.

**Por qué API Gateway y no Function URL**: la idea original era exponer la
Lambda directo con una Function URL (`auth-type NONE`) — más simple, un
recurso menos. En la práctica, esta cuenta de AWS devolvía **403 Forbidden**
en toda invocación anónima pese a que el resource policy estaba bien armado
(`lambda:InvokeFunctionUrl` para principal `*`, condición
`FunctionUrlAuthType: NONE` — la config exacta que pide la documentación).
Confirmado con los logs de CloudWatch: el log group ni existía, es decir,
la invocación se frenaba en el borde, antes de llegar a ejecutar la función.
Es una restricción anti-abuso a nivel de cuenta sobre Function URLs públicas,
independiente de IAM, y la única forma de sacarla es un caso de soporte a
AWS. En vez de esperar eso, la Lambda quedó detrás de un **API Gateway HTTP
API** (`apigatewayv2 create-api --target <lambda-arn>`, quick-create):
mismo `lambda_handler.py` sin tocar una línea, porque Function URL y HTTP
API v2 comparten el mismo formato de payload que ya entiende `mangum`. Un
paso extra de setup (`add-permission` para `apigateway.amazonaws.com`,
acotado por `source-arn` a este API), pero sin depender de que AWS levante
una restricción de cuenta.

**Por qué una imagen custom y no `public.ecr.aws/lambda/python`**:
`requirements.txt` fija `numpy==2.5.1`, que solo publica wheels `cp314` (ver
más arriba), y no hay garantía de que la imagen base oficial de Lambda ya
publique esa versión de Python. `Dockerfile.lambda` arma un runtime custom
sobre el mismo `python:3.14-slim` que ya usan `Dockerfile.api` y
`Dockerfile.dashboard`, agregando el Runtime Interface Client
(`awslambdaric`) — patrón soportado oficialmente por AWS para bases no-Amazon.
`services/inference_api/lambda_handler.py` adapta la misma app de FastAPI con
`mangum`, sin tocar `main.py`: un solo código de negocio, tres formas de
invocarlo (uvicorn local, Docker Compose, Lambda).

Los 13MB de histórico (`data/bronze/footballdata/`) y los 212KB de modelos
(`data/models/`) entran cómodos dentro de la imagen — no hace falta EFS ni
leerlos de S3 en cada cold start (mover esos 13MB a S3 no cambiaría nada: lo
que pesa la imagen no es la data, son las librerías, ver abajo).

**Tamaño de la imagen — de 1.44GB a 1.19GB**: `requirements-api.txt` hacía
`-r requirements.txt`, que **también** trae `streamlit` y `plotly` (deps
exclusivas del dashboard) aunque la API nunca las importa — pese a que el
propio comentario del archivo decía "no necesita streamlit/plotly". Se
separaron las dependencias realmente compartidas a `requirements-core.txt`
(pandas, deltalake, pyarrow, numpy, scikit-learn, pandera) y ahora
`requirements.txt` (dashboard) y `requirements-api.txt` (API/Lambda) parten
de ese archivo en vez de uno del otro. Ahorro real: ~250MB (streamlit +
plotly + sus dependencias — altair, pydeck, pillow.libs).

**Por qué no se llega a menos de 500MB**: `pyarrow` (153MB) + `deltalake`
(123MB) + `scipy` (110MB) + `pandas` (77MB) + `scikit-learn` (49MB) +
`numpy` (43MB) sola ya suman 555MB — son las librerías que la API
efectivamente ejecuta (Delta Lake + feature engineering + inferencia), no
hay grasa ahí para cortar sin cambiar de arquitectura. Se podría sacar
`deltalake` leyendo un Parquet plano pre-exportado en vez del formato Delta
completo, pero eso significa un código de lectura distinto para la API que
para el resto del proyecto — exactamente lo que este proyecto evita en
todos lados (un solo código de negocio, no duplicar lógica) — a cambio de
ahorrar centavos de dólar al mes en storage de ECR. No vale la pena.

**Tres bugs reales que aparecieron armando esto** (documentados porque no son
obvios):

1. `mangum==0.19.0` rompe en Python 3.14 con
   `RuntimeError: There is no current event loop in thread 'MainThread'`.
   Llama `asyncio.get_event_loop()` fuera de un loop corriendo, y 3.14
   finalmente eliminó el auto-create implícito (deprecado desde 3.10). Fix:
   `mangum==0.21.0` (feb 2026), que agregó soporte real para 3.14 — ver el
   comentario en `requirements-lambda.txt`, no bajar ese pin.
2. `docker build` con BuildKit genera por defecto un manifest OCI con
   attestations de provenance/SBOM que Lambda todavía no acepta para
   imágenes de contenedor (`InvalidParameterValueException: image manifest
   ... not supported`). Fix: buildear con
   `docker buildx build --provenance=false --sbom=false`.
3. Cold start fallaba con `Status: timeout` en la fase `init` de
   CloudWatch, a los ~10 segundos justos. Lambda impone un **tope duro de 10s
   para la fase INIT** (imports a nivel de módulo, antes de que corra el
   handler) que no depende del timeout configurado de la función — y
   `import numpy/pandas/scikit-learn` en frío tarda más que eso con 1024MB
   (que en Lambda determina también la CPU asignada). La carga del histórico
   ya era lazy (`lru_cache` en `dependencies.py`), así que no era eso: eran
   los imports pesados mismos. Fix: subir memoria a 3008MB — más memoria =
   más CPU proporcional también durante INIT, no solo durante la ejecución.

**Build y deploy** (imagen ya construida y funcionando en
`sports-ml-inference-api`, para reconstruir tras un cambio):

```bash
docker buildx build --provenance=false --sbom=false --load \
  -f Dockerfile.lambda -t sports-ml-inference-api:latest .

aws ecr get-login-password --region sa-east-1 | \
  docker login --username AWS --password-stdin \
  821672147613.dkr.ecr.sa-east-1.amazonaws.com

docker tag sports-ml-inference-api:latest \
  821672147613.dkr.ecr.sa-east-1.amazonaws.com/sports-ml-inference-api:latest
docker push \
  821672147613.dkr.ecr.sa-east-1.amazonaws.com/sports-ml-inference-api:latest

aws lambda update-function-code \
  --function-name sports-ml-inference-api \
  --image-uri 821672147613.dkr.ecr.sa-east-1.amazonaws.com/sports-ml-inference-api:latest \
  --region sa-east-1
```

**Gotcha post-deploy**: justo después de `update-function-code` con una
imagen de ~1.2GB, las primeras invocaciones pueden tardar mucho más de lo
normal o directamente devolver 503 — Lambda todavía está distribuyendo la
imagen nueva por su infraestructura interna. Se resuelve solo en un par de
minutos; no es necesario re-desplegar ni es un bug del código.

**Setup del API Gateway** (una sola vez; `create-api --target` arma la
integración, la ruta `$default` y el stage con auto-deploy en un solo
comando, pero no agrega el permiso de invocación — eso es aparte):

```bash
aws apigatewayv2 create-api \
  --name sports-ml-inference-api \
  --protocol-type HTTP \
  --target arn:aws:lambda:sa-east-1:821672147613:function:sports-ml-inference-api \
  --region sa-east-1

aws lambda add-permission \
  --function-name sports-ml-inference-api \
  --statement-id ApiGatewayInvoke \
  --action lambda:InvokeFunction \
  --principal apigateway.amazonaws.com \
  --source-arn "arn:aws:execute-api:sa-east-1:821672147613:<api-id>/*/*" \
  --region sa-east-1
```

**Recursos** (región `sa-east-1`, cuenta `821672147613`): rol IAM
`sports-ml-inference-lambda-role` (solo `AWSLambdaBasicExecutionRole` — logs
a CloudWatch, nada más), repo ECR `sports-ml-inference-api` con
scan-on-push, función Lambda `sports-ml-inference-api` (3008MB, timeout 45s
— ver el bug #3 de arriba sobre por qué no quedó en 1024MB/30s), y el HTTP
API `sports-ml-inference-api` sin auth propia (la API en sí es de solo
lectura/predicción, sin cookies ni datos sensibles — no hay nada que
proteger con un API key todavía).

**Latencia real** medida contra la URL pública (imagen de 1.19GB, post
recorte de streamlit/plotly): **~15.8s en cold start** (imports pesados +
primera lectura del histórico) y **~466ms en caliente** (mismo entorno de
ejecución, todo ya cacheado en memoria del proceso).

Para que el dashboard en Streamlit Community Cloud le pegue a esta API en
vez de a `localhost`, la URL del API Gateway
(`https://bfpdri7tfg.execute-api.sa-east-1.amazonaws.com`) va como secret
`INFERENCE_API_URL` en la config de la app en Streamlit Cloud (Settings →
Secrets) — `config.py` ya lee esa variable de entorno, Streamlit Cloud
expone los secrets también como env vars, no hace falta tocar código.

### CloudWatch — alarmas y dashboard

Los logs básicos (`AWSLambdaBasicExecutionRole`) ya estaban desde el deploy
inicial. Lo que faltaba era **enterarse** cuando algo falla, sin tener que
entrar a leer logs a mano, y no dejar los logs creciendo para siempre.

**Retención de logs**: por defecto, un log group de Lambda **nunca expira**
(`retentionInDays: null`) — con el tiempo junta storage por logs que a los
pocos meses no le sirven a nadie. Se fijó en 14 días:

```bash
aws logs put-retention-policy \
  --log-group-name /aws/lambda/sports-ml-inference-api \
  --retention-in-days 14 \
  --region sa-east-1
```

**Alarmas** (SNS topic `sports-ml-inference-alerts`, con notificación por
mail — hace falta confirmar la suscripción desde el mail que manda AWS la
primera vez):

| Alarma | Métrica | Dispara si |
|---|---|---|
| `sports-ml-inference-lambda-errors` | `AWS/Lambda Errors` | Cualquier error en 5 min |
| `sports-ml-inference-lambda-throttles` | `AWS/Lambda Throttles` | Cualquier throttle en 5 min (concurrencia insuficiente) |
| `sports-ml-inference-lambda-duration-alta` | `AWS/Lambda Duration` (max) | Una invocación tarda más de 35s (el timeout está en 45s — es aviso temprano antes de que empiecen los timeouts reales) |
| `sports-ml-inference-apigw-5xx` | `AWS/ApiGateway 5xxError` | Cualquier 5xx del API Gateway en 5 min |

Umbral en 0 (no en un porcentaje de error rate) a propósito: es una API de
tráfico bajo, un solo error ya es una señal real, no ruido estadístico.

**Dashboard** `sports-ml-inference-api`: invocaciones/errores/throttles de
Lambda, duración (avg/max/p95) con una línea de referencia en el timeout,
concurrencia, y del lado de API Gateway requests/4xx/5xx y latencia
(avg/p95).

### Drift monitoring — ¿el modelo en producción sigue sirviendo?

**El prerequisito que casi no estaba**: drift monitoring necesita partidos
nuevos para tener algo que auditar. `FOOTBALLDATA_TEMPORADAS` se había
quedado fija hasta la temporada 2024-2025 bastante después de que arrancara
la 2025-2026 — el pipeline semanal (`refresh-ml.yml`) llevaba meses
reentrenando contra exactamente el mismo dataset estático, sin que nada
avisara porque la ingesta es idempotente (no duplica, pero tampoco trae
partidos que no pidas). Se corrigió agregando las temporadas 2025-2026 y
2026-2027 (en curso): **18.011 → 20.013 partidos**, 160 → 167 equipos.

**`ml/drift.py`, distinto de `ml/promotion.py`**: la promoción compara un
candidato recién entrenado contra la ficha histórica del modelo en
producción, sobre los mismos folds de siempre. El drift monitoring hace una
pregunta distinta — audita al modelo **que ya está sirviendo predicciones**
contra los partidos que jugó **después** de promoverse, la porción de
realidad que ni el entrenamiento ni el gate original llegaron a ver. Mismo
`construir_features` de siempre, mismo `metadata["features"]` con el que
ese modelo se entrenó (igual que `ml/inference.py`): no hay un camino de
evaluación aparte que pueda divergir.

Marca drift si la métrica principal cae más de una tolerancia **o** si el
Brier empeora más de la suya — alcanza con una sola de las dos, a
diferencia del quality gate de promoción, que exige que fallen ambas para
rechazar un candidato: acá no se compite contra una alternativa, cualquier
degradación real importa. Con menos de 20 partidos frescos, no hay
veredicto (`evaluado=False`): una muestra así de chica es ruido, no señal,
y decir "sin drift" en ese caso escondería el problema en vez de admitir
que no se pudo medir.

Corre cada domingo junto al reentrenamiento (`pipelines/ml_pipeline.py`), y
el resultado queda auditable en `data/gold/ml/drift` y visible en el
dashboard (`Historial de drift`, junto al de promociones) — igual que el
resto del tracking de este proyecto, sin infraestructura nueva.

**Notificación, a propósito, sin AWS todavía**: el veredicto queda en la
tabla Delta y como `WARNING` en el log del GitHub Action si detecta drift.
No está conectado al SNS de las alarmas de CloudWatch — eso requeriría
credenciales de AWS nuevas viviendo en GitHub Actions, una decisión aparte
que todavía no se tomó.

**Pendiente** (roadmap de Martín, último paso): Terraform, para dejar todo
esto (Lambda, API Gateway, IAM, CloudWatch, SNS) como código en vez de
comandos de `aws cli` corridos a mano.

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
FOOTBALLDATA_TEMPORADAS=1516,1617,1718,1819,1920,2021,2122,2223,2324,2425,2526,2627
```

Resultado de la carga por defecto: **20.013 partidos**, 12 temporadas
(2015-2027, incluida la temporada en curso), 167 equipos, 5 ligas →
`data/bronze/footballdata/partidos`.

**Gotcha real**: `FOOTBALLDATA_TEMPORADAS` se quedó fija en `...2425` (hasta
2024-2025) bastante después de que arrancara la temporada 2025-2026 — el
pipeline semanal (`refresh-ml.yml`) llevaba meses reentrenando contra
exactamente el mismo dataset estático, sin que nada avisara porque no rompía
nada, solo dejaba de traer partidos nuevos. Se encontró armando el drift
monitoring (ver más abajo): sin esto, no había datos nuevos que vigilar.

**Idempotencia**: el CSV no trae ids, así que se sintetizan por hash del
partido (`FD-` + sha1 de división, temporada, fecha y equipos normalizados).
Al ser deterministas, re-ingerir una temporada reemplaza los mismos registros
en lugar de duplicarlos. La tabla particiona por `(id_liga, temporada)`, que es
la unidad natural de carga.

**Nota sobre `arbitro`**: solo la Premier League lo publica (100% de cobertura);
el resto de las ligas viene vacío. No es una feature utilizable cross-liga.

---

## CI — la suite corre sola en cada push

`refresh-ml.yml` y `refresh-thesportsdb.yml` corren con cron y commitean
datos: no son CI, no corren en push/PR, no bloquean nada. Antes de
`.github/workflows/ci.yml`, los 256 tests de `tests/` solo corrían si alguien
se acordaba de correrlos a mano.

```
push / PR
   ↓
pytest (256 tests: contratos Pandera, leakage, quality gate, contrato HTTP)
   ↓
docker build (Dockerfile.api + Dockerfile.dashboard)
```

El job de Docker no es cosmético: la primera vez que se armaron
`Dockerfile.api`/`Dockerfile.dashboard`, un mismatch de versión de Python
(3.11 en vez de la 3.14 que pide `requirements.txt`) rompió el build en el
momento de correrlo a mano. Este job es lo que hubiera atajado eso en el
commit, no en el próximo deploy.

**Qué no cubre, a propósito**: no reentrena los modelos contra el dataset
real — eso ya lo hace `refresh-ml.yml` semanalmente, y duplicarlo en cada
push sería caro y redundante. `tests/test_ml_promotion.py` ya prueba el
quality gate con métricas sintéticas, sin necesitar Delta ni los 18k
partidos reales.

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

