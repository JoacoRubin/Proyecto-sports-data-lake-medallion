# =============================================================================
# services/inference_api/lambda_handler.py — adapta main:app a AWS Lambda.
#
# Mangum traduce el evento de Function URL (formato API Gateway v2 payload) a
# un request ASGI y la respuesta de vuelta. No se toca main.py: la misma app
# de FastAPI sirve local (uvicorn), en Docker Compose y en Lambda -- un solo
# codigo de negocio, tres formas de invocarlo. Ver Dockerfile.lambda para el
# runtime que ejecuta este handler.
# =============================================================================

from mangum import Mangum

from services.inference_api.main import app

handler = Mangum(app)
