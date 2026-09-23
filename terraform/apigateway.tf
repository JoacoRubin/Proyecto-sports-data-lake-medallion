# =============================================================================
# apigateway.tf — HTTP API delante de la Lambda.
#
# POR QUE API GATEWAY Y NO FUNCTION URL
# ---------------------------------------
# La idea original era una Function URL (auth-type NONE): un recurso menos.
# Esta cuenta de AWS bloquea invocaciones anonimas a Function URLs a nivel de
# cuenta -- 403 Forbidden pese a un resource policy correcto, confirmado
# porque el log group de CloudWatch ni se creaba (la invocacion se frenaba
# en el borde, antes de llegar a la Lambda). Es una restriccion anti-abuso
# de cuenta, no arreglable via IAM/CLI, solo con un caso de soporte a AWS.
# API Gateway no tiene esa restriccion y no requirio tocar codigo: Function
# URL y HTTP API v2 comparten el mismo formato de payload que ya entiende
# `mangum` en services/inference_api/lambda_handler.py.
#
# Timeout de integracion en 30000ms: es el tope DURO de un HTTP API, no
# configurable mas alla de eso (ver variables.tf, lambda_timeout_seconds).
#
# LO QUE NO ESTA ACA, A PROPOSITO: la integracion, la ruta y el stage
# --------------------------------------------------------------------
# El API se creo con el atajo `apigatewayv2 create-api --target`
# (quick-create), que arma integracion+ruta+stage automaticamente y los
# marca `ApiGatewayManaged: true`. Se intento reemplazarlos por versiones
# manejables por Terraform (borrar + crear sueltos) y AWS los protege de
# verdad: `delete-route`/`delete-integration`/`delete-stage` devuelven
# exito (200, sin error) pero NO borran nada -- se verifico en vivo,
# `get-routes` seguia mostrando el recurso "borrado" despues de la
# llamada exitosa. No es una limitacion de este provider: `terraform
# import` los rechaza por la misma razon ("was created via quick
# create"), y el propio AWS los protege contra borrado individual, no
# solo Terraform.
#
# La unica forma de traerlos a Terraform seria borrar y recrear el API
# COMPLETO, lo que cambia `api_id` y por lo tanto la URL publica
# (`api_endpoint` en outputs.tf) -- decision que se tomo NO tomar: la
# URL ya esta en uso. El 95% de la infra (IAM, ECR, Lambda, el API
# Gateway en si, permisos, logs, alarmas, dashboard, SNS) queda en
# Terraform; estos tres sub-recursos quedan gestionados por AWS via
# quick-create, fuera del alcance de este repo, documentado aca en vez
# de escondido.
# =============================================================================

resource "aws_apigatewayv2_api" "inference_api" {
  name          = var.project_name
  protocol_type = "HTTP"
}

resource "aws_lambda_permission" "apigateway_invoke" {
  statement_id  = "ApiGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.inference_api.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.inference_api.execution_arn}/*/*"
}
