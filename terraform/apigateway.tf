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
# =============================================================================

resource "aws_apigatewayv2_api" "inference_api" {
  name          = var.project_name
  protocol_type = "HTTP"
}

resource "aws_apigatewayv2_integration" "inference_api" {
  api_id                 = aws_apigatewayv2_api.inference_api.id
  integration_type       = "AWS_PROXY"
  integration_method     = "POST"
  integration_uri        = aws_lambda_function.inference_api.invoke_arn
  payload_format_version = "2.0"
  timeout_milliseconds   = 30000
}

resource "aws_apigatewayv2_route" "default" {
  api_id    = aws_apigatewayv2_api.inference_api.id
  route_key = "$default"
  target    = "integrations/${aws_apigatewayv2_integration.inference_api.id}"
}

resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.inference_api.id
  name        = "$default"
  auto_deploy = true
}

resource "aws_lambda_permission" "apigateway_invoke" {
  statement_id  = "ApiGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.inference_api.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.inference_api.execution_arn}/*/*"
}
