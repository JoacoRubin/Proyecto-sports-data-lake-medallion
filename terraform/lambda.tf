# =============================================================================
# lambda.tf — la inference API como funcion Lambda basada en imagen.
#
# `image_uri` apunta al tag `:latest`, no a un digest fijo: el ciclo de
# deploy real sigue siendo `docker build` + `docker push` +
# `aws lambda update-function-code` (ver README, seccion "Deploy en AWS
# Lambda") -- Terraform fija la CONFIGURACION del recurso (memoria, timeout,
# rol, permisos), no gestiona cada imagen nueva. Como el texto del URI no
# cambia (sigue siendo `:latest`), un `terraform plan` no ve diferencia
# aunque la imagen a la que apunta ese tag haya cambiado.
# =============================================================================

resource "aws_cloudwatch_log_group" "inference_lambda" {
  name              = "/aws/lambda/${var.project_name}"
  retention_in_days = 14
}

resource "aws_lambda_function" "inference_api" {
  function_name = var.project_name
  role          = aws_iam_role.inference_lambda.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.inference_api.repository_url}:latest"

  memory_size = var.lambda_memory_size
  timeout     = var.lambda_timeout_seconds

  architectures = ["x86_64"]

  logging_config {
    log_format = "Text"
    log_group  = aws_cloudwatch_log_group.inference_lambda.name
  }

  depends_on = [aws_iam_role_policy_attachment.lambda_basic_execution]
}
