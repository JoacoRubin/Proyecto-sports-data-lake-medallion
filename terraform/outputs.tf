# =============================================================================
# outputs.tf
# =============================================================================

output "api_endpoint" {
  description = "URL publica de la inference API. Va como secret INFERENCE_API_URL en Streamlit Cloud."
  value       = aws_apigatewayv2_api.inference_api.api_endpoint
}

output "ecr_repository_url" {
  description = "Donde pushear la imagen con docker."
  value       = aws_ecr_repository.inference_api.repository_url
}

output "lambda_function_name" {
  value = aws_lambda_function.inference_api.function_name
}

output "sns_topic_arn" {
  description = "Topic de alarmas. La suscripcion por mail necesita confirmarse manualmente."
  value       = aws_sns_topic.alerts.arn
}
