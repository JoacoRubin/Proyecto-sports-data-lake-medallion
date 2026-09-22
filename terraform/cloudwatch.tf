# =============================================================================
# cloudwatch.tf — alarmas y dashboard.
#
# Umbral en 0 (no un % de error rate) a proposito en errors/throttles/5xx:
# es una API de trafico bajo, un solo error ya es señal real, no ruido
# estadistico. La alarma de duracion usa 25000ms (no el timeout de 30000ms
# configurado): tiene que avisar ANTES del corte real, no en el mismo
# instante.
# =============================================================================

resource "aws_cloudwatch_metric_alarm" "lambda_errors" {
  alarm_name          = "${local.short_name}-lambda-errors"
  alarm_description   = "Cualquier error en la Lambda de inference API en 5 min"
  namespace           = "AWS/Lambda"
  metric_name         = "Errors"
  dimensions          = { FunctionName = aws_lambda_function.inference_api.function_name }
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 0
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  ok_actions          = [aws_sns_topic.alerts.arn]
}

resource "aws_cloudwatch_metric_alarm" "lambda_throttles" {
  alarm_name          = "${local.short_name}-lambda-throttles"
  alarm_description   = "La Lambda esta siendo throttled (concurrencia insuficiente)"
  namespace           = "AWS/Lambda"
  metric_name         = "Throttles"
  dimensions          = { FunctionName = aws_lambda_function.inference_api.function_name }
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 0
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  ok_actions          = [aws_sns_topic.alerts.arn]
}

resource "aws_cloudwatch_metric_alarm" "lambda_duration_alta" {
  alarm_name          = "${local.short_name}-lambda-duration-alta"
  alarm_description   = "Una invocacion tardo mas de 25s (timeout real: 30s, el limite duro de API Gateway HTTP API, no el de Lambda) -- señal temprana antes de que empiecen los timeouts reales"
  namespace           = "AWS/Lambda"
  metric_name         = "Duration"
  dimensions          = { FunctionName = aws_lambda_function.inference_api.function_name }
  statistic           = "Maximum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 25000
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  ok_actions          = [aws_sns_topic.alerts.arn]
}

resource "aws_cloudwatch_metric_alarm" "apigw_5xx" {
  alarm_name          = "${local.short_name}-apigw-5xx"
  alarm_description   = "El API Gateway devolvio un 5xx en los ultimos 5 min"
  namespace           = "AWS/ApiGateway"
  metric_name         = "5xxError"
  dimensions          = { ApiId = aws_apigatewayv2_api.inference_api.id }
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 0
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  ok_actions          = [aws_sns_topic.alerts.arn]
}

resource "aws_cloudwatch_dashboard" "inference_api" {
  dashboard_name = var.project_name

  dashboard_body = jsonencode({
    widgets = [
      {
        type = "metric", x = 0, y = 0, width = 12, height = 6
        properties = {
          title  = "Lambda — Invocations / Errors / Throttles"
          region = var.aws_region
          metrics = [
            ["AWS/Lambda", "Invocations", "FunctionName", aws_lambda_function.inference_api.function_name, { stat = "Sum" }],
            ["AWS/Lambda", "Errors", "FunctionName", aws_lambda_function.inference_api.function_name, { stat = "Sum" }],
            ["AWS/Lambda", "Throttles", "FunctionName", aws_lambda_function.inference_api.function_name, { stat = "Sum" }],
          ]
          period = 300
          view   = "timeSeries"
        }
      },
      {
        type = "metric", x = 12, y = 0, width = 12, height = 6
        properties = {
          title  = "Lambda — Duration (ms)"
          region = var.aws_region
          metrics = [
            ["AWS/Lambda", "Duration", "FunctionName", aws_lambda_function.inference_api.function_name, { stat = "Average", label = "avg" }],
            ["AWS/Lambda", "Duration", "FunctionName", aws_lambda_function.inference_api.function_name, { stat = "Maximum", label = "max" }],
            ["AWS/Lambda", "Duration", "FunctionName", aws_lambda_function.inference_api.function_name, { stat = "p95", label = "p95" }],
          ]
          period = 300
          view   = "timeSeries"
          annotations = {
            horizontal = [{ label = "timeout real (30s, tope de API Gateway)", value = 30000 }]
          }
        }
      },
      {
        type = "metric", x = 0, y = 6, width = 12, height = 6
        properties = {
          title  = "Lambda — Concurrent Executions"
          region = var.aws_region
          metrics = [
            ["AWS/Lambda", "ConcurrentExecutions", "FunctionName", aws_lambda_function.inference_api.function_name, { stat = "Maximum" }],
          ]
          period = 300
          view   = "timeSeries"
        }
      },
      {
        type = "metric", x = 12, y = 6, width = 12, height = 6
        properties = {
          title  = "API Gateway — Requests / 4xx / 5xx"
          region = var.aws_region
          metrics = [
            ["AWS/ApiGateway", "Count", "ApiId", aws_apigatewayv2_api.inference_api.id, { stat = "Sum", label = "requests" }],
            ["AWS/ApiGateway", "4xxError", "ApiId", aws_apigatewayv2_api.inference_api.id, { stat = "Sum" }],
            ["AWS/ApiGateway", "5xxError", "ApiId", aws_apigatewayv2_api.inference_api.id, { stat = "Sum" }],
          ]
          period = 300
          view   = "timeSeries"
        }
      },
      {
        type = "metric", x = 0, y = 12, width = 12, height = 6
        properties = {
          title  = "API Gateway — Latency (ms)"
          region = var.aws_region
          metrics = [
            ["AWS/ApiGateway", "Latency", "ApiId", aws_apigatewayv2_api.inference_api.id, { stat = "Average", label = "avg" }],
            ["AWS/ApiGateway", "Latency", "ApiId", aws_apigatewayv2_api.inference_api.id, { stat = "p95", label = "p95" }],
          ]
          period = 300
          view   = "timeSeries"
        }
      },
    ]
  })
}
