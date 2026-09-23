# =============================================================================
# sns.tf — topic de alertas para las alarmas de CloudWatch.
#
# La suscripcion por mail queda "pending confirmation" hasta que se clickee
# el link que manda AWS -- Terraform crea la suscripcion, no puede confirmarla
# por vos.
# =============================================================================

resource "aws_sns_topic" "alerts" {
  name = "${local.short_name}-alerts"
}

resource "aws_sns_topic_subscription" "alerts_email" {
  topic_arn                       = aws_sns_topic.alerts.arn
  protocol                        = "email"
  endpoint                        = var.alert_email
  confirmation_timeout_in_minutes = 1
  endpoint_auto_confirms          = false

  # confirmation_timeout_in_minutes y endpoint_auto_confirms solo se usan al
  # CREAR la suscripcion (le dicen a SNS como mandar/interpretar la
  # confirmacion) -- la API de SNS no los devuelve al leer una suscripcion ya
  # existente, asi que `terraform plan` los muestra como diff fantasma
  # (+add) para siempre, ya importada o no. No es un problema real: aplicar
  # ese "cambio" no modificaria nada en AWS.
  lifecycle {
    ignore_changes = [confirmation_timeout_in_minutes, endpoint_auto_confirms]
  }
}
