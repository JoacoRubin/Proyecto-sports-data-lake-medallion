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
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}
