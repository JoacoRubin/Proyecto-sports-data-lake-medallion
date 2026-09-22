# =============================================================================
# locals.tf
#
# Dos prefijos, no uno, porque asi quedaron nombrados los recursos reales
# cuando se armaron a mano durante la sesion: ECR/Lambda/API Gateway/
# Dashboard usan el nombre completo (`sports-ml-inference-api`), pero el rol
# IAM, el topic de SNS y las alarmas usan el prefijo corto
# (`sports-ml-inference`, sin el "-api"). Es una inconsistencia real de
# cuando se crearon via CLI, no un capricho -- Terraform importa lo que
# existe, no lo que hubiera sido mas prolijo.
# =============================================================================

locals {
  short_name = "sports-ml-inference"
}
