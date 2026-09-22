# =============================================================================
# variables.tf — lo que cambia si esto se despliega en otra cuenta/region.
# =============================================================================

variable "aws_region" {
  description = "Region donde vive todo (Sao Paulo: menor latencia para Argentina que us-east-1)."
  type        = string
  default     = "sa-east-1"
}

variable "aws_account_id" {
  description = "Cuenta de AWS donde vive la infra. Solo se usa para armar ARNs explicitos."
  type        = string
  default     = "821672147613"
}

variable "project_name" {
  description = "Prefijo de nombre para todos los recursos de este proyecto."
  type        = string
  default     = "sports-ml-inference-api"
}

variable "alert_email" {
  description = "Mail que recibe las alarmas de CloudWatch via SNS."
  type        = string
  default     = "joaquinrubinstein6@gmail.com"
}

variable "lambda_memory_size" {
  description = <<-EOT
    MB asignados a la Lambda. No es solo memoria: Lambda asigna CPU
    proporcional. 3008MB (no el default de 1024MB) porque la fase INIT tiene
    un tope duro de 10s no configurable, y los imports de
    numpy/pandas/scikit-learn en frio superaban ese tope con menos CPU.
  EOT
  type        = number
  default     = 3008
}

variable "lambda_timeout_seconds" {
  description = <<-EOT
    30s, no mas: es el tope DURO del timeout de integracion de un HTTP API de
    API Gateway (no configurable mas alla de eso). Poner un timeout mayor en
    la Lambda no da mas margen real -- API Gateway corta antes igual.
  EOT
  type        = number
  default     = 30
}
