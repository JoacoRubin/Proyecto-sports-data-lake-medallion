# =============================================================================
# versions.tf — versiones fijadas del provider y del propio Terraform.
#
# State local, sin backend remoto: es un proyecto de una sola persona
# operando desde una sola maquina. Un backend en S3+DynamoDB resuelve
# colaboracion entre varios operadores -- infraestructura de mas para este
# caso. Si algun dia hay mas de un operador, ese es el momento de agregarlo,
# no antes.
# =============================================================================

terraform {
  required_version = ">= 1.9"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.70"
    }
  }
}

provider "aws" {
  region = var.aws_region
}
