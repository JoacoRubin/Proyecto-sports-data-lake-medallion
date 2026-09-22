# =============================================================================
# ecr.tf — repo de la imagen de la inference API.
# =============================================================================

resource "aws_ecr_repository" "inference_api" {
  name                 = var.project_name
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }
}
