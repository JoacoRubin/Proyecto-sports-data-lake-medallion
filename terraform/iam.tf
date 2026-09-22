# =============================================================================
# iam.tf — rol de ejecucion de la Lambda.
#
# Solo AWSLambdaBasicExecutionRole (escribir logs a CloudWatch), nada mas.
# El usuario deploy-cli que corre estos comandos tiene AdministratorAccess
# (heredado de antes de este proyecto) -- la Lambda en si NO hereda eso, solo
# lo minimo que necesita para correr.
# =============================================================================

data "aws_iam_policy_document" "lambda_assume_role" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "inference_lambda" {
  name               = "${local.short_name}-lambda-role"
  description        = "Ejecucion de la Lambda de inference API del proyecto sports-ml. Solo logs a CloudWatch."
  assume_role_policy = data.aws_iam_policy_document.lambda_assume_role.json
}

resource "aws_iam_role_policy_attachment" "lambda_basic_execution" {
  role       = aws_iam_role.inference_lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}
