output "api_url" { value = "http://${aws_lb.api.dns_name}" }
output "ecr_repository" { value = aws_ecr_repository.api.repository_url }
output "raw_bucket" { value = aws_s3_bucket.raw.bucket }
output "alerts_topic" { value = aws_sns_topic.alerts.arn }
