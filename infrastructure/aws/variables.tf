variable "region" {
  type    = string
  default = "us-west-2"
}

variable "image_tag" {
  type        = string
  description = "Tag of the API image in ECR (use the git SHA; tags are immutable)."
}

variable "desired_count" {
  type    = number
  default = 1
}

variable "allowed_cidrs" {
  type        = list(string)
  default     = ["0.0.0.0/0"]
  description = "Who may reach the API. Narrow this for anything beyond a demo."
}
