# AWS deployment

This deploys the API to **ECS Fargate** behind an Application Load Balancer. It also creates:
- a versioned, private S3 bucket for raw data
- an ECR repository with immutable tags
- CloudWatch logs and alarms (p95 latency, 5xx rate, CPU, memory), which notify an SNS topic

Kubernetes would be unnecessary at this scale.

```bash
# 1. Infrastructure. The first apply needs an image tag that you push in step 2.
cd infrastructure/aws
terraform init
terraform apply -var="image_tag=$(git rev-parse --short HEAD)"

# 2. Build and push the image. Models and artifacts come from `make evaluate`.
REPO=$(terraform output -raw ecr_repository)
aws ecr get-login-password | docker login --username AWS --password-stdin "${REPO%/*}"
docker build -f apps/api/Dockerfile -t "$REPO:$(git rev-parse --short HEAD)" ../..
docker push "$REPO:$(git rev-parse --short HEAD)"

# 3. Copy the raw data to S3 (the immutable copy of data/raw).
aws s3 sync ../../data/raw "s3://$(terraform output -raw raw_bucket)/raw/" --exclude "*/.gitkeep"

# 4. Benchmark the deployed API.
uv run python evaluation/benchmark_serving.py --url "$(terraform output -raw api_url)"
```

Cost: an ALB plus one 1 vCPU / 2 GB Fargate task runs roughly $1.50 per day. Run
`terraform destroy` when you are not demoing.

The listener uses plain HTTP to keep the demo simple. For real traffic, add an ACM
certificate and an HTTPS listener, and narrow `allowed_cidrs`.
