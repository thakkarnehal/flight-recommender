#!/usr/bin/env bash
# Deploy flight-recommender to AWS ECR + EC2
set -euo pipefail

REPO_NAME="flight-recommender"
REGION="us-east-1"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}"
IMAGE_TAG="latest"

EC2_HOST="44.202.128.221"
EC2_USER="ec2-user"
SSH_KEY="$HOME/Downloads/flight-recommender-key.pem"

echo "==> Account: ${ACCOUNT_ID}  Region: ${REGION}"
echo "==> Image:   ${ECR_URI}:${IMAGE_TAG}"

# 1. Create ECR repo (idempotent)
echo ""
echo "==> [1/5] Ensuring ECR repository exists..."
aws ecr describe-repositories --repository-names "${REPO_NAME}" --region "${REGION}" \
    > /dev/null 2>&1 || \
aws ecr create-repository \
    --repository-name "${REPO_NAME}" \
    --region "${REGION}" \
    --image-scanning-configuration scanOnPush=false \
    --encryption-configuration encryptionType=AES256 \
    > /dev/null
echo "    OK: ${REPO_NAME}"

# 2. Build image
echo ""
echo "==> [2/5] Building Docker image..."
docker build --platform linux/amd64 -t "${REPO_NAME}:${IMAGE_TAG}" .
echo "    OK: build complete"

# 3. Quick local smoke-test
echo ""
echo "==> [3/5] Local smoke test..."
CONTAINER_ID=$(docker run -d --rm -p 8081:8080 "${REPO_NAME}:${IMAGE_TAG}")
sleep 6
HEALTH=$(curl -sf http://localhost:8081/health || echo "FAILED")
docker stop "${CONTAINER_ID}" > /dev/null
if echo "${HEALTH}" | grep -q '"ok"'; then
    echo "    OK: /health → ${HEALTH}"
else
    echo "    FAILED: ${HEALTH}"
    exit 1
fi

# 4. Push to ECR
echo ""
echo "==> [4/5] Pushing to ECR..."
aws ecr get-login-password --region "${REGION}" \
    | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
docker tag "${REPO_NAME}:${IMAGE_TAG}" "${ECR_URI}:${IMAGE_TAG}"
docker push "${ECR_URI}:${IMAGE_TAG}"
echo "    OK: pushed ${ECR_URI}:${IMAGE_TAG}"

# 5. SSH into EC2 and redeploy
echo ""
echo "==> [5/5] Deploying to EC2 (${EC2_HOST})..."
chmod 400 "${SSH_KEY}"

ssh -i "${SSH_KEY}" -o StrictHostKeyChecking=no "${EC2_USER}@${EC2_HOST}" bash <<EOF
  set -e
  aws ecr get-login-password --region ${REGION} \
    | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com
  docker pull ${ECR_URI}:${IMAGE_TAG}
  docker stop flight-recommender 2>/dev/null || true
  docker rm flight-recommender 2>/dev/null || true
  docker stop \$(docker ps -q --filter publish=8080) 2>/dev/null || true
  docker run -d --name flight-recommender --restart always -p 8080:8080 ${ECR_URI}:${IMAGE_TAG}
  echo "    OK: container started"
EOF

echo ""
echo "=========================================="
echo "  DEPLOYED"
echo "  URL:    http://${EC2_HOST}:8080"
echo "  Health: http://${EC2_HOST}:8080/health"
echo "=========================================="
echo ""
echo "Test with:"
echo "  curl http://${EC2_HOST}:8080/health"
