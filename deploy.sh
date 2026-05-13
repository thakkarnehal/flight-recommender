#!/usr/bin/env bash
# Deploy flight-recommender to AWS ECR + App Runner
set -euo pipefail

REPO_NAME="flight-recommender"
REGION="us-east-1"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}"
IMAGE_TAG="latest"

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
echo "==> [2/5] Building Docker image (this takes a few minutes)..."
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

# 5. Create or update App Runner service
echo ""
echo "==> [5/5] Deploying to App Runner..."

SERVICE_NAME="${REPO_NAME}"

# Check if service already exists
EXISTING=$(aws apprunner list-services --region "${REGION}" \
    --query "ServiceSummaryList[?ServiceName=='${SERVICE_NAME}'].ServiceArn" \
    --output text 2>/dev/null || echo "")

if [ -n "${EXISTING}" ]; then
    echo "    Service exists — triggering redeployment..."
    aws apprunner start-deployment \
        --service-arn "${EXISTING}" \
        --region "${REGION}" > /dev/null
    SERVICE_ARN="${EXISTING}"
else
    echo "    Creating new App Runner service..."

    # App Runner needs an ECR access role
    ROLE_NAME="AppRunnerECRAccessRole"
    ROLE_ARN=$(aws iam get-role --role-name "${ROLE_NAME}" \
        --query "Role.Arn" --output text 2>/dev/null || echo "")

    if [ -z "${ROLE_ARN}" ]; then
        echo "    Creating IAM role ${ROLE_NAME}..."
        ROLE_ARN=$(aws iam create-role \
            --role-name "${ROLE_NAME}" \
            --assume-role-policy-document '{
                "Version":"2012-10-17",
                "Statement":[{
                    "Effect":"Allow",
                    "Principal":{"Service":"build.apprunner.amazonaws.com"},
                    "Action":"sts:AssumeRole"
                }]
            }' \
            --query "Role.Arn" --output text)
        aws iam attach-role-policy \
            --role-name "${ROLE_NAME}" \
            --policy-arn "arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess"
        sleep 10   # IAM propagation
    fi

    SERVICE_ARN=$(aws apprunner create-service \
        --service-name "${SERVICE_NAME}" \
        --region "${REGION}" \
        --source-configuration "{
            \"ImageRepository\": {
                \"ImageIdentifier\": \"${ECR_URI}:${IMAGE_TAG}\",
                \"ImageConfiguration\": {
                    \"Port\": \"8080\"
                },
                \"ImageRepositoryType\": \"ECR\"
            },
            \"AuthenticationConfiguration\": {
                \"AccessRoleArn\": \"${ROLE_ARN}\"
            },
            \"AutoDeploymentsEnabled\": false
        }" \
        --instance-configuration '{
            "Cpu": "1 vCPU",
            "Memory": "2 GB"
        }' \
        --health-check-configuration '{
            "Protocol": "HTTP",
            "Path": "/health",
            "Interval": 10,
            "Timeout": 5,
            "HealthyThreshold": 1,
            "UnhealthyThreshold": 3
        }' \
        --query "Service.ServiceArn" --output text)
    echo "    Created: ${SERVICE_ARN}"
fi

# Wait for service to be running
echo "    Waiting for service to reach RUNNING state..."
for i in $(seq 1 30); do
    STATUS=$(aws apprunner describe-service \
        --service-arn "${SERVICE_ARN}" \
        --region "${REGION}" \
        --query "Service.Status" --output text)
    URL=$(aws apprunner describe-service \
        --service-arn "${SERVICE_ARN}" \
        --region "${REGION}" \
        --query "Service.ServiceUrl" --output text 2>/dev/null || echo "")
    printf "\r    [%02d/30] Status: %-15s" "${i}" "${STATUS}"
    if [ "${STATUS}" = "RUNNING" ]; then
        echo ""
        echo ""
        echo "=========================================="
        echo "  DEPLOYED"
        echo "  URL:    https://${URL}"
        echo "  Health: https://${URL}/health"
        echo "=========================================="
        echo ""
        echo "Test with:"
        echo "  curl https://${URL}/health"
        echo "  curl -X POST https://${URL}/recommend \\"
        echo "    -H 'Content-Type: application/json' \\"
        echo "    -d '{\"freq_tier\":\"high\",\"budget_tier\":\"business\",\"time_pref\":\"morning\",\"origin\":\"JFK\",\"destination\":\"LAX\"}'"
        exit 0
    fi
    if [ "${STATUS}" = "CREATE_FAILED" ] || [ "${STATUS}" = "UPDATE_FAILED" ]; then
        echo ""
        echo "ERROR: Service entered ${STATUS} state."
        exit 1
    fi
    sleep 10
done

echo ""
echo "WARNING: service did not reach RUNNING within 5 minutes."
echo "Check the App Runner console for details."
