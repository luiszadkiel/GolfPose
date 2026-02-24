# ============================================================
#  GolfPose AI - Deploy automatizado a EC2 via ECR
# ============================================================
# Uso: .\deploy.ps1 -KeyPath "C:\ruta\golfpose-keyssh-zadkiel.pem"
# ============================================================

param(
    [Parameter(Mandatory=$true)]
    [string]$KeyPath,

    [string]$EC2Host = "13.220.199.23",
    [string]$EC2User = "ec2-user",
    [string]$ECR_URI = "293926505005.dkr.ecr.us-east-1.amazonaws.com",
    [string]$IMAGE   = "golfpose-gpu",
    [string]$TAG     = "latest"
)

$ErrorActionPreference = "Stop"
$FULL_IMAGE = "$ECR_URI/${IMAGE}:${TAG}"

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  GOLFPOSE AI - DEPLOY A EC2 CON GPU"     -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# --- PASO 1: Subir proyecto al EC2 via SCP ---
Write-Host ""
Write-Host "[1/4] Subiendo proyecto al EC2..." -ForegroundColor Yellow

$excludes = @("__pycache__", ".git", "*.pyc", "web_app/uploads", "web_app/results", "common/__pycache__")

scp -i $KeyPath -r `
    common `
    configs `
    entrenamiento_kaggle/work_dirs `
    web_app `
    requirements.txt `
    Dockerfile `
    .dockerignore `
    "${EC2User}@${EC2Host}:~/golf-pose/"

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Fallo al subir archivos" -ForegroundColor Red
    exit 1
}
Write-Host "Archivos subidos OK" -ForegroundColor Green

# --- PASO 2: Build en EC2 ---
Write-Host ""
Write-Host "[2/4] Construyendo imagen Docker en EC2..." -ForegroundColor Yellow

ssh -i $KeyPath "${EC2User}@${EC2Host}" @"
cd ~/golf-pose
docker build -t ${IMAGE}:${TAG} .
"@

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Fallo en docker build" -ForegroundColor Red
    exit 1
}
Write-Host "Imagen construida OK" -ForegroundColor Green

# --- PASO 3: Push a ECR desde EC2 ---
Write-Host ""
Write-Host "[3/4] Subiendo imagen a ECR..." -ForegroundColor Yellow

ssh -i $KeyPath "${EC2User}@${EC2Host}" @"
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $ECR_URI
docker tag ${IMAGE}:${TAG} $FULL_IMAGE
docker push $FULL_IMAGE
"@

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Fallo en push a ECR" -ForegroundColor Red
    exit 1
}
Write-Host "Imagen en ECR OK" -ForegroundColor Green

# --- PASO 4: Ejecutar contenedor ---
Write-Host ""
Write-Host "[4/4] Iniciando contenedor con GPU..." -ForegroundColor Yellow

ssh -i $KeyPath "${EC2User}@${EC2Host}" @"
docker stop golfpose 2>/dev/null
docker rm golfpose 2>/dev/null
docker run -d --gpus all --name golfpose -p 8000:8000 --restart unless-stopped $FULL_IMAGE
"@

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Fallo al iniciar contenedor" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "  DEPLOY EXITOSO"                          -ForegroundColor Green
Write-Host "  http://${EC2Host}:8000"                  -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
