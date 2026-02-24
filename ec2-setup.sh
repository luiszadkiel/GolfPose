#!/usr/bin/env bash
# ============================================================
#  Configuración inicial del EC2 (ejecutar UNA sola vez)
#  ssh -i key.pem ec2-user@IP "bash -s" < ec2-setup.sh
# ============================================================
set -e

echo "=========================================="
echo "  EC2 Setup - GolfPose AI"
echo "=========================================="

# Docker
if ! command -v docker &> /dev/null; then
    echo "[1/3] Instalando Docker..."
    sudo yum install -y docker
    sudo systemctl enable docker
    sudo systemctl start docker
    sudo usermod -aG docker ec2-user
    echo "Docker instalado. Se requiere reconectar SSH para aplicar grupo docker."
else
    echo "[1/3] Docker ya instalado"
fi

# NVIDIA Container Toolkit
if ! command -v nvidia-container-toolkit &> /dev/null; then
    echo "[2/3] Instalando NVIDIA Container Toolkit..."
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
        sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
    sudo yum install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
else
    echo "[2/3] NVIDIA Container Toolkit ya instalado"
fi

# AWS CLI (para ECR login)
if ! command -v aws &> /dev/null; then
    echo "[3/3] Instalando AWS CLI..."
    sudo yum install -y aws-cli
else
    echo "[3/3] AWS CLI ya instalado"
fi

# Crear directorio del proyecto
mkdir -p ~/golf-pose

echo ""
echo "=========================================="
echo "  Setup completo. Verificando GPU:"
echo "=========================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo ""
echo "Reconecta SSH para que el grupo 'docker' aplique:"
echo "  exit"
echo "  ssh -i key.pem ec2-user@<IP>"
