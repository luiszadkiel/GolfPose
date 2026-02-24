#!/usr/bin/env bash
set -e

echo "=========================================="
echo "   INICIANDO GOLFPOSE AI ANALYZER"
echo "=========================================="

if command -v nvidia-smi &> /dev/null; then
    echo "GPU detectada:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠️  Sin GPU detectada — ejecutando en CPU"
fi

echo ""
echo "Cargando modelos de IA... (esto puede tardar 1 min)"
python web_app/main.py
