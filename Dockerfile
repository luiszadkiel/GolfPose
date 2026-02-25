FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip git curl \
    ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

# Crear directorios necesarios
RUN mkdir -p entrenamiento_kaggle/work_dirs/detector_yolox_2cls \
    entrenamiento_kaggle/work_dirs/pose2d_hrnet \
    entrenamiento_kaggle/work_dirs/pose3d_golfpose \
    web_app/uploads web_app/results

# 1. Copiar requirements primero
COPY requirements.txt .

# 2. Instalar PyTorch PRIMERO para que 'mim' pueda detectarlo
RUN pip install --no-cache-dir torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# 3. Ahora instalar OpenMMLab con mim (ya encontrará torch disponible)
RUN pip install --no-cache-dir -U openmim && \
    mim install mmengine==0.10.3 mmcv==2.1.0 mmdet==3.2.0 mmpose==1.3.1

# 4. Instalar el resto de dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la app
COPY common/ common/
COPY configs/ configs/
COPY web_app/ web_app/

# Copiamos la carpeta de modelos completa. 
# En GitHub Actions esto tendrá los archivos de S3.
# En local, si no la tienes, no fallará por el .dockerignore o carpeta vacía.
COPY entrenamiento_kaggle/ entrenamiento_kaggle/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

CMD ["python", "web_app/main.py"]
