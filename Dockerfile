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

# Crear directorios necesarios para que la app no falle al arrancar
RUN mkdir -p entrenamiento_kaggle/work_dirs/detector_yolox_2cls \
    entrenamiento_kaggle/work_dirs/pose2d_hrnet \
    entrenamiento_kaggle/work_dirs/pose3d_golfpose \
    web_app/uploads web_app/results

# Instalar OpenMMLab primero para evitar conflictos de versiones luego
RUN pip install --no-cache-dir -U openmim && \
    mim install mmengine==0.10.3 mmcv==2.1.0 mmdet==3.2.0 mmpose==1.3.1

# Instalar el resto de dependencias (esto fijará numpy y otros si mim los cambió)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la app
COPY common/ common/
COPY configs/ configs/
COPY web_app/ web_app/

# TRUCO: Copiar modelos solo si existen. 
# Como requirements.txt siempre existe, este comando no fallará en GitHub Actions
# aunque la carpeta work_dirs no esté en Git.
COPY requirements.txt entrenamiento_kaggle/work_dirs* entrenamiento_kaggle/work_dirs/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

CMD ["python", "web_app/main.py"]
