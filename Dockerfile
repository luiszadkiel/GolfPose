FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Instalar dependencias del sistema y limpiar cache de apt en el mismo paso
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip git curl \
    ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

# Instalar librerías de Python y limpiar cache de pip inmediatamente
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install -U openmim && \
    mim install mmengine==0.10.3 mmcv==2.1.0 mmdet==3.2.0 mmpose==1.3.1 && \
    rm -rf ~/.cache/pip && rm -rf ~/.cache/mim

# Copiar el código de la app
COPY common/ common/
COPY configs/ configs/
# Solo copiamos lo necesario de work_dirs para ahorrar espacio
COPY entrenamiento_kaggle/work_dirs/ entrenamiento_kaggle/work_dirs/
COPY web_app/ web_app/

RUN mkdir -p web_app/uploads web_app/results

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

CMD ["python", "web_app/main.py"]
