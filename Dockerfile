# Presence Biometric Sidecar — FastAPI contract shim over serengil/deepface.
# Serves only the /v1/liveness-and-* endpoints behind X-Biometric-Secret auth;
# the upstream stock API surface (analyze/verify/represent) is not exposed.
FROM python:3.11-slim
LABEL org.opencontainers.image.source=https://github.com/NQB8Limited/deepface

# System libs required by opencv-python-headless / deepface.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/deepface

# Model weights cache — mount a volume at DEEPFACE_HOME so the
# MiniFASNet/ArcFace/retinaface weights don't re-download per container.
ENV DEEPFACE_HOME=/opt/deepface
RUN mkdir -p /opt/deepface

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY tests ./tests

ENV PORT=8001
EXPOSE 8001

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8001} --workers 1
