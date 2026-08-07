FROM --platform=linux/arm64 python:3.8-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# build-essential covers native extensions; gfortran/openblas help SciPy/sklearn source builds
# when a wheel is unavailable on this platform.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gfortran \
        libopenblas-dev \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
