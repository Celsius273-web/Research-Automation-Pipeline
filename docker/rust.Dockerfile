FROM --platform=linux/arm64 rust:1.78-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        pkg-config \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
