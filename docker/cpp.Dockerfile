FROM --platform=linux/arm64 ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        make \
        git \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
