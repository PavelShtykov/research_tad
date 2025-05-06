FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    vim \
    build-essential

RUN apt-get update && \
    apt-get install -y python3.11 python3.11-venv python3.11-dev python3.11-distutils && \
    ln -sf python3.11 /usr/bin/python3

RUN apt-get install -y python3-pip

RUN apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV HOME=/home
WORKDIR /home/workspace

COPY requirements.txt ./requirements1.txt
RUN pip install --no-cache-dir -r requirements1.txt

RUN MAX_JOBS=4 pip install flash-attn~=2.7.4.post1 --no-build-isolation

RUN apt update && apt install -y gdb python3-dbg

ENV PYTHONUNBUFFERED=1
ENV WANDB_SILENT=true
