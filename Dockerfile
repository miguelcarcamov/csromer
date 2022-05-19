FROM ubuntu:latest

RUN apt-get update -y && \
    apt-get install -y build-essential && \
    apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip \
    python3-wheel \
    python3-setuptools && \
    apt-get install -y libblas-dev && \
    apt-get install -y liblapack-dev && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN python3 --version
RUN pip3 --version