FROM ubuntu:latest

RUN apt-get update -y && \
    apt-get install -y build-essential && \
    apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip \
    python3-wheel \
    python3-setuptools && \
    apt-get install -y libblas-dev && \
    apt-get install liblapack-dev && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* \

RUN pip3 install --no-cache-dir -U install setuptools pip