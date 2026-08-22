# Pinned, and pinned to a Python rather than to a distro's incidental python3.
#
# `ubuntu:latest` was both unpinned and wrong: it now resolves to 24.04, which ships
# Python 3.12. Several of csromer's pinned dependencies publish no wheels beyond
# cp311 (scipy, astropy, matplotlib, PyWavelets), so on 3.12 an install silently
# falls back to compiling them from source -- slow, memory-hungry, and prone to
# failing outright. 3.11 is the top of the supported range; see tox.ini.
#
# Debian slim rather than Ubuntu: same toolchain, markedly smaller base layer, and
# the Python version is stated in the tag instead of being whatever the distro
# happens to ship.
FROM python:3.11-slim-bookworm

# PIP_NO_CACHE_DIR stops pip from leaving a wheel cache in the image; the bytecode
# and version-check settings trim more still. DEBIAN_FRONTEND avoids apt trying to
# prompt during a non-interactive build.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# One RUN, so the package lists never reach a committed layer -- deleting them in a
# later instruction would not shrink the image, since the earlier layer still holds
# them. `--no-install-recommends` is attached to the install that needs it; it was
# previously passed on a line of its own with no packages, where it did nothing.
#
#   build-essential  prox_tv and pynufft are source-only and always compile
#   libblas/liblapack/liblapacke  prox_tv links against LAPACK
#   git              setuptools_scm derives the version from git metadata
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libblas-dev \
        liblapack-dev \
        liblapacke-dev \
        git \
        ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Single layer for the Python tooling. The previous file spent four layers on
# `python3 --version`, `pip3 --version`, a bare `pip3` that only printed usage, and
# an echo.
RUN python -m pip install --upgrade pip setuptools wheel setuptools_scm

LABEL org.opencontainers.image.source="https://github.com/miguelcarcamov/csromer"
LABEL org.opencontainers.image.description="Base container image for CS-ROMER"
LABEL org.opencontainers.image.licenses=GPL3
