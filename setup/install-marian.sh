#!/bin/bash
# Title: Install Marian
# Description: Commands taken from https://groups.google.com/g/marian-nmt/c/uQC6q1PIf3k/m/JGP_4B5NBwAJ?utm_medium=email&utm_source=footer

apt-get update
apt-get install -y gcc-11 g++-11
apt-get install -y git cmake build-essential libboost-all-dev libprotobuf10 protobuf-compiler libprotobuf-dev openssl libssl-dev libgoogle-perftools-dev doxygen
apt -y install libboost-tools-dev libboost-thread-dev magics++
apt-get install -y cmake
apt-get install libboost-all-dev
apt-get install libpthread-stubs0-dev
sudo apt-get install google-perftools libgoogle-perftools-dev
sudo apt install doxygen
sudo apt install libopenblas-dev
sudo apt-get -y install intel-mkl

# CUDA pre-reqs (Marian Doc. CUDA 10.1, Ubuntu 20.04)
sudo apt-get install git cmake build-essential libboost-system-dev libprotobuf17 protobuf-compiler libprotobuf-dev openssl libssl-dev libgoogle-perftools-dev

# CUDA ChatGPT
apt-get update && apt-get install -y --no-install-recommends \
     cuda-cudart-dev-11-4 \
     cuda-command-line-tools-11-4 \
     libcublas-dev-11-4 \
     libcudnn8-dev \
     && apt-get clean \
     && rm -rf /var/lib/apt/lists/*

sudo apt remove cmake
sudo apt purge --auto-remove cmake
wget https://cmake.org/files/v3.12/cmake-3.12.3-Linux-x86_64.sh
sudo mkdir /opt/cmake
sudo sh cmake-3.12.3-Linux-x86_64.sh --prefix=/opt/cmake --skip-license
rm /usr/local/bin/cmake
sudo ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
git clone https://github.com/marian-nmt/marian
cd marian && git rev-parse --short HEAD
rm -rf build
mkdir -p build
ldconfig
cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_STATIC_LIBS=on -DCOMPILE_CUDA=ON
make -j1 -C marian/build
cd ../..