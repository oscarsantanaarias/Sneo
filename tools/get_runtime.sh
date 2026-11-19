#!/bin/sh
# el compilador de kernels de nvidia (nvrtc) y cublas para linux, de las mismas ruedas oficiales que usa windows
set -e
cd "$(dirname "$0")/.."
mkdir -p runtime
work=$(mktemp -d)
if [ ! -f runtime/libnvrtc.so.12 ]; then
    python3 -m pip download --no-deps --only-binary=:all: --platform manylinux2014_x86_64 --platform manylinux_2_27_x86_64 nvidia-cuda-nvrtc-cu12==12.9.86 -d "$work/nvrtc" -q
    unzip -q -o "$work"/nvrtc/*.whl -d "$work/nvrtc/x"
    cp -f "$work"/nvrtc/x/nvidia/cuda_nvrtc/lib/libnvrtc.so.12 "$work"/nvrtc/x/nvidia/cuda_nvrtc/lib/libnvrtc-builtins.so.12.9 runtime/
fi
if [ ! -f runtime/libcublasLt.so.12 ]; then
    python3 -m pip download --no-deps --only-binary=:all: --platform manylinux2014_x86_64 --platform manylinux_2_27_x86_64 nvidia-cublas-cu12==12.9.2.10 -d "$work/cublas" -q
    unzip -q -o "$work"/cublas/*.whl -d "$work/cublas/x"
    cp -f "$work"/cublas/x/nvidia/cublas/lib/libcublas.so.12 "$work"/cublas/x/nvidia/cublas/lib/libcublasLt.so.12 runtime/
fi
rm -rf "$work"
echo "runtime ready: $(ls runtime | tr '\n' ' ')"
