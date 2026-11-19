#!/bin/sh
# compila el motor para linux: g++ 11 o mas nuevo, el driver de nvidia (libcuda) y nvrtc y cublas de cuda 12
set -e
cd "$(dirname "$0")"
if ! command -v g++ >/dev/null 2>&1; then
    echo "g++ was not found. On Ubuntu or Debian run: sudo apt install build-essential"
    exit 1
fi
mkdir -p build
flags="-std=c++17 -O2 -mavx2 -mfma -mf16c -pthread -w -static-libstdc++ -static-libgcc"

echo "Building the engine tools..."
for t in fetchdraft draftbuild repack; do
    g++ $flags "tools/$t.cpp" -o "build/$t" -ldl
done

echo "Building the engine..."
g++ $flags master.cpp -o build/master -ldl
g++ $flags worker.cpp -o build/worker -ldl
cp -f build/master master
echo "Done: master, worker and the tools are up to date."
if ! ldconfig -p 2>/dev/null | grep -q libnvrtc.so.12; then
    echo "libnvrtc.so.12 was not found: install the cuda 12 runtime (for example sudo apt install nvidia-cuda-toolkit or the cuda-nvrtc-12 and libcublas-12 packages)"
fi
