#!/bin/bash
# arranca sneo en linux: paquetes de la pagina, runtime de nvidia, motor compilado y servidor, y abre la pagina en el navegador
cd "$(dirname "$0")"
root="$(pwd)"

if ! command -v node >/dev/null 2>&1; then
    echo "Node.js is not installed. On Ubuntu or Debian run: sudo apt install nodejs npm"
    exit 1
fi

if [ ! -f ui/node_modules/.package-lock.json ] || [ ui/package.json -nt ui/node_modules/.package-lock.json ]; then
    echo "Installing the Sneo UI packages..."
    (cd ui && PUPPETEER_SKIP_DOWNLOAD=true npm install --no-audit --no-fund) || { echo "npm install failed, check the internet connection and run this again."; exit 1; }
    touch ui/node_modules/.package-lock.json
fi

if [ ! -f ui/dist/index.html ] || [ -n "$(find ui/src -newer ui/dist/index.html -type f -print -quit)" ]; then
    echo "Building the Sneo page..."
    (cd ui && npm run build) || { echo "The Sneo page could not be built."; exit 1; }
fi

if [ ! -f runtime/libnvrtc.so.12 ] && { command -v nvidia-smi >/dev/null 2>&1 || [ -e /dev/dxg ]; }; then
    echo "Downloading the NVIDIA kernel compiler (NVRTC) and cuBLAS..."
    sh tools/get_runtime.sh || echo "The NVIDIA runtime could not be downloaded, the engine will run on the CPU."
fi

# el motor se recompila si el codigo es mas nuevo que el programa
if [ ! -x master ] || [ ! -x worker ] || [ -n "$(find master.cpp worker.cpp src -newer master -type f -print -quit)" ]; then
    if [ -x master ] && [ -x worker ] && ! command -v g++ >/dev/null 2>&1; then
        echo "The engine code is newer than the engine but g++ is not installed, keeping the current engine."
    else
        for pid in $(pgrep -x master); do
            [ "$(readlink -f /proc/$pid/exe 2>/dev/null)" = "$root/master" ] && kill -9 "$pid"
        done
        sh build.sh || { echo "The engine could not be built. On Ubuntu install the compiler with: sudo apt install build-essential"; exit 1; }
    fi
fi

listening() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }

if ! listening 8000; then
    nohup ./master 8000 - - modelos > master.log 2> master.log.err < /dev/null &
fi

sneo_up() { curl -s --max-time 2 http://127.0.0.1:3001/api/sneo 2>/dev/null | grep -q '"sneo":true'; }

# exec deja solo a node con la salida en su archivo, asi la terminal no queda esperando
if ! sneo_up; then
    (cd ui && exec nohup node server/server.js > server.log 2> server.log.err < /dev/null) &
fi

for i in $(seq 1 30); do
    sneo_up && break
    sleep 1
done

echo "Sneo is on http://127.0.0.1:3001"
if command -v xdg-open >/dev/null 2>&1; then
    xdg-open http://127.0.0.1:3001 >/dev/null 2>&1 < /dev/null &
fi
exit 0
