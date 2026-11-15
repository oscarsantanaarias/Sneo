#include "../src/net.h"
#include <mmsystem.h>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <iostream>
#include <random>

#pragma comment(lib, "winmm.lib")

// imita en esta misma pc el enlace entre las dos pcs del pool: ancho de banda y demora de ida en cada sentido
// solo frena la conexion de computo (el hello con pad 1), los pesos y las estadisticas pasan directo
// linksim <puerto de entrada> <puerto del worker> <mbit> <ms de ida> [ms de variacion] [archivo]
// si se da un archivo con "mbit ms variacion" el enlace cambia al momento cada vez que el archivo cambia

struct Shape {
    double bytesPerMicro = 0;
    double delayMicros = 0;
    double jitterMicros = 0;
};

std::mutex shapeLock;
Shape liveShape;

Shape currentShape() {
    std::lock_guard<std::mutex> guard(shapeLock);
    return liveShape;
}

void watchShape(std::string path) {
    std::string seen;
    while(true){
        std::string text;
        FILE* f = nullptr;
        if(fopen_s(&f, path.c_str(), "rb") == 0 && f){
            char buffer[256] = {};
            text.assign(buffer, fread(buffer, 1, sizeof(buffer) - 1, f));
            fclose(f);
        }
        double mbit = 0, delay = -1, jitter = 0;
        if(text != seen && sscanf_s(text.c_str(), "%lf %lf %lf", &mbit, &delay, &jitter) >= 2 && mbit > 0 && delay >= 0 && jitter >= 0){
            seen = text;
            std::lock_guard<std::mutex> guard(shapeLock);
            liveShape.bytesPerMicro = mbit / 8;
            liveShape.delayMicros = delay * 1000;
            liveShape.jitterMicros = jitter * 1000;
            std::cout << "link now " << mbit << " Mbit/s, " << delay << " ms each way, up to " << jitter << " ms more" << std::endl;
        }
        Sleep(200);
    }
}

struct Chunk {
    std::vector<char> data;
    double release = 0;
};

struct Pipe {
    std::mutex lock;
    std::condition_variable moved;
    std::deque<Chunk> queue;
    uint64_t queued = 0;
    bool closed = false;
    uint64_t total = 0;
};

// lo que el enlace guarda en camino: el buffer de envio de un lado y el de recepcion del otro
const uint64_t inFlight = 16ULL << 20;

double clockMicros() {
    static LARGE_INTEGER frequency = []{ LARGE_INTEGER f; QueryPerformanceFrequency(&f); return f; }();
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    return (double)now.QuadPart * 1e6 / (double)frequency.QuadPart;
}

void waitUntil(double when) {
    while(true){
        double left = when - clockMicros();
        if(left <= 0) return;
        if(left > 2000) Sleep(1);
        else if(left > 100) SwitchToThread();
        else _mm_pause();
    }
}

void readSide(SOCKET from, Pipe& pipe, unsigned seed) {
    std::vector<char> buffer(16 << 10);
    std::mt19937 random(seed);
    double busyUntil = 0, lastRelease = 0;
    while(true){
        {
            std::unique_lock<std::mutex> guard(pipe.lock);
            pipe.moved.wait(guard, [&]{ return pipe.queued < inFlight || pipe.closed; });
            if(pipe.closed) break;
        }
        int got = recv(from, buffer.data(), (int)buffer.size(), 0);
        if(got <= 0) break;
        Shape shape = currentShape();
        double now = clockMicros();
        double start = std::max(now, busyUntil);
        busyUntil = start + got / shape.bytesPerMicro;
        double jitter = shape.jitterMicros > 0 ? std::uniform_real_distribution<double>(0, shape.jitterMicros)(random) : 0;
        double release = std::max(lastRelease, busyUntil + shape.delayMicros + jitter);
        lastRelease = release;
        std::lock_guard<std::mutex> guard(pipe.lock);
        pipe.queue.push_back({std::vector<char>(buffer.begin(), buffer.begin() + got), release});
        pipe.queued += got;
        pipe.total += got;
        pipe.moved.notify_all();
    }
    std::lock_guard<std::mutex> guard(pipe.lock);
    pipe.closed = true;
    pipe.moved.notify_all();
}

void writeSide(SOCKET to, Pipe& pipe) {
    while(true){
        Chunk chunk;
        {
            std::unique_lock<std::mutex> guard(pipe.lock);
            pipe.moved.wait(guard, [&]{ return !pipe.queue.empty() || pipe.closed; });
            if(pipe.queue.empty()) break;
            chunk = std::move(pipe.queue.front());
            pipe.queue.pop_front();
        }
        waitUntil(chunk.release);
        bool sent = sendAll(to, chunk.data.data(), chunk.data.size());
        std::lock_guard<std::mutex> guard(pipe.lock);
        pipe.queued -= chunk.data.size();
        pipe.moved.notify_all();
        if(!sent){
            pipe.closed = true;
            break;
        }
    }
    shutdown(to, SD_SEND);
}

void passThrough(SOCKET from, SOCKET to) {
    std::vector<char> buffer(1 << 20);
    while(true){
        int got = recv(from, buffer.data(), (int)buffer.size(), 0);
        if(got <= 0 || !sendAll(to, buffer.data(), got)) break;
    }
    shutdown(to, SD_SEND);
}

void serve(SOCKET client, int targetPort) {
    SOCKET target = connectTcp("127.0.0.1", targetPort, 3000);
    if(target == INVALID_SOCKET){
        std::cout << "nothing answers on port " << targetPort << std::endl;
        closesocket(client);
        return;
    }
    tuneSocket(client);
    PacketHead head = {};
    HelloBody hello = {};
    if(!recvAll(client, &head, sizeof(head)) || head.kind != PACKET_HELLO || head.size != sizeof(hello) || !recvAll(client, &hello, sizeof(hello)) || !sendAll(target, &head, sizeof(head)) || !sendAll(target, &hello, sizeof(hello))){
        closesocket(client);
        closesocket(target);
        return;
    }
    bool compute = hello.pad == 1;
    std::cout << (compute ? "compute link opened, shaped" : "control link opened, passes through") << std::endl;
    if(!compute){
        std::thread up(passThrough, client, target);
        passThrough(target, client);
        up.join();
        closesocket(client);
        closesocket(target);
        return;
    }
    Pipe toWorker, toMaster;
    std::thread readUp(readSide, client, std::ref(toWorker), 7u);
    std::thread writeUp(writeSide, target, std::ref(toWorker));
    std::thread readDown(readSide, target, std::ref(toMaster), 11u);
    writeSide(client, toMaster);
    readDown.join();
    {
        std::lock_guard<std::mutex> guard(toWorker.lock);
        toWorker.closed = true;
        toWorker.moved.notify_all();
    }
    shutdown(client, SD_BOTH);
    shutdown(target, SD_BOTH);
    readUp.join();
    writeUp.join();
    closesocket(client);
    closesocket(target);
    std::cout << "compute link closed: " << toWorker.total / 1048576.0 << " MB to the worker, " << toMaster.total / 1048576.0 << " MB back" << std::endl;
}

int main(int argc, char** argv) {
    if(argc < 5){
        std::cout << "linksim <listen port> <worker port> <mbit> <one way ms> [jitter ms] [file with \"mbit ms jitter\" to change it live]" << std::endl;
        return 1;
    }
    int listenPort = atoi(argv[1]);
    int targetPort = atoi(argv[2]);
    double mbit = atof(argv[3]);
    double delay = atof(argv[4]);
    double jitter = argc > 5 ? atof(argv[5]) : 0;
    if(mbit <= 0 || delay < 0 || jitter < 0){
        std::cout << "the speed must be above 0 and the delays 0 or more" << std::endl;
        return 1;
    }
    liveShape.bytesPerMicro = mbit / 8;
    liveShape.delayMicros = delay * 1000;
    liveShape.jitterMicros = jitter * 1000;
    if(argc > 6) std::thread(watchShape, std::string(argv[6])).detach();
    timeBeginPeriod(1);
    SOCKET door = listenTcp(listenPort, "127.0.0.1");
    if(door == INVALID_SOCKET){
        std::cout << "cannot listen on 127.0.0.1:" << listenPort << std::endl;
        return 1;
    }
    std::cout << "link " << mbit << " Mbit/s, " << delay << " ms each way" << (jitter > 0 ? ", up to " + std::to_string(jitter) + " ms more" : std::string()) << ", 127.0.0.1:" << listenPort << " -> " << targetPort << std::endl;
    while(true){
        SOCKET client = accept(door, nullptr, nullptr);
        if(client == INVALID_SOCKET) continue;
        std::thread(serve, client, targetPort).detach();
    }
}
