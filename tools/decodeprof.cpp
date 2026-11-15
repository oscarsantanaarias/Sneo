#include "../src/brain.h"
#include "../src/tokenizer.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <numeric>

double millisSince(std::chrono::steady_clock::time_point from) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - from).count();
}

int bestOf(const std::vector<float>& logits) {
    return (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
}

int main(int argc, char** argv) {
    std::string promptFile = argc > 1 ? argv[1] : "build/gpugen/long.txt";
    int steps = argc > 2 ? atoi(argv[2]) : 96;
    double ram = argc > 3 ? atof(argv[3]) : 24.0;
    std::string modelPath = argc > 13 ? argv[13] : "modelos/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf";
    double vram = argc > 14 ? atof(argv[14]) : 23.5;
    const double gigabyte = (double)(1ULL << 30);

    ModelFile file;
    std::string error;
    Tokenizer tokenizer;
    if(!openModel(modelPath, file, error) || !tokenizer.load(file, error)){
        std::cout << "cannot read the model: " << error << std::endl;
        return 1;
    }
    std::ifstream in(promptFile, std::ios::binary);
    std::stringstream text;
    text << in.rdbuf();
    std::vector<int> prompt = tokenizer.encode(text.str(), true);

    GpuModel gpu;
    gpu.hostBudget = (uint64_t)(ram * gigabyte);
    gpu.vramCap = (uint64_t)(vram * gigabyte);
    gpu.contextLimit = 65536;
    if(argc > 5) gpu.adaptEvery = atoi(argv[5]);
    if(argc > 6) gpu.adaptSwaps = atoi(argv[6]);
    if(argc > 7 && std::string(argv[7]) == "pin") gpu.pinWarm = true;
    if(argc > 8) gpu.warmShare = atoi(argv[8]);
    if(argc > 9 && std::string(argv[9]) == "nolock") gpu.locking = false;
    if(argc > 10 && std::string(argv[10]) != "-") gpu.heatPath = argv[10];
    if(argc > 11) gpu.tailSkip = atoi(argv[11]);
    if(argc > 18) gpu.exchanging = std::string(argv[18]) != "noexchange";
    if(argc > 19) gpu.exchangeBatch = atoi(argv[19]);
    if(argc > 20) gpu.recentFade = (float)atof(argv[20]);
    if(argc > 21) gpu.ramHeadroom = (uint64_t)(atof(argv[21]) * gigabyte);
    if(argc > 22) gpu.cpuPlan = argv[22];
    if(!gpu.load(modelPath, 16)){
        std::cout << "load failed: " << gpu.problem << std::endl;
        return 1;
    }

    size_t freeBytes = 0, totalBytes = 0;
    cudaLink().info(&freeBytes, &totalBytes);
    printf("resident experts per layer %d, vram free after load %.2f GB\n", gpu.residentExperts, freeBytes / gigabyte);

    auto roundTime = [&](const char* when){
        int threads = (int)gpu.pool.threads.size() + 1;
        auto start = std::chrono::steady_clock::now();
        for(int i = 0; i < 200; i++) gpu.pool.run(threads, [](int){});
        printf("empty pool round %s: %.1f us\n", when, millisSince(start) * 1000.0 / 200);
    };
    roundTime("after load");
    std::vector<int> positions(prompt.size());
    std::iota(positions.begin(), positions.end(), 0);
    auto clock = std::chrono::steady_clock::now();
    gpu.profile = argc > 15 && std::string(argv[15]) == "profile";
    int tailTokens = argc > 17 ? std::min((int)prompt.size() - 1, atoi(argv[17])) : 0;
    std::vector<int> promptTrace;
    std::vector<float> logits = gpu.forward(std::vector<int>(prompt.begin(), prompt.end() - tailTokens), std::vector<int>(positions.begin(), positions.end() - tailTokens));
    for(int i = (int)prompt.size() - tailTokens; i < (int)prompt.size() && !logits.empty(); i++){
        logits = gpu.forward({prompt[i]}, {i});
        promptTrace.insert(promptTrace.end(), gpu.hostIds, gpu.hostIds + gpu.layers.size() * expertsUsed);
    }
    if(gpu.profile){
        std::vector<std::pair<double, std::string>> ranked;
        for(auto& k : gpu.kernelMs) ranked.push_back({k.second, k.first});
        std::sort(ranked.rbegin(), ranked.rend());
        for(size_t i = 0; i < ranked.size() && i < 18; i++) printf("  kernel %-32s %8.0f ms\n", ranked[i].second.c_str(), ranked[i].first);
        gpu.profile = false;
    }
    printf("prefill %zu tokens: %.0f ms (streamed experts %llu: copy %.0f ms, read %.0f ms, stream %.0f ms; waiting %.0f ms, embed %.0f ms, ple %.0f ms, layers %.0f ms)\n", prompt.size(), millisSince(clock), (unsigned long long)gpu.streamedCount, gpu.copyMs, gpu.readMs, gpu.streamMs, gpu.waitMs, gpu.embedMs, gpu.pleMs, std::accumulate(gpu.layerMs.begin(), gpu.layerMs.end(), 0.0));
    roundTime("after prefill");

    std::atomic<bool> watching{argc > 4 && std::string(argv[4]) == "watch"};
    std::thread watcher([&]{
        while(watching){
            NodeStats stats = readNode("modelos");
            (void)stats;
            for(int i = 0; i < 10 && watching; i++) Sleep(100);
        }
    });

    std::vector<int> shares;
    std::stringstream picks(argc > 12 ? std::string(argv[12]) : std::to_string(gpu.warmShare));
    for(std::string part; std::getline(picks, part, ',');) shares.push_back(atoi(part.c_str()));
    if(shares.size() > 1){
        gpu.adaptEvery = 0;
        gpu.saveState(0);
    }
    std::vector<float> first = logits;
    for(size_t round = 0; round < shares.size(); round++){
        if(round > 0){
            gpu.restoreState(0);
            for(auto& g : gpu.tokenGraphs){
                if(g.second) cudaQueue().execDestroy(g.second);
            }
            gpu.tokenGraphs.clear();
        }
        gpu.warmShare = shares[round];
        gpu.decodeRouted = 0;
        gpu.decodeResident = 0;
        logits = first;
        printf("warm share %d of 8\n", gpu.warmShare);
        std::vector<double> wall, wait, experts, layers;
        std::vector<int> window;
        std::vector<int> trace;
        int windowStart = 0;
        int next = bestOf(logits);
        for(int step = 0; step < steps; step++){
            if(step == steps - 8){
                gpu.saveState(1);
                windowStart = (int)gpu.tokens.size();
            }
            if(step >= steps - 8) window.push_back(next);
            clock = std::chrono::steady_clock::now();
            logits = gpu.forward({next}, {(int)gpu.tokens.size()});
            wall.push_back(millisSince(clock));
            if(round == 0 && argc > 16) trace.insert(trace.end(), gpu.hostIds, gpu.hostIds + gpu.layers.size() * expertsUsed);
            wait.push_back(gpu.waitMs);
            experts.push_back(gpu.expertMs);
            layers.push_back(std::accumulate(gpu.layerMs.begin(), gpu.layerMs.end(), 0.0));
            if(logits.empty()){
                printf("decode failed: %s\n", gpu.problem.c_str());
                return 1;
            }
            next = bestOf(logits);
        }
        auto mean = [&](const std::vector<double>& v){
            size_t from = v.size() / 3;
            return std::accumulate(v.begin() + from, v.end(), 0.0) / (double)(v.size() - from);
        };
        for(size_t from = 0; from < wall.size(); from += 16){
            size_t to = std::min(wall.size(), from + 16);
            double sum = 0, cpu = 0;
            for(size_t i = from; i < to; i++){
                sum += wall[i];
                cpu += experts[i];
            }
            printf("  tokens %zu-%zu: %.2f ms per token, cpu experts %.2f ms\n", from, to - 1, sum / (to - from), cpu / (to - from));
        }
        printf("decode steady state: %.2f ms per token (%.1f tok/s), host waiting on the gpu %.2f ms, cpu experts %.2f ms, layers %.2f ms, vram hit %.1f%%, exchanges %llu taking %.3f ms per token on the host\n", mean(wall), 1000.0 / mean(wall), mean(wait), mean(experts), mean(layers), gpu.decodeRouted ? 100.0 * gpu.decodeResident / gpu.decodeRouted : 0.0, (unsigned long long)gpu.exchangeCount, gpu.exchangeMs / std::max(1, steps));
        if(round == 0 && argc > 16){
            FILE* f = nullptr;
            fopen_s(&f, argv[16], "wb");
            if(f){
                int head[5] = {(int)gpu.layers.size(), expertsUsed, steps, expertCount, tailTokens};
                fwrite(head, 4, 5, f);
                fwrite(promptTrace.data(), 4, promptTrace.size(), f);
                fwrite(trace.data(), 4, trace.size(), f);
                for(GpuLayer& L : gpu.layers) fwrite(L.slot.data(), 4, L.slot.size(), f);
                fclose(f);
                printf("routing trace written to %s\n", argv[16]);
            }
        }

        for(int n : {1, 2, 3, 4}){
            std::vector<int> ids(window.begin(), window.begin() + n);
            double total = 0;
            for(int trial = 0; trial < 5; trial++){
                gpu.restoreState(1);
                std::vector<int> at(n);
                for(int i = 0; i < n; i++) at[i] = windowStart + i;
                clock = std::chrono::steady_clock::now();
                std::vector<float> rows = n == 1 ? gpu.forward(ids, at) : gpu.forwardAll(ids, at);
                total += millisSince(clock);
            }
            printf("verify %d real token%s: %.2f ms, missed routes %llu, distinct missed experts %llu\n", n, n == 1 ? "" : "s", total / 5, (unsigned long long)gpu.missedRoutes, (unsigned long long)gpu.missedUnique);
            printf("forward of %d token%s: %.2f ms (last one: waiting on the gpu %.2f ms, cpu experts %.2f ms, layers %.2f ms, head %.2f ms, cpu expert rows %llu)\n", n, n == 1 ? "" : "s", total / 5, gpu.waitMs, gpu.expertMs, std::accumulate(gpu.layerMs.begin(), gpu.layerMs.end(), 0.0), gpu.headMs, (unsigned long long)gpu.spilledRows);
        }
    }

    GpuLayer& L = gpu.layers[20];
    std::vector<int> cold;
    for(int e = 0; e < expertCount && cold.size() < 10; e++){
        if(L.slot[e] < 0) cold.push_back(e);
    }
    std::vector<float> input(hidden);
    for(int i = 0; i < hidden; i++) input[i] = (float)((i * 37) % 101 - 50) / 50.0f;
    HostPacked packed;
    packed.fill(input.data(), hidden);
    uint64_t gateSpan = (uint64_t)L.expertGate.rows * L.expertGate.stride;
    uint64_t upSpan = (uint64_t)L.expertUp.rows * L.expertUp.stride;
    uint64_t downSpan = (uint64_t)L.expertDown.rows * L.expertDown.stride;
    std::vector<float> gate(expertWidth), up(expertWidth), out(hidden);
    HostPacked middle;

    auto expert = [&](const uint8_t* g, const uint8_t* u, const uint8_t* d) {
        Matrix mg = L.expertGate, mu = L.expertUp, md = L.expertDown;
        mg.data = g;
        mu.data = u;
        md.data = d;
        hostMultiply(gpu.pool, {{mg, &packed, 0, gate.data(), 1}, {mu, &packed, 0, up.data(), 1}});
        for(int i = 0; i < expertWidth; i++) gate[i] = gate[i] * sigmoid(gate[i]) * up[i];
        middle.fill(gate.data(), expertWidth);
        hostMultiply(gpu.pool, {{md, &middle, 0, out.data(), 1}});
    };

    for(int pass = 0; pass < 2; pass++){
        clock = std::chrono::steady_clock::now();
        for(int e : cold) expert(L.expertGate.data + e * gateSpan, L.expertUp.data + e * upSpan, L.expertDown.data + e * downSpan);
        printf("cpu expert from the mapped model file, pass %d: %.3f ms each\n", pass + 1, millisSince(clock) / cold.size());
    }

    uint64_t perExpert = gateSpan + upSpan + downSpan;
    uint8_t* arena = (uint8_t*)VirtualAlloc(nullptr, perExpert * cold.size(), MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    SetProcessWorkingSetSizeEx(GetCurrentProcess(), 1ULL << 30, 48ULL << 30, QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE);
    bool locked = arena && VirtualLock(arena, perExpert * cold.size());
    for(size_t i = 0; i < cold.size(); i++){
        int e = cold[i];
        memcpy(arena + i * perExpert, L.expertGate.data + e * gateSpan, gateSpan);
        memcpy(arena + i * perExpert + gateSpan, L.expertUp.data + e * upSpan, upSpan);
        memcpy(arena + i * perExpert + gateSpan + upSpan, L.expertDown.data + e * downSpan, downSpan);
    }
    for(int pass = 0; pass < 2; pass++){
        clock = std::chrono::steady_clock::now();
        for(size_t i = 0; i < cold.size(); i++) expert(arena + i * perExpert, arena + i * perExpert + gateSpan, arena + i * perExpert + gateSpan + upSpan);
        printf("cpu expert from private %s memory, pass %d: %.3f ms each\n", locked ? "locked" : "unlocked", pass + 1, millisSince(clock) / cold.size());
    }

    int threads = (int)gpu.pool.threads.size() + 1;
    clock = std::chrono::steady_clock::now();
    for(int i = 0; i < 1000; i++) gpu.pool.run(threads, [](int){});
    printf("empty pool round with %d threads: %.1f us\n", threads, millisSince(clock));
    cudaLink().info(&freeBytes, &totalBytes);
    printf("vram free at the end %.2f GB, graphs %zu, snapshots %d\n", freeBytes / gigabyte, gpu.tokenGraphs.size(), (int)((gpu.snapshots[0] != 0) + (gpu.snapshots[1] != 0) + (gpu.snapshots[2] != 0) + (gpu.snapshots[3] != 0)));
    watching = false;
    watcher.join();
    return 0;
}
