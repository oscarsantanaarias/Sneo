#include "../src/hardware.h"
#include "../src/hostdot.h"
#include <cstdio>
#include <random>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if(argc < 2){
        printf("hostbench model.gguf [layer] [experts per layer] [layers] [tokens] [threads] [residual] [repeats] [cpu plan]\n");
        return 1;
    }
    std::string path = argv[1];
    int layer = argc > 2 ? atoi(argv[2]) : 20;
    int perLayer = argc > 3 ? atoi(argv[3]) : 4;
    int depth = argc > 4 ? atoi(argv[4]) : 48;
    int tokens = argc > 5 ? atoi(argv[5]) : 1;
    int threads = argc > 6 ? atoi(argv[6]) : 16;
    bool residual = argc > 7 ? atoi(argv[7]) != 0 : true;
    int repeats = argc > 8 ? atoi(argv[8]) : 10;
    std::string plan = argc > 9 ? argv[9] : "all";

    ModelFile file;
    std::string error;
    if(!openModel(path, file, error)){
        printf("%s\n", error.c_str());
        return 1;
    }
    const char* names[3] = {"ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"};
    Matrix shape[3];
    uint64_t span[3] = {};
    std::vector<uint8_t*> held(3, nullptr);
    int loaded = perLayer * depth;
    for(int p = 0; p < 3; p++){
        const TensorInfo* t = findTensor(file, "blk." + std::to_string(layer) + "." + names[p] + ".weight");
        if(!t){
            printf("no %s in layer %d\n", names[p], layer);
            return 1;
        }
        TypeInfo info = typeInfo(t->type);
        int cols = (int)t->dims[0], rows = (int)t->dims[1], experts = (int)t->dims[2];
        if(loaded > experts) loaded = experts;
        shape[p].type = t->type;
        shape[p].cols = cols;
        shape[p].rows = rows;
        shape[p].stride = (uint64_t)cols / info.blockElements * info.blockBytes;
        span[p] = (uint64_t)rows * shape[p].stride;
        held[p] = (uint8_t*)VirtualAlloc(nullptr, (size_t)(span[p] * loaded), MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        FILE* f = nullptr;
        fopen_s(&f, file.paths[t->file].c_str(), "rb");
        if(!f || !held[p]){
            printf("cannot read %s\n", names[p]);
            return 1;
        }
        _fseeki64(f, (long long)t->offset, SEEK_SET);
        fread(held[p], 1, (size_t)(span[p] * loaded), f);
        fclose(f);
        printf("%s: type %u (%s), %d x %d, %.2f MB per expert\n", names[p], t->type, info.name, rows, cols, span[p] / 1048576.0);
    }
    int hidden = shape[0].cols, width = shape[0].rows;
    perLayer = std::min(perLayer, loaded);
    depth = loaded / perLayer;

    ThreadPool pool;
    std::vector<GROUP_AFFINITY> places = coresFor(plan);
    if(places.size() > 1){
        int count = std::min((int)places.size(), threads);
        SetThreadGroupAffinity(GetCurrentThread(), &places[0], nullptr);
        pool.start(count - 1, std::vector<GROUP_AFFINITY>(places.begin() + 1, places.begin() + count));
        threads = count;
    } else pool.start(std::max(0, threads - 1));
    std::mt19937 random(7);
    std::normal_distribution<float> spread(0.0f, 1.0f);
    std::vector<float> x((size_t)tokens * hidden), gate((size_t)perLayer * tokens * width), up(gate.size()), out((size_t)perLayer * tokens * hidden);
    for(float& v : x) v = spread(random);
    HostPacked input, middle;
    double phase[5] = {};
    auto mark = [](std::chrono::steady_clock::time_point& at){
        auto now = std::chrono::steady_clock::now();
        double us = std::chrono::duration<double, std::micro>(now - at).count();
        at = now;
        return us;
    };

    auto step = [&](int l){
        auto at = std::chrono::steady_clock::now();
        input.fill(x.data(), tokens * hidden);
        phase[0] += mark(at);
        std::vector<HostProduct> work;
        for(int c = 0; c < perLayer; c++){
            int e = l * perLayer + c;
            Matrix g = shape[0], u = shape[1];
            g.data = held[0] + (uint64_t)e * span[0];
            u.data = held[1] + (uint64_t)e * span[1];
            work.push_back({g, &input, 0, gate.data() + (size_t)c * tokens * width, tokens});
            work.push_back({u, &input, 0, up.data() + (size_t)c * tokens * width, tokens});
        }
        hostMultiply(pool, work, residual);
        phase[1] += mark(at);
        hostSwiglu(gate.data(), up.data(), gate.size());
        middle.fill(gate.data(), perLayer * tokens * width);
        phase[2] += mark(at);
        work.clear();
        for(int c = 0; c < perLayer; c++){
            int e = l * perLayer + c;
            Matrix d = shape[2];
            d.data = held[2] + (uint64_t)e * span[2];
            work.push_back({d, &middle, c * tokens, out.data() + (size_t)c * tokens * hidden, tokens});
        }
        phase[3] += mark(at);
        hostMultiply(pool, work, residual);
        phase[4] += mark(at);
    };

    for(int l = 0; l < depth; l++) step(l);
    for(double& p : phase) p = 0.0;
    double best = 1e30, sum = 0.0;
    std::vector<double> each;
    for(int r = 0; r < repeats; r++){
        auto clock = std::chrono::steady_clock::now();
        for(int l = 0; l < depth; l++){
            auto layerClock = std::chrono::steady_clock::now();
            step(l);
            each.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - layerClock).count());
        }
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - clock).count();
        best = std::min(best, ms);
        sum += ms;
    }
    double steps = (double)repeats * depth;
    printf("per layer phases: input %.2f us, gate and up %.2f, swiglu and pack %.2f, lists %.2f, down %.2f\n", phase[0] / steps, phase[1] / steps, phase[2] / steps, phase[3] / steps, phase[4] / steps);
    std::sort(each.begin(), each.end());
    auto rank = [&](double p){ return each.empty() ? 0.0 : each[std::min(each.size() - 1, (size_t)(p * each.size()))]; };
    printf("per layer: p50 %.1f us, p90 %.1f, p99 %.1f, max %.1f (plan %s)\n", rank(0.5), rank(0.9), rank(0.99), each.empty() ? 0.0 : each.back(), plan.c_str());
    double bytes = (double)(span[0] + span[1] + span[2]) * perLayer * depth;
    double average = sum / repeats;
    double check = 0.0;
    uint64_t print = 1469598103934665603ULL;
    for(float v : out){
        check += v;
        uint32_t bits = 0;
        memcpy(&bits, &v, 4);
        print = (print ^ bits) * 1099511628211ULL;
    }
    printf("%d layers x %d experts, %d tokens, %d threads, residual %d: %.2f ms per pass (best %.2f), %.1f GB/s, %.0f MB read, check %.6g, bits %016llx\n", depth, perLayer, tokens, threads, residual ? 1 : 0, average, best, bytes / (average / 1000.0) / 1e9, bytes / 1048576.0, check, (unsigned long long)print);
    return 0;
}
