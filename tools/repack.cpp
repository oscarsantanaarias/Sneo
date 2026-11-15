#include "../src/pack.h"
#include <iostream>
#include <random>
#include <chrono>

int main(int argc, char** argv) {
    std::string modelPath = argc > 1 ? argv[1] : "modelos/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf";
    bool readOnly = argc > 2 && std::string(argv[2]) == "--read";

    ModelFile model;
    std::string error;
    if(!openModel(modelPath, model, error)){
        std::cout << error << std::endl;
        return 1;
    }

    std::string path = packPathFor(modelPath);
    if(!readOnly){
        auto start = std::chrono::steady_clock::now();
        bool ok = buildPack(model, path, error, [&](int done, int total){
            double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            printf("\rpacking layer %d of %d (%.0f s)   ", done, total, seconds);
            fflush(stdout);
        });
        printf("\n");
        if(!ok){
            std::cout << "pack failed: " << error << std::endl;
            return 1;
        }
        double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::cout << "pack written to " << path << " in " << seconds << " s" << std::endl;
    }

    ExpertPack pack;
    if(!pack.open(path, model)){
        std::cout << pack.problem << std::endl;
        return 1;
    }

    std::mt19937 random(5);
    int count = 2048;
    std::vector<std::pair<int, int>> wanted;
    uint64_t largest = 0;
    for(int i = 0; i < count; i++){
        int layer = (int)(random() % pack.layers.size());
        wanted.push_back({layer, (int)(random() % pack.experts)});
        largest = std::max(largest, pack.spanOf(layer));
    }

    int batch = 256;
    std::vector<char*> buffers(batch);
    for(char*& b : buffers) b = (char*)VirtualAlloc(nullptr, (size_t)largest, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);

    uint64_t bytes = 0;
    auto start = std::chrono::steady_clock::now();
    for(int at = 0; at < count; at += batch){
        std::vector<std::pair<int, int>> part(wanted.begin() + at, wanted.begin() + std::min(count, at + batch));
        std::vector<char*> targets(buffers.begin(), buffers.begin() + part.size());
        if(!pack.readMany(part, targets)){
            std::cout << "read failed" << std::endl;
            return 1;
        }
        for(auto& w : part) bytes += pack.spanOf(w.first);
    }
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << "random expert reads: " << count << " experts, " << bytes / 1e9 << " GB in " << seconds << " s = " << bytes / seconds / 1e9 << " GB/s, " << count / seconds << " experts/s" << std::endl;

    const PackLayer& p = pack.layers[0];
    std::vector<char> check(p.gateBytes);
    const TensorInfo *gate, *up, *down;
    expertTensors(model, 0, gate, up, down);
    FILE* f = nullptr;
    fopen_s(&f, model.paths[gate->file].c_str(), "rb");
    _fseeki64(f, (long long)(gate->offset + (uint64_t)7 * p.gateBytes), SEEK_SET);
    fread(check.data(), 1, check.size(), f);
    fclose(f);
    std::vector<std::pair<int, int>> one = {{0, 7}};
    std::vector<char*> into = {buffers[0]};
    pack.readMany(one, into);
    std::cout << "layer 0 expert 7 gate slice matches the model file: " << (memcmp(check.data(), buffers[0], check.size()) == 0 ? "yes" : "NO") << std::endl;
    return 0;
}
