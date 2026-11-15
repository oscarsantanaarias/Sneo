#include "../src/gpumodel.h"
#include "../src/tokenizer.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <algorithm>

std::string readAll(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::stringstream text;
    text << in.rdbuf();
    return text.str();
}

int main(int argc, char** argv) {
    if(argc < 3){
        std::cout << "usage: prefillprof model.gguf tokens [root] [graphs]" << std::endl;
        return 1;
    }
    std::string modelPath = argv[1];
    size_t wanted = (size_t)atoi(argv[2]);
    std::string root = argc > 3 ? argv[3] : ".";

    ModelFile file;
    std::string error;
    Tokenizer tokenizer;
    if(!openModel(modelPath, file, error) || !tokenizer.load(file, error)){
        std::cout << "cannot read the model: " << error << std::endl;
        return 1;
    }
    std::string text;
    for(const char* name : {"\\master.cpp", "\\src\\gpumodel.h", "\\src\\gpukernels.h", "\\src\\serve.h", "\\src\\kernels.h"}) text += readAll(root + name) + "\n\n";
    std::vector<int> ids = tokenizer.encode(text, true);
    if(ids.size() > wanted) ids.resize(wanted);
    std::vector<int> positions(ids.size());
    for(size_t i = 0; i < ids.size(); i++) positions[i] = (int)i;

    GpuModel gpu;
    gpu.contextLimit = (int)ids.size() + 256;
    gpu.adaptEvery = 0;
    if(!gpu.load(modelPath, 16)){
        std::cout << "load failed: " << gpu.problem << std::endl;
        return 1;
    }
    std::vector<int> warm(ids.begin(), ids.begin() + std::min<size_t>(ids.size(), 512));
    std::vector<int> warmAt(positions.begin(), positions.begin() + warm.size());
    gpu.forward(warm, warmAt);
    gpu.reset();

    auto clock = std::chrono::steady_clock::now();
    if(gpu.forward(ids, positions).empty()){
        std::cout << "forward failed: " << gpu.problem << std::endl;
        return 1;
    }
    double plainMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - clock).count();
    gpu.reset();

    gpu.profile = true;
    gpu.kernelMs.clear();
    gpu.kernelCalls.clear();
    if(gpu.forward(ids, positions).empty()){
        std::cout << "profiled forward failed: " << gpu.problem << std::endl;
        return 1;
    }
    std::vector<std::pair<double, std::string>> ranked;
    double total = 0;
    for(auto& k : gpu.kernelMs){
        ranked.push_back({k.second, k.first});
        total += k.second;
    }
    std::sort(ranked.rbegin(), ranked.rend());
    printf("%zu tokens read in %.0f ms (%.0f tok/s), kernels add up to %.0f ms when timed one by one\n", ids.size(), plainMs, ids.size() * 1000.0 / plainMs, total);
    for(size_t i = 0; i < ranked.size() && i < 25; i++) printf("  %-36s %9.1f ms %5.1f%%  %7d calls\n", ranked[i].second.c_str(), ranked[i].first, 100.0 * ranked[i].first / total, gpu.kernelCalls[ranked[i].second]);
    return 0;
}
