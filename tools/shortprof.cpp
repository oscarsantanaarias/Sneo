#include "../src/brain.h"
#include "../src/tokenizer.h"
#include "../src/chat.h"
#include <iostream>
#include <chrono>
#include <numeric>

double millisSince(std::chrono::steady_clock::time_point from) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - from).count();
}

int main(int argc, char** argv) {
    std::string modelPath = argc > 1 ? argv[1] : "modelos/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf";
    const double gigabyte = (double)(1ULL << 30);
    ModelFile file;
    std::string error;
    Tokenizer tokenizer;
    if(!openModel(modelPath, file, error) || !tokenizer.load(file, error)){
        std::cout << "cannot read the model: " << error << std::endl;
        return 1;
    }
    std::vector<std::string> asks = {
        "<|im_start|>user\nWhat is the capital of France, and why is it famous?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        "<|im_start|>user\nGive me three tips to sleep better at night, short please.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        "<|im_start|>user\nExplain in two lines what a hash table is and when to use one.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    };
    GpuModel gpu;
    gpu.hostBudget = (uint64_t)(24.0 * gigabyte);
    gpu.vramCap = (uint64_t)(23.5 * gigabyte);
    gpu.contextLimit = 65536;
    gpu.pinWarm = true;
    gpu.tailSkip = 7;
    gpu.relayLimit = 9;
    if(!gpu.load(modelPath, 16)){
        std::cout << "load failed: " << gpu.problem << std::endl;
        return 1;
    }
    auto bestOf = [](const std::vector<float>& v){ return (int)(std::max_element(v.begin(), v.end()) - v.begin()); };
    for(int rows : {5, 0}){
        gpu.graphRows = rows > 0 ? rows : 5;
        gpu.shortRead = rows > 0 ? 64 : 0;
        for(auto& g : gpu.tokenGraphs){
            if(g.second) cudaQueue().execDestroy(g.second);
        }
        gpu.tokenGraphs.clear();
        gpu.graphless.clear();
        for(int round = 0; round < 3; round++){
            for(const std::string& ask : asks){
                gpu.reset();
                std::vector<int> ids = tokenizer.encode(ask, false);
                std::vector<int> at(ids.size());
                std::iota(at.begin(), at.end(), 0);
                auto clock = std::chrono::steady_clock::now();
                std::vector<float> logits = gpu.forward(ids, at);
                double read = millisSince(clock);
                int next = bestOf(logits);
                clock = std::chrono::steady_clock::now();
                for(int i = 0; i < 16 && !logits.empty(); i++){
                    logits = gpu.forward({next}, {(int)gpu.tokens.size()});
                    next = bestOf(logits);
                }
                double write = millisSince(clock) / 16;
                if(round == 2) printf("%s: prompt %zu tokens read in %.1f ms, then %.2f ms per token\n", rows > 0 ? "windows of 5 with graphs" : "one batched pass", ids.size(), read, write);
            }
        }
    }
    return 0;
}
