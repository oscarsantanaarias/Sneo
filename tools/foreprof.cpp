#include "../src/brain.h"
#include "../src/tokenizer.h"
#include <iostream>
#include <set>
#include <numeric>

int main(int argc, char** argv) {
    std::string modelPath = argc > 1 ? argv[1] : "modelos/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf";
    int window = argc > 2 ? atoi(argv[2]) : 5;
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
        "<|im_start|>user\nExplain in two lines what a hash table is and when to use one.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        "<|im_start|>user\nWrite a haiku about debugging at three in the morning.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    };
    GpuModel gpu;
    gpu.hostBudget = (uint64_t)(24.0 * gigabyte);
    gpu.vramCap = (uint64_t)(23.5 * gigabyte);
    gpu.contextLimit = 65536;
    gpu.pinWarm = true;
    gpu.tailSkip = 7;
    gpu.relayLimit = 9;
    gpu.exchanging = false;
    gpu.adaptEvery = 0;
    if(!gpu.load(modelPath, 16)){
        std::cout << "load failed: " << gpu.problem << std::endl;
        return 1;
    }
    gpu.foresee = true;
    int layers = (int)gpu.layers.size();
    uint64_t routes = 0, misses = 0, guessed = 0, pulled = 0, pulledUsed = 0, distinctMisses = 0, distinctCovered = 0, windows = 0, distinctAll = 0, distinctHeld = 0, heldRoutes = 0;
    for(const std::string& ask : asks){
        gpu.reset();
        std::vector<int> ids = tokenizer.encode(ask, false);
        for(size_t at = 0; at < ids.size(); at += window){
            int n = (int)std::min((size_t)window, ids.size() - at);
            std::vector<int> part(ids.begin() + at, ids.begin() + at + n), where(n);
            std::iota(where.begin(), where.end(), (int)at);
            std::vector<float> logits = gpu.forward(part, where);
            if(logits.empty()){
                std::cout << "forward failed: " << gpu.problem << std::endl;
                return 1;
            }
            windows++;
            for(int l = 1; l < layers; l++){
                GpuLayer& L = gpu.layers[l];
                const int* real = gpu.hostIds + (size_t)l * expertsUsed * n;
                const int* guess = gpu.hostNext + (size_t)(l - 1) * expertsUsed * n;
                std::set<int> wanted, missing, predicted;
                for(int t = 0; t < n; t++){
                    for(int k = 0; k < expertsUsed; k++){
                        int e = real[t * expertsUsed + k];
                        routes++;
                        if(k < gpu.tailSkip && L.slot[e] < 0){
                            misses++;
                            missing.insert(e);
                            bool seen = false;
                            for(int j = 0; j < expertsUsed; j++) seen = seen || guess[t * expertsUsed + j] == e;
                            guessed += seen ? 1 : 0;
                        }
                        wanted.insert(e);
                    }
                    for(int j = 0; j < gpu.tailSkip; j++){
                        int e = guess[t * expertsUsed + j];
                        if(e >= 0 && e < expertCount && L.slot[e] < 0) predicted.insert(e);
                    }
                }
                distinctAll += wanted.size();
                std::set<int> held;
                for(int e : wanted){
                    if(L.slot[e] >= 0) held.insert(e);
                }
                distinctHeld += held.size();
                for(int t = 0; t < n; t++){
                    for(int k = 0; k < expertsUsed; k++) heldRoutes += L.slot[real[t * expertsUsed + k]] >= 0 ? 1 : 0;
                }
                distinctMisses += missing.size();
                for(int e : missing) distinctCovered += predicted.count(e);
                pulled += predicted.size();
                for(int e : predicted) pulledUsed += missing.count(e);
            }
        }
    }
    printf("windows of %d: %llu windows, per window and layer: %.2f missed routes, %.2f distinct missed experts\n", window, (unsigned long long)windows, (double)misses / windows / (layers - 1), (double)distinctMisses / windows / (layers - 1));
    printf("routes per window and layer: %.1f, distinct experts %.1f; routes to experts in vram %.1f, distinct %.1f (%.0f%% of those reads are repeats)\n", (double)routes / windows / (layers - 1), (double)distinctAll / windows / (layers - 1), (double)heldRoutes / windows / (layers - 1), (double)distinctHeld / windows / (layers - 1), 100.0 * (1.0 - (double)distinctHeld / std::max<uint64_t>(1, heldRoutes)));
    printf("missed routes the next-layer guess saw coming: %.1f%%\n", 100.0 * guessed / std::max<uint64_t>(1, misses));
    printf("distinct missed experts covered by the guess: %.1f%%\n", 100.0 * distinctCovered / std::max<uint64_t>(1, distinctMisses));
    printf("experts the guess would copy per window and layer: %.2f, of which really missed: %.1f%%\n", (double)pulled / windows / (layers - 1), 100.0 * pulledUsed / std::max<uint64_t>(1, pulled));
    return 0;
}
