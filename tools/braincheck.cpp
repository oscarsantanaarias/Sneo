#include "../src/brain.h"
#include "../src/tokenizer.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>

int main(int argc, char** argv) {
    std::string mode = argc > 1 ? argv[1] : "auto";
    double vram = argc > 2 ? atof(argv[2]) : 1.0;
    std::string source = argc > 3 ? argv[3] : "src/pool.h";
    int steps = argc > 4 ? atoi(argv[4]) : 64;
    std::string modelPath = "modelos/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf";

    ModelFile file;
    std::string error;
    Tokenizer tokenizer;
    if(!openModel(modelPath, file, error) || !tokenizer.load(file, error)){
        std::cout << "cannot read the model: " << error << std::endl;
        return 1;
    }

    std::ifstream in(source, std::ios::binary);
    std::stringstream text;
    text << in.rdbuf();
    std::string prompt = "<|im_start|>user\nExplain what this file does, briefly.\n\n" + text.str() + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    std::vector<int> ids = tokenizer.encode(prompt, true);

    Brain brain;
    auto start = std::chrono::steady_clock::now();
    if(!brain.load(modelPath, mode, 16, (uint64_t)(vram * (double)(1ULL << 30)), 0, (int)ids.size() + steps + 64, "")){
        std::cout << "load failed: " << brain.problem << std::endl;
        return 1;
    }
    double loadSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    std::vector<int> positions(ids.size());
    for(size_t i = 0; i < ids.size(); i++) positions[i] = (int)i;
    auto prefill = std::chrono::steady_clock::now();
    std::vector<float> logits = brain.forward(ids, positions);
    double prefillSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - prefill).count();
    if(logits.empty()){
        std::cout << "prefill failed: " << brain.problem << std::endl;
        return 1;
    }

    int best = (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
    std::string reply = tokenizer.tokenText(best);
    auto decode = std::chrono::steady_clock::now();
    int made = 0;
    for(int i = 0; i < steps; i++){
        if(tokenizer.isStop(best)) break;
        std::vector<int> next = {best};
        std::vector<int> at = {(int)brain.tokens().size()};
        logits = brain.forward(next, at);
        if(logits.empty()){
            std::cout << "decode failed: " << brain.problem << std::endl;
            return 1;
        }
        best = (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
        reply += tokenizer.tokenText(best);
        made++;
    }
    double decodeSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - decode).count();

    std::cout << brain.kind << " backend, experts in vram " << vram << " GB, load " << loadSeconds << " s" << std::endl;
    std::cout << "prefill " << ids.size() << " tokens in " << prefillSeconds << " s = " << ids.size() / prefillSeconds << " tok/s" << std::endl;
    std::cout << "decode " << made << " tokens in " << decodeSeconds << " s = " << made / decodeSeconds << " tok/s" << std::endl;
    std::cout << "reply start: " << tokenizer.decode(tokenizer.encode(reply.substr(0, 400), false)) << std::endl;
    return 0;
}
