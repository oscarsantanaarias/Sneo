#include "../src/brain.h"
#include "../src/draft.h"
#include "../src/tokenizer.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>

int bestOf(const float* row, size_t count) {
    return (int)(std::max_element(row, row + count) - row);
}

int main(int argc, char** argv) {
    std::string source = argc > 1 ? argv[1] : "src/pool.h";
    int steps = argc > 2 ? atoi(argv[2]) : 48;
    int slice = argc > 3 ? atoi(argv[3]) : 3000;
    double ram = argc > 4 ? atof(argv[4]) : 10.0;
    bool onCpu = argc > 5 && std::string(argv[5]) == "cpu";
    int bits = argc > 6 ? atoi(argv[6]) : 2;
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
    std::string prompt = "<|im_start|>user\nExplain what this code does, step by step.\n\n" + text.str().substr(0, (size_t)slice) + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    std::vector<int> sequence = tokenizer.encode(prompt, true);
    size_t promptLength = sequence.size();

    Model cpu;
    cpu.keepRows = onCpu;
    if(!cpu.load(modelPath, 16)){
        std::cout << "cpu load failed: " << cpu.problem << std::endl;
        return 1;
    }

    std::unique_ptr<GpuModel> gpu;
    if(!onCpu){
        gpu = std::make_unique<GpuModel>();
        gpu->hostBudget = (uint64_t)(ram * (double)(1ULL << 30));
        gpu->contextLimit = (int)promptLength + steps + 64;
        if(!gpu->load(modelPath, 16)){
            std::cout << "gpu load failed: " << gpu->problem << std::endl;
            return 1;
        }
    }

    std::vector<float> rows;
    auto run = [&](const std::vector<int>& ids, const std::vector<int>& at) {
        if(onCpu) return cpu.forward(ids, at);
        std::vector<float> out = gpu->forward(ids, at);
        if(out.empty() || !gpu->residualRows) return std::vector<float>();
        size_t first = rows.size();
        size_t count = (size_t)gpu->residualCount * hcWidth;
        rows.resize(first + count);
        if(cudaLink().copyDown(rows.data() + first, gpu->residualRows, count * 4) != 0) return std::vector<float>();
        return out;
    };

    auto started = std::chrono::steady_clock::now();
    std::vector<int> positions(promptLength);
    for(size_t i = 0; i < promptLength; i++) positions[i] = (int)i;
    std::vector<float> logits = run(sequence, positions);
    if(onCpu) rows = cpu.rows;
    if(logits.empty() || rows.size() != promptLength * hcWidth){
        std::cout << "prefill gave " << rows.size() / hcWidth << " residual rows for " << promptLength << " tokens" << std::endl;
        return 1;
    }
    std::cout << "prefill " << promptLength << " tokens in " << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() << " s" << std::endl;

    for(int i = 0; i < steps; i++){
        int next = bestOf(logits.data(), logits.size());
        sequence.push_back(next);
        if(tokenizer.isStop(next)) break;
        logits = run({next}, {(int)sequence.size() - 1});
        if(logits.empty()){
            std::cout << "decode failed" << std::endl;
            return 1;
        }
    }
    if(onCpu) rows = cpu.rows;
    if(!tokenizer.isStop(sequence.back())) sequence.push_back(bestOf(logits.data(), logits.size()));

    std::string reply;
    for(size_t i = promptLength; i < sequence.size(); i++) reply += tokenizer.tokenText(sequence[i]);
    std::cout << "prompt " << promptLength << " tokens, reply " << sequence.size() - promptLength << " tokens:" << std::endl << reply << std::endl;

    int cells = (int)(rows.size() / hcWidth);
    if(cells != (int)sequence.size() - 1){
        std::cout << "row count " << cells << " does not match " << sequence.size() - 1 << " fed tokens" << std::endl;
        return 1;
    }

    DraftModel draft;
    if(!draft.load(draftRuntimePath("modelos", bits))){
        std::cout << "draft load failed: " << draft.problem << std::endl;
        return 1;
    }

    std::vector<int> next(cells), places(cells);
    for(int i = 0; i < cells; i++){
        next[i] = sequence[i + 1];
        places[i] = i;
    }
    int firstLogit = (int)promptLength - 2;
    size_t vocab = (size_t)cpu.head.rows;
    for(int variant = 0; variant < 2; variant++){
        draft.reset();
        draft.streamNorm = variant == 1;
        std::vector<float> residual, guesses;
        auto start = std::chrono::steady_clock::now();
        draft.run(cpu, rows.data(), next.data(), places.data(), cells, residual, guesses, firstLogit);
        double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        int agree = 0, total = 0;
        for(int i = firstLogit; i <= cells - 2; i++){
            int guess = bestOf(guesses.data() + (size_t)(i - firstLogit) * vocab, vocab);
            agree += guess == sequence[i + 2];
            total++;
        }
        std::cout << (variant == 0 ? "one norm over all streams" : "one norm per stream") << ": draft agrees with the model on " << agree << " of " << total << " next-next tokens (" << (total ? 100.0 * agree / total : 0.0) << "%), " << cells << " cells in " << seconds << " s" << std::endl;
    }
    return 0;
}
