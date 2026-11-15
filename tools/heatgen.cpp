#include "../src/gpumodel.h"
#include "../src/tokenizer.h"
#include "../src/chat.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <chrono>

namespace fs = std::filesystem;

std::string readSlice(const std::string& path, size_t from, size_t limit) {
    std::ifstream file(path, std::ios::binary);
    std::stringstream text;
    text << file.rdbuf();
    std::string all = text.str();
    if(from >= all.size()) return "";
    return all.substr(from, limit);
}

int main(int argc, char** argv) {
    std::string root = argc > 1 ? argv[1] : ".";
    int perPrompt = argc > 2 ? atoi(argv[2]) : 320;
    double ram = argc > 3 ? atof(argv[3]) : 25.0;
    std::string modelPath = root + "\\modelos\\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf";
    std::string out = argc > 4 ? argv[4] : root + "\\profiles\\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.heat";

    std::vector<std::string> asks = {
        "Explain step by step what this C++ code does and how the pieces connect:\n\n" + readSlice(root + "\\worker.cpp", 0, 6000),
        "Explain what this header is for and walk me through the main functions:\n\n" + readSlice(root + "\\src\\gguf.h", 0, 6000),
        "Review this code, list the risky parts and suggest fixes:\n\n" + readSlice(root + "\\src\\net.h", 0, 5000),
        "Explain how this React component works and what state it keeps:\n\n" + readSlice(root + "\\ui\\src\\EngineView.jsx", 0, 6000),
        "Explain this Node.js module and how a request travels through it:\n\n" + readSlice(root + "\\ui\\server\\agents.js", 0, 6000),
        "Explica en espanol que hace este codigo y como lo mejorarias:\n\n" + readSlice(root + "\\src\\tokenizer.h", 0, 5000),
        "Write a C++ function that parses a CSV line with quoted fields and returns the columns. Explain the edge cases.",
        "Write a Python script that watches a folder and prints new files, with error handling, and explain it.",
        "Write a small React component for a searchable list with keyboard navigation and explain the choices.",
        "Implement an LRU cache in Rust with get and put in O(1) and explain how it works.",
        "What is the difference between a mutex and a spinlock, and when would you use each? Give examples.",
        "Explain how a mixture of experts language model routes tokens and why it saves compute.",
        "Explain how speculative decoding works in language model inference and when it pays off.",
        "Hola, puedes explicarme como funciona la memoria virtual en Windows y que es el working set?",
        "Necesito un plan para aprender CUDA en un mes, con ejercicios por semana.",
        "Escribe un correo corto a un cliente explicando que la entrega se retrasa una semana y por que.",
        "I found this in a decompiled game client: a function walks a list of actors, calls a virtual method at offset 0x34 and compares a float. What is it likely doing and how would you confirm it?",
        "Explain how x86 calling conventions work on Windows x64 and how arguments are passed.",
        "Summarize the main ideas of the actor model of concurrency and its pros and cons.",
        "Give me a detailed explanation of how TCP congestion control works, with the main algorithms.",
        "Explain this CSS and how to add a dark theme without breaking the layout:\n\n" + readSlice(root + "\\ui\\src\\style.css", 6000, 4000),
        "Debug this: my program crashes only in release builds with optimizations. List the likely causes and how to find them.",
        "Que diferencia hay entre una GPU y un CPU para correr modelos de lenguaje? Explicalo simple.",
        "Write unit tests in JavaScript for a function that formats file sizes like 1.5 GB, and explain each test."
    };

    std::vector<std::string> topics = {
        "hash tables", "garbage collection", "database indexes", "HTTP caching", "TLS handshakes", "virtual memory",
        "CPU caches", "GPU warps and occupancy", "quantization of neural networks", "attention in transformers",
        "React rendering", "async and await in JavaScript", "Rust ownership", "C++ move semantics", "SQL joins",
        "Docker containers", "git rebase versus merge", "binary search trees", "graph shortest paths", "regular expressions",
        "floating point precision", "multithreading deadlocks", "game loops and frame timing", "network packet loss",
        "reverse engineering a Windows executable", "memory leaks", "unit testing strategy", "REST versus gRPC",
        "load balancing", "file systems and journaling", "compression algorithms", "the PE file format"
    };
    std::vector<std::string> styles = {
        "Explain %s to an experienced developer, with a concrete example.",
        "Explica %s en espanol de forma sencilla y con un ejemplo practico.",
        "Write a short code example that shows %s and walk through it line by line.",
        "What are the most common mistakes people make with %s, and how do you avoid them?"
    };
    for(const std::string& topic : topics){
        for(const std::string& style : styles){
            std::string ask = style;
            ask.replace(ask.find("%s"), 2, topic);
            asks.push_back(ask);
        }
    }
    std::vector<std::pair<std::string, size_t>> files = {
        {"\\src\\hardware.h", 0}, {"\\src\\pack.h", 0}, {"\\src\\residency.h", 0}, {"\\src\\json.h", 0},
        {"\\src\\chat.h", 0}, {"\\src\\quant.h", 6000}, {"\\ui\\src\\App.jsx", 30000}, {"\\ui\\server\\tools.js", 10000},
        {"\\ui\\server\\runs.js", 0}, {"\\ui\\src-tauri\\src\\lib.rs", 0}, {"\\tools\\repack.cpp", 0}, {"\\src\\draft.h", 0}
    };
    for(const auto& f : files) asks.push_back("Explain what this code does, how it is organized and anything you would improve:\n\n" + readSlice(root + f.first, f.second, 5000));

    ModelFile file;
    std::string error;
    Tokenizer tokenizer;
    if(!openModel(modelPath, file, error) || !tokenizer.load(file, error)){
        std::cout << "cannot read the model: " << error << std::endl;
        return 1;
    }

    GpuModel gpu;
    gpu.hostBudget = (uint64_t)(ram * (double)(1ULL << 30));
    gpu.contextLimit = 16384;
    gpu.adaptEvery = 0;
    if(!gpu.load(modelPath, 16)){
        std::cout << "load failed: " << gpu.problem << std::endl;
        return 1;
    }

    std::vector<uint64_t> counts(gpu.expertHits.size(), 0);
    uint64_t generated = 0;
    auto started = std::chrono::steady_clock::now();
    for(size_t a = 0; a < asks.size(); a++){
        Json messages = parseJson("[{\"role\":\"user\",\"content\":" + quoteJson(asks[a]) + "}]");
        ChatOptions options;
        std::string prompt;
        renderChat(messages, Json(), options, prompt, error);
        std::vector<int> ids = tokenizer.encode(prompt, true);
        gpu.reset();
        std::vector<int> positions(ids.size());
        for(size_t i = 0; i < ids.size(); i++) positions[i] = (int)i;
        std::vector<float> logits = gpu.forward(ids, positions);
        if(logits.empty()){
            std::cout << "prompt " << a << " failed: " << gpu.problem << std::endl;
            continue;
        }
        std::vector<uint64_t> before = gpu.expertHits;
        int made = 0;
        for(; made < perPrompt; made++){
            int next = (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
            if(tokenizer.isStop(next)) break;
            logits = gpu.forward({next}, {(int)gpu.tokens.size()});
            if(logits.empty()) break;
        }
        for(size_t i = 0; i < counts.size(); i++) counts[i] += gpu.expertHits[i] - before[i];
        generated += made;
        double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        std::cout << "prompt " << a + 1 << " of " << asks.size() << ": " << ids.size() << " tokens in, " << made << " generated, " << generated << " so far, " << seconds << " s" << std::endl;
    }

    if(fs::exists(out)) fs::copy_file(out, out + ".prefill", fs::copy_options::overwrite_existing);
    if(!gpu.writeHeat(out, counts, generated)){
        std::cout << "cannot write " << out << std::endl;
        return 1;
    }
    std::cout << "decode heat profile from " << generated << " generated tokens written to " << out << std::endl;
    return 0;
}
