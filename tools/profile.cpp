#include "../src/qwen4.h"
#include "../src/gpumodel.h"
#include "../src/tokenizer.h"
#include "../src/chat.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <chrono>

namespace fs = std::filesystem;

struct HeatHeader {
    char tag[4];
    uint32_t version;
    uint32_t layers;
    uint32_t experts;
    uint64_t tokens;
};

std::string readSlice(const std::string& path, size_t from, size_t limit) {
    std::ifstream file(path, std::ios::binary);
    std::stringstream text;
    text << file.rdbuf();
    std::string all = text.str();
    if(from >= all.size()) return "";
    return all.substr(from, limit);
}

std::string turn(const std::string& role, const std::string& text) {
    return "{\"role\":\"" + role + "\",\"content\":" + quoteJson(text) + "}";
}

std::string buildCorpus(const std::string& root) {
    std::string tools = readSlice(root + "\\ui\\server\\toolSchemas.js", 0, 6000);
    std::string list = "["
        + turn("system", "You are SNEO, a coding and reverse engineering agent. Use the tools to read, edit and run code in the user's project. Tools available:\n" + tools) + ","
        + turn("user", "Revisa este archivo y dime por que el pool no conecta con la laptop, despues arreglalo:\n" + readSlice(root + "\\master.cpp", 0, 7000)) + ","
        + turn("assistant", "I will read the networking part first and then check how the master connects to the node, then fix the handshake.") + ","
        + turn("user", "Now update this React view so the priority rows use GB by default and keep the layout clean:\n" + readSlice(root + "\\ui\\src\\PoolView.jsx", 0, 7000)) + ","
        + turn("assistant", "The rows read the mode from the slot, so I will default new rows to GB and convert through the live limit.") + ","
        + turn("user", "Explain what this CUDA kernel source does and find the slowest part:\n" + readSlice(root + "\\src\\kernels.h", 0, 7000)) + ","
        + turn("assistant", "The row kernels give each warp one output row and each lane a group of 32 weights, so short rows leave lanes idle.") + ","
        + turn("user", "Optimiza este bucle del agente, se queda colgado cuando una herramienta falla:\n" + readSlice(root + "\\ui\\server\\agents.js", 20000, 7000)) + ","
        + turn("assistant", "El error de la herramienta vuelve al modelo como resultado y el bucle sigue hasta el limite de vueltas.") + ","
        + turn("user", "Here is the forward pass, tell me where the hyper connection mix happens:\n" + readSlice(root + "\\src\\qwen4.h", 15000, 7000)) + ","
        + turn("assistant", "The mix reads the four normalized streams, applies the gate and averages them before the block runs.") + ","
        + turn("user", "Add dark mode tokens to this stylesheet without changing the layout:\n" + readSlice(root + "\\ui\\src\\style.css", 0, 5000)) + ","
        + turn("assistant", "I will define the colors as variables on the root and switch them under the dark scheme media query.") + ","
        + turn("user", "Lee mis reglas de estilo y dime si este archivo las cumple:\n" + readSlice(root + "\\MiEstilo.MD", 0, 3000) + "\n\n" + readSlice(root + "\\src\\serve.h", 0, 5000)) + ","
        + turn("assistant", "Cumple casi todo: nombres en ingles, sin comentarios y returns tempranos. Hay dos if con espacio que conviene arreglar.") + ","
        + turn("user", "Decompile notes: the function at 0x00401A20 walks a linked list of actors and calls a virtual method at offset 0x34 on each one, then compares the result with a float threshold before spawning an effect. What is it most likely doing in a game client?") + ","
        + turn("assistant", "It looks like a visibility or distance check per actor: the virtual call returns the distance to the camera and the threshold decides whether the effect is spawned.")
        + "]";

    Json messages = parseJson(list);
    ChatOptions options;
    std::string prompt, error;
    renderChat(messages, Json(), options, prompt, error);
    return prompt;
}

void report(const std::vector<uint64_t>& counts, uint32_t layers, uint32_t experts) {
    double quarter = 0, half = 0;
    int counted = 0;
    for(uint32_t l = 0; l < layers; l++){
        std::vector<uint64_t> sorted(counts.begin() + (size_t)l * experts, counts.begin() + (size_t)(l + 1) * experts);
        std::sort(sorted.begin(), sorted.end(), std::greater<uint64_t>());
        uint64_t total = 0, top = 0, topHalf = 0;
        for(size_t e = 0; e < sorted.size(); e++){
            total += sorted[e];
            if(e < sorted.size() / 4) top += sorted[e];
            if(e < sorted.size() / 2) topHalf += sorted[e];
        }
        if(total == 0) continue;
        quarter += (double)top / total;
        half += (double)topHalf / total;
        counted++;
    }
    if(!counted) return;
    std::cout << "hottest 25% of experts serve " << 100.0 * quarter / counted << "% of routes on average" << std::endl;
    std::cout << "hottest 50% of experts serve " << 100.0 * half / counted << "% of routes on average" << std::endl;
}

int main(int argc, char** argv) {
    std::string modelPath = "modelos\\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf";
    std::string root = ".";
    size_t limit = 16384;
    int threads = 16;
    bool onGpu = false;
    double budget = 8;

    for(int i = 1; i < argc; i++){
        std::string flag = argv[i];
        if(flag == "--gpu") onGpu = true;
        else if(i + 1 >= argc) break;
        else if(flag == "--model") modelPath = argv[++i];
        else if(flag == "--root") root = argv[++i];
        else if(flag == "--tokens") limit = (size_t)atoll(argv[++i]);
        else if(flag == "--threads") threads = atoi(argv[++i]);
        else if(flag == "--vram") budget = atof(argv[++i]);
    }

    ModelFile file;
    std::string error;
    Tokenizer tokenizer;
    if(!openModel(modelPath, file, error) || !tokenizer.load(file, error)){
        std::cout << "cannot read the model: " << error << std::endl;
        return 1;
    }

    std::vector<int> ids = tokenizer.encode(buildCorpus(root), true);
    if(ids.size() > limit) ids.resize(limit);
    std::vector<int> positions(ids.size());
    for(size_t i = 0; i < ids.size(); i++) positions[i] = (int)i;
    std::cout << "profiling " << ids.size() << " tokens on " << (onGpu ? "gpu" : "cpu") << std::endl;

    fs::create_directories(root + "\\profiles");
    std::string out = root + "\\profiles\\" + fs::path(modelPath).stem().string() + ".heat";

    if(onGpu){
        GpuModel gpu;
        gpu.heatPath = out + ".none";
        gpu.contextLimit = std::max(32768, (int)ids.size() + 64);
        gpu.expertBudget = (uint64_t)(budget * (double)(1ULL << 30));
        if(!gpu.load(modelPath, threads)){
            std::cout << "cannot load on the gpu: " << gpu.problem << std::endl;
            return 1;
        }
        auto start = std::chrono::steady_clock::now();
        if(gpu.forward(ids, positions).empty()){
            std::cout << "gpu forward failed: " << gpu.problem << std::endl;
            return 1;
        }
        double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::cout << "done in " << seconds << " s (" << ids.size() / seconds << " tok/s)" << std::endl;
        uint32_t layers = (uint32_t)gpu.layers.size();
        uint32_t experts = (uint32_t)(gpu.expertHits.size() / std::max<size_t>(1, layers));
        if(!gpu.writeHeat(out, gpu.expertHits, ids.size())){
            std::cout << "cannot write " << out << std::endl;
            return 1;
        }
        std::cout << "saved " << out << std::endl;
        report(gpu.expertHits, layers, experts);
        return 0;
    }

    Model model;
    if(!model.load(modelPath, threads)){
        std::cout << "cannot load the model: " << model.problem << std::endl;
        return 1;
    }
    auto start = std::chrono::steady_clock::now();
    model.forward(ids, positions);
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << "done in " << seconds << " s (" << ids.size() / seconds << " tok/s)" << std::endl;

    uint32_t layers = (uint32_t)model.layers.size();
    uint32_t experts = 0;
    for(const Layer& L : model.layers) experts = std::max<uint32_t>(experts, (uint32_t)L.routeCount.size());
    std::vector<uint64_t> counts((size_t)layers * experts, 0);
    for(uint32_t l = 0; l < layers; l++){
        for(size_t e = 0; e < model.layers[l].routeCount.size(); e++) counts[(size_t)l * experts + e] = model.layers[l].routeCount[e];
    }

    std::ofstream heat(out, std::ios::binary | std::ios::trunc);
    HeatHeader header = {{'H', 'E', 'A', 'T'}, 1, layers, experts, (uint64_t)ids.size()};
    heat.write((const char*)&header, sizeof(header));
    heat.write((const char*)counts.data(), counts.size() * sizeof(uint64_t));
    std::cout << "saved " << out << std::endl;
    report(counts, layers, experts);
    return 0;
}
