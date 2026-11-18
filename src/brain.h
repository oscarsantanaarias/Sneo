#pragma once
#include "qwen4.h"
#include "gpumodel.h"
#include "json.h"
#include <iostream>
#include <memory>
#include <fstream>
#include <sstream>

// un modelo bajado por otra app puede traer su ayudante para adivinar tokens en el mismo manifiesto
inline std::string sharedDraftOf(const std::string& path) {
    namespace fs = std::filesystem;
    fs::path blob(path);
    std::string name = blob.filename().string();
    if(blob.parent_path().filename() != "blobs" || name.rfind("sha256-", 0) != 0) return "";
    std::string digest = "sha256:" + name.substr(7);
    fs::path store = blob.parent_path().parent_path();
    std::error_code walk;
    if(!fs::exists(store / "manifests", walk)) return "";
    for(auto& entry : fs::recursive_directory_iterator(store / "manifests", walk)){
        if(!entry.is_regular_file(walk)) continue;
        std::ifstream in(entry.path(), std::ios::binary);
        std::stringstream text;
        text << in.rdbuf();
        if(text.str().find(digest) == std::string::npos) continue;
        Json manifest = parseJson(text.str());
        std::string model, draft;
        for(const Json& layer : manifest["layers"].items){
            std::string kind = layer["mediaType"].str();
            if(kind == "application/vnd.ollama.image.model") model = layer["digest"].str();
            if(kind == "application/vnd.ollama.image.draft") draft = layer["digest"].str();
        }
        if(model != digest || draft.rfind("sha256:", 0) != 0) continue;
        fs::path found = store / "blobs" / ("sha256-" + draft.substr(7));
        if(fs::exists(found, walk)) return found.string();
    }
    return "";
}

struct Brain {
    std::unique_ptr<Model> cpu;
    std::unique_ptr<GpuModel> gpu;
    std::string problem;
    std::string kind = "off";
    uint64_t ramBudget = 0;
    std::string gpuProblem;
    bool wantDraft = false;
    int draftLength = 3;
    int copyReach = 8;
    std::string cpuPlan = "all";
    int warmShare = 4;
    bool holdRam = true;
    bool ramWhole = false;
    int pinShare = 50;
    double pinGb = 0;
    bool vision = false;
    int visionGpu = -1;
    bool foresee = false;
    int firstLayer = 0;
    bool freshGpu = true;
    bool gpuUsed = false;
    int mainGpu = -1;
    bool useHelper = true;
    bool helperFirst = false;
    int helperGpu = -1;
    double helperGb = 0;
    uint64_t extraKeep = 0;
    bool gpuLost = false;

    bool ready() const {
        return cpu != nullptr || gpu != nullptr;
    }

    std::vector<int>& tokens() {
        return gpu ? gpu->tokens : cpu->tokens;
    }

    void reset() {
        if(gpu){
            gpu->reset();
            return;
        }
        if(cpu) cpu->reset();
    }

    std::vector<float> forward(const std::vector<int>& ids, const std::vector<int>& positions) {
        if(gpu){
            std::vector<float> logits = gpu->forward(ids, positions);
            if(logits.empty()) problem = gpu->problem;
            return logits;
        }
        if(!cpu) return {};
        std::vector<float> logits = cpu->forward(ids, positions);
        if(logits.empty()) problem = cpu->problem;
        return logits;
    }

    bool load(const std::string& path, const std::string& mode, int threads, uint64_t vramCap, uint64_t ramCap, int context, const std::string& heat) {
        cpu.reset();
        gpu.reset();
        bool switched = !cudaLink().ready || pickMainGpu(mainGpu);
        if(gpuUsed && freshGpu){
            if(freshContext()) gpuUsed = false;
            else std::cout << "the gpu context could not be reset before loading, the old one is used" << std::endl;
        }
        problem.clear();
        gpuProblem = switched ? "" : "could not switch to the chosen main gpu";
        kind = "off";

        MEMORYSTATUSEX memory = {};
        memory.dwLength = sizeof(memory);
        GlobalMemoryStatusEx(&memory);
        uint64_t room = memory.ullTotalPhys > (3ULL << 30)
            ? memory.ullTotalPhys - (3ULL << 29)
            : 1ULL << 30;
        if(ramCap > room) ramCap = room;
        ramBudget = ramCap;

        bool wantGpu = mode == "gpu" || (mode != "cpu" && useCuda());
        std::string arch;
        gpuLost = false;
        for(int attempt = 0; wantGpu && attempt < 2; attempt++){
            int alive = cudaLink().ready ? checkContext() : 0;
            if(alive != 0 && !reviveContext()){
                gpuProblem = "the gpu stopped answering (cuda error " + std::to_string(alive) + ") and could not be opened again";
                problem = gpuProblem;
                gpuLost = true;
                break;
            }
            gpuUsed = true;
            auto next = std::make_unique<GpuModel>();
            next->vramCap = vramCap;
            next->extraKeep = extraKeep;
            next->hostBudget = ramCap;
            next->useHelper = useHelper;
            next->helperFirst = helperFirst;
            next->helperOrdinal = helperGpu;
            next->helperCap = (uint64_t)(helperGb * 1073741824.0);
            std::filesystem::path beside = std::filesystem::path(path).parent_path() / "Qwen3.8-Flash-Next.draft-q2.gguf";
            std::filesystem::path shared = std::filesystem::path(runtimeFolder()).parent_path() / "modelos" / "Qwen3.8-Flash-Next.draft-q2.gguf";
            bool blob = std::filesystem::path(path).parent_path().filename() == "blobs";
            if(wantDraft) next->draftPath = blob ? sharedDraftOf(path) : (std::filesystem::exists(beside) ? beside : shared).string();
            next->relayLimit = std::max(2, std::min(33, std::max(draftLength, copyReach) + 1));
            next->graphRows = std::max(5, std::min(9, std::max(draftLength, copyReach) + 1));
            next->foresee = foresee;
            next->cpuPlan = cpuPlan;
            next->warmShare = warmShare;
            next->holdRam = holdRam;
            next->ramWhole = ramWhole;
            next->pinShare = pinShare;
            next->pinGb = pinGb;
            next->vision = vision;
            next->visionGpu = visionGpu;
            next->firstLayer = firstLayer;
            if(context > 0) next->contextLimit = context;
            if(!heat.empty()) next->heatPath = heat;
            next->heatSave = (std::filesystem::path(runtimeFolder()).parent_path() / "profiles" / (std::filesystem::path(path).stem().string() + ".heat")).string();
            if(next->load(path, threads)){
                gpu = std::move(next);
                kind = "gpu";
                return true;
            }
            problem = next->problem;
            gpuProblem = next->problem;
            arch = next->file.arch;
            std::cout << "the model did not load on the gpu: " << problem << std::endl;
            bool pinLost = next->pinLost;
            next.reset();
            if(pinLost){
                gpuLost = true;
                break;
            }
            if(cudaLink().ready && checkContext() != 0){
                if(attempt == 0){
                    gpuProblem = "the gpu context was lost while loading (" + problem + "), it was opened again";
                    continue;
                }
                gpuLost = true;
            }
            break;
        }
        if(stopLoading || gpuLost) return false;
        if(wantGpu && (mode == "gpu" || firstLayer > 0 || (!arch.empty() && arch != "qwen4exp"))) return false;

        auto next = std::make_unique<Model>();
        next->memoryCap = ramCap;
        if(!next->load(path, threads)){
            problem = next->problem;
            return false;
        }
        cpu = std::move(next);
        kind = "cpu";
        return true;
    }
};
