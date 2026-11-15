#include "../src/gpumodel.h"
#include "../src/tokenizer.h"
#include <fstream>
#include <psapi.h>
#include <random>
#include <sstream>

const double gigabyte = 1073741824.0;

struct Run {
    std::vector<int> prompt;
    std::vector<float> logits;
    std::vector<int> produced;
    double prefillSeconds = 0;
    double decodeSeconds = 0;
    double waitMs = 0;
    double expertMs = 0;
    double pleMs = 0;
};

struct Prompt {
    std::string name;
    std::vector<int> ids;
};

double secondsSince(std::chrono::steady_clock::time_point from) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - from).count();
}

int best(const std::vector<float>& logits) {
    return (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
}

template<typename Engine>
Run generate(Engine& engine, const std::vector<int>& prompt, int count, bool stepwise) {
    Run run;
    run.prompt = prompt;
    engine.reset();
    std::vector<int> positions(prompt.size());
    std::iota(positions.begin(), positions.end(), 0);

    auto started = std::chrono::steady_clock::now();
    std::vector<float> logits;
    if(stepwise){
        for(size_t i = 0; i < prompt.size(); i++) logits = engine.forward({prompt[i]}, {positions[i]});
    } else {
        logits = engine.forward(prompt, positions);
    }
    run.prefillSeconds = secondsSince(started);
    run.logits = logits;
    if(logits.empty()) return run;

    started = std::chrono::steady_clock::now();
    int position = (int)prompt.size();
    for(int step = 0; step < count; step++){
        int next = best(logits);
        run.produced.push_back(next);
        if(step + 1 == count) break;
        logits = engine.forward({next}, {position++});
        if(logits.empty()) break;
    }
    run.decodeSeconds = secondsSince(started);
    return run;
}

uint64_t fingerprint(const std::vector<int>& ids, int count) {
    uint64_t h = 1469598103934665603ULL ^ (uint64_t)count;
    for(int id : ids){
        h ^= (uint64_t)(uint32_t)id;
        h *= 1099511628211ULL;
    }
    return h;
}

bool readRun(const std::string& path, Run& run) {
    FILE* f = nullptr;
    fopen_s(&f, path.c_str(), "rb");
    if(!f) return false;
    uint64_t sizes[3] = {};
    bool ok = fread(sizes, 8, 3, f) == 3;
    if(ok){
        run.prompt.resize(sizes[0]);
        run.logits.resize(sizes[1]);
        run.produced.resize(sizes[2]);
        ok = fread(run.prompt.data(), 4, sizes[0], f) == sizes[0] && fread(run.logits.data(), 4, sizes[1], f) == sizes[1] && fread(run.produced.data(), 4, sizes[2], f) == sizes[2];
        ok = ok && fread(&run.prefillSeconds, 8, 1, f) == 1 && fread(&run.decodeSeconds, 8, 1, f) == 1;
    }
    fclose(f);
    return ok;
}

void writeRun(const std::string& path, const Run& run) {
    FILE* f = nullptr;
    fopen_s(&f, path.c_str(), "wb");
    if(!f) return;
    uint64_t sizes[3] = {run.prompt.size(), run.logits.size(), run.produced.size()};
    fwrite(sizes, 8, 3, f);
    fwrite(run.prompt.data(), 4, run.prompt.size(), f);
    fwrite(run.logits.data(), 4, run.logits.size(), f);
    fwrite(run.produced.data(), 4, run.produced.size(), f);
    fwrite(&run.prefillSeconds, 8, 1, f);
    fwrite(&run.decodeSeconds, 8, 1, f);
    fclose(f);
}

std::string readText(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::stringstream text;
    text << in.rdbuf();
    return text.str();
}

std::string shown(const std::string& text) {
    std::string out;
    for(char c : text){
        if(c == '\n') out += "\\n";
        else if(c == '\r') out += "\\r";
        else if(c == '\t') out += "\\t";
        else out.push_back(c);
    }
    return out;
}

std::string hostMemory() {
    PROCESS_MEMORY_COUNTERS_EX counters = {};
    counters.cb = sizeof(counters);
    GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&counters, sizeof(counters));
    MEMORYSTATUSEX status = {};
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    char text[160];
    snprintf(text, sizeof(text), "host: working set %.2f GB, private %.2f GB, %.2f GB RAM free", (double)counters.WorkingSetSize / gigabyte, (double)counters.PrivateUsage / gigabyte, (double)status.ullAvailPhys / gigabyte);
    return text;
}

double totalVram() {
    size_t freeBytes = 0, totalBytes = 0;
    if(!useCuda()) return 0;
    cudaLink().info(&freeBytes, &totalBytes);
    return (double)totalBytes / gigabyte;
}

double freeVram() {
    size_t freeBytes = 0, totalBytes = 0;
    if(!useCuda()) return 0;
    cudaLink().info(&freeBytes, &totalBytes);
    return (double)freeBytes / gigabyte;
}

int main(int argc, char** argv) {
    std::string path = "modelos\\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf";
    std::string cacheFolder = "build\\gpugen";
    std::string shortFile = "ui\\node_modules\\@toon-format\\toon\\README.md";
    std::string longFile = "src\\qwen4.h";
    std::string useIn, useOut, only, lockstep, profileName, calibrate, verifyName, commitName;
    int threads = (int)std::thread::hardware_concurrency();
    int count = 32;
    int context = 32768;
    int chunk = 256;
    double vram = 0;
    double ram = 0;
    bool cpuOnly = false;
    bool skipCpu = false;
    bool stepwise = false;
    bool exact = false;
    bool noAdapt = false;
    bool hostFloat = false;
    bool tensorPlainOn = false;
    bool pinOn = false;
    bool checkPlain = false;
    bool deepCheck = false;
    int superRows = 0;
    int draftCount = 8;
    int tailRank = 0;
    bool refeedMode = false;
    bool plainBench = false;
    bool legacyMode = false;
    bool oldRows = false;
    bool singleLevel = false;
    int attendBudget = 0;
    bool noPairs = false;
    bool dumpBatch = false;
    int relayRowsLimit = 0;
    std::string dumpPath, perplexityName, lossPath, textPath;
    int textSpan = 6000;

    for(int i = 1; i < argc; i++){
        std::string arg = argv[i];
        bool more = i + 1 < argc;
        if(arg == "--threads" && more) threads = atoi(argv[++i]);
        else if(arg == "--tokens" && more) count = atoi(argv[++i]);
        else if(arg == "--context" && more) context = atoi(argv[++i]);
        else if(arg == "--chunk" && more) chunk = atoi(argv[++i]);
        else if(arg == "--vram" && more) vram = atof(argv[++i]);
        else if(arg == "--ram" && more) ram = atof(argv[++i]);
        else if(arg == "--cache" && more) cacheFolder = argv[++i];
        else if(arg == "--short" && more) shortFile = argv[++i];
        else if(arg == "--long" && more) longFile = argv[++i];
        else if(arg == "--use" && more) useIn = argv[++i];
        else if(arg == "--save-use" && more) useOut = argv[++i];
        else if(arg == "--only" && more) only = argv[++i];
        else if(arg == "--lockstep" && more) lockstep = argv[++i];
        else if(arg == "--profile" && more) profileName = argv[++i];
        else if(arg == "--calibrate" && more) calibrate = argv[++i];
        else if(arg == "--verify" && more) verifyName = argv[++i];
        else if(arg == "--cpu-only") cpuOnly = true;
        else if(arg == "--no-cpu") skipCpu = true;
        else if(arg == "--stepwise") stepwise = true;
        else if(arg == "--exact") exact = true;
        else if(arg == "--no-adapt") noAdapt = true;
        else if(arg == "--host-float") hostFloat = true;
        else if(arg == "--tensor-plain") tensorPlainOn = true;
        else if(arg == "--pin") pinOn = true;
        else if(arg == "--checkplain") checkPlain = true;
        else if(arg == "--deepcheck") deepCheck = true;
        else if(arg == "--super" && more) superRows = atoi(argv[++i]);
        else if(arg == "--drafts" && more) draftCount = atoi(argv[++i]);
        else if(arg == "--tail" && more) tailRank = atoi(argv[++i]);
        else if(arg == "--commit" && more) commitName = argv[++i];
        else if(arg == "--refeed") refeedMode = true;
        else if(arg == "--plainbench") plainBench = true;
        else if(arg == "--legacy") legacyMode = true;
        else if(arg == "--old-rows") oldRows = true;
        else if(arg == "--single") singleLevel = true;
        else if(arg == "--budget" && more) attendBudget = atoi(argv[++i]);
        else if(arg == "--nopairs") noPairs = true;
        else if(arg == "--dumpbatch") dumpBatch = true;
        else if(arg == "--relay" && more) relayRowsLimit = atoi(argv[++i]);
        else if(arg == "--dump" && more) dumpPath = argv[++i];
        else if(arg == "--perplexity" && more) perplexityName = argv[++i];
        else if(arg == "--losses" && more) lossPath = argv[++i];
        else if(arg == "--text" && more) textPath = argv[++i];
        else if(arg == "--span" && more) textSpan = atoi(argv[++i]);
        else if(arg.rfind("--", 0) != 0) path = arg;
        else {
            printf("usage: gpugen [model.gguf] [--threads T] [--tokens N] [--context N] [--chunk N] [--vram GB] [--ram GB] [--cache folder] [--short file] [--long file] [--use heat] [--save-use heat] [--only france,counting,short,long] [--cpu-only] [--no-cpu] [--stepwise] [--exact] [--no-adapt] [--host-float] [--lockstep prompt] [--profile prompt] [--verify prompt] [--calibrate file,file]\n");
            return 1;
        }
    }
    SetConsoleOutputCP(CP_UTF8);
    setvbuf(stdout, nullptr, _IONBF, 0);

    ModelFile header;
    std::string error;
    Tokenizer words;
    if(!openModel(path, header, error) || !words.load(header, error)){
        printf("cannot read the model: %s\n", error.c_str());
        return 1;
    }

    std::vector<Prompt> prompts;
    prompts.push_back({"france", {760, 6511, 314, 9338, 369}});
    prompts.push_back({"counting", words.encode("Count from 1 to 40: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,")});
    std::filesystem::create_directories(cacheFolder);
    auto pinned = [&](const std::string& source, const std::string& name){
        std::string kept = cacheFolder + "\\" + name;
        std::string text = readText(kept);
        if(!text.empty()) return text;
        text = readText(source);
        std::ofstream(kept, std::ios::binary) << text;
        return text;
    };
    std::vector<int> shortIds = words.encode(pinned(shortFile, "short.txt"));
    std::vector<int> longIds = words.encode(pinned(longFile, "long.txt"));
    if(shortIds.size() < 500 || longIds.size() < 2500){
        printf("prompt files are too short: %zu and %zu tokens\n", shortIds.size(), longIds.size());
        return 1;
    }
    shortIds.resize(500);
    longIds.resize(2500);
    prompts.push_back({"short", shortIds});
    prompts.push_back({"long", longIds});
    if(("," + only + ",").find(",huge,") != std::string::npos){
        std::string text = "Read these files and explain how a request flows from the user interface to the engine and back:\n\n" + readText("master.cpp").substr(0, 9000) + "\n\n" + readText("src\\serve.h").substr(0, 7000) + "\n\n" + readText("ui\\server\\engine.js").substr(0, 3000);
        std::string kept = cacheFolder + "\\huge.txt";
        std::string saved = readText(kept);
        if(saved.empty()) std::ofstream(kept, std::ios::binary) << text;
        else text = saved;
        prompts.push_back({"huge", words.encode(text)});
    }
    if(!textPath.empty()){
        std::vector<int> all = words.encode(readText(textPath));
        for(int c = 0; c < 8 && (size_t)(c + 1) * textSpan <= all.size(); c++) prompts.push_back({"text" + std::to_string(c), std::vector<int>(all.begin() + (size_t)c * textSpan, all.begin() + (size_t)(c + 1) * textSpan)});
    }
    if(!only.empty()){
        std::vector<Prompt> kept;
        for(const Prompt& p : prompts){
            if(("," + only + ",").find("," + p.name + ",") != std::string::npos) kept.push_back(p);
        }
        prompts = kept;
    }
    printf("prompts:");
    for(const Prompt& p : prompts) printf(" %s=%zu", p.name.c_str(), p.ids.size());
    printf("\n");

    std::filesystem::create_directories(cacheFolder);
    std::vector<Run> reference(prompts.size());
    std::vector<bool> haveReference(prompts.size(), false);
    if(!skipCpu){
        std::unique_ptr<Model> cpu;
        for(size_t i = 0; i < prompts.size(); i++){
            char name[64];
            snprintf(name, sizeof(name), "\\%016llx.bin", (unsigned long long)fingerprint(prompts[i].ids, count));
            std::string cached = cacheFolder + name;
            if(readRun(cached, reference[i]) && reference[i].prompt == prompts[i].ids){
                haveReference[i] = true;
                printf("cpu %s: cached reference, prefill %.1f s, decode %.2f tok/s\n", prompts[i].name.c_str(), reference[i].prefillSeconds, (count - 1) / std::max(1e-9, reference[i].decodeSeconds));
                continue;
            }
            if(!cpu){
                cpu = std::make_unique<Model>();
                if(!cpu->load(path, threads)){
                    printf("cpu load failed: %s\n", cpu->problem.c_str());
                    return 1;
                }
            }
            reference[i] = generate(*cpu, prompts[i].ids, count, false);
            if(reference[i].logits.empty()){
                printf("cpu forward failed for %s\n", prompts[i].name.c_str());
                return 1;
            }
            haveReference[i] = true;
            writeRun(cached, reference[i]);
            printf("cpu %s: prefill %.1f s (%.1f tok/s), decode %.2f tok/s: %s\n", prompts[i].name.c_str(), reference[i].prefillSeconds, prompts[i].ids.size() / reference[i].prefillSeconds, (count - 1) / std::max(1e-9, reference[i].decodeSeconds), shown(words.decode(reference[i].produced)).c_str());
        }
    }
    if(cpuOnly) return 0;

    double freeBefore = freeVram();
    auto started = std::chrono::steady_clock::now();
    GpuModel gpu;
    gpu.contextLimit = context;
    gpu.chunk = chunk;
    gpu.exact = exact;
    if(ram > 0) gpu.hostBudget = (uint64_t)(ram * gigabyte);
    gpu.hostInt8 = !hostFloat;
    if(tensorPlainOn) gpu.tensorPlain = true;
    if(superRows > 0) gpu.superLimit = superRows;
    gpu.tailSkip = tailRank;
    if(pinOn) gpu.pinWarm = true;
    if(vram > 0) gpu.expertBudget = (uint64_t)(vram * gigabyte);
    if(!useIn.empty()) gpu.heatPath = useIn;
    if(noAdapt) gpu.adaptEvery = 0;
    gpu.legacy = legacyMode;
    if(oldRows) gpu.deepTiles = false;
    gpu.residual = !singleLevel;
    gpu.pairRows = !noPairs;
    if(relayRowsLimit > 0) gpu.relayLimit = relayRowsLimit;
    if(!gpu.load(path, threads)){
        printf("gpu load failed: %s\n", gpu.problem.c_str());
        return 1;
    }
    if(attendBudget > 0) gpu.budget = attendBudget;
    double freeAfter = freeVram();
    double lowest = freeAfter;
    printf("gpu loaded in %.1f s: dense %.2f GB, caches %.2f GB, %d of %d experts per layer resident (%.2f GB), %.2f GB VRAM free before, %.2f GB after\n", secondsSince(started), (double)gpu.denseBytes / gigabyte, (double)gpu.cacheBytes / gigabyte, gpu.residentExperts, expertCount, (double)gpu.residentBytes / gigabyte, freeBefore, freeAfter);
    printf("expert placement: %s, live swaps every %d tokens\n", gpu.heat.empty() ? "index order (no heat profile)" : ("heat profile of " + std::to_string(gpu.heatTokens) + " tokens").c_str(), gpu.adaptEvery);
    printf("%s, working set capped at %.2f GB\n", hostMemory().c_str(), (double)gpu.hostCap / gigabyte);
    int vramExperts = gpu.residentExperts * (int)gpu.layers.size();
    int pinnedExperts = gpu.warmPerLayer * (int)gpu.layers.size();
    printf("experts: %d in VRAM, %d pinned in RAM for the gpu (%.2f GB), %d warming in the page cache (%.2f GB), %d left on the SSD\n", vramExperts, pinnedExperts, (double)gpu.warmPinnedBytes / gigabyte, gpu.warmExperts, (double)gpu.warmBytes / gigabyte, (int)gpu.layers.size() * expertCount - vramExperts - pinnedExperts - gpu.warmExperts);

    if(deepCheck){
        std::vector<std::pair<std::string, GpuMatrix>> picks;
        for(int l : {0, 3}){
            GpuLayer& L = gpu.layers[l];
            std::vector<std::pair<const char*, GpuMatrix*>> dense = {{"qkv", &L.qkv}, {"gate", &L.gate}, {"ssm out", &L.ssmOut}, {"query", &L.query}, {"attn out", &L.attnOut}, {"shared gate", &L.sharedGate}, {"shared down", &L.sharedDown}};
            for(auto& d : dense){
                if(d.second->data && d.second->type != 30 && d.second->type != 1) picks.push_back({std::to_string(l) + " " + d.first, *d.second});
            }
            const Matrix* parts[3] = {&L.expertGate, &L.expertUp, &L.expertDown};
            const char* names[3] = {"expert gate", "expert up", "expert down"};
            for(int i = 0; i < 3; i++){
                if(!L.resident[i]) continue;
                GpuMatrix m;
                m.data = L.resident[i];
                m.type = parts[i]->type;
                m.rows = parts[i]->rows;
                m.cols = parts[i]->cols;
                m.stride = parts[i]->stride;
                picks.push_back({std::to_string(l) + " " + names[i], m});
            }
        }
        std::mt19937 random(7);
        std::normal_distribution<float> spread(0.0f, 1.0f);
        for(int n : {64, 200}){
            for(auto& pick : picks){
                GpuMatrix& m = pick.second;
                std::vector<float> input((size_t)n * m.cols), old((size_t)n * m.rows), deep((size_t)n * m.rows);
                for(float& v : input) v = spread(random);
                uint64_t in = 0, out = 0;
                cudaLink().alloc(&in, input.size() * 4);
                cudaLink().alloc(&out, old.size() * 4);
                cudaLink().copyUp(in, input.data(), input.size() * 4);
                for(int level = 0; level < 2; level++){
                    gpu.residual = level == 0;
                    gpu.pack(in, n * m.cols, 0);
                    for(int pass = 0; pass < 2; pass++){
                        gpu.deepTiles = pass == 1;
                        gpu.project(m, in, 0, out, n);
                        cudaLink().sync();
                        cudaLink().copyDown(pass == 0 ? old.data() : deep.data(), out, old.size() * 4);
                    }
                    double worst = 0, peak = 0;
                    for(size_t i = 0; i < old.size(); i++){
                        worst = std::max(worst, (double)fabs(old[i] - deep[i]));
                        peak = std::max(peak, (double)fabs(old[i]));
                    }
                    printf("%-16s n %3d type %2u %5d x %5d %s: rel diff %.2e%s\n", pick.first.c_str(), n, m.type, m.rows, m.cols, level == 0 ? "two level" : "one level", worst / std::max(1e-12, peak), gpu.failed ? ", launch failed" : "");
                    gpu.failed = false;
                }
                cudaLink().release(in);
                cudaLink().release(out);
            }
        }
        gpu.residual = true;
        gpu.deepTiles = true;
        return 0;
    }

    if(checkPlain){
        GpuLayer& L0 = gpu.layers[0];
        GpuLayer& L3 = gpu.layers[3];
        std::vector<std::pair<const char*, GpuMatrix*>> picks = {{"hc down", &L0.hcDown[0]}, {"hc up", &L0.hcUp[0]}, {"hc inject", &L0.hcInject[0]}, {"router", &L0.router}, {"alpha", &L0.alpha}, {"index query", &L3.indexQuery}, {"output down", &gpu.outDown}};
        int n = 256;
        std::mt19937 random(7);
        std::normal_distribution<float> spread(0.0f, 1.0f);
        for(auto& pick : picks){
            GpuMatrix& m = *pick.second;
            std::vector<float> input((size_t)n * m.cols), slow((size_t)n * m.rows), fast((size_t)n * m.rows);
            for(float& v : input) v = spread(random);
            uint64_t in = 0, out = 0;
            cudaLink().alloc(&in, input.size() * 4);
            cudaLink().alloc(&out, slow.size() * 4);
            cudaLink().copyUp(in, input.data(), input.size() * 4);
            double times[2] = {};
            for(int pass = 0; pass < 2; pass++){
                gpu.tensorPlain = pass == 1;
                for(int mode = 0; mode < 2; mode++){
                    gpu.plain(m, in, m.cols, out, m.rows, n, mode);
                    cudaLink().sync();
                    if(mode == (pick.second == &L0.hcDown[0] ? 1 : 0)) cudaLink().copyDown(pass == 0 ? slow.data() : fast.data(), out, slow.size() * 4);
                }
                auto clock = std::chrono::steady_clock::now();
                for(int i = 0; i < 20; i++) gpu.plain(m, in, m.cols, out, m.rows, n, 0);
                cudaLink().sync();
                times[pass] = secondsSince(clock) * 1000.0 / 20;
            }
            double worst = 0, peak = 0;
            for(size_t i = 0; i < slow.size(); i++){
                worst = std::max(worst, (double)fabs(slow[i] - fast[i]));
                peak = std::max(peak, (double)fabs(slow[i]));
            }
            printf("%-12s %5d x %5d: rel diff %.2e, plain rows %.3f ms, tensor %.3f ms\n", pick.first, m.rows, m.cols, worst / std::max(1e-12, peak), times[0], times[1]);
            cudaLink().release(in);
            cudaLink().release(out);
        }
        return 0;
    }


    if(!calibrate.empty()){
        std::string text;
        size_t at = 0;
        while(at <= calibrate.size()){
            size_t comma = calibrate.find(',', at);
            if(comma == std::string::npos) comma = calibrate.size();
            text += readText(calibrate.substr(at, comma - at)) + "\n";
            at = comma + 1;
        }
        std::vector<int> ids = words.encode(text);
        if(ids.size() > 4096) ids.resize(4096);
        std::fill(gpu.expertHits.begin(), gpu.expertHits.end(), 0);
        gpu.routedCount = 0;
        std::vector<int> positions(ids.size());
        std::iota(positions.begin(), positions.end(), 0);
        gpu.reset();
        auto clock = std::chrono::steady_clock::now();
        gpu.forward(ids, positions);
        printf("calibration prefill of %zu tokens in %.1f s\n", ids.size(), secondsSince(clock));
        if(!useOut.empty() && gpu.writeHeat(useOut, gpu.expertHits, ids.size())) printf("expert heat saved to %s\n", useOut.c_str());
        return 0;
    }

    if(!profileName.empty()){
        const Prompt* chosen = nullptr;
        for(const Prompt& p : prompts){
            if(p.name == profileName) chosen = &p;
        }
        if(!chosen){
            printf("unknown prompt %s\n", profileName.c_str());
            return 1;
        }
        auto report = [&](const char* title, double tokens){
            std::vector<std::pair<double, std::string>> rows;
            std::map<std::string, double> phases;
            double total = 0;
            for(auto& k : gpu.kernelMs){
                rows.push_back({k.second, k.first});
                phases[k.first.substr(0, k.first.find(' '))] += k.second;
                total += k.second;
            }
            std::sort(rows.rbegin(), rows.rend());
            printf("%s: %.3f ms of kernels per token, %.2f s in all\n", title, total / tokens, total / 1000.0);
            for(auto& p : phases) printf("  phase %-10s %9.2f s  %5.1f%%\n", p.first.c_str(), p.second / 1000.0, 100.0 * p.second / std::max(1e-9, total));
            for(size_t i = 0; i < rows.size() && i < 24; i++) printf("  %9.4f ms  %s\n", rows[i].first / tokens, rows[i].second.c_str());
            gpu.kernelMs.clear();
        };
        gpu.reset();
        std::vector<int> positions(chosen->ids.size());
        std::iota(positions.begin(), positions.end(), 0);
        double readBefore = gpu.readMs, copyBefore = gpu.copyMs, streamBefore = gpu.streamMs;
        auto wall = std::chrono::steady_clock::now();
        std::vector<float> logits = gpu.forward(chosen->ids, positions);
        double plain = secondsSince(wall);
        printf("prefill of %zu tokens without profiling: %.2f s (%.1f tok/s), host waiting on the gpu %.2f s, filling the expert ring %.2f s (pack reads %.2f s, page cache copies %.2f s), %llu experts streamed\n", chosen->ids.size(), plain, chosen->ids.size() / plain, gpu.waitMs / 1000.0, (gpu.streamMs - streamBefore) / 1000.0, (gpu.readMs - readBefore) / 1000.0, (gpu.copyMs - copyBefore) / 1000.0, (unsigned long long)gpu.streamedCount);
        gpu.reset();
        gpu.profile = true;
        logits = gpu.forward(chosen->ids, positions);
        report("prefill", (double)chosen->ids.size());
        int position = (int)chosen->ids.size();
        for(int step = 0; step < count; step++) logits = gpu.forward({best(logits)}, {position++});
        report("decode", (double)count);
        return 0;
    }

    if(!perplexityName.empty()){
        double grand = 0;
        int texts = 0, grandCount = 0;
        for(const Prompt& p : prompts){
            if(("," + perplexityName + ",").find("," + p.name + ",") == std::string::npos) continue;
            gpu.reset();
            double total = 0;
            int counted = 0, hits = 0;
            int step = gpu.relayLimit;
            std::vector<float> losses;
            auto clock = std::chrono::steady_clock::now();
            for(size_t at = 0; at + 1 < p.ids.size(); at += step){
                int n = (int)std::min<size_t>(step, p.ids.size() - 1 - at);
                std::vector<int> batch(p.ids.begin() + at, p.ids.begin() + at + n), places(n);
                std::iota(places.begin(), places.end(), (int)at);
                std::vector<float> rows = gpu.forwardAll(batch, places);
                if(rows.size() % n != 0 || rows.empty()){
                    printf("forward failed: %s\n", gpu.problem.c_str());
                    return 1;
                }
                size_t width = rows.size() / n;
                for(int k = 0; k < n; k++){
                    const float* row = rows.data() + (size_t)k * width;
                    int target = p.ids[at + k + 1];
                    float top = *std::max_element(row, row + width);
                    double sum = 0;
                    for(size_t j = 0; j < width; j++) sum += exp((double)row[j] - top);
                    double loss = log(sum) + top - row[target];
                    total += loss;
                    losses.push_back((float)loss);
                    hits += (int)(std::max_element(row, row + width) - row) == target;
                    counted++;
                }
            }
            if(!lossPath.empty()) std::ofstream(lossPath + "." + p.name, std::ios::binary).write((const char*)losses.data(), losses.size() * 4);
            printf("perplexity %s: %d tokens, mean loss %.5f nats, perplexity %.4f, top-1 %.2f%%, %.1f s\n", p.name.c_str(), counted, total / counted, exp(total / counted), 100.0 * hits / counted, secondsSince(clock));
            grand += total;
            grandCount += counted;
            texts++;
        }
        if(texts > 1) printf("all %d texts: %d tokens, mean loss %.5f nats, perplexity %.4f\n", texts, grandCount, grand / grandCount, exp(grand / grandCount));
        return 0;
    }

    if(!dumpPath.empty()){
        std::ofstream out(dumpPath, std::ios::binary);
        gpu.reset();
        int position = 0;
        std::vector<float> logits;
        for(const Prompt& p : prompts){
            if(dumpBatch){
                std::vector<int> places(p.ids.size());
                std::iota(places.begin(), places.end(), position);
                position += (int)p.ids.size();
                logits = gpu.forward(p.ids, places);
                out.write((const char*)logits.data(), logits.size() * 4);
            }
            for(int id : p.ids){
                if(dumpBatch) break;
                logits = gpu.forward({id}, {position++});
                out.write((const char*)logits.data(), logits.size() * 4);
            }
            for(int step = 0; step < count && !logits.empty(); step++){
                logits = gpu.forward({best(logits)}, {position++});
                out.write((const char*)logits.data(), logits.size() * 4);
            }
        }
        printf("dumped %d rows, last argmax %d\n", position, logits.empty() ? -1 : best(logits));
        return 0;
    }

    if(plainBench){
        auto measure = [&](const char* title, int pick, int n){
            uint64_t bytes = 0;
            int launches = 0;
            auto body = [&]{
                bytes = 0;
                launches = 0;
                for(GpuLayer& L : gpu.layers){
                    for(int h = 0; h < 2; h++){
                        if(pick == 0 || pick == 1){
                            gpu.plain(L.hcDown[h], gpu.xn, hcWidth, gpu.lo, hcRank, n, 1);
                            bytes += L.hcDown[h].stride * L.hcDown[h].rows;
                            launches++;
                        }
                        if(pick == 0 || pick == 2){
                            gpu.plain(L.hcUp[h], gpu.lo, hcRank, gpu.mixGate, hcWidth, n, 0);
                            bytes += L.hcUp[h].stride * L.hcUp[h].rows;
                            launches++;
                        }
                    }
                    if(pick == 0 || pick == 3){
                        gpu.plain(L.router, gpu.x, hidden, gpu.routerOut, expertCount, n, 0);
                        bytes += L.router.stride * L.router.rows;
                        launches++;
                    }
                }
            };
            void* exec = nullptr;
            gpu.replay(exec, body);
            cudaLink().sync();
            auto clock = std::chrono::steady_clock::now();
            for(int i = 0; i < 20; i++) gpu.replay(exec, body);
            cudaLink().sync();
            double ms = secondsSince(clock) * 1000.0 / 20;
            printf("%s: %.1f MB in %.3f ms per token (%.0f GB/s), %d launches, %.1f us each\n", title, bytes / 1048576.0, ms, bytes / ms / 1e6, launches, ms * 1000.0 / launches);
        };
        typedef int (__stdcall *FuncAttribute)(int*, int, void*);
        auto attribute = (FuncAttribute)GetProcAddress(GetModuleHandleA("nvcuda.dll"), "cuFuncGetAttribute");
        for(const char* name : {"rowsType14x1", "rowsType14x2", "rowsType14x4", "rowsType13x1", "rowsType12x1", "rowsType23x1", "rowsType21x1", "rowsMulti", "rowsStaged", "rowsRouted", "plainRows"}){
            void* fn = kernelFunction(gpu.module, name);
            int regs = -1, local = -1;
            if(fn && attribute){
                attribute(&regs, 4, fn);
                attribute(&local, 3, fn);
            }
            printf("%s: %d registers, %d bytes of local memory\n", name, regs, local);
        }
        measure("warm up", 0, 1);
        measure("hc down", 1, 1);
        measure("hc up", 2, 1);
        measure("router", 3, 1);
        measure("all, 2 rows", 0, 2);
        measure("all, 3 rows", 0, 3);
        measure("all, 4 rows", 0, 4);
        measure("all, 1 row", 0, 1);
        auto rows = [&](const char* title, int n, bool withHead, bool withLayers){
            uint64_t bytes = 0;
            auto add = [&](const GpuMatrix& m){ bytes += m.stride * m.rows; };
            auto body = [&]{
                bytes = 0;
                if(withLayers){
                    for(GpuLayer& L : gpu.layers){
                        if(L.attention){
                            gpu.projectMany({{&L.query, gpu.q}, {&L.key, gpu.k}, {&L.value, gpu.v}, {&L.indexQuery, gpu.iq}, {&L.indexKey, gpu.ik}}, gpu.x, 0, n);
                            gpu.project(L.attnOut, gpu.o, 0, gpu.y, n);
                            for(const GpuMatrix* m : {&L.query, &L.key, &L.value, &L.indexQuery, &L.indexKey, &L.attnOut}) add(*m);
                        } else {
                            gpu.projectMany({{&L.qkv, gpu.qkv}, {&L.gate, gpu.z}, {&L.alpha, gpu.alphaOut}, {&L.beta, gpu.betaOut}}, gpu.x, 0, n);
                            gpu.project(L.ssmOut, gpu.o, 0, gpu.y, n);
                            for(const GpuMatrix* m : {&L.qkv, &L.gate, &L.alpha, &L.beta, &L.ssmOut}) add(*m);
                        }
                        gpu.projectMany({{&L.sharedGate, gpu.sharedGateOut}, {&L.sharedUp, gpu.sharedUpOut}}, gpu.x, 0, n);
                        gpu.project(L.sharedDown, gpu.sharedHidden, 1, gpu.shared, n);
                        for(const GpuMatrix* m : {&L.sharedGate, &L.sharedUp, &L.sharedDown}) add(*m);
                    }
                }
                if(withHead){
                    gpu.project(gpu.head, gpu.fin, 0, gpu.vocabOut, n);
                    add(gpu.head);
                }
            };
            void* exec = nullptr;
            gpu.replay(exec, body);
            cudaLink().sync();
            auto clock = std::chrono::steady_clock::now();
            for(int i = 0; i < 20; i++) gpu.replay(exec, body);
            cudaLink().sync();
            double ms = secondsSince(clock) * 1000.0 / 20;
            printf("%s, %d row%s: %.1f MB in %.3f ms (%.0f GB/s)\n", title, n, n == 1 ? "" : "s", bytes / 1048576.0, ms, bytes / ms / 1e6);
        };
        for(int setting = 0; setting < 5; setting++){
            gpu.typedRows = setting > 0;
            gpu.groupedRows = setting < 2 || setting == 3;
            gpu.directRows = setting >= 3;
            printf("%s\n", setting == 0 ? "generic kernels" : setting == 1 ? "typed kernels, grouped single row" : setting == 2 ? "typed kernels, no grouping" : setting == 3 ? "typed kernels reading the weights directly" : "direct typed kernels, no grouping");
            for(int n : {1, 2, 3, 4}) rows("dense projections", n, false, true);
            for(int n : {1, 2, 3, 4}) rows("output head", n, true, false);
        }
        return 0;
    }

    if(!commitName.empty()){
        const Prompt* chosen = nullptr;
        for(const Prompt& p : prompts){
            if(p.name == commitName) chosen = &p;
        }
        if(!chosen){
            printf("unknown prompt %s\n", commitName.c_str());
            return 1;
        }
        std::vector<int> positions(chosen->ids.size());
        std::iota(positions.begin(), positions.end(), 0);
        gpu.reset();
        std::vector<float> logits = gpu.forward(chosen->ids, positions);
        std::vector<float> promptRow = logits;
        if(logits.empty() || !gpu.saveState(0)){
            printf("prompt forward failed: %s\n", gpu.problem.c_str());
            return 1;
        }
        std::vector<std::vector<float>> plainRows;
        std::vector<int> plainTokens;
        int position = (int)chosen->ids.size();
        for(int step = 0; step <= count && !logits.empty(); step++){
            plainRows.push_back(logits);
            plainTokens.push_back(best(logits));
            if(step < count) logits = gpu.forward({plainTokens.back()}, {position++});
        }
        if((int)plainRows.size() != count + 1){
            printf("plain decode failed: %s\n", gpu.problem.c_str());
            return 1;
        }
        size_t width = plainRows[0].size();
        auto compare = [&](const float* row, int index, int& rows, int& differing, double& worst){
            const std::vector<float>& want = plainRows[index];
            double gap = 0;
            for(size_t j = 0; j < width; j++) gap = std::max(gap, (double)fabsf(row[j] - want[j]));
            rows++;
            differing += memcmp(row, want.data(), width * 4) == 0 ? 0 : 1;
            worst = std::max(worst, gap);
        };

        if(!gpu.restoreState(0)){
            printf("restore failed: %s\n", gpu.problem.c_str());
            return 1;
        }
        logits = promptRow;
        position = (int)chosen->ids.size();
        int made = 0, rounds = 0, kept = 0, commits = 0;
        int batchRows = 0, batchDiffering = 0, afterRows = 0, afterDiffering = 0;
        double batchWorst = 0, afterWorst = 0;
        int startDiffering = 0, startRows = 0;
        double startWorst = 0;
        compare(logits.data(), 0, startRows, startDiffering, startWorst);
        int token = best(logits);
        int diverged = -1;
        double spent = 0;
        while(made + 2 < (int)plainTokens.size()){
            if(token != plainTokens[made]){
                diverged = made;
                break;
            }
            int room = std::min(draftCount, (int)plainTokens.size() - made - 2);
            int keep = rounds % (room + 1);
            std::vector<int> batch = {token};
            for(int i = 0; i < room; i++){
                int right = plainTokens[made + 1 + i];
                batch.push_back(i == keep ? (right + 1) % (int)width : right);
            }
            std::vector<int> batchPositions(batch.size());
            std::iota(batchPositions.begin(), batchPositions.end(), position);
            if(!gpu.saveState(1)){
                printf("snapshot failed: %s\n", gpu.problem.c_str());
                return 1;
            }
            std::vector<float> all = gpu.forwardAll(batch, batchPositions);
            if(all.size() != batch.size() * width){
                printf("forwardAll failed: %s\n", gpu.problem.c_str());
                return 1;
            }
            for(int k = 0; k <= keep; k++) compare(all.data() + (size_t)k * width, made + 1 + k, batchRows, batchDiffering, batchWorst);
            int fix = best(std::vector<float>(all.begin() + (size_t)keep * width, all.begin() + (size_t)(keep + 1) * width));
            kept += keep;
            rounds++;
            if(keep == room){
                position += (int)batch.size();
                made += (int)batch.size();
                token = fix;
                continue;
            }
            std::vector<int> accepted(batch.begin(), batch.begin() + 1 + keep);
            std::vector<int> acceptedPositions(accepted.size());
            std::iota(acceptedPositions.begin(), acceptedPositions.end(), position);
            auto clock = std::chrono::steady_clock::now();
            bool ok = refeedMode
                ? gpu.restoreState(1) && !gpu.forward(accepted, acceptedPositions).empty()
                : gpu.commitRows(accepted);
            spent += secondsSince(clock) * 1000.0;
            commits++;
            if(!ok){
                printf("%s failed: %s\n", refeedMode ? "refeed" : "commit", gpu.problem.c_str());
                return 1;
            }
            position += (int)accepted.size();
            made += (int)accepted.size();
            if(fix != plainTokens[made]){
                diverged = made;
                break;
            }
            logits = gpu.forward({fix}, {position++});
            if(logits.empty()){
                printf("forward after %s failed: %s\n", refeedMode ? "refeed" : "commit", gpu.problem.c_str());
                return 1;
            }
            made++;
            compare(logits.data(), made, afterRows, afterDiffering, afterWorst);
            token = best(logits);
        }
        printf("%s %s: %d rounds, %d drafts kept, prompt row %s (max gap %.3g)\n", refeedMode ? "refeed" : "commit", chosen->name.c_str(), rounds, kept, startDiffering ? "differs" : "identical", startWorst);
        printf("  verify rows: %d of %d bitwise identical to plain decode, max gap %.3g\n", batchRows - batchDiffering, batchRows, batchWorst);
        printf("  rows decoded right after a %s: %d of %d bitwise identical, max gap %.3g, %.2f ms per %s\n", refeedMode ? "refeed" : "commit", afterRows - afterDiffering, afterRows, afterWorst, spent / std::max(1, commits), refeedMode ? "refeed" : "commit");
        if(diverged >= 0) printf("  greedy text diverged at token %d of %d\n", diverged, count);
        else printf("  greedy text identical for %d tokens\n", made);
        return 0;
    }

    if(!verifyName.empty()){
        int index = -1;
        for(size_t i = 0; i < prompts.size(); i++){
            if(prompts[i].name == verifyName) index = (int)i;
        }
        if(index < 0 || !haveReference[index]){
            printf("verify needs a prompt with a cpu reference\n");
            return 1;
        }
        const Prompt& p = prompts[index];
        const Run& ref = reference[index];
        int drafts = std::min(draftCount, (int)ref.produced.size() - 1);
        gpu.reset();
        std::vector<int> positions(p.ids.size());
        std::iota(positions.begin(), positions.end(), 0);
        gpu.forward(p.ids, positions);
        if(!gpu.saveState()){
            printf("snapshot failed: %s\n", gpu.problem.c_str());
            return 1;
        }
        std::vector<int> batch(ref.produced.begin(), ref.produced.begin() + drafts), batchPositions(drafts);
        std::iota(batchPositions.begin(), batchPositions.end(), (int)p.ids.size());
        auto clock = std::chrono::steady_clock::now();
        std::vector<float> all = gpu.forwardAll(batch, batchPositions);
        double batchMs = secondsSince(clock) * 1000.0;
        if(all.size() != (size_t)drafts * ref.logits.size()){
            printf("forwardAll failed: %s\n", gpu.problem.c_str());
            return 1;
        }
        int accepted = 0;
        for(int k = 0; k < drafts; k++){
            std::vector<float> row(all.begin() + (size_t)k * ref.logits.size(), all.begin() + (size_t)(k + 1) * ref.logits.size());
            if(best(row) != ref.produced[k + 1]) break;
            accepted++;
        }
        gpu.restoreState();
        double worst = 0;
        for(int k = 0; k < drafts; k++){
            std::vector<float> single = gpu.forward({batch[k]}, {batchPositions[k]});
            double diff2 = 0, ref2 = 0;
            for(size_t j = 0; j < single.size(); j++){
                double d = (double)single[j] - all[(size_t)k * single.size() + j];
                diff2 += d * d;
                ref2 += (double)single[j] * single[j];
            }
            worst = std::max(worst, sqrt(diff2 / std::max(1e-12, ref2)));
        }
        printf("verify %s: %d drafted tokens checked in one forward (%.1f ms), %d of %d agree with the cpu greedy text, batch vs single rms rel diff at most %.2e after restoring the state\n", p.name.c_str(), drafts, batchMs, accepted, drafts, worst);
        return 0;
    }

    if(!lockstep.empty()){
        const Prompt* chosen = nullptr;
        for(const Prompt& p : prompts){
            if(p.name == lockstep) chosen = &p;
        }
        if(!chosen){
            printf("unknown prompt %s\n", lockstep.c_str());
            return 1;
        }
        Model cpu;
        if(!cpu.load(path, threads)){
            printf("cpu load failed: %s\n", cpu.problem.c_str());
            return 1;
        }
        cpu.reset();
        gpu.reset();
        std::vector<int> feed = chosen->ids;
        for(int step = 0; step < (int)chosen->ids.size() + count; step++){
            std::vector<float> a = cpu.forward({feed[step]}, {step});
            std::vector<float> b = gpu.forward({feed[step]}, {step});
            if(a.empty() || b.empty()){
                printf("forward failed at step %d: %s\n", step, gpu.problem.c_str());
                return 1;
            }
            double worst = 0, peak = 0, diff2 = 0, ref2 = 0;
            for(size_t j = 0; j < a.size(); j++){
                double d = (double)b[j] - a[j];
                worst = std::max(worst, fabs(d));
                peak = std::max(peak, fabs((double)a[j]));
                diff2 += d * d;
                ref2 += (double)a[j] * a[j];
            }
            int flips = 0, firstFlip = -1;
            for(size_t l = 0; l < gpu.layers.size(); l++){
                std::vector<int> x = cpu.layers[l].lastRouting, y = gpu.layers[l].lastRouting;
                std::sort(x.begin(), x.end());
                std::sort(y.begin(), y.end());
                if(x != y){
                    flips++;
                    if(firstFlip < 0) firstFlip = (int)l;
                }
            }
            printf("step %4d token %6d: max rel %.2e rms rel %.2e argmax %d/%d routing differs in %d layers (first %d)\n", step, feed[step], worst / peak, sqrt(diff2 / ref2), best(a), best(b), flips, firstFlip);
            if(step + 1 >= (int)feed.size()) feed.push_back(best(a));
        }
        return 0;
    }

    for(size_t i = 0; i < prompts.size(); i++){
        uint64_t routedBefore = gpu.routedCount, residentBefore = gpu.residentCount;
        uint64_t decodeBefore = gpu.decodeRouted, vramBefore = gpu.decodeResident, pinnedBefore = gpu.warmHits, streamedBefore = gpu.streamedCount;
        double streamBefore = gpu.streamMs;
        Run run = generate(gpu, prompts[i].ids, count, stepwise);
        lowest = std::min(lowest, freeVram());
        if(run.logits.empty()){
            printf("gpu forward failed for %s: %s\n", prompts[i].name.c_str(), gpu.problem.c_str());
            return 1;
        }
        int steps = (int)run.produced.size() - 1;
        double hitRate = gpu.routedCount > routedBefore ? (double)(gpu.residentCount - residentBefore) / (double)(gpu.routedCount - routedBefore) : 0;
        printf("\n[%s] %zu tokens, %llu expert swaps so far: gpu prefill %.2f s (%.1f tok/s), decode %d tokens %.2f s (%.2f tok/s), resident hit rate %.1f%%\n", prompts[i].name.c_str(), prompts[i].ids.size(), (unsigned long long)gpu.swapCount, run.prefillSeconds, prompts[i].ids.size() / run.prefillSeconds, steps, run.decodeSeconds, steps / std::max(1e-9, run.decodeSeconds), 100.0 * hitRate);
        double routedDecode = (double)std::max<uint64_t>(1, gpu.decodeRouted - decodeBefore);
        printf("  prefill streaming: %llu experts uploaded, %.2f s spent reading them\n", (unsigned long long)(gpu.streamedCount - streamedBefore), (gpu.streamMs - streamBefore) / 1000.0);
        printf("  decode routing: %.1f%% from VRAM, %.1f%% from pinned RAM over PCIe, %.1f%% on the CPU\n", 100.0 * (gpu.decodeResident - vramBefore) / routedDecode, 100.0 * (gpu.warmHits - pinnedBefore) / routedDecode, 100.0 * (1.0 - (double)(gpu.decodeResident - vramBefore + gpu.warmHits - pinnedBefore) / routedDecode));
        printf("  last decode step: wait %.2f ms, cpu experts %.2f ms, ple %.2f ms, head %.2f ms, layers %.2f ms\n", gpu.waitMs, gpu.expertMs, gpu.pleMs, gpu.headMs, std::accumulate(gpu.layerMs.begin(), gpu.layerMs.end(), 0.0));
        printf("  gpu: %s\n", shown(words.decode(run.produced)).c_str());
        if(!haveReference[i]) continue;

        const Run& ref = reference[i];
        printf("  cpu: %s\n", shown(words.decode(ref.produced)).c_str());
        double worst = 0, peak = 0, diff2 = 0, ref2 = 0;
        for(size_t j = 0; j < ref.logits.size(); j++){
            double d = (double)run.logits[j] - ref.logits[j];
            worst = std::max(worst, fabs(d));
            peak = std::max(peak, fabs((double)ref.logits[j]));
            diff2 += d * d;
            ref2 += (double)ref.logits[j] * ref.logits[j];
        }
        auto topFive = [](const std::vector<float>& l){
            std::vector<int> order(l.size());
            std::iota(order.begin(), order.end(), 0);
            std::partial_sort(order.begin(), order.begin() + 5, order.end(), [&](int a, int b){ return l[a] > l[b]; });
            order.resize(5);
            return order;
        };
        std::vector<int> cpuTop = topFive(ref.logits), gpuTop = topFive(run.logits);
        int overlap = 0;
        for(int a : gpuTop) overlap += std::find(cpuTop.begin(), cpuTop.end(), a) != cpuTop.end();
        size_t same = 0;
        while(same < ref.produced.size() && same < run.produced.size() && ref.produced[same] == run.produced[same]) same++;
        printf("  prefill logits: max abs diff %.4f, max rel diff %.2e, rms rel diff %.2e, top-5 overlap %d/5, argmax cpu %d gpu %d\n", worst, worst / std::max(1e-12, peak), sqrt(diff2 / std::max(1e-12, ref2)), overlap, cpuTop[0], gpuTop[0]);
        printf("  top-5 cpu:");
        for(int a : cpuTop) printf(" %d(%.3f)", a, ref.logits[a]);
        printf("  gpu:");
        for(int a : gpuTop) printf(" %d(%.3f)", a, run.logits[a]);
        printf("\n  greedy continuation: %zu of %zu tokens identical%s\n", same, ref.produced.size(), same == ref.produced.size() ? "" : " (diverged)");
    }

    printf("\npeak VRAM in use on the device %.2f GB, this process about %.2f GB\n", totalVram() - lowest, freeBefore - lowest);
    printf("%s\n", hostMemory().c_str());
    if(!useOut.empty() && gpu.writeHeat(useOut, gpu.expertHits, gpu.routedCount / expertsUsed / gpu.layers.size())) printf("expert heat saved to %s\n", useOut.c_str());
    return 0;
}
