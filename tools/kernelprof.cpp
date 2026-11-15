#include "../src/brain.h"
#include "../src/tokenizer.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <numeric>
#include <random>

double millisSince(std::chrono::steady_clock::time_point from) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - from).count();
}

int bestOf(const std::vector<float>& logits) {
    return (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::string promptFile = argc > 1 ? argv[1] : "build/gpugen/benchlong.txt";
    std::string modelPath = argc > 2 ? argv[2] : "modelos/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf";
    int tail = argc > 3 ? atoi(argv[3]) : 7;
    int window = argc > 4 ? atoi(argv[4]) : 1;
    const double gigabyte = (double)(1ULL << 30);

    ModelFile file;
    std::string error;
    Tokenizer tokenizer;
    if(!openModel(modelPath, file, error) || !tokenizer.load(file, error)){
        std::cout << "cannot read the model: " << error << std::endl;
        return 1;
    }
    std::ifstream in(promptFile, std::ios::binary);
    std::stringstream text;
    text << in.rdbuf();
    std::vector<int> prompt = tokenizer.encode(text.str(), true);
    // cuantos tokens del texto caen debajo de cada limite: dice cuanto se pierde si los borradores miran solo los primeros ids
    if(getenv("KERNELPROF_TOKEN_SPREAD")){
        int below[3] = {};
        const int marks[3] = {32768, 65536, 131072};
        for(int id : prompt){
            for(int i = 0; i < 3; i++) below[i] += id < marks[i];
        }
        printf("prompt tokens %zu, under 32K %.2f%%, under 64K %.2f%%, under 128K %.2f%%\n", prompt.size(), 100.0 * below[0] / std::max<size_t>(1, prompt.size()), 100.0 * below[1] / std::max<size_t>(1, prompt.size()), 100.0 * below[2] / std::max<size_t>(1, prompt.size()));
        return 0;
    }

    GpuModel gpu;
    gpu.hostBudget = (uint64_t)(24.0 * gigabyte);
    gpu.vramCap = (uint64_t)(23.5 * gigabyte);
    gpu.contextLimit = getenv("KERNELPROF_CONTEXT") ? atoi(getenv("KERNELPROF_CONTEXT")) : 65536;
    gpu.pinWarm = true;
    gpu.tailSkip = tail;
    gpu.exchanging = argc > 7 && std::string(argv[7]) == "exchange";
    if(!gpu.exchanging) gpu.adaptEvery = 0;
    if(argc > 8) gpu.warmShare = atoi(argv[8]);
    if(const char* helping = getenv("KERNELPROF_HELPER")) gpu.useHelper = helping[0] == '1';
    gpu.timeHelper = getenv("KERNELPROF_HELPER_TIMES") != nullptr;
    if(const char* recent = getenv("KERNELPROF_RECENT")) gpu.recentSlots = atoi(recent);
    if(const char* rows = getenv("KERNELPROF_GRAPH_ROWS")) gpu.graphRows = atoi(rows);
    if(const char* relay = getenv("KERNELPROF_RELAY")) gpu.relayLimit = atoi(relay);
    if(const char* quick = getenv("KERNELPROF_PROMPT_RELAY")) gpu.promptRelay = atoi(quick);
    if(const char* team = getenv("KERNELPROF_TEAM_ROWS")) gpu.teamRows = atoi(team);
    if(const char* reading = getenv("KERNELPROF_SHORT_READ")) gpu.shortRead = atoi(reading);
    if(const char* cold = getenv("KERNELPROF_COLD")) gpu.coldRoutes = atoi(cold);
    if(const char* gemmaWide = getenv("KERNELPROF_GEMMA_WIDE")) gpu.gemmaWide = gemmaWide[0] != '0';
    if(const char* gemmaGroups = getenv("KERNELPROF_GEMMA_GROUPS")) gpu.gemmaGroups = gemmaGroups[0] != '0';
    if(const char* gemmaBlock = getenv("KERNELPROF_GEMMA_BLOCK")) gpu.gemmaBlock = atoi(gemmaBlock);
    if(const char* gemmaFew = getenv("KERNELPROF_GEMMA_FEW")) gpu.gemmaFew = gemmaFew[0] != '0';
    if(const char* pad = getenv("KERNELPROF_PAD_THREE")) gpu.padThree = pad[0] != '0';
    if(const char* flash = getenv("KERNELPROF_GEMMA_FLASH")) gpu.gemmaFlash = flash[0] != '0';
    if(const char* packed = getenv("KERNELPROF_PACKED")) gpu.packedAttention = packed[0] != '0';
    if(const char* direct = getenv("KERNELPROF_DIRECT_ROWS")) gpu.directRows = direct[0] != '0';
    if(const char* few = getenv("KERNELPROF_FEWMMA")) gpu.fewMma = few[0] != '0';
    if(const char* from = getenv("KERNELPROF_FEWMMA_FROM")) gpu.fewMmaFrom = atoi(from);
    if(const char* pair = getenv("KERNELPROF_PAIR_ROWS")) gpu.pairRows = pair[0] != '0';    if(const char* coldRows = getenv("KERNELPROF_COLD_ROWS")) gpu.coldRows = atoi(coldRows);
    if(const char* teams = getenv("KERNELPROF_TEAMS")) gpu.teams = teams[0] == '1';
    if(const char* grow = getenv("KERNELPROF_GROW")) gpu.growKv = grow[0] == '1';
    if(const char* start = getenv("KERNELPROF_KV_START")) gpu.kvStart = atoi(start);
    if(const char* margin = getenv("KERNELPROF_GROW_MARGIN_MB")) gpu.growMargin = (uint64_t)atoi(margin) << 20;
    if(const char* delay = getenv("KERNELPROF_HELPER_DELAY")) gpu.helperDelay = atoi(delay);
    if(const char* late = getenv("KERNELPROF_HOST_DELAY")) gpu.hostDelay = atoi(late);
    if(const char* seeing = getenv("KERNELPROF_VISION")) gpu.vision = seeing[0] == '1';
    if(const char* card = getenv("KERNELPROF_VISION_GPU")) gpu.visionGpu = atoi(card);
    if(const char* share = getenv("KERNELPROF_PIN_SHARE")) gpu.pinShare = atoi(share);
    if(const char* size = getenv("KERNELPROF_PIN_GB")) gpu.pinGb = atof(size);
    if(const char* lifts = getenv("KERNELPROF_DISK_LIFTS")){
        gpu.diskLifts = atoi(lifts);
        gpu.diskLimit = 2 * gpu.diskLifts;
    }
    if(const char* vram = getenv("KERNELPROF_VRAM_GB")) gpu.vramCap = (uint64_t)(atof(vram) * gigabyte);
    if(const char* plan = getenv("KERNELPROF_CPU_PLAN")) gpu.cpuPlan = plan;
    if(const char* group = getenv("KERNELPROF_GROUP")) gpu.groupAttention = group[0] != '0';
    if(const char* blas = getenv("KERNELPROF_BLAS")) gpu.useBlas = blas[0] != '0';
    if(const char* scan = getenv("KERNELPROF_GROUPSCAN")) gpu.groupScan = scan[0] != '0';
    if(const char* feed = getenv("KERNELPROF_FASTFEED")) gpu.fastFeed = feed[0] != '0';
    if(const char* joined = getenv("KERNELPROF_FUSEDNORM")) gpu.fusedNorm = joined[0] != '0';
    if(const char* gates = getenv("KERNELPROF_HALFGATES")) gpu.halfGates = gates[0] != '0';
    if(const char* together = getenv("KERNELPROF_JOININJECT")) gpu.joinInject = together[0] != '0';
    if(const char* mma = getenv("KERNELPROF_MMAATTN")) gpu.mmaAttention = mma[0] != '0';
    if(const char* wide = getenv("KERNELPROF_INTATTN")) gpu.intAttention = wide[0] != '0';
    if(const char* fold = getenv("KERNELPROF_MIXUP")) gpu.mixUp = fold[0] != '0';
    if(const char* tiles = getenv("KERNELPROF_TILESCORES")) gpu.tileScores = tiles[0] != '0';
    if(const char* wide = getenv("KERNELPROF_DEEPWIDE")) gpu.deepWide = wide[0] != '0';
    if(const char* exact = getenv("KERNELPROF_EXACTMIX")) gpu.exactMix = exact[0] != '0';
    if(const char* tall = getenv("KERNELPROF_TALL")) gpu.tallTiles = tall[0] != '0';
    if(const char* least = getenv("KERNELPROF_WIDELEAST")) gpu.wideLeast = atoi(least);
    if(const char* wide = getenv("KERNELPROF_WIDE")) gpu.wideTiles = wide[0] != '0';
    if(const char* mixHalf = getenv("KERNELPROF_HALFMIX")) gpu.halfMix = mixHalf[0] != '0';
    if(const char* sides = getenv("KERNELPROF_SIDES")) gpu.stageSides = atoi(sides);
    if(const char* fineRows = getenv("KERNELPROF_BATCHFINE")) gpu.batchFine = fineRows[0] != '0';
    if(const char* rows = getenv("KERNELPROF_CHUNK")) gpu.chunk = atoi(rows);
    if(const char* rows = getenv("KERNELPROF_CREST")) gpu.crestRows = atoi(rows);
    if(const char* rows = getenv("KERNELPROF_LOAN_CREST")) gpu.loanCrest = atoi(rows);
    if(const char* loan = getenv("KERNELPROF_LOAN")) gpu.loanFrom = loan[0] == '0' ? (1 << 30) : atoi(loan);
    if(const char* ram = getenv("KERNELPROF_RAM_GB")) gpu.hostBudget = (uint64_t)(atof(ram) * gigabyte);
    if(const char* draft = getenv("KERNELPROF_DRAFT")) gpu.draftPath = draft;
    if(const char* rows = getenv("KERNELPROF_DRAFTBATCH")) gpu.draftBatch = atoi(rows);
    if(getenv("KERNELPROF_SHAPES")) gpu.profileShapes = true;
    if(!gpu.load(modelPath, 16)){
        std::cout << "load failed: " << gpu.problem << std::endl;
        return 1;
    }
    printf("resident experts per layer %d, warm %d\n", gpu.residentExperts, gpu.warmPerLayer);
    if(const char* bench = getenv("KERNELPROF_DENSEBENCH")){
        int n = std::max(1, atoi(bench));
        int repeats = getenv("KERNELPROF_BENCH_REPEATS") ? atoi(getenv("KERNELPROF_BENCH_REPEATS")) : 300;
        std::string list = getenv("KERNELPROF_BENCH_KERNELS") ? getenv("KERNELPROF_BENCH_KERNELS") : "rowsDirect:8,rowsPair:16";
        std::vector<std::pair<std::string, const GpuMatrix*>> picked;
        // KERNELPROF_BENCH_MATS elige por nombre las matrices de la primera capa que las tenga, de cualquier tipo
        if(const char* mats = getenv("KERNELPROF_BENCH_MATS")){
            std::stringstream names(mats);
            std::string name;
            while(std::getline(names, name, ',')){
                // en gemma las matrices de la primera capa: gq consulta, gk clave, go salida, gg compuerta, gd bajada
                if(gpu.gemma && !gpu.gemmaLayers.empty()){
                    GemmaLayer& G = gpu.gemmaLayers[0];
                    const GpuMatrix* m = name == "gq" ? &G.query : name == "gk" ? &G.key : name == "go" ? &G.out : name == "gg" ? &G.gate : name == "gd" ? &G.down : name == "head" ? &gpu.head : nullptr;
                    if(m && m->data) picked.push_back({name + " of gemma layer 0", m});
                    continue;
                }
                for(int l = gpu.firstLayer; l < gpu.endLayer; l++){
                    GpuLayer& L = gpu.layers[l];
                    const GpuMatrix* m = name == "qkv" ? &L.qkv : name == "ssmOut" ? &L.ssmOut : name == "gate" ? &L.gate : name == "sharedGate" ? &L.sharedGate : name == "sharedUp" ? &L.sharedUp : name == "sharedDown" ? &L.sharedDown : name == "query" ? &L.query : name == "attnOut" ? &L.attnOut : nullptr;
                    if(m && m->data){
                        picked.push_back({name + " of layer " + std::to_string(l), m});
                        break;
                    }
                }
            }
        }
        for(int l = gpu.firstLayer; l < gpu.endLayer && picked.size() < 3 && !getenv("KERNELPROF_BENCH_MATS"); l++){
            GpuLayer& L = gpu.layers[l];
            if(L.qkv.type == 14 && picked.empty()) picked.push_back({"qkv of layer " + std::to_string(l), &L.qkv});
            if(L.ssmOut.type == 14 && picked.size() == 1) picked.push_back({"ssm out of layer " + std::to_string(l), &L.ssmOut});
            if(L.gate.type == 14 && picked.size() == 2) picked.push_back({"gate of layer " + std::to_string(l), &L.gate});
        }
        bool oneLevel = getenv("KERNELPROF_BENCH_FINE") && getenv("KERNELPROF_BENCH_FINE")[0] == '0';
        std::mt19937 random(5);
        std::normal_distribution<float> spread(0.0f, 1.0f);
        auto upload = [&](const void* data, uint64_t bytes){
            uint64_t ptr = 0;
            cudaLink().alloc(&ptr, (size_t)bytes);
            if(data) cudaLink().copyUp(ptr, data, (size_t)bytes);
            return ptr;
        };
        for(auto& p : picked){
            const GpuMatrix& m = *p.second;
            int count = n * m.cols;
            std::vector<float> x((size_t)count);
            for(float& v : x) v = spread(random);
            uint64_t xOnGpu = upload(x.data(), 4ULL * count);
            PackedInput a, b;
            a.values = upload(nullptr, (uint64_t)count);
            a.scales = upload(nullptr, 4ULL * count / 32);
            a.sums = upload(nullptr, 4ULL * count / 32);
            b.values = upload(nullptr, (uint64_t)count);
            b.scales = upload(nullptr, 4ULL * count / 32);
            b.sums = upload(nullptr, 4ULL * count / 32);
            gpu.go("quantizeFine", (unsigned int)((count / 32 + 7) / 8), 1, 32, 8, xOnGpu, count, a.values, a.scales, a.sums, b.values, b.scales, b.sums);
            uint64_t out = upload(nullptr, 4ULL * n * m.rows);
            double bytes = (double)m.rows * m.stride;
            printf("%s: type %u, %d x %d, %.1f MB, %d tokens\n", p.first.c_str(), m.type, m.rows, m.cols, bytes / 1048576.0, n);
            std::vector<float> reference;
            for(int round = 0; round < 3; round++){
                std::stringstream kernels(list);
                std::string item;
                while(std::getline(kernels, item, ',')){
                    size_t colon = item.find(':');
                    bool tensor = item.rfind("rowsMma", 0) == 0;
                    std::string name = tensor
                        ? "rowsMma" + std::to_string(m.type) + "t" + std::to_string((n + 7) / 8)
                        : item.substr(0, colon) + std::to_string(m.type) + "x" + std::to_string(n);
                    int per = tensor ? 16 : colon == std::string::npos ? 16 : atoi(item.c_str() + colon + 1);
                    unsigned int blocks = (unsigned int)((m.rows + per - 1) / per);
                    int stage = (int)((m.stride + 31) / 16 * 16);
                    uint64_t none = 0;
                    auto call = [&]{
                        if(tensor) gpu.goShared(0, name.c_str(), blocks, 1, 256, 1, m.data, m.stride, a.values, a.scales, a.sums, oneLevel ? none : b.values, oneLevel ? none : b.scales, oneLevel ? none : b.sums, out, m.rows, m.cols, n, gpu.tables);
                        else gpu.goShared(0, name.c_str(), blocks, 1, 256, 1, m.data, m.stride, a.values, a.scales, a.sums, oneLevel ? none : b.values, oneLevel ? none : b.scales, oneLevel ? none : b.sums, out, m.rows, m.cols, stage, gpu.tables);
                    };
                    gpu.failed = false;
                    call();
                    if(gpu.waitStream() != 0 || gpu.failed){
                        printf("%s: launch failed %s\n", name.c_str(), gpu.launchProblem.c_str());
                        gpu.failed = false;
                        continue;
                    }
                    auto clock = std::chrono::steady_clock::now();
                    for(int i = 0; i < repeats; i++) call();
                    gpu.waitStream();
                    double ms = millisSince(clock) / repeats;
                    std::vector<float> result((size_t)n * m.rows);
                    cudaLink().copyDown(result.data(), out, 4ULL * n * m.rows);
                    if(reference.empty()) reference = result;
                    size_t same = 0;
                    double worst = 0.0;
                    for(size_t i = 0; i < result.size(); i++){
                        same += result[i] == reference[i];
                        worst = std::max(worst, (double)std::fabs(result[i] - reference[i]));
                    }
                    if(round > 0) printf("  %-16s %7.1f us (%5.0f GB/s), same as the first %zu of %zu, worst gap %.3g\n", name.c_str(), ms * 1000, bytes / ms / 1e6, same, result.size(), worst);
                }
            }
        }
        return 0;
    }
    if(const char* bench = getenv("KERNELPROF_EXPERTBENCH")){
        int n = std::max(1, atoi(bench));
        int layer = getenv("KERNELPROF_BENCH_LAYER") ? atoi(getenv("KERNELPROF_BENCH_LAYER")) : gpu.firstLayer + 10;
        int reach = getenv("KERNELPROF_BENCH_POOL") ? atoi(getenv("KERNELPROF_BENCH_POOL")) : 6 * n;
        int repeats = getenv("KERNELPROF_BENCH_REPEATS") ? atoi(getenv("KERNELPROF_BENCH_REPEATS")) : 300;
        std::string list = getenv("KERNELPROF_BENCH_KERNELS") ? getenv("KERNELPROF_BENCH_KERNELS") : "rowsStagedSet:1,rowsStagedSet2:2,rowsStagedSet4:4";
        GpuLayer& L = gpu.layers[layer];
        int per = expertsUsed * n;
        int pick = getenv("KERNELPROF_BENCH_PICK") ? std::max(1, std::min(expertsUsed, atoi(getenv("KERNELPROF_BENCH_PICK")))) : 6;
        std::mt19937 random(7);
        std::vector<int> slots(per, 0), inputs(per, 0), used(per, 0);
        int pool = std::max(pick, std::min(gpu.residentExperts, reach));
        int routes = 0;
        for(int t = 0; t < n; t++){
            std::vector<int> picked;
            while((int)picked.size() < pick){
                int e = (int)(random() % pool);
                if(std::find(picked.begin(), picked.end(), e) == picked.end()) picked.push_back(e);
            }
            for(int e : picked){
                slots[routes] = e;
                inputs[routes] = t;
                used[routes] = 1;
                routes++;
            }
        }
        std::set<int> distinct(slots.begin(), slots.begin() + routes);
        std::vector<int> crews(4ULL * per, -1), seatOf(pool, -1), filled(per, 0);
        int seats = 0;
        for(int g = 0; g < routes; g++){
            int s = seatOf[slots[g]];
            if(s < 0 || filled[s] == 4){
                s = seats++;
                seatOf[slots[g]] = s;
            }
            crews[4 * s + filled[s]++] = g;
        }
        auto upload = [&](const void* data, uint64_t bytes){
            uint64_t ptr = 0;
            cudaLink().alloc(&ptr, (size_t)bytes);
            if(data) cudaLink().copyUp(ptr, data, (size_t)bytes);
            return ptr;
        };
        std::normal_distribution<float> spread(0.0f, 1.0f);
        std::vector<float> x((size_t)n * hidden), h((size_t)per * expertWidth);
        for(float& v : x) v = spread(random);
        for(float& v : h) v = 0.1f * spread(random);
        uint64_t slotOnGpu = upload(slots.data(), 4ULL * per), inputOnGpu = upload(inputs.data(), 4ULL * per), usedOnGpu = upload(used.data(), 4ULL * per);
        uint64_t xOnGpu = upload(x.data(), 4ULL * x.size()), hOnGpu = upload(h.data(), 4ULL * h.size());
        uint64_t crewOnGpu = upload(crews.data(), 4ULL * crews.size());
        auto quantized = [&](uint64_t from, int count, PackedInput& a, PackedInput& b){
            a.values = upload(nullptr, (uint64_t)count);
            a.scales = upload(nullptr, 4ULL * count / 32);
            a.sums = upload(nullptr, 4ULL * count / 32);
            b.values = upload(nullptr, (uint64_t)count);
            b.scales = upload(nullptr, 4ULL * count / 32);
            b.sums = upload(nullptr, 4ULL * count / 32);
            gpu.go("quantizeFine", (unsigned int)((count / 32 + 7) / 8), 1, 32, 8, from, count, a.values, a.scales, a.sums, b.values, b.scales, b.sums);
        };
        PackedInput a, b, c, d;
        quantized(xOnGpu, n * hidden, a, b);
        quantized(hOnGpu, per * expertWidth, c, d);
        uint64_t gateOut = upload(nullptr, 4ULL * per * expertWidth), upOut = upload(nullptr, 4ULL * per * expertWidth), downOut = upload(nullptr, 4ULL * per * hidden);
        auto build = [&](const Matrix& m, uint64_t weights, uint64_t input, uint64_t output, StagedSet& set){
            int i = set.count++;
            set.per = per;
            set.type[i] = (int)m.type;
            set.shift[i] = 0;
            set.weights[i] = weights;
            set.rowBytes[i] = m.stride;
            set.expertBytes[i] = (uint64_t)m.rows * m.stride;
            set.slot[i] = slotOnGpu;
            set.used[i] = usedOnGpu;
            set.input[i] = input;
            set.output[i] = output;
        };
        StagedSet up = {}, down = {};
        build(L.expertGate, L.resident[0], inputOnGpu, gateOut, up);
        build(L.expertUp, L.resident[1], inputOnGpu, upOut, up);
        build(L.expertDown, L.resident[2], 0, downOut, down);
        double upBytes = (double)routes * ((double)L.expertGate.rows * L.expertGate.stride + (double)L.expertUp.rows * L.expertUp.stride);
        double downBytes = (double)routes * L.expertDown.rows * L.expertDown.stride;
        printf("layer %d: gate and up type %u, down type %u, %d tokens, %d routes over %zu experts in %d teams, %.2f MB gate and up, %.2f MB down\n", layer, L.expertGate.type, L.expertDown.type, n, routes, distinct.size(), seats, upBytes / 1048576.0, downBytes / 1048576.0);
        std::vector<float> firstUp, firstDown;
        for(int round = 0; round < 3; round++){
        std::stringstream kernels(list);
        std::string item;
        while(std::getline(kernels, item, ',')){
            size_t colon = item.find(':');
            std::string name = item.substr(0, colon);
            size_t slash = name.find('/');
            std::string downName = slash == std::string::npos ? name : name.substr(slash + 1);
            name = name.substr(0, slash);
            int many = colon == std::string::npos ? 1 : atoi(item.c_str() + colon + 1);
            size_t split = colon == std::string::npos ? std::string::npos : item.find('/', colon);
            int downMany = split == std::string::npos ? many : atoi(item.c_str() + split + 1);
            int upStage = (int)((many * std::max(L.expertGate.stride, L.expertUp.stride) + 31) / 16 * 16);
            int downStage = (int)((downMany * L.expertDown.stride + 31) / 16 * 16);
            unsigned int upGrid = (unsigned int)((expertWidth + 8 * many - 1) / (8 * many));
            unsigned int downGrid = (unsigned int)((hidden + 8 * downMany - 1) / (8 * downMany));
            up.team[0] = up.team[1] = name.rfind("rowsTeam", 0) == 0 ? crewOnGpu : 0;
            down.team[0] = downName.rfind("rowsTeam", 0) == 0 ? crewOnGpu : 0;
            auto upCall = [&]{ gpu.goShared(8 * upStage, name.c_str(), upGrid, (unsigned int)(up.count * up.per), 256, 1, up, hidden, a.values, a.scales, a.sums, b.values, b.scales, b.sums, expertWidth, hidden, upStage, gpu.tables); };
            auto downCall = [&]{ gpu.goShared(8 * downStage, downName.c_str(), downGrid, (unsigned int)(down.count * down.per), 256, 1, down, expertWidth, c.values, c.scales, c.sums, d.values, d.scales, d.sums, hidden, expertWidth, downStage, gpu.tables); };
            gpu.failed = false;
            std::vector<float> zeros((size_t)per * hidden, 0.0f);
            cudaLink().copyUp(gateOut, zeros.data(), 4ULL * per * expertWidth);
            cudaLink().copyUp(upOut, zeros.data(), 4ULL * per * expertWidth);
            cudaLink().copyUp(downOut, zeros.data(), 4ULL * per * hidden);
            upCall();
            downCall();
            if(gpu.waitStream() != 0 || gpu.failed){
                printf("%s: launch failed %s\n", name.c_str(), gpu.launchProblem.c_str());
                gpu.failed = false;
                continue;
            }
            auto clock = std::chrono::steady_clock::now();
            for(int i = 0; i < repeats; i++) upCall();
            gpu.waitStream();
            double upMs = millisSince(clock) / repeats;
            clock = std::chrono::steady_clock::now();
            for(int i = 0; i < repeats; i++) downCall();
            gpu.waitStream();
            double downMs = millisSince(clock) / repeats;
            std::vector<float> upResult((size_t)2 * per * expertWidth), downResult((size_t)per * hidden);
            cudaLink().copyDown(upResult.data(), gateOut, 4ULL * per * expertWidth);
            cudaLink().copyDown(upResult.data() + (size_t)per * expertWidth, upOut, 4ULL * per * expertWidth);
            cudaLink().copyDown(downResult.data(), downOut, 4ULL * per * hidden);
            if(firstUp.empty()){
                firstUp = upResult;
                firstDown = downResult;
            }
            size_t upSame = 0, downSame = 0, upCount = 0, downCount = 0;
            double worst = 0.0;
            for(int g = 0; g < routes; g++){
                for(int half = 0; half < 2; half++){
                    for(int r = 0; r < expertWidth; r++){
                        size_t at = (size_t)half * per * expertWidth + (size_t)g * expertWidth + r;
                        upSame += upResult[at] == firstUp[at];
                        upCount++;
                        worst = std::max(worst, (double)std::fabs(upResult[at] - firstUp[at]));
                    }
                }
                for(int r = 0; r < hidden; r++){
                    size_t at = (size_t)g * hidden + r;
                    downSame += downResult[at] == firstDown[at];
                    downCount++;
                    worst = std::max(worst, (double)std::fabs(downResult[at] - firstDown[at]));
                }
            }
            if(round == 0) continue;
            printf("%-18s rows per warp %d: gate and up %7.1f us (%5.0f GB/s), down %7.1f us (%5.0f GB/s), same as the first %zu of %zu and %zu of %zu, worst gap %.3g\n", name.c_str(), many, upMs * 1000, upBytes / upMs / 1e6, downMs * 1000, downBytes / downMs / 1e6, upSame, upCount, downSame, downCount, worst);
        }
        }
        return 0;
    }
    if(getenv("KERNELPROF_COPYBENCH")){
        int layer = gpu.firstLayer + 10;
        GpuLayer& L = gpu.layers[layer];
        const Matrix* parts[3] = {&L.expertGate, &L.expertUp, &L.expertDown};
        CudaQueue& queue = cudaQueue();
        int count = std::min(gpu.stageSlots, gpu.warmPerLayer);
        std::vector<int> warmed;
        for(int e = 0; e < expertCount && (int)warmed.size() < count; e++) if(L.warm[e] >= 0) warmed.push_back(e);
        uint64_t total = 0;
        for(int i = 0; i < 3; i++) total += (uint64_t)parts[i]->rows * parts[i]->stride * warmed.size();
        printf("copy bench: %zu experts of layer %d, %.1f MB\n", warmed.size(), layer, total / 1048576.0);
        for(int round = 0; round < 3; round++){
            auto clock = std::chrono::steady_clock::now();
            for(int r = 0; r < 20; r++){
                for(size_t j = 0; j < warmed.size(); j++){
                    for(int i = 0; i < 3; i++){
                        uint64_t span = (uint64_t)parts[i]->rows * parts[i]->stride;
                        queue.copyUp(gpu.stage[0] + (uint64_t)j * L.slice + L.piece[i], L.warmHost[i] + (uint64_t)L.warm[warmed[j]] * span, (size_t)span, gpu.copyStream);
                    }
                }
            }
            queue.streamSync(gpu.copyStream);
            double pieces = millisSince(clock) / 20;
            clock = std::chrono::steady_clock::now();
            for(int r = 0; r < 20; r++){
                uint64_t at = 0;
                for(int i = 0; i < 3; i++){
                    uint64_t span = (uint64_t)parts[i]->rows * parts[i]->stride * warmed.size();
                    queue.copyUp(gpu.stage[0] + at, L.warmHost[i], (size_t)span, gpu.copyStream);
                    at += span;
                }
            }
            queue.streamSync(gpu.copyStream);
            double whole = millisSince(clock) / 20;
            static void* second = nullptr;
            if(!second) queue.streamCreate(&second, 1);
            clock = std::chrono::steady_clock::now();
            for(int r = 0; r < 20; r++){
                for(size_t j = 0; j < warmed.size(); j++){
                    for(int i = 0; i < 3; i++){
                        uint64_t span = (uint64_t)parts[i]->rows * parts[i]->stride;
                        queue.copyUp(gpu.stage[0] + (uint64_t)j * L.slice + L.piece[i], L.warmHost[i] + (uint64_t)L.warm[warmed[j]] * span, (size_t)span, (j * 3 + i) % 2 ? second : gpu.copyStream);
                    }
                }
            }
            queue.streamSync(gpu.copyStream);
            queue.streamSync(second);
            double twin = millisSince(clock) / 20;
            clock = std::chrono::steady_clock::now();
            int planed = 0;
            for(int r = 0; r < 20; r++){
                for(int i = 0; i < 3; i++){
                    uint64_t span = (uint64_t)parts[i]->rows * parts[i]->stride;
                    planed += queue.planeUp(gpu.stage[0] + L.piece[i], L.slice, L.warmHost[i], span, span, warmed.size(), gpu.copyStream) == 0;
                }
            }
            queue.streamSync(gpu.copyStream);
            double plane = millisSince(clock) / 20;
            if(round > 0) printf("  planes: %.2f ms (%.1f GB/s), %d of 60 accepted\n", plane, total / plane / 1e6, planed);
            if(round > 0) printf("  one copy per part of each expert: %.2f ms (%.1f GB/s), one copy per part: %.2f ms (%.1f GB/s), per part on two queues: %.2f ms (%.1f GB/s)\n", pieces, total / pieces / 1e6, whole, total / whole / 1e6, twin, total / twin / 1e6);
        }
        return 0;
    }
    if(const char* bench = getenv("KERNELPROF_MMABENCH")){
        int n = std::max(1, atoi(bench));
        int layer = getenv("KERNELPROF_BENCH_LAYER") ? atoi(getenv("KERNELPROF_BENCH_LAYER")) : gpu.firstLayer + 10;
        int count = getenv("KERNELPROF_BENCH_EXPERTS") ? atoi(getenv("KERNELPROF_BENCH_EXPERTS")) : 12;
        int repeats = getenv("KERNELPROF_BENCH_REPEATS") ? atoi(getenv("KERNELPROF_BENCH_REPEATS")) : 50;
        int pool = getenv("KERNELPROF_BENCH_TOKENS") ? atoi(getenv("KERNELPROF_BENCH_TOKENS")) : 23819;
        GpuLayer& Lt = gpu.layers[getenv("KERNELPROF_BENCH_LAYER") ? atoi(getenv("KERNELPROF_BENCH_LAYER")) : gpu.firstLayer + 10];
        std::string list = getenv("KERNELPROF_BENCH_KERNELS") ? getenv("KERNELPROF_BENCH_KERNELS") : gpu.tiled(Lt.expertGate.type, false, 0) + ":64," + gpu.tiled(Lt.expertGate.type, true, 0) + ":128";
        std::string downList = getenv("KERNELPROF_BENCH_DOWN") ? getenv("KERNELPROF_BENCH_DOWN") : gpu.tiled(Lt.expertDown.type, false, 0) + ":64," + gpu.tiled(Lt.expertDown.type, true, 0) + ":128";
        for(int l = gpu.firstLayer; l < gpu.endLayer; l++) printf("%u/%u/%u ", gpu.layers[l].expertGate.type, gpu.layers[l].expertUp.type, gpu.layers[l].expertDown.type);
        printf("\n");
        GpuLayer& L = gpu.layers[layer];
        count = std::min(count, gpu.residentExperts);
        std::mt19937 random(11);
        std::vector<int> slots(count), starts(count), counts(count, n);
        double skew = getenv("KERNELPROF_BENCH_SKEW") ? atof(getenv("KERNELPROF_BENCH_SKEW")) : 0.0;
        int routes = 0, widest = 1;
        for(int g = 0; g < count; g++){
            if(skew > 0.0) counts[g] = g == 0 ? (int)(skew * n) : (int)(n * (0.3 + 1.4 * (random() % 1000) / 1000.0));
            counts[g] = std::max(1, counts[g]);
            starts[g] = routes;
            routes += counts[g];
            widest = std::max(widest, counts[g]);
        }
        pool = std::max(pool, widest);
        std::vector<int> inputs(routes);
        std::vector<float> weights(routes, 0.1f);
        for(int g = 0; g < count; g++){
            slots[g] = g;
            std::vector<int> picked;
            std::set<int> seen;
            while((int)picked.size() < counts[g]){
                int token = (int)(random() % pool);
                if(seen.insert(token).second) picked.push_back(token);
            }
            std::sort(picked.begin(), picked.end());
            for(int j = 0; j < counts[g]; j++) inputs[starts[g] + j] = picked[j];
        }
        auto upload = [&](const void* data, uint64_t bytes){
            uint64_t ptr = 0;
            cudaLink().alloc(&ptr, (size_t)std::max<uint64_t>(bytes, 256));
            if(data) cudaLink().copyUp(ptr, data, (size_t)bytes);
            return ptr;
        };
        std::normal_distribution<float> spread(0.0f, 1.0f);
        std::vector<float> x((size_t)pool * hidden), h((size_t)routes * expertWidth);
        for(float& v : x) v = spread(random);
        for(float& v : h) v = 0.1f * spread(random);
        uint64_t slotOnGpu = upload(slots.data(), 4ULL * count), startOnGpu = upload(starts.data(), 4ULL * count), countOnGpu = upload(counts.data(), 4ULL * count);
        uint64_t inputOnGpu = upload(inputs.data(), 4ULL * routes), weightOnGpu = upload(weights.data(), 4ULL * routes);
        uint64_t xOnGpu = upload(x.data(), 4ULL * x.size()), hOnGpu = upload(h.data(), 4ULL * h.size());
        auto quantized = [&](uint64_t from, uint64_t total, PackedInput& a, PackedInput& b){
            a.values = upload(nullptr, total);
            a.scales = upload(nullptr, 4ULL * total / 32);
            a.sums = upload(nullptr, 4ULL * total / 32);
            b.values = upload(nullptr, total);
            b.scales = upload(nullptr, 4ULL * total / 32);
            b.sums = upload(nullptr, 4ULL * total / 32);
            gpu.go("quantizeFine", (unsigned int)((total / 32 + 7) / 8), 1, 32, 8, from, (int)total, a.values, a.scales, a.sums, b.values, b.scales, b.sums);
        };
        PackedInput a, b, c, d;
        quantized(xOnGpu, (uint64_t)pool * hidden, a, b);
        quantized(hOnGpu, (uint64_t)routes * expertWidth, c, d);
        uint64_t upOut = upload(nullptr, 4ULL * routes * expertWidth), gathered = upload(nullptr, 8ULL * pool * hidden);
        uint64_t none = 0;
        uint64_t gateSpan = (uint64_t)L.expertGate.rows * L.expertGate.stride, downSpan = (uint64_t)L.expertDown.rows * L.expertDown.stride;
        double upMacs = (double)routes * expertWidth * hidden, downMacs = (double)routes * hidden * expertWidth;
        printf("layer %d: gate type %u, down type %u, %d experts of %d tokens out of %d\n", layer, L.expertGate.type, L.expertDown.type, count, n, pool);
        auto run = [&](const std::string& names, bool down){
            std::vector<float> first;
            for(int round = 0; round < 2; round++){
                std::stringstream kernels(names);
                std::string item;
                while(std::getline(kernels, item, ',')){
                    size_t colon = item.find(':');
                    std::string name = item.substr(0, colon);
                    int width = colon == std::string::npos ? 64 : atoi(item.c_str() + colon + 1);
                    unsigned int threads = width == 128 ? 256 : 128;
                    int rowTile = name.size() > 0 && name.back() == 'X' ? 128 : 64;
                    int tiles = (widest + width - 1) / width;
                    auto call = [&]{
                        if(!down) gpu.go(name.c_str(), (unsigned int)(expertWidth / rowTile), (unsigned int)(count * tiles), threads, 1, (int)L.expertGate.type, L.resident[0], L.expertGate.stride, gateSpan, slotOnGpu, startOnGpu, countOnGpu, inputOnGpu, a.values, a.scales, a.sums, none, none, none, upOut, expertWidth, hidden, 0, tiles, none, none, none, gpu.tables);
                        else gpu.go(name.c_str(), (unsigned int)(hidden / rowTile), (unsigned int)(count * tiles), threads, 1, (int)L.expertDown.type, L.resident[2], L.expertDown.stride, downSpan, slotOnGpu, startOnGpu, countOnGpu, none, c.values, c.scales, c.sums, none, none, none, gathered, hidden, expertWidth, 0, tiles, inputOnGpu, weightOnGpu, gathered, gpu.tables);
                    };
                    gpu.failed = false;
                    cudaLink().memset(gathered, 0, 8ULL * pool * hidden);
                    call();
                    if(gpu.waitStream() != 0 || gpu.failed){
                        printf("%s: launch failed %s\n", name.c_str(), gpu.launchProblem.c_str());
                        gpu.failed = false;
                        continue;
                    }
                    std::vector<float> result;
                    if(!down){
                        result.resize((size_t)routes * expertWidth);
                        cudaLink().copyDown(result.data(), upOut, 4ULL * result.size());
                    } else {
                        std::vector<long long> fixed((size_t)pool * hidden);
                        cudaLink().copyDown(fixed.data(), gathered, 8ULL * fixed.size());
                        result.resize(fixed.size());
                        for(size_t i = 0; i < fixed.size(); i++) result[i] = (float)fixed[i];
                    }
                    if(first.empty()) first = result;
                    size_t same = 0;
                    for(size_t i = 0; i < result.size(); i++) same += result[i] == first[i];
                    auto clock = std::chrono::steady_clock::now();
                    for(int i = 0; i < repeats; i++) call();
                    gpu.waitStream();
                    double ms = millisSince(clock) / repeats;
                    if(round == 0) continue;
                    printf("  %-22s %-4s %8.1f us, %6.1f T multiply adds per second, same as the first %zu of %zu\n", name.c_str(), down ? "down" : "gate", ms * 1000, (down ? downMacs : upMacs) / ms / 1e9, same, result.size());
                }
            }
        };
        run(list, false);
        run(downList, true);
        return 0;
    }
    auto drifted = [&](const char* when){
        std::vector<int> table(gpu.layers[gpu.firstLayer].slot.size());
        size_t drift = 0, held = 0, first = 0;
        int firstLayer = -1;
        for(int l = gpu.firstLayer; l < gpu.endLayer; l++){
            GpuLayer& L = gpu.layers[l];
            int code = cudaLink().copyDown(table.data(), L.slots, table.size() * 4);
            for(size_t e = 0; e < table.size(); e++){
                held += L.slot[e] >= 0;
                if(table[e] == L.slot[e]) continue;
                drift++;
                if(firstLayer < 0){
                    firstLayer = l;
                    first = e;
                    printf("  first mismatch: layer %d expert %zu, gpu %d, host %d, copy code %d\n", l, e, table[e], L.slot[e], code);
                }
            }
        }
        printf("slot tables %s: %zu entries out of step, %zu resident on the host side\n", when, drift, held);
    };
    if(getenv("KERNELPROF_TABLES")) drifted("after load");
    if(gpu.vision) printf("vision: %s\n", gpu.seeing ? ("the picture reader runs on " + gpu.visionCard).c_str() : gpu.visionProblem.c_str());
    if(const char* forced = getenv("KERNELPROF_RESIDUAL")) gpu.residual = forced[0] == '1';
    if(getenv("KERNELPROF_REGS")){
        typedef int (__stdcall *FuncAttribute)(int*, int, void*);
        typedef int (__stdcall *Occupancy)(int*, void*, int, size_t);
        HMODULE lib = LoadLibraryA("nvcuda.dll");
        auto attribute = (FuncAttribute)GetProcAddress(lib, "cuFuncGetAttribute");
        auto occupancy = (Occupancy)GetProcAddress(lib, "cuOccupancyMaxActiveBlocksPerMultiprocessor");
        std::string asked = getenv("KERNELPROF_REGS");
        std::vector<std::string> names = {"attentionSpread", "attentionTiles", "rowsMulti", "rowsDirect14x1", "rowsDirect12x1", "mmaDeepLean", "mmaDeepSplitLean"};
        if(asked != "1"){
            names.clear();
            std::stringstream list(asked);
            std::string item;
            while(std::getline(list, item, ',')) names.push_back(item);
        }
        for(const std::string& kernel : names){
            const char* name = kernel.c_str();
            void* fn = kernelFunction(gpu.module, name);
            int regs = 0, spill = 0, shared = 0, blocks = 0;
            if(fn){
                attribute(&regs, 4, fn);
                attribute(&spill, 3, fn);
                attribute(&shared, 1, fn);
                occupancy(&blocks, fn, name[0] == 'm' ? 128 : 256, 0);
            }
            printf("%-20s %3d registers %5d spill bytes %6d shared bytes %d blocks per SM\n", name, regs, spill, shared, blocks);
        }
    }
    if(getenv("KERNELPROF_HELPER_BENCH") && gpu.helper.ready){
        typedef int (__stdcall *FuncAttribute)(int*, int, void*);
        auto attribute = (FuncAttribute)GetProcAddress(LoadLibraryA("nvcuda.dll"), "cuFuncGetAttribute");
        for(const char* name : {"helperGateUp", "helperDown", "expertRowsQ8"}){
            void* fn = kernelFunction(gpu.helper.module, name);
            int regs = 0, spill = 0;
            if(fn){
                attribute(&regs, 4, fn);
                attribute(&spill, 3, fn);
            }
            printf("second gpu kernel %s: %d registers, %d bytes of local memory\n", name, regs, spill);
        }
        GpuLayer& L = gpu.layers[gpu.layers.size() / 2];
        {
            for(int i = 0; i < 2; i++) gpu.helper.slotList[i] = i;
            cudaLink().setCurrent(gpu.helper.context);
            cudaQueue().copyUp(gpu.helper.inRowsOnCard, gpu.helper.inRows, sizeof(float) * 2 * hidden, nullptr);
            cudaQueue().copyUp(gpu.helper.slotsOnCard, gpu.helper.slotList, sizeof(int) * 2, nullptr);
            int gateType = (int)L.expertGate.type, upType = (int)L.expertUp.type;
            uint64_t gateRow = L.expertGate.stride, upRow = L.expertUp.stride;
            uint64_t gateSpan = (uint64_t)L.expertGate.rows * gateRow, upSpan = (uint64_t)L.expertUp.rows * upRow;
            int wide = hidden, upStage = (int)((std::max(gateRow, upRow) + 31) / 16 * 16);
            unsigned int upShared = (unsigned int)(3232 + 8 * 2 * upStage);
            packInput(gpu.helper.module, gpu.helper.inRowsOnCard, 2 * hidden, gpu.helper.packedIn);
            void* fn = kernelFunction(gpu.helper.module, "helperGateUp");
            for(int width : {expertWidth, 0}){
                void* args[] = {&gateType, &upType, &L.helpBase[0], &L.helpBase[1], &gateRow, &upRow, &gateSpan, &upSpan, &gpu.helper.slotsOnCard, &gpu.helper.inRowsOnCard, &gpu.helper.packedIn.values, &gpu.helper.packedIn.scales, &gpu.helper.packedIn.sums, &gpu.helper.hiddenOut, &width, &wide, &upStage, &gpu.helper.tables};
                cudaLink().sync();
                auto clock = std::chrono::steady_clock::now();
                for(int k = 0; k < 500; k++) launchKernel(fn, (unsigned int)((expertWidth + 7) / 8), 2, 32, 8, upShared, args);
                cudaLink().sync();
                printf("second gpu gate and up kernel alone, %s: %.1f us per launch\n", width ? "full" : "packing only", millisSince(clock) * 1000 / 500);
            }
            cudaLink().setCurrent(cudaLink().context);
        }
        {
            int rows = 2;
            for(int i = 0; i < rows; i++) gpu.helper.slotList[i] = (i * 7) % std::max(1, gpu.helper.perLayer);
            for(int i = 0; i < rows * hidden; i++) gpu.helper.inRows[i] = (float)((i * 2654435761u) % 2001) / 1000.0f - 1.0f;
            gpu.helperLaunch(L, rows);
            cudaLink().setCurrent(gpu.helper.context);
            cudaLink().sync();
            std::vector<float> fused(gpu.helper.outRows, gpu.helper.outRows + rows * hidden);
            uint64_t gateSpan = (uint64_t)L.expertGate.rows * L.expertGate.stride, upSpan = (uint64_t)L.expertUp.rows * L.expertUp.stride, downSpan = (uint64_t)L.expertDown.rows * L.expertDown.stride;
            int count = rows * expertWidth;
            void* swiglu = kernelFunction(gpu.helper.module, "swiglu");
            void* swigluArgs[] = {&gpu.helper.gateOut, &gpu.helper.upOut, &gpu.helper.hiddenOut, &count};
            packInput(gpu.helper.module, gpu.helper.inRowsOnCard, rows * hidden, gpu.helper.packedIn);
            launchExpertsQ8(gpu.helper.module, (int)L.expertGate.type, L.helpBase[0], L.expertGate.stride, gateSpan, gpu.helper.slotsOnCard, rows, gpu.helper.inRowsOnCard, gpu.helper.packedIn, (uint64_t)hidden, gpu.helper.gateOut, expertWidth, hidden, gpu.helper.tables);
            launchExpertsQ8(gpu.helper.module, (int)L.expertUp.type, L.helpBase[1], L.expertUp.stride, upSpan, gpu.helper.slotsOnCard, rows, gpu.helper.inRowsOnCard, gpu.helper.packedIn, (uint64_t)hidden, gpu.helper.upOut, expertWidth, hidden, gpu.helper.tables);
            launchKernel(swiglu, (unsigned int)((count + 255) / 256), 1, 256, 1, 0, swigluArgs);
            packInput(gpu.helper.module, gpu.helper.hiddenOut, count, gpu.helper.packedHidden);
            launchExpertsQ8(gpu.helper.module, (int)L.expertDown.type, L.helpBase[2], L.expertDown.stride, downSpan, gpu.helper.slotsOnCard, rows, gpu.helper.hiddenOut, gpu.helper.packedHidden, (uint64_t)expertWidth, gpu.helper.outMapped, hidden, expertWidth, gpu.helper.tables);
            cudaLink().sync();
            std::vector<float> stepped(gpu.helper.outRows, gpu.helper.outRows + rows * hidden);
            launchExperts(gpu.helper.module, (int)L.expertGate.type, L.helpBase[0], L.expertGate.stride, gateSpan, gpu.helper.slotsOnCard, rows, gpu.helper.inRowsOnCard, (uint64_t)hidden, gpu.helper.gateOut, expertWidth, hidden, gpu.helper.tables);
            launchExperts(gpu.helper.module, (int)L.expertUp.type, L.helpBase[1], L.expertUp.stride, upSpan, gpu.helper.slotsOnCard, rows, gpu.helper.inRowsOnCard, (uint64_t)hidden, gpu.helper.upOut, expertWidth, hidden, gpu.helper.tables);
            launchKernel(swiglu, (unsigned int)((count + 255) / 256), 1, 256, 1, 0, swigluArgs);
            launchExperts(gpu.helper.module, (int)L.expertDown.type, L.helpBase[2], L.expertDown.stride, downSpan, gpu.helper.slotsOnCard, rows, gpu.helper.hiddenOut, (uint64_t)expertWidth, gpu.helper.outMapped, hidden, expertWidth, gpu.helper.tables);
            cudaLink().sync();
            std::vector<float> exact(gpu.helper.outRows, gpu.helper.outRows + rows * hidden);
            cudaLink().setCurrent(cudaLink().context);
            auto compare = [&](const std::vector<float>& a, const std::vector<float>& b){
                double worst = 0, scale = 0, dot = 0, na = 0, nb = 0;
                for(size_t i = 0; i < a.size(); i++){
                    worst = std::max(worst, (double)fabsf(a[i] - b[i]));
                    scale = std::max(scale, (double)fabsf(b[i]));
                    dot += (double)a[i] * b[i];
                    na += (double)a[i] * a[i];
                    nb += (double)b[i] * b[i];
                }
                return std::make_pair(worst / std::max(1e-12, scale), dot / std::max(1e-30, sqrt(na * nb)));
            };
            auto same = compare(fused, stepped), close = compare(fused, exact), old = compare(stepped, exact);
            printf("second gpu check: new vs old int8 path worst %.3g of peak, cosine %.7f; new vs float path worst %.3g, cosine %.7f; old int8 vs float worst %.3g, cosine %.7f\n", same.first, same.second, close.first, close.second, old.first, old.second);
        }
        if(const char* hold = getenv("KERNELPROF_HELPER_HOLD")){
            for(int i = 0; i < 2; i++) gpu.helper.slotList[i] = i;
            auto clock = std::chrono::steady_clock::now();
            int calls = 0;
            while(millisSince(clock) < 1000.0 * atoi(hold)){
                gpu.helperLaunch(L, 2);
                calls++;
                if(calls % 64 == 0){
                    cudaLink().setCurrent(gpu.helper.context);
                    cudaLink().sync();
                    cudaLink().setCurrent(cudaLink().context);
                }
            }
            cudaLink().setCurrent(gpu.helper.context);
            cudaLink().sync();
            cudaLink().setCurrent(cudaLink().context);
            printf("second gpu held busy: %.1f us per 2 row call\n", millisSince(clock) * 1000 / calls);
        }
        for(int rows : {1, 2, 4, 8}){
            for(int i = 0; i < rows; i++) gpu.helper.slotList[i] = i % std::max(1, gpu.helper.perLayer);
            memset(gpu.helper.inRows, 0, sizeof(float) * rows * hidden);
            for(int i = 0; i < rows * hidden; i++) gpu.helper.inRows[i] = (float)((i * 2654435761u) % 1000) / 1000.0f - 0.5f;
            gpu.timeHelper = true;
            double gateMs = 0, downMs = 0;
            auto clock = std::chrono::steady_clock::now();
            for(int k = 0; k < 200; k++){
                gpu.helperLaunch(L, rows);
                cudaLink().setCurrent(gpu.helper.context);
                cudaLink().sync();
                float part = 0.0f;
                if(cudaQueue().eventElapsed(&part, gpu.helper.marks[0], gpu.helper.marks[1]) == 0) gateMs += part;
                if(cudaQueue().eventElapsed(&part, gpu.helper.marks[3], gpu.helper.marks[4]) == 0) downMs += part;
                cudaLink().setCurrent(cudaLink().context);
            }
            double each = millisSince(clock) / 200;
            gpu.timeHelper = getenv("KERNELPROF_HELPER_TIMES") != nullptr;
            printf("second gpu alone, %d rows: gate and up %.1f us, down %.1f us\n", rows, gateMs * 1000 / 200, downMs * 1000 / 200);
            clock = std::chrono::steady_clock::now();
            for(int k = 0; k < 200; k++) gpu.helperLaunch(L, rows);
            cudaLink().setCurrent(gpu.helper.context);
            cudaLink().sync();
            cudaLink().setCurrent(cudaLink().context);
            double queued = millisSince(clock) / 200;
            printf("second gpu alone, %d rows: %.1f us per call waiting each time, %.1f us per call queued\n", rows, each * 1000, queued * 1000);
        }
    }
    std::vector<int> positions(prompt.size());
    std::iota(positions.begin(), positions.end(), 0);
    const char* reading = getenv("KERNELPROF_PROMPT");
    if(reading){
        if(reading[0] == '2' || reading[0] == '3'){
            gpu.forward(prompt, positions);
            gpu.reset();
            gpu.coldMs = 0;
            gpu.coldRouted = 0;
        }
        gpu.profile = reading[0] == '1' || reading[0] == '2';
        gpu.kernelMs.clear();
        gpu.kernelCalls.clear();
        size_t freeBytes = 0, totalBytes = 0;
        cudaLink().info(&freeBytes, &totalBytes);
        printf("free video memory before the prompt: %.2f GB of %.2f GB, context room %d tokens, recent slots per layer %d, fixed %d\n", freeBytes / 1073741824.0, totalBytes / 1073741824.0, gpu.kvRoom, gpu.recentPerLayer, gpu.fixedExperts);
    }
    auto readClock = std::chrono::steady_clock::now();
    std::vector<float> logits = gpu.forward(prompt, positions);
    double readMs = millisSince(readClock);
    if(logits.empty()){
        printf("prefill failed: %s\n", gpu.problem.c_str());
        return 1;
    }
    if(const char* second = getenv("KERNELPROF_SECOND")){
        std::ifstream more(second, std::ios::binary);
        std::stringstream body;
        body << more.rdbuf();
        std::vector<int> next = tokenizer.encode(body.str(), true);
        int rows = getenv("KERNELPROF_SECOND_ROWS") ? atoi(getenv("KERNELPROF_SECOND_ROWS")) : 9237;
        rows = std::max(1, std::min(rows, (int)next.size() - 6));
        auto bad = [](const std::vector<float>& v){
            int count = 0;
            for(float f : v) count += !std::isfinite(f);
            return count;
        };
        gpu.reset();
        std::vector<int> head(next.begin(), next.begin() + rows), where(rows);
        std::iota(where.begin(), where.end(), 0);
        std::vector<float> got = gpu.forward(head, where);
        printf("second prompt: %d rows after a reset, %d non finite logits of %zu, borrowed rows %d%s%s\n", rows, bad(got), got.size(), gpu.loanUsed, got.empty() ? ", failed: " : "", got.empty() ? gpu.problem.c_str() : "");
        std::vector<int> ending(next.begin() + rows, next.begin() + rows + 6), at(6);
        std::iota(at.begin(), at.end(), rows);
        got = gpu.forward(ending, at);
        printf("second prompt tail: 6 rows, %d non finite logits of %zu%s%s\n", bad(got), got.size(), got.empty() ? ", failed: " : "", got.empty() ? gpu.problem.c_str() : "");
    }
    if(reading){
        if(const char* keep = getenv("KERNELPROF_LOGITS")){
            std::ofstream saved(keep, std::ios::binary);
            saved.write((const char*)logits.data(), logits.size() * 4);
        }
        if(const char* keep = getenv("KERNELPROF_DRAFTDUMP")){
            if(gpu.drafting && gpu.draft.keys){
                size_t bytes = prompt.size() * (size_t)kvHeads * headWidth;
                std::vector<uint8_t> keys(bytes), values(bytes);
                cudaLink().copyDown(keys.data(), gpu.draft.keys, bytes);
                cudaLink().copyDown(values.data(), gpu.draft.values, bytes);
                std::ofstream saved(keep, std::ios::binary);
                saved.write((const char*)keys.data(), bytes);
                saved.write((const char*)values.data(), bytes);
            }
        }
        printf("prompt read: %zu tokens in %.2f s, %.0f tokens per second, %s, borrowed rows %d\n", prompt.size(), readMs / 1000.0, prompt.size() * 1000.0 / readMs, gpu.profile ? "with kernel marks" : "plain", gpu.loanUsed);
        if(gpu.drafting) printf("draft feed %.0f ms\n", gpu.feedMs);
        printf("host side: embeddings %.0f ms, ple %.0f ms, waiting for routes %.0f ms, expert stream %.0f ms (ring copies %.0f ms, pack reads %.0f ms), cpu experts %.0f ms, cold experts on the cpu %.0f ms for %llu routes\n", gpu.embedMs, gpu.pleMs, gpu.waitMs, gpu.streamMs, gpu.copyMs, gpu.readMs, gpu.expertMs, gpu.coldMs, (unsigned long long)gpu.coldRouted);
        std::vector<std::pair<double, std::string>> ranked;
        std::map<std::string, double> byPhase;
        double total = 0.0;
        for(auto& k : gpu.kernelMs){
            ranked.push_back({k.second, k.first});
            total += k.second;
            byPhase[k.first.substr(0, k.first.find(' '))] += k.second;
        }
        std::sort(ranked.rbegin(), ranked.rend());
        printf("kernels %.0f ms of %.0f ms wall\n", total, readMs);
        for(auto& p : byPhase) printf("  phase %-12s %9.1f ms  %5.1f%%\n", p.first.c_str(), p.second, 100.0 * p.second / std::max(1.0, total));
        for(size_t i = 0; i < ranked.size() && i < 40; i++) printf("  %-40s %9.1f ms  %7d calls  %8.1f us each\n", ranked[i].second.c_str(), ranked[i].first, gpu.kernelCalls[ranked[i].second], 1000.0 * ranked[i].first / std::max(1, gpu.kernelCalls[ranked[i].second]));
        return 0;
    }
    int next = bestOf(logits);
    // KERNELPROF_SPECULATE=N escribe N tokens con borradores y verificacion como el servidor y despues uno por paso desde el mismo prompt
    // y dice si los dos textos coinciden y cuanto tarda cada uno (KERNELPROF_SPEC_DEPTH borradores por vuelta, KERNELPROF_SPEC_SURE el piso)
    if(const char* speculating = getenv("KERNELPROF_SPECULATE")){
        int count = std::max(16, atoi(speculating));
        int depth = getenv("KERNELPROF_SPEC_DEPTH") ? std::max(1, atoi(getenv("KERNELPROF_SPEC_DEPTH"))) : 3;
        if(getenv("KERNELPROF_SPEC_SURE")) gpu.draftSure = (float)atof(getenv("KERNELPROF_SPEC_SURE"));
        if(!gpu.drafting){
            printf("no draft is loaded: %s\n", gpu.draftProblem.c_str());
            return 1;
        }
        if(!gpu.saveState(2)){
            printf("the state could not be saved\n");
            return 1;
        }
        std::vector<int> made = {next};
        int checks = 0, drafted = 0, kept = 0, singles = 0;
        double proposeMs = 0, checkMs = 0;
        auto clock = std::chrono::steady_clock::now();
        while((int)made.size() <= count){
            int token = made.back();
            auto asked = std::chrono::steady_clock::now();
            std::vector<int> drafts = gpu.draftPropose(token, depth);
            proposeMs += millisSince(asked);
            if(drafts.empty()){
                std::vector<float> one = gpu.forward({token}, {(int)gpu.tokens.size()});
                if(one.empty()){
                    printf("a single step failed: %s\n", gpu.problem.c_str());
                    return 1;
                }
                made.push_back(bestOf(one));
                singles++;
                continue;
            }
            std::vector<int> batch = {token};
            batch.insert(batch.end(), drafts.begin(), drafts.end());
            std::vector<int> where(batch.size());
            for(size_t i = 0; i < batch.size(); i++) where[i] = (int)gpu.tokens.size() + (int)i;
            asked = std::chrono::steady_clock::now();
            if(!gpu.markRows((int)batch.size())){
                printf("the rows could not be marked\n");
                return 1;
            }
            std::vector<float> rows = gpu.forwardAll(batch, where);
            checkMs += millisSince(asked);
            if(rows.empty() || rows.size() % batch.size() != 0){
                printf("a check failed: %s\n", gpu.problem.c_str());
                return 1;
            }
            size_t width = rows.size() / batch.size();
            int agreed = 0, fix = -1;
            for(size_t i = 0; i < drafts.size(); i++){
                int best = (int)(std::max_element(rows.begin() + i * width, rows.begin() + (i + 1) * width) - (rows.begin() + i * width));
                if(best != drafts[i]){
                    fix = best;
                    break;
                }
                agreed++;
            }
            checks++;
            drafted += (int)drafts.size();
            kept += agreed;
            for(int i = 0; i < agreed; i++) made.push_back(drafts[i]);
            if(agreed == (int)drafts.size()){
                gpu.settleRows(batch);
                made.push_back((int)(std::max_element(rows.end() - width, rows.end()) - (rows.end() - width)));
                continue;
            }
            std::vector<int> refeed(batch.begin(), batch.begin() + 1 + agreed);
            if(!gpu.commitRows(refeed)){
                printf("the kept rows could not be committed\n");
                return 1;
            }
            made.push_back(fix);
        }
        double speculateMs = millisSince(clock);
        int written = (int)made.size() - 1;
        if(!gpu.restoreState(2)){
            printf("the state could not be restored\n");
            return 1;
        }
        std::vector<int> plain = {next};
        clock = std::chrono::steady_clock::now();
        while((int)plain.size() < (int)made.size()){
            std::vector<float> one = gpu.forward({plain.back()}, {(int)gpu.tokens.size()});
            if(one.empty()){
                printf("a plain step failed: %s\n", gpu.problem.c_str());
                return 1;
            }
            plain.push_back(bestOf(one));
        }
        double plainMs = millisSince(clock);
        int same = 0;
        while(same < (int)made.size() && made[same] == plain[same]) same++;
        printf("with drafts: %d tokens in %.0f ms (%.1f tok/s), %d checks, %d single steps, %d of %d drafts kept (%.1f%%), %.2f tokens per check, propose %.2f ms and check %.2f ms per check\n", written, speculateMs, written * 1000.0 / speculateMs, checks, singles, kept, drafted, 100.0 * kept / std::max(1, drafted), checks ? (double)(kept + checks) / checks : 0.0, proposeMs / std::max(1, checks), checkMs / std::max(1, checks));
        printf("one per step: %d tokens in %.0f ms (%.1f tok/s)\n", written, plainMs, written * 1000.0 / plainMs);
        printf("graphs: %zu recorded, %zu could not be recorded%s%s\n", gpu.tokenGraphs.size(), gpu.graphless.size(), gpu.graphProblem.empty() ? "" : ", last problem: ", gpu.graphProblem.c_str());
        printf("same tokens for the first %d of %zu%s\n", same, made.size(), same == (int)made.size() ? " (all)" : "");
        if(same < (int)made.size()){
            int from = std::max(0, same - 8);
            std::vector<int> a(made.begin() + from, made.begin() + std::min(made.size(), (size_t)same + 8));
            std::vector<int> b(plain.begin() + from, plain.begin() + std::min(plain.size(), (size_t)same + 8));
            printf("  with drafts: %s\n  one per step: %s\n", tokenizer.decode(a).c_str(), tokenizer.decode(b).c_str());
        }
        printf("text: %s\n", tokenizer.decode(std::vector<int>(made.begin(), made.begin() + std::min<size_t>(made.size(), 160))).c_str());
        return 0;
    }
    // KERNELPROF_GEMMA_ATTN_CHECK=1 compara la atencion de pocas filas con la de siempre sobre las claves reales del prompt
    if(getenv("KERNELPROF_GEMMA_ATTN_CHECK") && gpu.gemma){
        GemmaShape& s = gpu.gemmaShape;
        GemmaBuffers& b = gpu.gemmaBuffers;
        int top = (int)gpu.tokens.size() - 1;
        std::mt19937 random(7);
        std::normal_distribution<float> spread(0.0f, 1.0f);
        for(int n : {1, 4, 77}){
            for(int l = 0; l < s.layers; l++){
                GemmaLayer& L = gpu.gemmaLayers[l];
                if(l > 0 && L.sliding == gpu.gemmaLayers[l - 1].sliding) continue;
                int width = s.heads * L.headSize;
                std::vector<float> query((size_t)n * width);
                for(float& v : query) v = spread(random);
                for(int t = 0; t < n; t++){
                    float* row = query.data() + (size_t)t * width;
                    for(int h = 0; h < s.heads; h++){
                        double sum = 0;
                        for(int d = 0; d < L.headSize; d++) sum += row[h * L.headSize + d] * row[h * L.headSize + d];
                        float inv = (float)(1.0 / sqrt(sum / L.headSize));
                        for(int d = 0; d < L.headSize; d++) row[h * L.headSize + d] *= inv;
                    }
                }
                std::vector<int> places(n);
                for(int t = 0; t < n; t++) places[t] = top - n + 1 + t;
                cudaLink().copyUp(b.q, query.data(), query.size() * 4);
                cudaLink().copyUp(b.positions, places.data(), (size_t)n * 4);
                int ring = L.sliding ? s.ring : 0, window = L.sliding ? s.window : 0;
                std::vector<float> old((size_t)n * width), fresh((size_t)n * width);
                int tileTokens = L.headSize == 256 ? 8 : 2;
                gpu.go(L.headSize == 256 ? "gemmaAttention256" : "gemmaAttention512", (unsigned int)L.kvHeads, (unsigned int)((n + tileTokens - 1) / tileTokens), 256, 1, b.q, L.keys, L.values, b.positions, b.o, b.scratch, n, s.heads, L.kvHeads, ring, window, 1);
                cudaLink().sync();
                cudaLink().copyDown(old.data(), b.o, old.size() * 4);
                for(int splits : {1, 3, 8}){
                    if(n > 16){
                        if(splits > 1) break;
                        bool narrow = L.headSize == 256;
                        int per = narrow ? 32 : 8;
                        gpu.goShared(narrow ? 67968 : 100000, narrow ? "gemmaFlash256" : "gemmaFlash512", (unsigned int)L.kvHeads, (unsigned int)((n + per - 1) / per), narrow ? 128 : 256, 1, b.q, L.keys, L.values, b.positions, b.o, n, s.heads, L.kvHeads, ring, window);
                    } else {
                        gpu.go(L.headSize == 256 ? "gemmaFew256" : "gemmaFew512", (unsigned int)L.kvHeads, (unsigned int)(n * splits), 256, 1, b.q, L.keys, L.values, b.positions, b.o, b.scratch, s.heads, L.kvHeads, ring, window, splits);
                        if(splits > 1) gpu.go("gemmaMerge", (unsigned int)s.heads, (unsigned int)n, 256, 1, b.scratch, b.o, s.heads, L.headSize, splits);
                    }
                    cudaLink().sync();
                    cudaLink().copyDown(fresh.data(), b.o, fresh.size() * 4);
                    double worst = 0, peak = 0;
                    for(size_t i = 0; i < old.size(); i++){
                        worst = std::max(worst, (double)fabsf(old[i] - fresh[i]));
                        peak = std::max(peak, (double)fabsf(old[i]));
                    }
                    printf("layer %d (%s, head %d), %d rows at %d, %d splits: largest difference %.3e of values up to %.3e%s\n", l, L.sliding ? "window" : "full", L.headSize, n, top, splits, worst, peak, gpu.failed ? ", a launch failed" : "");
                }
            }
        }
        return 0;
    }
    // KERNELPROF_GEMMA_FUSE_CHECK=1 compara los kernels juntados de gemma con la cadena vieja sobre los buffers reales del prompt
    if(getenv("KERNELPROF_GEMMA_FUSE_CHECK") && gpu.gemma){
        GemmaShape& s = gpu.gemmaShape;
        GemmaBuffers& b = gpu.gemmaBuffers;
        GemmaLayer& L = gpu.gemmaLayers[0];
        int n = 8, d = s.hidden;
        uint64_t none = 0;
        float one = 1.0f;
        auto grab = [&](uint64_t bytes){ uint64_t p = 0; cudaLink().alloc(&p, (size_t)bytes); cudaLink().memset(p, 0, (size_t)bytes); return p; };
        auto down = [&](uint64_t p, size_t count){ std::vector<float> v(count); cudaLink().copyDown(v.data(), p, count * 4); return v; };
        auto bytesDown = [&](uint64_t p, size_t count){ std::vector<int8_t> v(count); cudaLink().copyDown(v.data(), p, count); return v; };
        auto worst = [&](const std::vector<float>& a, const std::vector<float>& c){ double w = 0; for(size_t i = 0; i < a.size(); i++) w = std::max(w, (double)fabsf(a[i] - c[i])); return w; };
        uint64_t outA = grab(4ULL * n * d), outB = grab(4ULL * n * d), xnA = grab(4ULL * n * d), xnB = grab(4ULL * n * d);
        uint64_t qa = grab((uint64_t)n * d), qb = grab((uint64_t)n * d), sa = grab(4ULL * n * d / 32), sb = grab(4ULL * n * d / 32), ua = grab(4ULL * n * d / 32), ub = grab(4ULL * n * d / 32);
        uint64_t fq = grab((uint64_t)n * d), fs = grab(4ULL * n * d / 32), fu = grab(4ULL * n * d / 32);
        gpu.go("gemmaRmsAdd", (unsigned int)n, 1, 256, 1, b.attn, L.postAttnNorm, b.x, outA, d, s.eps, none);
        gpu.go("gemmaRms", (unsigned int)n, 1, 256, 1, outA, L.ffnNorm, xnA, d, s.eps, one, none);
        gpu.go("quantizeFine", (unsigned int)((n * d / 32 + 7) / 8), 1, 32, 8, xnA, n * d, qa, sa, ua, fq, fs, fu);
        gpu.go("gemmaAddRmsPack", (unsigned int)n, 1, 256, 1, b.attn, L.postAttnNorm, b.x, outB, L.ffnNorm, xnB, d, s.eps, qb, sb, ub);
        cudaLink().sync();
        auto pa = bytesDown(qa, (size_t)n * d), pb = bytesDown(qb, (size_t)n * d);
        size_t packDiff = 0;
        for(size_t i = 0; i < pa.size(); i++) packDiff += pa[i] != pb[i];
        printf("add and norm: residual out %.3e, normed %.3e, packed values differing %zu of %zu, scales %.3e%s\n", worst(down(outA, (size_t)n * d), down(outB, (size_t)n * d)), worst(down(xnA, (size_t)n * d), down(xnB, (size_t)n * d)), packDiff, pa.size(), worst(down(sa, (size_t)n * d / 32), down(sb, (size_t)n * d / 32)), gpu.failed ? ", a launch failed" : "");
        if(L.mixture){
            int used = s.expertsUsed;
            uint64_t moe = grab(4ULL * n * d), pair = grab(4ULL * n * d);
            gpu.go("gemmaMix", (unsigned int)n, 1, 256, 1, b.expertOut, b.weights, L.downScale, b.ids, moe, used, d);
            gpu.go("gemmaPair", (unsigned int)n, 1, 256, 1, b.mlp, L.postMlpNorm, moe, L.postMoeNorm, pair, d, s.eps);
            gpu.go("gemmaRmsAdd", (unsigned int)n, 1, 256, 1, pair, L.postFfnNorm, b.a, outA, d, s.eps, L.outScale);
            gpu.goShared((unsigned int)(8 * d), "gemmaMoeOut", (unsigned int)n, 1, 256, 1, b.expertOut, b.weights, L.downScale, b.ids, none, b.mlp, L.postMlpNorm, L.postMoeNorm, L.postFfnNorm, b.a, outB, L.outScale, used, d, s.eps);
            cudaLink().sync();
            printf("experts out: %.3e%s\n", worst(down(outA, (size_t)n * d), down(outB, (size_t)n * d)), gpu.failed ? ", a launch failed" : "");
        }
        return 0;
    }
    // KERNELPROF_STEP_MARKS=N mide N pasos de un token sin marcas y despues con una marca en cada kernel
    if(const char* marking = getenv("KERNELPROF_STEP_MARKS")){
        int count = std::max(1, atoi(marking));
        auto steps = [&](int many){
            auto clock = std::chrono::steady_clock::now();
            for(int i = 0; i < many; i++){
                logits = gpu.forward({next}, {(int)gpu.tokens.size()});
                if(logits.empty()) return -1.0;
                next = bestOf(logits);
            }
            return millisSince(clock) / many;
        };
        steps(8);
        double plain = steps(count);
        gpu.profile = true;
        gpu.kernelMs.clear();
        gpu.kernelCalls.clear();
        double marked = steps(count);
        gpu.profile = false;
        std::vector<std::pair<double, std::string>> ranked;
        double total = 0;
        for(auto& k : gpu.kernelMs){
            ranked.push_back({k.second / count, k.first});
            total += k.second / count;
        }
        std::sort(ranked.rbegin(), ranked.rend());
        printf("one token per step: %.3f ms plain, %.3f ms with marks, kernels %.3f ms per step\n", plain, marked, total);
        for(size_t i = 0; i < ranked.size() && i < 40; i++) printf("  %-48s %7.3f ms  %6.1f calls  %7.1f us each\n", ranked[i].second.c_str(), ranked[i].first, (double)gpu.kernelCalls[ranked[i].second] / count, 1000.0 * ranked[i].first * count / std::max(1, gpu.kernelCalls[ranked[i].second]));
        return 0;
    }
    // KERNELPROF_PROPOSE=N mide N llamadas a los borradores despues del prompt: cuanto cuestan con este contexto y en que kernels
    if(const char* proposing = getenv("KERNELPROF_PROPOSE")){
        int calls = std::max(1, atoi(proposing));
        int depth = getenv("KERNELPROF_PROPOSE_DEPTH") ? std::max(1, atoi(getenv("KERNELPROF_PROPOSE_DEPTH"))) : 3;
        gpu.draftSure = 0.0f;
        if(!gpu.drafting){
            printf("no draft layer is loaded\n");
            return 1;
        }
        gpu.draftPropose(next, depth);
        gpu.profile = getenv("KERNELPROF_PROPOSE_MARKS") != nullptr;
        gpu.kernelMs.clear();
        gpu.kernelCalls.clear();
        auto clock = std::chrono::steady_clock::now();
        size_t made = 0;
        for(int i = 0; i < calls; i++) made += gpu.draftPropose(next, depth).size();
        double each = millisSince(clock) / calls;
        printf("draft proposals with %zu tokens of context: %.3f ms per call of up to %d drafts, %.2f made on average\n", gpu.tokens.size(), each, depth, made / (double)calls);
        std::vector<std::pair<double, std::string>> ranked;
        double total = 0;
        for(auto& k : gpu.kernelMs){
            ranked.push_back({k.second / calls, k.first});
            total += k.second / calls;
        }
        std::sort(ranked.rbegin(), ranked.rend());
        if(gpu.profile) printf("kernels per call %.3f ms\n", total);
        for(size_t i = 0; i < ranked.size() && i < 20; i++) printf("  %-40s %7.3f ms  %5.1f calls\n", ranked[i].second.c_str(), ranked[i].first, (double)gpu.kernelCalls[ranked[i].second] / calls);
        return 0;
    }
    std::vector<float> firstLogits = logits;
    std::vector<int> trail;
    double markMs = 0.0, commitMs = 0.0, verifyMs = 0.0;
    bool settle = getenv("KERNELPROF_SNAPSHOT") == nullptr;
    int keep = getenv("KERNELPROF_KEEP") ? std::max(1, std::min(window, atoi(getenv("KERNELPROF_KEEP")))) : 1;
    if(const char* ahead = getenv("KERNELPROF_FUTURE")){
        int count = std::max(16, atoi(ahead));
        int span = std::max(2, window);
        if(!gpu.saveState(2)){
            printf("the state could not be saved for the future run\n");
            return 1;
        }
        std::vector<int> future = {next};
        bool drifting = getenv("KERNELPROF_DRIFT") != nullptr;
        std::vector<std::vector<float>> truth;
        double driftSum = 0.0, driftWorst = 0.0;
        int driftRows = 0;
        auto clock = std::chrono::steady_clock::now();
        for(int i = 0; i < count; i++){
            std::vector<float> one = gpu.forward({future.back()}, {(int)gpu.tokens.size()});
            if(one.empty()){
                printf("the future run failed: %s\n", gpu.problem.c_str());
                return 1;
            }
            future.push_back(bestOf(one));
            if(drifting) truth.push_back(std::move(one));
        }
        printf("one token per step: %.2f ms per token over %d tokens\n", millisSince(clock) / count, count);
        uint64_t missedPairs = 0, missedKinds = 0;
        int passCount = 0;
        gpu.timing = getenv("KERNELPROF_TIMELINE") != nullptr;
        int depth = gpu.endLayer - gpu.firstLayer;
        std::vector<double> frontSum(depth, 0.0), residentSum(depth, 0.0), waitLayer(depth, 0.0), cpuLayer(depth, 0.0);
        std::vector<int> waitBins(8, 0), cpuBins(8, 0), rowCount(24, 0);
        std::vector<double> rowCpu(24, 0.0), rowWait(24, 0.0), rowWorst(24, 0.0), rowGather(24, 0.0);
        int timed = 0;
        int profiledCount = 24;
        double serveSum = 0.0, waitSum = 0.0, expertSum = 0.0;
        auto passes = [&](bool profiled, int most){
            gpu.restoreState(2);
            gpu.profile = profiled;
            gpu.useGraphs = !profiled;
            gpu.relayAlways = profiled;
            gpu.kernelMs.clear();
            gpu.kernelCalls.clear();
            size_t cursor = 0;
            int done = 0, agree = 0;
            auto start = std::chrono::steady_clock::now();
            while(cursor + span < future.size() && done < most){
                std::vector<int> ids(future.begin() + cursor, future.begin() + cursor + span), where(span);
                for(int i = 0; i < span; i++) where[i] = (int)gpu.tokens.size() + i;
                if(settle) gpu.markRows(span);
                std::vector<float> rows = gpu.forwardAll(ids, where);
                missedPairs += gpu.missedRoutes;
                missedKinds += gpu.missedUnique;
                passCount++;
                for(double ms : gpu.layerMs) serveSum += ms;
                waitSum += gpu.waitMs;
                expertSum += gpu.expertMs;
                if(gpu.timing && !profiled && !rows.empty()){
                    for(int l = gpu.firstLayer; l < gpu.endLayer; l++){
                        int i = l - gpu.firstLayer;
                        const volatile unsigned long long* t = gpu.hostTimeline + 4 * l;
                        double resident = (double)(t[1] - t[0]) / 1000.0, wait = (double)(t[2] - t[1]) / 1000.0;
                        double front = l > gpu.firstLayer ? (double)(t[0] - gpu.hostTimeline[4 * (l - 1) + 2]) / 1000.0 : 0.0;
                        double cpu = (gpu.cpuDone[l] - gpu.cpuSaw[l]) * 1000.0;
                        frontSum[i] += front;
                        residentSum[i] += resident;
                        waitLayer[i] += wait;
                        cpuLayer[i] += cpu;
                        waitBins[std::min(7, (int)(wait / 50.0))]++;
                        int seen = std::min(23, gpu.cpuSeen[l]);
                        rowCount[seen]++;
                        rowCpu[seen] += cpu;
                        rowWait[seen] += wait;
                        rowGather[seen] += gpu.cpuGather[l] * 1000.0;
                        rowWorst[seen] = std::max(rowWorst[seen], cpu);
                        cpuBins[std::min(7, (int)(cpu / 50.0))]++;
                    }
                    timed++;
                }
                if(rows.empty()){
                    printf("a verify pass failed: %s\n", gpu.problem.c_str());
                    return 0.0;
                }
                size_t width = rows.size() / span;
                for(int i = 0; i < span; i++) agree += bestOf(std::vector<float>(rows.begin() + i * width, rows.begin() + (i + 1) * width)) == future[cursor + i + 1];
                for(int i = 0; i < span && drifting && !profiled; i++){
                    const float* a = truth[cursor + i].data();
                    const float* b = rows.data() + i * width;
                    double topA = -1e30, topB = -1e30, sumA = 0.0, sumB = 0.0, kl = 0.0;
                    for(size_t v = 0; v < width; v++){
                        topA = std::max(topA, (double)a[v]);
                        topB = std::max(topB, (double)b[v]);
                    }
                    for(size_t v = 0; v < width; v++){
                        sumA += exp(a[v] - topA);
                        sumB += exp(b[v] - topB);
                    }
                    for(size_t v = 0; v < width; v++){
                        double p = exp(a[v] - topA) / sumA;
                        if(p > 1e-12) kl += p * ((a[v] - topA - log(sumA)) - (b[v] - topB - log(sumB)));
                    }
                    driftSum += kl;
                    driftWorst = std::max(driftWorst, kl);
                    driftRows++;
                }
                if(settle) gpu.settleRows(ids);
                cursor += span;
                done++;
            }
            double each = millisSince(start) / std::max(1, done);
            if(profiled) profiledCount = std::max(1, done);
            printf("%s: %d passes of %d tokens, %.2f ms per pass, %.1f tokens per second if every draft is kept, %.1f%% of the rows pick the same next token\n", profiled ? "profiled passes" : "verify passes", done, span, each, span * 1000.0 / each, 100.0 * agree / std::max(1, done * span));
            return each;
        };
        passes(false, 1 << 20);
        missedPairs = 0;
        missedKinds = 0;
        passCount = 0;
        serveSum = 0.0;
        waitSum = 0.0;
        expertSum = 0.0;
        passes(false, 1 << 20);
        printf("context room %d tokens in %zu stretched buffers, %zu mapped pieces, %d of %d recent slots per layer given back, %d resident experts per layer, %llu exchanges, %llu lifts\n", gpu.kvRoom, gpu.stretches.size(), gpu.stretchPages.size(), gpu.givenBack, gpu.recentPerLayer, gpu.residentExperts, (unsigned long long)gpu.exchangeCount, (unsigned long long)gpu.liftCount);
        printf("cpu per pass: %.2f ms serving layers, %.2f ms waiting for the gpu, %.2f ms on cpu experts\n", serveSum / std::max(1, passCount), waitSum / std::max(1, passCount), expertSum / std::max(1, passCount));
        if(driftRows > 0) printf("drift from one token per step: mean kl %.3e, worst %.3e over %d rows\n", driftSum / driftRows, driftWorst, driftRows);
        double perLayer = (double)(gpu.endLayer - gpu.firstLayer) * std::max(1, passCount);
        printf("missed per layer and pass: %.1f routes over %.1f experts\n", missedPairs / perLayer, missedKinds / perLayer);
        if(timed > 0){
            double f = 0, r = 0, w = 0, c = 0;
            for(int i = 0; i < depth; i++){
                f += frontSum[i];
                r += residentSum[i];
                w += waitLayer[i];
                c += cpuLayer[i];
            }
            printf("timeline per pass (us): gpu before telling the cpu %.0f, gpu after %.0f, gpu waiting %.0f, cpu busy %.0f over %d timed passes\n", f / timed, r / timed, w / timed, c / timed, timed);
            printf("gpu wait per layer in 50 us bins:");
            for(int b = 0; b < 8; b++) printf(" %d", waitBins[b]);
            printf("\ncpu busy per layer in 50 us bins:");
            for(int b = 0; b < 8; b++) printf(" %d", cpuBins[b]);
            printf("\ncpu rows: layers, cpu busy, gather, gpu wait, worst cpu (us)\n");
            for(int k = 0; k < 24; k++){
                if(rowCount[k] == 0) continue;
                printf("  %2d: %5d %7.0f %6.1f %7.0f %8.0f\n", k, rowCount[k], rowCpu[k] / rowCount[k], rowGather[k] / rowCount[k], rowWait[k] / rowCount[k], rowWorst[k]);
            }
            printf("layer: before after wait cpu (us)\n");
            for(int i = 0; i < depth; i++) printf("  %2d: %6.0f %6.0f %6.0f %6.0f\n", i + gpu.firstLayer, frontSum[i] / timed, residentSum[i] / timed, waitLayer[i] / timed, cpuLayer[i] / timed);
        }
        printf("graphs: %zu recorded, %zu could not be recorded%s%s\n", gpu.tokenGraphs.size(), gpu.graphless.size(), gpu.graphProblem.empty() ? "" : ", last problem: ", gpu.graphProblem.c_str());
        double wall = passes(true, 24);
        std::vector<std::pair<double, std::string>> ranked;
        double total = 0;
        for(auto& k : gpu.kernelMs){
            ranked.push_back({k.second / profiledCount, k.first});
            total += k.second / profiledCount;
        }
        std::sort(ranked.rbegin(), ranked.rend());
        printf("kernels per pass %.2f ms of %.2f ms wall\n", total, wall);
        for(size_t i = 0; i < ranked.size() && i < 24; i++) printf("  %-40s %7.3f ms  %5.1f calls\n", ranked[i].second.c_str(), ranked[i].first, (double)gpu.kernelCalls[ranked[i].second] / profiledCount);
        return 0;
    }
    auto step = [&](){
        if(window <= 1){
            logits = gpu.forward({next}, {(int)gpu.tokens.size()});
            next = bestOf(logits);
            return;
        }
        int at = (int)gpu.tokens.size();
        std::vector<int> ids(window, next), where(window);
        for(int i = 0; i < window; i++) where[i] = at + i;
        auto mark = std::chrono::steady_clock::now();
        if(settle) gpu.markRows(window);
        else gpu.saveState(1);
        markMs += millisSince(mark);
        mark = std::chrono::steady_clock::now();
        std::vector<float> rows = gpu.forwardAll(ids, where);
        verifyMs += millisSince(mark);
        mark = std::chrono::steady_clock::now();
        std::vector<int> kept(keep, next);
        if(keep < window) gpu.commitRows(kept);
        else if(settle) gpu.settleRows(ids);
        commitMs += millisSince(mark);
        size_t width = rows.size() / window;
        logits.assign(rows.begin() + (keep - 1) * width, rows.begin() + keep * width);
        next = bestOf(logits);
    };
    auto run = [&](const char* label, int count){
        auto clock = std::chrono::steady_clock::now();
        uint64_t routed = gpu.decodeRouted, resident = gpu.decodeResident;
        for(int i = 0; i < count; i++){
            step();
            trail.push_back(next);
        }
        double each = millisSince(clock) / count;
        printf("%s: %.2f ms per step, vram hit %.1f%%\n", label, each, gpu.decodeRouted > routed ? 100.0 * (gpu.decodeResident - resident) / (gpu.decodeRouted - routed) : 0.0);
        return each;
    };
    if(const char* tracePath = getenv("KERNELPROF_TRACE")){
        int steps = getenv("KERNELPROF_TRACE_STEPS") ? atoi(getenv("KERNELPROF_TRACE_STEPS")) : 400;
        std::string text;
        auto filling = std::chrono::steady_clock::now();
        while(gpu.warmer.joinable() && !gpu.warmDone && millisSince(filling) < 180000) Sleep(50);
        printf("waited %.1f s for the ram cache to fill\n", millisSince(filling) / 1000);
        if(const char* keep = getenv("KERNELPROF_WARM_KEEP")) gpu.warmKeep = keep[0] == '1' ? gpu.warmKeep : 0;
        PROCESS_MEMORY_COUNTERS memoryBefore = {};
        GetProcessMemoryInfo(GetCurrentProcess(), &memoryBefore, sizeof(memoryBefore));
        auto clock = std::chrono::steady_clock::now();
        auto part = clock;
        double spilled = 0.0;
        uint64_t lifts = gpu.liftCount;
        for(int i = 0; i < steps; i++){
            logits = gpu.forward({next}, {(int)gpu.tokens.size()});
            if(logits.empty()){
                printf("stopped at step %d: %s\n", i, gpu.problem.c_str());
                break;
            }
            text += tokenizer.decode({next});
            next = bestOf(logits);
            spilled += gpu.expertMs;
            if((i + 1) % 100 != 0) continue;
            printf("  steps %d to %d: %.2f ms per step, experts off the main card %.2f ms, %llu recent experts lifted\n", i - 99, i + 1, millisSince(part) / 100, spilled / 100, (unsigned long long)(gpu.liftCount - lifts));
            part = std::chrono::steady_clock::now();
            spilled = 0.0;
            lifts = gpu.liftCount;
        }
        PROCESS_MEMORY_COUNTERS memoryAfter = {};
        GetProcessMemoryInfo(GetCurrentProcess(), &memoryAfter, sizeof(memoryAfter));
        printf("ram cache: %zu experts listed, %.1f GB budget, %.1f GB kept under pressure, locking %s, %lu page faults during the answer, working set %.1f GB (peak %.1f)\n", gpu.warmOrder.size(), gpu.warmBudget / 1073741824.0, gpu.warmKeep / 1073741824.0, gpu.locking ? "on" : "off", (unsigned long)(memoryAfter.PageFaultCount - memoryBefore.PageFaultCount), memoryAfter.WorkingSetSize / 1073741824.0, memoryAfter.PeakWorkingSetSize / 1073741824.0);
        if(!gpu.helper.problem.empty()) printf("second gpu problem: %s\n", gpu.helper.problem.c_str());
        printf("%d steps, %.2f ms per step, %d recent slots per layer, %llu experts lifted, vram hit %.1f%%, second gpu %.2f rows per step, %llu cpu experts outside pinned memory\n", steps, millisSince(clock) / steps, gpu.recentPerLayer, (unsigned long long)gpu.liftCount, gpu.decodeRouted > 0 ? 100.0 * gpu.decodeResident / gpu.decodeRouted : 0.0, (double)gpu.helper.rows / steps, (unsigned long long)gpu.coldExperts);
        std::ofstream(tracePath, std::ios::binary) << text;
        while(gpu.liftPending > 0){
            Sleep(20);
            gpu.refresh();
        }
        cudaLink().sync();
        size_t checked = 0, wrong = 0, drift = 0;
        std::vector<uint8_t> held;
        std::vector<int> table(gpu.layers[gpu.firstLayer].slot.size());
        for(int l = gpu.firstLayer; l < gpu.endLayer; l++){
            GpuLayer& L = gpu.layers[l];
            cudaLink().copyDown(table.data(), L.slots, table.size() * 4);
            for(size_t e = 0; e < table.size(); e++) drift += table[e] != L.slot[e];
            const Matrix* parts[3] = {&L.expertGate, &L.expertUp, &L.expertDown};
            for(int j = 0; j < gpu.recentPerLayer; j++){
                int e = gpu.recentHolds[(size_t)l * gpu.recentPerLayer + j];
                if(e < 0) continue;
                checked++;
                bool same = L.slot[e] == gpu.fixedExperts + j;
                for(int p = 0; p < 3 && same; p++){
                    uint64_t span = (uint64_t)parts[p]->rows * parts[p]->stride;
                    held.resize((size_t)span);
                    same = cudaLink().copyDown(held.data(), L.resident[p] + (uint64_t)(gpu.fixedExperts + j) * span, (size_t)span) == 0 && memcmp(held.data(), parts[p]->data + (uint64_t)e * span, (size_t)span) == 0;
                }
                wrong += !same;
            }
        }
        printf("recent tier check: %zu slots compared against the model file, %zu differ, %zu slot table entries out of step\n", checked, wrong, drift);
        size_t pinnedChecked = 0, pinnedWrong = 0, shared = 0, adrift = 0, homeless = 0;
        for(int l = gpu.firstLayer; l < gpu.endLayer; l++){
            GpuLayer& L = gpu.layers[l];
            if(!L.warmHost[0]) continue;
            cudaLink().copyDown(table.data(), L.warmSlots, expertCount * 4);
            const Matrix* parts[3] = {&L.expertGate, &L.expertUp, &L.expertDown};
            std::vector<char> taken(expertCount, 0);
            for(int room : gpu.ramFree[l]) taken[room] = 1;
            for(int e = 0; e < expertCount; e++){
                adrift += table[e] != L.warm[e];
                homeless += L.slot[e] < 0 && L.warm[e] < 0 && (L.help.empty() || L.help[e] < 0);
                if(L.warm[e] < 0) continue;
                shared += taken[L.warm[e]];
                taken[L.warm[e]] = 1;
                pinnedChecked++;
                bool same = true;
                for(int p = 0; p < 3 && same; p++){
                    uint64_t span = (uint64_t)parts[p]->rows * parts[p]->stride;
                    same = memcmp(L.warmHost[p] + (uint64_t)L.warm[e] * span, parts[p]->data + (uint64_t)e * span, (size_t)span) == 0;
                }
                pinnedWrong += !same;
            }
        }
        printf("pinned memory check: %zu experts compared against the model file, %zu differ, %zu shared places, %zu table entries out of step, %zu experts only on disk\n", pinnedChecked, pinnedWrong, shared, adrift, homeless);
        printf("trace done\n");
        return 0;
    }
    run("warm up with graphs", 48);
    double launchBefore = gpu.helperLaunchMs, cpuBefore = gpu.helperCpuMs, waitBefore = gpu.helperWaitMs;
    uint64_t callsBefore = gpu.helperCalls, rowsBefore = gpu.helper.rows;
    double partBefore[4] = {gpu.helper.partMs[0], gpu.helper.partMs[1], gpu.helper.partMs[2], gpu.helper.partMs[3]};
    markMs = commitMs = verifyMs = 0.0;
    run("with graphs", 32);
    if(window > 1) printf("per step: mark %.3f ms, verify %.3f ms, commit %.3f ms, keeping %d of %d rows\n", markMs / 32, verifyMs / 32, commitMs / 32, keep, window);
    printf("recent slots per layer %d, experts lifted %llu\n", gpu.recentPerLayer, (unsigned long long)gpu.liftCount);
    printf("second gpu per step: %.2f calls, %.2f rows, launch %.3f ms, cpu experts beside it %.3f ms, wait %.3f ms\n", (gpu.helperCalls - callsBefore) / 32.0, (gpu.helper.rows - rowsBefore) / 32.0, (gpu.helperLaunchMs - launchBefore) / 32, (gpu.helperCpuMs - cpuBefore) / 32, (gpu.helperWaitMs - waitBefore) / 32);
    printf("second gpu kernels per step: gate %.3f ms, up %.3f ms, swiglu %.3f ms, down %.3f ms\n", (gpu.helper.partMs[0] - partBefore[0]) / 32, (gpu.helper.partMs[1] - partBefore[1]) / 32, (gpu.helper.partMs[2] - partBefore[2]) / 32, (gpu.helper.partMs[3] - partBefore[3]) / 32);
    if(const char* dump = getenv("KERNELPROF_DUMP")){
        std::ofstream saved(dump, std::ios::binary);
        int count = (int)trail.size();
        saved.write((const char*)&count, 4);
        saved.write((const char*)trail.data(), count * 4);
        saved.write((const char*)firstLogits.data(), firstLogits.size() * 4);
    }
    std::string switchName = argc > 6 ? argv[6] : "-";
    bool* flag = switchName == "attention"
        ? &gpu.splitAttention
        : switchName == "mix"
            ? &gpu.fusedMix
            : switchName == "staged"
                ? &gpu.stagedSets
                : switchName == "residual"
                    ? &gpu.residual
                    : nullptr;
    if(flag){
        printf("state saved for the comparison: %s\n", gpu.saveState(2) ? "yes" : "no");
        int start = next;
        auto fresh = [&]{
            for(auto& g : gpu.tokenGraphs){
                if(g.second) cudaQueue().execDestroy(g.second);
            }
            gpu.tokenGraphs.clear();
        };
        std::vector<std::vector<float>> ends;
        std::vector<std::vector<int>> picks;
        for(int pass = 0; pass < 4; pass++){
            gpu.restoreState(2);
            next = start;
            *flag = pass % 2 == 1;
            fresh();
            std::vector<int> chosen;
            auto clock = std::chrono::steady_clock::now();
            for(int i = 0; i < 32; i++){
                step();
                chosen.push_back(next);
            }
            printf("%s %s: %.2f ms per step\n", switchName.c_str(), *flag ? "new" : "old", millisSince(clock) / 32);
            ends.push_back(logits);
            picks.push_back(chosen);
        }
        double gap = 0;
        for(size_t i = 0; i < ends[0].size() && i < ends[1].size(); i++) gap = std::max(gap, (double)fabsf(ends[0][i] - ends[1][i]));
        int same = 0;
        while(same < 32 && picks[0][same] == picks[1][same]) same++;
        printf("old and new agree on the first %d of 32 greedy tokens, largest logit gap at the end %.4g\n", same, gap);
        *flag = true;
        fresh();
    }
    gpu.useGraphs = false;
    gpu.relayAlways = true;
    run("without graphs", 32);
    gpu.profile = true;
    gpu.profileShapes = argc > 5 && std::string(argv[5]) == "shapes";
    gpu.kernelMs.clear();
    gpu.kernelCalls.clear();
    int profiled = 32;
    double wall = run("profiled", profiled);
    std::map<std::string, double> byPhase;
    std::vector<std::pair<double, std::string>> ranked;
    double total = 0;
    for(auto& k : gpu.kernelMs){
        ranked.push_back({k.second / profiled, k.first});
        total += k.second / profiled;
        byPhase[k.first.substr(0, k.first.find(' '))] += k.second / profiled;
    }
    std::sort(ranked.rbegin(), ranked.rend());
    printf("kernels per step %.2f ms of %.2f ms wall\n", total, wall);
    for(auto& p : byPhase) printf("  phase %-10s %7.3f ms\n", p.first.c_str(), p.second);
    for(size_t i = 0; i < ranked.size() && i < 45; i++) printf("  %-40s %7.3f ms  %5.1f calls  %6.2f us each\n", ranked[i].second.c_str(), ranked[i].first, (double)gpu.kernelCalls[ranked[i].second] / profiled, 1000.0 * ranked[i].first * profiled / std::max(1, gpu.kernelCalls[ranked[i].second]));
    return 0;
}
