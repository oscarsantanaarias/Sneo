#pragma once

inline bool GpuModel::loadGemma(const std::string& path, int threads) {
    gemma = true;
    dense = true;
    usePle = false;
    streamPrefill = false;
    pinWarm = false;
    foresee = false;
    residual = false;
    firstLayer = 0;
    endLayer = 0;
    layers.clear();
    GemmaShape& s = gemmaShape;
    GemmaBuffers& b = gemmaBuffers;
    std::string a = "gemma4.";
    auto real = [&](const std::string& key, double fallback) {
        auto found = file.meta.find(a + key);
        return found == file.meta.end() ? fallback : found->second.real;
    };
    auto list = [&](const std::string& key) {
        auto found = file.meta.find(a + key);
        return found == file.meta.end() ? std::vector<int64_t>() : found->second.wholes;
    };
    s.layers = (int)metaWhole(file, a + "block_count", 0);
    s.hidden = (int)metaWhole(file, a + "embedding_length", 0);
    s.heads = (int)metaWhole(file, a + "attention.head_count", 0);
    s.window = (int)metaWhole(file, a + "attention.sliding_window", 1024);
    s.feedForward = (int)metaWhole(file, a + "feed_forward_length", 0);
    s.experts = (int)metaWhole(file, a + "expert_count", 0);
    s.expertsUsed = (int)metaWhole(file, a + "expert_used_count", 0);
    s.expertWidth = (int)metaWhole(file, a + "expert_feed_forward_length", 0);
    s.eps = (float)real("attention.layer_norm_rms_epsilon", 1e-6);
    s.softcap = (float)real("final_logit_softcapping", 0.0);
    s.ropeBase = (float)real("rope.freq_base", 1e6);
    s.ropeBaseSliding = (float)real("rope.freq_base_swa", 1e4);
    int fullSize = (int)metaWhole(file, a + "attention.key_length", 512);
    int slidingSize = (int)metaWhole(file, a + "attention.key_length_swa", 256);
    std::vector<int64_t> kvList = list("attention.head_count_kv");
    std::vector<int64_t> pattern = list("attention.sliding_window_pattern");
    if(kvList.empty()) kvList.assign(s.layers, (int64_t)metaWhole(file, a + "attention.head_count_kv", 0));
    if(metaWhole(file, a + "attention.shared_kv_layers", 0) != 0 || metaWhole(file, a + "embedding_length_per_layer_input", 0) != 0){
        problem = "this gemma 4 size shares key value layers or uses per layer embeddings, which the engine does not run yet";
        return false;
    }
    if(s.layers <= 0 || s.hidden <= 0 || s.heads <= 0 || (int)kvList.size() < s.layers || (int)pattern.size() < s.layers || s.hidden % 32 != 0 || s.feedForward % 32 != 0){
        problem = "unexpected gemma 4 shape";
        return false;
    }
    if(s.experts > 0 && (s.expertsUsed <= 0 || s.expertsUsed > 16 || s.expertWidth % 32 != 0)){
        problem = "unexpected gemma 4 expert shape";
        return false;
    }
    if(!useCuda()){
        problem = "no cuda device";
        return false;
    }
    CudaQueue& queue = cudaQueue();
    if(!queue.ready || !cudaLink().hostAlloc || !cudaLink().memset){
        problem = "cuda driver entry points missing";
        return false;
    }
    size_t startFree = 0, startTotal = 0;
    cudaLink().info(&startFree, &startTotal);
    freeAtStart = startFree;
    legacy = legacy || cudaLink().major < 8;

    std::string sizes = "#define HIDDEN " + std::to_string(hidden)
        + "\n#define HC " + std::to_string(hcCount)
        + "\n#define HCW " + std::to_string(hcWidth)
        + "\n#define HCR " + std::to_string(std::max(hcRank, 32))
        + "\n#define KUSED " + std::to_string(expertsUsed)
        + "\n#define EXPERTS 32"
        + "\n#define CONVW " + std::to_string(convWidth)
        + "\n#define INNER " + std::to_string(innerWidth)
        + "\n#define VHEADS " + std::to_string(valueHeads)
        + "\n#define KHEADS " + std::to_string(keyHeads)
        + "\n#define QHEADS " + std::to_string(queryHeads)
        + "\n#define KVHEADS " + std::to_string(kvHeads)
        + "\n#define KVW " + std::to_string(kvHeads * headWidth)
        + "\n#define IHEADS 0"
        + "\n#define EXACT_MIX 0"
        + "\n#define SILU_NORM 1\n";
    if(!buildModule(sizes + quantKernelSource + modelKernelSource + gemmaKernelSource, module)){
        problem = "kernel build failed: " + module.log;
        return false;
    }

    for(const std::string& p : file.paths){
        MappedFile m;
        m.file = CreateFileA(p.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
        if(m.file != INVALID_HANDLE_VALUE) m.mapping = CreateFileMappingA(m.file, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if(m.mapping) m.base = (const uint8_t*)MapViewOfFile(m.mapping, FILE_MAP_READ, 0, 0, 0);
        maps.push_back(m);
        if(!m.base){
            problem = "cannot map " + p;
            return false;
        }
    }

    auto grab = [&](uint64_t bytes) -> uint64_t {
        uint64_t ptr = 0;
        int code = cudaLink().alloc(&ptr, (size_t)std::max<uint64_t>(bytes, 256));
        if(code != 0){
            if(problem.empty()) problem = code == 2
                ? "out of video memory allocating " + std::to_string(bytes >> 20) + " MB"
                : "cuda error " + std::to_string(code) + " allocating " + std::to_string(bytes >> 20) + " MB";
            return 0;
        }
        allocations.push_back(ptr);
        return ptr;
    };
    auto shared = [&](uint64_t bytes, uint64_t& alias) -> void* {
        void* p = nullptr;
        bytes = std::max<uint64_t>(bytes, 256);
        if(cudaLink().hostAlloc(&p, (size_t)bytes, 3) != 0 || queue.hostDevice(&alias, p, 0) != 0){
            if(problem.empty()) problem = "cannot map " + std::to_string(bytes >> 20) + " MB of host memory for the gpu";
            return nullptr;
        }
        pinned.push_back(p);
        return p;
    };

    std::vector<std::pair<const TensorInfo*, uint64_t*>> wanted;
    auto matrix = [&](const std::string& name, GpuMatrix& out, bool optional = false) {
        const TensorInfo* t = findTensor(file, name);
        if(!t && optional) return;
        if(!t || !knownType(t->type)){
            if(problem.empty()) problem = t
                ? name + " is " + typeInfo(t->type).name + ", a tensor type the engine cannot read yet"
                : "missing tensor " + name;
            return;
        }
        out.type = t->type;
        out.cols = (int)t->dims[0];
        out.rows = (int)t->dims[1];
        out.stride = rowBytes(t->type, t->dims[0]);
        wanted.push_back({t, &out.data});
    };
    auto floats = [&](const std::string& name, uint64_t& out, bool optional = false) {
        const TensorInfo* t = findTensor(file, name);
        if(!t && optional) return;
        if(!t || t->type != 0){
            if(problem.empty()) problem = name + " is missing or not F32";
            return;
        }
        wanted.push_back({t, &out});
    };

    const TensorInfo* table = findTensor(file, "token_embd.weight");
    if(!table || !knownType(table->type) || (int)table->dims[0] != s.hidden){
        problem = "missing or unsupported token_embd.weight";
        return false;
    }
    embed.data = maps[table->file].base + table->offset;
    embed.type = table->type;
    embed.cols = (int)table->dims[0];
    embed.rows = (int)table->dims[1];
    embed.stride = rowBytes(table->type, table->dims[0]);
    s.vocab = embed.rows;
    matrix(findTensor(file, "output.weight") ? "output.weight" : "token_embd.weight", head);
    floats("output_norm.weight", b.outNorm);
    uint64_t ropeFactors = 0;
    floats("rope_freqs.weight", ropeFactors, true);

    chunk = std::max(64, gemmaBlock);
    s.ring = s.window + 2 * chunk;
    gemmaLayers.assign(s.layers, GemmaLayer());
    for(int l = 0; l < s.layers; l++){
        GemmaLayer& L = gemmaLayers[l];
        std::string p = "blk." + std::to_string(l) + ".";
        L.sliding = pattern[l] != 0;
        L.kvHeads = (int)kvList[l];
        L.headSize = L.sliding ? slidingSize : fullSize;
        bool fits = L.kvHeads > 0 && (L.sliding ? L.headSize == 256 && L.kvHeads * 2 == s.heads : L.headSize == 512 && L.kvHeads * 8 == s.heads);
        if(!fits){
            problem = "gemma 4 layer " + std::to_string(l) + " has " + std::to_string(s.heads) + " query heads over " + std::to_string(L.kvHeads) + " key value heads of " + std::to_string(L.headSize) + ", the engine runs 2 per head at 256 and 8 per head at 512";
            return false;
        }
        floats(p + "attn_norm.weight", L.attnNorm);
        matrix(p + "attn_q.weight", L.query);
        matrix(p + "attn_k.weight", L.key);
        matrix(p + "attn_v.weight", L.value, true);
        matrix(p + "attn_output.weight", L.out);
        floats(p + "attn_q_norm.weight", L.queryNorm);
        floats(p + "attn_k_norm.weight", L.keyNorm);
        floats(p + "post_attention_norm.weight", L.postAttnNorm);
        floats(p + "ffn_norm.weight", L.ffnNorm);
        matrix(p + "ffn_gate.weight", L.gate);
        matrix(p + "ffn_up.weight", L.up);
        matrix(p + "ffn_down.weight", L.down);
        floats(p + "post_ffw_norm.weight", L.postFfnNorm);
        floats(p + "layer_output_scale.weight", L.outScale, true);
        L.mixture = findTensor(file, p + "ffn_gate_inp.weight") != nullptr;
        if(L.mixture){
            floats(p + "ffn_gate_inp.weight", L.router);
            floats(p + "ffn_gate_inp.scale", L.routerScale);
            floats(p + "pre_ffw_norm_2.weight", L.preMoeNorm);
            floats(p + "post_ffw_norm_1.weight", L.postMlpNorm);
            floats(p + "post_ffw_norm_2.weight", L.postMoeNorm);
            floats(p + "ffn_down_exps.scale", L.downScale, true);
            if(const TensorInfo* scaleInfo = findTensor(file, p + "ffn_down_exps.scale")){
                if(scaleInfo->type == 0){
                    const float* host = (const float*)(maps[scaleInfo->file].base + scaleInfo->offset);
                    L.downScaleHost.assign(host, host + scaleInfo->dims[0]);
                }
            }
            const TensorInfo* both = findTensor(file, p + "ffn_gate_up_exps.weight");
            const TensorInfo* down = findTensor(file, p + "ffn_down_exps.weight");
            if(!both || !down || !knownType(both->type) || !knownType(down->type) || s.experts <= 0 || (int)both->dims[2] != s.experts || (int)down->dims[2] != s.experts){
                problem = "gemma 4 layer " + std::to_string(l) + " has no fused gate and up experts the engine can run";
                return false;
            }
            L.expertGateUp = {0, both->type, (int)both->dims[1], (int)both->dims[0], rowBytes(both->type, both->dims[0])};
            L.expertDown = {0, down->type, (int)down->dims[1], (int)down->dims[0], rowBytes(down->type, down->dims[0])};
            L.expertGateUpBytes = (uint64_t)L.expertGateUp.rows * L.expertGateUp.stride;
            L.expertDownBytes = (uint64_t)L.expertDown.rows * L.expertDown.stride;
            if(L.expertGateUp.rows != 2 * s.expertWidth || L.expertGateUp.cols != s.hidden || L.expertDown.rows != s.hidden || L.expertDown.cols != s.expertWidth){
                problem = "unexpected gemma 4 expert tensor shapes";
                return false;
            }
            wanted.push_back({both, &L.expertGateUp.data});
            wanted.push_back({down, &L.expertDown.data});
        }
        if(!L.sliding && !findTensor(file, "rope_freqs.weight")){
            problem = "gemma 4 needs rope_freqs.weight for the full attention layers";
            return false;
        }
        s.widest = std::max(s.widest, L.query.rows);
        s.widestKv = std::max(s.widestKv, L.key.rows);
        if(!problem.empty()) return false;
        if(L.query.rows != s.heads * L.headSize || L.key.rows != L.kvHeads * L.headSize || (L.value.rows != 0 && L.value.rows != L.key.rows) || L.out.cols != L.query.rows || L.gate.rows != s.feedForward || L.down.cols != s.feedForward){
            problem = "unexpected gemma 4 tensor shapes in layer " + std::to_string(l);
            return false;
        }
    }
    if(!problem.empty()) return false;

    std::vector<uint8_t> blob(33936);
    memcpy(blob.data(), grid2xxs, 2048);
    memcpy(blob.data() + 2048, grid2xs, 4096);
    memcpy(blob.data() + 6144, grid2s, 8192);
    memcpy(blob.data() + 14336, grid3xxs, 1024);
    memcpy(blob.data() + 15360, grid3s, 2048);
    memcpy(blob.data() + 17408, signTable, 128);
    memcpy(blob.data() + 17536, nonLinear4, 16);
    memcpy(blob.data() + 17552, grid1s, 16384);
    uint64_t tablesOnGpu = grab(blob.size());
    if(!tablesOnGpu || cudaLink().copyUp(tablesOnGpu, blob.data(), blob.size()) != 0) return false;
    tables = {tablesOnGpu, tablesOnGpu + 2048, tablesOnGpu + 6144, tablesOnGpu + 14336, tablesOnGpu + 15360, tablesOnGpu + 17408, tablesOnGpu + 17536, tablesOnGpu + 17552};

    for(auto& w : wanted) denseBytes += (w.first->bytes + 255) / 256 * 256;
    // el ayudante que adivina tokens va antes de medir el contexto para que su memoria no se la coma la cache
    ModelFile draftFile;
    bool withDraft = false;
    if(!draftPath.empty() && std::filesystem::exists(draftPath)){
        std::string why;
        withDraft = openModel(draftPath, draftFile, why) && draftFile.arch == "gemma4-assistant";
        if(!withDraft) draftProblem = why.empty() ? "no gemma 4 assistant beside this model" : why;
    }
    b.everyCap = std::min(std::max(relayLimit, verifyLimit), 64);
    uint64_t draftReserve = withDraft ? draftFile.totalBytes + (64ULL << 20) : 0;
    uint64_t everyBytes = (uint64_t)b.everyCap * ((uint64_t)s.vocab + (uint64_t)s.hidden) * 4;
    uint64_t C = 0, bufferBytes = 0, slidingBytes = 0, perToken = 0, fixedBytes = 0;
    uint64_t d = (uint64_t)s.hidden;
    uint64_t E = (uint64_t)std::max(1, s.expertsUsed);
    uint64_t W = (uint64_t)std::max(32, s.expertWidth);
    uint64_t widestInput = std::max<uint64_t>({d, (uint64_t)s.widest, (uint64_t)s.feedForward});
    uint64_t scratchBytes = 4ULL * s.splitLimit * s.heads * (512 + 2) * 4;
    uint64_t margin = 900ULL << 20;
    for(;;){
        s.ring = s.window + 2 * chunk;
        C = (uint64_t)chunk;
        bufferBytes = C * d * 4 * 9 + C * (uint64_t)s.widest * 4 * 2 + C * (uint64_t)s.widestKv * 4 * 2 + C * (uint64_t)s.feedForward * 4 * 3
            + (s.experts > 0 ? C * E * (8 + 2 * W * 4 + W * 4 + d * 4) + C * d * 2 + C * E * W * 2 : 0)
            + scratchBytes + (uint64_t)s.vocab * 4 + widestInput * C * 3;
        slidingBytes = 0;
        perToken = 0;
        for(const GemmaLayer& L : gemmaLayers){
            uint64_t width = (uint64_t)L.kvHeads * L.headSize * 4;
            if(L.sliding) slidingBytes += (uint64_t)s.ring * width;
            else perToken += width;
        }
        fixedBytes = denseBytes + bufferBytes + slidingBytes + margin + draftReserve + everyBytes;
        bool roomy = fixedBytes < startFree && (perToken == 0 || (startFree - fixedBytes) / perToken >= (uint64_t)contextLimit);
        if(roomy || chunk <= 256) break;
        chunk /= 2;
    }
    if(fixedBytes >= startFree){
        problem = "gemma 4 needs " + std::to_string((fixedBytes >> 20) + 1) + " MB of video memory before any context and " + std::to_string(startFree >> 20) + " MB are free";
        return false;
    }
    uint64_t room = perToken > 0 ? (startFree - fixedBytes) / perToken : (uint64_t)contextLimit;
    if((uint64_t)contextLimit > room){
        int fitted = (int)(room / 256 * 256);
        if(fitted < 4096){
            problem = "gemma 4 fits only " + std::to_string(fitted) + " tokens of context in the free video memory";
            return false;
        }
        contextLimit = fitted;
    }
    contextLimit = std::max(contextLimit, s.ring);

    uint64_t arena = grab(denseBytes + 256);
    if(!arena) return false;
    uint64_t used = 0;
    std::vector<std::array<uint64_t, 4>> reads;
    for(auto& w : wanted){
        *w.second = arena + used;
        reads.push_back({w.first->file, w.first->offset, w.first->bytes, arena + used});
        used += (w.first->bytes + 255) / 256 * 256;
    }
    if(!fetchAll(reads)) return false;
    for(GemmaLayer& L : gemmaLayers){
        if(!L.sliding) L.ropeFactors = ropeFactors;
    }
    reads.clear();
    for(GemmaLayer& L : gemmaLayers){
        L.slots = L.sliding ? s.ring : contextLimit;
        uint64_t bytes = (uint64_t)L.slots * L.kvHeads * L.headSize * 2;
        L.keys = grab(bytes);
        L.values = grab(bytes);
        cacheBytes += 2 * bytes;
    }

    b.x = grab(C * d * 4);
    b.xn = grab(C * d * 4);
    b.attn = grab(C * d * 4);
    b.a = grab(C * d * 4);
    b.mlp = grab(C * d * 4);
    b.moeIn = grab(C * d * 4);
    b.routerIn = grab(C * d * 4);
    b.moe = grab(C * d * 4);
    b.pair = grab(C * d * 4);
    b.q = grab(C * s.widest * 4);
    b.o = grab(C * s.widest * 4);
    b.k = grab(C * s.widestKv * 4);
    b.v = grab(C * s.widestKv * 4);
    b.gate = grab(C * s.feedForward * 4);
    b.up = grab(C * s.feedForward * 4);
    b.hid = grab(C * s.feedForward * 4);
    b.positions = grab(C * 4);
    b.scratch = grab(scratchBytes);
    b.last = grab(d * 4);
    b.vocab = grab((uint64_t)s.vocab * 4);
    b.normed = grab((uint64_t)b.everyCap * d * 4);
    b.every = grab((uint64_t)b.everyCap * s.vocab * 4);
    if(s.experts > 0){
        b.ids = grab(C * E * 4);
        b.weights = grab(C * E * 4);
        b.both = grab(C * E * 2 * W * 4);
        b.expertHid = grab(C * E * W * 4);
        b.expertOut = grab(C * E * d * 4);
        for(PackedInput* packed : {&b.expertIn, &b.expertMid}){
            int capacity = packed == &b.expertIn ? (int)(C * d) : (int)(C * E * W);
            packed->values = grab(capacity);
            packed->scales = grab(capacity / 8);
            packed->sums = grab(capacity / 8);
            packed->capacity = capacity;
        }
    }
    for(PackedInput* packed : {&packs[0], &fines[0]}){
        int capacity = (int)(C * widestInput);
        if(cudaLink().alloc(&packed->values, capacity) != 0 || cudaLink().alloc(&packed->scales, capacity / 8) != 0 || cudaLink().alloc(&packed->sums, capacity / 8) != 0){
            problem = "out of video memory for packed inputs";
            return false;
        }
        packed->capacity = capacity;
    }
    b.hostEmbed = (float*)shared(C * d * 4, b.mappedEmbed);
    b.hostLogits = (float*)shared((uint64_t)s.vocab * 4, b.mappedLogits);
    b.hostEvery = (float*)shared((uint64_t)b.everyCap * s.vocab * 4, b.mappedEvery);
    if(s.experts > 0){
        b.routeCap = (int)(C * E);
        b.hostRoutes = (int*)shared(2ULL * b.routeCap * 4, b.mappedRoutes);
        b.hostPlan = (int*)shared((3ULL * b.routeCap + 3ULL * s.experts) * 4, b.mappedPlan);
        b.plan = grab((3ULL * b.routeCap + 3ULL * s.experts) * 4);
        if(!b.hostRoutes || !b.hostPlan) problem = "cannot pin host memory for the gemma 4 expert lists";
    }
    void* hostPos = nullptr;
    if(cudaLink().hostAlloc(&hostPos, (size_t)(C * 4), 0) == 0){
        pinned.push_back(hostPos);
        b.hostPositions = (int*)hostPos;
    }
    if(!problem.empty() || !b.hostEmbed || !b.hostLogits || !b.hostEvery || !b.hostPositions){
        if(problem.empty()) problem = "cannot pin host memory for gemma 4";
        return false;
    }
    if(queue.streamCreate(&stream, 0) != 0){
        problem = "cannot create a cuda stream";
        return false;
    }
    if(withDraft && !loadGemmaDraft(draftPath, draftFile)) gemmaDraft.ready = false;
    size_t nowFree = 0, nowTotal = 0;
    cudaLink().info(&nowFree, &nowTotal);
    vramHeld = startFree > nowFree ? startFree - nowFree : 0;
    residentBytes = 0;
    for(const TensorInfo& t : file.tensors) residentBytes += t.expert ? t.bytes : 0;
    residentExperts = s.experts;
    pool.start(std::max(0, threads - 1));
    ready = true;
    reset();
    return true;
}

inline void GpuModel::gemmaPack(uint64_t input, int count, PackedInput& packed) {
    go("quantizeInput", (unsigned int)((count / 32 + 7) / 8), 1, 32, 8, input, count, packed.values, packed.scales, packed.sums);
}

inline void GpuModel::gemmaExpertRows(const GpuMatrix& m, uint64_t expertBytes, int perToken, uint64_t input, PackedInput& packed, uint64_t output, int slots) {
    bool typed = m.type == 12 || m.type == 13 || m.type == 14 || m.type == 20 || m.type == 21 || m.type == 23 || m.type == 42 || m.type == 7 || m.type == 8;
    if(typed && m.cols % 32 == 0 && !legacy){
        std::string name = "gemmaExperts" + std::to_string(m.type);
        go(name.c_str(), (unsigned int)((m.rows + 7) / 8), (unsigned int)slots, 256, 1, m.data, m.stride, expertBytes, gemmaBuffers.ids, perToken, packed.values, packed.scales, packed.sums, output, m.rows, m.cols, tables);
        return;
    }
    go("gemmaExperts", (unsigned int)((m.rows + 7) / 8), (unsigned int)slots, 32, 8, (int)m.type, m.data, m.stride, expertBytes, gemmaBuffers.ids, perToken, input, packed.values, packed.scales, packed.sums, output, m.rows, m.cols, tables);
}

inline int GpuModel::gemmaSplits(const GemmaLayer& L, int n, int top) {
    GemmaShape& s = gemmaShape;
    int reach = L.sliding ? std::min(top + 1, s.window + n) : top + 1;
    // con pocas filas cada bloque toma unos dos tramos de claves (128 en las de ventana y 32 en las completas) mientras quepan en la memoria de partes
    int tile = L.headSize == 256 ? 128 : 32;
    int splits = n <= 16 ? std::max(1, std::min({s.splitLimit, (reach + 2 * tile - 1) / (2 * tile), 4 * s.splitLimit / n})) : 1;
    if(!gemmaRounded) return splits;
    int whole = 1;
    while(whole * 2 <= splits) whole *= 2;
    return whole;
}

inline void GpuModel::gemmaRun(int l, int n, int top) {
    GemmaShape& s = gemmaShape;
    GemmaBuffers& b = gemmaBuffers;
    GemmaLayer& L = gemmaLayers[l];
    int d = s.hidden;
    uint64_t none = 0;
    PackedInput& packed = packs[0];

    go("gemmaRmsPack", (unsigned int)n, 1, 256, 1, b.x, L.attnNorm, b.xn, d, s.eps, packed.values, packed.scales, packed.sums);
    if(L.value.data) projectMany({{&L.query, b.q}, {&L.key, b.k}, {&L.value, b.v}}, b.xn, 0, n);
    else projectMany({{&L.query, b.q}, {&L.key, b.k}}, b.xn, 0, n);

    int ring = L.sliding ? s.ring : 0;
    float base = L.sliding ? s.ropeBaseSliding : s.ropeBase;
    uint64_t source = L.value.data ? b.v : b.k;
    go("gemmaHeads", (unsigned int)(s.heads + L.kvHeads), (unsigned int)n, (unsigned int)(L.headSize / 2), 1, b.q, b.k, source, L.queryNorm, L.keyNorm, L.keys, L.values, b.positions, s.heads, L.kvHeads, L.headSize, L.query.rows, L.key.rows, ring, base, L.ropeFactors, s.eps);

    int window = L.sliding ? s.window : 0;
    int tileTokens = L.headSize == 256 ? 8 : 2;
    int tiles = (n + tileTokens - 1) / tileTokens;
    int splits = gemmaSplits(L, n, top);
    // prompt con nucleos tensoriales: 64 filas por bloque, 32 tokens en las de ventana y 8 en las completas
    if(n > 16 && gemmaFlash && !legacy && !amdCards && cudaLink().funcAttribute){
        bool narrow = L.headSize == 256;
        unsigned int bytes = narrow ? 67968 : 100000;
        int per = narrow ? 32 : 8;
        goShared(bytes, narrow ? "gemmaFlash256" : "gemmaFlash512", (unsigned int)L.kvHeads, (unsigned int)((n + per - 1) / per), narrow ? 128 : 256, 1, b.q, L.keys, L.values, b.positions, b.o, n, s.heads, L.kvHeads, ring, window);
        splits = 1;
    }
    else if(n <= 16 && gemmaFew) go(L.headSize == 256 ? "gemmaFew256" : "gemmaFew512", (unsigned int)L.kvHeads, (unsigned int)(n * splits), 256, 1, b.q, L.keys, L.values, b.positions, b.o, b.scratch, s.heads, L.kvHeads, ring, window, splits);
    else go(L.headSize == 256 ? "gemmaAttention256" : "gemmaAttention512", (unsigned int)L.kvHeads, (unsigned int)(tiles * splits), 256, 1, b.q, L.keys, L.values, b.positions, b.o, b.scratch, n, s.heads, L.kvHeads, ring, window, splits);
    if(splits > 1) go("gemmaMerge", (unsigned int)s.heads, (unsigned int)n, 256, 1, b.scratch, b.o, s.heads, L.headSize, splits);
    pack(b.o, n * L.query.rows, 0);
    project(L.out, b.o, 0, b.attn, n);
    go("gemmaAddRmsPack", (unsigned int)n, 1, 256, 1, b.attn, L.postAttnNorm, b.x, b.a, L.ffnNorm, b.xn, d, s.eps, packed.values, packed.scales, packed.sums);
    projectMany({{&L.gate, b.gate}, {&L.up, b.up}}, b.xn, 0, n);
    int ff = s.feedForward;
    go("gemmaGeluPack", (unsigned int)((n * ff / 32 + 7) / 8), 1, 32, 8, b.gate, b.up, b.hid, n * ff, packed.values, packed.scales, packed.sums);
    project(L.down, b.hid, 0, b.mlp, n);

    if(!L.mixture){
        go("gemmaRmsAdd", (unsigned int)n, 1, 256, 1, b.mlp, L.postFfnNorm, b.a, b.x, d, s.eps, L.outScale);
        return;
    }
    int used = s.expertsUsed;
    int width = s.expertWidth;
    int slots = n * used;
    float shrink = 1.0f / sqrtf((float)d);
    go("gemmaMoeIn", (unsigned int)n, 1, 256, 1, b.a, L.preMoeNorm, L.routerScale, shrink, d, s.eps, b.moeIn, b.routerIn, b.expertIn.values, b.expertIn.scales, b.expertIn.sums);
    go("gemmaRouteLogits", (unsigned int)((s.experts + 7) / 8), (unsigned int)n, 256, 1, b.routerIn, L.router, b.both, d, s.experts);
    go("gemmaRoutePick", (unsigned int)n, 1, 32, 1, b.both, b.ids, b.weights, s.experts, used);
    uint64_t rankOnGpu = 0;
    unsigned int mixBytes = (unsigned int)(8 * d);
    if(gemmaGroups && n >= 16 && b.plan && slots <= b.routeCap){
        CudaQueue& queue = cudaQueue();
        go("copyWords", (unsigned int)((slots + 255) / 256), 1, 256, 1, b.ids, b.mappedRoutes, slots);
        go("copyWords", (unsigned int)((slots + 255) / 256), 1, 256, 1, b.weights, b.mappedRoutes + 4ULL * b.routeCap, slots);
        if(!routed && queue.eventCreate(&routed, 2) != 0) failed = true;
        if(queue.eventRecord(routed, stream) != 0 || queue.eventSync(routed) != 0) failed = true;
        const int* ids = b.hostRoutes;
        const float* weights = (const float*)(b.hostRoutes + b.routeCap);
        int* input = b.hostPlan;
        float* weight = (float*)(b.hostPlan + b.routeCap);
        int* groupSlot = b.hostPlan + 2 * b.routeCap;
        int* groupStart = groupSlot + s.experts;
        int* groupCount = groupStart + s.experts;
        std::vector<int> tally(s.experts, 0), begin(s.experts, 0);
        for(int i = 0; i < slots; i++){
            int e = ids[i];
            if(e < 0 || e >= s.experts){
                failed = true;
                continue;
            }
            tally[e]++;
        }
        int groups = 0, cursor = 0, widest = 1;
        for(int e = 0; e < s.experts; e++){
            if(!tally[e]) continue;
            groupSlot[groups] = e;
            groupStart[groups] = cursor;
            groupCount[groups] = tally[e];
            begin[e] = cursor;
            cursor += tally[e];
            widest = std::max(widest, tally[e]);
            groups++;
        }
        int* rankOf = groupCount + s.experts;
        for(int i = 0; i < slots; i++){
            int e = ids[i];
            if(e < 0 || e >= s.experts) continue;
            int a = begin[e]++;
            input[a] = i / used;
            weight[a] = weights[i];
            rankOf[i] = a;
        }
        int words = 3 * b.routeCap + 3 * s.experts;
        go("copyWords", (unsigned int)((words + 255) / 256), 1, 256, 1, b.mappedPlan, b.plan, words);
        uint64_t inputOnGpu = b.plan;
        uint64_t weightOnGpu = b.plan + 4ULL * b.routeCap;
        uint64_t slotOnGpu = b.plan + 8ULL * b.routeCap;
        uint64_t startOnGpu = slotOnGpu + 4ULL * s.experts;
        uint64_t countOnGpu = startOnGpu + 4ULL * s.experts;
        bool wide = gemmaWide && slots >= wideLeast * groups;
        int tileWide = tileWidth(wide);
        int tiles = (widest + tileWide - 1) / tileWide;
        unsigned int threads = tileWide == 128 ? 256 : 128;
        const GpuMatrix& gu = L.expertGateUp;
        const GpuMatrix& dn = L.expertDown;
        rankOnGpu = countOnGpu + 4ULL * s.experts;
        go(tiled(gu.type, wide, 0).c_str(), (unsigned int)((gu.rows + rowTile(gu.type, wide, 0) - 1) / rowTile(gu.type, wide, 0)), (unsigned int)(groups * tiles), threads, 1, (int)gu.type, gu.data, gu.stride, L.expertGateUpBytes, slotOnGpu, startOnGpu, countOnGpu, inputOnGpu, b.expertIn.values, b.expertIn.scales, b.expertIn.sums, none, none, none, b.both, gu.rows, gu.cols, 0, tiles, none, none, none, tables);
        go("gemmaGeluBothPack", (unsigned int)((slots * width / 32 + 7) / 8), 1, 32, 8, b.both, b.expertHid, width, slots, b.expertMid.values, b.expertMid.scales, b.expertMid.sums);
        go(tiled(dn.type, wide, 0).c_str(), (unsigned int)((dn.rows + rowTile(dn.type, wide, 0) - 1) / rowTile(dn.type, wide, 0)), (unsigned int)(groups * tiles), threads, 1, (int)dn.type, dn.data, dn.stride, L.expertDownBytes, slotOnGpu, startOnGpu, countOnGpu, none, b.expertMid.values, b.expertMid.scales, b.expertMid.sums, none, none, none, b.expertOut, dn.rows, dn.cols, 0, tiles, none, none, none, tables);
    } else {
        gemmaExpertRows(L.expertGateUp, L.expertGateUpBytes, used, b.moeIn, b.expertIn, b.both, slots);
        go("gemmaGeluBothPack", (unsigned int)((slots * width / 32 + 7) / 8), 1, 32, 8, b.both, b.expertHid, width, slots, b.expertMid.values, b.expertMid.scales, b.expertMid.sums);
        gemmaExpertRows(L.expertDown, L.expertDownBytes, 1, b.expertHid, b.expertMid, b.expertOut, slots);
    }
    goShared(mixBytes, "gemmaMoeOut", (unsigned int)n, 1, 256, 1, b.expertOut, b.weights, L.downScale, b.ids, rankOnGpu, b.mlp, L.postMlpNorm, L.postMoeNorm, L.postFfnNorm, b.a, b.x, L.outScale, used, d, s.eps);
}

inline bool GpuModel::gemmaChunk(const int* ids, const int* positions, int n, bool last) {
    GemmaShape& s = gemmaShape;
    GemmaBuffers& b = gemmaBuffers;
    CudaQueue& queue = cudaQueue();
    int d = s.hidden;
    if(cudaLink().sync() != 0) failed = true;
    pool.run(n, [&](int t){
        int id = ids[t];
        if(id < 0 || id >= embed.rows) id = 0;
        dequantRow(embed.type, embed.data + (uint64_t)id * embed.stride, b.hostEmbed + (size_t)t * d, d);
    });
    int top = 0;
    for(int t = 0; t < n; t++){
        b.hostPositions[t] = positions[t];
        top = std::max(top, positions[t]);
    }
    if(queue.copyUp(b.positions, b.hostPositions, (size_t)n * 4, stream) != 0) failed = true;
    float scale = sqrtf((float)d);
    auto work = [&]{
        go("gemmaEmbed", (unsigned int)((n * d + 255) / 256), 1, 256, 1, b.mappedEmbed, b.x, n * d, scale);
        for(int l = 0; l < s.layers && !failed; l++) gemmaRun(l, n, top);
        if(!last) return;
        uint64_t none = 0;
        float one = 1.0f;
        // con lista corta solo vuelven los mejores de cada fila con sus ids y la cpu no ordena todo el vocabulario
        int keep = shortlist > 0 ? shortlist : 0;
        PackedInput& packed = packs[0];
        if(everyRow){
            int count = n * s.vocab;
            go("gemmaRmsPack", (unsigned int)n, 1, 256, 1, b.x, b.outNorm, b.normed, d, s.eps, packed.values, packed.scales, packed.sums);
            project(head, b.normed, 0, b.every, n);
            go("gemmaCap", (unsigned int)((count + 255) / 256), 1, 256, 1, b.every, keep > 0 ? b.every : b.mappedEvery, count, s.softcap);
            if(keep > 0) go("shortRows", (unsigned int)n, 1, 512, 1, b.every, s.vocab, keep, b.mappedEvery);
            return;
        }
        go("gemmaRmsPack", 1, 1, 256, 1, b.x + 4ULL * (n - 1) * d, b.outNorm, b.last, d, s.eps, packed.values, packed.scales, packed.sums);
        project(head, b.last, 0, b.vocab, 1);
        go("gemmaCap", (unsigned int)((s.vocab + 255) / 256), 1, 256, 1, b.vocab, keep > 0 ? b.vocab : b.mappedLogits, s.vocab, s.softcap);
        if(keep > 0) go("shortRows", 1, 1, 512, 1, b.vocab, s.vocab, keep, b.mappedLogits);
    };
    // de 16 filas para arriba los expertos esperan a la cpu en medio de la capa y eso no se puede grabar
    bool single = n == 1 && last && !everyRow;
    bool checking = n > 1 && n < 16 && last && everyRow;
    bool graphed = (single || checking) && useGraphs && !profile;
    if(graphed){
        gemmaRounded = true;
        int sliding = 1, whole = 1;
        for(const GemmaLayer& L : gemmaLayers){
            if(L.sliding) sliding = gemmaSplits(L, n, top);
            else whole = gemmaSplits(L, n, top);
        }
        int key = (((single ? 1 : 3) * 64 + n) * 2 + (shortlist > 0 ? 1 : 0)) * 65536 + sliding * 256 + whole;
        bool recorded = !graphless.count(key);
        if(recorded && !replay(tokenGraphs[key], work)){
            graphless.insert(key);
            recorded = false;
        }
        if(!recorded) work();
        gemmaRounded = false;
    } else work();
    b.ringTop = std::max(b.ringTop, top);
    return !failed;
}

inline std::vector<float> GpuModel::forwardGemma(const std::vector<int>& ids, const std::vector<int>& positions) {
    std::vector<float> logits;
    GemmaShape& s = gemmaShape;
    GemmaBuffers& b = gemmaBuffers;
    if(!ready || ids.empty() || ids.size() != positions.size()) return logits;
    for(int p : positions){
        if(p < 0 || p >= contextLimit){
            problem = "position " + std::to_string(p) + " is past the context limit of " + std::to_string(contextLimit);
            return logits;
        }
    }
    if(!useCuda()){
        problem = "cuda context is not available";
        return logits;
    }
    if(everyRow && (int)ids.size() > std::min(b.everyCap, chunk)){
        problem = "at most " + std::to_string(std::min(b.everyCap, chunk)) + " rows can be checked per call";
        return logits;
    }
    failed = false;
    auto clock = std::chrono::steady_clock::now();
    int start = positions[0];
    gemmaDraft.pendingAt = -1;
    if(start > 0 && start <= (int)tokens.size() && start - s.window + 1 + s.ring <= b.ringTop){
        int from = std::max(0, start - s.window + 1);
        std::vector<int> again(tokens.begin() + from, tokens.begin() + start);
        std::vector<int> places(again.size());
        for(size_t i = 0; i < again.size(); i++) places[i] = from + (int)i;
        for(size_t at = 0; at < again.size() && !failed && !stopGenerating; at += chunk){
            int n = (int)std::min(again.size() - at, (size_t)chunk);
            gemmaChunk(again.data() + at, places.data() + at, n, false);
        }
        b.ringTop = start - 1;
    }
    tokens.resize(std::min(tokens.size(), (size_t)start));
    for(size_t at = 0; at < ids.size() && !failed && !stopGenerating; at += chunk){
        int n = (int)std::min(ids.size() - at, (size_t)chunk);
        tokens.insert(tokens.end(), ids.begin() + at, ids.begin() + at + n);
        gemmaChunk(ids.data() + at, positions.data() + at, n, at + n == ids.size());
    }
    int code = cudaLink().sync();
    if(stopGenerating && !failed){
        problem = "stopped";
        return logits;
    }
    if(failed || code != 0){
        problem = "gpu forward failed" + (code != 0 ? ", cuda error " + std::to_string(code) : "") + (launchProblem.empty() ? "" : ", " + launchProblem);
        launchProblem.clear();
        failed = false;
        return logits;
    }
    size_t width = shortlist > 0 ? 2 * (size_t)shortlist : (size_t)s.vocab;
    if(everyRow) logits.assign(b.hostEvery, b.hostEvery + ids.size() * width);
    else logits.assign(b.hostLogits, b.hostLogits + width);
    // el ayudante sigue desde el estado normado de la ultima fila leida y en una verificacion lo elige quien acepta las filas
    if(!everyRow){
        gemmaDraft.pendingFrom = b.last;
        gemmaDraft.pendingAt = positions.back();
    }
    headMs = since(clock);
    for(size_t i = 0; i < markUsed; i++){
        float ms = 0;
        cudaQueue().eventElapsed(&ms, marks[i].second.first, marks[i].second.second);
        kernelMs[marks[i].first] += ms;
        kernelCalls[marks[i].first]++;
    }
    markUsed = 0;
    return logits;
}

inline bool GpuModel::loadGemmaDraft(const std::string& path, const ModelFile& f) {
    GemmaShape& s = gemmaShape;
    GemmaDraft& g = gemmaDraft;
    std::string a = f.arch + ".";
    auto real = [&](const std::string& key, double fallback) {
        auto found = f.meta.find(a + key);
        return found == f.meta.end() ? fallback : found->second.real;
    };
    auto list = [&](const std::string& key) {
        auto found = f.meta.find(a + key);
        return found == f.meta.end() ? std::vector<int64_t>() : found->second.wholes;
    };
    int blocks = (int)metaWhole(f, a + "block_count", 0);
    g.hidden = (int)metaWhole(f, a + "embedding_length", 0);
    g.heads = (int)metaWhole(f, a + "attention.head_count", 0);
    g.feedForward = (int)metaWhole(f, a + "feed_forward_length", 0);
    g.eps = (float)real("attention.layer_norm_rms_epsilon", 1e-6);
    g.ropeBase = (float)real("rope.freq_base", 1e6);
    g.ropeBaseSliding = (float)real("rope.freq_base_swa", 1e4);
    int outWidth = (int)metaWhole(f, a + "embedding_length_out", 0);
    int fullSize = (int)metaWhole(f, a + "attention.key_length", 512);
    int slidingSize = (int)metaWhole(f, a + "attention.key_length_swa", 256);
    std::vector<int64_t> kvList = list("attention.head_count_kv");
    std::vector<int64_t> pattern = list("attention.sliding_window_pattern");
    if(kvList.empty()) kvList.assign(std::max(0, blocks), (int64_t)metaWhole(f, a + "attention.head_count_kv", 0));
    if(blocks <= 0 || g.hidden <= 0 || g.hidden % 32 != 0 || g.heads <= 0 || g.heads > s.heads || g.feedForward <= 0 || g.feedForward % 32 != 0 || outWidth != s.hidden || (int)kvList.size() < blocks || (int)pattern.size() < blocks){
        draftProblem = "unexpected gemma 4 assistant shape";
        return false;
    }
    int lastSliding = -1, lastFull = -1;
    for(int l = 0; l < s.layers; l++){
        if(gemmaLayers[l].sliding) lastSliding = l;
        else lastFull = l;
    }
    g.layers.assign(blocks, GemmaDraftLayer());
    for(int l = 0; l < blocks; l++){
        GemmaDraftLayer& L = g.layers[l];
        L.sliding = pattern[l] != 0;
        L.kvHeads = (int)kvList[l];
        L.headSize = L.sliding ? slidingSize : fullSize;
        L.target = L.sliding ? lastSliding : lastFull;
        bool fits = L.target >= 0 && gemmaLayers[L.target].kvHeads == L.kvHeads && gemmaLayers[L.target].headSize == L.headSize
            && (L.sliding ? L.headSize == 256 && L.kvHeads * 2 == g.heads : L.headSize == 512 && L.kvHeads * 8 == g.heads);
        if(!fits){
            draftProblem = "the gemma 4 assistant layer " + std::to_string(l) + " does not match the attention of the model";
            return false;
        }
    }

    HANDLE handle = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
    HANDLE mapping = handle != INVALID_HANDLE_VALUE ? CreateFileMappingA(handle, nullptr, PAGE_READONLY, 0, 0, nullptr) : nullptr;
    const uint8_t* base = mapping ? (const uint8_t*)MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0) : nullptr;
    std::string missing;
    auto upload = [&](const TensorInfo* t) -> uint64_t {
        uint64_t ptr = 0;
        if(cudaLink().alloc(&ptr, (size_t)std::max<uint64_t>(t->bytes, 256)) != 0){
            if(missing.empty()) missing = "out of video memory for the gemma 4 assistant";
            return 0;
        }
        allocations.push_back(ptr);
        g.bytes += t->bytes;
        if(cudaLink().copyUp(ptr, base + t->offset, (size_t)t->bytes) != 0 && missing.empty()) missing = "cannot copy " + t->name + " of the gemma 4 assistant";
        return ptr;
    };
    auto matrix = [&](const std::string& name, GpuMatrix& m, int cols, int rows) {
        const TensorInfo* t = findTensor(f, name);
        if(!t || !knownType(t->type) || t->type == 0 || (int)t->dims[0] != cols || (int)(t->elements / t->dims[0]) != rows){
            if(missing.empty()) missing = "the gemma 4 assistant has no usable " + name;
            return;
        }
        m.type = t->type;
        m.cols = cols;
        m.rows = rows;
        m.stride = rowBytes(t->type, t->dims[0]);
        m.data = upload(t);
    };
    auto floats = [&](const std::string& name, uint64_t& out, uint64_t count) {
        const TensorInfo* t = findTensor(f, name);
        if(!t || t->type != 0 || t->elements != count){
            if(missing.empty()) missing = "the gemma 4 assistant has no usable " + name;
            return;
        }
        out = upload(t);
    };
    auto grab = [&](uint64_t bytes) -> uint64_t {
        uint64_t ptr = 0;
        if(cudaLink().alloc(&ptr, (size_t)std::max<uint64_t>(bytes, 256)) != 0){
            if(missing.empty()) missing = "out of video memory for the gemma 4 assistant";
            return 0;
        }
        allocations.push_back(ptr);
        g.bytes += bytes;
        return ptr;
    };
    if(!base) missing = "cannot map " + path;
    else {
        matrix("nextn.pre_projection.weight", g.join, 2 * s.hidden, g.hidden);
        matrix("nextn.post_projection.weight", g.back, g.hidden, s.hidden);
        matrix("token_embd.weight", g.head, g.hidden, s.vocab);
        floats("output_norm.weight", g.outNorm, (uint64_t)g.hidden);
        if(findTensor(f, "rope_freqs.weight")) floats("rope_freqs.weight", g.ropeFactors, (uint64_t)fullSize / 2);
        else g.ropeFactors = lastFull >= 0 ? gemmaLayers[lastFull].ropeFactors : 0;
        for(int l = 0; l < blocks && missing.empty(); l++){
            GemmaDraftLayer& L = g.layers[l];
            std::string p = "blk." + std::to_string(l) + ".";
            int width = g.heads * L.headSize;
            floats(p + "attn_norm.weight", L.attnNorm, (uint64_t)g.hidden);
            matrix(p + "attn_q.weight", L.query, g.hidden, width);
            floats(p + "attn_q_norm.weight", L.queryNorm, (uint64_t)L.headSize);
            matrix(p + "attn_output.weight", L.out, width, g.hidden);
            floats(p + "post_attention_norm.weight", L.postAttnNorm, (uint64_t)g.hidden);
            floats(p + "ffn_norm.weight", L.ffnNorm, (uint64_t)g.hidden);
            matrix(p + "ffn_gate.weight", L.gate, g.hidden, g.feedForward);
            matrix(p + "ffn_up.weight", L.up, g.hidden, g.feedForward);
            matrix(p + "ffn_down.weight", L.down, g.feedForward, g.hidden);
            floats(p + "post_ffw_norm.weight", L.postFfnNorm, (uint64_t)g.hidden);
            if(findTensor(f, p + "layer_output_scale.weight")) floats(p + "layer_output_scale.weight", L.outScale, 1);
        }
    }
    if(base) UnmapViewOfFile(base);
    if(mapping) CloseHandle(mapping);
    if(handle != INVALID_HANDLE_VALUE) CloseHandle(handle);
    if(!missing.empty()){
        draftProblem = missing;
        return false;
    }

    int widest = g.heads * 512;
    g.input = grab(8ULL * s.hidden);
    g.x = grab(4ULL * g.hidden);
    g.xn = grab(4ULL * g.hidden);
    g.a = grab(4ULL * g.hidden);
    g.attn = grab(4ULL * g.hidden);
    g.mlp = grab(4ULL * g.hidden);
    g.fin = grab(4ULL * g.hidden);
    g.q = grab(4ULL * widest);
    g.o = grab(4ULL * widest);
    g.gate = grab(4ULL * g.feedForward);
    g.up = grab(4ULL * g.feedForward);
    g.hid = grab(4ULL * g.feedForward);
    g.vocab = grab(4ULL * s.vocab);
    g.position = grab(256);
    void* place = nullptr;
    if(cudaLink().hostAlloc(&place, 256, 0) == 0){
        pinned.push_back(place);
        g.hostPosition = (int*)place;
    }
    void* pick = nullptr;
    if(cudaLink().hostAlloc(&pick, 256, 3) == 0 && cudaQueue().hostDevice(&g.mappedPick, pick, 0) == 0){
        pinned.push_back(pick);
        g.hostPick = (volatile uint32_t*)pick;
    }
    if(!missing.empty() || !g.hostPosition || !g.hostPick || cudaLink().sync() != 0){
        draftProblem = missing.empty() ? "cannot pin host memory for the gemma 4 assistant" : missing;
        return false;
    }
    g.ready = true;
    drafting = true;
    tiedDraft = false;
    return true;
}

inline void GpuModel::gemmaDraftStep(int top) {
    GemmaShape& s = gemmaShape;
    GemmaBuffers& b = gemmaBuffers;
    GemmaDraft& g = gemmaDraft;
    int d = s.hidden;
    int h = g.hidden;
    float one = 1.0f;
    uint64_t none = 0;
    go("gemmaEmbed", (unsigned int)((d + 255) / 256), 1, 256, 1, b.mappedEmbed, g.input, d, sqrtf((float)d));
    pack(g.input, 2 * d, 0);
    project(g.join, g.input, 0, g.x, 1);
    PackedInput& packed = packs[0];
    for(const GemmaDraftLayer& L : g.layers){
        const GemmaLayer& T = gemmaLayers[L.target];
        go("gemmaRmsPack", 1, 1, 256, 1, g.x, L.attnNorm, g.xn, h, g.eps, packed.values, packed.scales, packed.sums);
        project(L.query, g.xn, 0, g.q, 1);
        go("gemmaHeads", (unsigned int)g.heads, 1, (unsigned int)(L.headSize / 2), 1, g.q, none, none, L.queryNorm, none, none, none, g.position, g.heads, 0, L.headSize, L.query.rows, 0, 0, L.sliding ? g.ropeBaseSliding : g.ropeBase, L.sliding ? none : g.ropeFactors, g.eps);
        int splits = gemmaSplits(T, 1, top);
        if(gemmaFew) go(L.headSize == 256 ? "gemmaFew256" : "gemmaFew512", (unsigned int)L.kvHeads, (unsigned int)splits, 256, 1, g.q, T.keys, T.values, g.position, g.o, b.scratch, g.heads, L.kvHeads, L.sliding ? s.ring : 0, L.sliding ? s.window : 0, splits);
        else go(L.headSize == 256 ? "gemmaAttention256" : "gemmaAttention512", (unsigned int)L.kvHeads, (unsigned int)splits, 256, 1, g.q, T.keys, T.values, g.position, g.o, b.scratch, 1, g.heads, L.kvHeads, L.sliding ? s.ring : 0, L.sliding ? s.window : 0, splits);
        if(splits > 1) go("gemmaMerge", (unsigned int)g.heads, 1, 256, 1, b.scratch, g.o, g.heads, L.headSize, splits);
        pack(g.o, L.query.rows, 0);
        project(L.out, g.o, 0, g.attn, 1);
        go("gemmaAddRmsPack", 1, 1, 256, 1, g.attn, L.postAttnNorm, g.x, g.a, L.ffnNorm, g.xn, h, g.eps, packed.values, packed.scales, packed.sums);
        projectMany({{&L.gate, g.gate}, {&L.up, g.up}}, g.xn, 0, 1);
        go("gemmaGeluPack", (unsigned int)((g.feedForward / 32 + 7) / 8), 1, 32, 8, g.gate, g.up, g.hid, g.feedForward, packed.values, packed.scales, packed.sums);
        project(L.down, g.hid, 0, g.mlp, 1);
        go("gemmaRmsAdd", 1, 1, 256, 1, g.mlp, L.postFfnNorm, g.a, g.x, h, g.eps, L.outScale);
    }
    go("gemmaRmsPack", 1, 1, 256, 1, g.x, g.outNorm, g.fin, h, g.eps, packed.values, packed.scales, packed.sums);
    project(g.head, g.fin, 0, g.vocab, 1);
    project(g.back, g.fin, 0, g.input + 4ULL * d, 1);
    go("gemmaCap", (unsigned int)((g.head.rows + 255) / 256), 1, 256, 1, g.vocab, g.vocab, g.head.rows, s.softcap);
    go("pickTop", 1, 1, 1024, 1, g.vocab, g.head.rows, g.mappedPick);
}

// cada paso del ayudante mira el contexto desde la ultima posicion que leyo el modelo y adivina el token que sigue al que le dan
inline std::vector<int> GpuModel::gemmaPropose(int token, int most) {
    std::vector<int> drafts;
    GemmaShape& s = gemmaShape;
    GemmaBuffers& b = gemmaBuffers;
    GemmaDraft& g = gemmaDraft;
    if(!g.ready || most <= 0 || g.pendingAt < 0 || g.pendingAt != (int)tokens.size() - 1 || !g.pendingFrom) return drafts;
    CudaQueue& queue = cudaQueue();
    int d = s.hidden;
    int top = g.pendingAt;
    if(cudaLink().sync() != 0) return drafts;
    g.hostPosition[0] = top;
    if(queue.copyUp(g.position, g.hostPosition, 4, stream) != 0) return drafts;
    queue.copyAcross(g.input + 4ULL * d, g.pendingFrom, (size_t)d * 4, stream);
    gemmaRounded = true;
    int sliding = 1, whole = 1;
    for(const GemmaDraftLayer& L : g.layers){
        if(L.sliding) sliding = gemmaSplits(gemmaLayers[L.target], 1, top);
        else whole = gemmaSplits(gemmaLayers[L.target], 1, top);
    }
    int key = 5 * 128 * 65536 + sliding * 256 + whole;
    for(int k = 0; k < most; k++){
        int id = token < 0 || token >= embed.rows ? 0 : token;
        dequantRow(embed.type, embed.data + (uint64_t)id * embed.stride, b.hostEmbed, d);
        bool recorded = useGraphs && !profile && !graphless.count(key);
        if(recorded && !replay(tokenGraphs[key], [&]{ gemmaDraftStep(top); })){
            graphless.insert(key);
            recorded = false;
        }
        if(!recorded) gemmaDraftStep(top);
        if(waitStream() != 0 || failed){
            failed = false;
            break;
        }
        int guess = (int)g.hostPick[0];
        uint32_t bits = g.hostPick[1];
        float sure = 0.0f;
        memcpy(&sure, &bits, 4);
        if(guess < 0 || guess >= s.vocab || sure < draftSure) break;
        drafts.push_back(guess);
        token = guess;
    }
    gemmaRounded = false;
    for(size_t i = 0; i < markUsed; i++){
        float ms = 0;
        cudaQueue().eventElapsed(&ms, marks[i].second.first, marks[i].second.second);
        kernelMs[marks[i].first] += ms;
        kernelCalls[marks[i].first]++;
    }
    markUsed = 0;
    return drafts;
}

// tras una verificacion el ayudante sigue desde la ultima fila aceptada sin volver a pasar nada por el modelo
inline bool GpuModel::gemmaKeep(int count) {
    if(count < 1 || count > gemmaBuffers.everyCap) return false;
    gemmaDraft.pendingFrom = gemmaBuffers.normed + 4ULL * (count - 1) * gemmaShape.hidden;
    gemmaDraft.pendingAt = (int)tokens.size() - 1;
    return true;
}

inline bool GpuModel::moveGemma(int slot, bool keep) {
    if(!ready || slot < 0 || slot >= 4) return false;
    if(keep){
        savedTokens[slot] = tokens.size();
        gemmaBuffers.saved[slot] = true;
        return true;
    }
    if(!gemmaBuffers.saved[slot]) return false;
    tokens.resize(std::min(tokens.size(), savedTokens[slot]));
    return true;
}
