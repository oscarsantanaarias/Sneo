#pragma once

inline bool GpuModel::loadInk(const std::string& path, int threads) {
    ink = true;
    dense = true;
    usePle = false;
    streamPrefill = false;
    pinWarm = false;
    foresee = false;
    residual = false;
    firstLayer = 0;
    endLayer = 0;
    layers.clear();
    InkShape& s = inkShape;
    InkBuffers& b = inkBuffers;
    std::string a = "inkling2_loop.";
    auto real = [&](const std::string& key, double fallback) {
        auto found = file.meta.find(a + key);
        return found == file.meta.end() ? fallback : found->second.real;
    };
    s.layers = (int)file.layers;
    s.hidden = (int)metaWhole(file, a + "embedding_length", 0);
    s.experts = (int)metaWhole(file, a + "expert_count", 0);
    s.used = (int)metaWhole(file, a + "expert_used_count", 0);
    s.expertWidth = (int)metaWhole(file, a + "expert_feed_forward_length", 0);
    s.window = (int)metaWhole(file, a + "attention.sliding_window", 1024);
    s.extent = (int)metaWhole(file, a + "attention.relative_extent", 1024);
    s.relDim = (int)metaWhole(file, a + "attention.relative_length", 16);
    s.loopStart = (int)metaWhole(file, a + "loop.start", 6);
    s.loopEnd = (int)metaWhole(file, a + "loop.end", 18);
    s.passes = (int)metaWhole(file, a + "loop.count", 3);
    s.rank = (int)metaWhole(file, a + "loop.gate_rank", 64);
    s.block = (int)metaWhole(file, a + "attn_res.block_size", 6);
    s.unpadded = (int)metaWhole(file, a + "vocab_size_unpadded", 0);
    s.eps = (float)real("attention.layer_norm_rms_epsilon", 1e-6);
    s.qkClip = (float)real("attention.qk_clip", 22.627417);
    s.relClip = (float)real("attention.relative_clip", 8.0);
    s.logFloor = (float)real("attention.log_scale_floor", 8192.0);
    s.logAlpha = (float)real("attention.log_scale_alpha", 0.1);
    s.softcap = (float)real("final_logit_softcapping", 30.0);
    s.gateCap = (float)real("activation.situ_beta", 4.0);
    s.upCap = (float)real("activation.situ_linear_beta", 25.0);
    s.routeScale = (float)real("expert_weights_scale", 8.0);
    std::vector<int64_t> pattern;
    if(auto found = file.meta.find(a + "attention.sliding_window_pattern"); found != file.meta.end()) pattern = found->second.wholes;
    if(s.layers <= 0 || s.hidden != 2304 || s.relDim != 16 || s.used > 32 || s.experts > 1024 || (int)pattern.size() < s.layers || s.loopStart % s.block != 0 || s.loopEnd % s.block != 0 || s.loopEnd > s.layers || s.rank != 64){
        problem = "this inkling2_loop layout is not the one the engine runs";
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
    if(!buildModule(sizes + quantKernelSource + modelKernelSource + gemmaKernelSource + hyKernelSource + glmKernelSource + inkKernelSource, module)){
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
    auto host = [&](const TensorInfo* t) { return maps[t->file].base + t->offset; };

    std::vector<std::pair<const TensorInfo*, uint64_t*>> wanted;
    std::vector<std::pair<const TensorInfo*, uint64_t*>> widened;
    auto matrix = [&](const std::string& name, GpuMatrix& out) {
        const TensorInfo* t = findTensor(file, name);
        if(!t || (!knownType(t->type) && t->type != 0)){
            if(problem.empty()) problem = t
                ? name + " is " + typeInfo(t->type).name + ", a tensor type the engine cannot read yet"
                : "missing tensor " + name;
            return;
        }
        out.type = t->type;
        out.cols = (int)t->dims[0];
        out.rows = (int)(t->dims[1] * t->dims[2]);
        out.stride = rowBytes(out.type, t->dims[0]);
        wanted.push_back({t, &out.data});
    };
    auto floats = [&](const std::string& name, uint64_t& out, bool optional = false) {
        const TensorInfo* t = findTensor(file, name);
        if(!t){
            if(!optional && problem.empty()) problem = "missing tensor " + name;
            return;
        }
        if(t->type == 0){
            wanted.push_back({t, &out});
            return;
        }
        widened.push_back({t, &out});
    };

    const TensorInfo* table = findTensor(file, "token_embd.weight");
    if(!table || !knownType(table->type) || (int)table->dims[0] != s.hidden){
        problem = "missing or unsupported token_embd.weight";
        return false;
    }
    embed.data = host(table);
    embed.type = table->type;
    embed.cols = (int)table->dims[0];
    embed.rows = (int)table->dims[1];
    embed.stride = rowBytes(table->type, table->dims[0]);
    s.vocab = embed.rows;
    if(s.unpadded <= 0 || s.unpadded > s.vocab) s.unpadded = s.vocab;
    matrix("output.weight", head);
    floats("output_norm.weight", b.outNorm);
    floats("output_res_query.weight", b.outQuery);
    floats("token_embd_norm.weight", b.embNorm);

    inkLayers.assign(s.layers, InkLayer());
    for(int l = 0; l < s.layers && problem.empty(); l++){
        InkLayer& L = inkLayers[l];
        std::string p = "blk." + std::to_string(l) + ".";
        L.full = pattern[l] == 0;
        L.loop = l >= s.loopStart && l < s.loopEnd;
        matrix(p + "attn_q.weight", L.q);
        matrix(p + "attn_k.weight", L.k);
        matrix(p + "attn_v.weight", L.v);
        matrix(p + "attn_gate.weight", L.gate);
        matrix(p + "attn_output.weight", L.out);
        matrix(p + "attn_rel.weight", L.rel);
        matrix(p + "attn_sink_proj.weight", L.sink);
        floats(p + "attn_q_clip.weight", L.qClip);
        floats(p + "attn_rel_clip.weight", L.relClip);
        floats(p + "attn_rel_bias.weight", L.relBank);
        floats(p + "attn_k_conv.weight", L.kConv);
        floats(p + "attn_v_conv.weight", L.vConv);
        floats(p + "attn_post_conv.weight", L.postConv);
        floats(p + "ffn_post_conv.weight", L.ffnConv);
        floats(p + "ffn_res_query.weight", L.ffnQuery);
        floats(p + "ffn_norm.weight", L.ffnNorm);
        floats(p + "ffn_gate_inp.weight", L.router);
        floats(p + "exp_probs_b.bias", L.routerBias);
        floats(p + "ffn_gate_up_exps.scale", L.guScale, true);
        floats(p + "ffn_down_exps.scale", L.dScale, true);
        if(l > 0){
            floats(p + "attn_res_query.weight", L.resQuery);
            const TensorInfo* rq = findTensor(file, p + "attn_res_query.weight");
            L.queries = rq ? (int)(rq->elements / s.hidden) : 0;
        }
        if(L.full){
            floats(p + "attn_q_inp_norm.weight", L.qNorm);
            floats(p + "attn_k_inp_norm.weight", L.kNorm);
            floats(p + "attn_v_inp_norm.weight", L.vNorm);
            matrix(p + "attn_k_conv_proj.weight", L.kProj);
            matrix(p + "attn_v_conv_proj.weight", L.vProj);
        } else floats(p + "attn_norm.weight", L.norm);
        if(L.loop){
            const char* kinds[4] = {"q", "k", "v", "ffn"};
            for(int k = 0; k < 4; k++){
                floats(p + "loop_" + kinds[k] + "_ctrl_down.weight", L.ctrlDown[k]);
                floats(p + "loop_" + kinds[k] + "_ctrl_forget.weight", L.ctrlForget[k]);
                floats(p + "loop_" + kinds[k] + "_ctrl_gain.weight", L.ctrlGain[k]);
            }
            floats(p + "loop_msg_query.weight", L.msgQuery);
        }
        const TensorInfo* gu = findTensor(file, p + "ffn_gate_up_exps.weight");
        const TensorInfo* dn = findTensor(file, p + "ffn_down_exps.weight");
        if(!gu || !dn || (int)gu->dims[2] != s.experts || (int)gu->dims[1] != 2 * s.expertWidth || (int)dn->dims[0] != s.expertWidth){
            problem = "unexpected inkling2_loop experts in layer " + std::to_string(l);
            return false;
        }
        matrix(p + "ffn_gate_up_exps.weight", L.gateUp);
        matrix(p + "ffn_down_exps.weight", L.down);
        L.gateUpBytes = (uint64_t)gu->dims[1] * rowBytes(gu->type, gu->dims[0]);
        L.downBytes = (uint64_t)dn->dims[1] * rowBytes(dn->type, dn->dims[0]);
        if(problem.empty() && ((L.full ? 3 : 1) != L.queries && l > 0)){
            problem = "unexpected inkling2_loop residual queries in layer " + std::to_string(l);
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

    uint64_t widenedBytes = 0;
    for(auto& w : widened) widenedBytes += (w.first->elements * 4 + 255) / 256 * 256;
    for(auto& w : wanted) denseBytes += (w.first->bytes + 255) / 256 * 256;
    denseBytes += widenedBytes;

    const char* rowsAsked = getenv("SNEO_INK_ROWS");
    chunk = rowsAsked ? std::max(64, atoi(rowsAsked)) / 64 * 64 : 256;
    uint64_t C = (uint64_t)chunk, d = (uint64_t)s.hidden, A = (uint64_t)s.attendRows;
    uint64_t pairs = C * s.used, width = (uint64_t)s.expertWidth;
    b.pairsMost = (int)pairs;
    s.runs = s.loopStart + (s.loopEnd - s.loopStart) * s.passes + (s.layers - s.loopEnd);
    int fullRuns = 0;
    for(int r = 0; r < s.runs; r++){
        int l = r < s.loopStart ? r : r < s.loopStart + (s.loopEnd - s.loopStart) * s.passes ? s.loopStart + (r - s.loopStart) % (s.loopEnd - s.loopStart) : r - (s.loopEnd - s.loopStart) * (s.passes - 1);
        fullRuns += inkLayers[l].full ? 1 : 0;
    }
    int ring = s.window + (int)C;
    uint64_t slidingBytes = (uint64_t)(s.runs - fullRuns) * ring * s.kvHeads * 128 * 2 * 2;
    uint64_t perToken = (uint64_t)fullRuns * s.kvHeads * 128 * 2 * 2;
    uint64_t bufferBytes = C * 4 * (d * 14 + 3072 * 3 + 1024 * 2 + 384 + 24 + 64 + 128 + 16 * 3) + pairs * 4 * (2 * width + width + d + 4) + A * s.heads * 130 * 4 * 16
        + (uint64_t)s.vocab * 4 + d * 4 + std::max<uint64_t>(C * 3072, pairs * width) * 2 * 2 + C * d * 2 + pairs * width * 2;
    uint64_t historyBytes = (uint64_t)s.runs * 3 * (1024 * 2 + d * 2) * 4;
    uint64_t margin = 700ULL << 20;
    uint64_t fixedBytes = denseBytes + bufferBytes + slidingBytes + historyBytes + margin;
    if(fixedBytes >= startFree){
        problem = "inkling2_loop needs " + std::to_string((fixedBytes >> 20) + 1) + " MB of video memory before any context and " + std::to_string(startFree >> 20) + " MB are free";
        return false;
    }
    uint64_t room = perToken > 0 ? (startFree - fixedBytes) / perToken : (uint64_t)contextLimit;
    if((uint64_t)contextLimit > room){
        int fitted = (int)(room / 256 * 256);
        if(fitted < 4096){
            problem = "inkling2_loop fits only " + std::to_string(fitted) + " tokens of context in the free video memory";
            return false;
        }
        contextLimit = fitted;
    }

    loadSay("Copying the model to the gpu");
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
    for(auto& w : widened){
        const TensorInfo* t = w.first;
        int cols = (int)t->dims[0];
        uint64_t rows = t->elements / cols;
        uint64_t stride = rowBytes(t->type, cols);
        std::vector<float> full((size_t)t->elements);
        for(uint64_t r = 0; r < rows; r++) dequantRow(t->type, host(t) + r * stride, full.data() + r * cols, cols);
        *w.second = arena + used;
        if(cudaLink().copyUp(arena + used, full.data(), full.size() * 4) != 0){
            problem = "cannot copy " + t->name + " to the gpu";
            return false;
        }
        used += (full.size() * 4 + 255) / 256 * 256;
    }

    inkRuns.assign(s.runs, InkRun());
    for(int r = 0; r < s.runs && problem.empty(); r++){
        int l = r < s.loopStart ? r : r < s.loopStart + (s.loopEnd - s.loopStart) * s.passes ? s.loopStart + (r - s.loopStart) % (s.loopEnd - s.loopStart) : r - (s.loopEnd - s.loopStart) * (s.passes - 1);
        InkRun& R = inkRuns[r];
        R.ring = inkLayers[l].full ? 0 : ring;
        uint64_t slots = inkLayers[l].full ? (uint64_t)contextLimit : (uint64_t)ring;
        R.kCache = grab(slots * s.kvHeads * 128 * 2);
        R.vCache = grab(slots * s.kvHeads * 128 * 2);
        R.kHist = grab(3ULL * 1024 * 4);
        R.vHist = grab(3ULL * 1024 * 4);
        R.postHist = grab(3ULL * d * 4);
        R.ffnHist = grab(3ULL * d * 4);
        cacheBytes += slots * s.kvHeads * 128 * 4;
    }
    for(int i = 0; i < 2; i++) b.h[i] = grab(C * d * 4);
    for(int i = 0; i < 4; i++) b.blocks[i] = grab(C * d * 4);
    for(int i = 0; i < 4; i++) b.msgs[i] = grab(C * d * 4);
    b.prevBlock = grab(C * d * 4);
    b.prevH = grab(C * d * 4);
    b.xin = grab(C * d * 4);
    b.hq = grab(C * d * 4);
    b.hk = grab(C * d * 4);
    b.hv = grab(C * d * 4);
    b.q = grab(C * 3072 * 4);
    b.k = grab(C * 1024 * 4);
    b.v = grab(C * 1024 * 4);
    b.rel = grab(C * 384 * 4);
    b.sink = grab(C * 24 * 4);
    b.gateRaw = grab(C * 3072 * 4);
    b.attn = grab(C * 3072 * 4);
    b.dynK = grab(C * 32 * 4);
    b.dynV = grab(C * 32 * 4);
    b.proj = grab(C * d * 4);
    b.resid = grab(C * d * 4);
    b.mlpIn = grab(C * d * 4);
    b.xn = grab(C * d * 4);
    b.logits = grab(C * s.experts * 4);
    b.moe = grab(C * d * 4);
    b.ids = grab(pairs * 4);
    b.weights = grab(pairs * 4);
    b.rowScale = grab(pairs * 4);
    b.rowWeight = grab(pairs * 4);
    b.guOut = grab(pairs * 2 * width * 4);
    b.act = grab(pairs * width * 4);
    b.downOut = grab(pairs * d * 4);
    b.inValues = grab(C * d);
    b.inScales = grab(C * d / 8);
    b.inSums = grab(C * d / 8);
    b.hidValues = grab(pairs * width);
    b.hidScales = grab(pairs * width / 8);
    b.hidSums = grab(pairs * width / 8);
    b.positions = grab(C * 4);
    b.last = grab(d * 4);
    b.vocab = grab((uint64_t)s.vocab * 4);
    b.scratchFloats = A * s.heads * 130 * 16;
    b.scratch = grab(b.scratchFloats * 4);
    b.gates = grab(C * s.rank * 4);
    splitScratch = grab(4ULL << 20);
    std::vector<int> identity(pairs);
    for(uint64_t i = 0; i < pairs; i++) identity[i] = (int)i;
    b.pairs = grab(pairs * 4);
    if(b.pairs) cudaLink().copyUp(b.pairs, identity.data(), pairs * 4);
    for(int slot = 0; slot < 2; slot++){
        PackedInput* packed = &packs[slot];
        int capacity = (int)((std::max<uint64_t>(C * 3072, pairs * std::max(width, d)) + 255) / 256 * 256);
        if(cudaLink().alloc(&packed->values, capacity) != 0 || cudaLink().alloc(&packed->scales, capacity / 8) != 0 || cudaLink().alloc(&packed->sums, capacity / 8) != 0){
            problem = "out of video memory for packed inputs";
            return false;
        }
        packed->capacity = capacity;
    }
    b.hostEmbed = (float*)shared(C * d * 4, b.mappedEmbed);
    b.hostLogits = (float*)shared((uint64_t)s.vocab * 4, b.mappedLogits);
    b.hostIds = (int*)shared(pairs * 4, b.mappedIds);
    b.hostWeights = (float*)shared(pairs * 4, b.mappedWeights);
    b.hostGroups = (int*)shared((uint64_t)(3 * s.experts + 4 * pairs) * 4, b.groupsOn);
    void* hostPos = nullptr;
    if(cudaLink().hostAlloc(&hostPos, (size_t)(C * 4), 0) == 0){
        pinned.push_back(hostPos);
        b.hostPositions = (int*)hostPos;
    }
    if(!problem.empty() || !b.hostEmbed || !b.hostLogits || !b.hostIds || !b.hostWeights || !b.hostGroups || !b.hostPositions){
        if(problem.empty()) problem = "cannot pin host memory for inkling2_loop";
        return false;
    }
    if(queue.streamCreate(&stream, 0) != 0){
        problem = "cannot create a cuda stream";
        return false;
    }
    b.saved.assign(4, std::vector<uint8_t>());
    b.savedTokens.assign(4, 0);
    size_t nowFree = 0, nowTotal = 0;
    cudaLink().info(&nowFree, &nowTotal);
    vramHeld = startFree > nowFree ? startFree - nowFree : 0;
    residentBytes = 0;
    for(const TensorInfo& t : file.tensors) residentBytes += t.expert ? t.bytes : 0;
    residentExperts = s.experts;
    pool.start(std::max(0, threads - 1));
    profile = getenv("SNEO_INK_PROFILE") != nullptr;
    std::cout << "inkling2_loop: " << s.layers << " layers run as " << s.runs << " (layers " << s.loopStart << " to " << s.loopEnd - 1 << " " << s.passes << " times), all " << s.experts << " experts on the gpu, context " << contextLimit << ", " << (nowFree >> 20) << " MB left" << std::endl;
    ready = true;
    reset();
    return true;
}

inline void GpuModel::inkReset() {
    InkShape& s = inkShape;
    if(!ready || !useCuda()) return;
    for(InkRun& R : inkRuns){
        cudaLink().memset(R.kHist, 0, 3ULL * 1024 * 4);
        cudaLink().memset(R.vHist, 0, 3ULL * 1024 * 4);
        cudaLink().memset(R.postHist, 0, 3ULL * s.hidden * 4);
        cudaLink().memset(R.ffnHist, 0, 3ULL * s.hidden * 4);
    }
}

inline bool GpuModel::moveInk(int slot, bool keep) {
    InkShape& s = inkShape;
    InkBuffers& b = inkBuffers;
    if(!ready || slot < 0 || slot >= 4 || !useCuda()) return false;
    uint64_t narrowBytes = 3ULL * 1024 * 4, broadBytes = 3ULL * s.hidden * 4;
    uint64_t total = (uint64_t)inkRuns.size() * (2 * narrowBytes + 2 * broadBytes);
    if(waitStream() != 0) return false;
    std::vector<uint8_t>& copy = b.saved[slot];
    if(keep){
        copy.resize((size_t)total);
        uint64_t at = 0;
        for(InkRun& R : inkRuns){
            for(auto part : {std::make_pair(R.kHist, narrowBytes), std::make_pair(R.vHist, narrowBytes), std::make_pair(R.postHist, broadBytes), std::make_pair(R.ffnHist, broadBytes)}){
                if(cudaLink().copyDown(copy.data() + at, part.first, (size_t)part.second) != 0) return false;
                at += part.second;
            }
        }
        b.savedTokens[slot] = tokens.size();
        savedTokens[slot] = tokens.size();
        return true;
    }
    if(copy.size() != total || b.savedTokens[slot] > tokens.size()) return false;
    if(tokens.size() - b.savedTokens[slot] > (size_t)chunk) return false;
    uint64_t at = 0;
    for(InkRun& R : inkRuns){
        for(auto part : {std::make_pair(R.kHist, narrowBytes), std::make_pair(R.vHist, narrowBytes), std::make_pair(R.postHist, broadBytes), std::make_pair(R.ffnHist, broadBytes)}){
            if(cudaLink().copyUp(part.first, copy.data() + at, (size_t)part.second) != 0) return false;
            at += part.second;
        }
    }
    tokens.resize(b.savedTokens[slot]);
    return true;
}

inline void GpuModel::inkProject(const GpuMatrix& m, uint64_t input, uint64_t output, int n) {
    if(m.type == 0){
        go("hyF32Rows", (unsigned int)((m.rows + 7) / 8), 1, 256, 1, m.data, input, m.cols, output, m.rows, m.rows, m.cols, n);
        return;
    }
    if(m.type != 30 && m.type != 1) hyPack(input, n * m.cols, 0);
    project(m, input, 0, output, n);
}

inline void GpuModel::inkExperts(InkLayer& L, int n) {
    InkShape& s = inkShape;
    InkBuffers& b = inkBuffers;
    CudaQueue& queue = cudaQueue();
    int used = s.used, d = s.hidden, width = s.expertWidth, pairs = n * used;
    PackedInput& p = packs[0];
    uint64_t none = 0;
    auto blocks = [](uint64_t count){ return (unsigned int)((count + 255) / 256); };
    if(n <= s.fewRows){
        go("inkPairs", blocks(pairs), 1, 256, 1, b.ids, b.weights, L.guScale, L.dScale, b.rowScale, b.rowWeight, pairs);
        go("hyGatherPack", (unsigned int)(((uint64_t)pairs * d / 32 + 7) / 8), 1, 32, 8, b.xn, b.pairs, b.downOut, d, used, pairs, p.values, p.scales, p.sums);
        go("glmExpertRows", (unsigned int)((2 * width + 7) / 8), (unsigned int)pairs, 256, 1, L.gateUp.data, L.gateUp.stride, L.gateUpBytes, none, b.ids, p.values, p.scales, p.sums, b.guOut, 2 * width, d, (int)L.gateUp.type, tables);
        go("inkAct", (unsigned int)(((uint64_t)pairs * width / 32 + 7) / 8), 1, 32, 8, b.guOut, b.rowScale, b.act, pairs, width, s.gateCap, s.upCap, p.values, p.scales, p.sums);
        go("glmExpertRows", (unsigned int)((d + 7) / 8), (unsigned int)pairs, 256, 1, L.down.data, L.down.stride, L.downBytes, none, b.ids, p.values, p.scales, p.sums, b.downOut, d, width, (int)L.down.type, tables);
        go("inkScatter", (unsigned int)n, 1, 256, 1, b.downOut, b.rowWeight, b.moe, d, used);
        return;
    }
    if(queue.copyAcross(b.mappedIds, b.ids, (size_t)pairs * 4, stream) != 0 || queue.copyAcross(b.mappedWeights, b.weights, (size_t)pairs * 4, stream) != 0) failed = true;
    if(!b.routed && queue.eventCreate(&b.routed, 2) != 0) failed = true;
    if(queue.eventRecord(b.routed, stream) != 0 || queue.eventSync(b.routed) != 0) failed = true;
    std::vector<float> guScale(s.experts, 1.0f), dScale(s.experts, 1.0f);
    if(L.guScale) cudaLink().copyDown(guScale.data(), L.guScale, (size_t)s.experts * 4);
    if(L.dScale) cudaLink().copyDown(dScale.data(), L.dScale, (size_t)s.experts * 4);
    std::vector<std::vector<int>> byExpert(s.experts);
    for(int i = 0; i < pairs; i++){
        int e = b.hostIds[i];
        if(e >= 0 && e < s.experts) byExpert[e].push_back(i);
    }
    int* groupSlot = b.hostGroups;
    int* groupStart = groupSlot + s.experts;
    int* groupCount = groupStart + s.experts;
    int* inputIndex = groupCount + s.experts;
    int* tokenOf = inputIndex + b.pairsMost;
    float* weightOf = (float*)(tokenOf + b.pairsMost);
    float* scaleOf = weightOf + b.pairsMost;
    int groups = 0, c = 0, widest = 0;
    for(int e = 0; e < s.experts; e++){
        if(byExpert[e].empty()) continue;
        groupSlot[groups] = e;
        groupStart[groups] = c;
        groupCount[groups] = (int)byExpert[e].size();
        widest = std::max(widest, groupCount[groups]);
        for(int pair : byExpert[e]){
            inputIndex[c] = pair / used;
            tokenOf[c] = pair / used;
            weightOf[c] = b.hostWeights[pair] * dScale[e];
            scaleOf[c] = guScale[e];
            c++;
        }
        groups++;
    }
    uint64_t slotOn = b.groupsOn, startOn = slotOn + 4ULL * s.experts, countOn = startOn + 4ULL * s.experts, inputOn = countOn + 4ULL * s.experts;
    uint64_t tokenOn = inputOn + 4ULL * b.pairsMost, weightOn = tokenOn + 4ULL * b.pairsMost, scaleOn = weightOn + 4ULL * b.pairsMost;
    go("hyFill", blocks((uint64_t)n * d), 1, 256, 1, b.moe, n * d);
    go("quantizeInput", (unsigned int)(((uint64_t)n * d / 32 + 7) / 8), 1, 32, 8, b.xn, n * d, b.inValues, b.inScales, b.inSums);
    bool wide = c >= wideLeast * groups;
    int tile = tileWidth(wide);
    int tiles = (widest + tile - 1) / tile;
    unsigned int threadsUsed = tile == 128 ? 256 : 128;
    int guTile = rowTile(L.gateUp.type, wide, 0), downTile = rowTile(L.down.type, wide, 0);
    go(tiled(L.gateUp.type, wide, 0).c_str(), (unsigned int)((2 * width + guTile - 1) / guTile), (unsigned int)(groups * tiles), threadsUsed, 1, (int)L.gateUp.type, L.gateUp.data, L.gateUp.stride, L.gateUpBytes, slotOn, startOn, countOn, inputOn, b.inValues, b.inScales, b.inSums, none, none, none, b.guOut, 2 * width, d, 0, tiles, none, none, none, tables);
    go("inkAct", (unsigned int)(((uint64_t)c * width / 32 + 7) / 8), 1, 32, 8, b.guOut, scaleOn, b.act, c, width, s.gateCap, s.upCap, b.hidValues, b.hidScales, b.hidSums);
    go(tiled(L.down.type, wide, 0).c_str(), (unsigned int)((d + downTile - 1) / downTile), (unsigned int)(groups * tiles), threadsUsed, 1, (int)L.down.type, L.down.data, L.down.stride, L.downBytes, slotOn, startOn, countOn, none, b.hidValues, b.hidScales, b.hidSums, none, none, none, none, d, width, 0, tiles, tokenOn, weightOn, b.moe, tables);
}

inline void GpuModel::inkLayerRun(int l, int r, uint64_t h, uint64_t next, int count, int boundary, int n, bool messages) {
    InkShape& s = inkShape;
    InkBuffers& b = inkBuffers;
    InkLayer& L = inkLayers[l];
    InkRun& R = inkRuns[r];
    int d = s.hidden;
    unsigned int spread = (unsigned int)n;
    auto blocks = [](uint64_t c){ return (unsigned int)((c + 255) / 256); };
    PackedInput& pk = packs[0];
    uint64_t none = 0;
    static const bool debug = getenv("SNEO_INK_DEBUG") != nullptr;
    auto check = [&](const char* where){
        if(!debug || failed) return;
        if(waitStream() != 0){
            std::cout << "inkling2_loop: cuda failed after " << where << " in layer " << l << " run " << r << " with " << n << " rows" << std::endl;
            failed = true;
        }
    };
    uint64_t st[5] = {b.blocks[0], b.blocks[1], b.blocks[2], b.blocks[3], none};
    auto gate = [&](uint64_t from, int k, uint64_t into){
        go("inkMsgDown", spread, (unsigned int)((s.rank + 3) / 4), 128, 1, from, b.msgs[k], L.ctrlDown[k], b.gates, d, s.rank, s.eps);
        go("inkMsgMix", (unsigned int)((d + 7) / 8), spread, 256, 1, from, b.msgs[k], b.gates, L.ctrlForget[k], L.ctrlGain[k], into, d, s.rank);
    };
    uint64_t hq = b.hq, hk = b.hq, hv = b.hq;
    if(L.full){
        if(l > 0) go("inkBlockRes", spread, 1, 256, 1, st[0], st[1], st[2], st[3], h, count, L.resQuery, 3, b.hq, b.hk, b.hv, none, d, s.eps);
        else {
            go("inkCopy", blocks((uint64_t)n * d), 1, 256, 1, h, b.hq, n * d);
            go("inkCopy", blocks((uint64_t)n * d), 1, 256, 1, h, b.hk, n * d);
            go("inkCopy", blocks((uint64_t)n * d), 1, 256, 1, h, b.hv, n * d);
        }
        if(messages){
            gate(b.hq, 0, b.xin);
            go("inkCopy", blocks((uint64_t)n * d), 1, 256, 1, b.xin, b.hq, n * d);
            gate(b.hk, 1, b.xin);
            go("inkCopy", blocks((uint64_t)n * d), 1, 256, 1, b.xin, b.hk, n * d);
            gate(b.hv, 2, b.xin);
            go("inkCopy", blocks((uint64_t)n * d), 1, 256, 1, b.xin, b.hv, n * d);
        }
        hyNormRows(b.hq, d, b.hq, d, L.qNorm, 0, n, 1, d, s.eps);
        hyNormRows(b.hk, d, b.hk, d, L.kNorm, 0, n, 1, d, s.eps);
        hyNormRows(b.hv, d, b.hv, d, L.vNorm, 0, n, 1, d, s.eps);
        hk = b.hk;
        hv = b.hv;
    } else {
        if(l > 0) go("inkBlockRes", spread, 1, 256, 1, st[0], st[1], st[2], st[3], h, count, L.resQuery, 1, b.xin, none, none, none, d, s.eps);
        else go("inkCopy", blocks((uint64_t)n * d), 1, 256, 1, h, b.xin, n * d);
        if(messages){
            gate(b.xin, 0, b.hq);
            gate(b.xin, 1, b.hk);
            gate(b.xin, 2, b.hv);
            hyNormRows(b.hq, d, b.hq, d, L.norm, 0, n, 1, d, s.eps);
            hyNormRows(b.hk, d, b.hk, d, L.norm, 0, n, 1, d, s.eps);
            hyNormRows(b.hv, d, b.hv, d, L.norm, 0, n, 1, d, s.eps);
            hk = b.hk;
            hv = b.hv;
        } else hyNormRows(b.xin, d, b.hq, d, L.norm, 0, n, 1, d, s.eps);
    }
    check("the attention input");
    inkProject(L.q, hq, b.q, n);
    inkProject(L.rel, hq, b.rel, n);
    inkProject(L.gate, hq, b.gateRaw, n);
    inkProject(L.sink, hq, b.sink, n);
    inkProject(L.k, hk, b.k, n);
    if(L.full) inkProject(L.kProj, hk, b.dynK, n);
    inkProject(L.v, hv, b.v, n);
    if(L.full) inkProject(L.vProj, hv, b.dynV, n);
    check("the projections");
    go("inkClip", spread * 24, 1, 128, 1, b.q, L.qClip, b.positions, 24, 128, s.qkClip, L.full ? 1 : 0, s.logAlpha, s.logFloor);
    go("inkClip", spread * 24, 1, 32, 1, b.rel, L.relClip, b.positions, 24, 16, s.relClip, L.full ? 1 : 0, s.logAlpha, s.logFloor);
    go("inkConv", blocks(1024), 1, 256, 1, b.k, L.kConv, L.full ? b.dynK : none, R.kHist, n, 1024, 128);
    go("inkConv", blocks(1024), 1, 256, 1, b.v, L.vConv, L.full ? b.dynV : none, R.vHist, n, 1024, 128);
    check("the clips and convolutions");
    go("inkStore", spread * 8, 1, 128, 1, b.k, b.v, R.kCache, R.vCache, b.positions, R.ring, 8, s.qkClip, s.eps);
    check("the cache store");
    int window = L.full ? 0 : s.window;
    int reach = L.full ? inkTop + 1 : std::min(s.window, inkTop + 1);
    float scale = 1.0f / sqrtf(128.0f);
    for(int first = 0; first < n && !failed; first += s.attendRows){
        int m = std::min(s.attendRows, n - first);
        int most = (int)std::max<uint64_t>(1, b.scratchFloats / ((uint64_t)m * 24 * 130));
        int span = std::max(16, (reach + most - 1) / most);
        int splits = std::max(1, (reach + span - 1) / span);
        go("inkAttendPart", 3, (unsigned int)(m * splits), 256, 1, b.q + 4ULL * first * 3072, R.kCache, R.vCache, R.ring, b.positions + 4ULL * first, b.rel + 4ULL * first * 384, L.relBank, b.scratch, 24, 8, window, s.extent, scale, span, splits);
        go("inkAttendMerge", 24, (unsigned int)m, 128, 1, b.scratch, b.sink + 4ULL * first * 24, b.attn + 4ULL * first * 3072, 24, splits);
    }
    check("the attention");
    go("hyGateMulPack", (unsigned int)(((uint64_t)n * 3072 / 32 + 7) / 8), 1, 32, 8, b.attn, b.gateRaw, n * 3072, pk.values, pk.scales, pk.sums);
    project(L.out, b.attn, 0, b.proj, n);
    go("inkConv", blocks(d), 1, 256, 1, b.proj, L.postConv, none, R.postHist, n, d, d);
    check("the output projection");
    int inner = count;
    if(boundary >= 0){
        go("inkCopy", blocks((uint64_t)n * d), 1, 256, 1, b.proj, b.resid, n * d);
        inner = count + 1;
    } else go("inkAdd", blocks((uint64_t)n * d), 1, 256, 1, h, b.proj, b.resid, n * d);
    go("inkBlockRes", spread, 1, 256, 1, st[0], st[1], st[2], st[3], b.resid, inner, L.ffnQuery, 1, b.mlpIn, none, none, none, d, s.eps);
    if(messages){
        gate(b.mlpIn, 3, b.xin);
        hyNormRows(b.xin, d, b.xn, d, L.ffnNorm, 0, n, 1, d, s.eps);
    } else hyNormRows(b.mlpIn, d, b.xn, d, L.ffnNorm, 0, n, 1, d, s.eps);
    check("the mlp input");
    go("hyF32Wide", (unsigned int)s.experts, spread, 128, 1, L.router, b.xn, d, b.logits, s.experts, d);
    go("hyRouteFast", spread, 1, 512, 1, b.logits, L.routerBias, b.ids, b.weights, s.experts, s.used, s.routeScale, 1);
    check("the router");
    inkExperts(L, n);
    check("the experts");
    go("inkConv", blocks(d), 1, 256, 1, b.moe, L.ffnConv, none, R.ffnHist, n, d, d);
    go("inkAdd", blocks((uint64_t)n * d), 1, 256, 1, b.resid, b.moe, next, n * d);
}

inline bool GpuModel::inkChunk(const int* ids, const int* positions, int n, bool last) {
    InkShape& s = inkShape;
    InkBuffers& b = inkBuffers;
    CudaQueue& queue = cudaQueue();
    int d = s.hidden;
    auto blocks = [](uint64_t c){ return (unsigned int)((c + 255) / 256); };
    uint64_t none = 0;
    if(waitStream() != 0) failed = true;
    pool.run(n, [&](int t){
        int id = ids[t];
        if(id < 0 || id >= embed.rows) id = 0;
        dequantRow(embed.type, embed.data + (uint64_t)id * embed.stride, b.hostEmbed + (size_t)t * d, d);
    });
    inkTop = 0;
    for(int t = 0; t < n; t++){
        b.hostPositions[t] = positions[t];
        inkTop = std::max(inkTop, positions[t]);
    }
    if(queue.copyUp(b.positions, b.hostPositions, (size_t)n * 4, stream) != 0) failed = true;
    static const bool debugging = getenv("SNEO_INK_DEBUG") != nullptr;
    bool graphed = useGraphs && !profile && !debugging && n <= s.fewRows;
    int reachLog = 8;
    if(graphed){
        while((1 << reachLog) < inkTop + 1 && (1 << reachLog) < contextLimit) reachLog++;
        inkTop = std::min(1 << reachLog, contextLimit) - 1;
    }
    auto work = [&]{
        hyNormRows(b.mappedEmbed, d, b.h[0], d, b.embNorm, 0, n, 1, d, s.eps);
        int at = 0, count = 0, r = 0;
        auto step = [&](int l, bool messages){
            int boundary = l % s.block == 0 ? l / s.block : -1;
            if(boundary >= 0) go("inkCopy", blocks((uint64_t)n * d), 1, 256, 1, b.h[at], b.blocks[boundary], n * d);
            inkLayerRun(l, r, b.h[at], b.h[1 - at], count, boundary, n, messages);
            if(boundary >= 0) count = boundary + 1;
            at = 1 - at;
            r++;
        };
        for(int l = 0; l < s.loopStart && !failed; l++) step(l, false);
        int preCount = count;
        for(int l = s.loopStart; l < s.loopEnd && !failed; l++) step(l, false);
        for(int pass = 1; pass < s.passes && !failed; pass++){
            go("inkCopy", blocks((uint64_t)n * d), 1, 256, 1, b.blocks[s.loopStart / s.block + 1], b.prevBlock, n * d);
            go("inkCopy", blocks((uint64_t)n * d), 1, 256, 1, b.h[at], b.prevH, n * d);
            go("inkCopy", blocks((uint64_t)n * d), 1, 256, 1, b.blocks[s.loopStart / s.block], b.h[at], n * d);
            count = preCount;
            for(int l = s.loopStart; l < s.loopEnd && !failed; l++){
                go("inkBlockRes", (unsigned int)n, 1, 256, 1, b.prevBlock, none, none, none, b.prevH, 1, inkLayers[l].msgQuery, 4, b.msgs[0], b.msgs[1], b.msgs[2], b.msgs[3], d, s.eps);
                step(l, true);
            }
        }
        for(int l = s.loopEnd; l < s.layers && !failed; l++) step(l, false);
        if(!last || failed) return;
        uint64_t lastRow = b.h[at] + 4ULL * (n - 1) * d;
        uint64_t st[4];
        for(int k = 0; k < 4; k++) st[k] = b.blocks[k] + 4ULL * (n - 1) * d;
        go("inkBlockRes", 1, 1, 256, 1, st[0], st[1], st[2], st[3], lastRow, count, b.outQuery, 1, b.xin, none, none, none, d, s.eps);
        hyNormRows(b.xin, d, b.last, d, b.outNorm, 0, 1, 1, d, s.eps);
        inkProject(head, b.last, b.vocab, 1);
        go("inkHead", blocks(s.vocab), 1, 256, 1, b.vocab, s.vocab, s.unpadded, s.softcap);
        int keep = shortlist > 0 ? shortlist : 0;
        if(keep > 0) go("shortRows", 1, 1, 512, 1, b.vocab, s.vocab, keep, b.mappedLogits);
        else go("hyCopy", blocks(s.vocab), 1, 256, 1, b.vocab, b.mappedLogits, s.vocab);
    };
    if(!graphed || failed){
        work();
        return !failed;
    }
    int key = ((reachLog * 32 + n) * 2 + (last ? 1 : 0)) * 65536 + std::min(std::max(shortlist, 0), 65535);
    if(!graphless.count(key) && replay(tokenGraphs[key], work)) return !failed;
    if(graphless.insert(key).second && !graphProblem.empty()) std::cout << "inkling2_loop: " << n << " rows run without a graph: " << graphProblem << std::endl;
    work();
    return !failed;
}

inline std::vector<float> GpuModel::forwardInk(const std::vector<int>& ids, const std::vector<int>& positions) {
    std::vector<float> logits;
    InkShape& s = inkShape;
    InkBuffers& b = inkBuffers;
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
    if(everyRow){
        problem = "inkling2_loop does not check drafted rows yet";
        return logits;
    }
    int start = positions[0];
    if(start != (int)tokens.size()){
        if(start != 0){
            problem = "inkling2_loop keeps a running state and can only continue at position " + std::to_string(tokens.size()) + ", not " + std::to_string(start);
            return logits;
        }
        tokens.clear();
        inkReset();
    }
    failed = false;
    auto clock = std::chrono::steady_clock::now();
    for(size_t at = 0; at < ids.size() && !failed && !stopGenerating;){
        int n = (int)std::min(ids.size() - at, (size_t)chunk);
        tokens.insert(tokens.end(), ids.begin() + at, ids.begin() + at + n);
        inkChunk(ids.data() + at, positions.data() + at, n, at + n == ids.size());
        at += n;
    }
    int code = waitStream();
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
    logits.assign(b.hostLogits, b.hostLogits + width);
    static const char* dump = getenv("SNEO_INK_DUMP");
    if(dump && ids.size() > 1){
        std::vector<float> all(s.vocab);
        FILE* f = fopen(dump, "wb");
        if(f && cudaLink().copyDown(all.data(), b.vocab, all.size() * 4) == 0){
            int count = (int)ids.size();
            fwrite(&count, 4, 1, f);
            fwrite(ids.data(), 4, ids.size(), f);
            fwrite(all.data(), 4, all.size(), f);
        }
        if(f) fclose(f);
    }
    headMs = since(clock);
    for(size_t i = 0; i < markUsed; i++){
        float ms = 0;
        cudaQueue().eventElapsed(&ms, marks[i].second.first, marks[i].second.second);
        kernelMs[marks[i].first] += ms;
        kernelCalls[marks[i].first]++;
    }
    markUsed = 0;
    static int profiled = 0;
    if(profile && (++profiled % 64 == 0 || ids.size() > 1)){
        std::vector<std::pair<double, std::string>> ranked;
        for(auto& k : kernelMs) ranked.push_back({k.second, k.first});
        std::sort(ranked.rbegin(), ranked.rend());
        double all = 0;
        for(auto& k : ranked) all += k.first;
        std::cout << "inkling2_loop kernels over " << (ids.size() > 1 ? 1 : 64) << " calls with " << ids.size() << " rows, " << (int)all << " ms in all:";
        for(size_t i = 0; i < ranked.size() && i < 24; i++) std::cout << " | " << ranked[i].second << " " << (int)ranked[i].first << " ms x" << kernelCalls[ranked[i].second];
        std::cout << std::endl;
        kernelMs.clear();
        kernelCalls.clear();
    }
    return logits;
}
