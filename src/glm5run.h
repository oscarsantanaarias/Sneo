#pragma once

inline void GpuModel::glmHcIn(uint64_t fn, uint64_t scale, uint64_t base, uint64_t norm, int n) {
    HyShape& s = hyShape;
    HyBuffers& b = hyBuffers;
    int width = s.hc * s.hidden, chunk = 512;
    int parts = (width + chunk - 1) / chunk;
    go("glmHcDots", (unsigned int)(n * parts), 1, 256, 1, b.streams, fn, b.mixes, width, chunk);
    PackedInput& p = packs[0];
    go("glmHcCombine", (unsigned int)n, 1, 512, 1, b.streams, b.mixes, scale, base, norm, b.xn, glmBuffers.hcw, s.hidden, s.eps, s.hcEps, glmShape.sinkhorn, p.values, p.scales, p.sums);
}

inline void GpuModel::glmHcOut(uint64_t x, int n) {
    HyShape& s = hyShape;
    int chunk = 256;
    int parts = (s.hidden + chunk - 1) / chunk;
    go("glmHcSpread", (unsigned int)(n * parts), 1, 256, 1, hyBuffers.streams, x, glmBuffers.hcw, s.hidden, chunk);
}

inline void GpuModel::glmPacked(const GpuMatrix& m, uint64_t input, uint64_t output, int n) {
    if(m.type == 0 || m.type == 30 || m.type == 1){
        hyProject(m, input, output, n);
        return;
    }
    project(m, input, 0, output, n);
}

inline bool GpuModel::loadGlm(const std::string& path, int threads) {
    glm = true;
    dense = true;
    usePle = false;
    streamPrefill = false;
    pinWarm = false;
    foresee = false;
    residual = false;
    firstLayer = 0;
    endLayer = 0;
    layers.clear();
    HyShape& s = hyShape;
    HyBuffers& b = hyBuffers;
    GlmShape& g = glmShape;
    GlmBuffers& gb = glmBuffers;
    std::string a = "glm5-next.";
    auto real = [&](const std::string& key, double fallback) {
        auto found = file.meta.find(a + key);
        return found == file.meta.end() ? fallback : found->second.real;
    };
    auto list = [&](const std::string& key) {
        auto found = file.meta.find(a + key);
        return found == file.meta.end() ? std::vector<int64_t>() : found->second.wholes;
    };
    auto reals = [&](const std::string& key) {
        auto found = file.meta.find(a + key);
        return found == file.meta.end() ? std::vector<double>() : found->second.reals;
    };
    s.layers = (int)file.layers;
    s.hidden = (int)metaWhole(file, a + "embedding_length", 0);
    s.hc = (int)metaWhole(file, a + "hyper_connection.count", 4);
    s.qLora = (int)metaWhole(file, a + "attention.q_lora_rank", 0);
    s.kvLora = (int)metaWhole(file, a + "attention.kv_lora_rank", 0);
    s.rope = 64;
    s.nope = (int)metaWhole(file, a + "attention.key_length_mla", 256);
    s.headValue = (int)metaWhole(file, a + "attention.value_length_mla", 256);
    s.experts = (int)metaWhole(file, a + "expert_count", 0);
    s.used = (int)metaWhole(file, a + "expert_used_count", 0);
    s.latent = s.hidden;
    s.expertWidth = (int)metaWhole(file, a + "expert_feed_forward_length", 0);
    s.feedForward = (int)metaWhole(file, a + "feed_forward_length", 0);
    s.eps = (float)real("attention.layer_norm_rms_epsilon", 1e-5);
    s.hcEps = (float)real("hyper_connection.epsilon", 1e-6);
    s.expertScale = (float)real("expert_weights_scale", 1.0);
    s.expertNorm = metaWhole(file, a + "expert_weights_norm", 1) != 0;
    g.sinkhorn = (int)metaWhole(file, a + "hyper_connection.sinkhorn_iterations", 20);
    g.linearDim = (int)metaWhole(file, a + "kda.head_dim", 128);
    g.convTaps = (int)metaWhole(file, a + "ssm.conv_kernel", 4);
    g.lowerBound = (float)real("kda.gate_lower_bound", -5.0);
    g.headKey = s.nope;
    int gating = (int)metaWhole(file, a + "expert_gating_func", 2);
    int denseLead = (int)metaWhole(file, a + "leading_dense_block_count", 3);
    int heads = (int)metaWhole(file, a + "attention.head_count", 64);
    std::vector<int64_t> pattern = list("attention.head_count_kv");
    std::vector<double> clamps = reals("swiglu_clamp_exp");
    int ropeDims = (int)metaWhole(file, a + "rope.dimension_count", 0);
    if(s.layers <= 0 || s.hidden != 4096 || s.hc != 4 || (int)pattern.size() < s.layers){
        problem = "unexpected glm5-next shape";
        return false;
    }
    if(s.kvLora != 512 || ropeDims != 0 || s.nope != 256 || s.headValue != 256 || s.used > 32 || s.experts > 1024 || gating != 2 || s.expertWidth % 256 != 0 || g.linearDim != 128 || g.convTaps != 4 || heads % 8 != 0 || heads > 128){
        problem = "this glm5-next layout is not the one the engine runs (latent 512 without rope, kda heads of 128, sigmoid routing)";
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
    if(!buildModule(sizes + quantKernelSource + modelKernelSource + gemmaKernelSource + hyKernelSource + glmKernelSource, module)){
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
        out.rows = (int)t->dims[1];
        out.stride = rowBytes(out.type, t->dims[0]);
        wanted.push_back({t, &out.data});
    };
    auto floats = [&](const std::string& name, uint64_t& out) {
        const TensorInfo* t = findTensor(file, name);
        if(!t){
            if(problem.empty()) problem = "missing tensor " + name;
            return;
        }
        if(t->type == 0){
            wanted.push_back({t, &out});
            return;
        }
        if(!knownType(t->type)){
            if(problem.empty()) problem = name + " is " + typeInfo(t->type).name + ", a tensor type the engine cannot read yet";
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
    matrix("output.weight", head);
    floats("output_norm.weight", b.outNorm);

    const char* rowsAsked = getenv("SNEO_GLM_ROWS");
    chunk = rowsAsked ? std::max(64, atoi(rowsAsked)) / 64 * 64 : 1024;
    hyLayers.assign(s.layers, HyLayer());
    g.linear.assign(s.layers, 0);
    g.linearSlot.assign(s.layers, -1);
    glmLinears.clear();
    int linearCount = 0;
    for(int l = 0; l < s.layers; l++){
        if(pattern[l] == 0) g.linearSlot[l] = linearCount++;
        g.linear[l] = pattern[l] == 0 ? 1 : 0;
    }
    glmLinears.assign(linearCount, GlmLinear());
    gb.idxGate.assign(s.layers, GpuMatrix());
    gb.idxGateRing.assign(s.layers, 0);
    s.idxKeep = 512;
    s.idxRing = chunk + 64;
    s.idxWidth = 2052;
    s.idxScale = 1.0f / 64.0f;
    for(int l = 0; l < s.layers; l++){
        HyLayer& L = hyLayers[l];
        std::string p = "blk." + std::to_string(l) + ".";
        L.heads = heads;
        L.dense = l < denseLead;
        L.clamp = l < (int)clamps.size() ? (float)clamps[l] : 0.0f;
        s.widestHeads = std::max(s.widestHeads, L.heads);
        floats(p + "attn_norm.weight", L.attnNorm);
        floats(p + "ffn_norm.weight", L.ffnNorm);
        floats(p + "hc_attn_fn.weight", L.hcAttnFn);
        floats(p + "hc_attn_base.weight", L.hcAttnBase);
        floats(p + "hc_attn_scale.weight", L.hcAttnScale);
        floats(p + "hc_ffn_fn.weight", L.hcFfnFn);
        floats(p + "hc_ffn_base.weight", L.hcFfnBase);
        floats(p + "hc_ffn_scale.weight", L.hcFfnScale);
        if(g.linear[l]){
            GlmLinear& K = glmLinears[g.linearSlot[l]];
            matrix(p + "attn_q.weight", K.q);
            matrix(p + "attn_k.weight", K.k);
            matrix(p + "attn_v.weight", K.v);
            matrix(p + "attn_output.weight", K.out);
            matrix(p + "ssm_beta.weight", K.beta);
            matrix(p + "ssm_f_a.weight", K.fa);
            matrix(p + "ssm_f_b.weight", K.fb);
            matrix(p + "ssm_g_a.weight", K.ga);
            matrix(p + "ssm_g_b.weight", K.gb);
            floats(p + "ssm_conv1d_q.weight", K.convQ);
            floats(p + "ssm_conv1d_k.weight", K.convK);
            floats(p + "ssm_conv1d_v.weight", K.convV);
            floats(p + "ssm_dt.bias", K.dtBias);
            floats(p + "ssm_norm.weight", K.norm);
            if(problem.empty() && (K.q.rows != g.linearWidth || K.q.cols != s.hidden || K.out.cols != g.linearWidth || K.beta.rows != g.linearHeads || K.fb.rows != g.linearWidth || K.gb.rows != g.linearWidth)){
                problem = "unexpected glm5-next linear attention shapes in layer " + std::to_string(l);
                return false;
            }
        } else {
            matrix(p + "attn_q_a.weight", L.qa);
            floats(p + "attn_q_a_norm.weight", L.qaNorm);
            matrix(p + "attn_q_b.weight", L.qb);
            matrix(p + "attn_kv_a_mqa.weight", L.kva);
            floats(p + "attn_kv_a_norm.weight", L.kvNorm);
            matrix(p + "attn_output.weight", L.out);
            if(findTensor(file, p + "indexer.attn_k.weight") && !getenv("SNEO_GLM_NO_INDEX")){
                L.indexed = true;
                s.indexed = true;
                matrix(p + "indexer.attn_k.weight", L.idxK);
                matrix(p + "indexer.attn_q_b.weight", L.idxQ);
                matrix(p + "indexer_compressor_gate.weight", gb.idxGate[l]);
                floats(p + "indexer.k_norm.weight", L.idxKNorm);
                floats(p + "indexer.k_norm.bias", L.idxNorm);
                floats(p + "indexer.proj.weight", L.idxProj);
                floats(p + "indexer_compressor_ape.weight", L.idxApe);
                if(problem.empty() && (L.idxK.rows != 128 || L.idxK.cols != s.hidden || L.idxQ.rows != 4096 || L.idxQ.cols != s.qLora || gb.idxGate[l].rows != 128)){
                    problem = "unexpected glm5-next indexer shapes in layer " + std::to_string(l);
                    return false;
                }
            }
            if(problem.empty() && (L.qa.rows != s.qLora || L.qb.rows != L.heads * s.nope || L.kva.rows != s.kvLora || L.out.cols != L.heads * s.headValue)){
                problem = "unexpected glm5-next attention shapes in layer " + std::to_string(l);
                return false;
            }
        }
        if(L.dense){
            matrix(p + "ffn_gate.weight", L.ffnGate);
            matrix(p + "ffn_up.weight", L.ffnUp);
            matrix(p + "ffn_down.weight", L.ffnDown);
        } else {
            floats(p + "ffn_gate_inp.weight", L.router);
            floats(p + "exp_probs_b.bias", L.routerBias);
            matrix(p + "ffn_gate_shexp.weight", L.sharedGate);
            matrix(p + "ffn_up_shexp.weight", L.sharedUp);
            matrix(p + "ffn_down_shexp.weight", L.sharedDown);
            const TensorInfo* gt = findTensor(file, p + "ffn_gate_exps.weight");
            const TensorInfo* ut = findTensor(file, p + "ffn_up_exps.weight");
            const TensorInfo* dt = findTensor(file, p + "ffn_down_exps.weight");
            if(!gt || !ut || !dt || !hostDotType(gt->type) || !hostDotType(ut->type) || !hostDotType(dt->type) || (int)gt->dims[2] != s.experts){
                problem = "glm5-next layer " + std::to_string(l) + " has experts the cpu cannot read";
                return false;
            }
            L.hostGate = {host(gt), gt->type, (int)gt->dims[0], (int)gt->dims[1], rowBytes(gt->type, gt->dims[0])};
            L.hostUp = {host(ut), ut->type, (int)ut->dims[0], (int)ut->dims[1], rowBytes(ut->type, ut->dims[0])};
            L.hostDown = {host(dt), dt->type, (int)dt->dims[0], (int)dt->dims[1], rowBytes(dt->type, dt->dims[0])};
            L.gateBytes = (uint64_t)L.hostGate.rows * L.hostGate.stride;
            L.upBytes = (uint64_t)L.hostUp.rows * L.hostUp.stride;
            L.downBytes = (uint64_t)L.hostDown.rows * L.hostDown.stride;
            if(L.hostGate.cols != s.latent || L.hostGate.rows != s.expertWidth || L.hostDown.cols != s.expertWidth || L.hostDown.rows != s.latent){
                problem = "unexpected glm5-next expert shapes in layer " + std::to_string(l);
                return false;
            }
        }
        if(!problem.empty()) return false;
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

    uint64_t perHeadBytes = 0;
    for(int l = 0; l < s.layers; l++){
        if(!g.linear[l]) perHeadBytes += (uint64_t)hyLayers[l].heads * ((uint64_t)s.kvLora * s.nope + (uint64_t)s.headValue * s.kvLora) * 2;
    }
    uint64_t widenedBytes = 0;
    for(auto& w : widened) widenedBytes += (w.first->dims[0] * std::max<uint64_t>(1, w.first->dims[1]) * 4 + 255) / 256 * 256;
    for(auto& w : wanted) denseBytes += (w.first->bytes + 255) / 256 * 256;
    denseBytes += perHeadBytes + widenedBytes + (uint64_t)linearCount * 256;

    uint64_t C = (uint64_t)chunk;
    uint64_t d = (uint64_t)s.hidden;
    uint64_t H = (uint64_t)s.widestHeads;
    uint64_t W = (uint64_t)g.linearWidth;
    uint64_t widestInput = std::max<uint64_t>({d * s.hc, H * s.headValue, (uint64_t)s.feedForward, W});
    uint64_t stateBytes = (uint64_t)linearCount * ((uint64_t)g.linearHeads * g.linearDim * g.linearDim * 4 + 3 * 3 * W * 4);
    uint64_t A = (uint64_t)g.attendRows;
    uint64_t hitPairs = (uint64_t)g.hitRows * s.used;
    uint64_t hitWidest = std::max<uint64_t>(s.latent, s.expertWidth);
    gb.pairsMost = 8192;
    uint64_t P = (uint64_t)gb.pairsMost, ew = (uint64_t)s.expertWidth;
    uint64_t slotBytes = 0;
    for(const HyLayer& L : hyLayers){
        if(!L.dense) slotBytes = std::max(slotBytes, (L.gateBytes + 255) / 256 * 256 + (L.upBytes + 255) / 256 * 256 + (L.downBytes + 255) / 256 * 256);
    }
    gb.stageSlots = getenv("SNEO_GLM_STAGE") ? std::max(2, atoi(getenv("SNEO_GLM_STAGE"))) / 2 * 2 : 64;
    uint64_t bufferBytes = C * 4 * (d * s.hc + 32 + 20 + d * 4 + s.qLora + H * s.nope + 64 + 512 + 576 + H * s.headValue
        + 3 * (uint64_t)s.feedForward + 3 * (uint64_t)s.expertWidth + (uint64_t)s.experts + 7 * W + 2 * 128 + 64 + 4096 + 32) + A * 4 * (H * 576 + H * 512 + (uint64_t)s.idxGroups + 2052)
        + d * 4 + (uint64_t)s.vocab * 4 + std::max<uint64_t>(C * widestInput, hitPairs * hitWidest) * 2 * 2 + hitPairs * 4 * (2 * s.latent + 3 * ew)
        + P * ew * 4 * 3 + P * ew * 2 + C * d * 2 + (uint64_t)gb.stageSlots * slotBytes;
    uint64_t perToken = 0;
    for(int l = 0; l < s.layers; l++) perToken += g.linear[l] ? 0 : 576 * 2;
    uint64_t margin = 900ULL << 20;
    uint64_t fixedBytes = denseBytes + bufferBytes + stateBytes + (16ULL << 22) + margin;
    if(fixedBytes >= startFree){
        problem = "glm5-next needs " + std::to_string((fixedBytes >> 20) + 1) + " MB of video memory before any context and " + std::to_string(startFree >> 20) + " MB are free";
        return false;
    }
    uint64_t room = perToken > 0 ? (startFree - fixedBytes) / perToken : (uint64_t)contextLimit;
    if((uint64_t)contextLimit > room){
        int fitted = (int)(room / 256 * 256);
        if(fitted < 4096){
            problem = "glm5-next fits only " + std::to_string(fitted) + " tokens of context in the free video memory";
            return false;
        }
        contextLimit = fitted;
    }

    loadSay("Copying the attention and router weights to the gpu");
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
        uint64_t rows = std::max<uint64_t>(1, t->dims[1]);
        uint64_t stride = rowBytes(t->type, cols);
        std::vector<float> full((size_t)(rows * cols));
        pool.run((int)((rows + 15) / 16), [&](int part){
            for(uint64_t r = (uint64_t)part * 16; r < std::min<uint64_t>(rows, (uint64_t)part * 16 + 16); r++) dequantRow(t->type, host(t) + r * stride, full.data() + r * cols, cols);
        });
        *w.second = arena + used;
        if(cudaLink().copyUp(arena + used, full.data(), full.size() * 4) != 0){
            problem = "cannot copy " + t->name + " to the gpu";
            return false;
        }
        used += (full.size() * 4 + 255) / 256 * 256;
    }
    for(int l = 0; l < s.layers && problem.empty(); l++){
        HyLayer& L = hyLayers[l];
        std::string p = "blk." + std::to_string(l) + ".";
        if(g.linear[l]){
            GlmLinear& K = glmLinears[g.linearSlot[l]];
            const TensorInfo* t = findTensor(file, p + "ssm_a");
            if(!t || t->type != 0 || (int)t->dims[0] != g.linearHeads){
                problem = "unexpected glm5-next ssm_a in layer " + std::to_string(l);
                return false;
            }
            std::vector<float> rate(g.linearHeads);
            const float* raw = (const float*)host(t);
            for(int h = 0; h < g.linearHeads; h++) rate[h] = -raw[h];
            K.decay = arena + used;
            if(cudaLink().copyUp(K.decay, rate.data(), rate.size() * 4) != 0){
                problem = "cannot copy the kda decay to the gpu";
                return false;
            }
            used += 256;
            continue;
        }
        const TensorInfo* kb = findTensor(file, p + "attn_k_b.weight");
        const TensorInfo* vb = findTensor(file, p + "attn_v_b.weight");
        if(!kb || !vb || (int)kb->dims[0] != s.nope || (int)kb->dims[1] != s.kvLora || (int)kb->dims[2] != L.heads || (int)vb->dims[0] != s.kvLora || (int)vb->dims[1] != s.headValue || (int)vb->dims[2] != L.heads){
            problem = "unexpected glm5-next k_b or v_b in layer " + std::to_string(l);
            return false;
        }
        for(int which = 0; which < 2; which++){
            const TensorInfo* t = which == 0 ? kb : vb;
            int cols = (int)t->dims[0];
            uint64_t rows = t->dims[1] * t->dims[2];
            uint64_t stride = rowBytes(t->type, cols);
            std::vector<uint16_t> halves((size_t)(rows * cols));
            pool.run((int)((rows + 63) / 64), [&](int part){
                std::vector<float> line(cols);
                for(uint64_t r = (uint64_t)part * 64; r < std::min<uint64_t>(rows, (uint64_t)part * 64 + 64); r++){
                    dequantRow(t->type, host(t) + r * stride, line.data(), cols);
                    for(int c = 0; c < cols; c++) halves[(size_t)(r * cols + c)] = (uint16_t)_mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(line[c]), 0), 0);
                }
            });
            uint64_t at = arena + used;
            used += (halves.size() * 2 + 255) / 256 * 256;
            if(cudaLink().copyUp(at, halves.data(), halves.size() * 2) != 0){
                problem = "cannot copy k_b or v_b to the gpu";
                return false;
            }
            (which == 0 ? L.kb : L.vb) = at;
        }
    }
    for(int l = 0; l < s.layers; l++){
        HyLayer& L = hyLayers[l];
        if(g.linear[l]) continue;
        L.slots = contextLimit;
        L.cache = grab((uint64_t)L.slots * 576 * 2);
        cacheBytes += (uint64_t)L.slots * 576 * 2;
    }
    s.idxGroups = contextLimit / 4 + 1;
    for(int l = 0; l < s.layers; l++){
        HyLayer& L = hyLayers[l];
        if(!L.indexed) continue;
        L.idxKeys = grab((uint64_t)s.idxRing * 128 * 4);
        gb.idxGateRing[l] = grab((uint64_t)s.idxRing * 128 * 4);
        L.idxPools = grab((uint64_t)s.idxGroups * 128 * 4);
        cacheBytes += (uint64_t)s.idxGroups * 128 * 4;
    }
    for(GlmLinear& K : glmLinears){
        K.state = grab((uint64_t)g.linearHeads * g.linearDim * g.linearDim * 4);
        K.history = grab(3 * 3 * W * 4);
    }

    b.streams = grab(C * d * s.hc * 4);
    b.mixes = grab(C * 32 * 4);
    if(b.mixes) cudaLink().memset(b.mixes, 0, (size_t)(C * 32 * 4));
    gb.hcw = grab(C * 20 * 4);
    b.xn = grab(C * d * 4);
    b.routedIn = b.xn;
    b.qa = grab(C * s.qLora * 4);
    b.q = grab(C * H * s.nope * 4 + 256);
    gb.kvRaw = grab(C * 512 * 4);
    b.kv = grab(C * 576 * 4);
    b.qfull = grab(A * H * 576 * 4);
    b.latentOut = grab(A * H * 512 * 4);
    b.heads = grab(C * H * s.headValue * 4);
    b.attn = grab(C * d * 4);
    b.bigGate = grab(C * s.feedForward * 4);
    b.bigUp = grab(C * s.feedForward * 4);
    b.bigHid = grab(C * s.feedForward * 4);
    b.ffn = grab(C * d * 4);
    b.shared = grab(C * d * 4);
    b.sGate = grab(C * s.expertWidth * 4);
    b.sUp = grab(C * s.expertWidth * 4);
    b.sHid = grab(C * s.expertWidth * 4);
    b.routedOut = grab(C * s.latent * 4);
    b.logits = grab(C * s.experts * 4);
    b.positions = grab(C * 4);
    if(s.indexed){
        b.idxKey = grab(C * 128 * 4);
        gb.idxGateRaw = grab(C * 128 * 4);
        b.idxQ = grab(C * 4096 * 4);
        b.idxW = grab(C * 32 * 4);
        b.idxScores = grab(A * s.idxGroups * 4);
        b.idxList = grab(A * s.idxWidth * 4);
        b.idxCount = grab(C * 4);
    }
    gb.lq = grab(C * W * 4);
    gb.lk = grab(C * W * 4);
    gb.lv = grab(C * W * 4);
    gb.lf = grab(C * W * 4);
    gb.lgate = grab(C * W * 4);
    gb.lo = grab(C * W * 4);
    gb.lon = grab(C * W * 4);
    gb.lfa = grab(C * 128 * 4);
    gb.lga = grab(C * 128 * 4);
    gb.lbeta = grab(C * 64 * 4);
    gb.sinks = grab(H * 4);
    if(gb.sinks){
        std::vector<float> none(H, -1e30f);
        cudaLink().copyUp(gb.sinks, none.data(), none.size() * 4);
    }
    b.last = grab(d * 4);
    b.vocab = grab((uint64_t)s.vocab * 4);
    b.everyCap = 8;
    b.scratchFloats = 16ULL << 20;
    b.attnScratch = grab(b.scratchFloats * 4);
    uint64_t pairsMost = hitPairs;
    b.hitIn = grab(pairsMost * s.latent * 4);
    b.hitGate = grab(pairsMost * s.expertWidth * 4);
    b.hitUp = grab(pairsMost * s.expertWidth * 4);
    b.hitHid = grab(pairsMost * s.expertWidth * 4);
    b.hitDown = grab(pairsMost * s.latent * 4);
    b.hostHits = (int*)shared(pairsMost * 4 * 4, b.mappedHits);
    for(int slot = 0; slot < 2; slot++){
        PackedInput* packed = &packs[slot];
        int capacity = (int)std::max<uint64_t>(C * widestInput, pairsMost * hitWidest);
        capacity = (capacity + 255) / 256 * 256;
        if(cudaLink().alloc(&packed->values, capacity) != 0 || cudaLink().alloc(&packed->scales, capacity / 8) != 0 || cudaLink().alloc(&packed->sums, capacity / 8) != 0){
            problem = "out of video memory for packed inputs";
            return false;
        }
        packed->capacity = capacity;
    }
    b.hostEmbed = (float*)shared(C * d * 4, b.mappedEmbed);
    b.hostLogits = (float*)shared((uint64_t)s.vocab * 4, b.mappedLogits);
    b.hostLatent = (float*)shared(C * s.latent * 4, b.mappedLatent);
    b.hostMoe = (float*)shared(C * s.latent * 4, b.mappedMoe);
    b.hostIds = (int*)shared(C * s.used * 4, b.mappedIds);
    b.hostWeights = (float*)shared(C * s.used * 4, b.mappedWeights);
    void* hostPos = nullptr;
    if(cudaLink().hostAlloc(&hostPos, (size_t)(C * 4), 0) == 0){
        pinned.push_back(hostPos);
        b.hostPositions = (int*)hostPos;
    }
    if(!problem.empty() || !b.hostEmbed || !b.hostLogits || !b.hostLatent || !b.hostMoe || !b.hostIds || !b.hostWeights || !b.hostPositions || !b.hostHits){
        if(problem.empty()) problem = "cannot pin host memory for glm5-next";
        return false;
    }
    if(queue.streamCreate(&stream, 0) != 0){
        problem = "cannot create a cuda stream";
        return false;
    }
    gb.inValues = grab(C * d);
    gb.inScales = grab(C * d / 8);
    gb.inSums = grab(C * d / 8);
    gb.gateOut = grab(P * ew * 4);
    gb.upOut = grab(P * ew * 4);
    gb.hidOut = grab(P * ew * 4);
    gb.hidValues = grab(P * ew);
    gb.hidScales = grab(P * ew / 8);
    gb.hidSums = grab(P * ew / 8);
    gb.stage = grab((uint64_t)gb.stageSlots * slotBytes);
    gb.hostGroups = (int*)shared(2ULL * (192 + 3 * P) * 4, gb.groupsOn);
    void* pinnedStage = nullptr;
    if(problem.empty() && cudaLink().hostAlloc(&pinnedStage, (size_t)gb.stageSlots * slotBytes, 0) == 0){
        pinned.push_back(pinnedStage);
        gb.pinned = (uint8_t*)pinnedStage;
    }
    bool wideReady = gb.pinned && gb.hostGroups && queue.streamCreate(&gb.copyStream, 1) == 0;
    for(int h = 0; h < 2 && wideReady; h++) wideReady = queue.eventCreate(&gb.copied[h], 2) == 0 && queue.eventCreate(&gb.done[h], 2) == 0;
    if(!problem.empty() || !wideReady){
        if(problem.empty()) problem = "cannot set up the glm5-next prompt blocks";
        return false;
    }
    gb.saved.assign(4, std::vector<uint8_t>());
    gb.savedTokens.assign(4, 0);
    residentBytes = 0;
    residentExperts = 0;

    HyCache& cache = hyCache;
    for(const HyLayer& L : hyLayers){
        if(L.dense) continue;
        uint64_t one = (L.gateBytes + 255) / 256 * 256 + (L.upBytes + 255) / 256 * 256 + (L.downBytes + 255) / 256 * 256;
        cache.slotBytes = std::max(cache.slotBytes, one);
    }
    size_t leftFree = 0, leftTotal = 0;
    cudaLink().info(&leftFree, &leftTotal);
    uint64_t spare = 1200ULL << 20;
    if(cache.slotBytes && leftFree > spare){
        int want = (int)((leftFree - spare) / cache.slotBytes);
        want = std::min(want, s.layers * s.experts);
        while(want > 0){
            uint64_t ptr = 0;
            if(cudaLink().alloc(&ptr, (size_t)want * cache.slotBytes) == 0){
                cache.base = ptr;
                allocations.push_back(ptr);
                break;
            }
            want = want * 9 / 10;
        }
        cache.slots = cache.base ? want : 0;
    }
    cache.slotOf.assign((size_t)s.layers * s.experts, -1);
    cache.owner.assign(cache.slots, -1);
    cache.heat.assign((size_t)s.layers * s.experts, 0);
    cache.free.clear();
    for(int i = cache.slots - 1; i >= 0; i--) cache.free.push_back(i);
    cache.warmOf.assign((size_t)s.layers * s.experts, -1);
    cache.warmOwner.clear();
    cache.warmFree.clear();
    std::cout << "glm5-next: " << s.layers << " layers (" << linearCount << " linear), " << cache.slots << " expert slots in video memory (" << (((uint64_t)cache.slots * cache.slotBytes) >> 20) << " MB), the rest run on the cpu, context " << contextLimit << std::endl;
    size_t nowFree = 0, nowTotal = 0;
    cudaLink().info(&nowFree, &nowTotal);
    vramHeld = startFree > nowFree ? startFree - nowFree : 0;
    residentBytes = (uint64_t)cache.slots * cache.slotBytes;
    {
        std::error_code e;
        std::filesystem::path folder = std::filesystem::path(runtimeFolder()).parent_path() / "profiles";
        std::filesystem::create_directories(folder, e);
        std::string stem = std::filesystem::path(path).filename().string();
        size_t cut = stem.find("-00001-of-");
        if(cut != std::string::npos) stem = stem.substr(0, cut);
        if(stem.size() > 5 && stem.substr(stem.size() - 5) == ".gguf") stem = stem.substr(0, stem.size() - 5);
        cache.heatPath = (folder / (stem + ".glmheat")).string();
        cache.ageEvery = getenv("SNEO_GLM_AGE") ? atoi(getenv("SNEO_GLM_AGE")) : 64;
        uint32_t ceiling = cache.ageEvery > 0 ? 64 : 1024;
        FILE* f = fopen(cache.heatPath.c_str(), "rb");
        if(f){
            std::vector<uint32_t> saved(cache.heat.size());
            if(fread(saved.data(), 4, saved.size(), f) == saved.size()){
                uint32_t top = 0;
                for(uint32_t v : saved) top = std::max(top, v);
                double scale = top > ceiling ? (double)ceiling / top : 0.5;
                for(size_t i = 0; i < saved.size(); i++) cache.heat[i] = saved[i] ? (uint32_t)(saved[i] * scale) + 1 : 0;
            }
            fclose(f);
        }
        std::vector<int> order;
        for(size_t i = 0; i < cache.heat.size(); i++){
            if(cache.heat[i] && !hyLayers[i / s.experts].dense) order.push_back((int)i);
        }
        std::sort(order.begin(), order.end(), [&](int x, int y){ return cache.heat[x] > cache.heat[y]; });
        int place = std::min((int)order.size(), (int)cache.free.size());
        if(place > 0) loadSay("Placing the " + std::to_string(place) + " most used experts on the gpu");
        for(int i = 0; i < place && !failed; i++){
            int slot = cache.free.back();
            cache.free.pop_back();
            hyUpload(order[i] / s.experts, order[i] % s.experts, slot);
        }
        cache.uploads = 0;
        std::cout << "glm5-next: " << place << " experts placed in video memory from " << (order.empty() ? std::string("no heat map yet, it learns while you use it") : std::filesystem::path(cache.heatPath).filename().string()) << std::endl;
    }
    hyFillRam();
    cache.cpuPlain = !getenv("SNEO_HY_CPU_FINE");
    cache.cpuRows = getenv("SNEO_HY_CPU_ROWS") ? atoi(getenv("SNEO_HY_CPU_ROWS")) : 192;
    if(const char* asked = getenv("SNEO_HY_THREADS")) threads = std::max(1, atoi(asked));
    pool.start(std::max(0, threads - 1));
    if(cache.ageEvery > 0) hyStartUploads();
    std::cout << "glm5-next: " << std::max(1, threads) << " cpu threads for the experts" << std::endl;
    ready = true;
    reset();
    return true;
}

inline void GpuModel::glmReset() {
    GlmShape& g = glmShape;
    if(!ready || !useCuda()) return;
    for(GlmLinear& K : glmLinears){
        cudaLink().memset(K.state, 0, (size_t)g.linearHeads * g.linearDim * g.linearDim * 4);
        cudaLink().memset(K.history, 0, (size_t)3 * 3 * g.linearWidth * 4);
    }
}

inline bool GpuModel::moveGlm(int slot, bool keep) {
    GlmShape& g = glmShape;
    GlmBuffers& gb = glmBuffers;
    if(!ready || slot < 0 || slot >= 4 || !useCuda()) return false;
    uint64_t stateBytes = (uint64_t)g.linearHeads * g.linearDim * g.linearDim * 4;
    uint64_t historyBytes = 3ULL * 3 * g.linearWidth * 4;
    uint64_t total = (uint64_t)glmLinears.size() * (stateBytes + historyBytes);
    if(waitStream() != 0) return false;
    std::vector<uint8_t>& copy = gb.saved[slot];
    if(keep){
        copy.resize((size_t)total);
        uint64_t at = 0;
        for(GlmLinear& K : glmLinears){
            if(cudaLink().copyDown(copy.data() + at, K.state, (size_t)stateBytes) != 0) return false;
            at += stateBytes;
            if(cudaLink().copyDown(copy.data() + at, K.history, (size_t)historyBytes) != 0) return false;
            at += historyBytes;
        }
        gb.savedTokens[slot] = tokens.size();
        savedTokens[slot] = tokens.size();
        return true;
    }
    if(copy.size() != total || gb.savedTokens[slot] > tokens.size()) return false;
    uint64_t at = 0;
    for(GlmLinear& K : glmLinears){
        if(cudaLink().copyUp(K.state, copy.data() + at, (size_t)stateBytes) != 0) return false;
        at += stateBytes;
        if(cudaLink().copyUp(K.history, copy.data() + at, (size_t)historyBytes) != 0) return false;
        at += historyBytes;
    }
    tokens.resize(gb.savedTokens[slot]);
    return true;
}

inline void GpuModel::glmLinearPass(int l, int n) {
    HyShape& s = hyShape;
    HyBuffers& b = hyBuffers;
    GlmShape& g = glmShape;
    GlmBuffers& gb = glmBuffers;
    HyLayer& L = hyLayers[l];
    GlmLinear& K = glmLinears[g.linearSlot[l]];
    PackedInput& p = packs[0];
    glmHcIn(L.hcAttnFn, L.hcAttnScale, L.hcAttnBase, L.attnNorm, n);
    hyProjectSame({{&K.q, gb.lq}, {&K.k, gb.lk}, {&K.v, gb.lv}, {&K.fa, gb.lfa}, {&K.ga, gb.lga}, {&K.beta, gb.lbeta}}, b.xn, n, true);
    hyProject(K.fb, gb.lfa, gb.lf, n);
    hyProject(K.gb, gb.lga, gb.lgate, n);
    go("glmConv", (unsigned int)((3 * g.linearWidth + 255) / 256), 1, 256, 1, gb.lq, gb.lk, gb.lv, K.convQ, K.convK, K.convV, K.history, n, g.linearWidth);
    go("glmKdaScan", (unsigned int)g.linearHeads, 1, 128, 1, gb.lq, gb.lk, gb.lv, gb.lf, gb.lbeta, K.dtBias, K.decay, K.state, gb.lo, n, g.linearHeads, g.lowerBound);
    bool packed = K.out.type != 0 && K.out.type != 30 && K.out.type != 1;
    uint64_t none = 0;
    go("glmGatedNorm", (unsigned int)(n * g.linearHeads), 1, 128, 1, gb.lo, gb.lgate, K.norm, gb.lon, g.linearHeads, s.eps, packed ? p.values : none, packed ? p.scales : none, packed ? p.sums : none);
    glmPacked(K.out, gb.lon, b.attn, n);
    glmHcOut(b.attn, n);
}

inline void GpuModel::glmAttention(int l, int n) {
    HyShape& s = hyShape;
    HyBuffers& b = hyBuffers;
    GlmBuffers& gb = glmBuffers;
    HyLayer& L = hyLayers[l];
    unsigned int spread = (unsigned int)n;
    glmHcIn(L.hcAttnFn, L.hcAttnScale, L.hcAttnBase, L.attnNorm, n);
    std::vector<std::pair<const GpuMatrix*, uint64_t>> first = {{&L.qa, b.qa}, {&L.kva, gb.kvRaw}};
    if(L.indexed){
        first.push_back({&L.idxK, b.idxKey});
        first.push_back({&gb.idxGate[l], gb.idxGateRaw});
    }
    hyProjectSame(first, b.xn, n, true);
    bool scored = L.indexed && hyTop >= 4 * s.idxKeep + 3;
    if(scored) go("hyF32Wide", 32, spread, 128, 1, L.idxProj, b.xn, s.hidden, b.idxW, 32, s.hidden);
    hyNormRows(b.qa, s.qLora, b.qa, s.qLora, L.qaNorm, 0, n, 1, s.qLora, s.eps);
    hyProject(L.qb, b.qa, b.q, n);
    if(scored) glmPacked(L.idxQ, b.qa, b.idxQ, n);
    if(L.indexed){
        go("glmIdxKey", spread, 1, 128, 1, b.idxKey, gb.idxGateRaw, L.idxKNorm, L.idxNorm, L.idxKeys, gb.idxGateRing[l], b.positions, s.idxRing, 1e-6f);
        go("glmIdxPool", spread, 1, 128, 1, L.idxKeys, gb.idxGateRing[l], L.idxApe, b.positions, s.idxRing, L.idxPools);
    }
    go("glmLatent", spread, 1, 256, 1, gb.kvRaw, L.kvNorm, b.kv, s.eps);
    go("hyStore", spread, 1, 256, 1, b.kv, L.cache, b.positions, 0, 576);
    int reach = scored ? s.idxWidth : hyTop + 1;
    float scale = 1.0f / sqrtf((float)glmShape.headKey);
    uint64_t heads = (uint64_t)L.heads;
    for(int first = 0; first < n && !failed; first += glmShape.attendRows){
        int m = std::min(glmShape.attendRows, n - first);
        unsigned int rows = (unsigned int)m;
        uint64_t positions = b.positions + 4ULL * first;
        if(scored){
            go("glmIdxScore", (unsigned int)((s.idxGroups + 127) / 128), rows, 128, 1, b.idxQ + 4ULL * first * 4096, b.idxW + 4ULL * first * 32, L.idxPools, positions, b.idxScores, s.idxGroups, s.idxKeep, s.idxScale);
            go("glmSelect", rows, 1, 1024, 1, b.idxScores, s.idxGroups, positions, b.idxList, b.idxCount, s.idxWidth, s.idxKeep);
        }
        unsigned int partsQ = m >= 16 ? 1 : 4;
        go("glmAbsorbRows", (unsigned int)L.heads, rows * partsQ, 256, 1, b.q + 4ULL * first * heads * 256, L.kb, b.qfull, L.heads, (int)partsQ);
        int most = (int)std::max<uint64_t>(1, b.scratchFloats / ((uint64_t)m * heads * 514));
        int span = std::max(hySpanLeast(), (reach + most - 1) / most);
        int splits = std::max(1, (reach + span - 1) / span);
        if(scored) go("hyAttendList", (unsigned int)(L.heads / 8), rows * (unsigned int)splits, 256, 1, b.qfull, L.cache, b.idxList, b.idxCount, s.idxWidth, b.attnScratch, L.heads, scale, span, splits);
        else go("hyAttendPart", (unsigned int)(L.heads / 8), rows * (unsigned int)splits, 256, 1, b.qfull, L.cache, 0, positions, b.attnScratch, L.heads, 0, scale, span, splits);
        go("hyAttendMerge", (unsigned int)L.heads, rows, 256, 1, b.attnScratch, gb.sinks, b.latentOut, L.heads, splits);
        go("hyExpandRows", (unsigned int)L.heads, rows * partsQ, 256, 1, b.latentOut, L.vb, b.heads + 4ULL * first * heads * s.headValue, L.heads, s.kvLora, s.headValue, (int)partsQ);
    }
    hyProject(L.out, b.heads, b.attn, n);
    glmHcOut(b.attn, n);
}

inline void GpuModel::glmBatch(HyLayer& L, int count) {
    HyShape& s = hyShape;
    HyBuffers& b = hyBuffers;
    HyCache& cache = hyCache;
    int latent = s.latent, width = s.expertWidth;
    PackedInput& p = packs[0];
    uint64_t start = 0;
    uint64_t upAt = (L.gateBytes + 255) / 256 * 256;
    uint64_t downAt = upAt + (L.upBytes + 255) / 256 * 256;
    uint64_t slotsOn = b.mappedHits;
    uint64_t pairsOn = b.mappedHits + 4ULL * s.used * glmShape.hitRows;
    go("hyGatherPack", (unsigned int)(((uint64_t)count * latent / 32 + 7) / 8), 1, 32, 8, b.routedIn, pairsOn, b.hitIn, latent, s.used, count, p.values, p.scales, p.sums);
    go("glmExpertRows", (unsigned int)((L.hostGate.rows + 7) / 8), (unsigned int)count, 256, 1, cache.base, L.hostGate.stride, cache.slotBytes, start, slotsOn, p.values, p.scales, p.sums, b.hitGate, L.hostGate.rows, L.hostGate.cols, (int)L.hostGate.type, tables);
    go("glmExpertRows", (unsigned int)((L.hostUp.rows + 7) / 8), (unsigned int)count, 256, 1, cache.base, L.hostUp.stride, cache.slotBytes, upAt, slotsOn, p.values, p.scales, p.sums, b.hitUp, L.hostUp.rows, L.hostUp.cols, (int)L.hostUp.type, tables);
    go("hySwigluPack", (unsigned int)(((uint64_t)count * width / 32 + 7) / 8), 1, 32, 8, b.hitGate, b.hitUp, b.hitHid, count * width, L.clamp, p.values, p.scales, p.sums);
    go("glmExpertRows", (unsigned int)((L.hostDown.rows + 7) / 8), (unsigned int)count, 256, 1, cache.base, L.hostDown.stride, cache.slotBytes, downAt, slotsOn, p.values, p.scales, p.sums, b.hitDown, L.hostDown.rows, L.hostDown.cols, (int)L.hostDown.type, tables);
    go("hyScatter", (unsigned int)count, 1, 256, 1, b.hitDown, pairsOn, b.mappedWeights, b.routedOut, latent, s.used);
}

inline void GpuModel::glmExperts(int l, int n) {
    HyShape& s = hyShape;
    HyBuffers& b = hyBuffers;
    HyCache& cache = hyCache;
    HyLayer& L = hyLayers[l];
    int used = s.used, latent = s.latent;
    int pairs = n * used;
    auto blocks = [](uint64_t count){ return (unsigned int)((count + 255) / 256); };
    std::vector<char> onCpu(pairs, 1);
    go("hyFill", blocks((uint64_t)n * latent), 1, 256, 1, b.routedOut, n * latent);
    int hits = 0;
    int* hitSlot = b.hostHits;
    int* hitPair = b.hostHits + used * glmShape.hitRows;
    for(int i = 0; i < pairs; i++){
        int e = b.hostIds[i];
        if(e < 0 || e >= s.experts) continue;
        size_t key = (size_t)l * s.experts + e;
        cache.heat[key]++;
        cache.total++;
        int slot = cache.slotOf[key];
        if(slot < 0) continue;
        onCpu[i] = 0;
        cache.hits++;
        hitSlot[hits] = slot;
        hitPair[hits] = i;
        hits++;
    }
    if(hits > 0) glmBatch(L, hits);
    if(hyFlushOn()) cudaQueue().query(stream);
    auto cpuFrom = std::chrono::steady_clock::now();
    hyExpertsHost(L, n, onCpu);
    cache.cpuMs += since(cpuFrom);
    go("hyAdd", blocks((uint64_t)n * latent), 1, 256, 1, b.routedOut, b.mappedMoe, n * latent);
    if(hyFlushOn()) cudaQueue().query(stream);
    hySwapIn(l, n, b.hostIds);
}

inline void GpuModel::glmExpertsWide(int l, int n) {
    HyShape& s = hyShape;
    HyBuffers& b = hyBuffers;
    GlmBuffers& gb = glmBuffers;
    HyCache& cache = hyCache;
    HyLayer& L = hyLayers[l];
    CudaQueue& queue = cudaQueue();
    int used = s.used, latent = s.latent, ew = s.expertWidth;
    int half = gb.stageSlots / 2, most = gb.pairsMost;
    uint64_t none = 0;
    uint64_t gateAt = 0, upAt = (L.gateBytes + 255) / 256 * 256, downAt = upAt + (L.upBytes + 255) / 256 * 256;
    auto started = std::chrono::steady_clock::now();
    go("hyFill", (unsigned int)(((uint64_t)n * latent + 255) / 256), 1, 256, 1, b.routedOut, n * latent);
    go("quantizeInput", (unsigned int)(((uint64_t)n * latent / 32 + 7) / 8), 1, 32, 8, b.xn, n * latent, gb.inValues, gb.inScales, gb.inSums);
    std::vector<std::vector<int>> byExpert(s.experts);
    for(int i = 0; i < n * used; i++){
        int e = b.hostIds[i];
        if(e < 0 || e >= s.experts) continue;
        byExpert[e].push_back(i);
        size_t key = (size_t)l * s.experts + e;
        cache.heat[key]++;
        cache.total++;
        if(cache.slotOf[key] >= 0) cache.hits++;
    }
    struct Piece {
        int expert = 0, from = 0, count = 0;
    };
    struct Job {
        bool staged = false;
        std::vector<Piece> experts;
    };
    std::vector<Job> jobs;
    for(int pass = 0; pass < 2; pass++){
        Job job;
        job.staged = pass == 1;
        int pairs = 0;
        for(int e = 0; e < s.experts; e++){
            int total = (int)byExpert[e].size();
            bool resident = cache.slotOf[(size_t)l * s.experts + e] >= 0;
            if(!total || resident == job.staged) continue;
            for(int from = 0; from < total; from += most){
                int count = std::min(most, total - from);
                if(!job.experts.empty() && ((int)job.experts.size() == half || pairs + count > most)){
                    jobs.push_back(job);
                    job.experts.clear();
                    pairs = 0;
                }
                job.experts.push_back({e, from, count});
                pairs += count;
            }
        }
        if(!job.experts.empty()) jobs.push_back(job);
    }
    uint64_t region = 192 + 3ULL * most;
    for(size_t k = 0; k < jobs.size() && !failed; k++){
        const Job& job = jobs[k];
        int h = (int)(k % 2);
        int groups = (int)job.experts.size();
        if(k >= 2 && queue.eventSync(gb.done[h]) != 0) failed = true;
        uint64_t base = cache.base;
        if(job.staged){
            auto from = std::chrono::steady_clock::now();
            uint8_t* host = gb.pinned + (uint64_t)h * half * cache.slotBytes;
            pool.run(groups, [&](int j){
                int e = job.experts[j].expert;
                uint8_t* at = host + (uint64_t)j * cache.slotBytes;
                memcpy(at + gateAt, L.hostGate.data + (uint64_t)e * L.gateBytes, (size_t)L.gateBytes);
                memcpy(at + upAt, L.hostUp.data + (uint64_t)e * L.upBytes, (size_t)L.upBytes);
                memcpy(at + downAt, L.hostDown.data + (uint64_t)e * L.downBytes, (size_t)L.downBytes);
            });
            gb.uploadMs += since(from);
            gb.uploaded += (uint64_t)groups * (L.gateBytes + L.upBytes + L.downBytes);
            base = gb.stage + (uint64_t)h * half * cache.slotBytes;
            if(queue.copyUp(base, host, (size_t)groups * cache.slotBytes, gb.copyStream) != 0 || queue.eventRecord(gb.copied[h], gb.copyStream) != 0) failed = true;
        }
        int* groupSlot = gb.hostGroups + h * region;
        int* groupStart = groupSlot + 64;
        int* groupCount = groupSlot + 128;
        int* inputIndex = groupSlot + 192;
        int* tokenOf = inputIndex + most;
        float* weightOf = (float*)(tokenOf + most);
        int c = 0, widest = 0;
        for(int g = 0; g < groups; g++){
            const Piece& piece = job.experts[g];
            int e = piece.expert;
            groupSlot[g] = job.staged ? g : cache.slotOf[(size_t)l * s.experts + e];
            groupStart[g] = c;
            groupCount[g] = piece.count;
            widest = std::max(widest, piece.count);
            for(int i = piece.from; i < piece.from + piece.count; i++){
                int pair = byExpert[e][i];
                inputIndex[c] = pair / used;
                tokenOf[c] = pair / used;
                weightOf[c] = b.hostWeights[pair];
                c++;
            }
        }
        if(job.staged && queue.streamWait(stream, gb.copied[h], 0) != 0) failed = true;
        uint64_t slotOn = gb.groupsOn + 4ULL * h * region;
        uint64_t startOn = slotOn + 4ULL * 64, countOn = slotOn + 4ULL * 128, inputOn = slotOn + 4ULL * 192;
        uint64_t tokenOn = inputOn + 4ULL * most, weightOn = tokenOn + 4ULL * most;
        bool wide = c >= wideLeast * groups;
        int width = tileWidth(wide);
        int tiles = (widest + width - 1) / width;
        unsigned int threads = width == 128 ? 256 : 128;
        int gateTile = rowTile(L.hostGate.type, wide, 0), upTile = rowTile(L.hostUp.type, wide, 0), downTile = rowTile(L.hostDown.type, wide, 0);
        go(tiled(L.hostGate.type, wide, 0).c_str(), (unsigned int)((ew + gateTile - 1) / gateTile), (unsigned int)(groups * tiles), threads, 1, (int)L.hostGate.type, base + gateAt, L.hostGate.stride, cache.slotBytes, slotOn, startOn, countOn, inputOn, gb.inValues, gb.inScales, gb.inSums, none, none, none, gb.gateOut, ew, latent, 0, tiles, none, none, none, tables);
        go(tiled(L.hostUp.type, wide, 0).c_str(), (unsigned int)((ew + upTile - 1) / upTile), (unsigned int)(groups * tiles), threads, 1, (int)L.hostUp.type, base + upAt, L.hostUp.stride, cache.slotBytes, slotOn, startOn, countOn, inputOn, gb.inValues, gb.inScales, gb.inSums, none, none, none, gb.upOut, ew, latent, 0, tiles, none, none, none, tables);
        go("hySwigluPack", (unsigned int)(((uint64_t)c * ew / 32 + 7) / 8), 1, 32, 8, gb.gateOut, gb.upOut, gb.hidOut, c * ew, L.clamp, gb.hidValues, gb.hidScales, gb.hidSums);
        go(tiled(L.hostDown.type, wide, 0).c_str(), (unsigned int)((latent + downTile - 1) / downTile), (unsigned int)(groups * tiles), threads, 1, (int)L.hostDown.type, base + downAt, L.hostDown.stride, cache.slotBytes, slotOn, startOn, countOn, none, gb.hidValues, gb.hidScales, gb.hidSums, none, none, none, none, latent, ew, 0, tiles, tokenOn, weightOn, b.routedOut, tables);
        if(queue.eventRecord(gb.done[h], stream) != 0) failed = true;
    }
    if(cudaLink().sync() != 0) failed = true;
    gb.wideMs += since(started);
}

inline void GpuModel::glmLayer(int l, int n) {
    HyShape& s = hyShape;
    HyBuffers& b = hyBuffers;
    HyLayer& L = hyLayers[l];
    int d = s.hidden;
    auto blocks = [](uint64_t count){ return (unsigned int)((count + 255) / 256); };
    PackedInput& pk = packs[0];
    if(glmShape.linear[l]) glmLinearPass(l, n);
    else glmAttention(l, n);
    glmHcIn(L.hcFfnFn, L.hcFfnScale, L.hcFfnBase, L.ffnNorm, n);
    if(L.dense){
        int ff = s.feedForward;
        hyProjectSame({{&L.ffnGate, b.bigGate}, {&L.ffnUp, b.bigUp}}, b.xn, n, true);
        go("hySwigluPack", (unsigned int)(((uint64_t)n * ff / 32 + 7) / 8), 1, 32, 8, b.bigGate, b.bigUp, b.bigHid, n * ff, L.clamp, pk.values, pk.scales, pk.sums);
        glmPacked(L.ffnDown, b.bigHid, b.ffn, n);
        glmHcOut(b.ffn, n);
        return;
    }
    CudaQueue& queue = cudaQueue();
    go("hyF32Wide", (unsigned int)s.experts, (unsigned int)n, 128, 1, L.router, b.xn, d, b.logits, s.experts, d);
    go("hyRouteFast", (unsigned int)n, 1, 512, 1, b.logits, L.routerBias, b.mappedIds, b.mappedWeights, s.experts, s.used, s.expertScale, s.expertNorm ? 1 : 0);
    go("hyCopy", blocks((uint64_t)n * d), 1, 256, 1, b.xn, b.mappedLatent, n * d);
    if(!hyRouted && queue.eventCreate(&hyRouted, 2) != 0) failed = true;
    if(queue.eventRecord(hyRouted, stream) != 0) failed = true;
    hyProjectSame({{&L.sharedGate, b.sGate}, {&L.sharedUp, b.sUp}}, b.xn, n, true);
    int ew = s.expertWidth;
    go("hySwigluPack", (unsigned int)(((uint64_t)n * ew / 32 + 7) / 8), 1, 32, 8, b.sGate, b.sUp, b.sHid, n * ew, L.clamp, pk.values, pk.scales, pk.sums);
    glmPacked(L.sharedDown, b.sHid, b.shared, n);
    auto waited = std::chrono::steady_clock::now();
    if(queue.eventSync(hyRouted) != 0) failed = true;
    hyCache.waitMs += since(waited);
    if(!failed && n <= glmShape.hitRows) glmExperts(l, n);
    else if(!failed) glmExpertsWide(l, n);
    go("hyAdd", blocks((uint64_t)n * d), 1, 256, 1, b.routedOut, b.shared, n * d);
    glmHcOut(b.routedOut, n);
}

inline bool GpuModel::glmChunk(const int* ids, const int* positions, int n, bool last) {
    HyShape& s = hyShape;
    HyBuffers& b = hyBuffers;
    CudaQueue& queue = cudaQueue();
    int d = s.hidden;
    if(waitStream() != 0) failed = true;
    pool.run(n, [&](int t){
        int id = ids[t];
        if(id < 0 || id >= embed.rows) id = 0;
        dequantRow(embed.type, embed.data + (uint64_t)id * embed.stride, b.hostEmbed + (size_t)t * d, d);
    });
    hyTop = 0;
    b.firstPos = positions[0];
    for(int t = 0; t < n; t++){
        b.hostPositions[t] = positions[t];
        hyTop = std::max(hyTop, positions[t]);
    }
    if(queue.copyUp(b.positions, b.hostPositions, (size_t)n * 4, stream) != 0) failed = true;
    go("hySpread", (unsigned int)n, 1, 256, 1, b.mappedEmbed, b.streams, d, s.hc);
    for(int l = 0; l < s.layers && !failed && !stopGenerating; l++) glmLayer(l, n);
    if(!last || failed) return !failed;
    int keep = shortlist > 0 ? shortlist : 0;
    uint64_t lastRow = b.streams + 4ULL * (n - 1) * d * s.hc;
    go("glmHeadNorm", 1, 1, 256, 1, lastRow, b.outNorm, b.last, d, s.eps);
    hyProject(head, b.last, b.vocab, 1);
    if(keep > 0) go("shortRows", 1, 1, 512, 1, b.vocab, s.vocab, keep, b.mappedLogits);
    else go("hyCopy", (unsigned int)((s.vocab + 255) / 256), 1, 256, 1, b.vocab, b.mappedLogits, s.vocab);
    return !failed;
}

inline std::vector<float> GpuModel::forwardGlm(const std::vector<int>& ids, const std::vector<int>& positions) {
    std::vector<float> logits;
    HyShape& s = hyShape;
    HyBuffers& b = hyBuffers;
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
        problem = "glm5-next does not check drafted rows yet";
        return logits;
    }
    int start = positions[0];
    if(start != (int)tokens.size()){
        if(start != 0){
            problem = "glm5-next keeps a running state and can only continue at position " + std::to_string(tokens.size()) + ", not " + std::to_string(start);
            return logits;
        }
        tokens.clear();
        glmReset();
    }
    failed = false;
    hyCollectUploads(false);
    auto clock = std::chrono::steady_clock::now();
    for(size_t at = 0; at < ids.size() && !failed && !stopGenerating;){
        int n = (int)std::min(ids.size() - at, (size_t)chunk);
        tokens.insert(tokens.end(), ids.begin() + at, ids.begin() + at + n);
        glmChunk(ids.data() + at, positions.data() + at, n, at + n == ids.size());
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
    static const char* dump = getenv("SNEO_GLM_DUMP");
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
    if(hyCache.ageEvery > 0 && ++hyCache.aged % (uint64_t)hyCache.ageEvery == 0){
        for(uint32_t& h : hyCache.heat) h >>= 1;
    }
    headMs = since(clock);
    static int calls = 0;
    if(hyCache.total - hyCache.heatSaved > 200000) hySaveHeat();
    if(ids.size() > 1 || ++calls % 64 == 0){
        std::cout << "glm5-next: experts found in video memory " << (hyCache.total ? 100 * hyCache.hits / hyCache.total : 0) << "% of " << hyCache.total << ", uploads " << hyCache.uploads << ", free slots " << hyCache.free.size() << ", ram free " << hyRamFreeGb() << " GB, ms in this call: total " << since(clock) << " cpu experts " << hyCache.cpuMs << " (" << (int)(hyCache.cpuBytes / 1048576.0 / std::max(0.001, hyCache.cpuMs)) << " GB/s) waiting for the gpu " << hyCache.waitMs << " uploads " << hyCache.uploadMs << std::endl;
        hyCache.cpuMs = hyCache.waitMs = hyCache.uploadMs = 0;
        hyCache.cpuBytes = 0;
    }
    return logits;
}
