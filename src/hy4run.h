#pragma once

inline bool GpuModel::moveHy4(int slot, bool keep) {
    if(!ready || slot < 0 || slot >= 4) return false;
    if(keep){
        savedTokens[slot] = tokens.size();
        hyBuffers.saved[slot] = true;
        return true;
    }
    if(!hyBuffers.saved[slot]) return false;
    tokens.resize(std::min(tokens.size(), savedTokens[slot]));
    return true;
}

inline uint16_t hyHalfBits(float v) {
    return (uint16_t)_mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(v), 0), 0);
}

// una fila de floats a q8_0: bloques de 32 con su escala en f16
inline void hyQuantQ8(const float* x, uint8_t* out, int cols) {
    for(int b = 0; b < cols / 32; b++){
        const float* v = x + b * 32;
        float top = 0.0f;
        for(int i = 0; i < 32; i++) top = std::max(top, fabsf(v[i]));
        float d = top / 127.0f;
        float inv = d > 0.0f ? 1.0f / d : 0.0f;
        uint8_t* block = out + b * 34;
        uint16_t h = hyHalfBits(d);
        memcpy(block, &h, 2);
        for(int i = 0; i < 32; i++) block[2 + i] = (uint8_t)(int8_t)std::lrintf(v[i] * inv);
    }
}

// una fila a iq4_nl: 16 niveles no lineales, la escala se ajusta por minimos cuadrados con los niveles elegidos
inline void hyQuantNl(const float* x, uint8_t* out, int cols) {
    for(int b = 0; b < cols / 32; b++){
        const float* v = x + b * 32;
        float peak = 0.0f;
        for(int i = 0; i < 32; i++) if(fabsf(v[i]) > fabsf(peak)) peak = v[i];
        float d = peak / -127.0f;
        uint8_t pick[32] = {};
        for(int round = 0; round < 3 && d != 0.0f; round++){
            float inv = 1.0f / d, top = 0.0f, bottom = 0.0f;
            for(int i = 0; i < 32; i++){
                float want = v[i] * inv;
                int best = 0;
                float gap = fabsf(want - nonLinear4[0]);
                for(int k = 1; k < 16; k++){
                    float g = fabsf(want - nonLinear4[k]);
                    if(g < gap){
                        gap = g;
                        best = k;
                    }
                }
                pick[i] = (uint8_t)best;
                top += v[i] * nonLinear4[best];
                bottom += (float)nonLinear4[best] * nonLinear4[best];
            }
            if(round < 2 && bottom > 0.0f) d = top / bottom;
        }
        uint8_t* block = out + b * 18;
        uint16_t h = hyHalfBits(d);
        memcpy(block, &h, 2);
        for(int i = 0; i < 16; i++) block[2 + i] = (uint8_t)(pick[i] | (pick[i + 16] << 4));
    }
}

// pasa un tensor del archivo a otro tipo con todos los nucleos y deja los bytes listos para subir
inline std::vector<uint8_t> hyRequant(const uint8_t* src, uint32_t from, int cols, int rows, uint32_t into) {
    uint64_t fromStride = rowBytes(from, cols), intoStride = rowBytes(into, cols);
    std::vector<uint8_t> out((size_t)rows * intoStride);
    int workers = (int)std::max(1u, std::thread::hardware_concurrency());
    std::atomic<int> next{0};
    std::vector<std::thread> crew;
    for(int w = 0; w < workers; w++){
        crew.emplace_back([&]{
            std::vector<float> line(cols);
            for(int r = next++; r < rows; r = next++){
                dequantRow(from, src + (uint64_t)r * fromStride, line.data(), cols);
                if(into == 8) hyQuantQ8(line.data(), out.data() + (uint64_t)r * intoStride, cols);
                else hyQuantNl(line.data(), out.data() + (uint64_t)r * intoStride, cols);
            }
        });
    }
    for(std::thread& t : crew) t.join();
    return out;
}

inline void GpuModel::hyPack(uint64_t input, int count, int slot) {
    go("quantizeInput", (unsigned int)((count / 32 + 7) / 8), 1, 32, 8, input, count, packs[slot].values, packs[slot].scales, packs[slot].sums);
}

inline void GpuModel::hyProject(const GpuMatrix& m, uint64_t input, uint64_t output, int n) {
    if(m.type == 0){
        go("hyF32Rows", (unsigned int)((m.rows + 7) / 8), 1, 256, 1, m.data, input, m.cols, output, m.rows, m.rows, m.cols, n);
        return;
    }
    if(m.type != 30 && m.type != 1) hyPack(input, n * m.cols, 0);
    project(m, input, 0, output, n);
}

inline void GpuModel::hyProjectSame(const std::vector<std::pair<const GpuMatrix*, uint64_t>>& list, uint64_t input, int n, bool ready) {
    bool packed = ready;
    for(const auto& item : list){
        const GpuMatrix& m = *item.first;
        if(m.type == 0){
            go("hyF32Wide", (unsigned int)m.rows, (unsigned int)n, 128, 1, m.data, input, m.cols, item.second, m.rows, m.cols);
            continue;
        }
        if(m.type != 30 && m.type != 1 && !packed){
            hyPack(input, n * m.cols, 0);
            packed = true;
        }
        project(m, input, 0, item.second, n);
    }
}

inline void GpuModel::hyHcIn(uint64_t streams, uint64_t fn, uint64_t scale, uint64_t base, uint64_t norm, uint64_t out, uint64_t post, int rows, int n, bool pack) {
    HyShape& s = hyShape;
    HyBuffers& b = hyBuffers;
    int width = s.hc * s.hidden, chunk = 512;
    int parts = (width + chunk - 1) / chunk;
    go("hyHcDots", (unsigned int)(n * parts), 1, 256, 1, streams, fn, b.mixes, width, rows, chunk);
    uint64_t none = 0;
    PackedInput& p = packs[0];
    go("hyHcCombine", (unsigned int)n, 1, 512, 1, streams, b.mixes, scale, base, norm, out, post, s.hidden, s.hc, s.eps, s.hcEps, s.hcMag, pack ? p.values : none, pack ? p.scales : none, pack ? p.sums : none);
}

inline void GpuModel::hyHcOut(uint64_t x, uint64_t norm, int n) {
    HyShape& s = hyShape;
    HyBuffers& b = hyBuffers;
    int chunk = 256;
    int parts = (s.hidden + chunk - 1) / chunk;
    go("hyHcSpread", (unsigned int)(n * parts), 1, 256, 1, b.streams, x, norm, b.post, s.hidden, s.hc, s.eps, chunk);
}

inline void GpuModel::hyNormRows(uint64_t in, int inStride, uint64_t out, int outStride, uint64_t weight, int weightStride, int rows, int groups, int cols, float eps) {
    go("hyNorm", (unsigned int)(rows * groups), 1, 256, 1, in, inStride, out, outStride, weight, weightStride, groups, cols, eps);
}

inline bool GpuModel::loadHy4(const std::string& path, int threads) {
    hy4 = true;
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
    std::string a = "hy_v4_flash.";
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
    s.rope = (int)metaWhole(file, a + "rope.dimension_count", 64);
    int headKey = (int)metaWhole(file, a + "attention.key_length_mla", 256);
    s.nope = headKey - s.rope;
    s.headValue = (int)metaWhole(file, a + "attention.value_length_mla", 256);
    s.window = (int)metaWhole(file, a + "attention.sliding_window", 512);
    s.experts = (int)metaWhole(file, a + "expert_count", 0);
    s.used = (int)metaWhole(file, a + "expert_used_count", 0);
    s.latent = (int)metaWhole(file, a + "expert_latent_length", 0);
    s.expertWidth = (int)metaWhole(file, a + "expert_feed_forward_length", 0);
    s.feedForward = (int)metaWhole(file, a + "feed_forward_length", 0);
    s.eps = (float)real("attention.layer_norm_rms_epsilon", 1e-6);
    s.hcEps = (float)real("hyper_connection.epsilon", 1e-6);
    s.hcMag = (float)real("hyper_connection.magnitude", 2.0);
    s.ropeBase = (float)real("rope.freq_base", 1e7);
    s.ropeBaseSliding = (float)real("rope.freq_base_swa", 1e4);
    s.expertScale = (float)real("expert_weights_scale", 1.0);
    s.expertNorm = metaWhole(file, a + "expert_weights_norm", 1) != 0;
    int gating = (int)metaWhole(file, a + "expert_gating_func", 2);
    int denseLead = (int)metaWhole(file, a + "leading_dense_block_count", 1);
    std::vector<int64_t> headList = list("attention.head_count");
    std::vector<int64_t> pattern = list("attention.sliding_window_pattern");
    std::vector<double> clamps = reals("swiglu_clamp_exp");
    if(s.layers <= 0 || s.hidden <= 0 || s.hidden % 256 != 0 || (int)headList.size() < s.layers || (int)pattern.size() < s.layers){
        problem = "unexpected hy_v4_flash shape";
        return false;
    }
    if(s.kvLora != 512 || s.rope != 64 || s.nope > 256 || s.headValue > 512 || s.hc > 8 || s.used > 32 || s.experts > 1024 || gating != 2 || s.latent % 256 != 0 || s.expertWidth % 256 != 0){
        problem = "this hy_v4_flash layout is not the one the engine runs (latent 512 with rope 64, sigmoid routing)";
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
    if(!buildModule(sizes + quantKernelSource + modelKernelSource + gemmaKernelSource + hyKernelSource, module)){
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
    // los que se achican en la carga: la atencion de 8 bits pasa a 4 y la cabeza de 16 a 8, asi entran mas expertos en la vram
    struct Shrink {
        const TensorInfo* t;
        GpuMatrix* out;
    };
    std::vector<Shrink> shrink;
    bool lean = !getenv("SNEO_HY_FULL_ATTN");
    auto matrix = [&](const std::string& name, GpuMatrix& out, bool optional = false, uint32_t into = 0) {
        const TensorInfo* t = findTensor(file, name);
        if(!t && optional) return;
        if(!t || (!knownType(t->type) && t->type != 0)){
            if(problem.empty()) problem = t
                ? name + " is " + typeInfo(t->type).name + ", a tensor type the engine cannot read yet"
                : "missing tensor " + name;
            return;
        }
        out.type = t->type;
        out.cols = (int)t->dims[0];
        out.rows = (int)t->dims[1];
        bool smaller = lean && into != 0 && t->dims[0] % 32 == 0 && ((into == 20 && t->type == 8) || (into == 8 && (t->type == 30 || t->type == 1)));
        if(smaller) out.type = into;
        out.stride = rowBytes(out.type, t->dims[0]);
        if(smaller) shrink.push_back({t, &out});
        else wanted.push_back({t, &out.data});
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
    embed.data = host(table);
    embed.type = table->type;
    embed.cols = (int)table->dims[0];
    embed.rows = (int)table->dims[1];
    embed.stride = rowBytes(table->type, table->dims[0]);
    s.vocab = embed.rows;
    matrix("output.weight", head, false, 8);
    floats("output_norm.weight", b.outNorm);
    floats("output_hc_fn.weight", b.headFn);
    floats("output_hc_base.weight", b.headBase);
    floats("output_hc_scale.weight", b.headScale);

    // engram: tablas de n-gramas que se leen en la cpu del archivo mapeado
    std::vector<int64_t> engramLayers;
    if(auto found = file.meta.find(a + "ple.layers"); found != file.meta.end()) engramLayers = found->second.wholes;
    if(!engramLayers.empty()){
        const TensorInfo* rows = findTensor(file, "per_layer_token_embd.weight");
        s.engramModules = (int)engramLayers.size();
        s.engramGram = (int)metaWhole(file, a + "ple.ngram_size", 3);
        s.engramPerGram = (int)metaWhole(file, a + "ple.heads_per_ngram", 8);
        s.engramDim = (int)metaWhole(file, a + "embedding_length_per_layer_input", 96);
        s.engramHeads = (s.engramGram - 1) * s.engramPerGram;
        s.engramPad = (int)metaWhole(file, a + "engram.pad_token_id", 0);
        for(int64_t v : list("ple.layer_multipliers")) s.engramMult.push_back((uint64_t)v);
        for(int64_t v : list("ple.head_vocab_sizes")) s.engramPrimes.push_back((uint32_t)v);
        for(int64_t v : list("ple.head_offsets")) s.engramOffsets.push_back((uint32_t)v);
        for(int64_t v : list("engram.token_map")) s.engramMap.push_back((int)v);
        size_t columns = (size_t)s.engramModules * s.engramHeads;
        if(!rows || rows->type != 43 || (int)rows->dims[0] != s.engramDim || s.engramMult.size() != (size_t)s.engramModules * s.engramGram || s.engramPrimes.size() != columns || s.engramOffsets.size() != columns || s.engramPad >= (int)s.engramMap.size() || s.engramHeads > 32 || s.engramDim > 256){
            problem = "the engram tables of this hy_v4_flash file are not the layout the engine reads";
            return false;
        }
        s.engramTable = host(rows);
    }

    chunk = 64;
    const char* subAsked = getenv("SNEO_HY_WIDE_SUB");
    b.wideSub = subAsked ? std::max(64, atoi(subAsked)) / 64 * 64 : 256;
    // la ventana deslizante guarda las filas del tramo mas largo ademas de la ventana
    s.ring = s.window + std::max(chunk, b.wideSub);
    hyLayers.assign(s.layers, HyLayer());
    for(int l = 0; l < s.layers; l++){
        HyLayer& L = hyLayers[l];
        std::string p = "blk." + std::to_string(l) + ".";
        L.sliding = pattern[l] != 0;
        L.heads = (int)headList[l];
        L.dense = l < denseLead;
        L.clamp = l < (int)clamps.size() ? (float)clamps[l] : 0.0f;
        if(L.heads % 8 != 0 || L.heads > 128){
            problem = "hy_v4_flash layer " + std::to_string(l) + " has " + std::to_string(L.heads) + " heads, the engine runs multiples of 8";
            return false;
        }
        s.widestHeads = std::max(s.widestHeads, L.heads);
        floats(p + "attn_norm.weight", L.attnNorm);
        floats(p + "attn_sinks.weight", L.sinks);
        matrix(p + "attn_q_a.weight", L.qa);
        floats(p + "attn_q_a_norm.weight", L.qaNorm);
        matrix(p + "attn_q_b.weight", L.qb, false, 20);
        matrix(p + "attn_kv_a_mqa.weight", L.kva);
        floats(p + "attn_kv_a_norm.weight", L.kvNorm);
        matrix(p + "attn_output.weight", L.out, false, 20);
        matrix(p + "attn_gate.weight", L.gate, false, 20);
        floats(p + "post_attention_norm.weight", L.postAttnNorm);
        floats(p + "hc_attn_fn.weight", L.hcAttnFn);
        floats(p + "hc_attn_base.weight", L.hcAttnBase);
        floats(p + "hc_attn_scale.weight", L.hcAttnScale);
        floats(p + "hc_ffn_fn.weight", L.hcFfnFn);
        floats(p + "hc_ffn_base.weight", L.hcFfnBase);
        floats(p + "hc_ffn_scale.weight", L.hcFfnScale);
        floats(p + "ffn_norm.weight", L.ffnNorm);
        floats(p + "post_ffw_norm.weight", L.postFfnNorm);
        if(L.dense){
            matrix(p + "ffn_gate.weight", L.ffnGate);
            matrix(p + "ffn_up.weight", L.ffnUp);
            matrix(p + "ffn_down.weight", L.ffnDown);
        } else {
            floats(p + "ffn_gate_inp.weight", L.router);
            floats(p + "exp_probs_b.bias", L.routerBias, true);
            matrix(p + "ffn_routed_down.weight", L.routedDown, false, 20);
            matrix(p + "ffn_routed_up.weight", L.routedUp, false, 20);
            matrix(p + "ffn_gate_shexp.weight", L.sharedGate, false, 20);
            matrix(p + "ffn_up_shexp.weight", L.sharedUp, false, 20);
            matrix(p + "ffn_down_shexp.weight", L.sharedDown, false, 20);
            const TensorInfo* g = findTensor(file, p + "ffn_gate_exps.weight");
            const TensorInfo* u = findTensor(file, p + "ffn_up_exps.weight");
            const TensorInfo* d = findTensor(file, p + "ffn_down_exps.weight");
            if(!g || !u || !d || !hostDotType(g->type) || !hostDotType(u->type) || !hostDotType(d->type) || (int)g->dims[2] != s.experts){
                problem = "hy_v4_flash layer " + std::to_string(l) + " has experts the cpu cannot read";
                return false;
            }
            L.hostGate = {host(g), g->type, (int)g->dims[0], (int)g->dims[1], rowBytes(g->type, g->dims[0])};
            L.hostUp = {host(u), u->type, (int)u->dims[0], (int)u->dims[1], rowBytes(u->type, u->dims[0])};
            L.hostDown = {host(d), d->type, (int)d->dims[0], (int)d->dims[1], rowBytes(d->type, d->dims[0])};
            L.gateBytes = (uint64_t)L.hostGate.rows * L.hostGate.stride;
            L.upBytes = (uint64_t)L.hostUp.rows * L.hostUp.stride;
            L.downBytes = (uint64_t)L.hostDown.rows * L.hostDown.stride;
            if(L.hostGate.cols != s.latent || L.hostGate.rows != s.expertWidth || L.hostDown.cols != s.expertWidth || L.hostDown.rows != s.latent){
                problem = "unexpected hy_v4_flash expert shapes in layer " + std::to_string(l);
                return false;
            }
        }
        for(size_t e = 0; e < engramLayers.size(); e++){
            if(engramLayers[e] != l) continue;
            L.engram = (int)e;
            matrix(p + "engram_kv.weight", L.engramKv);
            floats(p + "engram_kv_b.weight", L.engramBias);
            floats(p + "engram_gate.weight", L.engramGate);
            floats(p + "engram_lookup_norm.weight", L.engramNorm);
            floats(p + "engram_pre_gamma.weight", L.engramGamma);
            floats(p + "engram_hc_scale.weight", L.engramScale);
        }
        // las capas dispersas con indexador eligen a que posiciones mira cada consulta
        if(!L.sliding && findTensor(file, p + "indexer.attn_k.weight") && !getenv("SNEO_HY_NO_INDEX")){
            L.indexed = true;
            s.indexed = true;
            matrix(p + "indexer.attn_k.weight", L.idxK);
            matrix(p + "indexer.attn_q_b.weight", L.idxQ);
            floats(p + "indexer.k_norm.weight", L.idxKNorm);
            floats(p + "indexer.proj.weight", L.idxProj);
            floats(p + "indexer_compressor_ape.weight", L.idxApe);
            floats(p + "indexer_compressor_norm.weight", L.idxNorm);
            if(problem.empty() && (L.idxK.rows != 128 || L.idxK.cols != s.hidden || L.idxQ.rows != 2048 || L.idxQ.cols != s.qLora)){
                problem = "unexpected hy_v4_flash indexer shapes in layer " + std::to_string(l);
                return false;
            }
        }
        if(!problem.empty()) return false;
        if(L.qa.rows != s.qLora || L.qb.rows != L.heads * headKey || L.kva.rows != s.kvLora + s.rope || L.out.cols != L.heads * s.headValue || L.gate.rows != L.heads * s.headValue){
            problem = "unexpected hy_v4_flash attention shapes in layer " + std::to_string(l);
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

    // k_b y v_b van a f16 en la gpu para multiplicarlos por cabeza
    uint64_t perHeadBytes = 0;
    for(int l = 0; l < s.layers; l++) perHeadBytes += (uint64_t)hyLayers[l].heads * ((uint64_t)s.kvLora * s.nope + (uint64_t)s.headValue * s.kvLora) * 2;
    for(auto& w : wanted) denseBytes += (w.first->bytes + 255) / 256 * 256;
    for(auto& item : shrink) denseBytes += ((uint64_t)item.out->rows * item.out->stride + 255) / 256 * 256;
    denseBytes += perHeadBytes;
    for(const HyLayer& L : hyLayers) denseBytes += L.indexed ? 2 * 128 * 128 * 4 : 0;

    uint64_t C = (uint64_t)chunk;
    uint64_t d = (uint64_t)s.hidden;
    uint64_t H = (uint64_t)s.widestHeads;
    uint64_t widestInput = std::max<uint64_t>({d * s.hc, H * s.headValue, (uint64_t)s.feedForward, (uint64_t)s.engramHeads * s.engramDim});
    uint64_t bufferBytes = C * 4 * (d * s.hc + d * s.hc + 16 + 8 + d * 2 + s.qLora + H * (s.nope + s.rope) + 576 + H * 576 + H * 512 + H * s.headValue * 2 + d
        + 3 * (uint64_t)s.feedForward + d + 3 * (uint64_t)s.expertWidth + 2 * (uint64_t)s.latent + 1 + (uint64_t)s.experts
        + (uint64_t)s.engramHeads * s.engramDim * 2 + (s.hc + 1) * d) + d * 4 + (uint64_t)s.vocab * 4 + widestInput * C * 2 * 2;
    uint64_t slidingBytes = 0, perToken = 0;
    for(const HyLayer& L : hyLayers){
        if(L.sliding) slidingBytes += (uint64_t)s.ring * 576 * 2;
        else perToken += 576 * 2;
    }
    uint64_t margin = 900ULL << 20;
    uint64_t fixedBytes = denseBytes + bufferBytes + slidingBytes + margin;
    if(fixedBytes >= startFree){
        problem = "hy_v4_flash needs " + std::to_string((fixedBytes >> 20) + 1) + " MB of video memory before any context and " + std::to_string(startFree >> 20) + " MB are free";
        return false;
    }
    uint64_t room = perToken > 0 ? (startFree - fixedBytes) / perToken : (uint64_t)contextLimit;
    if((uint64_t)contextLimit > room){
        int fitted = (int)(room / 256 * 256);
        if(fitted < 4096){
            problem = "hy_v4_flash fits only " + std::to_string(fitted) + " tokens of context in the free video memory";
            return false;
        }
        contextLimit = fitted;
    }
    contextLimit = std::max(contextLimit, s.ring);

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
    if(!shrink.empty()) loadSay("Packing the attention into 4 bits for more experts in video memory");
    for(auto& item : shrink){
        std::vector<uint8_t> packed = hyRequant(host(item.t), item.t->type, item.out->cols, item.out->rows, item.out->type);
        item.out->data = arena + used;
        if(cudaLink().copyUp(item.out->data, packed.data(), packed.size()) != 0){
            problem = "cannot copy the packed attention to the gpu";
            return false;
        }
        used += (packed.size() + 255) / 256 * 256;
    }
    for(int l = 0; l < s.layers && problem.empty(); l++){
        HyLayer& L = hyLayers[l];
        std::string p = "blk." + std::to_string(l) + ".";
        const TensorInfo* kb = findTensor(file, p + "attn_k_b.weight");
        const TensorInfo* vb = findTensor(file, p + "attn_v_b.weight");
        if(!kb || !vb || (int)kb->dims[0] != s.nope || (int)kb->dims[1] != s.kvLora || (int)kb->dims[2] != L.heads || (int)vb->dims[0] != s.kvLora || (int)vb->dims[1] != s.headValue || (int)vb->dims[2] != L.heads){
            problem = "unexpected hy_v4_flash k_b or v_b in layer " + std::to_string(l);
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
        // compuerta y valores del indexador son de 128x128: van en f32 para el kernel que junta los grupos
        for(int which = 0; which < 2 && L.indexed; which++){
            const TensorInfo* t = findTensor(file, p + (which == 0 ? "indexer_compressor_gate.weight" : "indexer_compressor_value.weight"));
            if(!t || t->dims[0] != 128 || t->dims[1] != 128 || !knownType(t->type)){
                problem = "unexpected hy_v4_flash indexer compressor in layer " + std::to_string(l);
                return false;
            }
            std::vector<float> full(128 * 128);
            uint64_t stride = rowBytes(t->type, 128);
            for(int r = 0; r < 128; r++) dequantRow(t->type, host(t) + r * stride, full.data() + r * 128, 128);
            uint64_t at = arena + used;
            used += full.size() * 4;
            if(cudaLink().copyUp(at, full.data(), full.size() * 4) != 0){
                problem = "cannot copy the indexer compressor to the gpu";
                return false;
            }
            (which == 0 ? L.idxGate : L.idxValue) = at;
        }
    }
    for(HyLayer& L : hyLayers){
        L.slots = L.sliding ? s.ring : contextLimit;
        L.cache = grab((uint64_t)L.slots * 576 * 2);
        cacheBytes += (uint64_t)L.slots * 576 * 2;
    }
    s.idxGroups = contextLimit / 4 + 1;
    for(HyLayer& L : hyLayers){
        if(!L.indexed) continue;
        L.idxKeys = grab((uint64_t)s.idxRing * 128 * 4);
        L.idxPools = grab((uint64_t)s.idxGroups * 128 * 4);
        cacheBytes += (uint64_t)s.idxRing * 128 * 4 + (uint64_t)s.idxGroups * 128 * 4;
    }

    b.streams = grab(C * d * s.hc * 4);
    b.flat = grab(C * d * s.hc * 4);
    b.mixes = grab(C * 16 * 4);
    if(b.mixes) cudaLink().memset(b.mixes, 0, (size_t)(C * 16 * 4));
    b.post = grab(C * 8 * 4);
    b.cur = grab(C * d * 4);
    b.xn = grab(C * d * 4);
    b.qa = grab(C * s.qLora * 4);
    b.q = grab(C * H * (s.nope + s.rope) * 4);
    b.kv = grab(C * 576 * 4);
    b.qfull = grab(C * H * 576 * 4);
    b.latentOut = grab(C * H * 512 * 4);
    b.heads = grab(C * H * s.headValue * 4);
    b.gateOut = grab(C * H * s.headValue * 4);
    b.attn = grab(C * d * 4);
    b.bigGate = grab(C * s.feedForward * 4);
    b.bigUp = grab(C * s.feedForward * 4);
    b.bigHid = grab(C * s.feedForward * 4);
    b.ffn = grab(C * d * 4);
    b.shared = grab(C * d * 4);
    b.sGate = grab(C * s.expertWidth * 4);
    b.sUp = grab(C * s.expertWidth * 4);
    b.sHid = grab(C * s.expertWidth * 4);
    b.routedIn = grab(C * s.latent * 4);
    b.routedOut = grab(C * s.latent * 4);
    b.logits = grab(C * s.experts * 4);
    b.positions = grab(C * 4);
    if(s.indexed){
        b.idxKey = grab(C * 128 * 4);
        b.idxQ = grab(C * 2048 * 4);
        b.idxW = grab(C * 16 * 4);
        b.idxScores = grab(C * s.idxGroups * 4);
        b.idxList = grab(C * s.idxWidth * 4);
        b.idxCount = grab(C * 4);
    }
    b.last = grab(d * 4);
    b.vocab = grab((uint64_t)s.vocab * 4);
    b.everyCap = 8;
    b.normedRows = grab((uint64_t)b.everyCap * d * 4);
    b.every = grab((uint64_t)b.everyCap * s.vocab * 4);
    b.hostEvery = (float*)shared((uint64_t)b.everyCap * s.vocab * 4, b.mappedEvery);
    b.scratchFloats = 16ULL << 20;
    b.attnScratch = grab(b.scratchFloats * 4);
    if(s.engramModules > 0){
        b.engramNormed = grab(C * s.engramHeads * s.engramDim * 4);
        b.engramKv = grab(C * (s.hc + 1) * d * 4);
    }
    uint64_t pairsMost = C * (uint64_t)s.used;
    uint64_t hitWidest = std::max<uint64_t>(s.latent, s.expertWidth);
    b.hitIn = grab(pairsMost * s.latent * 4);
    b.hitGate = grab(pairsMost * s.expertWidth * 4);
    b.hitUp = grab(pairsMost * s.expertWidth * 4);
    b.hitHid = grab(pairsMost * s.expertWidth * 4);
    b.hitDown = grab(pairsMost * s.latent * 4);
    b.hostHits = (int*)shared(pairsMost * 4 * 4, b.mappedHits);
    for(int slot = 0; slot < 2; slot++){
        PackedInput* packed = &packs[slot];
        int capacity = (int)std::max<uint64_t>(C * widestInput, pairsMost * hitWidest);
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
    if(s.engramModules > 0) b.hostEngram = (float*)shared(C * s.engramModules * s.engramHeads * s.engramDim * 4, b.mappedEngram);
    const char* wideAsked = getenv("SNEO_HY_WIDE_ROWS");
    b.wideRows = wideAsked ? std::max(64, atoi(wideAsked)) / 64 * 64 : 16384;
    b.hostWideIds = (int*)shared((uint64_t)b.wideRows * s.used * 4, b.wideIds);
    b.hostWideWeights = (float*)shared((uint64_t)b.wideRows * s.used * 4, b.wideWeights);
    if(s.engramModules > 0) b.hostWideEngram = (float*)shared((uint64_t)b.wideRows * s.engramModules * s.engramHeads * s.engramDim * 4, b.wideEngram);
    void* hostPos = nullptr;
    if(cudaLink().hostAlloc(&hostPos, (size_t)(C * 4), 0) == 0){
        pinned.push_back(hostPos);
        b.hostPositions = (int*)hostPos;
    }
    if(!problem.empty() || !b.hostEmbed || !b.hostLogits || !b.hostLatent || !b.hostMoe || !b.hostIds || !b.hostWeights || !b.hostPositions || (s.engramModules > 0 && !b.hostEngram)){
        if(problem.empty()) problem = "cannot pin host memory for hy_v4_flash";
        return false;
    }
    if(queue.streamCreate(&stream, 0) != 0){
        problem = "cannot create a cuda stream";
        return false;
    }
    // cublas solo para la absorcion y expansion de muchas filas: si no esta se usan los kernels propios
    if(blasLink().ready && !getenv("SNEO_HY_NO_BLAS")){
        HMODULE lib = GetModuleHandleA("cublas64_12.dll");
        b.blasBatched = lib ? (void*)GetProcAddress(lib, "cublasGemmStridedBatchedEx") : nullptr;
        b.q16 = grab(C * H * (s.nope + s.rope) * 2);
        b.lat16 = grab(C * H * s.kvLora * 2);
        if(b.blasBatched && b.q16 && b.lat16 && blasLink().create(&b.blas) == 0 && blasLink().setStream(b.blas, stream) != 0){
            blasLink().destroy(b.blas);
            b.blas = nullptr;
        }
        if(!b.q16 || !b.lat16) problem.clear();
    }
    size_t nowFree = 0, nowTotal = 0;
    cudaLink().info(&nowFree, &nowTotal);
    vramHeld = startFree > nowFree ? startFree - nowFree : 0;
    residentBytes = 0;
    residentExperts = 0;
    if(!getenv("SNEO_HY_NO_DRAFT")){
        loadSay("Loading the dfly2 drafter");
        if(hyLoadDraft(path)){
            drafting = true;
            int most = std::max(chunk, b.wideSub);
            for(HyDraftLayer& D : hyDraft.layers){
                HyLayer& T = hyLayers[D.target];
                if(!T.indexed || T.idxSnap) continue;
                T.idxSnap = grab((uint64_t)most * s.idxWidth * 4);
                T.idxSnapCount = grab((uint64_t)most * 4);
                T.idxSnapCap = most;
            }
            hyDraft.listWidth = s.idxWidth + 64;
            hyDraft.list = grab(8ULL * hyDraft.listWidth * 4);
            hyDraft.counts = grab(8ULL * 4);
            std::cout << "hy_v4_flash: dfly2 drafter loaded (" << (hyDraft.bytes >> 20) << " MB), it guesses " << hyDraft.length << " tokens per step" << std::endl;
        } else std::cout << "hy_v4_flash: dfly2 drafter not loaded: " << hyDraft.problem << std::endl;
    }
    // con lo que queda libre se guardan expertos enteros: compuerta, subida y bajada seguidas
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
        int most = s.layers * s.experts;
        want = std::min(want, most);
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
    if(cache.slotBytes){
        MEMORYSTATUSEX mem = {sizeof(mem)};
        GlobalMemoryStatusEx(&mem);
        const char* asked = getenv("SNEO_HY_WARM_GB");
        // medido 2026-11-18 con la 3090 sola: que el kernel lea los pesos por pcie es lento (espera 103 ms contra 29), va apagado
        // con SNEO_HY_WARM_GB se llena al cargar y la cpu lee esas mismas copias, asi no duplica ram
        uint64_t want = asked
            ? (uint64_t)(atof(asked) * (1ULL << 30))
            : 0;
        if(const char* share = getenv("SNEO_HY_WARM_SHARE")) cache.warmShare = (float)atof(share);
        int count = (int)(want / cache.slotBytes);
        while(count > 0){
            void* p = nullptr;
            uint64_t alias = 0;
            if(cudaLink().hostAlloc(&p, (size_t)count * cache.slotBytes, 3) == 0){
                if(queue.hostDevice(&alias, p, 0) == 0){
                    pinned.push_back(p);
                    cache.warmHost = (uint8_t*)p;
                    cache.warmDevice = alias;
                    break;
                }
            }
            count = count * 3 / 4;
        }
        cache.warmSlots = cache.warmHost ? count : 0;
    }
    if(cache.slots > 0){
        uint64_t rows = (uint64_t)b.wideRows, need = 0, most = (uint64_t)b.widePairsMost, ew = (uint64_t)s.expertWidth;
        for(uint64_t bytes : {rows * d * s.hc * 4, rows * d * 4, rows * s.latent * 4, rows * s.latent * 4, rows * s.hc * 4, rows * 4,
            rows * s.latent, rows * s.latent / 8, rows * s.latent / 8, most * ew * 4, most * ew * 4, most * ew * 4, most * ew, most * ew / 8, most * ew / 8}) need += (bytes + 255) / 256 * 256;
        for(const auto& item : hySubBuffers(b.wideSub)) need += (item.second + 255) / 256 * 256;
        b.wideStage = 128;
        int borrow = b.wideStage + (int)((need + cache.slotBytes - 1) / cache.slotBytes);
        b.wideBorrow = cache.slots > borrow * 4 ? borrow : 0;
        const char* least = getenv("SNEO_HY_WIDE_MIN");
        b.wideMin = least ? atoi(least) : 512;
        void* host = nullptr;
        if(b.wideBorrow && !getenv("SNEO_HY_WIDE_SYNC") && cudaLink().hostAlloc(&host, (size_t)(b.wideStage * cache.slotBytes), 0) == 0){
            pinned.push_back(host);
            b.widePinned = (uint8_t*)host;
            b.hostWidePairs = (int*)shared(6ULL * b.wideRows * s.used * 4, b.widePairs);
            bool ok = b.hostWidePairs && queue.streamCreate(&b.wideStream, 1) == 0;
            for(int h = 0; h < 2 && ok; h++) ok = queue.eventCreate(&b.wideCopied[h], 2) == 0 && queue.eventCreate(&b.wideDone[h], 2) == 0;
            if(!ok) b.widePinned = nullptr;
        }
    }
    cache.warmOwner.assign(cache.warmSlots, -1);
    cache.warmFree.clear();
    for(int i = cache.warmSlots - 1; i >= 0; i--) cache.warmFree.push_back(i);
    std::cout << "hy_v4_flash: " << cache.slots << " expert slots in video memory (" << (((uint64_t)cache.slots * cache.slotBytes) >> 20) << " MB), " << cache.warmSlots << " in pinned ram the gpu reads directly (" << (((uint64_t)cache.warmSlots * cache.slotBytes) >> 20) << " MB), the rest run on the cpu" << std::endl;
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
        cache.heatPath = (folder / (stem + ".hy4heat")).string();
        FILE* f = fopen(cache.heatPath.c_str(), "rb");
        if(f){
            std::vector<uint32_t> saved(cache.heat.size());
            if(fread(saved.data(), 4, saved.size(), f) == saved.size()){
                // la mitad de lo viejo para que lo nuevo pese mas con el tiempo
                for(size_t i = 0; i < saved.size(); i++) cache.heat[i] = saved[i] / 2 + (saved[i] ? 1 : 0);
            }
            fclose(f);
        }
        std::vector<int> order;
        for(size_t i = 0; i < cache.heat.size(); i++){
            if(cache.heat[i] && !hyLayers[i / s.experts].dense) order.push_back((int)i);
        }
        std::sort(order.begin(), order.end(), [&](int a, int b){ return cache.heat[a] > cache.heat[b]; });
        int place = std::min((int)order.size(), (int)cache.free.size());
        if(place > 0) loadSay("Placing the " + std::to_string(place) + " most used experts on the gpu");
        for(int i = 0; i < place && !failed; i++){
            int slot = cache.free.back();
            cache.free.pop_back();
            hyUpload(order[i] / s.experts, order[i] % s.experts, slot);
        }
        cache.uploads = 0;
        std::cout << "hy_v4_flash: " << place << " experts placed in video memory from " << (order.empty() ? std::string("no heat map yet, it learns while you use it") : std::filesystem::path(cache.heatPath).filename().string()) << std::endl;
    }
    hyOpenHelper(sizes + quantKernelSource + modelKernelSource + gemmaKernelSource + hyKernelSource, blob);
    hyFillWarm();
    hyFillRam();
    profile = getenv("SNEO_HY_PROFILE") != nullptr;
    pool.start(std::max(0, threads - 1));
    ready = true;
    reset();
    return true;
}

// la segunda tarjeta: sus kernels, buffers para las filas de una capa y los expertos que siguen en calor despues de la 3090
inline void GpuModel::hyOpenHelper(const std::string& source, const std::vector<uint8_t>& blob) {
    CudaLink& cu = cudaLink();
    CudaQueue& queue = cudaQueue();
    HyShape& s = hyShape;
    HyCache& cache = hyCache;
    if(!useHelper || cu.count < 2 || !cu.deviceGet || !cu.retain || !cu.attribute || getenv("SNEO_HY_NO_HELPER")) return;
    for(int d = 0; d < cu.count && helper.device < 0; d++){
        int dev = -1, major = 0, minor = 0;
        if(helperOrdinal >= 0 && d != helperOrdinal) continue;
        if(cu.deviceGet(&dev, d) != 0 || dev == cu.device) continue;
        cu.attribute(&major, 75, dev);
        cu.attribute(&minor, 76, dev);
        if(major * 10 + minor < 61) continue;
        helper.device = dev;
        helper.major = major;
        helper.minor = minor;
    }
    if(helper.device < 0) return;
    if(cu.retain(&helper.context, helper.device) != 0){
        helper.context = nullptr;
        helper.device = -1;
        return;
    }
    char bus[32] = {};
    NvmlLink& nv = nvmlLink();
    void* card = nullptr;
    if(cu.busId && cu.busId(bus, (int)sizeof(bus), helper.device) == 0 && nv.ready && nv.byBus && nv.byBus(bus, &card) == 0 && card){
        char name[96] = {};
        if(nv.getName(card, name, sizeof(name)) == 0) helper.name = name;
        helper.card = card;
    }
    if(helper.name.empty()) helper.name = "gpu " + std::to_string(helper.device);
    loadSay("Building the kernels for " + helper.name);
    bool ok = buildModule(source, helper.module, helper.context, helper.major, helper.minor, false);
    if(!ok) helper.problem = "the kernels did not build for " + helper.name + ": " + helper.module.log;
    auto take = [&](uint64_t bytes) -> uint64_t {
        uint64_t ptr = 0;
        if(cu.alloc(&ptr, (size_t)std::max<uint64_t>(bytes, 256)) != 0){
            ok = false;
            return 0;
        }
        helper.allocations.push_back(ptr);
        return ptr;
    };
    uint64_t C = (uint64_t)chunk, pairs = C * s.used, latent = (uint64_t)s.latent, ew = (uint64_t)s.expertWidth, wide = std::max(latent, ew);
    uint64_t table = ok ? take(blob.size()) : 0;
    ok = ok && cu.copyUp(table, blob.data(), blob.size()) == 0;
    if(ok) helper.tables = {table, table + 2048, table + 6144, table + 14336, table + 15360, table + 17408, table + 17536, table + 17552};
    if(ok){
        cache.helpIn = take(C * latent * 4);
        cache.helpWeights = take(pairs * 4);
        cache.helpSlotsOn = take(pairs * 4);
        cache.helpPairs = take(pairs * 4);
        cache.helpOut = take(C * latent * 4);
        cache.helpHitIn = take(pairs * latent * 4);
        cache.helpGate = take(pairs * ew * 4);
        cache.helpUp = take(pairs * ew * 4);
        cache.helpHid = take(pairs * ew * 4);
        cache.helpDown = take(pairs * latent * 4);
        cache.helpValues = take(pairs * wide);
        cache.helpScales = take(pairs * wide / 8);
        cache.helpSums = take(pairs * wide / 8);
    }
    void* host = nullptr;
    if(ok && cu.hostAlloc(&host, (size_t)(C * latent * 4), 0) == 0){
        helper.pinned.push_back(host);
        cache.helpHostOut = (float*)host;
    } else ok = false;
    if(ok && cu.hostAlloc(&host, (size_t)(pairs * 2 * 4), 0) == 0){
        helper.pinned.push_back(host);
        cache.helpHostLists = (int*)host;
    } else ok = false;
    if(ok && queue.eventCreate(&helper.done, 2) != 0){
        helper.done = nullptr;
        ok = false;
    }
    // lo que queda libre en la tarjeta va a expertos enteros, como en la 3090
    size_t freeBytes = 0, totalBytes = 0;
    cu.info(&freeBytes, &totalBytes);
    const char* capAsked = getenv("SNEO_HY_HELPER_GB");
    uint64_t keep = 600ULL << 20;
    uint64_t room = freeBytes > keep ? freeBytes - keep : 0;
    if(capAsked) room = std::min<uint64_t>(room, (uint64_t)(atof(capAsked) * (1ULL << 30)));
    else if(helperCap > 0) room = std::min<uint64_t>(room, helperCap);
    int count = ok && cache.slotBytes ? (int)(room / cache.slotBytes) : 0;
    while(count > 0){
        uint64_t ptr = 0;
        if(cu.alloc(&ptr, (size_t)count * cache.slotBytes) == 0){
            helper.allocations.push_back(ptr);
            cache.helpBase = ptr;
            break;
        }
        count = count * 9 / 10;
    }
    cache.helpSlots = cache.helpBase ? count : 0;
    cache.helpOf.assign((size_t)s.layers * s.experts, -1);
    std::vector<int> order;
    for(size_t i = 0; i < cache.heat.size(); i++){
        if(cache.heat[i] && !hyLayers[i / s.experts].dense && cache.slotOf[i] < 0) order.push_back((int)i);
    }
    std::sort(order.begin(), order.end(), [&](int a, int b){ return cache.heat[a] > cache.heat[b]; });
    int place = std::min((int)order.size(), cache.helpSlots);
    if(place > 0) loadSay("Placing " + std::to_string(place) + " more experts on " + helper.name);
    for(int i = 0; i < place && ok; i++){
        int l = order[i] / s.experts, e = order[i] % s.experts;
        if(!hyCopyExpert(l, e, cache.helpBase + (uint64_t)i * cache.slotBytes)){
            ok = false;
            break;
        }
        cache.helpOf[order[i]] = i;
    }
    cu.setCurrent(cu.context);
    if(!ok || place <= 0){
        if(helper.problem.empty()) helper.problem = place <= 0 ? helper.name + " has no room or no heat map yet for experts" : "could not set up " + helper.name;
        std::cout << "hy_v4_flash: second card not used: " << helper.problem << std::endl;
        cache.helpOf.assign(cache.helpOf.size(), -1);
        cache.helpSlots = 0;
        closeHelper();
        return;
    }
    helper.ready = true;
    helper.perLayer = place / std::max(1, s.layers - 1);
    helper.bytes = (uint64_t)place * cache.slotBytes;
    std::cout << "hy_v4_flash: " << place << " more experts on " << helper.name << " (" << (helper.bytes >> 20) << " MB), it computes them while the cpu does the rest" << std::endl;
}

// la parte de la capa que esta en la segunda tarjeta: entra la fila latente y sale la suma pesada, todo en su cola
inline bool GpuModel::hyHelperLaunch(HyLayer& L, int n, const std::vector<int>& slots, const std::vector<int>& pairsList) {
    CudaLink& cu = cudaLink();
    CudaQueue& queue = cudaQueue();
    HyShape& s = hyShape;
    HyBuffers& b = hyBuffers;
    HyCache& cache = hyCache;
    int h = (int)slots.size();
    if(h == 0) return false;
    int used = s.used, latent = s.latent, ew = s.expertWidth;
    uint64_t upAt = (L.gateBytes + 255) / 256 * 256, downAt = upAt + (L.upBytes + 255) / 256 * 256;
    bool ok = cu.setCurrent(helper.context) == 0;
    auto run = [&](const char* name, unsigned int gx, unsigned int gy, unsigned int bx, unsigned int by, auto... values){
        void* args[] = {(void*)&values...};
        void* fn = kernelFunction(helper.module, name);
        if(!fn || cu.launch(fn, gx, gy, 1, bx, by, 1, 0, nullptr, args, nullptr) != 0) ok = false;
    };
    int* hostSlots = cache.helpHostLists;
    int* hostPairs = cache.helpHostLists + (uint64_t)chunk * used;
    memcpy(hostSlots, slots.data(), (size_t)h * 4);
    memcpy(hostPairs, pairsList.data(), (size_t)h * 4);
    ok = ok && queue.copyUp(cache.helpIn, b.hostLatent, (size_t)n * latent * 4, nullptr) == 0;
    ok = ok && queue.copyUp(cache.helpWeights, b.hostWeights, (size_t)n * used * 4, nullptr) == 0;
    ok = ok && queue.copyUp(cache.helpSlotsOn, hostSlots, (size_t)h * 4, nullptr) == 0;
    ok = ok && queue.copyUp(cache.helpPairs, hostPairs, (size_t)h * 4, nullptr) == 0;
    std::string gateName = "gemmaExperts" + std::to_string(L.hostGate.type);
    std::string upName = "gemmaExperts" + std::to_string(L.hostUp.type);
    std::string downName = "gemmaExperts" + std::to_string(L.hostDown.type);
    run("hyFill", (unsigned int)(((uint64_t)n * latent + 255) / 256), 1, 256, 1, cache.helpOut, n * latent);
    run("hyGatherPack", (unsigned int)(((uint64_t)h * latent / 32 + 7) / 8), 1, 32, 8, cache.helpIn, cache.helpPairs, cache.helpHitIn, latent, used, h, cache.helpValues, cache.helpScales, cache.helpSums);
    run(gateName.c_str(), (unsigned int)((L.hostGate.rows + 7) / 8), (unsigned int)h, 256, 1, cache.helpBase, L.hostGate.stride, cache.slotBytes, cache.helpSlotsOn, 1, cache.helpValues, cache.helpScales, cache.helpSums, cache.helpGate, L.hostGate.rows, L.hostGate.cols, helper.tables);
    run(upName.c_str(), (unsigned int)((L.hostUp.rows + 7) / 8), (unsigned int)h, 256, 1, cache.helpBase + upAt, L.hostUp.stride, cache.slotBytes, cache.helpSlotsOn, 1, cache.helpValues, cache.helpScales, cache.helpSums, cache.helpUp, L.hostUp.rows, L.hostUp.cols, helper.tables);
    run("hySwigluPack", (unsigned int)(((uint64_t)h * ew / 32 + 7) / 8), 1, 32, 8, cache.helpGate, cache.helpUp, cache.helpHid, h * ew, L.clamp, cache.helpValues, cache.helpScales, cache.helpSums);
    run(downName.c_str(), (unsigned int)((L.hostDown.rows + 7) / 8), (unsigned int)h, 256, 1, cache.helpBase + downAt, L.hostDown.stride, cache.slotBytes, cache.helpSlotsOn, 1, cache.helpValues, cache.helpScales, cache.helpSums, cache.helpDown, L.hostDown.rows, L.hostDown.cols, helper.tables);
    run("hyScatter", (unsigned int)h, 1, 256, 1, cache.helpDown, cache.helpPairs, cache.helpWeights, cache.helpOut, latent, used);
    ok = ok && queue.copyDown(cache.helpHostOut, cache.helpOut, (size_t)n * latent * 4, nullptr) == 0;
    ok = ok && queue.eventRecord(helper.done, nullptr) == 0;
    cu.setCurrent(cu.context);
    if(!ok){
        launchProblem = helper.name + " failed computing its experts";
        failed = true;
    }
    return ok;
}

// la ram fijada que la 3090 lee directo se llena al cargar con los que siguen en calor despues de los de la vram
// mientras escribe no se cambia: copiar de a uno en cada capa costaba mas de lo que ahorraba
inline void GpuModel::hyFillWarm() {
    HyShape& s = hyShape;
    HyCache& cache = hyCache;
    if(!cache.warmSlots) return;
    std::vector<int> order;
    for(int i = 0; i < (int)cache.slotOf.size(); i++){
        if(!hyLayers[i / s.experts].dense && cache.slotOf[i] < 0 && (cache.helpOf.empty() || cache.helpOf[i] < 0)) order.push_back(i);
    }
    std::stable_sort(order.begin(), order.end(), [&](int a, int b){ return cache.heat[a] > cache.heat[b]; });
    int place = std::min((int)order.size(), cache.warmSlots);
    if(place <= 0) return;
    loadSay("Placing " + std::to_string(place) + " experts in pinned memory the gpu reads directly");
    std::atomic<int> next{0};
    std::vector<std::thread> crew;
    for(int w = 0; w < 8; w++){
        crew.emplace_back([&]{
            for(int i = next++; i < place; i = next++){
                int l = order[i] / s.experts, e = order[i] % s.experts;
                const HyLayer& L = hyLayers[l];
                uint8_t* at = cache.warmHost + (uint64_t)i * cache.slotBytes;
                uint64_t upAt = (L.gateBytes + 255) / 256 * 256, downAt = upAt + (L.upBytes + 255) / 256 * 256;
                memcpy(at, L.hostGate.data + (uint64_t)e * L.gateBytes, (size_t)L.gateBytes);
                memcpy(at + upAt, L.hostUp.data + (uint64_t)e * L.upBytes, (size_t)L.upBytes);
                memcpy(at + downAt, L.hostDown.data + (uint64_t)e * L.downBytes, (size_t)L.downBytes);
            }
        });
    }
    for(std::thread& t : crew) t.join();
    for(int i = 0; i < place; i++){
        cache.warmOf[order[i]] = i;
        cache.warmOwner[i] = order[i];
    }
    cache.warmFree.clear();
    for(int i = cache.warmSlots - 1; i >= place; i--) cache.warmFree.push_back(i);
    cache.warmBudget = 0;
    std::cout << "hy_v4_flash: " << place << " experts in pinned memory (" << (((uint64_t)place * cache.slotBytes) >> 20) << " MB), the gpu takes " << (int)(cache.warmShare * 100) << "% of the missing ones from there while the cpu does the rest" << std::endl;
}

inline void GpuModel::hySaveHeat() {
    if(hyCache.heatPath.empty() || hyCache.total == hyCache.heatSaved) return;
    hyCache.heatSaved = hyCache.total;
    std::string temp = hyCache.heatPath + ".part";
    FILE* f = fopen(temp.c_str(), "wb");
    if(!f) return;
    bool ok = fwrite(hyCache.heat.data(), 4, hyCache.heat.size(), f) == hyCache.heat.size();
    fclose(f);
    std::error_code e;
    if(ok) std::filesystem::rename(temp, hyCache.heatPath, e);
}

// los expertos que calcula la cpu se leen a la ram antes de decir que cargo, los mas usados primero
// no entra todo: la tabla engram sola pesa 59 GB, asi que se deja margen para windows y lo demas
inline void GpuModel::hyFillRam() {
    HyShape& s = hyShape;
    HyCache& cache = hyCache;
    // lo que se leyo del archivo para subir a la gpu (atencion, k_b, v_b) queda en la cache de windows y no en este proceso
    SetProcessWorkingSetSizeEx(GetCurrentProcess(), (SIZE_T)-1, (SIZE_T)-1, 0);
    MEMORYSTATUSEX mem = {sizeof(mem)};
    GlobalMemoryStatusEx(&mem);
    // con la app, el navegador y la tabla engram que crece al usarse, 6 GB libres no alcanzaban: windows sacaba expertos de la ram
    const char* freeAsked = getenv("SNEO_HY_FREE_GB");
    uint64_t keep = freeAsked ? (uint64_t)(atof(freeAsked) * (1ULL << 30)) : std::max<uint64_t>(12ULL << 30, mem.ullTotalPhys / 8);
    uint64_t room = mem.ullAvailPhys > keep ? mem.ullAvailPhys - keep : 0;
    if(hostBudget > 0) room = std::min(room, hostBudget);
    std::vector<int> order;
    for(int i = 0; i < (int)cache.slotOf.size(); i++){
        if(!hyLayers[i / s.experts].dense && cache.slotOf[i] < 0 && (cache.helpOf.empty() || cache.helpOf[i] < 0) && cache.warmOf[i] < 0) order.push_back(i);
    }
    std::stable_sort(order.begin(), order.end(), [&](int a, int b){ return cache.heat[a] > cache.heat[b]; });
    std::vector<int> chosen;
    uint64_t want = 0;
    for(int id : order){
        const HyLayer& L = hyLayers[id / s.experts];
        uint64_t bytes = L.gateBytes + L.upBytes + L.downBytes;
        if(want + bytes > room) break;
        chosen.push_back(id);
        want += bytes;
    }
    if(chosen.empty()) return;
    SetProcessWorkingSetSizeEx(GetCurrentProcess(), (SIZE_T)(want + (1ULL << 30)), (SIZE_T)mem.ullTotalPhys, QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE);
    auto start = std::chrono::steady_clock::now();
    std::atomic<size_t> next{0};
    std::atomic<uint64_t> done{0};
    std::atomic<int> finished{0};
    const size_t batch = 24;
    std::vector<std::thread> readers;
    for(int t = 0; t < 8; t++){
        readers.emplace_back([&]{
            std::vector<WIN32_MEMORY_RANGE_ENTRY> ranges;
            uint32_t sink = 0;
            for(size_t at = next.fetch_add(batch); at < chosen.size(); at = next.fetch_add(batch)){
                ranges.clear();
                uint64_t bytes = 0;
                for(size_t i = at; i < std::min(at + batch, chosen.size()); i++){
                    int l = chosen[i] / s.experts, e = chosen[i] % s.experts;
                    const HyLayer& L = hyLayers[l];
                    ranges.push_back({(PVOID)(L.hostGate.data + (uint64_t)e * L.gateBytes), (SIZE_T)L.gateBytes});
                    ranges.push_back({(PVOID)(L.hostUp.data + (uint64_t)e * L.upBytes), (SIZE_T)L.upBytes});
                    ranges.push_back({(PVOID)(L.hostDown.data + (uint64_t)e * L.downBytes), (SIZE_T)L.downBytes});
                    bytes += L.gateBytes + L.upBytes + L.downBytes;
                }
                PrefetchVirtualMemory(GetCurrentProcess(), ranges.size(), ranges.data(), 0);
                for(const WIN32_MEMORY_RANGE_ENTRY& r : ranges){
                    const volatile uint8_t* p = (const volatile uint8_t*)r.VirtualAddress;
                    for(SIZE_T step = 0; step < r.NumberOfBytes; step += 4096) sink += p[step];
                }
                done += bytes;
            }
            touched += sink & 1;
            finished++;
        });
    }
    auto gb = [](uint64_t b){ char text[32]; snprintf(text, sizeof(text), "%.1f", b / 1073741824.0); return std::string(text); };
    while(finished < (int)readers.size()){
        loadSay("Reading the experts into ram " + gb(done) + " of " + gb(want) + " GB");
        Sleep(300);
    }
    for(std::thread& t : readers) t.join();
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << "hy_v4_flash: " << chosen.size() << " of " << order.size() << " cpu experts read into ram (" << gb(want) << " GB) in " << (int)seconds << " s, the rest stay on disk until used" << std::endl;
}

inline void GpuModel::hyBatch(HyLayer& L, uint64_t base, const int* slots, const int* pairs, uint64_t slotsOn, uint64_t pairsOn, int count) {
    HyShape& s = hyShape;
    HyBuffers& b = hyBuffers;
    HyCache& cache = hyCache;
    int latent = s.latent, width = s.expertWidth, used = s.used;
    auto blocks = [](uint64_t n){ return (unsigned int)((n + 255) / 256); };
    uint64_t upAt = base + (L.gateBytes + 255) / 256 * 256;
    uint64_t downAt = upAt + (L.upBytes + 255) / 256 * 256;
    PackedInput& p = packs[0];
    go("hyGatherPack", (unsigned int)(((uint64_t)count * latent / 32 + 7) / 8), 1, 32, 8, b.routedIn, pairsOn, b.hitIn, latent, used, count, p.values, p.scales, p.sums);
    std::string gateName = "gemmaExperts" + std::to_string(L.hostGate.type);
    std::string upName = "gemmaExperts" + std::to_string(L.hostUp.type);
    std::string downName = "gemmaExperts" + std::to_string(L.hostDown.type);
    go(gateName.c_str(), (unsigned int)((L.hostGate.rows + 7) / 8), (unsigned int)count, 256, 1, base, L.hostGate.stride, cache.slotBytes, slotsOn, 1, p.values, p.scales, p.sums, b.hitGate, L.hostGate.rows, L.hostGate.cols, tables);
    go(upName.c_str(), (unsigned int)((L.hostUp.rows + 7) / 8), (unsigned int)count, 256, 1, upAt, L.hostUp.stride, cache.slotBytes, slotsOn, 1, p.values, p.scales, p.sums, b.hitUp, L.hostUp.rows, L.hostUp.cols, tables);
    go("hySwigluPack", (unsigned int)(((uint64_t)count * width / 32 + 7) / 8), 1, 32, 8, b.hitGate, b.hitUp, b.hitHid, count * width, L.clamp, p.values, p.scales, p.sums);
    go(downName.c_str(), (unsigned int)((L.hostDown.rows + 7) / 8), (unsigned int)count, 256, 1, downAt, L.hostDown.stride, cache.slotBytes, slotsOn, 1, p.values, p.scales, p.sums, b.hitDown, L.hostDown.rows, L.hostDown.cols, tables);
    go("hyScatter", (unsigned int)count, 1, 256, 1, b.hitDown, pairsOn, b.mappedWeights, b.routedOut, latent, used);
}

inline bool GpuModel::hyCopyExpert(int l, int e, uint64_t at) {
    HyLayer& L = hyLayers[l];
    HyCache& cache = hyCache;
    int warm = cache.warmOf.empty() ? -1 : cache.warmOf[(size_t)l * hyShape.experts + e];
    if(warm >= 0) return cudaLink().copyUp(at, cache.warmHost + (uint64_t)warm * cache.slotBytes, (size_t)cache.slotBytes) == 0;
    uint64_t upAt = at + (L.gateBytes + 255) / 256 * 256;
    uint64_t downAt = upAt + (L.upBytes + 255) / 256 * 256;
    return cudaLink().copyUp(at, L.hostGate.data + (uint64_t)e * L.gateBytes, (size_t)L.gateBytes) == 0 &&
       cudaLink().copyUp(upAt, L.hostUp.data + (uint64_t)e * L.upBytes, (size_t)L.upBytes) == 0 &&
       cudaLink().copyUp(downAt, L.hostDown.data + (uint64_t)e * L.downBytes, (size_t)L.downBytes) == 0;
}

inline void GpuModel::hyUpload(int l, int e, int slot) {
    HyCache& cache = hyCache;
    if(!hyCopyExpert(l, e, cache.base + (uint64_t)slot * cache.slotBytes)){
        failed = true;
        return;
    }
    size_t key = (size_t)l * hyShape.experts + e;
    if(cache.owner[slot] >= 0) cache.slotOf[cache.owner[slot]] = -1;
    cache.owner[slot] = (int)key;
    cache.slotOf[key] = slot;
    cache.uploads++;
    // la copia en la ram fijada se queda: si despues sale de la vram la cpu la sigue leyendo de ahi y no del disco
}

// reparte los expertos de la capa: los de la vram en la gpu y el resto en la cpu al mismo tiempo
inline void GpuModel::hyExperts(int l, int n) {
    HyShape& s = hyShape;
    HyBuffers& b = hyBuffers;
    HyCache& cache = hyCache;
    HyLayer& L = hyLayers[l];
    int used = s.used, latent = s.latent, width = s.expertWidth;
    int pairs = n * used;
    std::vector<char> onCpu(pairs, 1);
    auto blocks = [](uint64_t count){ return (unsigned int)((count + 255) / 256); };
    go("hyFill", blocks((uint64_t)n * latent), 1, 256, 1, b.routedOut, n * latent);
    bool typed = !legacy;
    for(uint32_t type : {L.hostGate.type, L.hostUp.type, L.hostDown.type}) typed = typed && (type == 12 || type == 13 || type == 14 || type == 8 || type == 7 || type == 20 || type == 21 || type == 23 || type == 42);
    // con el enrutado rapido los de menos peso que no estan en la vram no se calculan, como en los otros modelos
    static const int ranksAsked = getenv("SNEO_HY_TAIL") ? atoi(getenv("SNEO_HY_TAIL")) : 0;
    // medido en la 3090 sola: con 9 rangos los que no estan en la vram van de 25 a 37 tok/s escribiendo codigo y la aguja se sigue encontrando
    int tail = ranksAsked > 0 ? std::min(used, ranksAsked) : tailSkip > 0 ? std::min(used, 9) : used;
    int hits = 0;
    int* hitSlot = b.hostHits;
    int* hitPair = b.hostHits + pairs;
    // los que estan en la segunda tarjeta los calcula ella, aunque sean de los de menos peso: no le cuestan a la cpu
    bool helping = helper.ready && typed && cache.helpSlots > 0;
    std::vector<int> helpSlot, helpPair;
    for(int i = 0; i < pairs; i++){
        int e = b.hostIds[i];
        if(e < 0 || e >= s.experts) continue;
        size_t key = (size_t)l * s.experts + e;
        cache.heat[key]++;
        cache.total++;
        int slot = cache.slotOf[key];
        if(slot < 0 && helping && cache.helpOf[key] >= 0){
            onCpu[i] = 0;
            helpSlot.push_back(cache.helpOf[key]);
            helpPair.push_back(i);
            cache.helpUsed++;
            continue;
        }
        if(slot < 0){
            if(i % used >= tail){
                onCpu[i] = 0;
                cache.skipped++;
            }
            continue;
        }
        onCpu[i] = 0;
        cache.hits++;
        if(typed){
            hitSlot[hits] = slot;
            hitPair[hits] = i;
            hits++;
            continue;
        }
        int t = i / used;
        uint64_t at = cache.base + (uint64_t)slot * cache.slotBytes;
        uint64_t upAt = at + (L.gateBytes + 255) / 256 * 256;
        uint64_t downAt = upAt + (L.upBytes + 255) / 256 * 256;
        GpuMatrix g = {at, L.hostGate.type, L.hostGate.rows, L.hostGate.cols, L.hostGate.stride};
        GpuMatrix u = {upAt, L.hostUp.type, L.hostUp.rows, L.hostUp.cols, L.hostUp.stride};
        GpuMatrix dn = {downAt, L.hostDown.type, L.hostDown.rows, L.hostDown.cols, L.hostDown.stride};
        uint64_t row = b.routedIn + 4ULL * t * latent;
        hyPack(row, latent, 0);
        project(g, row, 0, b.bigGate, 1);
        project(u, row, 0, b.bigUp, 1);
        go("hySwiglu", blocks(width), 1, 256, 1, b.bigGate, b.bigUp, b.bigHid, width, L.clamp);
        hyPack(b.bigHid, width, 0);
        project(dn, b.bigHid, 0, b.bigGate, 1);
        go("hyAxpy", blocks(latent), 1, 256, 1, b.routedOut + 4ULL * t * latent, b.bigGate, b.mappedWeights + 4ULL * i, latent);
    }
    // de los que faltan una parte la hace la gpu leyendo la ram fijada, asi no se queda esperando a la cpu
    int warmCount = 0;
    if(typed && cache.warmSlots){
        int missing = 0;
        for(int i = 0; i < pairs; i++) missing += onCpu[i] ? 1 : 0;
        int give = (int)(missing * cache.warmShare + 0.5f);
        int* warmSlot = b.hostHits + 2 * pairs;
        int* warmPair = b.hostHits + 3 * pairs;
        for(int i = 0; i < pairs && warmCount < give; i++){
            if(!onCpu[i]) continue;
            int e = b.hostIds[i];
            int slot = cache.warmOf[(size_t)l * s.experts + e];
            if(slot < 0) continue;
            warmSlot[warmCount] = slot;
            warmPair[warmCount] = i;
            warmCount++;
            onCpu[i] = 0;
        }
        cache.warmUsed += warmCount;
    }
    // la segunda tarjeta arranca primero: trabaja mientras la 3090 hace los suyos y la cpu el resto
    bool helped = !helpSlot.empty() && hyHelperLaunch(L, n, helpSlot, helpPair);
    if(warmCount > 0) hyBatch(L, cache.warmDevice, nullptr, nullptr, b.mappedHits + 8ULL * pairs, b.mappedHits + 12ULL * pairs, warmCount);
    if(hits > 0) hyBatch(L, cache.base, nullptr, nullptr, b.mappedHits, b.mappedHits + 4ULL * pairs, hits);
    auto cpuFrom = std::chrono::steady_clock::now();
    hyExpertsHost(L, n, onCpu);
    cache.cpuMs += since(cpuFrom);
    if(helped){
        auto helpFrom = std::chrono::steady_clock::now();
        CudaLink& cu = cudaLink();
        if(cu.setCurrent(helper.context) != 0 || cudaQueue().eventSync(helper.done) != 0) failed = true;
        cu.setCurrent(cu.context);
        // lo que hizo la segunda tarjeta se suma a lo de la cpu antes de subirlo a la 3090
        float* moe = b.hostMoe;
        const float* add = cache.helpHostOut;
        for(int i = 0; i < n * latent; i++) moe[i] += add[i];
        cache.helpMs += since(helpFrom);
    }
    go("hyAdd", blocks((uint64_t)n * latent), 1, 256, 1, b.routedOut, b.mappedMoe, n * latent);
    if(cache.warmSlots){
        int filled = 0;
        for(int i = 0; i < pairs && filled < cache.warmBudget; i++){
            int e = b.hostIds[i];
            if(e < 0 || e >= s.experts) continue;
            size_t key = (size_t)l * s.experts + e;
            if(cache.slotOf[key] >= 0 || cache.warmOf[key] >= 0) continue;
            int slot = -1;
            if(!cache.warmFree.empty()){
                slot = cache.warmFree.back();
                cache.warmFree.pop_back();
            } else {
                uint32_t lowest = cache.heat[key];
                for(int k = 0; k < 24; k++){
                    int pick = (int)((cache.uploads * 2654435761ULL + (uint64_t)k * 69069ULL + key * 7ULL + filled) % (uint64_t)cache.warmSlots);
                    int owner = cache.warmOwner[pick];
                    if(owner >= 0 && owner / s.experts == l) continue;
                    if(owner < 0 || cache.heat[owner] + 2 < lowest){
                        lowest = owner < 0 ? 0 : cache.heat[owner];
                        slot = pick;
                        if(owner < 0) break;
                    }
                }
            }
            if(slot < 0) continue;
            uint8_t* at = cache.warmHost + (uint64_t)slot * cache.slotBytes;
            memcpy(at, L.hostGate.data + (uint64_t)e * L.gateBytes, (size_t)L.gateBytes);
            memcpy(at + (L.gateBytes + 255) / 256 * 256, L.hostUp.data + (uint64_t)e * L.upBytes, (size_t)L.upBytes);
            memcpy(at + (L.gateBytes + 255) / 256 * 256 + (L.upBytes + 255) / 256 * 256, L.hostDown.data + (uint64_t)e * L.downBytes, (size_t)L.downBytes);
            if(cache.warmOwner[slot] >= 0) cache.warmOf[cache.warmOwner[slot]] = -1;
            cache.warmOwner[slot] = (int)key;
            cache.warmOf[key] = slot;
            filled++;
        }
    }
    if(!cache.slots) return;
    // sube a la vram los que fueron a la cpu: primero a lugares libres y despues en lugar de uno menos usado
    std::vector<char> busy(s.experts, 0);
    for(int i = 0; i < pairs; i++){
        int e = b.hostIds[i];
        if(e >= 0 && e < s.experts) busy[e] = 1;
    }
    int moved = 0;
    for(int i = 0; i < pairs && moved < cache.budget; i++){
        int e = b.hostIds[i];
        if(e < 0 || e >= s.experts) continue;
        size_t key = (size_t)l * s.experts + e;
        if(cache.slotOf[key] >= 0) continue;
        int slot = -1;
        if(!cache.free.empty()){
            slot = cache.free.back();
            cache.free.pop_back();
        } else {
            uint32_t lowest = cache.heat[key];
            for(int k = 0; k < 24; k++){
                int pick = (int)((cache.uploads * 2654435761ULL + (uint64_t)k * 40503ULL + key) % (uint64_t)cache.slots);
                int owner = cache.owner[pick];
                if(owner < 0){
                    slot = pick;
                    break;
                }
                int ownerLayer = owner / s.experts;
                if(ownerLayer == l && busy[owner % s.experts]) continue;
                if(cache.heat[owner] + 2 < lowest){
                    lowest = cache.heat[owner];
                    slot = pick;
                }
            }
        }
        if(slot < 0) continue;
        auto upFrom = std::chrono::steady_clock::now();
        hyUpload(l, e, slot);
        cache.uploadMs += since(upFrom);
        moved++;
    }
}

// expertos del bloque agrupados por experto: cada grupo lee su experto una vez para todas sus filas con tensor cores
// primero los que estan en la vram y despues los demas, que suben en dos mitades mientras la gpu calcula la otra
inline void GpuModel::hyExpertsGrouped(int l, int n, uint64_t stage, const std::vector<std::vector<int>>& byExpert) {
    HyShape& s = hyShape;
    HyBuffers& b = hyBuffers;
    HyCache& cache = hyCache;
    HyLayer& L = hyLayers[l];
    CudaQueue& queue = cudaQueue();
    int used = s.used, latent = s.latent, ew = s.expertWidth;
    int half = b.wideStage / 2, most = b.widePairsMost;
    uint64_t none = 0;
    uint64_t gateAt = 0, upAt = (L.gateBytes + 255) / 256 * 256, downAt = upAt + (L.upBytes + 255) / 256 * 256;
    go("quantizeInput", (unsigned int)(((uint64_t)n * latent / 32 + 7) / 8), 1, 32, 8, b.routedIn, n * latent, b.wideInValues, b.wideInScales, b.wideInSums);
    // cada grupo es un pedazo de las filas de un experto: uno muy usado puede tener mas filas que las que entran en un trabajo
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
    uint64_t region = 3ULL * b.wideRows * used;
    auto started = std::chrono::steady_clock::now();
    for(size_t k = 0; k < jobs.size() && !failed; k++){
        const Job& job = jobs[k];
        int h = (int)(k % 2);
        int groups = (int)job.experts.size();
        if(k >= 2 && queue.eventSync(b.wideDone[h]) != 0) failed = true;
        uint64_t base = cache.base;
        if(job.staged){
            auto from = std::chrono::steady_clock::now();
            uint8_t* host = b.widePinned + (uint64_t)h * half * cache.slotBytes;
            pool.run(groups, [&](int j){
                int e = job.experts[j].expert;
                uint8_t* at = host + (uint64_t)j * cache.slotBytes;
                int warm = cache.warmOf.empty() ? -1 : cache.warmOf[(size_t)l * s.experts + e];
                if(warm >= 0){
                    memcpy(at, cache.warmHost + (uint64_t)warm * cache.slotBytes, (size_t)cache.slotBytes);
                    return;
                }
                memcpy(at + gateAt, L.hostGate.data + (uint64_t)e * L.gateBytes, (size_t)L.gateBytes);
                memcpy(at + upAt, L.hostUp.data + (uint64_t)e * L.upBytes, (size_t)L.upBytes);
                memcpy(at + downAt, L.hostDown.data + (uint64_t)e * L.downBytes, (size_t)L.downBytes);
            });
            b.wideUploadMs += since(from);
            b.wideUploaded += (uint64_t)groups * (L.gateBytes + L.upBytes + L.downBytes);
            base = stage + (uint64_t)h * half * cache.slotBytes;
            if(queue.copyUp(base, host, (size_t)groups * cache.slotBytes, b.wideStream) != 0 || queue.eventRecord(b.wideCopied[h], b.wideStream) != 0) failed = true;
        }
        int* groupSlot = b.hostWidePairs + h * region;
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
            widest = std::max(widest, groupCount[g]);
            for(int i = piece.from; i < piece.from + piece.count; i++){
                int pair = byExpert[e][i];
                inputIndex[c] = pair / used;
                tokenOf[c] = pair / used;
                weightOf[c] = b.hostWeights[pair];
                c++;
            }
        }
        if(job.staged && queue.streamWait(stream, b.wideCopied[h], 0) != 0) failed = true;
        uint64_t slotOn = b.widePairs + 4ULL * h * region;
        uint64_t startOn = slotOn + 4ULL * 64, countOn = slotOn + 4ULL * 128, inputOn = slotOn + 4ULL * 192;
        uint64_t tokenOn = inputOn + 4ULL * most, weightOn = tokenOn + 4ULL * most;
        bool wide = c >= wideLeast * groups;
        int width = tileWidth(wide);
        int tiles = (widest + width - 1) / width;
        unsigned int threads = width == 128 ? 256 : 128;
        int gateTile = rowTile(L.hostGate.type, wide, 0), upTile = rowTile(L.hostUp.type, wide, 0), downTile = rowTile(L.hostDown.type, wide, 0);
        go(tiled(L.hostGate.type, wide, 0).c_str(), (unsigned int)((ew + gateTile - 1) / gateTile), (unsigned int)(groups * tiles), threads, 1, (int)L.hostGate.type, base + gateAt, L.hostGate.stride, cache.slotBytes, slotOn, startOn, countOn, inputOn, b.wideInValues, b.wideInScales, b.wideInSums, none, none, none, b.wideGate, ew, latent, 0, tiles, none, none, none, tables);
        go(tiled(L.hostUp.type, wide, 0).c_str(), (unsigned int)((ew + upTile - 1) / upTile), (unsigned int)(groups * tiles), threads, 1, (int)L.hostUp.type, base + upAt, L.hostUp.stride, cache.slotBytes, slotOn, startOn, countOn, inputOn, b.wideInValues, b.wideInScales, b.wideInSums, none, none, none, b.wideUp, ew, latent, 0, tiles, none, none, none, tables);
        go("hySwigluPack", (unsigned int)(((uint64_t)c * ew / 32 + 7) / 8), 1, 32, 8, b.wideGate, b.wideUp, b.wideHid, c * ew, L.clamp, b.wideHidValues, b.wideHidScales, b.wideHidSums);
        go(tiled(L.hostDown.type, wide, 0).c_str(), (unsigned int)((latent + downTile - 1) / downTile), (unsigned int)(groups * tiles), threads, 1, (int)L.hostDown.type, base + downAt, L.hostDown.stride, cache.slotBytes, slotOn, startOn, countOn, none, b.wideHidValues, b.wideHidScales, b.wideHidSums, none, none, none, none, latent, ew, 0, tiles, tokenOn, weightOn, b.routedOut, tables);
        if(queue.eventRecord(b.wideDone[h], stream) != 0) failed = true;
    }
    if(cudaLink().sync() != 0) failed = true;
    b.wideBatchMs += since(started);
}

// expertos de un bloque grande del prompt: todos en la gpu, los que no estan en la vram suben una vez por bloque
// b.routedIn, b.routedOut, b.hostIds y b.mappedWeights ya apuntan a las filas del bloque
inline void GpuModel::hyExpertsWide(int l, int n, uint64_t stage) {
    HyShape& s = hyShape;
    HyBuffers& b = hyBuffers;
    HyCache& cache = hyCache;
    HyLayer& L = hyLayers[l];
    int used = s.used, latent = s.latent;
    go("hyFill", (unsigned int)(((uint64_t)n * latent + 255) / 256), 1, 256, 1, b.routedOut, n * latent);
    if(cudaLink().sync() != 0){
        failed = true;
        return;
    }
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
    if(b.widePinned && !getenv("SNEO_HY_WIDE_PAIRS")){
        hyExpertsGrouped(l, n, stage, byExpert);
        return;
    }
    int most = chunk * used;
    int* hitSlot = b.hostHits;
    int* hitPair = b.hostHits + most;
    int count = 0;
    auto flush = [&](uint64_t base){
        if(!count || failed) return;
        auto from = std::chrono::steady_clock::now();
        hyBatch(L, base, nullptr, nullptr, b.mappedHits, b.mappedHits + 4ULL * most, count);
        if(cudaLink().sync() != 0) failed = true;
        b.wideBatchMs += since(from);
        count = 0;
    };
    for(int e = 0; e < s.experts; e++){
        int slot = cache.slotOf[(size_t)l * s.experts + e];
        if(slot < 0) continue;
        for(int pair : byExpert[e]){
            if(count == most) flush(cache.base);
            hitSlot[count] = slot;
            hitPair[count] = pair;
            count++;
        }
    }
    flush(cache.base);
    std::vector<int> waiting;
    for(int e = 0; e < s.experts; e++){
        if(!byExpert[e].empty() && cache.slotOf[(size_t)l * s.experts + e] < 0) waiting.push_back(e);
    }
    if(b.widePinned){
        // la cpu copia una mitad a la ram fijada, esa mitad viaja por su propia cola y la gpu la calcula
        // mientras tanto la cpu ya llena la otra mitad
        CudaQueue& queue = cudaQueue();
        uint64_t P = (uint64_t)b.wideRows * used;
        int half = b.wideStage / 2;
        uint64_t gateAt = 0, upAt = (L.gateBytes + 255) / 256 * 256, downAt = upAt + (L.upBytes + 255) / 256 * 256;
        auto started = std::chrono::steady_clock::now();
        for(size_t g = 0, gi = 0; g < waiting.size() && !failed; g += half, gi++){
            int h = (int)(gi % 2);
            int cnt = (int)std::min(waiting.size() - g, (size_t)half);
            if(gi >= 2 && queue.eventSync(b.wideDone[h]) != 0) failed = true;
            auto from = std::chrono::steady_clock::now();
            uint8_t* host = b.widePinned + (uint64_t)h * half * cache.slotBytes;
            pool.run(cnt, [&](int j){
                int e = waiting[g + j];
                uint8_t* at = host + (uint64_t)j * cache.slotBytes;
                memcpy(at + gateAt, L.hostGate.data + (uint64_t)e * L.gateBytes, (size_t)L.gateBytes);
                memcpy(at + upAt, L.hostUp.data + (uint64_t)e * L.upBytes, (size_t)L.upBytes);
                memcpy(at + downAt, L.hostDown.data + (uint64_t)e * L.downBytes, (size_t)L.downBytes);
            });
            b.wideUploadMs += since(from);
            b.wideUploaded += (uint64_t)cnt * (L.gateBytes + L.upBytes + L.downBytes);
            uint64_t dev = stage + (uint64_t)h * half * cache.slotBytes;
            if(queue.copyUp(dev, host, (size_t)cnt * cache.slotBytes, b.wideStream) != 0 || queue.eventRecord(b.wideCopied[h], b.wideStream) != 0) failed = true;
            int* slotList = b.hostWidePairs + (uint64_t)h * 2 * P;
            int* pairList = slotList + P;
            int c = 0;
            for(int j = 0; j < cnt; j++){
                for(int pair : byExpert[waiting[g + j]]){
                    slotList[c] = j;
                    pairList[c] = pair;
                    c++;
                }
            }
            if(queue.streamWait(stream, b.wideCopied[h], 0) != 0) failed = true;
            uint64_t slotsOn = b.widePairs + 4ULL * h * 2 * P, pairsOn = slotsOn + 4ULL * P;
            for(int off = 0; off < c && !failed; off += most) hyBatch(L, dev, nullptr, nullptr, slotsOn + 4ULL * off, pairsOn + 4ULL * off, std::min(most, c - off));
            if(queue.eventRecord(b.wideDone[h], stream) != 0) failed = true;
        }
        if(cudaLink().sync() != 0) failed = true;
        b.wideBatchMs += since(started);
        return;
    }
    for(size_t g = 0; g < waiting.size() && !failed; g += b.wideStage){
        size_t end = std::min(waiting.size(), g + b.wideStage);
        auto from = std::chrono::steady_clock::now();
        for(size_t j = g; j < end && !failed; j++){
            if(!hyCopyExpert(l, waiting[j], stage + (uint64_t)(j - g) * cache.slotBytes)) failed = true;
            b.wideUploaded += L.gateBytes + L.upBytes + L.downBytes;
        }
        b.wideUploadMs += since(from);
        for(size_t j = g; j < end; j++){
            for(int pair : byExpert[waiting[j]]){
                if(count == most) flush(stage);
                hitSlot[count] = (int)(j - g);
                hitPair[count] = pair;
                count++;
            }
        }
        flush(stage);
    }
}

// buffers de paso de una capa para tramos de rows filas: la carga los suma y el prompt largo los saca de lo prestado
inline std::vector<std::pair<uint64_t*, uint64_t>> GpuModel::hySubBuffers(int rows) {
    HyShape& s = hyShape;
    HyBuffers& b = hyBuffers;
    uint64_t W = (uint64_t)rows, d = (uint64_t)s.hidden, H = (uint64_t)s.widestHeads, ew = (uint64_t)s.expertWidth, ff = (uint64_t)s.feedForward;
    uint64_t widest = std::max<uint64_t>({d * s.hc, H * s.headValue, ff, (uint64_t)s.engramHeads * s.engramDim});
    std::vector<std::pair<uint64_t*, uint64_t>> list = {
        {&b.xn, W * d * 4}, {&b.qa, W * s.qLora * 4}, {&b.q, W * H * (s.nope + s.rope) * 4}, {&b.kv, W * 576 * 4},
        {&b.qfull, W * H * 576 * 4}, {&b.latentOut, W * H * 512 * 4}, {&b.heads, W * H * s.headValue * 4}, {&b.gateOut, W * H * s.headValue * 4},
        {&b.attn, W * d * 4}, {&b.ffn, W * d * 4}, {&b.sGate, W * ew * 4}, {&b.sUp, W * ew * 4}, {&b.sHid, W * ew * 4},
        {&b.logits, W * s.experts * 4}, {&b.bigGate, W * ff * 4}, {&b.bigUp, W * ff * 4}, {&b.bigHid, W * ff * 4}, {&b.mixes, W * 16 * 4},
        {&packs[0].values, W * widest}, {&packs[0].scales, W * widest / 8}, {&packs[0].sums, W * widest / 8}};
    if(s.engramModules > 0){
        list.push_back({&b.engramNormed, W * s.engramHeads * s.engramDim * 4});
        list.push_back({&b.engramKv, W * (s.hc + 1) * d * 4});
    }
    if(b.blas){
        list.push_back({&b.q16, W * H * (s.nope + s.rope) * 2});
        list.push_back({&b.lat16, W * H * s.kvLora * 2});
    }
    if(s.indexed){
        list.push_back({&b.idxKey, W * 128 * 4});
        list.push_back({&b.idxQ, W * 2048 * 4});
        list.push_back({&b.idxW, W * 16 * 4});
        list.push_back({&b.idxScores, W * s.idxGroups * 4});
        list.push_back({&b.idxList, W * s.idxWidth * 4});
        list.push_back({&b.idxCount, W * 4});
    }
    return list;
}

// prompt largo capa por capa: cada capa pasa por todas las filas del bloque en tramos de 64 y despues
// los expertos de todo el bloque van juntos en la gpu, asi la cpu no calcula nada
inline bool GpuModel::hyWide(const int* ids, const int* positions, int n, bool last) {
    HyShape& s = hyShape;
    HyBuffers& b = hyBuffers;
    HyCache& cache = hyCache;
    int d = s.hidden;
    if(cudaLink().sync() != 0){
        failed = true;
        return false;
    }
    // el final de la cache de expertos se presta: los que estaban ahi vuelven a subir cuando se usen
    int from = cache.slots - b.wideBorrow;
    for(int j = from; j < cache.slots; j++){
        if(cache.owner[j] >= 0) cache.slotOf[cache.owner[j]] = -1;
        cache.owner[j] = -1;
    }
    cache.free.erase(std::remove_if(cache.free.begin(), cache.free.end(), [&](int j){ return j >= from; }), cache.free.end());
    uint64_t at = cache.base + (uint64_t)from * cache.slotBytes;
    auto carve = [&](uint64_t bytes){ uint64_t p = at; at += (bytes + 255) / 256 * 256; return p; };
    uint64_t stage = carve((uint64_t)b.wideStage * cache.slotBytes);
    uint64_t streams = carve((uint64_t)n * d * s.hc * 4);
    uint64_t shared = carve((uint64_t)n * d * 4);
    uint64_t routedIn = carve((uint64_t)n * s.latent * 4);
    uint64_t routedOut = carve((uint64_t)n * s.latent * 4);
    uint64_t post = carve((uint64_t)n * s.hc * 4);
    uint64_t places = carve((uint64_t)n * 4);
    uint64_t most = (uint64_t)b.widePairsMost, ew = (uint64_t)s.expertWidth;
    b.wideInValues = carve((uint64_t)n * s.latent);
    b.wideInScales = carve((uint64_t)n * s.latent / 8);
    b.wideInSums = carve((uint64_t)n * s.latent / 8);
    b.wideGate = carve(most * ew * 4);
    b.wideUp = carve(most * ew * 4);
    b.wideHid = carve(most * ew * 4);
    b.wideHidValues = carve(most * ew);
    b.wideHidScales = carve(most * ew / 8);
    b.wideHidSums = carve(most * ew / 8);
    HyBuffers kept = b;
    PackedInput keptPack = packs[0];
    // los tramos de atencion van de a wideSub filas con sus propios buffers de paso
    int step = b.wideSub > chunk ? b.wideSub : chunk;
    if(step > chunk){
        for(const auto& item : hySubBuffers(step)){
            *item.first = carve(item.second);
            if(item.first == &packs[0].values) packs[0].capacity = (int)item.second;
        }
        if(cudaLink().memset(b.mixes, 0, (size_t)step * 16 * 4) != 0) failed = true;
    }
    std::vector<int> spots(positions, positions + n);
    if(cudaLink().copyUp(places, spots.data(), (size_t)n * 4) != 0) failed = true;
    if(b.hostEngram){
        auto from = std::chrono::steady_clock::now();
        b.hostEngram = b.hostWideEngram;
        hyEngramHost(positions, n);
        b.hostEngram = kept.hostEngram;
        b.wideEngramMs += since(from);
    }
    uint64_t engramRow = (uint64_t)s.engramModules * s.engramHeads * s.engramDim * 4;
    auto point = [&](int k, int rows){
        b.streams = streams + (uint64_t)k * d * s.hc * 4;
        b.shared = shared + (uint64_t)k * d * 4;
        b.routedIn = routedIn + (uint64_t)k * s.latent * 4;
        b.routedOut = routedOut + (uint64_t)k * s.latent * 4;
        b.post = post + (uint64_t)k * s.hc * 4;
        b.positions = places + (uint64_t)k * 4;
        b.mappedEngram = b.wideEngram + (uint64_t)k * engramRow;
        b.mappedIds = b.wideIds + (uint64_t)k * s.used * 4;
        b.hostIds = b.hostWideIds + (size_t)k * s.used;
        b.mappedWeights = b.wideWeights + (uint64_t)k * s.used * 4;
        b.hostWeights = b.hostWideWeights + (size_t)k * s.used;
        hyTop = 0;
        b.firstPos = positions[k];
        for(int t = k; t < k + rows; t++) hyTop = std::max(hyTop, positions[t]);
    };
    for(int k = 0; k < n && !failed; k += chunk){
        int rows = std::min(chunk, n - k);
        if(cudaLink().sync() != 0) failed = true;
        pool.run(rows, [&](int t){
            int id = ids[k + t];
            if(id < 0 || id >= embed.rows) id = 0;
            dequantRow(embed.type, embed.data + (uint64_t)id * embed.stride, b.hostEmbed + (size_t)t * d, d);
        });
        go("hySpread", (unsigned int)rows, 1, 256, 1, b.mappedEmbed, streams + (uint64_t)k * d * s.hc * 4, d, s.hc);
    }
    for(int l = 0; l < s.layers && !failed && !stopGenerating; l++){
        bool dense = hyLayers[l].dense;
        auto from = std::chrono::steady_clock::now();
        for(int k = 0; k < n && !failed; k += step){
            int rows = std::min(step, n - k);
            point(k, rows);
            hyLayer(l, rows, dense ? 0 : 1);
        }
        if(dense) continue;
        if(cudaLink().sync() != 0) failed = true;
        b.wideLayersMs += since(from);
        point(0, n);
        hyExpertsWide(l, n, stage);
        from = std::chrono::steady_clock::now();
        for(int k = 0; k < n && !failed; k += step){
            int rows = std::min(step, n - k);
            point(k, rows);
            hyLayer(l, rows, 2);
        }
        if(cudaLink().sync() != 0) failed = true;
        b.wideLayersMs += since(from);
    }
    if(last && !failed){
        int k = (n - 1) / chunk * chunk;
        point(k, n - k);
        uint64_t lastRow = b.streams + 4ULL * (n - 1 - k) * d * s.hc;
        uint64_t none = 0;
        hyHcIn(lastRow, b.headFn, b.headScale, b.headBase, b.outNorm, b.last, none, s.hc, 1, false);
        hyProject(head, b.last, b.vocab, 1);
        int keep = shortlist > 0 ? shortlist : 0;
        if(keep > 0) go("shortRows", 1, 1, 512, 1, b.vocab, s.vocab, keep, b.mappedLogits);
        else go("hyCopy", (unsigned int)((s.vocab + 255) / 256), 1, 256, 1, b.vocab, b.mappedLogits, s.vocab);
    }
    if(cudaLink().sync() != 0) failed = true;
    // vuelven todos los buffers de siempre, solo se quedan los contadores del bloque
    uint64_t uploaded = b.wideUploaded;
    double uploadMs = b.wideUploadMs, batchMs = b.wideBatchMs, engramMs = b.wideEngramMs, layersMs = b.wideLayersMs;
    b = kept;
    b.wideUploaded = uploaded;
    b.wideUploadMs = uploadMs;
    b.wideBatchMs = batchMs;
    b.wideEngramMs = engramMs;
    b.wideLayersMs = layersMs;
    packs[0] = keptPack;
    for(int j = cache.slots - 1; j >= from; j--) cache.free.push_back(j);
    // lo prestado se llena ya con los mas usados que no estan: si no, los primeros tokens escritos los subian de a uno y tardaban el doble
    if(!failed && !stopGenerating){
        HyShape& s = hyShape;
        std::vector<int> order;
        for(size_t i = 0; i < cache.heat.size(); i++){
            if(cache.heat[i] && !hyLayers[i / s.experts].dense && cache.slotOf[i] < 0) order.push_back((int)i);
        }
        int want = std::min((int)cache.free.size(), (int)order.size());
        std::partial_sort(order.begin(), order.begin() + want, order.end(), [&](int a, int c){ return cache.heat[a] > cache.heat[c]; });
        for(int i = 0; i < want && !failed && !cache.free.empty(); i++){
            int slot = cache.free.back();
            cache.free.pop_back();
            hyUpload(order[i] / s.experts, order[i] % s.experts, slot);
        }
    }
    return !failed;
}

// fp8 e4m3 sin escala: la norma de cada fila la borra igual
inline float hyFp8(uint8_t v) {
    int sign = v & 0x80 ? -1 : 1;
    int e = (v >> 3) & 15, m = v & 7;
    if(e == 15 && m == 7) return 0.0f;
    return sign * (e == 0 ? (m / 8.0f) * 0.015625f : (1.0f + m / 8.0f) * ldexpf(1.0f, e - 7));
}

// filas de las tablas engram para cada token: hash de los ultimos tokens, como el original
inline void GpuModel::hyEngramHost(const int* positions, int n) {
    HyShape& s = hyShape;
    HyBuffers& b = hyBuffers;
    static float lut[256];
    static bool filled = false;
    if(!filled){
        for(int v = 0; v < 256; v++) lut[v] = hyFp8((uint8_t)v);
        filled = true;
    }
    int cols = s.engramHeads, dim = s.engramDim, gram = s.engramGram;
    int pad = s.engramMap[s.engramPad];
    auto mapped = [&](int id){ return id >= 0 && id < (int)s.engramMap.size() ? s.engramMap[id] : pad; };
    pool.run(n, [&](int t){
        int pos = positions[t];
        uint64_t toks[8];
        for(int k = 0; k < gram; k++){
            int at = pos - k;
            toks[k] = (uint64_t)(at >= 0 && at < (int)tokens.size() ? mapped(tokens[at]) : pad);
        }
        for(int e = 0; e < s.engramModules; e++){
            const uint64_t* mult = s.engramMult.data() + (size_t)e * gram;
            uint64_t rolling = toks[0] * mult[0];
            for(int o = 1; o < gram; o++){
                rolling ^= toks[o] * mult[o];
                for(int h = 0; h < s.engramPerGram; h++){
                    int col = (o - 1) * s.engramPerGram + h;
                    size_t at = (size_t)e * cols + col;
                    uint64_t row = rolling % s.engramPrimes[at] + s.engramOffsets[at];
                    const uint8_t* src = s.engramTable + row * (uint64_t)dim;
                    float* dst = b.hostEngram + (((size_t)t * s.engramModules + e) * cols + col) * dim;
                    for(int k = 0; k < dim; k++) dst[k] = lut[src[k]];
                }
            }
        }
    });
}

// expertos en la cpu: compuerta y subida juntas, swiglu con tope, bajada y suma pesada por token
inline void GpuModel::hyExpertsHost(HyLayer& L, int n, const std::vector<char>& onCpu) {
    HyShape& s = hyShape;
    HyBuffers& b = hyBuffers;
    int used = s.used, width = s.expertWidth, latent = s.latent;
    int layer = (int)(&L - hyLayers.data());
    HyCache& cache = hyCache;
    auto warmCopy = [&](int e) -> const uint8_t* {
        int slot = cache.warmOf.empty() ? -1 : cache.warmOf[(size_t)layer * s.experts + e];
        return slot >= 0 ? cache.warmHost + (uint64_t)slot * cache.slotBytes : nullptr;
    };
    std::vector<std::vector<int>> byExpert(s.experts);
    for(int t = 0; t < n; t++){
        for(int k = 0; k < used; k++){
            int e = b.hostIds[t * used + k];
            if(e >= 0 && e < s.experts && onCpu[t * used + k]) byExpert[e].push_back(t * used + k);
        }
    }
    std::vector<int> order;
    std::vector<int> start(s.experts, 0);
    for(int e = 0; e < s.experts; e++){
        start[e] = (int)order.size();
        for(int slot : byExpert[e]) order.push_back(slot);
    }
    int rows = (int)order.size();
    memset(b.hostMoe, 0, (size_t)n * latent * 4);
    if(!rows) return;
    for(int e = 0; e < s.experts; e++) if(!byExpert[e].empty()) cache.cpuBytes += L.gateBytes + L.upBytes + L.downBytes;
    std::vector<float> input((size_t)rows * latent);
    for(int r = 0; r < rows; r++){
        int t = order[r] / used;
        memcpy(input.data() + (size_t)r * latent, b.hostLatent + (size_t)t * latent, latent * 4);
    }
    HostPacked packedIn, packedMid;
    packedIn.fill(input.data(), rows * latent);
    std::vector<float> gate((size_t)rows * width), up((size_t)rows * width), down((size_t)rows * latent);
    std::vector<HostProduct> jobs;
    for(int e = 0; e < s.experts; e++){
        int count = (int)byExpert[e].size();
        if(!count) continue;
        Matrix g = L.hostGate, u = L.hostUp;
        g.data += (uint64_t)e * L.gateBytes;
        u.data += (uint64_t)e * L.upBytes;
        if(const uint8_t* warm = warmCopy(e)){
            g.data = warm;
            u.data = warm + (L.gateBytes + 255) / 256 * 256;
        }
        jobs.push_back({g, &packedIn, start[e], gate.data() + (size_t)start[e] * width, count});
        jobs.push_back({u, &packedIn, start[e], up.data() + (size_t)start[e] * width, count});
    }
    hostMultiply(pool, jobs, true);
    float limit = L.clamp;
    pool.run((rows + 15) / 16, [&](int part){
        for(int r = part * 16; r < std::min(rows, part * 16 + 16); r++){
            float* g = gate.data() + (size_t)r * width;
            const float* u = up.data() + (size_t)r * width;
            for(int c = 0; c < width; c++){
                float a = g[c], v = u[c];
                if(limit > 0.0f){
                    a = std::min(a, limit);
                    v = std::min(std::max(v, -limit), limit);
                }
                g[c] = a / (1.0f + expf(-a)) * v;
            }
        }
    });
    packedMid.fill(gate.data(), rows * width);
    jobs.clear();
    for(int e = 0; e < s.experts; e++){
        int count = (int)byExpert[e].size();
        if(!count) continue;
        Matrix dn = L.hostDown;
        dn.data += (uint64_t)e * L.downBytes;
        if(const uint8_t* warm = warmCopy(e)) dn.data = warm + (L.gateBytes + 255) / 256 * 256 + (L.upBytes + 255) / 256 * 256;
        jobs.push_back({dn, &packedMid, start[e], down.data() + (size_t)start[e] * latent, count});
    }
    hostMultiply(pool, jobs, true);
    for(int r = 0; r < rows; r++){
        int slot = order[r];
        int t = slot / used;
        float w = b.hostWeights[slot];
        float* dst = b.hostMoe + (size_t)t * latent;
        const float* src = down.data() + (size_t)r * latent;
        for(int c = 0; c < latent; c++) dst[c] += w * src[c];
    }
}

inline void GpuModel::hyAttention(int l, int n) {
    HyShape& s = hyShape;
    HyBuffers& b = hyBuffers;
    HyLayer& L = hyLayers[l];
    int d = s.hidden, hc = s.hc;
    unsigned int spread = (unsigned int)n;
    auto blocks = [](uint64_t count){ return (unsigned int)((count + 255) / 256); };

    if(L.engram >= 0 && b.hostEngram){
        int cols = s.engramHeads, dim = s.engramDim;
        uint64_t rows = b.mappedEngram + 4ULL * (uint64_t)L.engram * cols * dim;
        go("hyNorm", (unsigned int)(n * cols), 1, 128, 1, rows, s.engramModules * cols * dim, b.engramNormed, cols * dim, L.engramNorm, dim, cols, dim, 1.1920929e-07f);
        hyProject(L.engramKv, b.engramNormed, b.engramKv, n);
        go("hyEngramMix", spread, 1, 256, 1, b.streams, b.engramKv, L.engramBias, L.engramGate, L.engramGamma, L.engramScale, d, hc, 1.1920929e-07f);
    }

    // atencion
    hyHcIn(b.streams, L.hcAttnFn, L.hcAttnScale, L.hcAttnBase, L.attnNorm, b.xn, b.post, 2 * hc, n, true);
    std::vector<std::pair<const GpuMatrix*, uint64_t>> first = {{&L.qa, b.qa}, {&L.kva, b.kv}, {&L.gate, b.gateOut}};
    if(L.indexed) first.push_back({&L.idxK, b.idxKey});
    hyProjectSame(first, b.xn, n, true);
    // con mas de 511 grupos antes del propio el indexador elige las posiciones, si no se mira todo como siempre
    bool scored = L.indexed && (hyTop >> 2) > s.idxKeep;
    if(!scored) L.idxSnapRows = 0;
    if(scored) go("hyF32Wide", 16, spread, 128, 1, L.idxProj, b.xn, d, b.idxW, 16, d);
    PackedInput& pk = packs[0];
    if(L.qb.type != 30 && L.qb.type != 1){
        go("hyNormPack", spread, 1, 256, 1, b.qa, L.qaNorm, s.qLora, s.eps, pk.values, pk.scales, pk.sums);
        hyNormRows(b.qa, s.qLora, b.qa, s.qLora, L.qaNorm, 0, n, 1, s.qLora, s.eps);
        project(L.qb, b.qa, 0, b.q, n);
        if(scored) project(L.idxQ, b.qa, 0, b.idxQ, n);
    } else {
        hyNormRows(b.qa, s.qLora, b.qa, s.qLora, L.qaNorm, 0, n, 1, s.qLora, s.eps);
        hyProject(L.qb, b.qa, b.q, n);
        if(scored) hyProject(L.idxQ, b.qa, b.idxQ, n);
    }
    // las claves del indexador y los grupos que se cierran se guardan siempre, aunque esta vez no se elija
    if(L.indexed){
        go("hyIdxKey", spread, 1, 128, 1, b.idxKey, L.idxKNorm, L.idxKeys, b.positions, s.idxRing, s.eps);
        go("hyIdxPool", spread, 1, 128, 1, L.idxKeys, s.idxRing, b.positions, L.idxGate, L.idxValue, L.idxApe, L.idxNorm, L.idxPools, s.eps, s.ropeBase);
    }
    if(scored){
        go("hyRope", spread, 16, 32, 1, b.idxQ, 2048, 128, 64, b.positions, s.ropeBase, 64);
        int groups = hyTop >> 2;
        go("hyIdxScore", (unsigned int)((groups + 127) / 128), spread, 128, 1, b.idxQ, b.idxW, L.idxPools, b.positions, b.idxScores, s.idxGroups, s.idxKeep, s.idxScale);
        go("hySelect", spread, 1, 1024, 1, b.idxScores, s.idxGroups, b.positions, b.idxList, b.idxCount, s.idxWidth, s.idxKeep);
        if(L.idxSnap && n <= L.idxSnapCap){
            CudaQueue& queue = cudaQueue();
            if(queue.copyAcross(L.idxSnap, b.idxList, (size_t)n * s.idxWidth * 4, stream) != 0 || queue.copyAcross(L.idxSnapCount, b.idxCount, (size_t)n * 4, stream) != 0) failed = true;
            L.idxSnapFirst = b.firstPos;
            L.idxSnapRows = n;
        }
    }
    hyNormRows(b.kv, 576, b.kv, 576, L.kvNorm, 0, n, 1, s.kvLora, s.eps);
    int headKey = s.nope + s.rope;
    float base = L.sliding ? s.ropeBaseSliding : s.ropeBase;
    go("hyRope", spread, (unsigned int)L.heads, 32, 1, b.q, L.heads * headKey, headKey, s.nope, b.positions, base, s.rope);
    go("hyRope", spread, 1, 32, 1, b.kv, 576, 576, s.kvLora, b.positions, base, s.rope);
    unsigned int partsQ = n >= 16 ? 1 : 4;
    // con muchas filas cada cabeza es una multiplicacion de matrices: todas juntas en cublas con tensor cores
    typedef int (*Batched)(void*, int, int, int, int, int, const void*, const void*, int, int, long long, const void*, int, int, long long, const void*, void*, int, int, long long, int, int, int);
    // desde 2 filas: los kernels propios releen los pesos de cada cabeza por fila y cublas los lee una vez
    Batched batched = b.blas && n >= 2 ? (Batched)b.blasBatched : nullptr;
    float one = 1.0f, zero = 0.0f;
    if(batched){
        go("hyQHalf", spread * (unsigned int)L.heads, 1, 256, 1, b.q, b.q16, b.qfull, L.heads, s.nope, s.rope, s.kvLora, headKey);
        if(batched(b.blas, 1, 0, s.kvLora, n, s.nope, &one, (const void*)L.kb, 2, s.nope, (long long)s.kvLora * s.nope, (const void*)b.q16, 2, L.heads * headKey, (long long)headKey, &zero, (void*)b.qfull, 0, L.heads * (s.kvLora + s.rope), (long long)(s.kvLora + s.rope), L.heads, 68, -1) != 0) failed = true;
    } else go("hyAbsorbRows", (unsigned int)L.heads, spread * partsQ, 256, 1, b.q, L.kb, b.qfull, L.heads, s.nope, s.rope, s.kvLora, headKey, (int)partsQ);
    go("hyStore", spread, 1, 256, 1, b.kv, L.cache, b.positions, L.sliding ? s.ring : 0, 576);
    int window = L.sliding ? s.window : 0;
    int reach = scored ? s.idxWidth : window > 0 ? std::min(window, hyTop + 1) : hyTop + 1;
    int most = (int)std::max<uint64_t>(1, b.scratchFloats / ((uint64_t)n * L.heads * 514));
    int span = std::max(64, (reach + most - 1) / most);
    int splits = std::max(1, (reach + span - 1) / span);
    if(scored) go("hyAttendList", (unsigned int)(L.heads / 8), spread * (unsigned int)splits, 256, 1, b.qfull, L.cache, b.idxList, b.idxCount, s.idxWidth, b.attnScratch, L.heads, 1.0f / sqrtf((float)headKey), span, splits);
    else go("hyAttendPart", (unsigned int)(L.heads / 8), spread * (unsigned int)splits, 256, 1, b.qfull, L.cache, L.sliding ? s.ring : 0, b.positions, b.attnScratch, L.heads, window, 1.0f / sqrtf((float)headKey), span, splits);
    go("hyAttendMerge", (unsigned int)L.heads, spread, 256, 1, b.attnScratch, L.sinks, b.latentOut, L.heads, splits);
    if(batched){
        go("hyToHalf", blocks((uint64_t)n * L.heads * s.kvLora), 1, 256, 1, b.latentOut, b.lat16, n * L.heads * s.kvLora);
        if(batched(b.blas, 1, 0, s.headValue, n, s.kvLora, &one, (const void*)L.vb, 2, s.kvLora, (long long)s.headValue * s.kvLora, (const void*)b.lat16, 2, L.heads * s.kvLora, (long long)s.kvLora, &zero, (void*)b.heads, 0, L.heads * s.headValue, (long long)s.headValue, L.heads, 68, -1) != 0) failed = true;
    } else go("hyExpandRows", (unsigned int)L.heads, spread * partsQ, 256, 1, b.latentOut, L.vb, b.heads, L.heads, s.kvLora, s.headValue, (int)partsQ);
    int width = L.heads * s.headValue;
    if(L.out.type != 30 && L.out.type != 1){
        go("hyGateMulPack", (unsigned int)(((uint64_t)n * width / 32 + 7) / 8), 1, 32, 8, b.heads, b.gateOut, n * width, pk.values, pk.scales, pk.sums);
        project(L.out, b.heads, 0, b.attn, n);
    } else {
        go("hySigmoidMul", blocks((uint64_t)n * width), 1, 256, 1, b.heads, b.gateOut, n * width);
        hyProject(L.out, b.heads, b.attn, n);
    }
    hyHcOut(b.attn, L.postAttnNorm, n);
}

// part 0 hace la capa entera, 1 hasta antes de los expertos y 2 desde los expertos ya sumados en routedOut
inline void GpuModel::hyLayer(int l, int n, int part) {
    HyShape& s = hyShape;
    HyBuffers& b = hyBuffers;
    HyLayer& L = hyLayers[l];
    int d = s.hidden, hc = s.hc;
    unsigned int spread = (unsigned int)n;
    auto blocks = [](uint64_t count){ return (unsigned int)((count + 255) / 256); };
    PackedInput& pk = packs[0];
    if(part != 2){
        hyAttention(l, n);
        // red de avance
        hyHcIn(b.streams, L.hcFfnFn, L.hcFfnScale, L.hcFfnBase, L.ffnNorm, b.xn, b.post, 2 * hc, n, true);
    }
    if(L.dense){
        int ff = s.feedForward;
        hyProjectSame({{&L.ffnGate, b.bigGate}, {&L.ffnUp, b.bigUp}}, b.xn, n, true);
        go("hySwigluPack", (unsigned int)(((uint64_t)n * ff / 32 + 7) / 8), 1, 32, 8, b.bigGate, b.bigUp, b.bigHid, n * ff, 0.0f, pk.values, pk.scales, pk.sums);
        if(L.ffnDown.type != 30 && L.ffnDown.type != 1) project(L.ffnDown, b.bigHid, 0, b.ffn, n);
        else hyProject(L.ffnDown, b.bigHid, b.ffn, n);
    } else {
        if(part != 2){
            go("hyF32Wide", (unsigned int)s.experts, spread, 128, 1, L.router, b.xn, d, b.logits, s.experts, d);
            go("hyRouteFast", spread, 1, 512, 1, b.logits, L.routerBias, b.mappedIds, b.mappedWeights, s.experts, s.used, s.expertScale, s.expertNorm ? 1 : 0);
            hyProjectSame({{&L.routedDown, b.routedIn}, {&L.sharedGate, b.sGate}, {&L.sharedUp, b.sUp}}, b.xn, n, true);
            CudaQueue& queue = cudaQueue();
            if(part == 0){
                go("hyCopy", blocks((uint64_t)n * s.latent), 1, 256, 1, b.routedIn, b.mappedLatent, n * s.latent);
                if(!hyRouted && queue.eventCreate(&hyRouted, 2) != 0) failed = true;
                if(queue.eventRecord(hyRouted, stream) != 0) failed = true;
            }
            int ew = s.expertWidth;
            go("hySwigluPack", (unsigned int)(((uint64_t)n * ew / 32 + 7) / 8), 1, 32, 8, b.sGate, b.sUp, b.sHid, n * ew, 0.0f, pk.values, pk.scales, pk.sums);
            if(L.sharedDown.type != 30 && L.sharedDown.type != 1) project(L.sharedDown, b.sHid, 0, b.shared, n);
            else hyProject(L.sharedDown, b.sHid, b.shared, n);
            if(part == 1) return;
            // la cpu calcula los expertos que no estan en la vram mientras la gpu hace los suyos
            auto waited = std::chrono::steady_clock::now();
            if(queue.eventSync(hyRouted) != 0) failed = true;
            hyCache.waitMs += since(waited);
            if(!failed) hyExperts(l, n);
        }
        hyProject(L.routedUp, b.routedOut, b.ffn, n);
        go("hyAdd", blocks((uint64_t)n * d), 1, 256, 1, b.ffn, b.shared, n * d);
    }
    hyHcOut(b.ffn, L.postFfnNorm, n);
}

inline bool GpuModel::hyChunk(const int* ids, const int* positions, int n, bool last) {
    HyShape& s = hyShape;
    HyBuffers& b = hyBuffers;
    CudaQueue& queue = cudaQueue();
    int d = s.hidden;
    if(cudaLink().sync() != 0) failed = true;
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
    if(b.hostEngram) hyEngramHost(positions, n);
    if(queue.copyUp(b.positions, b.hostPositions, (size_t)n * 4, stream) != 0) failed = true;
    go("hySpread", (unsigned int)n, 1, 256, 1, b.mappedEmbed, b.streams, d, s.hc);
    for(int l = 0; l < s.layers && !failed && !stopGenerating; l++) hyLayer(l, n);
    if(!last || failed) return !failed;
    if(everyRow){
        uint64_t none = 0;
        int keep = shortlist > 0 ? shortlist : 0;
        hyHcIn(b.streams, b.headFn, b.headScale, b.headBase, b.outNorm, b.normedRows, none, s.hc, n, false);
        hyProject(head, b.normedRows, b.every, n);
        if(keep > 0) go("shortRows", (unsigned int)n, 1, 512, 1, b.every, s.vocab, keep, b.mappedEvery);
        else go("hyCopy", (unsigned int)(((uint64_t)n * s.vocab + 255) / 256), 1, 256, 1, b.every, b.mappedEvery, n * s.vocab);
        return !failed;
    }
    uint64_t lastRow = b.streams + 4ULL * (n - 1) * d * s.hc;
    uint64_t none = 0;
    hyHcIn(lastRow, b.headFn, b.headScale, b.headBase, b.outNorm, b.last, none, s.hc, 1, false);
    int keep = shortlist > 0 ? shortlist : 0;
    hyProject(head, b.last, b.vocab, 1);
    if(keep > 0) go("shortRows", 1, 1, 512, 1, b.vocab, s.vocab, keep, b.mappedLogits);
    else go("hyCopy", (unsigned int)((s.vocab + 255) / 256), 1, 256, 1, b.vocab, b.mappedLogits, s.vocab);
    return !failed;
}

inline std::vector<float> GpuModel::forwardHy4(const std::vector<int>& ids, const std::vector<int>& positions) {
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
    if(everyRow && (int)ids.size() > std::min(b.everyCap, chunk)){
        problem = "at most " + std::to_string(std::min(b.everyCap, chunk)) + " rows can be checked per call";
        return logits;
    }
    failed = false;
    auto clock = std::chrono::steady_clock::now();
    int start = positions[0];
    // al volver atras la ventana deslizante tiene filas pisadas: se recalculan las ultimas del tramo que queda
    if(start > 0 && start <= (int)tokens.size() && start - s.window + 1 + s.ring <= b.ringTop){
        int from = std::max(0, start - s.window + 1);
        std::vector<int> again(tokens.begin() + from, tokens.begin() + start);
        std::vector<int> places(again.size());
        for(size_t i = 0; i < again.size(); i++) places[i] = from + (int)i;
        tokens.resize(start);
        for(size_t at = 0; at < again.size() && !failed && !stopGenerating; at += chunk){
            int n = (int)std::min(again.size() - at, (size_t)chunk);
            hyChunk(again.data() + at, places.data() + at, n, false);
        }
        b.ringTop = start - 1;
    }
    tokens.resize(std::min(tokens.size(), (size_t)start));
    for(size_t at = 0; at < ids.size() && !failed && !stopGenerating;){
        size_t left = ids.size() - at;
        bool wide = !everyRow && b.wideBorrow > 0 && (int)left >= b.wideMin;
        int n = (int)std::min(left, (size_t)(wide ? b.wideRows : chunk));
        tokens.insert(tokens.end(), ids.begin() + at, ids.begin() + at + n);
        if(wide) hyWide(ids.data() + at, positions.data() + at, n, at + n == ids.size());
        else hyChunk(ids.data() + at, positions.data() + at, n, at + n == ids.size());
        b.ringTop = std::max(b.ringTop, positions[at + n - 1]);
        at += n;
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
    else {
        logits.assign(b.hostLogits, b.hostLogits + width);
        hyDraft.from = b.last;
        hyDraft.at = positions.back();
    }
    headMs = since(clock);
    static int calls = 0;
    if(hyCache.total - hyCache.heatSaved > 200000) hySaveHeat();
    if(ids.size() > 1 || ++calls % 64 == 0) std::cout << "hy_v4_flash: experts found in video memory " << (hyCache.total ? 100 * hyCache.hits / hyCache.total : 0) << "% of " << hyCache.total << ", drafter proposed " << hyDraft.proposed << " (cut " << hyDraft.cut << " unsure, sparse " << hyDraft.sparseSteps << ") in " << (int)hyDraft.ms << " ms, steps " << hyDraft.steps << " first right " << hyDraft.firstKept << " kept " << hyDraft.kept << ", on the second card " << hyCache.helpUsed << " (" << (int)hyCache.helpMs << " ms waiting)" << ", from pinned ram " << hyCache.warmUsed << ", skipped " << hyCache.skipped << ", uploads " << hyCache.uploads << ", free slots " << hyCache.free.size() << ", ms in this call: total " << since(clock) << " cpu experts " << hyCache.cpuMs << " (" << (int)(hyCache.cpuBytes / 1048576.0 / std::max(0.001, hyCache.cpuMs)) << " GB/s)" << " waiting for the gpu " << hyCache.waitMs << " uploads " << hyCache.uploadMs << std::endl;
    bool wideCall = b.wideUploaded > 0;
    if(wideCall) std::cout << "hy_v4_flash: prompt on the gpu, experts uploaded " << (b.wideUploaded >> 20) << " MB, cpu copying " << (int)b.wideUploadMs << " ms (" << (int)(b.wideUploaded / 1048576.0 / std::max(1.0, b.wideUploadMs)) << " GB/s), experts " << (int)b.wideBatchMs << " ms, attention and the rest of the layers " << (int)b.wideLayersMs << " ms, engram rows " << (int)b.wideEngramMs << " ms, whole call " << (int)since(clock) << " ms" << std::endl;
    b.wideUploaded = 0;
    b.wideUploadMs = b.wideBatchMs = b.wideLayersMs = b.wideEngramMs = 0;
    if(ids.size() > 1 || calls % 64 == 0){
        hyCache.cpuMs = hyCache.waitMs = hyCache.uploadMs = 0;
        hyCache.cpuBytes = 0;
    }
    for(size_t i = 0; i < markUsed; i++){
        float ms = 0;
        cudaQueue().eventElapsed(&ms, marks[i].second.first, marks[i].second.second);
        kernelMs[marks[i].first] += ms;
        kernelCalls[marks[i].first]++;
    }
    markUsed = 0;
    static int profiled = 0;
    if(profile && (++profiled % 64 == 0 || wideCall)){
        std::vector<std::pair<double, std::string>> ranked;
        for(auto& k : kernelMs) ranked.push_back({k.second, k.first});
        std::sort(ranked.rbegin(), ranked.rend());
        std::cout << "hy_v4_flash kernels over 64 tokens:";
        for(size_t i = 0; i < ranked.size() && i < 14; i++) std::cout << " | " << ranked[i].second << " " << (int)ranked[i].first << " ms";
        std::cout << std::endl;
        kernelMs.clear();
        kernelCalls.clear();
    }
    return logits;
}
