#pragma once

// dfly2: el ayudante de hy4 que adivina varios tokens en una pasada leyendo la cache de las capas dispersas del modelo
inline bool GpuModel::hyLoadDraft(const std::string& modelPath) {
    HyShape& s = hyShape;
    HyDraft& g = hyDraft;
    std::error_code e;
    std::string found;
    if(const char* asked = getenv("SNEO_HY_DRAFT_FILE")) found = asked;
    for(const auto& folder : {std::filesystem::path(modelPath).parent_path() / "dfly2", std::filesystem::path(modelPath).parent_path()}){
        if(!found.empty() || !std::filesystem::exists(folder, e)) continue;
        for(const auto& item : std::filesystem::directory_iterator(folder, e)){
            if(item.path().extension() != ".safetensors") continue;
            SafeFile probe;
            std::string why;
            if(probe.open(item.path().string(), why) && probe.find("model.dfly_layers.0.input_layernorm.weight")) found = item.path().string();
            probe.close();
            if(!found.empty()) break;
        }
    }
    if(found.empty()){
        g.problem = "no dfly2 weights beside the model";
        return false;
    }
    SafeFile file;
    if(!file.open(found, g.problem)) return false;
    std::string p = "model.dfly_layers.";
    std::vector<uint8_t> scratch;
    auto grab = [&](uint64_t bytes) -> uint64_t {
        uint64_t ptr = 0;
        if(cudaLink().alloc(&ptr, (size_t)std::max<uint64_t>(bytes, 256)) != 0){
            if(g.problem.empty()) g.problem = "out of video memory for the dfly2 drafter";
            return 0;
        }
        allocations.push_back(ptr);
        g.bytes += bytes;
        return ptr;
    };
    auto matrix = [&](const std::string& name, GpuMatrix& out, int rows, int cols) {
        const SafeTensor* t = file.find(name);
        if(!t || t->dtype != "BF16" || t->shape.size() != 2 || (int)t->shape[0] != rows || (int)t->shape[1] != cols){
            if(g.problem.empty()) g.problem = "dfly2 tensor " + name + " is missing or has an unexpected shape";
            return;
        }
        // en 8 bits ocupa la mitad y deja lugar para mas expertos en la vram
        if(!getenv("SNEO_HY_DRAFT_BF16") && cols % 32 == 0){
            std::vector<uint8_t> packed = hyRequant(t->data, 30, cols, rows, 8);
            out = {grab(packed.size()), 8, rows, cols, rowBytes(8, cols)};
            if(out.data && cudaLink().copyUp(out.data, packed.data(), packed.size()) != 0 && g.problem.empty()) g.problem = "cannot copy " + name;
            return;
        }
        out = {grab(t->end - t->begin), 30, rows, cols, (uint64_t)cols * 2};
        if(out.data && cudaLink().copyUp(out.data, t->data, (size_t)(t->end - t->begin)) != 0 && g.problem.empty()) g.problem = "cannot copy " + name;
    };
    auto floats = [&](const std::string& name, uint64_t& out, int count) {
        const SafeTensor* t = file.find(name);
        if(!t || t->dtype != "BF16" || (int)((t->end - t->begin) / 2) != count){
            if(g.problem.empty()) g.problem = "dfly2 tensor " + name + " is missing or has an unexpected size";
            return;
        }
        std::vector<float> values(count);
        const uint16_t* raw = (const uint16_t*)t->data;
        for(int i = 0; i < count; i++){
            uint32_t bits = (uint32_t)raw[i] << 16;
            memcpy(&values[i], &bits, 4);
        }
        out = grab((uint64_t)count * 4);
        if(out && cudaLink().copyUp(out, values.data(), (size_t)count * 4) != 0 && g.problem.empty()) g.problem = "cannot copy " + name;
    };
    int d = s.hidden;
    int heads = 32, nope = s.nope, rope = s.rope, kv = s.kvLora, hv = s.headValue;
    // cada capa del ayudante lee la cache de una capa dispersa del modelo, como dice su config
    std::vector<int64_t> targets = {25, 29, 33, 37, 41};
    g.layers.assign(5, HyDraftLayer());
    for(int i = 0; i < 5 && g.problem.empty(); i++){
        HyDraftLayer& L = g.layers[i];
        std::string q = p + std::to_string(i) + ".";
        L.target = (int)targets[i];
        if(L.target >= s.layers || hyLayers[L.target].sliding || hyLayers[L.target].heads != heads){
            g.problem = "dfly2 layer " + std::to_string(i) + " reads target layer " + std::to_string(L.target) + ", which is not a 32 head sparse layer";
            return false;
        }
        floats(q + "input_layernorm.weight", L.inNorm, d);
        floats(q + "post_attention_layernorm.weight", L.postAttn, d);
        matrix(q + "self_attn.q_a_proj.weight", L.qa, s.qLora, d);
        floats(q + "self_attn.q_a_layernorm.weight", L.qaNorm, s.qLora);
        matrix(q + "self_attn.q_b_proj.weight", L.qb, heads * (nope + rope), s.qLora);
        matrix(q + "self_attn.kv_a_proj_with_mqa.weight", L.kva, kv + rope, d);
        floats(q + "self_attn.kv_a_layernorm.weight", L.kvNorm, kv);
        floats(q + "self_attn.learnable_sink_param", L.sinks, heads);
        matrix(q + "self_attn.linear_gate.weight", L.gate, heads * hv, d);
        matrix(q + "self_attn.o_proj.weight", L.out, d, heads * hv);
        floats(q + "self_attn.post_proj_norm.weight", L.postProj, d);
        matrix(q + "mlp.gate_proj.weight", L.mlpGate, 9216, d);
        matrix(q + "mlp.up_proj.weight", L.mlpUp, 9216, d);
        matrix(q + "mlp.down_proj.weight", L.mlpDown, d, 9216);
        // kv_b lleva por cabeza 192 filas de k sin rope y 256 de v: van a f16 con el orden que leen los kernels de absorber y expandir
        const SafeTensor* kvb = file.find(q + "self_attn.kv_b_proj.weight");
        if(!kvb || kvb->dtype != "BF16" || kvb->shape.size() != 2 || (int)kvb->shape[0] != heads * (nope + hv) || (int)kvb->shape[1] != kv){
            if(g.problem.empty()) g.problem = "dfly2 kv_b_proj of layer " + std::to_string(i) + " has an unexpected shape";
            return false;
        }
        const uint16_t* raw = (const uint16_t*)kvb->data;
        auto value = [&](int row, int col){
            uint32_t bits = (uint32_t)raw[(size_t)row * kv + col] << 16;
            float f;
            memcpy(&f, &bits, 4);
            return (uint16_t)_mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(f), 0), 0);
        };
        std::vector<uint16_t> kb((size_t)heads * kv * nope), vb((size_t)heads * hv * kv);
        for(int h = 0; h < heads; h++){
            for(int r = 0; r < kv; r++) for(int k = 0; k < nope; k++) kb[((size_t)h * kv + r) * nope + k] = value(h * (nope + hv) + k, r);
            for(int j = 0; j < hv; j++) for(int r = 0; r < kv; r++) vb[((size_t)h * hv + j) * kv + r] = value(h * (nope + hv) + nope + j, r);
        }
        L.kb = grab(kb.size() * 2);
        L.vb = grab(vb.size() * 2);
        if(!L.kb || !L.vb || cudaLink().copyUp(L.kb, kb.data(), kb.size() * 2) != 0 || cudaLink().copyUp(L.vb, vb.data(), vb.size() * 2) != 0){
            if(g.problem.empty()) g.problem = "cannot copy the dfly2 k_b and v_b of layer " + std::to_string(i);
            return false;
        }
    }
    matrix(p + "mtp_fusion.eh_proj.weight", g.eh, d, 2 * d);
    floats(p + "mtp_fusion.enorm.weight", g.enorm, d);
    floats(p + "mtp_fusion.hnorm.weight", g.hnorm, d);
    floats(p + "mtp_fusion.gate", g.fuseGate, d);
    floats(p + "final_norm.weight", g.finalNorm, d);
    matrix(p + "hidden_correction.gate_proj.weight", g.hcGate, d, 2 * d);
    matrix(p + "hidden_correction.up_proj.weight", g.hcUp, d, 2 * d);
    matrix(p + "hidden_correction.down_proj.weight", g.hcDown, d, d);
    floats(p + "hidden_correction.hidden_norm.weight", g.hcHidden, d);
    floats(p + "hidden_correction.embed_norm.weight", g.hcEmbed, d);
    file.close();
    if(!g.problem.empty()) return false;

    int rows = 6;
    g.x = grab((uint64_t)rows * d * 4);
    g.hidden = grab((uint64_t)rows * d * 4);
    g.cat = grab((uint64_t)2 * d * 4);
    g.fused = grab((uint64_t)d * 4);
    g.corr = grab((uint64_t)d * 4);
    g.mid = grab((uint64_t)3 * d * 4);
    void* host = nullptr;
    if(cudaLink().hostAlloc(&host, (size_t)(rows + 1) * d * 4 + 256, 3) != 0 || cudaQueue().hostDevice(&g.mappedRows, host, 0) != 0){
        g.problem = "cannot pin host memory for the dfly2 rows";
        return false;
    }
    pinned.push_back(host);
    g.hostRows = (float*)host;
    g.hostPick = (int*)(g.hostRows + (size_t)(rows + 1) * d);
    g.mappedPick = g.mappedRows + 4ULL * (rows + 1) * d;
    if(!g.problem.empty()) return false;
    int id = g.mask;
    if(id < 0 || id >= embed.rows){
        g.problem = "the dfly2 mask token is outside the vocabulary";
        return false;
    }
    g.maskRow.resize(d);
    dequantRow(embed.type, embed.data + (uint64_t)id * embed.stride, g.maskRow.data(), d);
    const char* length = getenv("SNEO_HY_DRAFTS");
    g.length = std::max(1, std::min(5, length ? atoi(length) : 5));
    const char* sure = getenv("SNEO_HY_DRAFT_SURE");
    g.sure = sure ? (float)atof(sure) : 0.6f;
    g.ready = true;
    return true;
}

inline std::vector<int> GpuModel::hyPropose(int token, int most) {
    HyShape& s = hyShape;
    HyBuffers& b = hyBuffers;
    HyDraft& g = hyDraft;
    std::vector<int> drafts;
    if(!g.ready || !g.from || most <= 0) return drafts;
    int anchor = g.at + 1;
    if(anchor != (int)tokens.size() || token < 0 || token >= embed.rows) return drafts;
    int count = std::min(most, g.length);
    int rows = count + 1;
    if(anchor + rows >= contextLimit) return drafts;
    int d = s.hidden;
    auto clock = std::chrono::steady_clock::now();
    auto blocks = [](uint64_t n){ return (unsigned int)((n + 255) / 256); };
    uint64_t none = 0;
    failed = false;

    // fila 0 con el token que acaba de salir y las demas con la mascara
    dequantRow(embed.type, embed.data + (uint64_t)token * embed.stride, g.hostRows, d);
    for(int r = 1; r < rows; r++) memcpy(g.hostRows + (size_t)r * d, g.maskRow.data(), (size_t)d * 4);
    for(int r = 0; r < rows; r++) b.hostPositions[r] = anchor + r;
    if(cudaQueue().copyUp(b.positions, b.hostPositions, (size_t)rows * 4, stream) != 0) failed = true;
    go("hyCopy", blocks((uint64_t)rows * d), 1, 256, 1, g.mappedRows, g.x, rows * d);

    // la fila 0 mezcla su embedding con el estado final del modelo
    hyNormRows(g.x, d, g.cat, 2 * d, g.enorm, 0, 1, 1, d, 1e-5f);
    hyNormRows(g.from, d, g.cat + 4ULL * d, 2 * d, g.hnorm, 0, 1, 1, d, 1e-5f);
    hyProject(g.eh, g.cat, g.fused, 1);
    go("hyFuseAdd", blocks(d), 1, 256, 1, g.x, g.fused, g.fuseGate, d);

    int heads = 32, headKey = s.nope + s.rope;
    unsigned int spread = (unsigned int)rows;
    for(HyDraftLayer& L : g.layers){
        HyLayer& T = hyLayers[L.target];
        hyNormRows(g.x, d, b.xn, d, L.inNorm, 0, rows, 1, d, 1e-5f);
        hyProject(L.qa, b.xn, b.qa, rows);
        hyNormRows(b.qa, s.qLora, b.qa, s.qLora, L.qaNorm, 0, rows, 1, s.qLora, 1e-5f);
        hyProject(L.qb, b.qa, b.q, rows);
        hyProject(L.kva, b.xn, b.kv, rows);
        hyNormRows(b.kv, 576, b.kv, 576, L.kvNorm, 0, rows, 1, s.kvLora, 1e-5f);
        go("hyRope", spread, (unsigned int)heads, 32, 1, b.q, heads * headKey, headKey, s.nope, b.positions, s.ropeBase, s.rope);
        go("hyRope", spread, 1, 32, 1, b.kv, 576, 576, s.kvLora, b.positions, s.ropeBase, s.rope);
        // con varias filas cada cabeza va en cublas: los kernels propios releen los pesos por fila
        typedef int (*Batched)(void*, int, int, int, int, int, const void*, const void*, int, int, long long, const void*, int, int, long long, const void*, void*, int, int, long long, int, int, int);
        Batched batched = b.blas && rows >= 2 ? (Batched)b.blasBatched : nullptr;
        float one = 1.0f, zero = 0.0f;
        if(batched){
            go("hyQHalf", spread * (unsigned int)heads, 1, 256, 1, b.q, b.q16, b.qfull, heads, s.nope, s.rope, s.kvLora, headKey);
            if(batched(b.blas, 1, 0, s.kvLora, rows, s.nope, &one, (const void*)L.kb, 2, s.nope, (long long)s.kvLora * s.nope, (const void*)b.q16, 2, heads * headKey, (long long)headKey, &zero, (void*)b.qfull, 0, heads * (s.kvLora + s.rope), (long long)(s.kvLora + s.rope), heads, 68, -1) != 0) failed = true;
        } else go("hyAbsorbRows", (unsigned int)heads, spread * 4, 256, 1, b.q, L.kb, b.qfull, heads, s.nope, s.rope, s.kvLora, headKey, 4);
        // las filas del bloque se escriben en la cache de la capa del modelo y la proxima verificacion las pisa
        go("hyStore", spread, 1, 256, 1, b.kv, T.cache, b.positions, 0, 576);
        // con contexto largo mira lo mismo que eligio el modelo para el ultimo token aceptado mas sus propias filas
        int row = anchor - 1 - T.idxSnapFirst;
        bool sparse = g.list && T.idxSnapRows > 0 && row >= 0 && row < T.idxSnapRows;
        int reach = sparse ? g.listWidth : anchor + rows;
        int most = (int)std::max<uint64_t>(1, b.scratchFloats / ((uint64_t)rows * heads * 514));
        int span = std::max(64, (reach + most - 1) / most);
        int splits = std::max(1, (reach + span - 1) / span);
        if(sparse){
            go("hyDraftList", spread, 1, 256, 1, T.idxSnap, T.idxSnapCount, row, anchor, g.list, g.counts, g.listWidth);
            go("hyAttendList", (unsigned int)(heads / 8), spread * (unsigned int)splits, 256, 1, b.qfull, T.cache, g.list, g.counts, g.listWidth, b.attnScratch, heads, 1.0f / sqrtf((float)headKey), span, splits);
            g.sparseSteps++;
        } else go("hyAttendPart", (unsigned int)(heads / 8), spread * (unsigned int)splits, 256, 1, b.qfull, T.cache, 0, b.positions, b.attnScratch, heads, 0, 1.0f / sqrtf((float)headKey), span, splits);
        go("hyAttendMerge", (unsigned int)heads, spread, 256, 1, b.attnScratch, L.sinks, b.latentOut, heads, splits);
        if(batched){
            go("hyToHalf", blocks((uint64_t)rows * heads * s.kvLora), 1, 256, 1, b.latentOut, b.lat16, rows * heads * s.kvLora);
            if(batched(b.blas, 1, 0, s.headValue, rows, s.kvLora, &one, (const void*)L.vb, 2, s.kvLora, (long long)s.headValue * s.kvLora, (const void*)b.lat16, 2, heads * s.kvLora, (long long)s.kvLora, &zero, (void*)b.heads, 0, heads * s.headValue, (long long)s.headValue, heads, 68, -1) != 0) failed = true;
        } else go("hyExpandRows", (unsigned int)heads, spread * 4, 256, 1, b.latentOut, L.vb, b.heads, heads, s.kvLora, s.headValue, 4);
        int width = heads * s.headValue;
        hyProject(L.gate, b.xn, b.gateOut, rows);
        go("hySigmoidMul", blocks((uint64_t)rows * width), 1, 256, 1, b.heads, b.gateOut, rows * width);
        hyProject(L.out, b.heads, b.attn, rows);
        hyNormRows(b.attn, d, b.attn, d, L.postProj, 0, rows, 1, d, 1e-5f);
        go("hyAdd", blocks((uint64_t)rows * d), 1, 256, 1, g.x, b.attn, rows * d);
        hyNormRows(g.x, d, b.xn, d, L.postAttn, 0, rows, 1, d, 1e-5f);
        hyProject(L.mlpGate, b.xn, b.bigGate, rows);
        hyProject(L.mlpUp, b.xn, b.bigUp, rows);
        go("hySwiglu", blocks((uint64_t)rows * 9216), 1, 256, 1, b.bigGate, b.bigUp, b.bigHid, rows * 9216, 0.0f);
        hyProject(L.mlpDown, b.bigHid, b.ffn, rows);
        go("hyAdd", blocks((uint64_t)rows * d), 1, 256, 1, g.x, b.ffn, rows * d);
    }
    hyNormRows(g.x, d, g.hidden, d, g.finalNorm, 0, rows, 1, d, 1e-5f);

    // cada token sale de su fila corregida con el token anterior y la cabeza del modelo
    int previous = token;
    float* prevRow = g.hostRows + (size_t)rows * d;
    for(int k = 0; k < count && !failed; k++){
        dequantRow(embed.type, embed.data + (uint64_t)previous * embed.stride, prevRow, d);
        uint64_t h = g.hidden + 4ULL * (k + 1) * d;
        hyNormRows(h, d, g.cat, 2 * d, g.hcHidden, 0, 1, 1, d, 1e-5f);
        hyNormRows(g.mappedRows + 4ULL * rows * d, d, g.cat + 4ULL * d, 2 * d, g.hcEmbed, 0, 1, 1, d, 1e-5f);
        hyProject(g.hcGate, g.cat, g.mid, 1);
        hyProject(g.hcUp, g.cat, g.mid + 4ULL * d, 1);
        go("hySwiglu", blocks(d), 1, 256, 1, g.mid, g.mid + 4ULL * d, g.mid + 8ULL * d, d, 0.0f);
        hyProject(g.hcDown, g.mid + 8ULL * d, g.corr, 1);
        go("hyAdd", blocks(d), 1, 256, 1, g.corr, h, d);
        hyProject(head, g.corr, b.vocab, 1);
        go("hyArgmaxProb", 1, 1, 1024, 1, b.vocab, s.vocab, g.mappedPick, g.mappedPick + 4);
        if(waitStream() != 0) failed = true;
        previous = g.hostPick[0];
        if(previous < 0 || previous >= s.vocab) break;
        // cada fila de mas en la verificacion le cuesta a la cpu sus expertos: sin seguridad no se propone
        float p = ((const float*)g.hostPick)[1];
        if(p < g.sure){
            g.cut++;
            break;
        }
        drafts.push_back(previous);
    }
    if(failed){
        failed = false;
        drafts.clear();
    }
    g.proposed += drafts.size();
    g.ms += since(clock);
    return drafts;
}
