#pragma once
#include "qwen4.h"
#include <thread>
#include <atomic>
#include <cmath>
#include <cstring>

inline std::string draftSourcePath(const std::string& folder) {
    return folder + "\\Qwen3.8-Flash-Next.draft.gguf";
}

inline std::string draftRuntimePath(const std::string& folder, int bits = 2) {
    return folder + (bits == 4 ? "\\Qwen3.8-Flash-Next.draft-q4.gguf" : "\\Qwen3.8-Flash-Next.draft-q2.gguf");
}

inline void packNonLinear(const float* x, uint8_t* out, int count) {
    for(int b = 0; b < count / 32; b++){
        const float* v = x + 32 * b;
        uint8_t* block = out + 18 * b;
        memset(block, 0, 18);
        float peak = 0;
        for(int i = 0; i < 32; i++) peak = std::max(peak, fabsf(v[i]));
        if(peak == 0) continue;

        float bestScale = 0;
        double bestError = 1e300;
        uint8_t best[32] = {};
        uint8_t codes[32];
        for(int sign = -1; sign <= 1; sign += 2){
            for(int step = 0; step < 8; step++){
                float scale = sign * peak / (127.0f - 4.0f * step);
                for(int round = 0; round < 2; round++){
                    double xq = 0, qq = 0;
                    for(int i = 0; i < 32; i++){
                        float target = v[i] / scale;
                        int pick = 0;
                        float gap = 1e30f;
                        for(int c = 0; c < 16; c++){
                            float miss = fabsf(target - (float)nonLinear4[c]);
                            if(miss >= gap) continue;
                            gap = miss;
                            pick = c;
                        }
                        codes[i] = (uint8_t)pick;
                        xq += (double)v[i] * nonLinear4[pick];
                        qq += (double)nonLinear4[pick] * nonLinear4[pick];
                    }
                    if(qq == 0) break;
                    scale = (float)(xq / qq);
                }
                double error = 0;
                for(int i = 0; i < 32; i++){
                    double miss = v[i] - nonLinear4[codes[i]] * (double)scale;
                    error += miss * miss;
                }
                if(error >= bestError) continue;
                bestError = error;
                bestScale = scale;
                memcpy(best, codes, 32);
            }
        }

        uint16_t half16 = (uint16_t)_mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(bestScale), 0), 0);
        memcpy(block, &half16, 2);
        for(int i = 0; i < 16; i++) block[2 + i] = (uint8_t)(best[i] | (best[i + 16] << 4));
    }
}

inline void packTwoBit(const float* x, uint8_t* out, int count) {
    for(int b = 0; b < count / 64; b++){
        const float* v = x + 64 * b;
        uint8_t* block = out + 18 * b;
        memset(block, 0, 18);
        float peak = 0;
        for(int i = 0; i < 64; i++) peak = std::max(peak, fabsf(v[i]));
        if(peak == 0) continue;

        float bestScale = 0;
        double bestError = 1e300;
        uint8_t best[64] = {};
        uint8_t codes[64];
        for(int sign = -1; sign <= 1; sign += 2){
            for(int step = 0; step < 6; step++){
                float scale = sign * peak / (1.0f + 0.2f * step);
                for(int round = 0; round < 2; round++){
                    double xq = 0, qq = 0;
                    for(int i = 0; i < 64; i++){
                        int c = (int)lrintf(v[i] / scale) + 1;
                        c = c < 0
                            ? 0
                            : c > 3
                                ? 3
                                : c;
                        codes[i] = (uint8_t)c;
                        xq += (double)v[i] * (c - 1);
                        qq += (double)(c - 1) * (c - 1);
                    }
                    if(qq == 0) break;
                    scale = (float)(xq / qq);
                }
                double error = 0;
                for(int i = 0; i < 64; i++){
                    double miss = v[i] - (codes[i] - 1) * scale;
                    error += miss * miss;
                }
                if(error >= bestError) continue;
                bestError = error;
                bestScale = scale;
                memcpy(best, codes, 64);
            }
        }

        uint16_t half16 = (uint16_t)_mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(bestScale), 0), 0);
        memcpy(block, &half16, 2);
        for(int i = 0; i < 64; i++) block[2 + i / 4] |= (uint8_t)(best[i] << (2 * (i % 4)));
    }
}

inline void widenHalves(const uint8_t* src, float* out, size_t count) {
    for(size_t i = 0; i < count; i++){
        uint32_t bits = (uint32_t)src[2 * i] << 16 | (uint32_t)src[2 * i + 1] << 24;
        memcpy(out + i, &bits, 4);
    }
}

struct DraftTensor {
    std::string name;
    uint32_t type = 0;
    std::vector<uint64_t> dims;
    std::vector<uint8_t> data;
    uint64_t offset = 0;
};

inline bool buildDraft(const std::string& source, const std::string& target, int threads, std::string& error, int bits = 2) {
    ModelFile file;
    if(!openModel(source, file, error)) return false;

    HANDLE input = CreateFileA(source.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, nullptr);
    if(input == INVALID_HANDLE_VALUE){
        error = "cannot open " + source;
        return false;
    }

    auto readRange = [&](uint64_t from, uint64_t length, std::vector<uint8_t>& out) {
        uint64_t start = from / 4096 * 4096;
        uint64_t span = (from + length - start + 4095) / 4096 * 4096;
        uint8_t* staging = (uint8_t*)VirtualAlloc(nullptr, span, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        if(!staging) return false;
        bool ok = true;
        for(uint64_t done = 0; done < span && ok;){
            DWORD piece = (DWORD)std::min<uint64_t>(span - done, 64ULL << 20);
            OVERLAPPED spot = {};
            spot.Offset = (DWORD)(start + done);
            spot.OffsetHigh = (DWORD)((start + done) >> 32);
            DWORD got = 0;
            ok = ReadFile(input, staging + done, piece, &got, &spot) && got > 0;
            done += got;
            if(got < piece) break;
        }
        if(ok){
            out.resize(length);
            memcpy(out.data(), staging + (from - start), length);
        }
        VirtualFree(staging, 0, MEM_RELEASE);
        return ok;
    };

    auto sourceOf = [&](const std::string& name) -> const TensorInfo* {
        const TensorInfo* t = findTensor(file, name);
        if(!t && error.empty()) error = "the draft file has no " + name;
        return t;
    };

    std::vector<DraftTensor> out;
    auto plain = [&](const std::string& from, const std::string& to) {
        const TensorInfo* t = sourceOf(from);
        if(!t) return;
        DraftTensor d;
        d.name = to;
        d.type = 30;
        for(uint32_t i = 0; i < t->dimCount; i++) d.dims.push_back(t->dims[i]);
        if(!readRange(t->offset, t->bytes, d.data) && error.empty()) error = "cannot read " + from;
        out.push_back(std::move(d));
    };
    auto norm = [&](const std::string& from, const std::string& to) {
        const TensorInfo* t = sourceOf(from);
        if(!t) return;
        std::vector<uint8_t> raw;
        if(!readRange(t->offset, t->bytes, raw)){
            if(error.empty()) error = "cannot read " + from;
            return;
        }
        DraftTensor d;
        d.name = to;
        d.type = 0;
        d.dims.push_back(t->elements);
        d.data.resize(t->elements * 4);
        float* values = (float*)d.data.data();
        widenHalves(raw.data(), values, t->elements);
        for(uint64_t i = 0; i < t->elements; i++) values[i] += 1.0f;
        out.push_back(std::move(d));
    };
    auto rowsOf = [&](const std::string& from, const std::string& to, uint64_t first, uint64_t count) {
        const TensorInfo* t = sourceOf(from);
        if(!t) return;
        DraftTensor d;
        d.name = to;
        d.type = 30;
        d.dims = {t->dims[0], count};
        if(!readRange(t->offset + first * t->dims[0] * 2, count * t->dims[0] * 2, d.data) && error.empty()) error = "cannot read " + from;
        out.push_back(std::move(d));
    };

    norm("draft.embed_norm", "draft.embed_norm");
    norm("draft.hidden_norm", "draft.hidden_norm");
    plain("draft.embed_proj", "draft.embed_proj");
    plain("draft.hidden_proj", "draft.hidden_proj");
    norm("draft.out_hc_norm", "draft.out_hc_norm");
    plain("draft.out_hc_down", "draft.out_hc_down");
    plain("draft.out_hc_up", "draft.out_hc_up");
    const char* halves[2] = {"hc_attn_", "hc_ffn_"};
    for(const char* h : halves){
        norm(std::string("draft.") + h + "norm", std::string("blk.0.") + h + "norm.weight");
        plain(std::string("draft.") + h + "down", std::string("blk.0.") + h + "down.weight");
        plain(std::string("draft.") + h + "up", std::string("blk.0.") + h + "up.weight");
        plain(std::string("draft.") + h + "inject", std::string("blk.0.") + h + "inject.weight");
    }
    plain("draft.attn_q", "blk.0.attn_q.weight");
    plain("draft.attn_k", "blk.0.attn_k.weight");
    plain("draft.attn_v", "blk.0.attn_v.weight");
    plain("draft.attn_output", "blk.0.attn_output.weight");
    norm("draft.attn_q_norm", "blk.0.attn_q_norm.weight");
    norm("draft.attn_k_norm", "blk.0.attn_k_norm.weight");
    rowsOf("draft.indexer.qk_proj", "blk.0.indexer.q_proj.weight", 0, 512);
    rowsOf("draft.indexer.qk_proj", "blk.0.indexer.k_proj.weight", 512, 128);
    norm("draft.indexer.q_norm", "blk.0.indexer.q_norm.weight");
    norm("draft.indexer.k_norm", "blk.0.indexer.k_norm.weight");
    plain("draft.ffn_gate_inp", "blk.0.ffn_gate_inp.weight");
    plain("draft.ffn_gate_inp_shexp", "blk.0.ffn_gate_inp_shexp.weight");
    plain("draft.ffn_gate_shexp", "blk.0.ffn_gate_shexp.weight");
    plain("draft.ffn_up_shexp", "blk.0.ffn_up_shexp.weight");
    plain("draft.ffn_down_shexp", "blk.0.ffn_down_shexp.weight");
    if(!error.empty()){
        CloseHandle(input);
        return false;
    }

    const TensorInfo* fused = sourceOf("draft.ffn_gate_up_exps");
    const TensorInfo* down = sourceOf("draft.ffn_down_exps");
    if(!fused || !down || fused->dims[0] != 2560 || fused->dims[1] != 1280 || down->dims[0] != 640 || down->dims[1] != 2560){
        if(error.empty()) error = "unexpected expert shapes in the draft file";
        CloseHandle(input);
        return false;
    }
    int experts = (int)fused->dims[2];
    uint32_t kind = bits == 4
        ? 20
        : 42;
    auto pack = [&](const float* x, uint8_t* to, int count){
        if(kind == 20) packNonLinear(x, to, count);
        else packTwoBit(x, to, count);
    };
    uint64_t gateBytes = rowBytes(kind, 2560) * 640;
    uint64_t downBytes = rowBytes(kind, 640) * 2560;
    DraftTensor gate, up, low;
    gate.name = "blk.0.ffn_gate_exps.weight";
    up.name = "blk.0.ffn_up_exps.weight";
    low.name = "blk.0.ffn_down_exps.weight";
    gate.type = up.type = low.type = kind;
    gate.dims = up.dims = {2560, 640, (uint64_t)experts};
    low.dims = {640, 2560, (uint64_t)experts};
    gate.data.resize(gateBytes * experts);
    up.data.resize(gateBytes * experts);
    low.data.resize(downBytes * experts);

    std::atomic<int> next{0};
    std::atomic<bool> failed{false};
    auto work = [&]{
        std::vector<uint8_t> raw;
        std::vector<float> wide(1280 * 2560);
        while(!failed){
            int e = next++;
            if(e >= experts) break;
            if(!readRange(fused->offset + (uint64_t)e * 1280 * 2560 * 2, 1280ULL * 2560 * 2, raw)){
                failed = true;
                break;
            }
            widenHalves(raw.data(), wide.data(), 1280ULL * 2560);
            for(int r = 0; r < 640; r++){
                pack(wide.data() + (size_t)r * 2560, gate.data.data() + gateBytes * e + rowBytes(kind, 2560) * r, 2560);
                pack(wide.data() + (size_t)(640 + r) * 2560, up.data.data() + gateBytes * e + rowBytes(kind, 2560) * r, 2560);
            }
            if(!readRange(down->offset + (uint64_t)e * 2560 * 640 * 2, 2560ULL * 640 * 2, raw)){
                failed = true;
                break;
            }
            widenHalves(raw.data(), wide.data(), 2560ULL * 640);
            for(int r = 0; r < 2560; r++) pack(wide.data() + (size_t)r * 640, low.data.data() + downBytes * e + rowBytes(kind, 640) * r, 640);
        }
    };
    std::vector<std::thread> crew;
    for(int i = 0; i < std::max(1, threads); i++) crew.emplace_back(work);
    for(auto& t : crew) t.join();
    CloseHandle(input);
    if(failed){
        error = "cannot read the draft experts";
        return false;
    }
    out.push_back(std::move(gate));
    out.push_back(std::move(up));
    out.push_back(std::move(low));

    const uint64_t align = 4096;
    std::string revision;
    auto found = file.meta.find("draft.revision");
    if(found != file.meta.end()) revision = found->second.text;

    std::string header;
    auto putText = [&](const std::string& text) {
        uint64_t length = text.size();
        header.append((const char*)&length, 8);
        header.append(text);
    };
    auto putWord = [&](uint32_t value) {
        header.append((const char*)&value, 4);
    };
    auto putLong = [&](uint64_t value) {
        header.append((const char*)&value, 8);
    };
    header.append("GGUF", 4);
    putWord(3);
    putLong(out.size());
    putLong(3);
    putText("general.alignment");
    putWord(4);
    putWord((uint32_t)align);
    putText("general.architecture");
    putWord(8);
    putText("qwen4exp-draft");
    putText("draft.revision");
    putWord(8);
    putText(revision);
    uint64_t cursor = 0;
    for(DraftTensor& d : out){
        d.offset = cursor;
        cursor += (d.data.size() + align - 1) / align * align;
        putText(d.name);
        putWord((uint32_t)d.dims.size());
        for(uint64_t v : d.dims) putLong(v);
        putWord(d.type);
        putLong(d.offset);
    }
    uint64_t dataStart = (header.size() + align - 1) / align * align;
    uint64_t total = dataStart + cursor;

    uint8_t* image = (uint8_t*)VirtualAlloc(nullptr, total, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if(!image){
        error = "not enough memory to assemble the draft file";
        return false;
    }
    memcpy(image, header.data(), header.size());
    for(DraftTensor& d : out) memcpy(image + dataStart + d.offset, d.data.data(), d.data.size());

    std::string partial = target + ".part";
    HANDLE output = CreateFileA(partial.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH, nullptr);
    bool written = output != INVALID_HANDLE_VALUE;
    for(uint64_t done = 0; written && done < total;){
        DWORD piece = (DWORD)std::min<uint64_t>(total - done, 64ULL << 20);
        DWORD wrote = 0;
        written = WriteFile(output, image + done, piece, &wrote, nullptr) && wrote == piece;
        done += piece;
    }
    if(output != INVALID_HANDLE_VALUE) CloseHandle(output);
    VirtualFree(image, 0, MEM_RELEASE);
    if(!written){
        DeleteFileA(partial.c_str());
        error = "cannot write " + target;
        return false;
    }
    DeleteFileA(target.c_str());
    if(!MoveFileA(partial.c_str(), target.c_str())){
        error = "cannot rename the draft file";
        return false;
    }
    return true;
}

struct DraftModel {
    ModelFile file;
    MappedFile map;
    Layer layer;
    const float* embedNorm = nullptr;
    const float* hiddenNorm = nullptr;
    const float* outNorm = nullptr;
    Matrix embedProj, hiddenProj, outDown, outUp;
    bool streamNorm = false;
    std::string problem;

    ~DraftModel() {
        if(map.base) UnmapViewOfFile(map.base);
        if(map.mapping) CloseHandle(map.mapping);
        if(map.file != INVALID_HANDLE_VALUE) CloseHandle(map.file);
    }

    bool load(const std::string& path) {
        if(!openModel(path, file, problem)) return false;
        map.file = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_RANDOM_ACCESS, nullptr);
        if(map.file != INVALID_HANDLE_VALUE) map.mapping = CreateFileMappingA(map.file, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if(map.mapping) map.base = (const uint8_t*)MapViewOfFile(map.mapping, FILE_MAP_READ, 0, 0, 0);
        if(!map.base){
            problem = "cannot map " + path;
            return false;
        }

        auto matrix = [&](const std::string& name, Matrix& out) {
            const TensorInfo* t = findTensor(file, name);
            if(!t || !knownType(t->type)){
                if(problem.empty()) problem = t
                    ? name + " is " + typeInfo(t->type).name + ", a tensor type the engine cannot read yet"
                    : "missing tensor " + name;
                return;
            }
            out.data = map.base + t->offset;
            out.type = t->type;
            out.cols = (int)t->dims[0];
            out.rows = (int)(t->elements / t->dims[0]);
            out.stride = rowBytes(t->type, t->dims[0]);
        };
        auto floats = [&](const std::string& name) -> const float* {
            Matrix m;
            matrix(name, m);
            return m.data && m.type == 0 ? (const float*)m.data : nullptr;
        };

        embedNorm = floats("draft.embed_norm");
        hiddenNorm = floats("draft.hidden_norm");
        outNorm = floats("draft.out_hc_norm");
        matrix("draft.embed_proj", embedProj);
        matrix("draft.hidden_proj", hiddenProj);
        matrix("draft.out_hc_down", outDown);
        matrix("draft.out_hc_up", outUp);

        layer.attention = true;
        const char* halves[2] = {"hc_attn_", "hc_ffn_"};
        for(int h = 0; h < 2; h++){
            layer.hcNorm[h] = floats(std::string("blk.0.") + halves[h] + "norm.weight");
            matrix(std::string("blk.0.") + halves[h] + "down.weight", layer.hcDown[h]);
            matrix(std::string("blk.0.") + halves[h] + "up.weight", layer.hcUp[h]);
            matrix(std::string("blk.0.") + halves[h] + "inject.weight", layer.hcInject[h]);
        }
        matrix("blk.0.attn_q.weight", layer.query);
        matrix("blk.0.attn_k.weight", layer.key);
        matrix("blk.0.attn_v.weight", layer.value);
        matrix("blk.0.attn_output.weight", layer.attnOut);
        matrix("blk.0.indexer.q_proj.weight", layer.indexQuery);
        matrix("blk.0.indexer.k_proj.weight", layer.indexKey);
        layer.queryNorm = floats("blk.0.attn_q_norm.weight");
        layer.keyNorm = floats("blk.0.attn_k_norm.weight");
        layer.indexQueryNorm = floats("blk.0.indexer.q_norm.weight");
        layer.indexKeyNorm = floats("blk.0.indexer.k_norm.weight");
        matrix("blk.0.ffn_gate_inp.weight", layer.router);
        matrix("blk.0.ffn_gate_inp_shexp.weight", layer.sharedScalar);
        matrix("blk.0.ffn_gate_shexp.weight", layer.sharedGate);
        matrix("blk.0.ffn_up_shexp.weight", layer.sharedUp);
        matrix("blk.0.ffn_down_shexp.weight", layer.sharedDown);
        matrix("blk.0.ffn_gate_exps.weight", layer.expertGate);
        matrix("blk.0.ffn_up_exps.weight", layer.expertUp);
        matrix("blk.0.ffn_down_exps.weight", layer.expertDown);
        layer.expertGate.rows /= expertCount;
        layer.expertUp.rows /= expertCount;
        layer.expertDown.rows /= expertCount;

        if(!problem.empty()) return false;
        if(!embedNorm || !hiddenNorm || !outNorm || !layer.hcNorm[0] || !layer.hcNorm[1] || !layer.queryNorm || !layer.keyNorm){
            problem = "the draft file is missing its norms";
            return false;
        }
        return true;
    }

    void reset() {
        layer.keys.clear();
        layer.values.clear();
        layer.indexRaw.clear();
        layer.pooled.clear();
        layer.lastRouting.clear();
    }

    void run(Model& main, const float* rows, const int* next, const int* positions, int n, std::vector<float>& residual, std::vector<float>& logits, int firstLogit) {
        std::vector<float> e((size_t)n * hidden), en((size_t)n * hidden), ep((size_t)n * hidden);
        for(int t = 0; t < n; t++){
            int id = next[t] < 0 || next[t] >= main.embed.rows ? 0 : next[t];
            dequantRow(main.embed.type, main.embed.data + (uint64_t)id * main.embed.stride, e.data() + (size_t)t * hidden, hidden);
            rmsNorm(e.data() + (size_t)t * hidden, embedNorm, en.data() + (size_t)t * hidden, hidden);
        }

        std::vector<float> hn((size_t)n * hcWidth), hp((size_t)n * hcWidth);
        for(int t = 0; t < n; t++){
            const float* r = rows + (size_t)t * hcWidth;
            float* out = hn.data() + (size_t)t * hcWidth;
            if(!streamNorm){
                rmsNorm(r, hiddenNorm, out, hcWidth);
                continue;
            }
            for(int c = 0; c < hcCount; c++) rmsNorm(r + (size_t)c * hidden, hiddenNorm + (size_t)c * hidden, out + (size_t)c * hidden, hidden);
        }
        multiply(main.pool, {{embedProj, en.data(), ep.data(), n}, {hiddenProj, hn.data(), hp.data(), n * hcCount}});

        residual.assign((size_t)n * hcWidth, 0.0f);
        for(int t = 0; t < n; t++){
            for(int c = 0; c < hcCount; c++){
                float* dst = residual.data() + (size_t)t * hcWidth + (size_t)c * hidden;
                const float* h = hp.data() + (size_t)t * hcWidth + (size_t)c * hidden;
                const float* x = ep.data() + (size_t)t * hidden;
                for(int d = 0; d < hidden; d++) dst[d] = h[d] + x[d];
            }
        }

        std::vector<float> x((size_t)n * hidden), y((size_t)n * hidden), inj((size_t)n * hcCount);
        main.mix(layer.hcNorm[0], layer.hcDown[0], layer.hcUp[0], &layer.hcInject[0], residual.data(), n, x.data(), inj.data());
        main.attend(layer, x.data(), n, positions, y.data());
        main.write(residual.data(), y.data(), inj.data(), n);
        main.mix(layer.hcNorm[1], layer.hcDown[1], layer.hcUp[1], &layer.hcInject[1], residual.data(), n, x.data(), inj.data());
        main.experts(layer, x.data(), n, y.data());
        main.write(residual.data(), y.data(), inj.data(), n);

        int first = std::max(0, std::min(firstLogit, n - 1));
        int count = n - first;
        std::vector<float> fin((size_t)count * hidden);
        main.mix(outNorm, outDown, outUp, nullptr, residual.data() + (size_t)first * hcWidth, count, fin.data(), nullptr);
        logits.assign((size_t)count * main.head.rows, 0.0f);
        multiply(main.pool, {{main.head, fin.data(), logits.data(), count}});
    }
};
