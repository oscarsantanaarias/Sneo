#pragma once
#include "hardware.h"
#include "gguf.h"
#include "quant.h"
#include <cmath>
#include <chrono>
#include <numeric>

inline int hidden = 2560;
inline int hcCount = 4;
inline int hcWidth = 10240;
inline int hcRank = 320;
const int ssmSize = 128;
inline int keyHeads = 16;
inline int valueHeads = 48;
inline int convWidth = 10240;
inline int innerWidth = 6144;
inline int queryHeads = 24;
inline int kvHeads = 2;
const int headWidth = 256;
inline int indexHeads = 4;
const int indexWidth = 128;
const int expertCount = 512;
inline int expertsUsed = 10;
inline int expertWidth = 640;
const int pleHeads = 16;
const int pleRowWidth = 160;
const int pleTaps = 9;
const int endToken = 248044;
const int chunkTokens = 32;

inline bool readShape(const ModelFile& file, std::string& problem) {
    std::string a = file.arch;
    if(a != "qwen4exp" && a != "qwen35moe" && a != "qwen35"){
        problem = "the engine does not run " + a + " models yet";
        return false;
    }
    bool single = a != "qwen4exp";
    bool dense = a == "qwen35";
    hidden = (int)metaWhole(file, a + ".embedding_length", 2560);
    queryHeads = (int)metaWhole(file, a + ".attention.head_count", 24);
    kvHeads = (int)metaWhole(file, a + ".attention.head_count_kv", 2);
    expertsUsed = dense ? 1 : (int)metaWhole(file, a + ".expert_used_count", 10);
    expertWidth = dense
        ? (int)metaWhole(file, a + ".feed_forward_length", 0)
        : (int)metaWhole(file, a + ".expert_feed_forward_length", 640);
    keyHeads = (int)metaWhole(file, a + ".ssm.group_count", 16);
    valueHeads = (int)metaWhole(file, a + ".ssm.time_step_rank", 48);
    innerWidth = (int)metaWhole(file, a + ".ssm.inner_size", 6144);
    convWidth = 2 * keyHeads * ssmSize + valueHeads * ssmSize;
    hcCount = single ? 1 : (int)metaWhole(file, a + ".hyper_connection.count", 4);
    hcRank = single ? 0 : (int)metaWhole(file, a + ".hyper_connection.low_rank", 320);
    hcWidth = hidden * hcCount;
    indexHeads = single ? 0 : (int)metaWhole(file, a + ".attention.indexer.head_count", 4);
    int keyLength = (int)metaWhole(file, a + ".attention.key_length", 256);
    int stateSize = (int)metaWhole(file, a + ".ssm.state_size", 128);
    bool fits = keyLength == headWidth
        && stateSize == ssmSize
        && expertsUsed >= 1 && expertsUsed <= 32
        && hidden % 256 == 0
        && kvHeads > 0 && queryHeads % kvHeads == 0
        && innerWidth == valueHeads * ssmSize
        && innerWidth == queryHeads * headWidth
        && expertWidth > 0 && expertWidth % 64 == 0;
    if(!fits){
        problem = "unexpected sizes for " + a;
        return false;
    }
    return true;
}

inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

inline void rmsNorm(const float* in, const float* weight, float* out, int n) {
    double sum = 0;
    for(int i = 0; i < n; i++) sum += (double)in[i] * in[i];
    float inv = (float)(1.0 / sqrt(sum / n + 1e-6));
    for(int i = 0; i < n; i++) out[i] = in[i] * inv * weight[i];
}

struct MappedFile {
    HANDLE file = INVALID_HANDLE_VALUE;
    HANDLE mapping = nullptr;
    const uint8_t* base = nullptr;
};

struct Layer {
    bool attention = false;
    const float* hcNorm[2] = {};
    Matrix hcDown[2], hcUp[2], hcInject[2];

    Matrix qkv, gate, alpha, beta, ssmOut;
    const float* convWeight = nullptr;
    const float* decay = nullptr;
    const float* dtBias = nullptr;
    const float* ssmNorm = nullptr;

    Matrix query, key, value, attnOut, indexQuery, indexKey;
    const float* queryNorm = nullptr;
    const float* keyNorm = nullptr;
    const float* indexQueryNorm = nullptr;
    const float* indexKeyNorm = nullptr;

    Matrix router, sharedScalar, sharedGate, sharedUp, sharedDown;
    Matrix expertGate, expertUp, expertDown;

    std::vector<float> state, convHistory;
    std::vector<float> keys, values, indexRaw, pooled;
    std::vector<int> lastRouting;
    std::vector<uint64_t> routeCount;
};

struct Model {
    ModelFile file;
    std::vector<MappedFile> maps;
    std::vector<Layer> layers;
    ThreadPool pool;
    Matrix embed, head, outDown, outUp, pleTable, pleKey, pleValue;
    const float* outNorm = nullptr;
    const float* pleNormKey = nullptr;
    const float* pleNormQuery = nullptr;
    const float* pleNormConv = nullptr;
    std::vector<float> pleConv, pleHistory;
    uint64_t pleMultipliers[3] = {};
    uint64_t pleVocab[pleHeads] = {};
    uint64_t pleOffset[pleHeads] = {};
    double ropeFreq[32] = {};
    std::vector<int> tokens;
    std::vector<double> layerMs;
    double embedMs = 0;
    double headMs = 0;
    uint64_t memoryCap = 0;
    bool lockDense = false;
    bool usePle = true;
    bool keepRows = false;
    std::vector<float> rows;
    int budget = 2051;
    int expertCount = 512;
    std::string problem;

    ~Model() {
        for(MappedFile& m : maps){
            if(m.base) UnmapViewOfFile(m.base);
            if(m.mapping) CloseHandle(m.mapping);
            if(m.file != INVALID_HANDLE_VALUE) CloseHandle(m.file);
        }
    }

    bool load(const std::string& path, int threads) {
        keepFullSpeed();
        std::string error;
        if(!openModel(path, file, error)){
            problem = error;
            return false;
        }
        if(file.arch != "qwen4exp"){
            problem = "the cpu path only runs qwen4exp models, " + file.arch + " needs a cuda gpu";
            return false;
        }
        if(!readShape(file, problem)) return false;
        expertCount = file.experts > 0 ? (int)file.experts : 512;

        for(const std::string& p : file.paths){
            MappedFile m;
            m.file = CreateFileA(p.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_RANDOM_ACCESS, nullptr);
            if(m.file != INVALID_HANDLE_VALUE) m.mapping = CreateFileMappingA(m.file, nullptr, PAGE_READONLY, 0, 0, nullptr);
            if(m.mapping) m.base = (const uint8_t*)MapViewOfFile(m.mapping, FILE_MAP_READ, 0, 0, 0);
            maps.push_back(m);
            if(!m.base){
                problem = "cannot map " + p;
                return false;
            }
        }

        auto matrix = [&](const std::string& name, Matrix& out) {
            const TensorInfo* t = findTensor(file, name);
            if(!t || !knownType(t->type)){
                if(problem.empty()) problem = t
                    ? name + " is " + typeInfo(t->type).name + ", a tensor type the engine cannot read yet"
                    : "missing tensor " + name;
                return;
            }
            out.data = maps[t->file].base + t->offset;
            out.type = t->type;
            out.cols = (int)t->dims[0];
            out.rows = (int)(t->elements / t->dims[0]);
            out.stride = rowBytes(t->type, t->dims[0]);
        };

        auto floats = [&](const std::string& name) -> const float* {
            Matrix m;
            matrix(name, m);
            if(m.data && m.type != 0 && problem.empty()) problem = name + " is not F32";
            return m.data && m.type == 0 ? (const float*)m.data : nullptr;
        };

        matrix("token_embd.weight", embed);
        matrix("output.weight", head);
        matrix("output_hc_down.weight", outDown);
        matrix("output_hc_up.weight", outUp);
        outNorm = floats("output_hc_norm.weight");
        matrix("per_layer_token_embd.weight", pleTable);
        matrix("blk.1.ple_key.weight", pleKey);
        matrix("blk.1.ple_value.weight", pleValue);
        pleNormKey = floats("blk.1.ple_norm_key.weight");
        pleNormQuery = floats("blk.1.ple_norm_query.weight");
        pleNormConv = floats("blk.1.ple_norm_conv.weight");

        Matrix conv;
        matrix("blk.1.ple_conv1d.weight", conv);
        if(conv.data){
            pleConv.resize((size_t)conv.cols * conv.rows);
            dequantRow(conv.type, conv.data, pleConv.data(), (int64_t)pleConv.size());
        }

        uint32_t interval = metaWhole(file, "qwen4exp.full_attention_interval", 4);
        layers.resize(file.layers);
        for(int l = 0; l < (int)layers.size(); l++){
            Layer& L = layers[l];
            std::string p = "blk." + std::to_string(l) + ".";
            L.attention = (l + 1) % interval == 0;

            const char* halves[2] = {"hc_attn_", "hc_ffn_"};
            for(int h = 0; h < 2; h++){
                L.hcNorm[h] = floats(p + halves[h] + "norm.weight");
                matrix(p + halves[h] + "down.weight", L.hcDown[h]);
                matrix(p + halves[h] + "up.weight", L.hcUp[h]);
                matrix(p + halves[h] + "inject.weight", L.hcInject[h]);
            }

            if(L.attention){
                matrix(p + "attn_q.weight", L.query);
                matrix(p + "attn_k.weight", L.key);
                matrix(p + "attn_v.weight", L.value);
                matrix(p + "attn_output.weight", L.attnOut);
                matrix(p + "indexer.q_proj.weight", L.indexQuery);
                matrix(p + "indexer.k_proj.weight", L.indexKey);
                L.queryNorm = floats(p + "attn_q_norm.weight");
                L.keyNorm = floats(p + "attn_k_norm.weight");
                L.indexQueryNorm = floats(p + "indexer.q_norm.weight");
                L.indexKeyNorm = floats(p + "indexer.k_norm.weight");
            } else {
                matrix(p + "attn_qkv.weight", L.qkv);
                matrix(p + "attn_gate.weight", L.gate);
                matrix(p + "ssm_alpha.weight", L.alpha);
                matrix(p + "ssm_beta.weight", L.beta);
                matrix(p + "ssm_out.weight", L.ssmOut);
                L.convWeight = floats(p + "ssm_conv1d.weight");
                L.decay = floats(p + "ssm_a");
                L.dtBias = floats(p + "ssm_dt.bias");
                L.ssmNorm = floats(p + "ssm_norm.weight");
            }

            matrix(p + "ffn_gate_inp.weight", L.router);
            matrix(p + "ffn_gate_inp_shexp.weight", L.sharedScalar);
            matrix(p + "ffn_gate_shexp.weight", L.sharedGate);
            matrix(p + "ffn_up_shexp.weight", L.sharedUp);
            matrix(p + "ffn_down_shexp.weight", L.sharedDown);
            matrix(p + "ffn_gate_exps.weight", L.expertGate);
            matrix(p + "ffn_up_exps.weight", L.expertUp);
            matrix(p + "ffn_down_exps.weight", L.expertDown);
            L.expertGate.rows /= expertCount;
            L.expertUp.rows /= expertCount;
            L.expertDown.rows /= expertCount;
        }

        if(!problem.empty()) return false;

        if(embed.cols != hidden || head.cols != hidden || layers.size() != 48 || pleTable.cols != pleRowWidth || pleConv.size() != (size_t)4 * hcWidth){
            problem = "unexpected tensor shapes for qwen4exp";
            return false;
        }

        auto multipliers = file.meta.find("qwen4exp.ple.layer_multipliers");
        auto vocab = file.meta.find("qwen4exp.ple.head_vocab_sizes");
        auto offsets = file.meta.find("qwen4exp.ple.head_offsets");
        if(multipliers == file.meta.end() || vocab == file.meta.end() || offsets == file.meta.end() || multipliers->second.wholes.size() != 3 || vocab->second.wholes.size() != pleHeads || offsets->second.wholes.size() != pleHeads){
            problem = "ple metadata missing";
            return false;
        }
        for(int i = 0; i < 3; i++) pleMultipliers[i] = (uint64_t)multipliers->second.wholes[i];
        for(int i = 0; i < pleHeads; i++){
            pleVocab[i] = (uint64_t)vocab->second.wholes[i];
            pleOffset[i] = (uint64_t)offsets->second.wholes[i];
        }

        budget = (int)metaWhole(file, "qwen4exp.attention.indexer.top_k", 2048) + 3;

        auto base = file.meta.find("qwen4exp.rope.freq_base");
        double theta = base == file.meta.end() ? 1e7 : base->second.real;
        for(int k = 0; k < 32; k++) ropeFreq[k] = pow(theta, -2.0 * k / 64.0);

        MEMORYSTATUSEX memory = {};
        memory.dwLength = sizeof(memory);
        GlobalMemoryStatusEx(&memory);
        if(memoryCap == 0) memoryCap = memory.ullTotalPhys / 2;

        uint64_t locked = 0;
        if(lockDense){
            for(const TensorInfo& t : file.tensors){
                if(t.expert || t.name == "token_embd.weight" || t.name == "per_layer_token_embd.weight") continue;
                locked += t.bytes;
            }
            memoryCap = std::max<uint64_t>(memoryCap, locked + (2ULL << 30));
        }
        SetProcessWorkingSetSizeEx(GetCurrentProcess(), locked + (256ULL << 20), (SIZE_T)memoryCap, QUOTA_LIMITS_HARDWS_MAX_ENABLE | QUOTA_LIMITS_HARDWS_MIN_DISABLE);

        for(const TensorInfo& t : file.tensors){
            if(!lockDense || t.expert || t.name == "token_embd.weight" || t.name == "per_layer_token_embd.weight") continue;
            if(!VirtualLock((LPVOID)(maps[t.file].base + t.offset), (SIZE_T)t.bytes)){
                problem = "could not keep " + t.name + " resident, continuing without the lock";
                lockDense = false;
            }
        }

        pool.start(std::max(0, threads - 1));
        reset();
        return true;
    }

    void reset() {
        tokens.clear();
        rows.clear();
        pleHistory.assign((size_t)pleTaps * hcWidth, 0.0f);
        for(Layer& L : layers){
            L.state.assign(L.attention ? 0 : (size_t)valueHeads * ssmSize * ssmSize, 0.0f);
            L.convHistory.assign(L.attention ? 0 : (size_t)3 * convWidth, 0.0f);
            L.keys.clear();
            L.values.clear();
            L.indexRaw.clear();
            L.pooled.clear();
            L.lastRouting.clear();
        }
    }

    void rope(float* v, int position) {
        for(int k = 0; k < 32; k++){
            double angle = position * ropeFreq[k];
            double c = cos(angle), s = sin(angle);
            double a = v[k], b = v[k + 32];
            v[k] = (float)(a * c - b * s);
            v[k + 32] = (float)(a * s + b * c);
        }
    }

    void mix(const float* norm, const Matrix& down, const Matrix& up, const Matrix* inject, const float* R, int n, float* x, float* inj) {
        std::vector<float> xn((size_t)n * hcWidth), lo((size_t)n * hcRank), gate((size_t)n * hcWidth);
        for(int t = 0; t < n; t++){
            for(int c = 0; c < hcCount; c++){
                size_t at = (size_t)t * hcWidth + (size_t)c * hidden;
                rmsNorm(R + at, norm + (size_t)c * hidden, xn.data() + at, hidden);
            }
        }

        std::vector<Product> list = {{down, xn.data(), lo.data(), n}};
        if(inject) list.push_back({*inject, xn.data(), inj, n});
        multiply(pool, list);

        for(float& v : lo){
            v *= 0.25f;
            v = v * sigmoid(v);
        }
        multiply(pool, {{up, lo.data(), gate.data(), n}});

        for(int t = 0; t < n; t++){
            const float* a = xn.data() + (size_t)t * hcWidth;
            const float* g = gate.data() + (size_t)t * hcWidth;
            for(int d = 0; d < hidden; d++){
                float sum = 0;
                for(int c = 0; c < hcCount; c++) sum += a[c * hidden + d] * sigmoid(g[c * hidden + d]);
                x[(size_t)t * hidden + d] = 0.25f * sum;
            }
        }
    }

    void write(float* R, const float* y, const float* inj, int n) {
        for(int t = 0; t < n; t++){
            for(int c = 0; c < hcCount; c++){
                float w = 2.0f * sigmoid(inj[t * hcCount + c] * 0.25f);
                float* r = R + (size_t)t * hcWidth + (size_t)c * hidden;
                const float* v = y + (size_t)t * hidden;
                for(int d = 0; d < hidden; d++) r[d] += v[d] * w;
            }
        }
    }

    void perLayer(float* R, int n, size_t first) {
        std::vector<float> emb((size_t)n * hidden), key((size_t)n * hcWidth), value((size_t)n * hidden);
        for(int t = 0; t < n; t++){
            size_t s = first + t;
            uint64_t ctx[3] = {(uint64_t)tokens[s], 0, 0};
            bool cut = false;
            for(int i = 1; i <= 2; i++){
                bool missing = s < (size_t)i;
                int previous = missing ? endToken : tokens[s - i];
                cut = cut || missing || previous == endToken;
                ctx[i] = cut ? (uint64_t)endToken : (uint64_t)previous;
            }
            for(int order = 2; order <= 3; order++){
                uint64_t mixed = ctx[0] * pleMultipliers[0] ^ ctx[1] * pleMultipliers[1];
                if(order == 3) mixed ^= ctx[2] * pleMultipliers[2];
                for(int g = 0; g < 8; g++){
                    int h = (order - 2) * 8 + g;
                    uint64_t row = mixed % pleVocab[h] + pleOffset[h];
                    dequantRow(pleTable.type, pleTable.data + row * pleTable.stride, emb.data() + (size_t)t * hidden + h * pleRowWidth, pleRowWidth);
                }
            }
        }

        multiply(pool, {{pleKey, emb.data(), key.data(), n}, {pleValue, emb.data(), value.data(), n}});

        std::vector<float> kn(hidden), qn(hidden), gated(hcWidth), normd(hcWidth);
        for(int t = 0; t < n; t++){
            float* r = R + (size_t)t * hcWidth;
            const float* v = value.data() + (size_t)t * hidden;
            for(int c = 0; c < hcCount; c++){
                size_t at = (size_t)c * hidden;
                rmsNorm(key.data() + (size_t)t * hcWidth + at, pleNormKey + at, kn.data(), hidden);
                rmsNorm(r + at, pleNormQuery + at, qn.data(), hidden);
                double s = 0;
                for(int d = 0; d < hidden; d++) s += (double)kn[d] * qn[d];
                s /= sqrt((double)hidden);
                double sign = s > 0
                    ? 1.0
                    : s < 0
                        ? -1.0
                        : 0.0;
                float g = sigmoid((float)(sign * sqrt(std::max(fabs(s), 1e-6))));
                for(int d = 0; d < hidden; d++) gated[at + d] = v[d] * g;
                rmsNorm(gated.data() + at, pleNormConv + at, normd.data() + at, hidden);
            }

            for(int ch = 0; ch < hcWidth; ch++){
                const float* w = pleConv.data() + (size_t)ch * 4;
                float conv = w[0] * pleHistory[(size_t)0 * hcWidth + ch] + w[1] * pleHistory[(size_t)3 * hcWidth + ch] + w[2] * pleHistory[(size_t)6 * hcWidth + ch] + w[3] * normd[ch];
                r[ch] += gated[ch] + conv * sigmoid(conv);
            }

            memmove(pleHistory.data(), pleHistory.data() + hcWidth, sizeof(float) * (size_t)(pleTaps - 1) * hcWidth);
            memcpy(pleHistory.data() + (size_t)(pleTaps - 1) * hcWidth, normd.data(), sizeof(float) * hcWidth);
        }
    }

    void deltaNet(Layer& L, const float* x, int n, float* y) {
        std::vector<float> qkv((size_t)n * convWidth), z((size_t)n * innerWidth), a((size_t)n * valueHeads), b((size_t)n * valueHeads);
        multiply(pool, {{L.qkv, x, qkv.data(), n}, {L.gate, x, z.data(), n}, {L.alpha, x, a.data(), n}, {L.beta, x, b.data(), n}});

        std::vector<float> u((size_t)n * convWidth);
        for(int t = 0; t < n; t++){
            for(int c = 0; c < convWidth; c++){
                const float* w = L.convWeight + (size_t)c * 4;
                float s = 0;
                for(int k = 0; k < 4; k++){
                    int when = t - 3 + k;
                    float v = when >= 0
                        ? qkv[(size_t)when * convWidth + c]
                        : L.convHistory[(size_t)(when + 3) * convWidth + c];
                    s += w[k] * v;
                }
                u[(size_t)t * convWidth + c] = s * sigmoid(s);
            }
            for(int h = 0; h < 2 * keyHeads; h++){
                float* v = u.data() + (size_t)t * convWidth + (size_t)h * ssmSize;
                double sum = 0;
                for(int i = 0; i < ssmSize; i++) sum += (double)v[i] * v[i];
                float inv = (float)(1.0 / sqrt(sum + 1e-6));
                for(int i = 0; i < ssmSize; i++) v[i] *= inv;
            }
        }

        std::vector<float> history((size_t)3 * convWidth);
        for(int k = 0; k < 3; k++){
            int when = n - 3 + k;
            const float* src = when >= 0
                ? qkv.data() + (size_t)when * convWidth
                : L.convHistory.data() + (size_t)(when + 3) * convWidth;
            memcpy(history.data() + (size_t)k * convWidth, src, sizeof(float) * convWidth);
        }
        L.convHistory.swap(history);

        std::vector<float> out((size_t)n * innerWidth);
        float scale = 1.0f / sqrtf((float)ssmSize);
        pool.run(valueHeads, [&](int h){
            float* S = L.state.data() + (size_t)h * ssmSize * ssmSize;
            int kh = h % keyHeads;
            alignas(32) float kv[ssmSize], delta[ssmSize], o[ssmSize];
            for(int t = 0; t < n; t++){
                const float* row = u.data() + (size_t)t * convWidth;
                const float* q = row + (size_t)kh * ssmSize;
                const float* k = row + (size_t)keyHeads * ssmSize + (size_t)kh * ssmSize;
                const float* v = row + (size_t)2 * keyHeads * ssmSize + (size_t)h * ssmSize;
                float raw = a[(size_t)t * valueHeads + h] + L.dtBias[h];
                float soft = raw > 20.0f ? raw : log1pf(expf(raw));
                float g = expf(soft * L.decay[h]);
                float beta = sigmoid(b[(size_t)t * valueHeads + h]);

                for(int j = 0; j < ssmSize; j++) kv[j] = 0;
                for(int i = 0; i < ssmSize; i++){
                    const float* s = S + (size_t)i * ssmSize;
                    float ki = k[i];
                    for(int j = 0; j < ssmSize; j++) kv[j] += s[j] * ki;
                }
                for(int j = 0; j < ssmSize; j++){
                    delta[j] = (v[j] - g * kv[j]) * beta;
                    o[j] = 0;
                }
                for(int i = 0; i < ssmSize; i++){
                    float* s = S + (size_t)i * ssmSize;
                    float ki = k[i], qi = q[i];
                    for(int j = 0; j < ssmSize; j++){
                        s[j] = g * s[j] + ki * delta[j];
                        o[j] += s[j] * qi;
                    }
                }

                double sum = 0;
                for(int j = 0; j < ssmSize; j++){
                    o[j] *= scale;
                    sum += (double)o[j] * o[j];
                }
                float inv = (float)(1.0 / sqrt(sum / ssmSize + 1e-6));
                const float* zh = z.data() + (size_t)t * innerWidth + (size_t)h * ssmSize;
                float* dst = out.data() + (size_t)t * innerWidth + (size_t)h * ssmSize;
                for(int j = 0; j < ssmSize; j++) dst[j] = o[j] * inv * L.ssmNorm[j] * sigmoid(zh[j]);
            }
        });

        multiply(pool, {{L.ssmOut, out.data(), y, n}});
    }

    void attend(Layer& L, const float* x, int n, const int* positions, float* y) {
        const int qWidth = queryHeads * 2 * headWidth;
        const int kvWidth = kvHeads * headWidth;
        std::vector<float> q((size_t)n * qWidth), k((size_t)n * kvWidth), v((size_t)n * kvWidth), iq((size_t)n * indexHeads * indexWidth), ik((size_t)n * indexWidth);
        multiply(pool, {{L.query, x, q.data(), n}, {L.key, x, k.data(), n}, {L.value, x, v.data(), n}, {L.indexQuery, x, iq.data(), n}, {L.indexKey, x, ik.data(), n}});

        int top = 0;
        for(int t = 0; t < n; t++) top = std::max(top, positions[t]);
        size_t cells = (size_t)top + 1;
        if(L.keys.size() < cells * kvWidth){
            L.keys.resize(cells * kvWidth, 0.0f);
            L.values.resize(cells * kvWidth, 0.0f);
            L.indexRaw.resize(cells * indexWidth, 0.0f);
            L.pooled.resize(cells / 4 * indexWidth + indexWidth, 0.0f);
        }

        for(int t = 0; t < n; t++){
            int p = positions[t];
            for(int h = 0; h < queryHeads; h++){
                float* qh = q.data() + (size_t)t * qWidth + (size_t)h * 2 * headWidth;
                rmsNorm(qh, L.queryNorm, qh, headWidth);
                rope(qh, p);
            }
            for(int h = 0; h < kvHeads; h++){
                float* kh = k.data() + (size_t)t * kvWidth + (size_t)h * headWidth;
                rmsNorm(kh, L.keyNorm, kh, headWidth);
                rope(kh, p);
            }
            memcpy(L.keys.data() + (size_t)p * kvWidth, k.data() + (size_t)t * kvWidth, sizeof(float) * kvWidth);
            memcpy(L.values.data() + (size_t)p * kvWidth, v.data() + (size_t)t * kvWidth, sizeof(float) * kvWidth);

            for(int h = 0; h < indexHeads; h++){
                float* qh = iq.data() + ((size_t)t * indexHeads + h) * indexWidth;
                rmsNorm(qh, L.indexQueryNorm, qh, indexWidth);
                rope(qh, p);
            }
            for(int i = 0; i < indexWidth; i++){
                float raw = ik[(size_t)t * indexWidth + i];
                L.indexRaw[(size_t)p * indexWidth + i] = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtps_ph(_mm_set_ss(raw), 0)));
            }
            if((p + 1) % 4 == 0){
                int block = (p + 1) / 4 - 1;
                float* pooled = L.pooled.data() + (size_t)block * indexWidth;
                for(int i = 0; i < indexWidth; i++){
                    float sum = 0;
                    for(int c = 0; c < 4; c++) sum += L.indexRaw[(size_t)(4 * block + c) * indexWidth + i];
                    pooled[i] = sum * 0.25f;
                }
                rmsNorm(pooled, L.indexKeyNorm, pooled, indexWidth);
                rope(pooled, 4 * block);
            }
        }

        std::vector<std::vector<int>> chosen(n);
        for(int t = 0; t < n; t++){
            int p = positions[t];
            if(p + 1 <= budget) continue;
            int blocks = (p + 1) / 4;
            std::vector<int>& set = chosen[t];
            for(int c = 4 * blocks; c <= p; c++) set.push_back(c);
            std::vector<std::pair<float, int>> ranked(blocks);
            for(int bl = 0; bl < blocks; bl++){
                float score = 0;
                for(int h = 0; h < indexHeads; h++){
                    score += std::max(0.0f, dot(iq.data() + ((size_t)t * indexHeads + h) * indexWidth, L.pooled.data() + (size_t)bl * indexWidth, indexWidth));
                }
                ranked[bl] = {score, bl};
            }
            std::sort(ranked.begin(), ranked.end(), [](const std::pair<float, int>& l, const std::pair<float, int>& r){
                return l.first != r.first ? l.first > r.first : l.second < r.second;
            });
            for(size_t i = 0; i < ranked.size() && set.size() < (size_t)budget; i++){
                for(int c = 0; c < 4 && set.size() < (size_t)budget; c++) set.push_back(4 * ranked[i].second + c);
            }
        }

        std::vector<float> o((size_t)n * innerWidth);
        float scale = 1.0f / sqrtf((float)headWidth);
        pool.run(n * queryHeads, [&](int job){
            int t = job / queryHeads, h = job % queryHeads, kvh = h / (queryHeads / kvHeads);
            int p = positions[t];
            const float* qh = q.data() + (size_t)t * qWidth + (size_t)h * 2 * headWidth;
            const float* gate = qh + headWidth;
            const std::vector<int>& set = chosen[t];
            int count = set.empty() ? p + 1 : (int)set.size();
            std::vector<float> score(count);
            float best = -1e30f;
            for(int i = 0; i < count; i++){
                int cell = set.empty() ? i : set[i];
                score[i] = dot(qh, L.keys.data() + (size_t)cell * kvWidth + (size_t)kvh * headWidth, headWidth) * scale;
                best = std::max(best, score[i]);
            }
            double sum = 0;
            for(int i = 0; i < count; i++){
                score[i] = expf(score[i] - best);
                sum += score[i];
            }
            alignas(32) float acc[headWidth] = {};
            for(int i = 0; i < count; i++){
                int cell = set.empty() ? i : set[i];
                float w = (float)(score[i] / sum);
                const float* vv = L.values.data() + (size_t)cell * kvWidth + (size_t)kvh * headWidth;
                for(int d = 0; d < headWidth; d++) acc[d] += w * vv[d];
            }
            float* dst = o.data() + (size_t)t * innerWidth + (size_t)h * headWidth;
            for(int d = 0; d < headWidth; d++) dst[d] = acc[d] * sigmoid(gate[d]);
        });

        multiply(pool, {{L.attnOut, o.data(), y, n}});
    }

    void experts(Layer& L, const float* x, int n, float* y) {
        std::vector<float> logits((size_t)n * expertCount), scalar(n), sg((size_t)n * expertWidth), su((size_t)n * expertWidth), shared((size_t)n * hidden);
        multiply(pool, {{L.router, x, logits.data(), n}, {L.sharedScalar, x, scalar.data(), n}, {L.sharedGate, x, sg.data(), n}, {L.sharedUp, x, su.data(), n}});

        std::vector<std::vector<std::pair<int, float>>> routed(expertCount);
        L.lastRouting.clear();
        if(L.routeCount.size() != (size_t)expertCount) L.routeCount.assign(expertCount, 0);
        std::vector<double> prob(expertCount);
        std::vector<int> order(expertCount);
        for(int t = 0; t < n; t++){
            const float* lg = logits.data() + (size_t)t * expertCount;
            float top = *std::max_element(lg, lg + expertCount);
            for(int e = 0; e < expertCount; e++) prob[e] = exp((double)lg[e] - top);
            std::iota(order.begin(), order.end(), 0);
            std::partial_sort(order.begin(), order.begin() + expertsUsed, order.end(), [&](int l, int r){
                return prob[l] != prob[r] ? prob[l] > prob[r] : l < r;
            });
            double kept = 0;
            for(int k = 0; k < expertsUsed; k++) kept += prob[order[k]];
            for(int k = 0; k < expertsUsed; k++){
                routed[order[k]].push_back({t, (float)(prob[order[k]] / kept)});
                L.routeCount[order[k]]++;
                if(t == n - 1) L.lastRouting.push_back(order[k]);
            }
        }

        std::vector<int> used, start;
        int total = 0;
        for(int e = 0; e < expertCount; e++){
            if(routed[e].empty()) continue;
            used.push_back(e);
            start.push_back(total);
            total += (int)routed[e].size();
        }

        uint64_t gateSpan = (uint64_t)L.expertGate.rows * L.expertGate.stride;
        uint64_t upSpan = (uint64_t)L.expertUp.rows * L.expertUp.stride;
        uint64_t downSpan = (uint64_t)L.expertDown.rows * L.expertDown.stride;

        std::vector<float> gathered((size_t)total * hidden), gateOut((size_t)total * expertWidth), upOut((size_t)total * expertWidth), downOut((size_t)total * hidden);
        std::vector<Product> list;
        for(size_t i = 0; i < used.size(); i++){
            int e = used[i];
            int count = (int)routed[e].size();
            for(int r = 0; r < count; r++){
                memcpy(gathered.data() + (size_t)(start[i] + r) * hidden, x + (size_t)routed[e][r].first * hidden, sizeof(float) * hidden);
            }
            Matrix g = L.expertGate, u = L.expertUp;
            g.data += e * gateSpan;
            u.data += e * upSpan;
            list.push_back({g, gathered.data() + (size_t)start[i] * hidden, gateOut.data() + (size_t)start[i] * expertWidth, count});
            list.push_back({u, gathered.data() + (size_t)start[i] * hidden, upOut.data() + (size_t)start[i] * expertWidth, count});
        }
        multiply(pool, list);

        for(size_t i = 0; i < gateOut.size(); i++) gateOut[i] = gateOut[i] * sigmoid(gateOut[i]) * upOut[i];
        for(size_t i = 0; i < sg.size(); i++) sg[i] = sg[i] * sigmoid(sg[i]) * su[i];

        list.clear();
        for(size_t i = 0; i < used.size(); i++){
            Matrix d = L.expertDown;
            d.data += used[i] * downSpan;
            list.push_back({d, gateOut.data() + (size_t)start[i] * expertWidth, downOut.data() + (size_t)start[i] * hidden, (int)routed[used[i]].size()});
        }
        list.push_back({L.sharedDown, sg.data(), shared.data(), n});
        multiply(pool, list);

        for(int t = 0; t < n; t++){
            float gs = sigmoid(scalar[t]);
            for(int d = 0; d < hidden; d++) y[(size_t)t * hidden + d] = gs * shared[(size_t)t * hidden + d];
        }
        for(size_t i = 0; i < used.size(); i++){
            const std::vector<std::pair<int, float>>& items = routed[used[i]];
            for(size_t r = 0; r < items.size(); r++){
                float* dst = y + (size_t)items[r].first * hidden;
                const float* src = downOut.data() + (size_t)(start[i] + r) * hidden;
                float w = items[r].second;
                for(int d = 0; d < hidden; d++) dst[d] += w * src[d];
            }
        }
    }

    std::vector<float> forward(const std::vector<int>& ids, const std::vector<int>& positions) {
        std::vector<float> logits;
        if(ids.empty() || ids.size() != positions.size() || maps.empty()) return logits;

        layerMs.assign(layers.size(), 0.0);
        embedMs = 0;
        headMs = 0;
        auto now = []{ return std::chrono::steady_clock::now(); };
        auto since = [](std::chrono::steady_clock::time_point from){ return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - from).count(); };

        std::vector<float> last(hcWidth);
        for(size_t at = 0; at < ids.size(); at += chunkTokens){
            int n = (int)std::min(ids.size() - at, (size_t)chunkTokens);
            const int* pos = positions.data() + at;
            size_t first = tokens.size();
            tokens.insert(tokens.end(), ids.begin() + at, ids.begin() + at + n);

            auto clock = now();
            std::vector<float> R((size_t)n * hcWidth), e(hidden);
            for(int t = 0; t < n; t++){
                int id = ids[at + t];
                if(id < 0 || id >= embed.rows) id = 0;
                dequantRow(embed.type, embed.data + (uint64_t)id * embed.stride, e.data(), hidden);
                for(int c = 0; c < hcCount; c++) memcpy(R.data() + (size_t)t * hcWidth + (size_t)c * hidden, e.data(), sizeof(float) * hidden);
            }
            embedMs += since(clock);

            std::vector<float> x((size_t)n * hidden), y((size_t)n * hidden), inj((size_t)n * hcCount);
            for(int l = 0; l < (int)layers.size(); l++){
                if(stopGenerating){
                    problem = "stopped";
                    return {};
                }
                Layer& L = layers[l];
                clock = now();
                if(l == 1 && usePle) perLayer(R.data(), n, first);

                mix(L.hcNorm[0], L.hcDown[0], L.hcUp[0], &L.hcInject[0], R.data(), n, x.data(), inj.data());
                if(L.attention) attend(L, x.data(), n, pos, y.data());
                else deltaNet(L, x.data(), n, y.data());
                write(R.data(), y.data(), inj.data(), n);

                mix(L.hcNorm[1], L.hcDown[1], L.hcUp[1], &L.hcInject[1], R.data(), n, x.data(), inj.data());
                experts(L, x.data(), n, y.data());
                write(R.data(), y.data(), inj.data(), n);
                layerMs[l] += since(clock);
            }

            memcpy(last.data(), R.data() + (size_t)(n - 1) * hcWidth, sizeof(float) * hcWidth);
            if(keepRows) rows.insert(rows.end(), R.begin(), R.end());
        }

        auto clock = now();
        std::vector<float> fin(hidden);
        mix(outNorm, outDown, outUp, nullptr, last.data(), 1, fin.data(), nullptr);
        logits.resize(head.rows);
        multiply(pool, {{head, fin.data(), logits.data(), 1}});
        headMs = since(clock);
        return logits;
    }
};
