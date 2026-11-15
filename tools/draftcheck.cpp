#include "../src/fetch.h"
#include "../src/json.h"
#include "../src/gguf.h"
#include "../src/quant.h"
#include <iostream>
#include <map>
#include <cmath>
#include <cstring>

struct Remote {
    WebLink link;
    std::string base;
    std::map<std::string, std::pair<Json, uint64_t>> heads;
    std::string error;

    bool locate(const std::string& shard, const std::string& name, uint64_t& begin, std::vector<uint64_t>& shape) {
        if(!heads.count(shard)){
            std::string raw;
            auto keep = [&](const char* data, size_t length){
                raw.append(data, length);
                return true;
            };
            if(!link.get(base + shard, 0, 8, keep, error)) return false;
            uint64_t length = 0;
            memcpy(&length, raw.data(), 8);
            raw.clear();
            if(!link.get(base + shard, 8, 8 + length, keep, error)) return false;
            heads[shard] = {parseJson(raw), 8 + length};
        }
        const Json& info = heads[shard].first[name];
        if(info["data_offsets"].items.size() != 2){
            error = name + " missing in " + shard;
            return false;
        }
        begin = heads[shard].second + (uint64_t)info["data_offsets"].items[0].num();
        shape.clear();
        for(const Json& d : info["shape"].items) shape.push_back((uint64_t)d.num());
        return true;
    }

    std::vector<float> rows(const std::string& shard, const std::string& name, uint64_t first, uint64_t count) {
        uint64_t begin = 0;
        std::vector<uint64_t> shape;
        if(!locate(shard, name, begin, shape)) return {};
        uint64_t cols = shape.back();
        std::string raw;
        uint64_t from = begin + first * cols * 2;
        if(!link.get(base + shard, from, from + count * cols * 2, [&](const char* data, size_t length){
            raw.append(data, length);
            return true;
        }, error)) return {};
        std::vector<float> out(count * cols);
        for(size_t i = 0; i < out.size(); i++){
            uint32_t bits = (uint32_t)(uint8_t)raw[2 * i] << 16 | (uint32_t)(uint8_t)raw[2 * i + 1] << 24;
            memcpy(&out[i], &bits, 4);
        }
        return out;
    }
};

std::vector<float> localRows(const ModelFile& model, const std::string& name, uint64_t first, uint64_t count) {
    const TensorInfo* t = findTensor(model, name);
    if(!t) return {};
    uint64_t cols = t->dims[0];
    uint64_t stride = rowBytes(t->type, cols);
    std::vector<uint8_t> raw(count * stride);
    FILE* f = fopen(model.paths[t->file].c_str(), "rb");
    if(!f) return {};
    _fseeki64(f, (int64_t)(t->offset + first * stride), SEEK_SET);
    size_t got = fread(raw.data(), 1, raw.size(), f);
    fclose(f);
    if(got != raw.size()) return {};
    std::vector<float> out(count * cols);
    for(uint64_t r = 0; r < count; r++) dequantRow(t->type, raw.data() + r * stride, out.data() + r * cols, (int64_t)cols);
    return out;
}

double correlation(const float* a, const float* b, size_t n) {
    double ma = 0, mb = 0;
    for(size_t i = 0; i < n; i++){
        ma += a[i];
        mb += b[i];
    }
    ma /= n;
    mb /= n;
    double ab = 0, aa = 0, bb = 0;
    for(size_t i = 0; i < n; i++){
        ab += (a[i] - ma) * (b[i] - mb);
        aa += (a[i] - ma) * (a[i] - ma);
        bb += (b[i] - mb) * (b[i] - mb);
    }
    return aa > 0 && bb > 0 ? ab / sqrt(aa * bb) : 0;
}

void compare(const char* label, const std::vector<float>& ours, const std::vector<float>& theirs) {
    if(ours.empty() || theirs.empty() || ours.size() != theirs.size()){
        std::cout << label << ": cannot compare (" << ours.size() << " vs " << theirs.size() << ")" << std::endl;
        return;
    }
    double same = 0, shifted = 0;
    for(size_t i = 0; i < ours.size(); i++){
        same = std::max(same, (double)fabs(ours[i] - theirs[i]));
        shifted = std::max(shifted, (double)fabs(ours[i] - theirs[i] - 1.0f));
    }
    std::cout << label << ": max diff as is " << same << ", with +1 " << shifted << ", correlation " << correlation(ours.data(), theirs.data(), ours.size()) << std::endl;
}

int main(int argc, char** argv) {
    std::string modelPath = argc > 1 ? argv[1] : "modelos\\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf";
    std::string revision = argc > 2 ? argv[2] : "main";
    ModelFile model;
    std::string error;
    if(!openModel(modelPath, model, error)){
        std::cerr << error << std::endl;
        return 1;
    }

    Remote hub;
    hub.base = "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/" + revision + "/";
    const std::string pre = "model.language_model.layers.47.";
    const std::string mixShard = "model-00118-of-00131.safetensors";
    const std::string upShard = "model-00119-of-00131.safetensors";
    const std::string restShard = "model-00120-of-00131.safetensors";

    compare("hc attn norm", localRows(model, "blk.47.hc_attn_norm.weight", 0, 1), hub.rows(mixShard, pre + "attn_hyper_connection.hc_norm.weight", 0, 1));
    compare("hc attn down", localRows(model, "blk.47.hc_attn_down.weight", 0, 320), hub.rows(mixShard, pre + "attn_hyper_connection.input_mix_weight_down.weight", 0, 320));
    compare("hc attn up", localRows(model, "blk.47.hc_attn_up.weight", 0, 10240), hub.rows(mixShard, pre + "attn_hyper_connection.input_mix_weight_up.weight", 0, 10240));
    compare("hc attn inject", localRows(model, "blk.47.hc_attn_inject.weight", 0, 4), hub.rows(mixShard, pre + "attn_hyper_connection.block_inject_weight.weight", 0, 4));
    compare("hc ffn norm", localRows(model, "blk.47.hc_ffn_norm.weight", 0, 1), hub.rows(restShard, pre + "mlp_hyper_connection.hc_norm.weight", 0, 1));
    compare("q norm", localRows(model, "blk.47.attn_q_norm.weight", 0, 1), hub.rows(restShard, pre + "self_attn.q_norm.weight", 0, 1));
    compare("k norm", localRows(model, "blk.47.attn_k_norm.weight", 0, 1), hub.rows(restShard, pre + "self_attn.k_norm.weight", 0, 1));
    compare("index q norm", localRows(model, "blk.47.indexer.q_norm.weight", 0, 1), hub.rows(restShard, pre + "self_attn.indexer.q_layernorm.weight", 0, 1));
    compare("router", localRows(model, "blk.47.ffn_gate_inp.weight", 0, 512), hub.rows(restShard, pre + "mlp.gate.weight", 0, 512));
    compare("shared gate", localRows(model, "blk.47.ffn_gate_inp_shexp.weight", 0, 1), hub.rows(restShard, pre + "mlp.shared_expert_gate.weight", 0, 1));
    compare("query rows 0-63", localRows(model, "blk.47.attn_q.weight", 0, 64), hub.rows(restShard, pre + "self_attn.q_proj.weight", 0, 64));
    compare("query rows 256-319", localRows(model, "blk.47.attn_q.weight", 256, 64), hub.rows(restShard, pre + "self_attn.q_proj.weight", 256, 64));
    compare("index q rows", localRows(model, "blk.47.indexer.q_proj.weight", 0, 512), hub.rows(restShard, pre + "self_attn.indexer.index_qk_proj.weight", 0, 512));
    compare("index k rows", localRows(model, "blk.47.indexer.k_proj.weight", 0, 128), hub.rows(restShard, pre + "self_attn.indexer.index_qk_proj.weight", 512, 128));
    compare("output mixer norm", localRows(model, "output_hc_norm.weight", 0, 1), hub.rows("model-00001-of-00131.safetensors", "model.language_model.hyper_connection_mixer.hc_norm.weight", 0, 1));
    compare("output mixer down", localRows(model, "output_hc_down.weight", 0, 320), hub.rows("model-00001-of-00131.safetensors", "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight", 0, 320));

    std::vector<float> gate = localRows(model, "blk.47.ffn_gate_exps.weight", 0, 640);
    std::vector<float> up = localRows(model, "blk.47.ffn_up_exps.weight", 0, 640);
    std::vector<float> fused = hub.rows(upShard, pre + "mlp.experts.gate_up_proj", 0, 1280);
    if(gate.size() == 640 * 2560 && fused.size() == 1280 * 2560){
        std::vector<float> even(640 * 2560), odd(640 * 2560);
        for(int r = 0; r < 640; r++){
            memcpy(&even[(size_t)r * 2560], &fused[(size_t)(2 * r) * 2560], 2560 * 4);
            memcpy(&odd[(size_t)r * 2560], &fused[(size_t)(2 * r + 1) * 2560], 2560 * 4);
        }
        std::cout << "expert 0 gate vs first half " << correlation(gate.data(), fused.data(), gate.size()) << ", second half " << correlation(gate.data(), fused.data() + 640 * 2560, gate.size()) << ", even rows " << correlation(gate.data(), even.data(), gate.size()) << ", odd rows " << correlation(gate.data(), odd.data(), gate.size()) << std::endl;
        std::cout << "expert 0 up vs first half " << correlation(up.data(), fused.data(), up.size()) << ", second half " << correlation(up.data(), fused.data() + 640 * 2560, up.size()) << ", even rows " << correlation(up.data(), even.data(), up.size()) << ", odd rows " << correlation(up.data(), odd.data(), up.size()) << std::endl;
    }
    else std::cout << "expert gate/up: cannot compare " << gate.size() << " " << fused.size() << " " << hub.error << std::endl;

    compare("expert 0 down", localRows(model, "blk.47.ffn_down_exps.weight", 0, 2560), hub.rows(restShard, pre + "mlp.experts.down_proj", 0, 2560));
    compare("shared gate proj", localRows(model, "blk.47.ffn_gate_shexp.weight", 0, 640), hub.rows(restShard, pre + "mlp.shared_expert.gate_proj.weight", 0, 640));
    compare("shared down proj", localRows(model, "blk.47.ffn_down_shexp.weight", 0, 2560), hub.rows(restShard, pre + "mlp.shared_expert.down_proj.weight", 0, 2560));
    if(!hub.error.empty()) std::cout << "last remote error: " << hub.error << std::endl;
    return 0;
}
