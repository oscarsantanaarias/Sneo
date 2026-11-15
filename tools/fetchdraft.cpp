#include "../src/fetch.h"
#include "../src/json.h"
#include <iostream>
#include <fstream>
#include <thread>
#include <atomic>
#include <mutex>
#include <set>
#include <map>
#include <chrono>
#include <algorithm>
#include <cstring>

struct Part {
    std::string source;
    std::string name;
    std::string shard;
    std::vector<uint64_t> shape;
    uint64_t begin = 0;
    uint64_t bytes = 0;
    uint64_t offset = 0;
};

struct Piece {
    int part;
    uint64_t from;
    uint64_t length;
};

const char* layerNames[][2] = {
    {"mtp.pre_fc_norm_embedding.weight", "draft.embed_norm"},
    {"mtp.pre_fc_norm_hidden.weight", "draft.hidden_norm"},
    {"mtp.fc_embedding.weight", "draft.embed_proj"},
    {"mtp.fc_hidden.weight", "draft.hidden_proj"},
    {"mtp.hyper_connection_mixer.hc_norm.weight", "draft.out_hc_norm"},
    {"mtp.hyper_connection_mixer.input_mix_weight_down.weight", "draft.out_hc_down"},
    {"mtp.hyper_connection_mixer.input_mix_weight_up.weight", "draft.out_hc_up"},
    {"mtp.layers.0.attn_hyper_connection.hc_norm.weight", "draft.hc_attn_norm"},
    {"mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight", "draft.hc_attn_down"},
    {"mtp.layers.0.attn_hyper_connection.input_mix_weight_up.weight", "draft.hc_attn_up"},
    {"mtp.layers.0.attn_hyper_connection.block_inject_weight.weight", "draft.hc_attn_inject"},
    {"mtp.layers.0.mlp_hyper_connection.hc_norm.weight", "draft.hc_ffn_norm"},
    {"mtp.layers.0.mlp_hyper_connection.input_mix_weight_down.weight", "draft.hc_ffn_down"},
    {"mtp.layers.0.mlp_hyper_connection.input_mix_weight_up.weight", "draft.hc_ffn_up"},
    {"mtp.layers.0.mlp_hyper_connection.block_inject_weight.weight", "draft.hc_ffn_inject"},
    {"mtp.layers.0.self_attn.q_proj.weight", "draft.attn_q"},
    {"mtp.layers.0.self_attn.k_proj.weight", "draft.attn_k"},
    {"mtp.layers.0.self_attn.v_proj.weight", "draft.attn_v"},
    {"mtp.layers.0.self_attn.o_proj.weight", "draft.attn_output"},
    {"mtp.layers.0.self_attn.q_norm.weight", "draft.attn_q_norm"},
    {"mtp.layers.0.self_attn.k_norm.weight", "draft.attn_k_norm"},
    {"mtp.layers.0.self_attn.indexer.index_qk_proj.weight", "draft.indexer.qk_proj"},
    {"mtp.layers.0.self_attn.indexer.q_layernorm.weight", "draft.indexer.q_norm"},
    {"mtp.layers.0.self_attn.indexer.k_layernorm.weight", "draft.indexer.k_norm"},
    {"mtp.layers.0.mlp.gate.weight", "draft.ffn_gate_inp"},
    {"mtp.layers.0.mlp.shared_expert_gate.weight", "draft.ffn_gate_inp_shexp"},
    {"mtp.layers.0.mlp.shared_expert.gate_proj.weight", "draft.ffn_gate_shexp"},
    {"mtp.layers.0.mlp.shared_expert.up_proj.weight", "draft.ffn_up_shexp"},
    {"mtp.layers.0.mlp.shared_expert.down_proj.weight", "draft.ffn_down_shexp"},
    {"mtp.layers.0.mlp.experts.gate_up_proj", "draft.ffn_gate_up_exps"},
    {"mtp.layers.0.mlp.experts.down_proj", "draft.ffn_down_exps"}
};

void putText(std::string& out, const std::string& text) {
    uint64_t length = text.size();
    out.append((const char*)&length, 8);
    out.append(text);
}

template<class T> void putValue(std::string& out, T value) {
    out.append((const char*)&value, sizeof(T));
}

int main(int argc, char** argv) {
    std::string target = argc > 1 ? argv[1] : "modelos\\Qwen3.8-Flash-Next.draft.gguf";
    std::string revision = argc > 2 ? argv[2] : "";
    const std::string repo = "Qwen/Qwen3.8-Flash-Next";
    const uint64_t align = 4096;
    const uint64_t chunk = 32ULL << 20;
    const int workers = 8;

    WebLink link;
    std::string error, body;
    if(revision.empty()){
        if(!link.text("https://huggingface.co/api/models/" + repo, body, error)){
            std::cerr << "cannot read the model card: " << error << std::endl;
            return 1;
        }
        revision = parseJson(body)["sha"].str();
        if(revision.empty()){
            std::cerr << "the model card has no revision" << std::endl;
            return 1;
        }
    }
    std::string base = "https://huggingface.co/" + repo + "/resolve/" + revision + "/";
    std::cout << "draft layer from " << repo << " at " << revision << std::endl;

    if(!link.text(base + "model.safetensors.index.json", body, error)){
        std::cerr << "cannot read the weight index: " << error << std::endl;
        return 1;
    }
    Json index = parseJson(body);
    const Json& where = index["weight_map"];

    std::vector<Part> parts;
    for(auto& entry : layerNames){
        Part p;
        p.source = entry[0];
        p.name = entry[1];
        p.shard = where[p.source].str();
        if(p.shard.empty()){
            std::cerr << "the checkpoint has no " << p.source << std::endl;
            return 1;
        }
        parts.push_back(p);
    }

    std::map<std::string, Json> headers;
    std::map<std::string, uint64_t> starts;
    for(Part& p : parts){
        if(!headers.count(p.shard)){
            std::string head;
            auto keep = [&](const char* data, size_t length){
                head.append(data, length);
                return true;
            };
            if(!link.get(base + p.shard, 0, 8, keep, error)){
                std::cerr << p.shard << ": " << error << std::endl;
                return 1;
            }
            uint64_t length = 0;
            memcpy(&length, head.data(), 8);
            head.clear();
            if(length == 0 || length > (100u << 20) || !link.get(base + p.shard, 8, 8 + length, keep, error)){
                std::cerr << p.shard << ": " << (error.empty() ? "bad header" : error) << std::endl;
                return 1;
            }
            headers[p.shard] = parseJson(head);
            starts[p.shard] = 8 + length;
        }

        const Json& info = headers[p.shard][p.source];
        if(info["dtype"].str() != "BF16" || info["data_offsets"].items.size() != 2){
            std::cerr << p.source << " is not a bf16 tensor" << std::endl;
            return 1;
        }
        uint64_t count = 2;
        for(const Json& d : info["shape"].items){
            p.shape.push_back((uint64_t)d.num());
            count *= p.shape.back();
        }
        uint64_t first = (uint64_t)info["data_offsets"].items[0].num();
        uint64_t last = (uint64_t)info["data_offsets"].items[1].num();
        p.begin = starts[p.shard] + first;
        p.bytes = last - first;
        if(count != p.bytes){
            std::cerr << p.source << " has " << p.bytes << " bytes for its shape" << std::endl;
            return 1;
        }
    }

    uint64_t cursor = 0;
    for(Part& p : parts){
        p.offset = cursor;
        cursor += (p.bytes + align - 1) / align * align;
    }

    std::string header;
    header.append("GGUF", 4);
    putValue<uint32_t>(header, 3);
    putValue<uint64_t>(header, parts.size());
    putValue<uint64_t>(header, 3);
    putText(header, "general.alignment");
    putValue<uint32_t>(header, 4);
    putValue<uint32_t>(header, (uint32_t)align);
    putText(header, "draft.source");
    putValue<uint32_t>(header, 8);
    putText(header, repo);
    putText(header, "draft.revision");
    putValue<uint32_t>(header, 8);
    putText(header, revision);
    for(Part& p : parts){
        putText(header, p.name);
        putValue<uint32_t>(header, (uint32_t)p.shape.size());
        for(size_t d = p.shape.size(); d-- > 0;) putValue<uint64_t>(header, p.shape[d]);
        putValue<uint32_t>(header, 30);
        putValue<uint64_t>(header, p.offset);
    }
    uint64_t dataStart = (header.size() + align - 1) / align * align;
    header.resize(dataStart, '\0');
    uint64_t total = dataStart + cursor;

    std::vector<Piece> pieces;
    for(int i = 0; i < (int)parts.size(); i++){
        for(uint64_t at = 0; at < parts[i].bytes; at += chunk) pieces.push_back({i, at, std::min(chunk, parts[i].bytes - at)});
    }

    std::string progressPath = target + ".progress";
    std::string stamp = revision + " " + std::to_string(total) + " " + std::to_string(pieces.size());
    std::set<size_t> done;
    {
        std::ifstream in(progressPath);
        std::string first;
        if(in && std::getline(in, first) && first == stamp){
            size_t value = 0;
            while(in >> value) done.insert(value);
        }
    }
    DWORD attributes = GetFileAttributesA(target.c_str());
    if(attributes == INVALID_FILE_ATTRIBUTES) done.clear();
    bool fresh = done.empty();

    HANDLE file = CreateFileA(target.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr, fresh ? CREATE_ALWAYS : OPEN_ALWAYS, FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH, nullptr);
    if(file == INVALID_HANDLE_VALUE){
        std::cerr << "cannot write " << target << " (" << GetLastError() << ")" << std::endl;
        return 1;
    }

    auto writeAt = [&](const void* data, uint64_t length, uint64_t at) {
        OVERLAPPED spot = {};
        spot.Offset = (DWORD)at;
        spot.OffsetHigh = (DWORD)(at >> 32);
        DWORD wrote = 0;
        return WriteFile(file, data, (DWORD)length, &wrote, &spot) && wrote == length;
    };

    uint8_t* top = (uint8_t*)VirtualAlloc(nullptr, dataStart, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    memcpy(top, header.data(), dataStart);
    bool headerWritten = writeAt(top, dataStart, 0);
    VirtualFree(top, 0, MEM_RELEASE);
    if(!headerWritten){
        std::cerr << "cannot write the header (" << GetLastError() << ")" << std::endl;
        CloseHandle(file);
        return 1;
    }

    uint64_t wanted = 0;
    uint64_t already = 0;
    for(size_t i = 0; i < pieces.size(); i++){
        if(done.count(i)) already += pieces[i].length;
        else wanted += pieces[i].length;
    }
    std::cout << parts.size() << " tensors in " << headers.size() << " shards, " << (double)(wanted + already) / 1e9 << " GB, " << (double)already / 1e9 << " GB already here" << std::endl;

    std::ofstream progress(progressPath, fresh ? std::ios::trunc : std::ios::app);
    if(fresh) progress << stamp << "\n";
    progress.flush();

    std::atomic<size_t> next{0};
    std::atomic<uint64_t> fetched{0};
    std::atomic<bool> failed{false};
    std::atomic<bool> finished{false};
    std::mutex note;
    auto started = std::chrono::steady_clock::now();

    auto work = [&]{
        WebLink own;
        uint8_t* staging = (uint8_t*)VirtualAlloc(nullptr, chunk + align, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        while(!failed){
            size_t pick = next++;
            if(pick >= pieces.size()) break;
            if(done.count(pick)) continue;
            const Piece& piece = pieces[pick];
            const Part& part = parts[piece.part];
            std::string problem;
            bool ok = false;
            for(int attempt = 0; attempt < 6 && !ok && !failed; attempt++){
                if(attempt) Sleep(1000u << attempt);
                uint64_t filled = 0;
                ok = own.get(base + part.shard, part.begin + piece.from, part.begin + piece.from + piece.length, [&](const char* data, size_t length){
                    if(filled + length > piece.length) return false;
                    memcpy(staging + filled, data, length);
                    filled += length;
                    return true;
                }, problem);
            }
            if(!ok){
                std::lock_guard<std::mutex> guard(note);
                std::cerr << part.name << ": " << problem << std::endl;
                failed = true;
                break;
            }
            uint64_t padded = (piece.length + align - 1) / align * align;
            memset(staging + piece.length, 0, (size_t)(padded - piece.length));
            if(!writeAt(staging, padded, dataStart + part.offset + piece.from)){
                std::lock_guard<std::mutex> guard(note);
                std::cerr << part.name << ": write failed (" << GetLastError() << ")" << std::endl;
                failed = true;
                break;
            }
            fetched += piece.length;
            std::lock_guard<std::mutex> guard(note);
            progress << pick << "\n";
            progress.flush();
        }
        VirtualFree(staging, 0, MEM_RELEASE);
    };

    std::thread watch([&]{
        while(!finished){
            for(int i = 0; i < 20 && !finished; i++) Sleep(100);
            double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            std::lock_guard<std::mutex> guard(note);
            std::cout << (double)fetched / 1e9 << " of " << (double)wanted / 1e9 << " GB, " << (seconds > 0 ? (double)fetched / 1e6 / seconds : 0) << " MB/s" << std::endl;
        }
    });

    std::vector<std::thread> crew;
    for(int i = 0; i < workers; i++) crew.emplace_back(work);
    for(auto& t : crew) t.join();
    finished = true;
    watch.join();
    CloseHandle(file);
    progress.close();

    if(failed){
        std::cerr << "download stopped, run again to resume" << std::endl;
        return 1;
    }
    DeleteFileA(progressPath.c_str());
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    std::cout << "draft layer ready: " << target << " (" << (double)total / 1e9 << " GB) in " << seconds << " s" << std::endl;
    return 0;
}
