#pragma once
#include "gguf.h"
#include "hardware.h"
#include <filesystem>
#include <functional>
#include <vector>

const uint64_t packAlign = 4096;

struct PackLayer {
    uint64_t start = 0;
    uint64_t stride = 0;
    uint32_t gateBytes = 0;
    uint32_t upBytes = 0;
    uint32_t downBytes = 0;
    uint32_t gateType = 0;
    uint32_t upType = 0;
    uint32_t downType = 0;
};

struct PackHead {
    char tag[4];
    uint32_t version;
    uint32_t layers;
    uint32_t experts;
    uint64_t sourceBytes;
    uint64_t sourceStamp;
};

inline uint64_t alignUp(uint64_t value, uint64_t step) {
    return (value + step - 1) / step * step;
}

inline std::string packPathFor(const std::string& modelPath) {
    std::filesystem::path p(modelPath);
    std::string stem = p.stem().string();
    size_t part = stem.find("-00001-of-");
    if(part != std::string::npos) stem = stem.substr(0, part);
    return (p.parent_path() / (stem + ".mxpack")).string();
}

inline uint64_t sourceStamp(const ModelFile& model) {
    uint64_t h = 1469598103934665603ULL;
    for(const TensorInfo& t : model.tensors){
        if(!t.expert) continue;
        for(unsigned char c : t.name){
            h ^= c;
            h *= 1099511628211ULL;
        }
        h ^= t.offset + t.bytes * 31 + t.type;
        h *= 1099511628211ULL;
    }
    return h;
}

inline bool expertTensors(const ModelFile& model, int layer, const TensorInfo*& gate, const TensorInfo*& up, const TensorInfo*& down) {
    std::string base = "blk." + std::to_string(layer) + ".";
    gate = findTensor(model, base + "ffn_gate_exps.weight");
    up = findTensor(model, base + "ffn_up_exps.weight");
    down = findTensor(model, base + "ffn_down_exps.weight");
    return gate && up && down;
}

inline std::vector<PackLayer> planPack(const ModelFile& model) {
    std::vector<PackLayer> layers(model.layers);
    uint64_t at = packAlign;
    for(uint32_t l = 0; l < model.layers; l++){
        const TensorInfo *gate, *up, *down;
        if(!expertTensors(model, (int)l, gate, up, down)) return {};
        PackLayer& p = layers[l];
        p.gateBytes = (uint32_t)(gate->bytes / model.experts);
        p.upBytes = (uint32_t)(up->bytes / model.experts);
        p.downBytes = (uint32_t)(down->bytes / model.experts);
        p.gateType = gate->type;
        p.upType = up->type;
        p.downType = down->type;
        p.stride = alignUp((uint64_t)p.gateBytes + p.upBytes + p.downBytes, packAlign);
        p.start = at;
        at += p.stride * model.experts;
    }
    return layers;
}

struct ExpertPack {
    HANDLE file = INVALID_HANDLE_VALUE;
    std::vector<PackLayer> layers;
    uint32_t experts = 0;
    std::string problem;

    ~ExpertPack() {
        if(file != INVALID_HANDLE_VALUE) CloseHandle(file);
    }

    bool open(const std::string& path, const ModelFile& model) {
        file = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, nullptr);
        if(file == INVALID_HANDLE_VALUE){
            problem = "no expert pack at " + path;
            return false;
        }

        std::vector<char> head(packAlign);
        OVERLAPPED at = {};
        at.hEvent = CreateEventA(nullptr, TRUE, FALSE, nullptr);
        DWORD got = 0;
        bool ok = ReadFile(file, head.data(), (DWORD)packAlign, nullptr, &at) || GetLastError() == ERROR_IO_PENDING;
        ok = ok && GetOverlappedResult(file, &at, &got, TRUE) && got == packAlign;
        CloseHandle(at.hEvent);
        if(!ok){
            problem = "cannot read the expert pack header";
            return false;
        }

        PackHead info;
        memcpy(&info, head.data(), sizeof(info));
        uint64_t total = 0;
        for(const TensorInfo& t : model.tensors) total += t.bytes;
        if(memcmp(info.tag, "MXPK", 4) != 0 || info.version != 1 || info.layers != model.layers || info.experts != model.experts || info.sourceBytes != total || info.sourceStamp != sourceStamp(model)){
            problem = "the expert pack does not match this model, rebuild it";
            return false;
        }

        layers.assign(info.layers, PackLayer());
        memcpy(layers.data(), head.data() + sizeof(PackHead), sizeof(PackLayer) * info.layers);
        experts = info.experts;
        return true;
    }

    uint64_t spanOf(int layer) const {
        return layers[layer].stride;
    }

    bool readMany(const std::vector<std::pair<int, int>>& wanted, const std::vector<char*>& targets) {
        const size_t depth = 32;
        std::vector<OVERLAPPED> slots(std::min(depth, wanted.size()));
        std::vector<HANDLE> events(slots.size());
        for(size_t i = 0; i < slots.size(); i++) events[i] = CreateEventA(nullptr, TRUE, FALSE, nullptr);

        bool ok = true;
        size_t next = 0;
        std::vector<int> busy(slots.size(), -1);
        auto issue = [&](size_t slot, size_t item) {
            const PackLayer& p = layers[wanted[item].first];
            uint64_t offset = p.start + (uint64_t)wanted[item].second * p.stride;
            slots[slot] = OVERLAPPED();
            slots[slot].Offset = (DWORD)(offset & 0xFFFFFFFF);
            slots[slot].OffsetHigh = (DWORD)(offset >> 32);
            slots[slot].hEvent = events[slot];
            ResetEvent(events[slot]);
            busy[slot] = (int)item;
            if(!ReadFile(file, targets[item], (DWORD)p.stride, nullptr, &slots[slot]) && GetLastError() != ERROR_IO_PENDING) ok = false;
        };

        for(size_t s = 0; s < slots.size() && next < wanted.size(); s++) issue(s, next++);
        size_t done = 0;
        while(done < wanted.size() && ok){
            for(size_t s = 0; s < slots.size(); s++){
                if(busy[s] < 0) continue;
                DWORD got = 0;
                if(!GetOverlappedResult(file, &slots[s], &got, TRUE)){
                    ok = false;
                    break;
                }
                done++;
                busy[s] = -1;
                if(next < wanted.size()) issue(s, next++);
            }
        }

        for(HANDLE e : events) CloseHandle(e);
        return ok;
    }
};

inline bool buildPack(const ModelFile& model, const std::string& path, std::string& problem, std::function<void(int, int)> progress) {
    std::vector<PackLayer> layers = planPack(model);
    if(layers.empty()){
        problem = "this model has no expert tensors to pack";
        return false;
    }

    std::string temporary = path + ".part";
    HANDLE out = CreateFileA(temporary.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH, nullptr);
    if(out == INVALID_HANDLE_VALUE){
        problem = "cannot create " + temporary;
        return false;
    }

    char* head = (char*)VirtualAlloc(nullptr, packAlign, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    PackHead info = {{'M', 'X', 'P', 'K'}, 1, model.layers, model.experts, 0, sourceStamp(model)};
    for(const TensorInfo& t : model.tensors) info.sourceBytes += t.bytes;
    memcpy(head, &info, sizeof(info));
    memcpy(head + sizeof(info), layers.data(), sizeof(PackLayer) * layers.size());
    DWORD wrote = 0;
    bool ok = WriteFile(out, head, (DWORD)packAlign, &wrote, nullptr) && wrote == packAlign;
    VirtualFree(head, 0, MEM_RELEASE);

    std::vector<HANDLE> sources;
    for(const std::string& p : model.paths) sources.push_back(CreateFileA(p.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, nullptr));

    auto readRange = [&](const TensorInfo* t, char* dst) {
        uint64_t offset = t->offset;
        uint64_t left = t->bytes;
        char* at = dst;
        while(left > 0){
            DWORD chunk = (DWORD)std::min<uint64_t>(left, 64ULL << 20);
            OVERLAPPED where = {};
            where.Offset = (DWORD)(offset & 0xFFFFFFFF);
            where.OffsetHigh = (DWORD)(offset >> 32);
            DWORD got = 0;
            if(!ReadFile(sources[t->file], at, chunk, &got, &where) || got != chunk) return false;
            offset += got;
            at += got;
            left -= got;
        }
        return true;
    };

    for(uint32_t l = 0; l < model.layers && ok; l++){
        const TensorInfo *gate, *up, *down;
        expertTensors(model, (int)l, gate, up, down);
        const PackLayer& p = layers[l];
        uint64_t blockBytes = p.stride * model.experts;
        char* block = (char*)VirtualAlloc(nullptr, (size_t)blockBytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        char* whole = (char*)VirtualAlloc(nullptr, (size_t)std::max({gate->bytes, up->bytes, down->bytes}), MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        if(!block || !whole){
            problem = "not enough memory to pack layer " + std::to_string(l);
            ok = false;
        }

        const TensorInfo* parts[3] = {gate, up, down};
        uint32_t sizes[3] = {p.gateBytes, p.upBytes, p.downBytes};
        uint32_t shifts[3] = {0, p.gateBytes, p.gateBytes + p.upBytes};
        for(int k = 0; k < 3 && ok; k++){
            if(!readRange(parts[k], whole)){
                problem = "cannot read " + parts[k]->name;
                ok = false;
                break;
            }
            for(uint32_t e = 0; e < model.experts; e++){
                memcpy(block + (uint64_t)e * p.stride + shifts[k], whole + (uint64_t)e * sizes[k], sizes[k]);
            }
        }

        uint64_t written = 0;
        while(ok && written < blockBytes){
            DWORD chunk = (DWORD)std::min<uint64_t>(blockBytes - written, 64ULL << 20);
            if(!WriteFile(out, block + written, chunk, &wrote, nullptr) || wrote != chunk){
                problem = "cannot write the expert pack";
                ok = false;
            }
            written += wrote;
        }

        if(block) VirtualFree(block, 0, MEM_RELEASE);
        if(whole) VirtualFree(whole, 0, MEM_RELEASE);
        if(progress) progress((int)l + 1, (int)model.layers);
    }

    for(HANDLE h : sources){
        if(h != INVALID_HANDLE_VALUE) CloseHandle(h);
    }
    CloseHandle(out);

    if(!ok){
        DeleteFileA(temporary.c_str());
        return false;
    }
    DeleteFileA(path.c_str());
    if(!MoveFileA(temporary.c_str(), path.c_str())){
        problem = "cannot finish " + path;
        return false;
    }
    return true;
}
