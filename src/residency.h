#pragma once
#include "cuda.h"
#include "pool.h"
#include <atomic>
#include <map>

const uint64_t ramFloor = 2ULL << 30;

struct Piece {
    std::string tensor;
    int layer = -1;
    int tier = SLOT_SSD;
    int firstExpert = 0;
    int expertCount = 0;
    uint64_t offset = 0;
    uint64_t bytes = 0;
    uint64_t gpu = 0;
    char* ram = nullptr;
};

struct TensorSource {
    std::string name;
    std::string path;
    uint64_t offset = 0;
    uint64_t bytes = 0;
    int layer = -1;
    bool expert = false;
};

struct Residency {
    std::mutex lock;
    std::vector<Piece> pieces;
    std::string state = "empty";
    std::string problem;
    uint64_t total = 0;
    uint64_t done = 0;
    uint64_t gpuBytes = 0;
    uint64_t ramBytes = 0;
    std::atomic<uint64_t> generation{0};
};

inline void releaseAll(Residency& store) {
    std::lock_guard<std::mutex> guard(store.lock);
    bool cuda = useCuda();
    for(Piece& p : store.pieces){
        if(p.gpu && cuda) cudaLink().release(p.gpu);
        if(p.ram) VirtualFree(p.ram, 0, MEM_RELEASE);
        p.gpu = 0;
        p.ram = nullptr;
    }
    store.pieces.clear();
    store.state = "empty";
    store.problem.clear();
    store.total = store.done = store.gpuBytes = store.ramBytes = 0;
}

inline bool readRange(HANDLE file, uint64_t offset, char* out, uint64_t bytes) {
    while(bytes > 0){
        DWORD chunk = bytes > (64ULL << 20) ? (DWORD)(64ULL << 20) : (DWORD)bytes;
        OVERLAPPED at = {};
        at.Offset = (DWORD)(offset & 0xFFFFFFFF);
        at.OffsetHigh = (DWORD)(offset >> 32);
        DWORD got = 0;
        if(!ReadFile(file, out, chunk, &got, &at) || got != chunk) return false;
        offset += got;
        out += got;
        bytes -= got;
    }
    return true;
}

inline uint64_t ramAvailable() {
    MEMORYSTATUSEX memory = {};
    memory.dwLength = sizeof(memory);
    GlobalMemoryStatusEx(&memory);
    return memory.ullAvailPhys;
}

inline std::vector<Piece> cutPieces(const std::vector<TensorSource>& sources, const std::vector<LayerShare>& layers, bool holdsEdges, int headTier, int tableTier, uint32_t expertCount) {
    std::map<int, LayerShare> byLayer;
    for(const LayerShare& l : layers) byLayer[l.layer] = l;

    std::vector<Piece> pieces;
    for(const TensorSource& s : sources){
        if(s.layer < 0){
            if(!holdsEdges) continue;
            Piece p;
            p.tensor = s.name;
            p.offset = s.offset;
            p.bytes = s.bytes;
            p.tier = s.name.rfind("output", 0) == 0
                ? headTier
                : s.name.rfind("per_layer_token_embd", 0) == 0
                    ? SLOT_SSD
                    : tableTier;
            pieces.push_back(p);
            continue;
        }

        auto found = byLayer.find(s.layer);
        if(found == byLayer.end()) continue;
        const LayerShare& share = found->second;

        if(!s.expert || expertCount == 0){
            Piece p;
            p.tensor = s.name;
            p.layer = s.layer;
            p.offset = s.offset;
            p.bytes = s.bytes;
            p.tier = share.denseTier;
            pieces.push_back(p);
            continue;
        }

        uint64_t each = s.bytes / expertCount;
        int counts[3] = {share.gpuExperts, share.ramExperts, (int)expertCount - share.gpuExperts - share.ramExperts};
        int first = 0;
        for(int tier = 0; tier < 3; tier++){
            if(counts[tier] <= 0) continue;
            Piece p;
            p.tensor = s.name;
            p.layer = s.layer;
            p.tier = tier;
            p.firstExpert = first;
            p.expertCount = counts[tier];
            p.offset = s.offset + (uint64_t)first * each;
            p.bytes = (uint64_t)counts[tier] * each;
            pieces.push_back(p);
            first += counts[tier];
        }
    }
    return pieces;
}

inline void loadShare(Residency& store, std::vector<TensorSource> sources, std::vector<LayerShare> layers, bool holdsEdges, int headTier, int tableTier, uint32_t expertCount) {
    releaseAll(store);
    uint64_t ticket = ++store.generation;

    std::vector<Piece> pieces = cutPieces(sources, layers, holdsEdges, headTier, tableTier, expertCount);
    std::map<std::string, std::pair<std::string, uint64_t>> where;
    for(const TensorSource& s : sources) where[s.name] = {s.path, s.offset};

    {
        std::lock_guard<std::mutex> guard(store.lock);
        store.state = "loading";
        for(const Piece& p : pieces){
            if(p.tier != SLOT_SSD) store.total += p.bytes;
        }
    }

    std::map<std::string, HANDLE> files;
    auto fileFor = [&](const std::string& path) {
        auto found = files.find(path);
        if(found != files.end()) return found->second;
        HANDLE h = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
        files[path] = h;
        return h;
    };

    bool cuda = useCuda();
    char* staging = nullptr;
    const uint64_t stagingSize = 64ULL << 20;
    if(cuda && cudaLink().hostAlloc && cudaLink().hostAlloc((void**)&staging, stagingSize, 0) != 0) staging = nullptr;
    bool pinned = staging != nullptr;
    if(!staging) staging = (char*)VirtualAlloc(nullptr, stagingSize, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);

    uint64_t demotedGpu = 0, skippedRam = 0;
    std::string failure;

    for(int pass = 0; pass < 2 && failure.empty(); pass++){
        int wanted = pass == 0 ? SLOT_GPU : SLOT_RAM;
        for(Piece& p : pieces){
            if(store.generation != ticket){
                failure = "replaced by a newer plan";
                break;
            }
            if(p.tier != wanted) continue;

            HANDLE file = fileFor(where[p.tensor].first);
            if(file == INVALID_HANDLE_VALUE){
                failure = "cannot open " + where[p.tensor].first;
                break;
            }

            if(p.tier == SLOT_GPU){
                if(!cuda || cudaLink().alloc(&p.gpu, (size_t)p.bytes) != 0){
                    p.gpu = 0;
                    p.tier = SLOT_RAM;
                    demotedGpu += p.bytes;
                    continue;
                }
                uint64_t moved = 0;
                while(moved < p.bytes){
                    uint64_t size = p.bytes - moved < stagingSize ? p.bytes - moved : stagingSize;
                    if(!readRange(file, p.offset + moved, staging, size) || cudaLink().copyUp(p.gpu + moved, staging, (size_t)size) != 0){
                        failure = "read or upload failed on " + p.tensor;
                        break;
                    }
                    moved += size;
                    std::lock_guard<std::mutex> guard(store.lock);
                    store.done += size;
                }
                std::lock_guard<std::mutex> guard(store.lock);
                store.gpuBytes += p.bytes;
                continue;
            }

            if(ramAvailable() < p.bytes + ramFloor){
                p.tier = SLOT_SSD;
                skippedRam += p.bytes;
                continue;
            }

            p.ram = (char*)VirtualAlloc(nullptr, (size_t)p.bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
            if(!p.ram || !readRange(file, p.offset, p.ram, p.bytes)){
                if(p.ram) VirtualFree(p.ram, 0, MEM_RELEASE);
                p.ram = nullptr;
                p.tier = SLOT_SSD;
                skippedRam += p.bytes;
                continue;
            }
            std::lock_guard<std::mutex> guard(store.lock);
            store.done += p.bytes;
            store.ramBytes += p.bytes;
        }
    }

    for(auto& entry : files){
        if(entry.second != INVALID_HANDLE_VALUE) CloseHandle(entry.second);
    }
    if(pinned) cudaLink().hostFree(staging);
    else if(staging) VirtualFree(staging, 0, MEM_RELEASE);

    std::lock_guard<std::mutex> guard(store.lock);
    store.pieces = std::move(pieces);
    if(!failure.empty()){
        store.state = "failed";
        store.problem = failure;
        return;
    }

    store.state = "ready";
    store.total = store.gpuBytes + store.ramBytes;
    store.done = store.total;
    char text[160] = {};
    if(demotedGpu){
        snprintf(text, sizeof(text), "%.1f GB did not fit in VRAM, another program is using it", (double)demotedGpu / gib);
        store.problem = text;
    }
    if(skippedRam){
        snprintf(text, sizeof(text), "%.1f GB stayed on SSD to keep 2 GB of RAM free", (double)skippedRam / gib);
        store.problem += (store.problem.empty() ? "" : " | ") + std::string(text);
    }
}

inline void residencyStats(Residency& store, HostStats& host) {
    std::lock_guard<std::mutex> guard(store.lock);
    host.heldGpu = store.gpuBytes;
    host.heldRam = store.ramBytes;
    host.loadTotal = store.total;
    host.loadDone = store.done;
}
