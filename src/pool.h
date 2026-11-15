#pragma once
#include "gguf.h"
#include "hardware.h"
#include <map>

enum SlotKind { SLOT_GPU = 0, SLOT_RAM = 1, SLOT_SSD = 2 };

const uint64_t gib = 1ULL << 30;
const uint64_t gpuScratch = 768ULL << 20;

struct PoolSlot {
    std::string node;
    int kind = SLOT_GPU;
    double gb = 0;
    double percent = 0;
    uint64_t limit = 0;
    uint64_t used = 0;
    double keep = 0;
};

struct LayerShare {
    int layer = 0;
    std::string node;
    uint64_t denseBytes = 0;
    uint64_t expertBytes = 0;
    int denseTier = SLOT_GPU;
    int gpuExperts = 0;
    int ramExperts = 0;
    int ssdExperts = 0;
};

struct NodeShare {
    std::string node;
    int firstLayer = -1;
    int lastLayer = -1;
    bool holdsEdges = false;
    int headTier = SLOT_GPU;
    int tableTier = SLOT_RAM;
    uint64_t gpuBytes = 0;
    uint64_t ramBytes = 0;
    uint64_t ssdBytes = 0;
};

struct ModelShape {
    uint64_t headBytes = 0;
    uint64_t tableBytes = 0;
    uint64_t diskBytes = 0;
    std::vector<uint64_t> dense;
    std::vector<uint64_t> experts;
    std::vector<uint64_t> kvPerToken;
    uint32_t expertCount = 0;
};

struct PoolPlan {
    bool ok = false;
    std::string problem;
    uint64_t demandBytes = 0;
    uint32_t context = 0;
    std::vector<PoolSlot> slots;
    std::vector<NodeShare> nodes;
    std::vector<LayerShare> layers;
};

inline ModelShape shapeOf(const ModelFile& model) {
    ModelShape shape;
    shape.expertCount = model.experts;
    shape.dense.assign(model.layers, 0);
    shape.experts.assign(model.layers, 0);
    shape.kvPerToken.assign(model.layers, 0);

    for(const TensorInfo& t : model.tensors){
        if(t.layer < 0 || t.layer >= (int)model.layers){
            if(t.name.rfind("output", 0) == 0) shape.headBytes += t.bytes;
            else if(t.name.rfind("per_layer_token_embd", 0) == 0) shape.diskBytes += t.bytes;
            else shape.tableBytes += t.bytes;
            continue;
        }
        if(t.expert) shape.experts[t.layer] += t.bytes;
        else shape.dense[t.layer] += t.bytes;
    }

    uint32_t interval = metaWhole(model, model.arch + ".full_attention_interval", 1);
    uint32_t kvHeads = metaWhole(model, model.arch + ".attention.head_count_kv", 0);
    uint32_t keyLength = metaWhole(model, model.arch + ".attention.key_length", 0);
    uint32_t valueLength = metaWhole(model, model.arch + ".attention.value_length", 0);
    for(uint32_t l = 0; l < model.layers; l++){
        bool fullAttention = interval <= 1 || (l + 1) % interval == 0;
        if(fullAttention) shape.kvPerToken[l] = (uint64_t)kvHeads * (keyLength + valueLength) * 2;
    }
    return shape;
}

inline uint64_t slotTotal(const PoolSlot& slot, const NodeStats& stats) {
    if(slot.kind == SLOT_GPU) return stats.gpu.present ? stats.gpu.vramTotal : 0;
    if(slot.kind == SLOT_RAM) return stats.host.ramTotal;
    return stats.host.diskFree;
}

inline PoolPlan planPool(const ModelShape& shape, std::vector<PoolSlot> slots, const std::map<std::string, NodeStats>& stats, const std::string& masterNode, uint32_t context) {
    PoolPlan plan;
    plan.context = context;
    size_t layerCount = shape.dense.size();

    for(PoolSlot& slot : slots){
        auto found = stats.find(slot.node);
        if(found == stats.end()){
            plan.problem = slot.node + " is not connected";
            slot.limit = 0;
            continue;
        }
        uint64_t total = slotTotal(slot, found->second);
        uint64_t keep = (uint64_t)(slot.keep * (double)gib);
        const NodeStats& seen = found->second;
        // lo libre de verdad mas lo que el motor ya tiene ahi: con otro programa usando la tarjeta de la laptop un slot fijo
        // de 7.4 GB no entraba en sus 4.5 GB libres y el nodo se negaba a cargar sus capas
        uint64_t reachable = slot.kind == SLOT_RAM
            ? std::min(total, seen.host.ramFree + seen.host.heldRam)
            : slot.kind == SLOT_GPU && seen.gpu.present && seen.gpu.vramFree > 0
                ? std::min(total, seen.gpu.vramFree + seen.host.heldGpu)
                : total;
        slot.limit = slot.keep > 0
            ? (reachable > keep ? reachable - keep : 0)
            : slot.gb > 0
                ? (uint64_t)(slot.gb * (double)gib)
                : slot.percent > 0
                    ? (uint64_t)(slot.percent / 100.0 * (double)total)
                    : reachable;
        if(slot.limit > total) slot.limit = total;
        if(slot.kind == SLOT_GPU && slot.limit > reachable) slot.limit = reachable;
    }

    std::vector<uint64_t> layerCost(layerCount);
    uint64_t demand = shape.headBytes + shape.tableBytes;
    for(size_t l = 0; l < layerCount; l++){
        layerCost[l] = shape.dense[l] + shape.experts[l] + shape.kvPerToken[l] * context;
        demand += layerCost[l];
    }

    std::vector<std::string> order = {masterNode};
    for(const PoolSlot& slot : slots){
        if(slot.limit == 0) continue;
        if(std::find(order.begin(), order.end(), slot.node) == order.end()) order.push_back(slot.node);
    }

    std::map<std::string, bool> scratchTaken;
    uint64_t remaining = demand;
    for(PoolSlot& slot : slots){
        uint64_t room = slot.limit;
        if(slot.kind == SLOT_GPU && room > 0 && !scratchTaken[slot.node]){
            scratchTaken[slot.node] = true;
            room = room > gpuScratch ? room - gpuScratch : 0;
        }
        slot.used = remaining < room ? remaining : room;
        remaining -= slot.used;
    }

    plan.demandBytes = demand;
    plan.slots = slots;

    if(remaining > 0){
        char text[160];
        snprintf(text, sizeof(text), "model does not fit, %.1f GB still missing: raise a limit or add an SSD slot", (double)remaining / (double)gib);
        plan.problem = text;
        return plan;
    }

    std::map<std::string, uint64_t> capacity;
    std::map<std::string, uint64_t> gpuRoom, ramRoom, ssdRoom;
    for(const PoolSlot& slot : slots){
        capacity[slot.node] += slot.used;
        if(slot.kind == SLOT_GPU) gpuRoom[slot.node] += slot.used;
        if(slot.kind == SLOT_RAM) ramRoom[slot.node] += slot.used;
        if(slot.kind == SLOT_SSD) ssdRoom[slot.node] += slot.used;
    }

    std::vector<std::string> active;
    for(const std::string& node : order){
        if(capacity[node] > 0 || node == masterNode) active.push_back(node);
    }

    uint64_t edges = shape.headBytes + shape.tableBytes;
    uint64_t passed = 0;
    uint64_t boundary = 0;
    size_t layer = 0;

    for(size_t n = 0; n < active.size(); n++){
        const std::string& node = active[n];
        NodeShare share;
        share.node = node;
        share.holdsEdges = node == masterNode;
        boundary += capacity[node];
        if(share.holdsEdges) passed += edges;

        bool last = n + 1 == active.size();
        while(layer < layerCount){
            uint64_t middle = passed + layerCost[layer] / 2;
            if(!last && middle > boundary) break;
            if(share.firstLayer < 0) share.firstLayer = (int)layer;
            share.lastLayer = (int)layer;
            passed += layerCost[layer];
            layer++;
        }

        uint64_t gpuLeft = gpuRoom[node];
        uint64_t ramLeft = ramRoom[node];
        uint64_t ssdLeft = ssdRoom[node];

        auto take = [&](uint64_t bytes) {
            if(bytes <= gpuLeft){
                gpuLeft -= bytes;
                share.gpuBytes += bytes;
                return (int)SLOT_GPU;
            }
            if(bytes <= ramLeft){
                ramLeft -= bytes;
                share.ramBytes += bytes;
                return (int)SLOT_RAM;
            }
            ssdLeft = ssdLeft > bytes ? ssdLeft - bytes : 0;
            share.ssdBytes += bytes;
            return (int)SLOT_SSD;
        };

        if(share.holdsEdges){
            share.headTier = take(shape.headBytes);
            if(shape.tableBytes <= ramLeft){
                ramLeft -= shape.tableBytes;
                share.ramBytes += shape.tableBytes;
                share.tableTier = SLOT_RAM;
            } else share.tableTier = take(shape.tableBytes);
        }

        std::vector<LayerShare> mine;
        for(int l = share.firstLayer; l >= 0 && l <= share.lastLayer; l++){
            LayerShare item;
            item.layer = l;
            item.node = node;
            item.denseBytes = shape.dense[l] + shape.kvPerToken[l] * context;
            item.expertBytes = shape.experts[l];
            item.denseTier = take(item.denseBytes);
            mine.push_back(item);
        }

        uint64_t* tiers[3] = {&gpuLeft, &ramLeft, &ssdLeft};
        uint64_t* totals[3] = {&share.gpuBytes, &share.ramBytes, &share.ssdBytes};
        for(int tier = 0; tier < 3; tier++){
            if(mine.empty()) break;
            for(size_t k = 0; k < mine.size(); k++){
                LayerShare& item = mine[k];
                int placed = item.gpuExperts + item.ramExperts + item.ssdExperts;
                int open = (int)shape.expertCount - placed;
                if(open <= 0 || shape.expertCount == 0) continue;

                uint64_t each = item.expertBytes / shape.expertCount;
                uint64_t fair = *tiers[tier] / (mine.size() - k);
                int count = tier == SLOT_SSD ? open : (int)(each ? fair / each : open);
                if(count > open) count = open;

                uint64_t bytes = (uint64_t)count * each;
                *tiers[tier] = *tiers[tier] > bytes ? *tiers[tier] - bytes : 0;
                *totals[tier] += bytes;
                if(tier == SLOT_GPU) item.gpuExperts = count;
                if(tier == SLOT_RAM) item.ramExperts = count;
                if(tier == SLOT_SSD) item.ssdExperts = count;
            }
        }

        for(const LayerShare& item : mine) plan.layers.push_back(item);
        plan.nodes.push_back(share);
    }

    plan.ok = plan.problem.empty();
    return plan;
}
