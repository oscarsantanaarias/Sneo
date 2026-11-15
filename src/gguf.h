#pragma once
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <algorithm>

struct TypeInfo {
    uint32_t blockElements;
    uint32_t blockBytes;
    const char* name;
};

inline TypeInfo typeInfo(uint32_t type) {
    switch(type){
        case 0: return {1, 4, "F32"};
        case 1: return {1, 2, "F16"};
        case 2: return {32, 18, "Q4_0"};
        case 3: return {32, 20, "Q4_1"};
        case 6: return {32, 22, "Q5_0"};
        case 7: return {32, 24, "Q5_1"};
        case 8: return {32, 34, "Q8_0"};
        case 10: return {256, 84, "Q2_K"};
        case 11: return {256, 110, "Q3_K"};
        case 12: return {256, 144, "Q4_K"};
        case 13: return {256, 176, "Q5_K"};
        case 14: return {256, 210, "Q6_K"};
        case 16: return {256, 66, "IQ2_XXS"};
        case 17: return {256, 74, "IQ2_XS"};
        case 18: return {256, 98, "IQ3_XXS"};
        case 19: return {256, 50, "IQ1_S"};
        case 20: return {32, 18, "IQ4_NL"};
        case 21: return {256, 110, "IQ3_S"};
        case 22: return {256, 82, "IQ2_S"};
        case 23: return {256, 136, "IQ4_XS"};
        case 24: return {1, 1, "I8"};
        case 25: return {1, 2, "I16"};
        case 26: return {1, 4, "I32"};
        case 27: return {1, 8, "I64"};
        case 28: return {1, 8, "F64"};
        case 29: return {256, 56, "IQ1_M"};
        case 30: return {1, 2, "BF16"};
        case 39: return {32, 17, "MXFP4"};
        case 42: return {64, 18, "Q2_0"};
        default: return {0, 0, "UNKNOWN"};
    }
}

struct TensorInfo {
    std::string name;
    uint32_t type = 0;
    uint32_t dimCount = 0;
    uint64_t dims[4] = {1, 1, 1, 1};
    uint64_t elements = 0;
    uint64_t bytes = 0;
    uint32_t file = 0;
    uint64_t offset = 0;
    int layer = -1;
    bool expert = false;
};

struct MetaValue {
    uint32_t type = 0;
    int64_t whole = 0;
    double real = 0;
    std::string text;
    std::vector<int64_t> wholes;
    std::vector<double> reals;
    std::vector<std::string> texts;
};

struct ModelFile {
    std::vector<std::string> paths;
    std::map<std::string, MetaValue> meta;
    std::vector<TensorInfo> tensors;
    std::map<std::string, size_t> lookup;
    uint64_t totalBytes = 0;
    std::string arch;
    std::string name;
    uint32_t layers = 0;
    uint32_t experts = 0;
    uint32_t expertsUsed = 0;
    uint32_t hidden = 0;
    uint32_t contextLength = 0;
};

inline bool readRaw(FILE* f, void* out, size_t size) {
    return fread(out, 1, size, f) == size;
}

inline bool readGgufText(FILE* f, std::string& out) {
    uint64_t length = 0;
    if(!readRaw(f, &length, 8) || length > (1ULL << 30)) return false;
    out.resize((size_t)length);
    if(length == 0) return true;
    return readRaw(f, &out[0], (size_t)length);
}

inline bool readScalar(FILE* f, uint32_t type, int64_t& whole, double& real) {
    switch(type){
        case 0: { uint8_t v; if(!readRaw(f, &v, 1)) return false; whole = v; real = v; return true; }
        case 1: { int8_t v; if(!readRaw(f, &v, 1)) return false; whole = v; real = v; return true; }
        case 2: { uint16_t v; if(!readRaw(f, &v, 2)) return false; whole = v; real = v; return true; }
        case 3: { int16_t v; if(!readRaw(f, &v, 2)) return false; whole = v; real = v; return true; }
        case 4: { uint32_t v; if(!readRaw(f, &v, 4)) return false; whole = v; real = v; return true; }
        case 5: { int32_t v; if(!readRaw(f, &v, 4)) return false; whole = v; real = v; return true; }
        case 6: { float v; if(!readRaw(f, &v, 4)) return false; whole = (int64_t)v; real = v; return true; }
        case 7: { uint8_t v; if(!readRaw(f, &v, 1)) return false; whole = v; real = v; return true; }
        case 10: { uint64_t v; if(!readRaw(f, &v, 8)) return false; whole = (int64_t)v; real = (double)v; return true; }
        case 11: { int64_t v; if(!readRaw(f, &v, 8)) return false; whole = v; real = (double)v; return true; }
        case 12: { double v; if(!readRaw(f, &v, 8)) return false; whole = (int64_t)v; real = v; return true; }
    }
    return false;
}

inline bool readMetaValue(FILE* f, uint32_t type, MetaValue& value) {
    value.type = type;
    if(type == 8) return readGgufText(f, value.text);
    if(type != 9) return readScalar(f, type, value.whole, value.real);

    uint32_t itemType = 0;
    uint64_t count = 0;
    if(!readRaw(f, &itemType, 4) || !readRaw(f, &count, 8)) return false;
    value.whole = itemType;

    if(itemType == 8){
        value.texts.resize((size_t)count);
        for(uint64_t i = 0; i < count; i++){
            if(!readGgufText(f, value.texts[(size_t)i])) return false;
        }
        return true;
    }

    if(itemType == 9) return false;

    value.wholes.resize((size_t)count);
    value.reals.resize((size_t)count);
    for(uint64_t i = 0; i < count; i++){
        if(!readScalar(f, itemType, value.wholes[(size_t)i], value.reals[(size_t)i])) return false;
    }
    return true;
}

inline std::string splitPath(const std::string& first, int index, int count) {
    char ending[64];
    snprintf(ending, sizeof(ending), "-%05d-of-%05d.gguf", 1, count);
    size_t at = first.rfind(ending);
    if(at == std::string::npos) return "";
    char wanted[64];
    snprintf(wanted, sizeof(wanted), "-%05d-of-%05d.gguf", index, count);
    return first.substr(0, at) + wanted;
}

inline bool readGgufFile(const std::string& path, uint32_t fileIndex, ModelFile& model, std::string& error) {
    FILE* f = nullptr;
    fopen_s(&f, path.c_str(), "rb");
    if(!f){
        error = "cannot open " + path;
        return false;
    }
    setvbuf(f, nullptr, _IOFBF, 1 << 20);

    uint32_t magic = 0, version = 0;
    uint64_t tensorCount = 0, metaCount = 0;
    if(!readRaw(f, &magic, 4) || magic != 0x46554747 || !readRaw(f, &version, 4) || !readRaw(f, &tensorCount, 8) || !readRaw(f, &metaCount, 8)){
        fclose(f);
        error = "not a gguf file: " + path;
        return false;
    }

    uint64_t alignment = 32;
    for(uint64_t i = 0; i < metaCount; i++){
        std::string key;
        uint32_t type = 0;
        MetaValue value;
        if(!readGgufText(f, key) || !readRaw(f, &type, 4) || !readMetaValue(f, type, value)){
            fclose(f);
            error = "broken metadata in " + path;
            return false;
        }
        if(key == "general.alignment") alignment = (uint64_t)value.whole;
        if(fileIndex == 0) model.meta[key] = std::move(value);
    }

    size_t firstTensor = model.tensors.size();
    for(uint64_t i = 0; i < tensorCount; i++){
        TensorInfo t;
        if(!readGgufText(f, t.name) || !readRaw(f, &t.dimCount, 4) || t.dimCount > 4){
            fclose(f);
            error = "broken tensor table in " + path;
            return false;
        }
        t.elements = 1;
        for(uint32_t d = 0; d < t.dimCount; d++){
            if(!readRaw(f, &t.dims[d], 8)) break;
            t.elements *= t.dims[d];
        }
        uint64_t relative = 0;
        if(!readRaw(f, &t.type, 4) || !readRaw(f, &relative, 8)){
            fclose(f);
            error = "broken tensor table in " + path;
            return false;
        }
        t.file = fileIndex;
        t.offset = relative;

        size_t block = t.name.find("blk.");
        if(block != std::string::npos) t.layer = atoi(t.name.c_str() + block + 4);
        t.expert = t.name.find("_exps") != std::string::npos;
        model.tensors.push_back(t);
    }

    uint64_t headerEnd = (uint64_t)_ftelli64(f);
    uint64_t dataStart = (headerEnd + alignment - 1) / alignment * alignment;
    _fseeki64(f, 0, SEEK_END);
    uint64_t fileEnd = (uint64_t)_ftelli64(f);
    fclose(f);

    std::vector<size_t> order;
    for(size_t i = firstTensor; i < model.tensors.size(); i++){
        model.tensors[i].offset += dataStart;
        order.push_back(i);
    }
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b){ return model.tensors[a].offset < model.tensors[b].offset; });

    for(size_t k = 0; k < order.size(); k++){
        TensorInfo& t = model.tensors[order[k]];
        uint64_t next = k + 1 < order.size() ? model.tensors[order[k + 1]].offset : fileEnd;
        TypeInfo info = typeInfo(t.type);
        t.bytes = info.blockElements ? t.elements / info.blockElements * info.blockBytes : next - t.offset;
    }

    model.paths.push_back(path);
    return true;
}

inline uint32_t metaWhole(const ModelFile& model, const std::string& key, uint32_t fallback) {
    auto it = model.meta.find(key);
    return it == model.meta.end() ? fallback : (uint32_t)it->second.whole;
}

inline bool openModel(const std::string& firstPath, ModelFile& model, std::string& error) {
    model = ModelFile();
    if(!readGgufFile(firstPath, 0, model, error)) return false;

    uint32_t splitCount = metaWhole(model, "split.count", 1);
    for(uint32_t i = 2; i <= splitCount; i++){
        std::string next = splitPath(firstPath, (int)i, (int)splitCount);
        if(next.empty() || !readGgufFile(next, i - 1, model, error)){
            if(error.empty()) error = "missing split file " + std::to_string(i) + " of " + std::to_string(splitCount);
            return false;
        }
    }

    for(size_t i = 0; i < model.tensors.size(); i++){
        model.lookup[model.tensors[i].name] = i;
        model.totalBytes += model.tensors[i].bytes;
    }

    auto general = model.meta.find("general.architecture");
    model.arch = general == model.meta.end() ? "" : general->second.text;
    auto named = model.meta.find("general.name");
    model.name = named == model.meta.end() ? "" : named->second.text;
    model.layers = metaWhole(model, model.arch + ".block_count", 0) - metaWhole(model, model.arch + ".nextn_predict_layers", 0);
    model.experts = metaWhole(model, model.arch + ".expert_count", 0);
    model.expertsUsed = metaWhole(model, model.arch + ".expert_used_count", 0);
    model.hidden = metaWhole(model, model.arch + ".embedding_length", 0);
    model.contextLength = metaWhole(model, model.arch + ".context_length", 0);
    return true;
}

inline std::string modelIdOf(const ModelFile& model) {
    uint64_t h = 1469598103934665603ULL;
    for(unsigned char c : model.name + "|" + std::to_string(model.totalBytes)){
        h ^= c;
        h *= 1099511628211ULL;
    }
    char out[24];
    snprintf(out, sizeof(out), "%016llx", (unsigned long long)h);
    return out;
}

inline const TensorInfo* findTensor(const ModelFile& model, const std::string& name) {
    auto it = model.lookup.find(name);
    return it == model.lookup.end() ? nullptr : &model.tensors[it->second];
}
