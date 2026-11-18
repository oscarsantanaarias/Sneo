#pragma once
#include "json.h"
#include "platform.h"
#include <string>
#include <vector>
#include <map>

// lector de safetensors: cabecera json con tipo, forma y tramo de bytes de cada tensor
struct SafeTensor {
    std::string dtype;
    std::vector<uint64_t> shape;
    uint64_t begin = 0;
    uint64_t end = 0;
    const uint8_t* data = nullptr;
};

struct SafeFile {
    HANDLE file = INVALID_HANDLE_VALUE;
    HANDLE mapping = nullptr;
    const uint8_t* base = nullptr;
    uint64_t size = 0;
    std::map<std::string, SafeTensor> tensors;

    bool open(const std::string& path, std::string& error) {
        file = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
        if(file == INVALID_HANDLE_VALUE){
            error = "cannot open " + path;
            return false;
        }
        LARGE_INTEGER length = {};
        GetFileSizeEx(file, &length);
        size = (uint64_t)length.QuadPart;
        mapping = CreateFileMappingA(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if(mapping) base = (const uint8_t*)MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
        if(!base || size < 8){
            error = "cannot map " + path;
            return false;
        }
        uint64_t headerBytes = 0;
        memcpy(&headerBytes, base, 8);
        if(headerBytes == 0 || 8 + headerBytes > size){
            error = path + " has a broken safetensors header";
            return false;
        }
        Json header = parseJson(std::string((const char*)base + 8, (size_t)headerBytes));
        const uint8_t* data = base + 8 + headerBytes;
        for(const auto& field : header.fields){
            if(field.first == "__metadata__") continue;
            const Json& t = field.second;
            SafeTensor info;
            info.dtype = t["dtype"].str();
            for(const Json& d : t["shape"].items) info.shape.push_back((uint64_t)d.num());
            if(t["data_offsets"].items.size() != 2) continue;
            info.begin = (uint64_t)t["data_offsets"].items[0].num();
            info.end = (uint64_t)t["data_offsets"].items[1].num();
            if(8 + headerBytes + info.end > size){
                error = field.first + " points past the end of " + path;
                return false;
            }
            info.data = data + info.begin;
            tensors[field.first] = info;
        }
        return true;
    }

    const SafeTensor* find(const std::string& name) const {
        auto it = tensors.find(name);
        return it == tensors.end() ? nullptr : &it->second;
    }

    void close() {
        if(base) UnmapViewOfFile(base);
        if(mapping) CloseHandle(mapping);
        if(file != INVALID_HANDLE_VALUE) CloseHandle(file);
        base = nullptr;
        mapping = nullptr;
        file = INVALID_HANDLE_VALUE;
    }
};
