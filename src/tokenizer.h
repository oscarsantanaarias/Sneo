#pragma once
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "platform.h"
#include <filesystem>
#include "gguf.h"
#include "json.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <queue>
#include <mutex>

enum : uint8_t {
    KIND_LETTER = 1,
    KIND_MARK = 2,
    KIND_NUMBER = 4,
    KIND_SPACE = 8
};

const uint32_t recentCodepoints[][3] = {
    {0x0870, 0x0887, KIND_LETTER}, {0x0889, 0x088E, KIND_LETTER}, {0x0897, 0x089F, KIND_MARK}, {0x08B5, 0x08B5, KIND_LETTER},
    {0x08BE, 0x08C9, KIND_LETTER}, {0x08CA, 0x08D2, KIND_MARK}, {0x0B55, 0x0B55, KIND_MARK}, {0x0C3C, 0x0C3C, KIND_MARK},
    {0x0C5D, 0x0C5D, KIND_LETTER}, {0x0CDD, 0x0CDD, KIND_LETTER}, {0x0CF3, 0x0CF3, KIND_MARK}, {0x0D04, 0x0D04, KIND_LETTER},
    {0x0D81, 0x0D81, KIND_MARK}, {0x0ECE, 0x0ECE, KIND_MARK}, {0x170D, 0x170D, KIND_LETTER}, {0x1715, 0x1715, KIND_MARK},
    {0x171F, 0x171F, KIND_LETTER}, {0x180F, 0x180F, KIND_MARK}, {0x1ABF, 0x1ACE, KIND_MARK}, {0x1B4C, 0x1B4C, KIND_LETTER},
    {0x1C89, 0x1C8A, KIND_LETTER}, {0x1DFA, 0x1DFA, KIND_MARK}, {0x2C2F, 0x2C2F, KIND_LETTER}, {0x2C5F, 0x2C5F, KIND_LETTER},
    {0x31BB, 0x31BF, KIND_LETTER}, {0x4DB6, 0x4DBF, KIND_LETTER}, {0x9FF0, 0x9FFF, KIND_LETTER}, {0xA7C0, 0xA7C1, KIND_LETTER},
    {0xA7C7, 0xA7CD, KIND_LETTER}, {0xA7D0, 0xA7D1, KIND_LETTER}, {0xA7D3, 0xA7D3, KIND_LETTER}, {0xA7D5, 0xA7DC, KIND_LETTER},
    {0xA7F2, 0xA7F6, KIND_LETTER}, {0xA82C, 0xA82C, KIND_MARK}, {0xAB68, 0xAB69, KIND_LETTER}, {0x10570, 0x1057A, KIND_LETTER},
    {0x1057C, 0x1058A, KIND_LETTER}, {0x1058C, 0x10592, KIND_LETTER}, {0x10594, 0x10595, KIND_LETTER}, {0x10597, 0x105A1, KIND_LETTER},
    {0x105A3, 0x105B1, KIND_LETTER}, {0x105B3, 0x105B9, KIND_LETTER}, {0x105BB, 0x105BC, KIND_LETTER}, {0x105C0, 0x105F3, KIND_LETTER},
    {0x10780, 0x10785, KIND_LETTER}, {0x10787, 0x107B0, KIND_LETTER}, {0x107B2, 0x107BA, KIND_LETTER}, {0x10D40, 0x10D49, KIND_NUMBER},
    {0x10D4A, 0x10D65, KIND_LETTER}, {0x10D69, 0x10D6D, KIND_MARK}, {0x10D6F, 0x10D85, KIND_LETTER}, {0x10E80, 0x10EA9, KIND_LETTER},
    {0x10EAB, 0x10EAC, KIND_MARK}, {0x10EB0, 0x10EB1, KIND_LETTER}, {0x10EC2, 0x10EC4, KIND_LETTER}, {0x10EFC, 0x10EFF, KIND_MARK},
    {0x10F70, 0x10F81, KIND_LETTER}, {0x10F82, 0x10F85, KIND_MARK}, {0x10FB0, 0x10FC4, KIND_LETTER}, {0x10FC5, 0x10FCB, KIND_NUMBER},
    {0x11070, 0x11070, KIND_MARK}, {0x11071, 0x11072, KIND_LETTER}, {0x11073, 0x11074, KIND_MARK}, {0x11075, 0x11075, KIND_LETTER},
    {0x110C2, 0x110C2, KIND_MARK}, {0x11147, 0x11147, KIND_LETTER}, {0x111CE, 0x111CF, KIND_MARK}, {0x1123F, 0x11240, KIND_LETTER},
    {0x11241, 0x11241, KIND_MARK}, {0x11380, 0x11389, KIND_LETTER}, {0x1138B, 0x1138B, KIND_LETTER}, {0x1138E, 0x1138E, KIND_LETTER},
    {0x11390, 0x113B5, KIND_LETTER}, {0x113B7, 0x113B7, KIND_LETTER}, {0x113B8, 0x113C0, KIND_MARK}, {0x113C2, 0x113C2, KIND_MARK},
    {0x113C5, 0x113C5, KIND_MARK}, {0x113C7, 0x113CA, KIND_MARK}, {0x113CC, 0x113D0, KIND_MARK}, {0x113D1, 0x113D1, KIND_LETTER},
    {0x113D2, 0x113D2, KIND_MARK}, {0x113D3, 0x113D3, KIND_LETTER}, {0x113E1, 0x113E2, KIND_MARK}, {0x11460, 0x11461, KIND_LETTER},
    {0x116D0, 0x116E3, KIND_NUMBER}, {0x11740, 0x11746, KIND_LETTER}, {0x11900, 0x11906, KIND_LETTER}, {0x11909, 0x11909, KIND_LETTER},
    {0x1190C, 0x11913, KIND_LETTER}, {0x11915, 0x11916, KIND_LETTER}, {0x11918, 0x1192F, KIND_LETTER}, {0x11930, 0x11935, KIND_MARK},
    {0x11937, 0x11938, KIND_MARK}, {0x1193B, 0x1193E, KIND_MARK}, {0x1193F, 0x1193F, KIND_LETTER}, {0x11940, 0x11940, KIND_MARK},
    {0x11941, 0x11941, KIND_LETTER}, {0x11942, 0x11943, KIND_MARK}, {0x11950, 0x11959, KIND_NUMBER}, {0x11AB0, 0x11ABF, KIND_LETTER},
    {0x11BC0, 0x11BE0, KIND_LETTER}, {0x11BF0, 0x11BF9, KIND_NUMBER}, {0x11F00, 0x11F01, KIND_MARK}, {0x11F02, 0x11F02, KIND_LETTER},
    {0x11F03, 0x11F03, KIND_MARK}, {0x11F04, 0x11F10, KIND_LETTER}, {0x11F12, 0x11F33, KIND_LETTER}, {0x11F34, 0x11F3A, KIND_MARK},
    {0x11F3E, 0x11F42, KIND_MARK}, {0x11F50, 0x11F59, KIND_NUMBER}, {0x11F5A, 0x11F5A, KIND_MARK}, {0x11FB0, 0x11FB0, KIND_LETTER},
    {0x12F90, 0x12FF0, KIND_LETTER}, {0x1342F, 0x1342F, KIND_LETTER}, {0x13440, 0x13440, KIND_MARK}, {0x13441, 0x13446, KIND_LETTER},
    {0x13447, 0x13455, KIND_MARK}, {0x13460, 0x143FA, KIND_LETTER}, {0x16100, 0x1611D, KIND_LETTER}, {0x1611E, 0x1612F, KIND_MARK},
    {0x16130, 0x16139, KIND_NUMBER}, {0x16A70, 0x16ABE, KIND_LETTER}, {0x16AC0, 0x16AC9, KIND_NUMBER}, {0x16D40, 0x16D6C, KIND_LETTER},
    {0x16D70, 0x16D79, KIND_NUMBER}, {0x16FE4, 0x16FE4, KIND_MARK}, {0x16FF0, 0x16FF1, KIND_MARK}, {0x18AF3, 0x18CD5, KIND_LETTER},
    {0x18CFF, 0x18D08, KIND_LETTER}, {0x1AFF0, 0x1AFF3, KIND_LETTER}, {0x1AFF5, 0x1AFFB, KIND_LETTER}, {0x1AFFD, 0x1AFFE, KIND_LETTER},
    {0x1B11F, 0x1B122, KIND_LETTER}, {0x1B132, 0x1B132, KIND_LETTER}, {0x1B155, 0x1B155, KIND_LETTER}, {0x1CCF0, 0x1CCF9, KIND_NUMBER},
    {0x1CF00, 0x1CF2D, KIND_MARK}, {0x1CF30, 0x1CF46, KIND_MARK}, {0x1D2C0, 0x1D2D3, KIND_NUMBER}, {0x1DF00, 0x1DF1E, KIND_LETTER},
    {0x1DF25, 0x1DF2A, KIND_LETTER}, {0x1E030, 0x1E06D, KIND_LETTER}, {0x1E08F, 0x1E08F, KIND_MARK}, {0x1E290, 0x1E2AD, KIND_LETTER},
    {0x1E2AE, 0x1E2AE, KIND_MARK}, {0x1E4D0, 0x1E4EB, KIND_LETTER}, {0x1E4EC, 0x1E4EF, KIND_MARK}, {0x1E4F0, 0x1E4F9, KIND_NUMBER},
    {0x1E5D0, 0x1E5ED, KIND_LETTER}, {0x1E5EE, 0x1E5EF, KIND_MARK}, {0x1E5F0, 0x1E5F0, KIND_LETTER}, {0x1E5F1, 0x1E5FA, KIND_NUMBER},
    {0x1E7E0, 0x1E7E6, KIND_LETTER}, {0x1E7E8, 0x1E7EB, KIND_LETTER}, {0x1E7ED, 0x1E7EE, KIND_LETTER}, {0x1E7F0, 0x1E7FE, KIND_LETTER},
    {0x1FBF0, 0x1FBF9, KIND_NUMBER}, {0x2A6D7, 0x2A6DF, KIND_LETTER}, {0x2B735, 0x2B739, KIND_LETTER}, {0x2EBF0, 0x2EE5D, KIND_LETTER},
    {0x30000, 0x3134A, KIND_LETTER}, {0x31350, 0x323AF, KIND_LETTER}
};

inline uint32_t readCodepoint(const std::string& s, size_t& i) {
    uint8_t c = (uint8_t)s[i];
    size_t need =
        c < 0x80 ? 0 :
        c >= 0xC2 && c <= 0xDF ? 1 :
        c >= 0xE0 && c <= 0xEF ? 2 :
        c >= 0xF0 && c <= 0xF4 ? 3 :
        4;

    if(need == 0){
        i++;
        return c;
    }

    if(need == 4 || i + need >= s.size()){
        i++;
        return 0xFFFD;
    }

    uint32_t code = c & (0x3F >> need);
    for(size_t k = 1; k <= need; k++){
        uint8_t d = (uint8_t)s[i + k];
        if((d & 0xC0) != 0x80){
            i++;
            return 0xFFFD;
        }
        code = (code << 6) | (d & 0x3F);
    }

    if((need == 2 && (code < 0x800 || (code >= 0xD800 && code <= 0xDFFF))) || (need == 3 && (code < 0x10000 || code > 0x10FFFF))){
        i++;
        return 0xFFFD;
    }

    i += need + 1;
    return code;
}

struct TextStream {
    std::string pending;

    std::string push(const std::string& bytes) {
        pending += bytes;
        std::string out;
        size_t n = pending.size();
        size_t i = 0;

        while(i < n){
            uint8_t c = (uint8_t)pending[i];
            if(c < 0x80){
                out.push_back((char)c);
                i++;
                continue;
            }

            size_t need =
                c >= 0xC2 && c <= 0xDF ? 1 :
                c >= 0xE0 && c <= 0xEF ? 2 :
                c >= 0xF0 && c <= 0xF4 ? 3 :
                0;

            if(need == 0){
                out += "\xEF\xBF\xBD";
                i++;
                continue;
            }

            size_t good = 0;
            while(good < need && i + 1 + good < n){
                uint8_t d = (uint8_t)pending[i + 1 + good];
                uint8_t low = 0x80;
                uint8_t high = 0xBF;
                if(good == 0){
                    if(c == 0xE0) low = 0xA0;
                    if(c == 0xED) high = 0x9F;
                    if(c == 0xF0) low = 0x90;
                    if(c == 0xF4) high = 0x8F;
                }
                if(d < low || d > high) break;
                good++;
            }

            if(good == need){
                out.append(pending, i, need + 1);
                i += need + 1;
                continue;
            }

            if(i + 1 + good == n) break;

            out += "\xEF\xBF\xBD";
            i += 1 + good;
        }

        pending.erase(0, i);
        return out;
    }

    std::string finish() {
        std::string out = pending.empty() ? "" : "\xEF\xBF\xBD";
        pending.clear();
        return out;
    }
};

struct Tokenizer {
    std::vector<std::string> tokens;
    std::vector<std::string> pieces;
    std::vector<int> types;
    std::unordered_map<std::string, int> vocab;
    std::unordered_map<uint64_t, uint64_t> merges;
    std::vector<int> specials;
    bool specialLead[256] = {};
    int byteToken[256] = {};
    std::vector<uint8_t> kinds;
    int eos = -1;
    int bos = -1;
    int pad = -1;
    bool addBos = false;
    bool spm = false;
    // hy_v4 corta primero los numeros de a tres y los trozos chinos o japoneses y despues el resto
    bool hySplit = false;
    std::vector<int> stops;
    std::string chatTemplate;
    std::mutex cacheLock;
    std::unordered_map<std::string, std::vector<int>> cache;

    bool load(const ModelFile& model, std::string& error) {
        auto tokenList = model.meta.find("tokenizer.ggml.tokens");
        auto mergeList = model.meta.find("tokenizer.ggml.merges");
        if(tokenList == model.meta.end() || mergeList == model.meta.end()){
            error = "the model has no tokenizer";
            return false;
        }

        auto family = model.meta.find("tokenizer.ggml.model");
        if(family != model.meta.end() && family->second.text == "gemma4") return loadSpm(model, tokenList->second.texts, mergeList->second.texts, error);
        if(family != model.meta.end() && family->second.text != "gpt2"){
            error = "unsupported tokenizer " + family->second.text;
            return false;
        }

        auto pre = model.meta.find("tokenizer.ggml.pre");
        hySplit = pre != model.meta.end() && pre->second.text == "hy_v4";
        if(pre != model.meta.end() && pre->second.text != "qwen35" && !hySplit){
            error = "unsupported pre-tokenizer " + pre->second.text;
            return false;
        }

#ifdef _WIN32
        HMODULE icu = LoadLibraryA("icu.dll");
        if(!icu) icu = LoadLibraryA("icuuc.dll");
        auto charType = icu ? (int8_t(*)(int32_t))GetProcAddress(icu, "u_charType") : nullptr;
        if(!charType){
            error = "icu.dll is missing, unicode classes are not available";
            return false;
        }
#else
        // en linux la icu lleva su version en el archivo y en cada funcion: libicuuc.so.78 tiene u_charType_78
        int8_t (*charType)(int32_t) = nullptr;
        for(const char* folder : {"/usr/lib/x86_64-linux-gnu", "/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/lib"}){
            std::error_code code;
            for(const auto& entry : std::filesystem::directory_iterator(folder, code)){
                std::string name = entry.path().filename().string();
                if(name.rfind("libicuuc.so.", 0) != 0) continue;
                std::string version = name.substr(12, name.find('.', 12) - 12);
                HMODULE icu = LoadLibraryA(entry.path().string().c_str());
                if(!icu) continue;
                charType = (int8_t(*)(int32_t))GetProcAddress(icu, ("u_charType_" + version).c_str());
                if(!charType) charType = (int8_t(*)(int32_t))GetProcAddress(icu, "u_charType");
                if(charType) break;
            }
            if(charType) break;
        }
        if(!charType){
            error = "the icu library is missing, install it (on Ubuntu: sudo apt install libicu-dev)";
            return false;
        }
#endif

        kinds.assign(0x110000, 0);
        for(int32_t c = 0; c < 0x110000; c++){
            int8_t type = charType(c);
            kinds[c] =
                type >= 1 && type <= 5 ? KIND_LETTER :
                type >= 6 && type <= 8 ? KIND_MARK :
                type >= 9 && type <= 11 ? KIND_NUMBER :
                0;
        }
        for(const auto& range : recentCodepoints){
            for(uint32_t c = range[0]; c <= range[1]; c++){
                if(charType((int32_t)c) == 0) kinds[c] = (uint8_t)range[2];
            }
        }
        for(uint32_t c : {0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20, 0x85, 0xA0, 0x1680, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000}) kinds[c] = KIND_SPACE;
        for(uint32_t c = 0x2000; c <= 0x200A; c++) kinds[c] = KIND_SPACE;

        tokens = tokenList->second.texts;
        types.assign(tokens.size(), 1);
        auto typeList = model.meta.find("tokenizer.ggml.token_type");
        if(typeList != model.meta.end()){
            for(size_t i = 0; i < tokens.size() && i < typeList->second.wholes.size(); i++) types[i] = (int)typeList->second.wholes[i];
        }

        vocab.clear();
        vocab.reserve(tokens.size() * 2);
        for(size_t i = 0; i < tokens.size(); i++) vocab.emplace(tokens[i], (int)i);

        int byteChar[256];
        int charByte[324];
        int shifted = 256;
        for(int b = 0; b < 324; b++) charByte[b] = -1;
        for(int b = 0; b < 256; b++){
            bool printable = (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
            byteChar[b] = printable ? b : shifted++;
            charByte[byteChar[b]] = b;

            std::string single;
            appendUtf8(single, (uint32_t)byteChar[b]);
            auto found = vocab.find(single);
            if(found == vocab.end()){
                error = "byte " + std::to_string(b) + " has no token";
                return false;
            }
            byteToken[b] = found->second;
        }

        pieces.assign(tokens.size(), "");
        for(size_t i = 0; i < tokens.size(); i++){
            const std::string& text = tokens[i];
            if(types[i] == 3 || types[i] == 4){
                pieces[i] = text;
                continue;
            }
            std::string& raw = pieces[i];
            for(size_t at = 0; at < text.size();){
                size_t from = at;
                uint32_t code = readCodepoint(text, at);
                if(code < 324 && charByte[code] >= 0) raw.push_back((char)charByte[code]);
                else raw.append(text, from, at - from);
            }
        }

        const std::vector<std::string>& list = mergeList->second.texts;
        merges.clear();
        merges.reserve(list.size() * 2);
        for(size_t rank = 0; rank < list.size(); rank++){
            const std::string& line = list[rank];
            size_t space = line.find(' ', 1);
            if(space == std::string::npos){
                error = "merge " + std::to_string(rank) + " is not a pair";
                return false;
            }
            auto left = vocab.find(line.substr(0, space));
            auto right = vocab.find(line.substr(space + 1));
            auto joined = vocab.find(line.substr(0, space) + line.substr(space + 1));
            if(left == vocab.end() || right == vocab.end() || joined == vocab.end()){
                error = "merge " + std::to_string(rank) + " uses a token outside the vocabulary";
                return false;
            }
            uint64_t key = ((uint64_t)(uint32_t)left->second << 32) | (uint32_t)right->second;
            merges.emplace(key, ((uint64_t)rank << 32) | (uint32_t)joined->second);
        }

        specials.clear();
        memset(specialLead, 0, sizeof(specialLead));
        for(size_t i = 0; i < tokens.size(); i++){
            if((types[i] != 3 && types[i] != 4) || tokens[i].empty()) continue;
            specials.push_back((int)i);
            specialLead[(uint8_t)tokens[i][0]] = true;
        }
        std::stable_sort(specials.begin(), specials.end(), [&](int a, int b){ return tokens[a].size() > tokens[b].size(); });

        eos = (int)metaWhole(model, "tokenizer.ggml.eos_token_id", (uint32_t)-1);
        bos = (int)metaWhole(model, "tokenizer.ggml.bos_token_id", (uint32_t)-1);
        pad = (int)metaWhole(model, "tokenizer.ggml.padding_token_id", (uint32_t)-1);
        addBos = metaWhole(model, "tokenizer.ggml.add_bos_token", 0) != 0;
        auto chat = model.meta.find("tokenizer.chat_template");
        chatTemplate = chat == model.meta.end() ? "" : chat->second.text;

        std::lock_guard<std::mutex> hold(cacheLock);
        cache.clear();
        return true;
    }

    bool loadSpm(const ModelFile& model, const std::vector<std::string>& list, const std::vector<std::string>& mergeTexts, std::string& error) {
        spm = true;
        tokens = list;
        types.assign(tokens.size(), 1);
        auto typeList = model.meta.find("tokenizer.ggml.token_type");
        if(typeList != model.meta.end()){
            for(size_t i = 0; i < tokens.size() && i < typeList->second.wholes.size(); i++) types[i] = (int)typeList->second.wholes[i];
        }
        vocab.clear();
        vocab.reserve(tokens.size() * 2);
        for(size_t i = 0; i < tokens.size(); i++) vocab.emplace(tokens[i], (int)i);

        for(int b = 0; b < 256; b++){
            char name[8];
            snprintf(name, sizeof(name), "<0x%02X>", b);
            auto found = vocab.find(name);
            byteToken[b] = found == vocab.end() ? -1 : found->second;
        }

        const std::string blank = "\xE2\x96\x81";
        pieces.assign(tokens.size(), "");
        for(size_t i = 0; i < tokens.size(); i++){
            const std::string& text = tokens[i];
            if(types[i] == 3 || types[i] == 4){
                pieces[i] = text;
                continue;
            }
            if(text.size() == 6 && text.rfind("<0x", 0) == 0 && text[5] == '>'){
                pieces[i] = std::string(1, (char)strtol(text.substr(3, 2).c_str(), nullptr, 16));
                continue;
            }
            std::string& raw = pieces[i];
            for(size_t at = 0; at < text.size();){
                if(text.compare(at, 3, blank) == 0){
                    raw += ' ';
                    at += 3;
                    continue;
                }
                raw += text[at++];
            }
        }

        merges.clear();
        merges.reserve(mergeTexts.size() * 2);
        for(size_t rank = 0; rank < mergeTexts.size(); rank++){
            const std::string& line = mergeTexts[rank];
            size_t space = line.find(' ', 1);
            if(space == std::string::npos) continue;
            auto left = vocab.find(line.substr(0, space));
            auto right = vocab.find(line.substr(space + 1));
            auto joined = vocab.find(line.substr(0, space) + line.substr(space + 1));
            if(left == vocab.end() || right == vocab.end() || joined == vocab.end()) continue;
            uint64_t key = ((uint64_t)(uint32_t)left->second << 32) | (uint32_t)right->second;
            merges.emplace(key, ((uint64_t)rank << 32) | (uint32_t)joined->second);
        }

        specials.clear();
        memset(specialLead, 0, sizeof(specialLead));
        for(size_t i = 0; i < tokens.size(); i++){
            if((types[i] != 3 && types[i] != 4) || tokens[i].empty()) continue;
            specials.push_back((int)i);
            specialLead[(uint8_t)tokens[i][0]] = true;
        }
        std::stable_sort(specials.begin(), specials.end(), [&](int a, int b){ return tokens[a].size() > tokens[b].size(); });

        eos = (int)metaWhole(model, "tokenizer.ggml.eos_token_id", (uint32_t)-1);
        bos = (int)metaWhole(model, "tokenizer.ggml.bos_token_id", (uint32_t)-1);
        pad = (int)metaWhole(model, "tokenizer.ggml.padding_token_id", (uint32_t)-1);
        addBos = true;
        stops.clear();
        for(const char* name : {"<eos>", "<turn|>", "<|tool_response>"}){
            int id = find(name);
            if(id >= 0) stops.push_back(id);
        }
        if(merges.empty()){
            error = "the tokenizer has no usable merges";
            return false;
        }
        auto chat = model.meta.find("tokenizer.chat_template");
        chatTemplate = chat == model.meta.end() ? "" : chat->second.text;

        std::lock_guard<std::mutex> hold(cacheLock);
        cache.clear();
        return true;
    }

    std::vector<int> mergeSpm(const std::string& word) {
        {
            std::lock_guard<std::mutex> hold(cacheLock);
            auto hit = cache.find(word);
            if(hit != cache.end()) return hit->second;
        }

        struct Candidate {
            uint32_t rank;
            int at;
            int left;
            int right;
            int joined;
        };
        auto later = [](const Candidate& a, const Candidate& b){
            return a.rank != b.rank ? a.rank > b.rank : a.at > b.at;
        };

        std::vector<std::string> chars;
        for(size_t at = 0; at < word.size();){
            size_t from = at;
            readCodepoint(word, at);
            chars.push_back(word.substr(from, at - from));
        }
        int n = (int)chars.size();
        std::vector<int> ids(n), next(n), prev(n);
        for(int i = 0; i < n; i++){
            auto found = vocab.find(chars[i]);
            ids[i] = found == vocab.end() ? -1 : found->second;
            next[i] = i + 1 < n ? i + 1 : -1;
            prev[i] = i - 1;
        }

        std::priority_queue<Candidate, std::vector<Candidate>, decltype(later)> queue(later);
        auto offer = [&](int left, int right){
            if(left < 0 || right < 0 || ids[left] < 0 || ids[right] < 0) return;
            auto found = merges.find(((uint64_t)(uint32_t)ids[left] << 32) | (uint32_t)ids[right]);
            if(found != merges.end()) queue.push({(uint32_t)(found->second >> 32), left, ids[left], ids[right], (int)(uint32_t)found->second});
        };
        for(int i = 0; i + 1 < n; i++) offer(i, i + 1);

        while(!queue.empty()){
            Candidate top = queue.top();
            queue.pop();
            int right = next[top.at];
            if(ids[top.at] != top.left || right < 0 || ids[right] != top.right) continue;
            ids[top.at] = top.joined;
            ids[right] = -2;
            next[top.at] = next[right];
            if(next[right] >= 0) prev[next[right]] = top.at;
            offer(top.at, next[top.at]);
            offer(prev[top.at], top.at);
        }

        std::vector<int> out;
        for(int i = n > 0 ? 0 : -1; i >= 0; i = next[i]){
            if(ids[i] >= 0){
                out.push_back(ids[i]);
                continue;
            }
            for(unsigned char b : chars[i]){
                if(byteToken[b] >= 0) out.push_back(byteToken[b]);
            }
        }

        if(word.size() <= 256){
            std::lock_guard<std::mutex> hold(cacheLock);
            if(cache.size() >= 200000) cache.clear();
            cache.emplace(word, out);
        }
        return out;
    }

    void encodeSpmText(const std::string& text, std::vector<int>& out) {
        std::string escaped;
        escaped.reserve(text.size() + 16);
        for(char c : text){
            if(c == ' ') escaped += "\xE2\x96\x81";
            else escaped += c;
        }
        size_t at = 0;
        while(at < escaped.size()){
            bool breaks = escaped[at] == '\n';
            size_t end = at;
            while(end < escaped.size() && (escaped[end] == '\n') == breaks) end++;
            std::string run = escaped.substr(at, end - at);
            int whole = breaks ? find(run) : -1;
            if(whole >= 0) out.push_back(whole);
            else {
                std::vector<int> ids = mergeSpm(run);
                out.insert(out.end(), ids.begin(), ids.end());
            }
            at = end;
        }
    }

    std::vector<int> encodeSpm(const std::string& text, bool control) {
        std::vector<int> out;
        size_t from = 0;
        size_t at = 0;
        while(true){
            int found = -1;
            if(at < text.size() && specialLead[(uint8_t)text[at]]){
                for(int id : specials){
                    if(!control && types[id] == 3) continue;
                    if(text.compare(at, tokens[id].size(), tokens[id]) == 0){
                        found = id;
                        break;
                    }
                }
            }
            if(found < 0 && at < text.size()){
                at++;
                continue;
            }
            if(at > from) encodeSpmText(text.substr(from, at - from), out);
            if(found < 0) break;
            out.push_back(found);
            at += tokens[found].size();
            from = at;
        }
        return out;
    }

    std::vector<int> mergeWord(const std::string& word) {
        {
            std::lock_guard<std::mutex> hold(cacheLock);
            auto hit = cache.find(word);
            if(hit != cache.end()) return hit->second;
        }

        struct Candidate {
            uint32_t rank;
            int at;
            int left;
            int right;
            int joined;
        };
        auto later = [](const Candidate& a, const Candidate& b){
            return a.rank != b.rank ? a.rank > b.rank : a.at > b.at;
        };

        int n = (int)word.size();
        std::vector<int> ids(n), next(n), prev(n);
        for(int i = 0; i < n; i++){
            ids[i] = byteToken[(uint8_t)word[i]];
            next[i] = i + 1 < n ? i + 1 : -1;
            prev[i] = i - 1;
        }

        std::priority_queue<Candidate, std::vector<Candidate>, decltype(later)> queue(later);
        for(int i = 0; i + 1 < n; i++){
            auto found = merges.find(((uint64_t)(uint32_t)ids[i] << 32) | (uint32_t)ids[i + 1]);
            if(found != merges.end()) queue.push({(uint32_t)(found->second >> 32), i, ids[i], ids[i + 1], (int)(uint32_t)found->second});
        }

        while(!queue.empty()){
            Candidate top = queue.top();
            queue.pop();

            int right = next[top.at];
            if(ids[top.at] != top.left || right < 0 || ids[right] != top.right) continue;

            ids[top.at] = top.joined;
            ids[right] = -1;
            next[top.at] = next[right];
            if(next[right] >= 0) prev[next[right]] = top.at;

            int after = next[top.at];
            if(after >= 0){
                auto found = merges.find(((uint64_t)(uint32_t)ids[top.at] << 32) | (uint32_t)ids[after]);
                if(found != merges.end()) queue.push({(uint32_t)(found->second >> 32), top.at, ids[top.at], ids[after], (int)(uint32_t)found->second});
            }

            int before = prev[top.at];
            if(before >= 0){
                auto found = merges.find(((uint64_t)(uint32_t)ids[before] << 32) | (uint32_t)ids[top.at]);
                if(found != merges.end()) queue.push({(uint32_t)(found->second >> 32), before, ids[before], ids[top.at], (int)(uint32_t)found->second});
            }
        }

        std::vector<int> out;
        for(int i = 0; i >= 0; i = next[i]) out.push_back(ids[i]);

        if(word.size() <= 256){
            std::lock_guard<std::mutex> hold(cacheLock);
            if(cache.size() >= 200000) cache.clear();
            cache.emplace(word, out);
        }
        return out;
    }

    bool hyPunct(uint32_t c) const {
        return kinds[c] == 0 && c >= 0x21 && c != 0x7F && !(c >= 0x80 && c < 0xA0);
    }

    static bool hyHan(uint32_t c) {
        return (c >= 0x4E00 && c <= 0x9FA5) || (c >= 0x3040 && c <= 0x309F) || (c >= 0x30A0 && c <= 0x30FF);
    }

    // las tres pasadas del pre tokenizador de hy_v4: lo que una pasada no toma queda como su propio trozo
    void splitHy(const std::vector<uint32_t>& codes, std::vector<std::pair<size_t, size_t>>& parts) const {
        size_t n = codes.size();
        std::vector<std::pair<size_t, size_t>> first, second;
        size_t loose = 0;
        for(size_t i = 0; i < n;){
            if(!(kinds[codes[i]] & KIND_NUMBER)){
                i++;
                continue;
            }
            if(loose < i) first.push_back({loose, i});
            size_t end = i;
            while(end < n && end - i < 3 && (kinds[codes[end]] & KIND_NUMBER)) end++;
            first.push_back({i, end});
            i = loose = end;
        }
        if(loose < n) first.push_back({loose, n});
        for(const auto& seg : first){
            loose = seg.first;
            for(size_t i = seg.first; i < seg.second;){
                if(!hyHan(codes[i])){
                    i++;
                    continue;
                }
                if(loose < i) second.push_back({loose, i});
                size_t end = i;
                while(end < seg.second && hyHan(codes[end])) end++;
                second.push_back({i, end});
                i = loose = end;
            }
            if(loose < seg.second) second.push_back({loose, seg.second});
        }
        auto letter = [&](size_t k){ return (kinds[codes[k]] & (KIND_LETTER | KIND_MARK)) != 0; };
        auto ascii = [&](size_t k){ uint32_t c = codes[k] | 0x20; return c >= 'a' && c <= 'z'; };
        auto lineBreak = [&](size_t k){ return codes[k] == 0x0D || codes[k] == 0x0A; };
        for(const auto& seg : second){
            size_t lo = seg.first, hi = seg.second;
            loose = lo;
            for(size_t i = lo; i < hi;){
                uint32_t c = codes[i];
                size_t end = 0;
                if(c < 0x80 && hyPunct(c) && i + 1 < hi && ascii(i + 1) && kinds[codes[i + 1]] & KIND_LETTER){
                    end = i + 1;
                    while(end < hi && codes[end] < 0x80 && ascii(end)) end++;
                }
                if(!end && letter(i)){
                    end = i + 1;
                    while(end < hi && letter(end)) end++;
                }
                if(!end && !lineBreak(i) && !(kinds[c] & KIND_LETTER) && !hyPunct(c) && i + 1 < hi && letter(i + 1)){
                    end = i + 2;
                    while(end < hi && letter(end)) end++;
                }
                if(!end){
                    size_t k = c == ' ' && i + 1 < hi && hyPunct(codes[i + 1]) ? i + 1 : i;
                    if(hyPunct(codes[k])){
                        end = k + 1;
                        while(end < hi && hyPunct(codes[end])) end++;
                        while(end < hi && lineBreak(end)) end++;
                    }
                }
                if(!end && (kinds[c] & KIND_SPACE)){
                    size_t run = i, lastBreak = hi;
                    while(run < hi && (kinds[codes[run]] & KIND_SPACE)){
                        if(lineBreak(run)) lastBreak = run;
                        run++;
                    }
                    end = lastBreak < hi ? lastBreak + 1 : run == hi || run - i == 1 ? run : run - 1;
                }
                if(!end){
                    i++;
                    continue;
                }
                if(loose < i) parts.push_back({loose, i});
                parts.push_back({i, end});
                i = loose = end;
            }
            if(loose < hi) parts.push_back({loose, hi});
        }
    }

    std::vector<int> encode(const std::string& text, bool control = true) {
        if(spm) return encodeSpm(text, control);
        std::vector<int> out;
        std::vector<uint32_t> codes;
        std::vector<size_t> starts;
        size_t from = 0;
        size_t at = 0;

        while(true){
            int found = -1;
            if(at < text.size() && specialLead[(uint8_t)text[at]]){
                for(int id : specials){
                    if(!control && types[id] == 3) continue;
                    if(text.compare(at, tokens[id].size(), tokens[id]) == 0){
                        found = id;
                        break;
                    }
                }
            }

            if(found < 0 && at < text.size()){
                at++;
                continue;
            }

            codes.clear();
            starts.clear();
            for(size_t i = from; i < at;){
                starts.push_back(i);
                codes.push_back(readCodepoint(text, i));
            }
            starts.push_back(at);

            size_t n = codes.size();
            if(hySplit){
                std::vector<std::pair<size_t, size_t>> parts;
                splitHy(codes, parts);
                for(const auto& part : parts){
                    std::vector<int> ids = mergeWord(text.substr(starts[part.first], starts[part.second] - starts[part.first]));
                    out.insert(out.end(), ids.begin(), ids.end());
                }
                n = 0;
            }
            size_t i = 0;
            while(i < n){
                uint32_t c = codes[i];
                uint8_t kind = kinds[c];
                size_t end = 0;

                if(c == '\'' && i + 1 < n){
                    uint32_t first = codes[i + 1] | 0x20;
                    uint32_t second = i + 2 < n ? codes[i + 2] | 0x20 : 0;
                    if(first == 's' || first == 't' || first == 'm' || first == 'd') end = i + 2;
                    else if((first == 'r' && second == 'e') || (first == 'v' && second == 'e') || (first == 'l' && second == 'l')) end = i + 3;
                }

                if(!end){
                    size_t word = i;
                    if(c != '\r' && c != '\n' && !(kind & (KIND_LETTER | KIND_NUMBER)) && i + 1 < n && (kinds[codes[i + 1]] & (KIND_LETTER | KIND_MARK))) word = i + 1;
                    if(kinds[codes[word]] & (KIND_LETTER | KIND_MARK)){
                        end = word + 1;
                        while(end < n && (kinds[codes[end]] & (KIND_LETTER | KIND_MARK))) end++;
                    }
                }

                if(!end && (kind & KIND_NUMBER)) end = i + 1;

                if(!end){
                    size_t mark = c == ' ' && i + 1 < n && kinds[codes[i + 1]] == 0 ? i + 1 : i;
                    if(kinds[codes[mark]] == 0){
                        end = mark + 1;
                        while(end < n && kinds[codes[end]] == 0) end++;
                        while(end < n && (codes[end] == '\r' || codes[end] == '\n')) end++;
                    }
                }

                if(!end){
                    size_t run = i;
                    size_t lastBreak = n;
                    while(run < n && (kinds[codes[run]] & KIND_SPACE)){
                        if(codes[run] == '\r' || codes[run] == '\n') lastBreak = run;
                        run++;
                    }
                    end =
                        lastBreak < n ? lastBreak + 1 :
                        run == n || run - i == 1 ? run :
                        run - 1;
                }

                std::vector<int> ids = mergeWord(text.substr(starts[i], starts[end] - starts[i]));
                out.insert(out.end(), ids.begin(), ids.end());
                i = end;
            }

            if(found < 0) break;
            out.push_back(found);
            at += tokens[found].size();
            from = at;
        }

        return out;
    }

    std::string tokenText(int id) const {
        if(id < 0 || (size_t)id >= pieces.size()) return "";
        return pieces[id];
    }

    std::string decode(const std::vector<int>& ids) const {
        TextStream stream;
        std::string out;
        for(int id : ids) out += stream.push(tokenText(id));
        return out + stream.finish();
    }

    int find(const std::string& text) const {
        auto found = vocab.find(text);
        return found == vocab.end() ? -1 : found->second;
    }

    bool isStop(int id) const {
        return id == eos || (bos >= 0 && id == bos) || (pad >= 0 && id == pad) || std::find(stops.begin(), stops.end(), id) != stops.end();
    }
};
