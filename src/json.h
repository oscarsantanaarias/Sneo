#pragma once
#include <string>
#include <vector>
#include <utility>
#include <cstdlib>
#include <cstdio>
#include <cstdint>

enum JsonKind { JSON_NULL, JSON_BOOL, JSON_NUMBER, JSON_TEXT, JSON_LIST, JSON_OBJECT };

struct Json {
    JsonKind kind = JSON_NULL;
    bool flag = false;
    double number = 0;
    std::string text;
    std::vector<Json> items;
    std::vector<std::pair<std::string, Json>> fields;

    const Json& operator[](const std::string& key) const {
        static Json empty;
        for(auto& f : fields){
            if(f.first == key) return f.second;
        }
        return empty;
    }

    bool has(const std::string& key) const {
        for(auto& f : fields){
            if(f.first == key) return true;
        }
        return false;
    }

    std::string str(const std::string& fallback = "") const {
        return kind == JSON_TEXT ? text : fallback;
    }

    double num(double fallback = 0) const {
        return kind == JSON_NUMBER ? number : fallback;
    }

    bool truthy(bool fallback = false) const {
        return kind == JSON_BOOL ? flag : fallback;
    }
};

inline void skipSpace(const std::string& s, size_t& i) {
    while(i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t')) i++;
}

inline void appendUtf8(std::string& out, uint32_t code) {
    if(code < 0x80){
        out.push_back((char)code);
    } else if(code < 0x800){
        out.push_back((char)(0xC0 | (code >> 6)));
        out.push_back((char)(0x80 | (code & 0x3F)));
    } else if(code < 0x10000){
        out.push_back((char)(0xE0 | (code >> 12)));
        out.push_back((char)(0x80 | ((code >> 6) & 0x3F)));
        out.push_back((char)(0x80 | (code & 0x3F)));
    } else {
        out.push_back((char)(0xF0 | (code >> 18)));
        out.push_back((char)(0x80 | ((code >> 12) & 0x3F)));
        out.push_back((char)(0x80 | ((code >> 6) & 0x3F)));
        out.push_back((char)(0x80 | (code & 0x3F)));
    }
}

inline bool readJsonText(const std::string& s, size_t& i, std::string& out) {
    if(i >= s.size() || s[i] != '"') return false;
    i++;
    while(i < s.size()){
        char c = s[i++];
        if(c == '"') return true;
        if(c != '\\'){
            out.push_back(c);
            continue;
        }
        if(i >= s.size()) return false;
        char e = s[i++];
        if(e == 'n') out.push_back('\n');
        else if(e == 't') out.push_back('\t');
        else if(e == 'r') out.push_back('\r');
        else if(e == 'b') out.push_back('\b');
        else if(e == 'f') out.push_back('\f');
        else if(e == 'u'){
            if(i + 4 > s.size()) return false;
            uint32_t code = (uint32_t)strtoul(s.substr(i, 4).c_str(), nullptr, 16);
            i += 4;
            if(code >= 0xD800 && code <= 0xDBFF && i + 6 <= s.size() && s[i] == '\\' && s[i + 1] == 'u'){
                uint32_t low = (uint32_t)strtoul(s.substr(i + 2, 4).c_str(), nullptr, 16);
                if(low >= 0xDC00 && low <= 0xDFFF){
                    code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
                    i += 6;
                }
            }
            appendUtf8(out, code);
        } else out.push_back(e);
    }
    return false;
}

inline bool readJson(const std::string& s, size_t& i, Json& out, int depth = 0) {
    if(depth > 200) return false;
    skipSpace(s, i);
    if(i >= s.size()) return false;

    char c = s[i];
    if(c == '{'){
        out.kind = JSON_OBJECT;
        i++;
        skipSpace(s, i);
        if(i < s.size() && s[i] == '}'){
            i++;
            return true;
        }
        while(i < s.size()){
            skipSpace(s, i);
            std::string key;
            if(!readJsonText(s, i, key)) return false;
            skipSpace(s, i);
            if(i >= s.size() || s[i] != ':') return false;
            i++;
            Json value;
            if(!readJson(s, i, value, depth + 1)) return false;
            out.fields.emplace_back(std::move(key), std::move(value));
            skipSpace(s, i);
            if(i < s.size() && s[i] == ','){
                i++;
                continue;
            }
            if(i < s.size() && s[i] == '}'){
                i++;
                return true;
            }
            return false;
        }
        return false;
    }

    if(c == '['){
        out.kind = JSON_LIST;
        i++;
        skipSpace(s, i);
        if(i < s.size() && s[i] == ']'){
            i++;
            return true;
        }
        while(i < s.size()){
            Json value;
            if(!readJson(s, i, value, depth + 1)) return false;
            out.items.push_back(std::move(value));
            skipSpace(s, i);
            if(i < s.size() && s[i] == ','){
                i++;
                continue;
            }
            if(i < s.size() && s[i] == ']'){
                i++;
                return true;
            }
            return false;
        }
        return false;
    }

    if(c == '"'){
        out.kind = JSON_TEXT;
        return readJsonText(s, i, out.text);
    }

    if(s.compare(i, 4, "true") == 0){
        out.kind = JSON_BOOL;
        out.flag = true;
        i += 4;
        return true;
    }

    if(s.compare(i, 5, "false") == 0){
        out.kind = JSON_BOOL;
        i += 5;
        return true;
    }

    if(s.compare(i, 4, "null") == 0){
        i += 4;
        return true;
    }

    char* end = nullptr;
    out.number = strtod(s.c_str() + i, &end);
    if(end == s.c_str() + i) return false;
    out.kind = JSON_NUMBER;
    i = end - s.c_str();
    return true;
}

inline Json parseJson(const std::string& s) {
    Json out;
    size_t i = 0;
    if(!readJson(s, i, out)) return Json();
    return out;
}

inline std::string quoteJson(const std::string& s) {
    std::string out = "\"";
    for(unsigned char c : s){
        if(c == '"') out += "\\\"";
        else if(c == '\\') out += "\\\\";
        else if(c == '\n') out += "\\n";
        else if(c == '\r') out += "\\r";
        else if(c == '\t') out += "\\t";
        else if(c < 0x20){
            char code[8];
            snprintf(code, sizeof(code), "\\u%04x", c);
            out += code;
        } else out.push_back((char)c);
    }
    out.push_back('"');
    return out;
}

inline std::string writeJson(const Json& v) {
    if(v.kind == JSON_NULL) return "null";
    if(v.kind == JSON_BOOL) return v.flag ? "true" : "false";
    if(v.kind == JSON_NUMBER){
        char buffer[64];
        snprintf(buffer, sizeof(buffer), "%.17g", v.number);
        return buffer;
    }
    if(v.kind == JSON_TEXT) return quoteJson(v.text);
    if(v.kind == JSON_LIST){
        std::string out = "[";
        for(size_t i = 0; i < v.items.size(); i++){
            if(i) out += ",";
            out += writeJson(v.items[i]);
        }
        return out + "]";
    }
    std::string out = "{";
    for(size_t i = 0; i < v.fields.size(); i++){
        if(i) out += ",";
        out += quoteJson(v.fields[i].first) + ":" + writeJson(v.fields[i].second);
    }
    return out + "}";
}
