#pragma once
#include "net.h"
#include "json.h"
#include <functional>
#include <random>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <array>
#include <ctime>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <cstring>

struct GenerateRequest {
    Json messages;
    Json tools;
    std::string model;
    bool stream = false;
    bool messagesApi = false;
    bool includeUsage = false;
    int maxTokens = 8192;
    float temperature = 1.0f;
    float topP = 0.95f;
    int topK = 20;
    float minP = 0.0f;
    float repeatPenalty = 1.0f;
    int shortlist = 0;
    std::string effort = "medium";
    int thinkBudget = 0;
    bool thinking = true;
    std::vector<std::string> stops;
    uint64_t seed = 0;
};

struct GenerateHooks {
    std::function<std::string(const GenerateRequest&)> render;
    std::function<std::vector<int>(const std::string&)> encode;
    std::function<std::string(int)> piece;
    std::function<std::string(int)> tokenText;
    std::function<bool(const std::vector<int>&, size_t, std::vector<float>&)> forward;
    std::function<bool(int)> isStop;
    std::function<uint64_t()> reusedTokens;
    std::function<uint64_t(const std::vector<int>&)> reusable;
    std::function<bool(const std::vector<int>&, std::vector<float>&)> verify;
    std::function<bool(int)> mark;
    std::function<bool()> rewind;
    std::function<bool(const std::vector<int>&)> settle;
    std::function<std::vector<int>(int, int)> propose;
    std::function<bool(const std::vector<int>&)> commit;
    std::function<std::string()> problem;
    int verifyLimit = 0;
    int draftLimit = 0;
    int copyLimit = 8;
    int guessStart = 0;
    std::string thinkOpen = "<think>";
    std::string thinkClose = "</think>";
    std::string toolOpen = "<tool_call>";
    std::string toolClose = "</tool_call>";
    std::string closeThinking = "\n</think>\n\n";
};

struct GenerateResult {
    uint64_t copied = 0;
    uint64_t copyKept = 0;
    std::string reasoning;
    std::string content;
    Json toolCalls;
    std::string finish = "stop";
    std::string stopHit;
    uint64_t promptTokens = 0;
    uint64_t outputTokens = 0;
    uint64_t cachedTokens = 0;
    uint64_t drafted = 0;
    uint64_t kept = 0;
    double promptMs = 0;
    double outputMs = 0;
    double modelMs = 0;
    bool clientGone = false;
    std::string thinkingCut;
    int thinkingTokens = 0;
};

inline GenerateRequest readGenerateRequest(const Json& body) {
    GenerateRequest req;
    req.messages = body["messages"];
    req.tools = body["tools"];
    req.model = body["model"].str("sneo");
    req.stream = body["stream"].truthy(false);
    req.includeUsage = body["stream_options"]["include_usage"].truthy(false);
    req.maxTokens = (int)body["max_tokens"].num(body["max_completion_tokens"].num(8192));
    req.temperature = (float)body["temperature"].num(1.0);
    req.topP = (float)body["top_p"].num(0.95);
    req.topK = (int)body["top_k"].num(20);
    req.minP = (float)body["min_p"].num(0.0);
    req.repeatPenalty = (float)body["repeat_penalty"].num(body["repetition_penalty"].num(1.0));
    req.effort = body["reasoning_effort"].str("medium");
    req.thinkBudget = (int)body["thinking_budget"].num(0);
    req.thinking = req.effort != "none" && body["chat_template_kwargs"]["enable_thinking"].truthy(true);
    req.seed = (uint64_t)body["seed"].num((double)time(nullptr));

    const Json& stop = body["stop"];
    if(stop.kind == JSON_TEXT) req.stops.push_back(stop.text);
    for(const Json& s : stop.items) req.stops.push_back(s.str());

    if(req.maxTokens < 1) req.maxTokens = 1;
    if(req.topK < 0) req.topK = 0;
    return req;
}

inline std::vector<std::pair<int, double>> shapeChoices(const std::vector<float>& sorted, const std::vector<int>& order, const GenerateRequest& req) {
    size_t keep = sorted.size();
    float top = sorted[0];
    std::vector<double> chance(keep);
    double total = 0;
    for(size_t i = 0; i < keep; i++){
        chance[i] = std::exp((double)(sorted[i] - top) / req.temperature);
        total += chance[i];
    }

    double running = 0;
    size_t cut = keep;
    for(size_t i = 0; i < keep; i++){
        chance[i] /= total;
        if(req.minP > 0 && chance[i] < req.minP * chance[0]){
            cut = i;
            break;
        }
        running += chance[i];
        if(running >= req.topP){
            cut = i + 1;
            break;
        }
    }
    if(cut == 0) cut = 1;

    double mass = 0;
    for(size_t i = 0; i < cut; i++) mass += chance[i];
    std::vector<std::pair<int, double>> choices(cut);
    for(size_t i = 0; i < cut; i++) choices[i] = {order[i], chance[i] / mass};
    return choices;
}

inline std::vector<std::pair<int, double>> buildChoices(std::vector<float>& logits, const GenerateRequest& req, const std::vector<int>& recent) {
    if(req.shortlist > 0 && logits.size() == 2 * (size_t)req.shortlist){
        size_t keep = req.topK > 0 ? std::min<size_t>((size_t)req.topK, (size_t)req.shortlist) : (size_t)req.shortlist;
        std::vector<float> sorted(logits.begin(), logits.begin() + keep);
        std::vector<int> order(keep);
        memcpy(order.data(), logits.data() + req.shortlist, keep * 4);
        if(req.temperature <= 0.0f) return {{order[0], 1.0}};
        return shapeChoices(sorted, order, req);
    }
    if(req.repeatPenalty != 1.0f){
        for(int id : recent){
            if(id < 0 || id >= (int)logits.size()) continue;
            logits[id] = logits[id] > 0 ? logits[id] / req.repeatPenalty : logits[id] * req.repeatPenalty;
        }
    }

    if(req.temperature <= 0.0f){
        int best = (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
        return {{best, 1.0}};
    }

    std::vector<int> order(logits.size());
    for(size_t i = 0; i < order.size(); i++) order[i] = (int)i;
    size_t keep = req.topK > 0 && req.topK < (int)order.size() ? (size_t)req.topK : std::min<size_t>(order.size(), 1024);
    std::partial_sort(order.begin(), order.begin() + keep, order.end(), [&](int a, int b){ return logits[a] > logits[b]; });
    order.resize(keep);
    std::vector<float> sorted(keep);
    for(size_t i = 0; i < keep; i++) sorted[i] = logits[order[i]];
    return shapeChoices(sorted, order, req);
}

inline int pickChoice(const std::vector<std::pair<int, double>>& choices, int banned, std::mt19937_64& random) {
    double mass = 0;
    for(const auto& c : choices){
        if(c.first != banned) mass += c.second;
    }
    if(mass <= 0) return choices.front().first;
    double pick = std::uniform_real_distribution<double>(0.0, mass)(random);
    for(const auto& c : choices){
        if(c.first == banned) continue;
        pick -= c.second;
        if(pick <= 0) return c.first;
    }
    for(auto it = choices.rbegin(); it != choices.rend(); ++it){
        if(it->first != banned) return it->first;
    }
    return choices.front().first;
}

inline int checkDraft(std::vector<float>& logits, int draft, const GenerateRequest& req, const std::vector<int>& recent, std::mt19937_64& random, bool& kept) {
    std::vector<std::pair<int, double>> choices = buildChoices(logits, req, recent);
    double chance = 0;
    for(const auto& c : choices){
        if(c.first == draft) chance = c.second;
    }
    if(req.temperature <= 0.0f){
        kept = choices.front().first == draft;
        return choices.front().first;
    }
    kept = chance > 0 && std::uniform_real_distribution<double>(0.0, 1.0)(random) < chance;
    if(kept) return draft;
    return pickChoice(choices, draft, random);
}

struct DraftIndex {
    std::vector<int> seen;
    std::unordered_map<uint64_t, std::array<int, 6>> ends;

    static uint64_t keyOf(int a, int b, int c) {
        uint64_t h = (uint32_t)a;
        h = h * 1000003ULL ^ (uint32_t)b;
        h = h * 1000003ULL ^ (uint32_t)c;
        return h * 0x9E3779B97F4A7C15ULL;
    }

    void add(int token) {
        seen.push_back(token);
        size_t last = seen.size() - 1;
        if(last < 2) return;
        auto found = ends.find(keyOf(seen[last - 2], seen[last - 1], seen[last]));
        if(found == ends.end()){
            std::array<int, 6> fresh;
            fresh.fill(-1);
            fresh[0] = (int)last;
            ends.emplace(keyOf(seen[last - 2], seen[last - 1], seen[last]), fresh);
            return;
        }
        std::array<int, 6>& slots = found->second;
        for(size_t k = slots.size() - 1; k > 0; k--) slots[k] = slots[k - 1];
        slots[0] = (int)last;
    }

    void addAll(const std::vector<int>& tokens) {
        seen.reserve(seen.size() + tokens.size() + 4096);
        for(int t : tokens) add(t);
    }

    std::vector<int> propose(int most, int& matched) {
        matched = 0;
        if(seen.size() < 4 || most <= 0) return {};
        int now = (int)seen.size() - 1;
        auto found = ends.find(keyOf(seen[now - 2], seen[now - 1], seen[now]));
        if(found == ends.end()) return {};

        int best = -1;
        int bestLength = 0;
        for(int at : found->second){
            if(at < 0 || at >= now) continue;
            int length = 0;
            while(length < 64 && length <= at && seen[at - length] == seen[now - length]) length++;
            if(length > bestLength){
                bestLength = length;
                best = at;
            }
        }
        if(best < 0 || bestLength < 3) return {};

        matched = bestLength;
        int wanted = std::min(most, std::max(2, bestLength));
        int count = std::min(wanted, now - best);
        for(int k = 0; k < count; k++){
            if(seen[best + 1 + k] >= 0) continue;
            count = k;
            break;
        }
        if(count <= 0) return {};
        return std::vector<int>(seen.begin() + best + 1, seen.begin() + best + 1 + count);
    }
};

inline bool repeatsAt(const std::vector<int>& seen, size_t period, size_t span) {
    size_t same = 0;
    for(size_t i = seen.size() - span; i < seen.size(); i++) same += seen[i] == seen[i - period];
    return same * 20 >= span * 19;
}

inline bool wordy(const std::string& text) {
    int first = -1;
    for(unsigned char b : text){
        bool letter = (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z') || b >= 0x80;
        if(!letter) continue;
        int folded = b >= 'A' && b <= 'Z' ? b + 32 : b;
        if(first < 0) first = folded;
        else if(folded != first) return true;
    }
    return false;
}

inline bool stuckThinking(const std::vector<int>& seen, size_t start, const std::function<std::string(int)>& tokenText) {
    size_t made = seen.size() - start;
    if(!tokenText || made < 256 || made % 16 != 0) return false;
    size_t widest = made % 64 == 0 ? 512 : 64;
    std::vector<int> words;
    for(size_t period = 1; period <= widest; period++){
        size_t span = std::max<size_t>(256, period * 4);
        if(span + period > made) return false;
        if(!repeatsAt(seen, period, span)) continue;
        if(words.empty()){
            size_t reach = std::min<size_t>(made, 2048);
            words.assign(reach + 1, 0);
            for(size_t k = 0; k < reach; k++) words[k + 1] = words[k] + (wordy(tokenText(seen[seen.size() - 1 - k])) ? 1 : 0);
        }
        if(words[span] * 4 >= (int)span) return true;
    }
    return false;
}

inline int sampleToken(std::vector<float>& logits, const GenerateRequest& req, const std::vector<int>& recent, std::mt19937_64& random) {
    return pickChoice(buildChoices(logits, req, recent), -1, random);
}

inline bool sendText(SOCKET s, const std::string& text) {
    return sendAll(s, text.data(), text.size());
}

inline bool sendEvent(SOCKET s, const std::string& json) {
    return sendText(s, "data: " + json + "\n\n");
}

inline std::string chunkJson(const std::string& id, const std::string& model, int64_t created, const std::string& delta, const char* finish) {
    std::string out = "{\"id\":" + quoteJson(id) + ",\"object\":\"chat.completion.chunk\",\"created\":" + std::to_string(created)
        + ",\"model\":" + quoteJson(model) + ",\"choices\":[{\"index\":0,\"delta\":" + delta + ",\"finish_reason\":";
    out += finish ? quoteJson(finish) : "null";
    return out + "}]}";
}

inline std::string usageJson(const GenerateResult& r) {
    char text[700];
    uint64_t fresh = r.promptTokens > r.cachedTokens ? r.promptTokens - r.cachedTokens : 0;
    double promptSpeed = r.promptMs > 0 ? fresh * 1000.0 / r.promptMs : 0;
    double outputSpeed = r.outputMs > 0 ? r.outputTokens * 1000.0 / r.outputMs : 0;
    snprintf(text, sizeof(text),
        "\"usage\":{\"prompt_tokens\":%llu,\"completion_tokens\":%llu,\"total_tokens\":%llu,\"prompt_tokens_details\":{\"cached_tokens\":%llu}},"
        "\"timings\":{\"prompt_n\":%llu,\"prompt_ms\":%.1f,\"prompt_per_second\":%.2f,\"cache_n\":%llu,\"predicted_n\":%llu,\"predicted_ms\":%.1f,\"predicted_per_second\":%.2f,\"drafted\":%llu,\"kept\":%llu}",
        (unsigned long long)r.promptTokens, (unsigned long long)r.outputTokens, (unsigned long long)(r.promptTokens + r.outputTokens), (unsigned long long)r.cachedTokens,
        (unsigned long long)fresh, r.promptMs, promptSpeed, (unsigned long long)r.cachedTokens, (unsigned long long)r.outputTokens, r.outputMs, outputSpeed, (unsigned long long)r.drafted, (unsigned long long)r.kept);
    return text;
}

// avisos sin texto para la app mientras lee el prompt o escribe una herramienta
inline std::string noticeJson(const std::string& id, const std::string& model, int64_t created, const std::string& field, const std::string& value) {
    return "{\"id\":" + quoteJson(id) + ",\"object\":\"chat.completion.chunk\",\"created\":" + std::to_string(created)
        + ",\"model\":" + quoteJson(model) + ",\"choices\":[]," + quoteJson(field) + ":" + value + "}";
}

inline size_t wholeChars(const std::string& text, size_t end) {
    size_t lead = end;
    while(lead > 0 && end - lead < 4 && ((unsigned char)text[lead - 1] & 0xC0) == 0x80) lead--;
    if(lead == 0) return end;
    unsigned char first = (unsigned char)text[lead - 1];
    size_t need = first >= 0xF0
        ? 4
        : first >= 0xE0
            ? 3
            : first >= 0xC0
                ? 2
                : 1;
    return end - (lead - 1) < need ? lead - 1 : end;
}

struct OutputSplitter {
    bool inThink = false;
    bool inTool = false;
    std::string pending;
    std::string toolText;
    std::vector<std::string> tools;
    bool started = false;
    std::string thinkOpen = "<think>";
    std::string thinkClose = "</think>";
    std::string toolOpen = "<tool_call>";
    std::string toolClose = "</tool_call>";
    bool skipBreak = false;

    void addContent(std::string& content, std::string text) {
        static const std::string blank = {' ', char(13), char(10)};
        if(!started){
            size_t first = text.find_first_not_of(blank);
            if(first == std::string::npos) return;
            text.erase(0, first);
            started = true;
        }
        content += text;
    }

    void push(const std::string& text, std::string& reasoning, std::string& content) {
        pending += text;
        while(!pending.empty()){
            if(inThink && skipBreak){
                if(pending[0] == '\n') pending.erase(0, 1);
                if(pending.empty()) return;
                skipBreak = false;
            }
            const std::string* open = inTool ? &toolClose : inThink ? &thinkClose : nullptr;
            if(open){
                size_t at = pending.find(*open);
                if(at == std::string::npos){
                    size_t keep = 0;
                    for(size_t k = 1; k < open->size() && k <= pending.size(); k++){
                        if(open->compare(0, k, pending, pending.size() - k, k) == 0) keep = k;
                    }
                    size_t safe = wholeChars(pending, pending.size() - keep);
                    if(inTool) toolText += pending.substr(0, safe);
                    else reasoning += pending.substr(0, safe);
                    pending.erase(0, safe);
                    return;
                }
                if(inTool){
                    toolText += pending.substr(0, at);
                    tools.push_back(toolText);
                    toolText.clear();
                    inTool = false;
                } else {
                    reasoning += pending.substr(0, at);
                    inThink = false;
                }
                pending.erase(0, at + open->size());
                continue;
            }

            size_t think = pending.find(thinkOpen);
            size_t tool = pending.find(toolOpen);
            size_t next = std::min(think, tool);
            if(next == std::string::npos){
                size_t keep = 0;
                size_t longest = std::max(thinkOpen.size(), toolOpen.size());
                for(size_t k = 1; k <= longest && k <= pending.size(); k++){
                    std::string tail = pending.substr(pending.size() - k);
                    if(toolOpen.rfind(tail, 0) == 0 || thinkOpen.rfind(tail, 0) == 0) keep = k;
                }
                size_t cut = wholeChars(pending, pending.size() - keep);
                addContent(content, pending.substr(0, cut));
                pending.erase(0, cut);
                return;
            }
            addContent(content, pending.substr(0, next));
            if(next == think){
                inThink = true;
                skipBreak = true;
                pending.erase(0, next + thinkOpen.size());
            } else {
                inTool = true;
                pending.erase(0, next + toolOpen.size());
            }
        }
    }

    void finish(std::string& reasoning, std::string& content) {
        if(inTool) tools.push_back(toolText + pending);
        else if(inThink) reasoning += pending;
        else addContent(content, pending);
        pending.clear();
        toolText.clear();
    }
};

inline Json parseGemmaValue(const std::string& s, size_t& at) {
    static const std::string quote = "<|\"|>";
    Json v;
    while(at < s.size() && isspace((unsigned char)s[at])) at++;
    if(s.compare(at, quote.size(), quote) == 0){
        size_t end = s.find(quote, at + quote.size());
        v.kind = JSON_TEXT;
        v.text = s.substr(at + quote.size(), end == std::string::npos ? std::string::npos : end - at - quote.size());
        at = end == std::string::npos ? s.size() : end + quote.size();
        return v;
    }
    if(at < s.size() && (s[at] == '{' || s[at] == '[')){
        bool object = s[at] == '{';
        char close = object ? '}' : ']';
        v.kind = object ? JSON_OBJECT : JSON_LIST;
        at++;
        while(at < s.size()){
            while(at < s.size() && (isspace((unsigned char)s[at]) || s[at] == ',')) at++;
            if(at >= s.size() || s[at] == close){
                at++;
                break;
            }
            if(!object){
                v.items.push_back(parseGemmaValue(s, at));
                continue;
            }
            std::string key;
            if(s.compare(at, quote.size(), quote) == 0) key = parseGemmaValue(s, at).text;
            else {
                size_t colon = s.find(':', at);
                if(colon == std::string::npos) break;
                key = s.substr(at, colon - at);
                at = colon;
            }
            while(at < s.size() && s[at] != ':') at++;
            at++;
            v.fields.push_back({key, parseGemmaValue(s, at)});
        }
        return v;
    }
    size_t end = at;
    while(end < s.size() && s[end] != ',' && s[end] != '}' && s[end] != ']') end++;
    std::string word = s.substr(at, end - at);
    while(!word.empty() && isspace((unsigned char)word.back())) word.pop_back();
    at = end;
    Json parsed = parseJson(word);
    if(parsed.kind != JSON_NULL || word == "null") return parsed;
    v.kind = JSON_TEXT;
    v.text = word;
    return v;
}

inline Json parseToolText(const std::string& text, int index, const Json& tools) {
    Json call;
    call.kind = JSON_OBJECT;
    std::string name, args = "{}";

    size_t lead = text.find_first_not_of(" \n\r\t");
    Json direct = parseJson(text);
    if(lead != std::string::npos && text.compare(lead, 5, "call:") == 0){
        size_t open = text.find('{', lead);
        name = text.substr(lead + 5, open == std::string::npos ? std::string::npos : open - lead - 5);
        while(!name.empty() && isspace((unsigned char)name.back())) name.pop_back();
        size_t at = open == std::string::npos ? text.size() : open;
        Json params = at < text.size() ? parseGemmaValue(text, at) : Json();
        if(params.kind != JSON_OBJECT){
            params = Json();
            params.kind = JSON_OBJECT;
        }
        args = writeJson(params);
    } else if(direct.kind == JSON_OBJECT && direct.has("name")){
        name = direct["name"].str();
        args = direct["arguments"].kind == JSON_TEXT ? direct["arguments"].text : writeJson(direct["arguments"]);
    } else {
        size_t fn = text.find("<function=");
        if(fn != std::string::npos){
            size_t end = text.find('>', fn);
            name = text.substr(fn + 10, end == std::string::npos ? std::string::npos : end - fn - 10);
            while(!name.empty() && (name.back() == ' ' || name.back() == '\n')) name.pop_back();
            Json params;
            params.kind = JSON_OBJECT;
            size_t at = end;
            while(at != std::string::npos){
                size_t p = text.find("<parameter=", at);
                if(p == std::string::npos) break;
                size_t close = text.find('>', p);
                size_t stop = text.find("</parameter>", close);
                if(close == std::string::npos) break;
                std::string key = text.substr(p + 11, close - p - 11);
                std::string value = text.substr(close + 1, stop == std::string::npos ? std::string::npos : stop - close - 1);
                while(!value.empty() && (value.front() == '\n')) value.erase(0, 1);
                while(!value.empty() && (value.back() == '\n')) value.pop_back();
                bool asText = true;
                for(const Json& t : tools.items){
                    const Json& known = t["function"].kind == JSON_OBJECT ? t["function"] : t;
                    if(known["name"].str() != name) continue;
                    const Json& type = known["parameters"]["properties"][key]["type"];
                    asText = type.kind != JSON_TEXT || type.text == "string";
                }
                Json parsed = asText ? Json() : parseJson(value);
                if(asText || (parsed.kind == JSON_NULL && value != "null")){
                    parsed.kind = JSON_TEXT;
                    parsed.text = value;
                }
                params.fields.push_back({key, parsed});
                at = stop;
            }
            args = writeJson(params);
        }
    }

    while(!name.empty() && (name.back() == ' ' || name.back() == '\n')) name.pop_back();
    char id[32];
    snprintf(id, sizeof(id), "call_%d_%04x", index, (unsigned)(rand() & 0xFFFF));

    Json fn;
    fn.kind = JSON_OBJECT;
    Json nameValue, argValue, idValue, typeValue;
    nameValue.kind = argValue.kind = idValue.kind = typeValue.kind = JSON_TEXT;
    nameValue.text = name;
    argValue.text = args;
    idValue.text = id;
    typeValue.text = "function";
    fn.fields = {{"name", nameValue}, {"arguments", argValue}};
    call.fields = {{"index", Json()}, {"id", idValue}, {"type", typeValue}, {"function", fn}};
    call.fields[0].second.kind = JSON_NUMBER;
    call.fields[0].second.number = index;
    return call;
}

inline GenerateResult runGenerate(SOCKET client, const GenerateRequest& req, GenerateHooks& hooks) {
    GenerateResult result;
    std::string id = (req.messagesApi ? "msg_" : "chatcmpl-") + std::to_string(nowMs());
    int64_t created = (int64_t)time(nullptr);
    int blockIndex = -1;
    std::string blockKind;
    auto messageEvent = [&](const char* name, const std::string& data) {
        return sendText(client, std::string("event: ") + name + "\ndata: " + data + "\n\n");
    };
    auto enterBlock = [&](const std::string& kind, const std::string& start) {
        if(blockKind == kind) return true;
        if(blockIndex >= 0 && !messageEvent("content_block_stop", "{\"type\":\"content_block_stop\",\"index\":" + std::to_string(blockIndex) + "}")) return false;
        blockIndex++;
        blockKind = kind;
        return messageEvent("content_block_start", "{\"type\":\"content_block_start\",\"index\":" + std::to_string(blockIndex) + ",\"content_block\":" + start + "}");
    };
    auto failText = [&] {
        std::string why = quoteJson(hooks.problem ? hooks.problem() : std::string("generation failed"));
        return req.messagesApi
            ? "{\"type\":\"error\",\"error\":{\"type\":\"api_error\",\"message\":" + why + "}}"
            : "{\"error\":{\"message\":" + why + "}}";
    };

    if(req.stream){
        sendText(client, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n");
        if(!req.messagesApi) sendEvent(client, chunkJson(id, req.model, created, "{\"role\":\"assistant\",\"content\":\"\"}", nullptr));
    }

    std::string prompt = hooks.render(req);
    std::vector<int> tokens = hooks.encode(prompt);
    result.promptTokens = tokens.size();
    bool notices = req.stream && !req.messagesApi;
    if(notices){
        uint64_t reuse = hooks.reusable ? hooks.reusable(tokens) : 0;
        sendEvent(client, noticeJson(id, req.model, created, "prefill", "{\"tokens\":" + std::to_string(tokens.size()) + ",\"cached\":" + std::to_string(reuse) + "}"));
    }
    if(req.stream && req.messagesApi){
        messageEvent("message_start", "{\"type\":\"message_start\",\"message\":{\"id\":" + quoteJson(id) + ",\"type\":\"message\",\"role\":\"assistant\",\"model\":" + quoteJson(req.model)
            + ",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":" + std::to_string(tokens.size()) + ",\"output_tokens\":0}}}");
    }

    std::string ping = req.messagesApi ? "event: ping\ndata: {\"type\":\"ping\"}\n\n" : ": ping\n\n";
    std::atomic<bool> filling{true};
    std::mutex pingLock;
    std::condition_variable pingWake;
    std::thread waiting;
    if(req.stream){
        waiting = std::thread([&]{
            std::unique_lock<std::mutex> hold(pingLock);
            while(!pingWake.wait_for(hold, std::chrono::seconds(10), [&]{ return !filling.load(); })) sendText(client, ping);
        });
    }

    std::vector<float> logits;
    uint64_t start = nowMs();
    bool filled = hooks.forward(tokens, 0, logits);
    {
        std::lock_guard<std::mutex> hold(pingLock);
        filling = false;
    }
    pingWake.notify_all();
    if(waiting.joinable()) waiting.join();
    uint64_t lastSent = nowMs();
    if(!filled){
        result.finish = "error";
        if(req.stream && req.messagesApi) messageEvent("error", failText());
        if(req.stream && !req.messagesApi){
            sendEvent(client, failText());
            sendText(client, "data: [DONE]\n\n");
        }
        return result;
    }
    result.cachedTokens = hooks.reusedTokens ? hooks.reusedTokens() : 0;
    result.promptMs = (double)(nowMs() - start);
    if(notices) sendEvent(client, noticeJson(id, req.model, created, "prefill", "{\"tokens\":" + std::to_string(result.promptTokens) + ",\"cached\":" + std::to_string(result.cachedTokens) + ",\"ms\":" + std::to_string((long long)result.promptMs) + "}"));
    uint64_t toldTool = 0;
    int toolTokens = 0;

    std::mt19937_64 random(req.seed);
    OutputSplitter splitter;
    splitter.thinkOpen = hooks.thinkOpen;
    splitter.thinkClose = hooks.thinkClose;
    splitter.toolOpen = hooks.toolOpen;
    splitter.toolClose = hooks.toolClose;
    std::string opening = hooks.thinkOpen + "\n";
    splitter.inThink = prompt.size() >= opening.size() && prompt.compare(prompt.size() - opening.size(), opening.size(), opening) == 0;
    std::vector<int> recent;
    DraftIndex index;
    index.addAll(tokens);
    std::string text;
    int thinkTokens = 0;
    size_t thinkStart = index.seen.size();
    bool looping = false;
    int quiet = 0;
    int misses = 0;
    bool drafting = hooks.verify && hooks.mark && hooks.rewind && hooks.verifyLimit > 1;
    int most = std::max(1, std::min(hooks.draftLimit > 0 ? hooks.draftLimit : 8, hooks.verifyLimit - 1));
    int depth = hooks.guessStart > 0 ? std::min(most, hooks.guessStart) : most;
    int longest = std::min(hooks.copyLimit, hooks.verifyLimit - 1);
    int reach = std::min(8, longest);
    uint64_t decodeStart = nowMs();
    uint64_t toldSpeed = decodeStart;

    auto emit = [&](int token) {
        if(stopGenerating){
            result.clientGone = true;
            return 2;
        }
        result.outputTokens++;
        if(req.stream && nowMs() - lastSent > 10000){
            if(!sendText(client, ping)){
                result.clientGone = true;
                return 2;
            }
            lastSent = nowMs();
        }
        // cada medio segundo los tokens escritos y el tiempo para que la app muestre la velocidad mientras escribe
        if(notices && nowMs() - toldSpeed > 500){
            toldSpeed = nowMs();
            if(!sendEvent(client, noticeJson(id, req.model, created, "speed", "{\"tokens\":" + std::to_string(result.outputTokens) + ",\"ms\":" + std::to_string((long long)(toldSpeed - decodeStart)) + "}"))){
                result.clientGone = true;
                return 2;
            }
            lastSent = toldSpeed;
        }
        if(hooks.isStop(token)) return 1;

        recent.push_back(token);
        if(recent.size() > 64) recent.erase(recent.begin());
        index.add(token);

        std::string piece = hooks.piece(token);
        text += piece;
        if(splitter.inThink) thinkTokens++;
        if(splitter.inThink && !looping) looping = stuckThinking(index.seen, thinkStart, hooks.tokenText);

        std::string reasoning, content;
        splitter.push(piece, reasoning, content);
        if(!splitter.inTool) toolTokens = 0;
        if(notices && splitter.inTool){
            toolTokens++;
            if(toolTokens >= 8 && nowMs() - toldTool > 700){
                toldTool = nowMs();
                std::string head = splitter.toolText.substr(0, wholeChars(splitter.toolText, std::min<size_t>(splitter.toolText.size(), 400)));
                if(!sendEvent(client, noticeJson(id, req.model, created, "writing", "{\"tokens\":" + std::to_string(toolTokens) + ",\"head\":" + quoteJson(head) + "}"))){
                    result.clientGone = true;
                    return 2;
                }
                lastSent = toldTool;
            }
        }
        if(req.stream && (!reasoning.empty() || !content.empty())){
            bool sent = req.messagesApi
                ? (reasoning.empty() || (enterBlock("thinking", "{\"type\":\"thinking\",\"thinking\":\"\",\"signature\":\"\"}")
                    && messageEvent("content_block_delta", "{\"type\":\"content_block_delta\",\"index\":" + std::to_string(blockIndex) + ",\"delta\":{\"type\":\"thinking_delta\",\"thinking\":" + quoteJson(reasoning) + "}}")))
                    && (content.empty() || (enterBlock("text", "{\"type\":\"text\",\"text\":\"\"}")
                    && messageEvent("content_block_delta", "{\"type\":\"content_block_delta\",\"index\":" + std::to_string(blockIndex) + ",\"delta\":{\"type\":\"text_delta\",\"text\":" + quoteJson(content) + "}}")))
                : (reasoning.empty() || sendEvent(client, chunkJson(id, req.model, created, "{\"reasoning_content\":" + quoteJson(reasoning) + "}", nullptr)))
                    && (content.empty() || sendEvent(client, chunkJson(id, req.model, created, "{\"content\":" + quoteJson(content) + "}", nullptr)));
            if(!sent){
                result.clientGone = true;
                return 2;
            }
            lastSent = nowMs();
        }
        result.reasoning += reasoning;
        result.content += content;

        for(const std::string& stop : req.stops){
            if(stop.empty() || text.size() < stop.size() || text.compare(text.size() - stop.size(), stop.size(), stop) != 0) continue;
            result.stopHit = stop;
            return 1;
        }
        if((int)result.outputTokens >= req.maxTokens){
            result.finish = "length";
            return 1;
        }
        return 0;
    };

    int token = sampleToken(logits, req, recent, random);
    int state = emit(token);

    while(state == 0){
        bool spent = req.thinkBudget > 0 && thinkTokens >= req.thinkBudget;
        if(splitter.inThink && (spent || looping)){
            result.thinkingCut = looping ? "repeating" : "budget";
            result.thinkingTokens = thinkTokens;
            std::vector<int> next = {token};
            std::vector<int> close = hooks.encode(hooks.closeThinking);
            next.insert(next.end(), close.begin(), close.end());
            splitter.push(hooks.closeThinking, result.reasoning, result.content);
            for(int t : close) index.add(t);
            if(!hooks.forward(next, 0, logits)){
                result.finish = "error";
                break;
            }
            token = sampleToken(logits, req, recent, random);
            state = emit(token);
            continue;
        }

        std::vector<int> draft;
        int matched = 0;
        bool copied = false;
        if(drafting && quiet == 0){
            std::vector<int> copy = longest > 0 ? index.propose(reach, matched) : std::vector<int>();
            std::vector<int> guess = hooks.propose && matched < 12 ? hooks.propose(token, depth) : std::vector<int>();
            copied = !copy.empty() && (matched >= 12 || guess.empty() || (copy.size() > guess.size() && copy[0] == guess[0]));
            draft = copied ? copy : guess;
        }
        if(quiet > 0) quiet--;

        if(draft.empty() || !hooks.mark((int)draft.size() + 1)){
            uint64_t asked = nowMs();
            bool answered = hooks.forward({token}, 0, logits);
            result.modelMs += (double)(nowMs() - asked);
            if(!answered){
                result.finish = "error";
                break;
            }
            token = sampleToken(logits, req, recent, random);
            state = emit(token);
            continue;
        }

        std::vector<int> batch = {token};
        batch.insert(batch.end(), draft.begin(), draft.end());
        std::vector<float> rows;
        if(!hooks.verify(batch, rows) || rows.size() % batch.size() != 0){
            result.finish = "error";
            break;
        }
        size_t width = rows.size() / batch.size();
        result.drafted += draft.size();

        size_t kept = 0;
        int fix = -1;
        for(size_t i = 0; i < draft.size() && state == 0; i++){
            std::vector<float> row(rows.begin() + i * width, rows.begin() + (i + 1) * width);
            bool agreed = false;
            int choice = checkDraft(row, draft[i], req, recent, random, agreed);
            if(!agreed){
                fix = choice;
                break;
            }
            kept++;
            state = emit(draft[i]);
        }
        result.kept += kept;
        if(copied){
            reach = kept == draft.size()
                ? std::min(longest, reach * 2)
                : kept * 2 < draft.size()
                    ? std::max(4, reach / 2)
                    : reach;
            result.copied += draft.size();
            result.copyKept += kept;
        } else if(hooks.guessStart > 0 && !draft.empty()){
            depth = kept == draft.size()
                ? std::min(most, depth + 1)
                : kept == 0
                    ? std::max(1, depth - 1)
                    : depth;
        }
        misses = kept == 0 ? misses + 1 : 0;
        if(misses >= 3){
            quiet = 16;
            misses = 0;
        }
        if(state != 0){
            if(hooks.settle) hooks.settle(batch);
            break;
        }

        if(kept == draft.size()){
            if(hooks.settle && !hooks.settle(batch)){
                result.finish = "error";
                break;
            }
            std::vector<float> row(rows.end() - width, rows.end());
            token = sampleToken(row, req, recent, random);
            state = emit(token);
            continue;
        }

        std::vector<int> refeed(batch.begin(), batch.begin() + 1 + kept);
        if(hooks.commit && hooks.commit(refeed)){
            token = fix;
            state = emit(token);
            continue;
        }
        if(!hooks.rewind() || !hooks.forward(refeed, 0, logits)){
            result.finish = "error";
            break;
        }
        token = fix;
        state = emit(token);
    }

    result.outputMs = (double)(nowMs() - decodeStart);
    splitter.finish(result.reasoning, result.content);
    if(!result.stopHit.empty() && result.content.size() >= result.stopHit.size() && result.content.compare(result.content.size() - result.stopHit.size(), result.stopHit.size(), result.stopHit) == 0){
        result.content.erase(result.content.size() - result.stopHit.size());
    }

    result.toolCalls.kind = JSON_LIST;
    for(size_t i = 0; i < splitter.tools.size(); i++) result.toolCalls.items.push_back(parseToolText(splitter.tools[i], (int)i, req.tools));
    if(!result.toolCalls.items.empty() && result.finish == "stop") result.finish = "tool_calls";

    if(result.clientGone) return result;

    if(req.messagesApi){
        std::string stopReason =
            result.finish == "tool_calls" ? "tool_use" :
            result.finish == "length" ? "max_tokens" :
            !result.stopHit.empty() ? "stop_sequence" :
            "end_turn";
        std::string stopSequence = result.stopHit.empty() ? "null" : quoteJson(result.stopHit);
        std::string usage = "{\"input_tokens\":" + std::to_string(result.promptTokens - result.cachedTokens) + ",\"cache_read_input_tokens\":" + std::to_string(result.cachedTokens) + ",\"output_tokens\":" + std::to_string(result.outputTokens) + "}";

        if(req.stream){
            for(size_t i = 0; i < result.toolCalls.items.size(); i++){
                const Json& call = result.toolCalls.items[i];
                Json input = parseJson(call["function"]["arguments"].str());
                enterBlock("tool" + std::to_string(i), "{\"type\":\"tool_use\",\"id\":" + quoteJson(call["id"].str()) + ",\"name\":" + quoteJson(call["function"]["name"].str()) + ",\"input\":{}}");
                messageEvent("content_block_delta", "{\"type\":\"content_block_delta\",\"index\":" + std::to_string(blockIndex) + ",\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":" + quoteJson(input.kind == JSON_OBJECT ? writeJson(input) : "{}") + "}}");
            }
            if(blockIndex >= 0) messageEvent("content_block_stop", "{\"type\":\"content_block_stop\",\"index\":" + std::to_string(blockIndex) + "}");
            if(result.finish == "error"){
                messageEvent("error", failText());
                return result;
            }
            messageEvent("message_delta", "{\"type\":\"message_delta\",\"delta\":{\"stop_reason\":" + quoteJson(stopReason) + ",\"stop_sequence\":" + stopSequence + "},\"usage\":" + usage + "}");
            messageEvent("message_stop", "{\"type\":\"message_stop\"}");
            return result;
        }

        if(result.finish == "error") return result;
        std::string blocks;
        if(!result.reasoning.empty()) blocks += "{\"type\":\"thinking\",\"thinking\":" + quoteJson(result.reasoning) + ",\"signature\":\"\"}";
        if(!result.content.empty()) blocks += (blocks.empty() ? "" : ",") + std::string("{\"type\":\"text\",\"text\":") + quoteJson(result.content) + "}";
        for(const Json& call : result.toolCalls.items){
            Json input = parseJson(call["function"]["arguments"].str());
            blocks += (blocks.empty() ? "" : ",") + std::string("{\"type\":\"tool_use\",\"id\":") + quoteJson(call["id"].str()) + ",\"name\":" + quoteJson(call["function"]["name"].str()) + ",\"input\":" + (input.kind == JSON_OBJECT ? writeJson(input) : "{}") + "}";
        }
        std::string body = "{\"id\":" + quoteJson(id) + ",\"type\":\"message\",\"role\":\"assistant\",\"model\":" + quoteJson(req.model) + ",\"content\":[" + blocks + "],\"stop_reason\":" + quoteJson(stopReason) + ",\"stop_sequence\":" + stopSequence + ",\"usage\":" + usage + "}";
        sendText(client, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: " + std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n" + body);
        return result;
    }

    if(req.stream){
        if(!result.toolCalls.items.empty()){
            sendEvent(client, chunkJson(id, req.model, created, "{\"tool_calls\":" + writeJson(result.toolCalls) + "}", nullptr));
        }
        if(result.finish == "error") sendEvent(client, failText());
        sendEvent(client, chunkJson(id, req.model, created, "{}", result.finish.c_str()));
        if(req.includeUsage){
            sendEvent(client, "{\"id\":" + quoteJson(id) + ",\"object\":\"chat.completion.chunk\",\"created\":" + std::to_string(created) + ",\"model\":" + quoteJson(req.model) + ",\"choices\":[]," + usageJson(result) + "}");
        }
        sendText(client, "data: [DONE]\n\n");
        return result;
    }

    std::string message = "{\"role\":\"assistant\",\"content\":" + (result.content.empty() && !result.toolCalls.items.empty() ? std::string("null") : quoteJson(result.content));
    if(!result.reasoning.empty()) message += ",\"reasoning_content\":" + quoteJson(result.reasoning);
    if(!result.toolCalls.items.empty()) message += ",\"tool_calls\":" + writeJson(result.toolCalls);
    message += "}";

    std::string body = "{\"id\":" + quoteJson(id) + ",\"object\":\"chat.completion\",\"created\":" + std::to_string(created) + ",\"model\":" + quoteJson(req.model)
        + ",\"choices\":[{\"index\":0,\"message\":" + message + ",\"finish_reason\":" + quoteJson(result.finish) + "}]," + usageJson(result) + "}";
    std::string head = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: " + std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n";
    sendText(client, head + body);
    return result;
}
