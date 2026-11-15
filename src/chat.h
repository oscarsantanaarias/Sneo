#pragma once
#include "tokenizer.h"
#include "json.h"
#include <optional>
#include <cmath>

struct ChatOptions {
    std::optional<bool> thinking;
    std::optional<bool> keepThinking;
    std::string effort;
    bool generation = true;
    bool effortNotes = true;
    bool keepByDefault = true;
    bool gemma = false;
};

struct ToolCall {
    std::string name;
    Json arguments;
    bool complete = false;
};

struct ChatReply {
    std::string reasoning;
    std::string content;
    std::vector<ToolCall> calls;
    bool thinkingClosed = false;
};

inline std::string trimText(const std::string& s) {
    auto blank = [](uint32_t c){
        return (c >= 0x09 && c <= 0x0D) || (c >= 0x1C && c <= 0x20) || c == 0x85 || c == 0xA0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
    };

    size_t begin = 0;
    size_t end = s.size();
    while(begin < end){
        size_t next = begin;
        if(!blank(readCodepoint(s, next))) break;
        begin = next;
    }

    while(end > begin){
        size_t back = end - 1;
        while(back > begin && ((uint8_t)s[back] & 0xC0) == 0x80) back--;
        size_t probe = back;
        uint32_t c = readCodepoint(s, probe);
        if(probe != end || !blank(c)) break;
        end = back;
    }

    return s.substr(begin, end - begin);
}

inline std::string quotePython(const std::string& s) {
    std::string out = "\"";
    for(unsigned char c : s){
        if(c == '"') out += "\\\"";
        else if(c == '\\') out += "\\\\";
        else if(c == '\n') out += "\\n";
        else if(c == '\r') out += "\\r";
        else if(c == '\t') out += "\\t";
        else if(c == '\b') out += "\\b";
        else if(c == '\f') out += "\\f";
        else if(c < 0x20){
            char code[8];
            snprintf(code, sizeof(code), "\\u%04x", c);
            out += code;
        } else out.push_back((char)c);
    }
    out.push_back('"');
    return out;
}

inline std::string dumpPython(const Json& v) {
    if(v.kind == JSON_NULL) return "null";
    if(v.kind == JSON_BOOL) return v.flag ? "true" : "false";
    if(v.kind == JSON_TEXT) return quotePython(v.text);

    if(v.kind == JSON_NUMBER){
        char buffer[40];
        if(v.number == std::floor(v.number) && std::fabs(v.number) < 1e16){
            snprintf(buffer, sizeof(buffer), "%.0f", v.number);
            return buffer;
        }
        for(int digits = 1; digits <= 17; digits++){
            snprintf(buffer, sizeof(buffer), "%.*g", digits, v.number);
            if(strtod(buffer, nullptr) == v.number) break;
        }
        return buffer;
    }

    if(v.kind == JSON_LIST){
        std::string out = "[";
        for(size_t i = 0; i < v.items.size(); i++) out += (i ? ", " : "") + dumpPython(v.items[i]);
        return out + "]";
    }

    std::string out = "{";
    for(size_t i = 0; i < v.fields.size(); i++) out += (i ? ", " : "") + quotePython(v.fields[i].first) + ": " + dumpPython(v.fields[i].second);
    return out + "}";
}

inline bool renderContent(const Json& content, bool system, std::string& out, std::string& error) {
    out.clear();
    if(content.kind == JSON_TEXT){
        out = content.text;
        return true;
    }

    if(content.kind == JSON_NULL) return true;

    if(content.kind != JSON_LIST){
        error = "Unexpected content type.";
        return false;
    }

    for(const Json& item : content.items){
        std::string type = item["type"].str();
        if(item.has("image") || item.has("image_url") || type == "image"){
            if(system){
                error = "System message cannot contain images.";
                return false;
            }
            out += "<|vision_start|><|image_pad|><|vision_end|>";
        } else if(item.has("video") || type == "video"){
            if(system){
                error = "System message cannot contain videos.";
                return false;
            }
            out += "<|vision_start|><|video_pad|><|vision_end|>";
        } else if(item.has("text")){
            out += item["text"].str();
        } else {
            error = "Unexpected item type in content.";
            return false;
        }
    }
    return true;
}

inline ChatOptions chatOptions(const Json& request) {
    ChatOptions options;
    const Json& extra = request["chat_template_kwargs"];

    if(extra["enable_thinking"].kind == JSON_BOOL) options.thinking = extra["enable_thinking"].flag;
    if(request["enable_thinking"].kind == JSON_BOOL) options.thinking = request["enable_thinking"].flag;
    if(extra["preserve_thinking"].kind == JSON_BOOL) options.keepThinking = extra["preserve_thinking"].flag;
    if(request["add_generation_prompt"].kind == JSON_BOOL) options.generation = request["add_generation_prompt"].flag;

    std::string effort = request["reasoning_effort"].str(request["reasoning"]["effort"].str(extra["reasoning_effort"].str()));
    if(effort == "none" || effort == "minimal" || effort == "off"){
        options.thinking = false;
        return options;
    }

    options.effort =
        effort == "max" ? "xhigh" :
        effort;
    return options;
}

inline std::string gemmaQuote(const std::string& text) {
    return "<|\"|>" + text + "<|\"|>";
}

inline std::string upperText(std::string text) {
    for(char& c : text) c = (char)toupper((unsigned char)c);
    return text;
}

inline std::vector<std::pair<std::string, Json>> sortedFields(const Json& object) {
    std::vector<std::pair<std::string, Json>> fields = object.fields;
    std::stable_sort(fields.begin(), fields.end(), [](const auto& a, const auto& b){
        std::string x = a.first, y = b.first;
        for(char& c : x) c = (char)tolower((unsigned char)c);
        for(char& c : y) c = (char)tolower((unsigned char)c);
        return x < y;
    });
    return fields;
}

inline std::string gemmaValue(const Json& v, bool quoteKeys = true) {
    if(v.kind == JSON_NULL) return "null";
    if(v.kind == JSON_TEXT) return gemmaQuote(v.text);
    if(v.kind == JSON_BOOL) return v.flag ? "true" : "false";
    if(v.kind == JSON_NUMBER) return dumpPython(v);
    if(v.kind == JSON_LIST){
        std::string out = "[";
        for(size_t i = 0; i < v.items.size(); i++) out += (i ? "," : "") + gemmaValue(v.items[i], quoteKeys);
        return out + "]";
    }
    std::string out = "{";
    bool first = true;
    for(const auto& field : sortedFields(v)){
        out += (first ? "" : ",") + (quoteKeys ? gemmaQuote(field.first) : field.first) + ":" + gemmaValue(field.second, quoteKeys);
        first = false;
    }
    return out + "}";
}

inline std::string gemmaRequired(const Json& list) {
    std::string out = "required:[";
    for(size_t i = 0; i < list.items.size(); i++) out += (i ? "," : "") + gemmaQuote(list.items[i].str());
    return out + "]";
}

inline std::string gemmaParameters(const Json& properties, bool filterKeys = false) {
    static const char* standard[] = {"description", "type", "properties", "required", "nullable"};
    std::string out;
    bool found = false;
    for(const auto& field : sortedFields(properties)){
        const std::string& key = field.first;
        const Json& value = field.second;
        bool reserved = false;
        for(const char* s : standard) reserved = reserved || key == s;
        if(filterKeys && reserved) continue;
        if(found) out += ",";
        found = true;
        out += key + ":{";
        bool comma = false;
        auto separate = [&]{
            if(comma) out += ",";
            comma = true;
        };
        if(!value["description"].str().empty()){
            out += "description:" + gemmaQuote(value["description"].str());
            comma = true;
        }
        std::string type = upperText(value["type"].str());
        if(type == "STRING" && value["enum"].kind == JSON_LIST && !value["enum"].items.empty()){
            separate();
            out += "enum:" + gemmaValue(value["enum"]);
        } else if(type == "ARRAY" && value["items"].kind == JSON_OBJECT && !value["items"].fields.empty()){
            separate();
            out += "items:{";
            bool inner = false;
            for(const auto& item : sortedFields(value["items"])){
                if(item.second.kind == JSON_NULL) continue;
                if(inner) out += ",";
                inner = true;
                if(item.first == "properties") out += "properties:{" + (item.second.kind == JSON_OBJECT ? gemmaParameters(item.second) : std::string()) + "}";
                else if(item.first == "required") out += gemmaRequired(item.second);
                else if(item.first == "type"){
                    if(item.second.kind == JSON_TEXT) out += "type:" + gemmaQuote(upperText(item.second.text));
                    else {
                        Json upper = item.second;
                        for(Json& u : upper.items) u.text = upperText(u.text);
                        out += "type:" + gemmaValue(upper);
                    }
                } else out += item.first + ":" + gemmaValue(item.second);
            }
            out += "}";
        }
        if(value["nullable"].truthy(false)){
            separate();
            out += "nullable:true";
        }
        if(type == "OBJECT"){
            separate();
            out += "properties:{" + (value["properties"].kind == JSON_OBJECT ? gemmaParameters(value["properties"]) : gemmaParameters(value, true)) + "}";
            if(value["required"].kind == JSON_LIST && !value["required"].items.empty()){
                out += ",";
                out += gemmaRequired(value["required"]);
            }
        }
        separate();
        out += "type:" + gemmaQuote(type) + "}";
    }
    return out;
}

inline std::string gemmaDeclaration(const Json& tool) {
    const Json& function = tool.has("function") ? tool["function"] : tool;
    std::string out = "declaration:" + function["name"].str() + "{description:" + gemmaQuote(function["description"].str());
    const Json& parameters = function.has("parameters") ? function["parameters"] : function["input_schema"];
    if(parameters.kind == JSON_OBJECT && !parameters.fields.empty()){
        out += ",parameters:{";
        if(parameters["properties"].kind == JSON_OBJECT && !parameters["properties"].fields.empty()) out += "properties:{" + gemmaParameters(parameters["properties"]) + "},";
        if(parameters["required"].kind == JSON_LIST && !parameters["required"].items.empty()) out += gemmaRequired(parameters["required"]) + ",";
        if(!parameters["type"].str().empty()) out += "type:" + gemmaQuote(upperText(parameters["type"].str())) + "}";
    }
    return out + "}";
}

inline std::string gemmaStripThinking(const std::string& text) {
    std::string out;
    size_t at = 0;
    while(at <= text.size()){
        size_t close = text.find("<channel|>", at);
        std::string part = text.substr(at, close == std::string::npos ? std::string::npos : close - at);
        size_t open = part.find("<|channel>");
        out += open == std::string::npos ? part : part.substr(0, open);
        if(close == std::string::npos) break;
        at = close + 10;
    }
    return trimText(out);
}

inline Json gemmaArguments(const Json& call) {
    Json arguments = call["arguments"];
    if(arguments.kind == JSON_TEXT){
        Json parsed = parseJson(arguments.text);
        if(parsed.kind == JSON_OBJECT) return parsed;
        Json wrapped;
        wrapped.kind = JSON_OBJECT;
        if(!trimText(arguments.text).empty()) wrapped.fields.emplace_back("arguments", arguments);
        return wrapped;
    }
    return arguments;
}

inline bool renderGemma(const Json& messages, const Json& tools, const ChatOptions& options, std::string& prompt, std::string& error) {
    prompt = "<bos>";
    if(messages.kind != JSON_LIST || messages.items.empty()){
        error = "No messages provided.";
        return false;
    }
    const std::vector<Json>& list = messages.items;
    bool thinking = options.thinking.value_or(false);
    bool keep = options.keepThinking.value_or(false);
    bool hasTools = tools.kind == JSON_LIST && !tools.items.empty();
    std::string role0 = list[0]["role"].str();
    bool system = role0 == "system" || role0 == "developer";
    std::string previousType;
    size_t begin = 0;
    if(thinking || hasTools || system){
        prompt += "<|turn>system\n";
        if(thinking){
            prompt += "<|think|>\n";
            previousType = "think";
        }
        if(system){
            const Json& content = list[0]["content"];
            if(content.kind == JSON_TEXT) prompt += trimText(content.text);
            for(const Json& item : content.items) prompt += trimText(item["text"].str()) + " ";
            begin = 1;
        }
        if(hasTools){
            for(const Json& tool : tools.items) prompt += "<|tool>" + trimText(gemmaDeclaration(tool)) + "<tool|>";
            previousType = "tool";
        }
        prompt += "<turn|>\n";
    }

    int lastUser = -1;
    for(size_t i = begin; i < list.size(); i++){
        if(list[i]["role"].str() == "user") lastUser = (int)i;
    }

    std::string previousRole;
    for(size_t i = begin; i < list.size(); i++){
        const Json& message = list[i];
        std::string original = message["role"].str();
        if(original == "tool") continue;
        previousType.clear();
        std::string role = original == "assistant" ? "model" : original;
        if(!(role == "model" && previousRole == "assistant")) prompt += "<|turn>" + role + "\n";

        std::string reasoning = message["reasoning"].str(message["reasoning_content"].str());
        const Json& calls = message["tool_calls"];
        bool hasCalls = calls.kind == JSON_LIST && !calls.items.empty();
        if(!reasoning.empty() && ((int)i > lastUser || (keep && hasCalls))) prompt += "<|channel>thought\n" + reasoning + "\n<channel|>";

        for(size_t c = 0; hasCalls && c < calls.items.size(); c++){
            const Json& call = calls.items[c].has("function") ? calls.items[c]["function"] : calls.items[c];
            Json arguments = gemmaArguments(call);
            prompt += "<|tool_call>call:" + call["name"].str() + "{";
            bool first = true;
            for(const auto& field : sortedFields(arguments)){
                prompt += (first ? "" : ",") + field.first + ":" + gemmaValue(field.second, false);
                first = false;
            }
            prompt += "}<tool_call|>";
            previousType = "tool_call";
        }

        bool answered = false;
        if(hasCalls){
            for(size_t k = i + 1; k < list.size() && list[k]["role"].str() == "tool"; k++){
                const Json& follow = list[k];
                std::string name = follow["name"].str("unknown");
                for(const Json& item : calls.items){
                    if(item["id"].str() == follow["tool_call_id"].str() && !follow["tool_call_id"].str().empty()) name = item["function"]["name"].str(name);
                }
                std::string body;
                if(follow["content"].kind == JSON_TEXT) body = follow["content"].text;
                for(const Json& part : follow["content"].items){
                    if(part["type"].str() == "text") body += part["text"].str();
                }
                Json value;
                value.kind = JSON_TEXT;
                value.text = body;
                prompt += "<|tool_response>response:" + name + "{value:" + gemmaValue(value, false) + "}<tool_response|>";
                answered = true;
                previousType = "tool_response";
            }
        }

        std::string content;
        if(message["content"].kind == JSON_TEXT) content = role == "model" ? gemmaStripThinking(message["content"].text) : trimText(message["content"].text);
        for(const Json& item : message["content"].items){
            if(item["type"].str() == "text") content += role == "model" ? gemmaStripThinking(item["text"].str()) : trimText(item["text"].str());
        }
        prompt += content;
        bool hasContent = !trimText(content).empty();

        std::string nextRole;
        for(size_t j = i + 1; j < list.size(); j++){
            if(list[j]["role"].str() == "tool") continue;
            nextRole = list[j]["role"].str();
            break;
        }
        bool continues = role == "model" && nextRole == "assistant" && (!hasCalls || answered);
        if(previousType == "tool_call" && !answered) prompt += "<|tool_response>";
        else if(continues){}
        else if(!(answered && !hasContent && nextRole.empty())) prompt += "<turn|>\n";
        previousRole = original;
    }

    if(!options.generation) return true;
    if(previousType != "tool_response" && previousType != "tool_call"){
        prompt += "<|turn>model\n";
        if(!thinking) prompt += "<|channel>thought\n<channel|>";
    } else if(previousType == "tool_response" && thinking) prompt += "<|channel>thought\n";
    return true;
}

inline bool renderChat(const Json& messages, const Json& tools, const ChatOptions& options, std::string& prompt, std::string& error) {
    if(options.gemma) return renderGemma(messages, tools, options, prompt, error);
    prompt.clear();

    if(messages.kind != JSON_LIST || messages.items.empty()){
        error = "No messages provided.";
        return false;
    }

    const std::vector<Json>& list = messages.items;
    size_t systemCount = 0;
    std::string merged;
    for(size_t i = 0; i < list.size(); i++){
        std::string role = list[i]["role"].str();
        if(role != "system" && role != "developer") break;

        std::string text;
        if(!renderContent(list[i]["content"], true, text, error)) return false;
        text = trimText(text);
        if(!text.empty()) merged += (merged.empty() ? "" : "\n") + text;
        systemCount++;
    }

    std::string instructions;
    if(options.thinking.value_or(true) && options.effortNotes){
        std::string effort = options.effort.empty() ? "xhigh" : options.effort;
        if(effort == "high") effort = "xhigh";

        if(effort != "xhigh" && effort != "medium" && effort != "low"){
            error = "Unexpected reasoning effort " + options.effort + ". Supported types are xhigh (default), medium, and low.";
            return false;
        }

        instructions =
            effort == "xhigh" ? "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer." :
            effort == "low" ? "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration." :
            "";
    }

    if(tools.kind == JSON_LIST && !tools.items.empty()){
        prompt += "<|im_start|>system\n";
        if(!instructions.empty()) prompt += instructions + "\n\n";
        prompt += "# Tools\n\nYou have access to the following functions:\n\n<tools>";
        for(const Json& tool : tools.items) prompt += "\n" + dumpPython(tool);
        prompt += "\n</tools>";
        prompt += "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";
        if(!merged.empty()) prompt += "\n\n" + merged;
        prompt += "<|im_end|>\n";
    } else if(!merged.empty()){
        prompt += "<|im_start|>system\n" + (instructions.empty() ? "" : instructions + "\n\n") + merged + "<|im_end|>\n";
    } else if(!instructions.empty()){
        prompt += "<|im_start|>system\n" + instructions + "<|im_end|>\n";
    }

    size_t lastQuery = list.size() - 1;
    for(size_t back = list.size(); back-- > 0;){
        if(list[back]["role"].str() != "user") continue;

        std::string text;
        if(!renderContent(list[back]["content"], false, text, error)) return false;
        text = trimText(text);

        bool wrapped = text.size() >= 31 && text.compare(0, 15, "<tool_response>") == 0 && text.compare(text.size() - 16, 16, "</tool_response>") == 0;
        if(!wrapped){
            lastQuery = back;
            break;
        }
    }

    for(size_t i = systemCount; i < list.size(); i++){
        const Json& message = list[i];
        std::string role = message["role"].str();

        std::string content;
        if(!renderContent(message["content"], false, content, error)) return false;
        content = trimText(content);

        if(role == "system" || role == "developer"){
            error = "System message must be at the beginning.";
            return false;
        }

        if(role == "user"){
            prompt += "<|im_start|>user\n" + content + "<|im_end|>\n";
            continue;
        }

        if(role == "tool"){
            if(i > 0 && list[i - 1]["role"].str() != "tool") prompt += "<|im_start|>user";
            prompt += "\n<tool_response>\n" + content + "\n</tool_response>";
            if(i + 1 == list.size() || list[i + 1]["role"].str() != "tool") prompt += "<|im_end|>\n";
            continue;
        }

        if(role != "assistant"){
            error = "Unexpected message role.";
            return false;
        }

        std::string reasoning = trimText(message["reasoning_content"].str());
        if(options.keepThinking.value_or(options.keepByDefault) || i > lastQuery) prompt += "<|im_start|>assistant\n<think>\n" + reasoning + "\n</think>\n\n" + content;
        else prompt += "<|im_start|>assistant\n" + content;

        const Json& calls = message["tool_calls"];
        for(size_t c = 0; calls.kind == JSON_LIST && c < calls.items.size(); c++){
            const Json& call = calls.items[c].has("function") ? calls.items[c]["function"] : calls.items[c];
            if(call["name"].kind != JSON_TEXT){
                error = "Tool call is missing a function name.";
                return false;
            }

            prompt +=
                c > 0 ? "\n<tool_call>\n<function=" :
                !content.empty() ? "\n\n<tool_call>\n<function=" :
                "<tool_call>\n<function=";
            prompt += call["name"].text + ">\n";

            Json arguments = call["arguments"];
            if(arguments.kind == JSON_TEXT && !trimText(arguments.text).empty()){
                Json parsed = parseJson(arguments.text);
                if(parsed.kind != JSON_OBJECT){
                    parsed = Json();
                    parsed.kind = JSON_OBJECT;
                    parsed.fields.emplace_back("arguments", arguments);
                }
                arguments = parsed;
            }

            if(arguments.kind == JSON_LIST || arguments.kind == JSON_NUMBER || arguments.kind == JSON_BOOL){
                error = "Tool call arguments for function \"" + call["name"].text + "\" must be an object/mapping or a JSON string.";
                return false;
            }

            for(const auto& field : arguments.fields){
                prompt += "<parameter=" + field.first + ">\n";
                prompt += field.second.kind == JSON_TEXT ? field.second.text : dumpPython(field.second);
                prompt += "\n</parameter>\n";
            }
            prompt += "</function>\n</tool_call>";
        }
        prompt += "<|im_end|>\n";
    }

    if(options.generation){
        prompt += "<|im_start|>assistant\n";
        prompt += options.thinking.value_or(true) ? "<think>\n" : "<think>\n\n</think>\n\n";
    }
    return true;
}

inline ChatReply readReply(const std::string& text, bool thinking, const Json& tools, bool done) {
    ChatReply reply;
    size_t limit = text.size();

    if(!done){
        for(const char* tag : {"</think>", "<tool_call>", "<function="}){
            size_t length = strlen(tag);
            for(size_t k = std::min(length - 1, text.size()); k > 0; k--){
                if(text.compare(text.size() - k, k, tag, k) == 0){
                    limit = std::min(limit, text.size() - k);
                    break;
                }
            }
        }
    }

    std::string body = text.substr(0, limit);
    size_t at = 0;
    size_t lead = body.find_first_not_of(" \t\r\n");
    if(lead != std::string::npos && body.compare(lead, 7, "<think>") == 0){
        thinking = true;
        at = lead + 7;
    }

    if(thinking){
        size_t close = body.find("</think>", at);
        size_t early = body.find("<tool_call>", at);

        if(close != std::string::npos){
            reply.reasoning = trimText(body.substr(at, close - at));
            reply.thinkingClosed = true;
            at = close + 8;
        } else {
            reply.reasoning = trimText(body.substr(at, early == std::string::npos ? std::string::npos : early - at));
            if(early == std::string::npos || !done) return reply;
            at = early;
        }
    }

    while(at < body.size()){
        size_t wrapped = body.find("<tool_call>", at);
        size_t bare = body.find("<function=", at);
        size_t start = std::min(wrapped, bare);

        std::string piece = trimText(body.substr(at, start == std::string::npos ? std::string::npos : start - at));
        if(!piece.empty()) reply.content += (reply.content.empty() ? "" : "\n\n") + piece;
        if(start == std::string::npos) break;

        bool open = start == wrapped;
        size_t inner = open ? start + 11 : start;
        size_t finish = body.find(open ? "</tool_call>" : "</function>", inner);
        size_t again = open ? body.find("<tool_call>", inner) : std::string::npos;
        bool closed = finish != std::string::npos && finish < again;
        size_t stop = std::min(finish, again);

        std::string chunk = body.substr(inner, stop == std::string::npos ? std::string::npos : stop - inner);
        at =
            stop == std::string::npos ? body.size() :
            closed ? finish + (open ? 12 : 11) :
            stop;

        ToolCall call;
        call.arguments.kind = JSON_OBJECT;
        size_t first = chunk.find_first_not_of(" \t\r\n");

        if(first != std::string::npos && chunk[first] == '{'){
            Json object = parseJson(chunk.substr(first));
            if(object.kind != JSON_OBJECT || object["name"].kind != JSON_TEXT) continue;

            call.name = object["name"].text;
            Json arguments = object.has("arguments") ? object["arguments"] : object["parameters"];
            if(arguments.kind == JSON_TEXT) arguments = parseJson(arguments.text);
            if(arguments.kind == JSON_OBJECT) call.arguments = arguments;
            call.complete = true;
            reply.calls.push_back(call);
            continue;
        }

        size_t named = chunk.find("<function=");
        size_t nameEnd = named == std::string::npos ? std::string::npos : chunk.find('>', named);
        if(nameEnd == std::string::npos) continue;

        call.name = trimText(chunk.substr(named + 10, nameEnd - named - 10));
        if(call.name.empty()) continue;

        const Json* properties = nullptr;
        for(const Json& tool : tools.items){
            const Json& function = tool.has("function") ? tool["function"] : tool;
            if(function["name"].str() == call.name) properties = &function["parameters"]["properties"];
        }

        size_t functionEnd = chunk.find("</function>", nameEnd);
        std::string block = chunk.substr(nameEnd + 1, functionEnd == std::string::npos ? std::string::npos : functionEnd - nameEnd - 1);
        call.complete = closed || functionEnd != std::string::npos;

        size_t cursor = 0;
        while(true){
            size_t opened = block.find("<parameter=", cursor);
            if(opened == std::string::npos) break;

            size_t keyEnd = block.find('>', opened);
            if(keyEnd == std::string::npos){
                call.complete = false;
                break;
            }

            std::string key = trimText(block.substr(opened + 11, keyEnd - opened - 11));
            size_t closing = block.find("</parameter>", keyEnd);
            size_t following = block.find("<parameter=", keyEnd);
            size_t valueEnd = std::min(closing, following);
            std::string value = block.substr(keyEnd + 1, valueEnd == std::string::npos ? std::string::npos : valueEnd - keyEnd - 1);
            cursor =
                valueEnd == std::string::npos ? block.size() :
                valueEnd == closing ? closing + 12 :
                following;

            if(!value.empty() && value.front() == '\n') value.erase(0, 1);
            if(!value.empty() && value.back() == '\n') value.pop_back();

            std::string type;
            const Json& schema = properties ? (*properties)[key]["type"] : Json();
            if(schema.kind == JSON_TEXT) type = schema.text;
            for(const Json& option : schema.items){
                if(type.empty() || option.str() == "string") type = option.str();
            }

            Json parsed;
            size_t used = 0;
            bool structured = type != "string" && readJson(value, used, parsed);
            if(structured) skipSpace(value, used);
            structured = structured && used == value.size() && (!type.empty() || parsed.kind != JSON_TEXT);

            if(!structured){
                parsed = Json();
                parsed.kind = JSON_TEXT;
                parsed.text = value;
            }
            call.arguments.fields.emplace_back(key, parsed);
        }

        reply.calls.push_back(call);
    }

    return reply;
}
