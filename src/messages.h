#pragma once
#include "json.h"
#include <string>

inline std::string plainText(const Json& content) {
    if(content.kind == JSON_TEXT) return content.text;
    std::string out;
    for(const Json& item : content.items){
        if(item["type"].str() != "text") continue;
        if(!out.empty()) out += "\n";
        out += item["text"].str();
    }
    return out;
}

inline Json chatFromMessages(const Json& body) {
    std::string out = "{\"model\":" + quoteJson(body["model"].str("sneo")) + ",\"messages\":[";
    bool empty = true;
    auto add = [&](const std::string& message) {
        out += empty ? "" : ",";
        out += message;
        empty = false;
    };

    std::string system = plainText(body["system"]);
    if(!system.empty()) add("{\"role\":\"system\",\"content\":" + quoteJson(system) + "}");

    for(const Json& message : body["messages"].items){
        std::string role = message["role"].str() == "assistant" ? "assistant" : "user";
        const Json& content = message["content"];
        if(content.kind == JSON_TEXT){
            add("{\"role\":\"" + role + "\",\"content\":" + quoteJson(content.text) + "}");
            continue;
        }

        std::string text, reasoning, calls;
        for(const Json& block : content.items){
            std::string type = block["type"].str();
            if(type == "text"){
                text += block["text"].str();
                continue;
            }
            if(type == "thinking"){
                reasoning += block["thinking"].str();
                continue;
            }
            if(type == "tool_use"){
                std::string input = block["input"].kind == JSON_OBJECT ? writeJson(block["input"]) : "{}";
                calls += calls.empty() ? "" : ",";
                calls += "{\"id\":" + quoteJson(block["id"].str()) + ",\"type\":\"function\",\"function\":{\"name\":" + quoteJson(block["name"].str()) + ",\"arguments\":" + quoteJson(input) + "}}";
                continue;
            }
            if(type != "tool_result") continue;
            std::string result = plainText(block["content"]);
            if(block["is_error"].truthy()) result = "error: " + result;
            add("{\"role\":\"tool\",\"tool_call_id\":" + quoteJson(block["tool_use_id"].str()) + ",\"content\":" + quoteJson(result) + "}");
        }

        if(role == "user"){
            if(!text.empty()) add("{\"role\":\"user\",\"content\":" + quoteJson(text) + "}");
            continue;
        }
        std::string reply = "{\"role\":\"assistant\",\"content\":" + quoteJson(text);
        if(!reasoning.empty()) reply += ",\"reasoning_content\":" + quoteJson(reasoning);
        if(!calls.empty()) reply += ",\"tool_calls\":[" + calls + "]";
        add(reply + "}");
    }
    out += "]";

    const Json& tools = body["tools"];
    if(!tools.items.empty() && body["tool_choice"]["type"].str("auto") != "none"){
        std::string list;
        for(const Json& tool : tools.items){
            if(!tool.has("input_schema")) continue;
            list += list.empty() ? "" : ",";
            list += "{\"type\":\"function\",\"function\":{\"name\":" + quoteJson(tool["name"].str()) + ",\"description\":" + quoteJson(tool["description"].str()) + ",\"parameters\":" + writeJson(tool["input_schema"]) + "}}";
        }
        if(!list.empty()) out += ",\"tools\":[" + list + "]";
    }

    out += ",\"max_tokens\":" + std::to_string((int)body["max_tokens"].num(4096));
    for(const char* key : {"temperature", "top_p", "top_k"}){
        if(body[key].kind == JSON_NUMBER) out += std::string(",\"") + key + "\":" + writeJson(body[key]);
    }
    if(body["stop_sequences"].kind == JSON_LIST) out += ",\"stop\":" + writeJson(body["stop_sequences"]);
    out += body["stream"].truthy() ? ",\"stream\":true" : ",\"stream\":false";

    std::string thinking = body["thinking"]["type"].str();
    bool reasons = thinking == "enabled" || thinking == "adaptive";
    out += reasons ? ",\"chat_template_kwargs\":{\"enable_thinking\":true}" : ",\"chat_template_kwargs\":{\"enable_thinking\":false}";
    int budget = (int)body["thinking"]["budget_tokens"].num(0);
    if(reasons && budget > 0) out += ",\"thinking_budget\":" + std::to_string(budget);
    std::string effort = body["output_config"]["effort"].str();
    if(reasons && !effort.empty()) out += ",\"reasoning_effort\":" + quoteJson(effort == "max" ? "xhigh" : effort);
    return parseJson(out + "}");
}
