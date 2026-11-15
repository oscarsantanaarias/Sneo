#include "../src/net.h"
#include "../src/json.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <ctime>

struct Trial {
    std::string suite;
    double firstMs = 0;
    double totalMs = 0;
    uint64_t promptTokens = 0;
    uint64_t outputTokens = 0;
    double promptSpeed = 0;
    double outputSpeed = 0;
    bool ok = false;
    std::string problem;
};

std::string readSlice(const std::string& path, size_t limit) {
    std::ifstream file(path, std::ios::binary);
    std::stringstream text;
    text << file.rdbuf();
    std::string all = text.str();
    return all.size() > limit ? all.substr(0, limit) : all;
}

std::string promptFor(const std::string& suite, const std::string& root) {
    if(suite == "short") return "Write four short lines about graphics cards.";
    if(suite == "code"){
        return "Review this C++ file and list three concrete bugs or risks with the line where each one is:\n\n" + readSlice(root + "\\src\\pool.h", 4500);
    }
    if(suite == "huge"){
        return "Read these two files and explain what each one does, then name the three most important functions of each:\n\n"
            + readSlice(root + "\\master.cpp", 40000) + "\n\n" + readSlice(root + "\\src\\gpumodel.h", 40000);
    }
    return "Read these files and explain how a request flows from the user interface to the engine and back:\n\n"
        + readSlice(root + "\\master.cpp", 9000) + "\n\n" + readSlice(root + "\\src\\serve.h", 7000) + "\n\n" + readSlice(root + "\\ui\\server\\engine.js", 3000);
}

struct Sampling {
    float temperature = 0.0f;
    float topP = 1.0f;
    int topK = 0;
    bool speculate = true;
};

Trial runTrial(const std::string& host, int port, const std::string& path, const std::string& model, const std::string& suite, const std::string& root, int maxTokens, bool thinking, int context, bool thinkField, const Sampling& sampling, std::mt19937& random) {
    Trial trial;
    trial.suite = suite;

    bool chatApi = path == "/api/chat";
    std::string nonce = "Session " + std::to_string(random()) + ". ";
    std::string messages = "\"messages\":[{\"role\":\"system\",\"content\":" + quoteJson(nonce + "You are a precise assistant.") + "},{\"role\":\"user\",\"content\":" + quoteJson(promptFor(suite, root)) + "}]";
    std::string picks = "\"temperature\":" + std::to_string(sampling.temperature) + ",\"top_p\":" + std::to_string(sampling.topP) + (sampling.topK > 0 ? ",\"top_k\":" + std::to_string(sampling.topK) : std::string()) + ",\"seed\":7" + (sampling.speculate ? std::string() : std::string(",\"speculate\":false"));
    std::string body = chatApi
        ? "{\"model\":" + quoteJson(model) + ",\"stream\":true" + (thinkField ? std::string(",\"think\":") + (thinking ? "true" : "false") : std::string())
            + ",\"options\":{" + picks + ",\"num_predict\":" + std::to_string(maxTokens) + (context > 0 ? ",\"num_ctx\":" + std::to_string(context) : std::string()) + "}," + messages + "}"
        : "{\"model\":" + quoteJson(model) + ",\"stream\":true,\"stream_options\":{\"include_usage\":true}," + picks + ",\"max_tokens\":" + std::to_string(maxTokens)
            + ",\"reasoning_effort\":\"" + (thinking ? "medium" : "none") + "\",\"chat_template_kwargs\":{\"enable_thinking\":" + (thinking ? "true" : "false") + "}," + messages + "}";
    std::string request = "POST " + path + (chatApi ? " HTTP/1.0" : " HTTP/1.1") + "\r\nHost: " + host + "\r\nContent-Type: application/json\r\nContent-Length: " + std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n" + body;

    SOCKET s = connectTcp(host, port, 3000);
    if(s == INVALID_SOCKET){
        trial.problem = "cannot connect";
        return trial;
    }

    uint64_t start = nowMs();
    sendAll(s, request.data(), request.size());

    std::string pending;
    uint64_t first = 0;
    uint64_t pieces = 0;
    char buffer[16384];
    while(true){
        int got = recv(s, buffer, sizeof(buffer), 0);
        if(got <= 0) break;
        pending.append(buffer, got);
        size_t end;
        while((end = pending.find('\n')) != std::string::npos){
            std::string line = pending.substr(0, end);
            pending.erase(0, end + 1);
            if(chatApi && line.rfind("{", 0) == 0){
                Json event = parseJson(line);
                if(!event["error"].str().empty()) trial.problem = event["error"].str();
                const Json& message = event["message"];
                if(!message["content"].str().empty() || !message["thinking"].str().empty()){
                    pieces++;
                    if(!first) first = nowMs();
                }
                if(event["done"].truthy(false)){
                    trial.promptTokens = (uint64_t)event["prompt_eval_count"].num();
                    trial.outputTokens = (uint64_t)event["eval_count"].num();
                }
                continue;
            }
            if(line.rfind("data: {", 0) != 0) continue;
            Json event = parseJson(line.substr(6));
            const Json& delta = event["choices"].items.empty() ? Json() : event["choices"].items[0]["delta"];
            bool text = !delta["content"].str().empty() || !delta["reasoning_content"].str().empty();
            if(text){
                pieces++;
                if(!first) first = nowMs();
            }
            const Json& usage = event["usage"];
            if(usage.kind == JSON_OBJECT){
                trial.promptTokens = (uint64_t)usage["prompt_tokens"].num();
                trial.outputTokens = (uint64_t)usage["completion_tokens"].num();
            }
        }
    }
    closesocket(s);
    if(chatApi && trial.problem.empty() && pending.rfind("{", 0) == 0) trial.problem = parseJson(pending)["error"].str();

    uint64_t finish = nowMs();
    if(!first){
        if(trial.problem.empty()) trial.problem = "no tokens streamed";
        return trial;
    }
    if(!trial.outputTokens) trial.outputTokens = pieces;

    trial.firstMs = (double)(first - start);
    trial.totalMs = (double)(finish - start);
    double decodeMs = (double)(finish - first);
    trial.promptSpeed = trial.firstMs > 0 ? trial.promptTokens * 1000.0 / trial.firstMs : 0;
    trial.outputSpeed = decodeMs > 0 && trial.outputTokens > 1 ? (trial.outputTokens - 1) * 1000.0 / decodeMs : 0;
    trial.ok = true;
    return trial;
}

int main(int argc, char** argv) {
    std::string url = "http://127.0.0.1:8000/v1/chat/completions";
    std::string label = "sneo";
    std::string model = "sneo";
    std::string root = ".";
    std::string out = "bench\\results.jsonl";
    std::string suites = "short,code,long";
    int runs = 2;
    int maxTokens = 128;
    bool thinking = false;
    int context = 0;
    bool thinkField = true;
    Sampling sampling;

    for(int i = 1; i + 1 < argc; i++){
        std::string flag = argv[i];
        if(flag == "--url") url = argv[++i];
        else if(flag == "--label") label = argv[++i];
        else if(flag == "--model") model = argv[++i];
        else if(flag == "--root") root = argv[++i];
        else if(flag == "--out") out = argv[++i];
        else if(flag == "--suites") suites = argv[++i];
        else if(flag == "--runs") runs = atoi(argv[++i]);
        else if(flag == "--max") maxTokens = atoi(argv[++i]);
        else if(flag == "--thinking") thinking = std::string(argv[++i]) == "on";
        else if(flag == "--context") context = atoi(argv[++i]);
        else if(flag == "--temperature") sampling.temperature = (float)atof(argv[++i]);
        else if(flag == "--top-p") sampling.topP = (float)atof(argv[++i]);
        else if(flag == "--top-k") sampling.topK = atoi(argv[++i]);
        else if(flag == "--speculate") sampling.speculate = std::string(argv[++i]) != "off";
    }

    size_t scheme = url.find("://");
    std::string rest = scheme == std::string::npos ? url : url.substr(scheme + 3);
    size_t slash = rest.find('/');
    std::string hostPort = rest.substr(0, slash);
    std::string path = slash == std::string::npos ? "/v1/chat/completions" : rest.substr(slash);
    size_t colon = hostPort.find(':');
    std::string host = hostPort.substr(0, colon);
    int port = colon == std::string::npos ? 80 : atoi(hostPort.c_str() + colon + 1);

    if(!startSockets()){
        std::cout << "winsock failed" << std::endl;
        return 1;
    }

    std::mt19937 random((unsigned)time(nullptr));
    std::ofstream log(out, std::ios::app);
    std::vector<std::string> list;
    std::stringstream split(suites);
    std::string item;
    while(std::getline(split, item, ',')) if(!item.empty()) list.push_back(item);

    printf("%-10s %-6s %8s %10s %12s %10s %12s\n", "engine", "suite", "prompt", "first ms", "prompt t/s", "output", "output t/s");
    for(const std::string& suite : list){
        for(int r = 0; r < runs; r++){
            Trial t = runTrial(host, port, path, model, suite, root, maxTokens, thinking, context, thinkField, sampling, random);
            if(!t.ok && thinkField && t.problem.find("think") != std::string::npos){
                thinkField = false;
                t = runTrial(host, port, path, model, suite, root, maxTokens, thinking, context, thinkField, sampling, random);
            }
            if(!t.ok){
                printf("%-10s %-6s failed: %s\n", label.c_str(), suite.c_str(), t.problem.c_str());
                continue;
            }
            printf("%-10s %-6s %8llu %10.0f %12.1f %10llu %12.2f\n", label.c_str(), suite.c_str(), (unsigned long long)t.promptTokens, t.firstMs, t.promptSpeed, (unsigned long long)t.outputTokens, t.outputSpeed);
            log << "{\"when\":" << time(nullptr) << ",\"engine\":" << quoteJson(label) << ",\"suite\":" << quoteJson(suite) << ",\"promptTokens\":" << t.promptTokens
                << ",\"firstMs\":" << t.firstMs << ",\"promptSpeed\":" << t.promptSpeed << ",\"outputTokens\":" << t.outputTokens << ",\"outputSpeed\":" << t.outputSpeed
                << ",\"totalMs\":" << t.totalMs << "}" << std::endl;
        }
    }
    return 0;
}
