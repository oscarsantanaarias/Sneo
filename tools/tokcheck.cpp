#include "../src/chat.h"
#include <fstream>
#include <sstream>
#include <chrono>

using Clock = std::chrono::steady_clock;

double secondsFrom(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

bool sameIds(const std::vector<int>& ours, const Json& theirs) {
    if(ours.size() != theirs.items.size()) return false;
    for(size_t i = 0; i < ours.size(); i++){
        if(ours[i] != (int)theirs.items[i].number) return false;
    }
    return true;
}

std::string describeCalls(const ChatReply& reply) {
    std::string out;
    for(const ToolCall& call : reply.calls){
        if(!out.empty()) out += "; ";
        out += call.name + " " + dumpPython(call.arguments) + (call.complete ? " complete" : " partial");
    }
    return out;
}

void showMismatch(const Tokenizer& tok, const std::string& text, const std::vector<int>& ours, const Json& theirs) {
    size_t at = 0;
    while(at < ours.size() && at < theirs.items.size() && ours[at] == (int)theirs.items[at].number) at++;

    printf("  text %s\n", quoteJson(text.substr(0, 160)).c_str());
    printf("  first difference at token %zu (ours %zu tokens, reference %zu)\n", at, ours.size(), theirs.items.size());
    std::string mine, other;
    for(size_t i = at; i < ours.size() && i < at + 6; i++) mine += quoteJson(tok.tokenText(ours[i])) + "(" + std::to_string(ours[i]) + ") ";
    for(size_t i = at; i < theirs.items.size() && i < at + 6; i++){
        int id = (int)theirs.items[i].number;
        other += quoteJson(tok.tokenText(id)) + "(" + std::to_string(id) + ") ";
    }
    printf("  ours      %s\n  reference %s\n", mine.c_str(), other.c_str());
}

int main(int argc, char** argv) {
    std::string modelPath = argc > 1 ? argv[1] : "modelos\\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf";
    std::string referencePath = argc > 2 ? argv[2] : "build\\tokref.json";

    Clock::time_point started = Clock::now();
    ModelFile model;
    std::string error;
    if(!openModel(modelPath, model, error)){
        printf("cannot open model: %s\n", error.c_str());
        return 1;
    }
    double metaTime = secondsFrom(started);

    started = Clock::now();
    Tokenizer tok;
    if(!tok.load(model, error)){
        printf("tokenizer failed: %s\n", error.c_str());
        return 1;
    }
    printf("tokenizer: %zu tokens, %zu merges, %zu specials, eos %d bos %d pad %d add_bos %d, metadata %.2fs, load %.2fs\n",
        tok.tokens.size(), tok.merges.size(), tok.specials.size(), tok.eos, tok.bos, tok.pad, (int)tok.addBos, metaTime, secondsFrom(started));
    printf("eos text %s, think %d, tool_call %d\n", quoteJson(tok.tokenText(tok.eos)).c_str(), tok.find("<think>"), tok.find("<tool_call>"));

    std::ifstream file(referencePath, std::ios::binary);
    if(!file){
        printf("cannot read %s, run tools\\tokref.py first\n", referencePath.c_str());
        return 1;
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    Json reference = parseJson(buffer.str());
    const Json& cases = reference["cases"];
    const Json& chats = reference["chats"];
    if(cases.items.empty()){
        printf("reference file has no cases\n");
        return 1;
    }

    size_t fullMatch = 0, plainMatch = 0, roundTrip = 0, streamTrip = 0, shown = 0, totalTokens = 0;
    for(const Json& item : cases.items){
        const std::string& text = item["text"].text;
        std::vector<int> ids = tok.encode(text, true);
        std::vector<int> plain = tok.encode(text, false);
        totalTokens += ids.size();

        bool full = sameIds(ids, item["ids"]);
        bool loose = sameIds(plain, item["plain"]);
        fullMatch += full;
        plainMatch += loose;
        roundTrip += tok.decode(ids) == text && tok.decode(plain) == text;

        TextStream stream;
        std::string streamed;
        bool clean = true;
        for(int id : ids){
            std::string part = stream.push(tok.tokenText(id));
            size_t at = 0;
            while(at < part.size()){
                size_t from = at;
                if(readCodepoint(part, at) == 0xFFFD && part.compare(from, 3, "\xEF\xBF\xBD") != 0) clean = false;
            }
            streamed += part;
        }
        streamed += stream.finish();
        streamTrip += clean && streamed == text;

        if((!full || !loose) && shown < 8){
            shown++;
            printf("MISMATCH (%s)\n", full ? "plain mode" : "special mode");
            if(!full) showMismatch(tok, text, ids, item["ids"]);
            else showMismatch(tok, text, plain, item["plain"]);
        }
    }

    size_t total = cases.items.size();
    printf("\nencode vs reference (HF tokenizers ByteLevel BPE, same split regex):\n");
    printf("  special tokens on : %zu / %zu exact (%.2f%%)\n", fullMatch, total, 100.0 * fullMatch / total);
    printf("  special tokens off: %zu / %zu exact (%.2f%%)\n", plainMatch, total, 100.0 * plainMatch / total);
    printf("  decode round trip : %zu / %zu\n", roundTrip, total);
    printf("  streamed decode   : %zu / %zu (only whole characters per step)\n", streamTrip, total);

    std::string corpus;
    for(const Json& item : cases.items) corpus += item["text"].text;
    {
        std::lock_guard<std::mutex> hold(tok.cacheLock);
        tok.cache.clear();
    }
    started = Clock::now();
    size_t coldCount = tok.encode(corpus, true).size();
    double cold = secondsFrom(started);
    started = Clock::now();
    size_t warmCount = tok.encode(corpus, true).size();
    double warm = secondsFrom(started);
    printf("  speed: %.2f MB -> %zu tokens, cold %.3fs (%.0f tok/s), warm %.3fs (%.0f tok/s)\n",
        corpus.size() / 1048576.0, coldCount, cold, coldCount / cold, warm, warmCount / warm);

    size_t chatOk = 0, chatIds = 0, chatTotal = 0;
    for(size_t c = 0; c < chats.items.size(); c++){
        const Json& chat = chats.items[c];
        const Json& given = chat["options"];
        ChatOptions options;
        if(given["enable_thinking"].kind == JSON_BOOL) options.thinking = given["enable_thinking"].flag;
        if(given["preserve_thinking"].kind == JSON_BOOL) options.keepThinking = given["preserve_thinking"].flag;
        if(given["add_generation_prompt"].kind == JSON_BOOL) options.generation = given["add_generation_prompt"].flag;
        options.effort = given["reasoning_effort"].str();

        std::string prompt, problem;
        bool rendered = renderChat(chat["messages"], chat["tools"], options, prompt, problem);
        chatTotal++;

        if(chat.has("error")){
            bool agree = !rendered && problem == chat["error"].text;
            chatOk += agree;
            chatIds += agree;
            if(!agree) printf("CHAT %zu: reference refused with %s, ours %s\n", c, quoteJson(chat["error"].text).c_str(), rendered ? "rendered" : quoteJson(problem).c_str());
            continue;
        }

        const std::string& expected = chat["prompt"].text;
        if(!rendered){
            printf("CHAT %zu: ours refused: %s\n", c, problem.c_str());
            continue;
        }

        if(prompt == expected) chatOk++;
        else {
            size_t at = 0;
            while(at < prompt.size() && at < expected.size() && prompt[at] == expected[at]) at++;
            size_t from = at > 60 ? at - 60 : 0;
            printf("CHAT %zu differs at byte %zu\n  ours      %s\n  reference %s\n", c, at, quoteJson(prompt.substr(from, 140)).c_str(), quoteJson(expected.substr(from, 140)).c_str());
        }
        chatIds += sameIds(tok.encode(prompt, true), chat["ids"]);
    }
    printf("\nchat template vs jinja2 (transformers settings):\n");
    printf("  prompt text exact : %zu / %zu\n", chatOk, chatTotal);
    printf("  prompt ids exact  : %zu / %zu\n", chatIds, chatTotal);

    const Json& tools = chats.items.size() > 2 ? chats.items[2]["tools"] : Json();
    struct Probe {
        const char* label;
        std::string text;
        bool thinking;
        std::string reasoning;
        std::string content;
        std::string calls;
    };
    std::vector<Probe> probes = {
        {"full reply", "Let me think.\n</think>\n\nSure.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nMadrid\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>", true,
            "Let me think.", "Sure.", "get_weather {\"city\": \"Madrid\", \"days\": 3} complete"},
        {"cut mid value", "ok\n</think>\n\n<tool_call>\n<function=run_code>\n<parameter=code>\nprint(", true,
            "ok", "", "run_code {\"code\": \"print(\"} partial"},
        {"missing closers", "<tool_call>\n<function=run_code>\n<parameter=code>\nx = 1\n<parameter=timeout>\n5\n</tool_call>", false,
            "", "", "run_code {\"code\": \"x = 1\", \"timeout\": 5} complete"},
        {"json body", "Checking.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>", false,
            "", "Checking.", "get_weather {\"city\": \"Paris\"} complete"},
        {"call in open thinking", "I should call it.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nLima\n</parameter>\n</function>\n</tool_call>", true,
            "I should call it.", "", "get_weather {\"city\": \"Lima\"} complete"},
        {"plain answer", "Hello there!", false, "", "Hello there!", ""},
        {"bare function", "Sure.\n<function=get_weather>\n<parameter=city>\nRome\n</parameter>\n</function>", false,
            "", "Sure.", "get_weather {\"city\": \"Rome\"} complete"},
        {"think tag echoed", "<think>\nhmm\n</think>\n\nAnswer.", false, "hmm", "Answer.", ""},
        {"two calls first open", "<tool_call>\n<function=get_weather>\n<parameter=city>\nA\n</parameter>\n<tool_call>\n<function=get_weather>\n<parameter=city>\nB\n</parameter>\n</function>\n</tool_call>", false,
            "", "", "get_weather {\"city\": \"A\"} partial; get_weather {\"city\": \"B\"} complete"},
        {"schema types", "<tool_call>\n<function=get_weather>\n<parameter=city>\n123\n</parameter>\n<parameter=days>\n4\n</parameter>\n</function>\n</tool_call>", false,
            "", "", "get_weather {\"city\": \"123\", \"days\": 4} complete"},
        {"no schema", "<tool_call>\n<function=other>\n<parameter=n>\n42\n</parameter>\n<parameter=s>\nhello\n</parameter>\n<parameter=o>\n{\"a\": [1, 2]}\n</parameter>\n</function>\n</tool_call>", false,
            "", "", "other {\"n\": 42, \"s\": \"hello\", \"o\": {\"a\": [1, 2]}} complete"},
        {"thinking only, unfinished", "Still thinking about it", true, "Still thinking about it", "", ""},
    };

    size_t probeOk = 0;
    for(const Probe& probe : probes){
        ChatReply reply = readReply(probe.text, probe.thinking, tools, true);
        std::string calls = describeCalls(reply);
        bool good = reply.reasoning == probe.reasoning && reply.content == probe.content && calls == probe.calls;
        probeOk += good;
        if(!good) printf("PARSE %s\n  reasoning %s\n  content %s\n  calls %s\n", probe.label, quoteJson(reply.reasoning).c_str(), quoteJson(reply.content).c_str(), calls.c_str());
    }

    size_t streamOk = 0;
    for(const Probe& probe : probes){
        std::vector<int> ids = tok.encode(probe.text, true);
        TextStream stream;
        std::string seen;
        ChatReply last;
        bool steady = true;
        for(size_t i = 0; i <= ids.size(); i++){
            seen += i < ids.size() ? stream.push(tok.tokenText(ids[i])) : stream.finish();
            ChatReply now = readReply(seen, probe.thinking, tools, i == ids.size());
            if(now.reasoning.compare(0, last.reasoning.size(), last.reasoning) != 0) steady = false;
            if(now.content.compare(0, last.content.size(), last.content) != 0) steady = false;
            if(now.calls.size() < last.calls.size()) steady = false;
            last = now;
        }
        bool good = steady && last.reasoning == probe.reasoning && last.content == probe.content && describeCalls(last) == probe.calls;
        streamOk += good;
        if(!good) printf("STREAM %s steady %d\n", probe.label, (int)steady);
    }

    size_t tripOk = 0, tripTotal = 0;
    for(const Json& chat : chats.items){
        const std::vector<Json>& messages = chat["messages"].items;
        for(size_t m = 1; m < messages.size(); m++){
            if(messages[m]["role"].str() != "assistant" || messages[m - 1]["role"].str() != "user") continue;
            Json pair;
            pair.kind = JSON_LIST;
            pair.items = {messages[m - 1], messages[m]};
            ChatOptions options;
            options.generation = false;
            std::string prompt, problem;
            if(!renderChat(pair, chat["tools"], options, prompt, problem)) continue;

            size_t begin = prompt.rfind("<|im_start|>assistant\n<think>\n");
            size_t end = prompt.rfind("<|im_end|>");
            if(begin == std::string::npos || end == std::string::npos || end < begin) continue;
            begin += 30;
            ChatReply reply = readReply(prompt.substr(begin, end - begin), true, chat["tools"], true);

            std::string expectedCalls;
            for(const Json& entry : messages[m]["tool_calls"].items){
                const Json& call = entry.has("function") ? entry["function"] : entry;
                Json arguments = call["arguments"].kind == JSON_TEXT ? parseJson(call["arguments"].text) : call["arguments"];
                if(arguments.kind != JSON_OBJECT) arguments.kind = JSON_OBJECT;
                if(!expectedCalls.empty()) expectedCalls += "; ";
                expectedCalls += call["name"].text + " " + dumpPython(arguments) + " complete";
            }

            std::string content;
            renderContent(messages[m]["content"], false, content, problem);
            tripTotal++;
            bool good = reply.reasoning == trimText(messages[m]["reasoning_content"].str()) && reply.content == trimText(content) && describeCalls(reply) == expectedCalls;
            tripOk += good;
            if(!good) printf("ROUND TRIP differs\n  calls ours %s\n  calls want %s\n", describeCalls(reply).c_str(), expectedCalls.c_str());
        }
    }

    printf("\nreply parser:\n");
    printf("  fixed probes      : %zu / %zu\n", probeOk, probes.size());
    printf("  token streaming   : %zu / %zu (reasoning and content only grow, same final result)\n", streamOk, probes.size());
    printf("  render then parse : %zu / %zu assistant turns\n", tripOk, tripTotal);

    bool perfect = fullMatch == total && plainMatch == total && roundTrip == total && streamTrip == total && chatOk == chatTotal && chatIds == chatTotal && probeOk == probes.size() && streamOk == probes.size() && tripOk == tripTotal;
    printf("\n%s\n", perfect ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED");
    return perfect ? 0 : 2;
}
