#include "../src/qwen4.h"
#include "../src/residency.h"
#include <psapi.h>
#include <cctype>

std::string pieceBytes(const std::string& piece) {
    static int byteOf[512];
    static bool ready = false;
    if(!ready){
        for(int i = 0; i < 512; i++) byteOf[i] = -1;
        int extra = 0;
        for(int b = 0; b < 256; b++){
            bool plain = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
            byteOf[plain ? b : 256 + extra++] = b;
        }
        ready = true;
    }

    std::string out;
    for(size_t i = 0; i < piece.size();){
        unsigned char c = piece[i];
        uint32_t code = c;
        int length = c < 0x80
            ? 1
            : c < 0xE0
                ? 2
                : c < 0xF0
                    ? 3
                    : 4;
        if(length == 2) code = c & 0x1F;
        if(length == 3) code = c & 0x0F;
        if(length == 4) code = c & 0x07;
        for(int k = 1; k < length && i + k < piece.size(); k++) code = (code << 6) | (piece[i + k] & 0x3F);
        if(code < 512 && byteOf[code] >= 0) out.push_back((char)byteOf[code]);
        else out += piece.substr(i, length);
        i += length;
    }
    return out;
}

std::string shown(const std::string& text) {
    std::string out;
    for(char c : text){
        if(c == '\n') out += "\\n";
        else if(c == '\t') out += "\\t";
        else out.push_back(c);
    }
    return out;
}

double privatePeakGb() {
    PROCESS_MEMORY_COUNTERS_EX counters = {};
    counters.cb = sizeof(counters);
    GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&counters, sizeof(counters));
    return (double)counters.PeakPagefileUsage / (double)gib;
}

int main(int argc, char** argv) {
    if(argc < 2){
        printf("usage: generate <model-00001-of-N.gguf> [--text \"The capital of France is\"] [--ids 1,2,3] [--tokens N] [--threads T] [--stepwise] [--perplexity] [--score last] [--budget cells] [--no-ple] [--ram-cap GB] [--lock-dense]\n");
        return 1;
    }

    std::string path = argv[1];
    std::string text;
    std::string idList;
    int count = 16;
    int threads = (int)readHost("").cores;
    int budget = 0;
    double ramCap = 0;
    bool lockDense = false;
    bool stepwise = false;
    bool perplexity = false;
    int scoreLast = 0;
    bool usePle = true;

    for(int i = 2; i < argc; i++){
        std::string arg = argv[i];
        bool more = i + 1 < argc;
        if(arg == "--text" && more) text = argv[++i];
        else if(arg == "--ids" && more) idList = argv[++i];
        else if(arg == "--tokens" && more) count = atoi(argv[++i]);
        else if(arg == "--threads" && more) threads = atoi(argv[++i]);
        else if(arg == "--budget" && more) budget = atoi(argv[++i]);
        else if(arg == "--stepwise") stepwise = true;
        else if(arg == "--perplexity") perplexity = true;
        else if(arg == "--score" && more) scoreLast = atoi(argv[++i]);
        else if(arg == "--no-ple") usePle = false;
        else if(arg == "--ram-cap" && more) ramCap = atof(argv[++i]);
        else if(arg == "--lock-dense") lockDense = true;
        else {
            printf("unknown argument %s\n", arg.c_str());
            return 1;
        }
    }

    if(text.empty() && idList.empty()) text = "The capital of France is";
    if(threads < 1) threads = (int)std::thread::hardware_concurrency();
    SetConsoleOutputCP(CP_UTF8);

    auto started = std::chrono::steady_clock::now();
    Model model;
    if(ramCap > 0) model.memoryCap = (uint64_t)(ramCap * (double)gib);
    model.lockDense = lockDense;
    if(!model.load(path, threads)){
        printf("load failed: %s\n", model.problem.c_str());
        return 1;
    }
    if(!model.problem.empty()) printf("warning: %s\n", model.problem.c_str());
    model.usePle = usePle;
    if(budget > 0) model.budget = budget;
    double loadSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();

    auto vocabEntry = model.file.meta.find("tokenizer.ggml.tokens");
    if(vocabEntry == model.file.meta.end()){
        printf("vocabulary missing\n");
        return 1;
    }
    const std::vector<std::string>& vocab = vocabEntry->second.texts;
    std::map<std::string, int> lookup;
    for(size_t i = 0; i < vocab.size(); i++) lookup[vocab[i]] = (int)i;

    printf("model %s, %u layers, %zu files mapped, %d threads, loaded in %.2f s, %.1f GB RAM free, working set capped at %.1f GB\n", model.file.name.c_str(), model.file.layers, model.maps.size(), threads, loadSeconds, (double)ramAvailable() / (double)gib, (double)model.memoryCap / (double)gib);

    std::vector<int> prompt;
    if(!idList.empty()){
        size_t at = 0;
        while(at < idList.size()){
            size_t comma = idList.find(',', at);
            if(comma == std::string::npos) comma = idList.size();
            prompt.push_back(atoi(idList.substr(at, comma - at).c_str()));
            at = comma + 1;
        }
    } else {
        size_t i = 0;
        while(i < text.size()){
            size_t start = i;
            bool space = text[i] == ' ';
            size_t j = space ? i + 1 : i;
            if(j < text.size() && isalpha((unsigned char)text[j])){
                while(j < text.size() && isalpha((unsigned char)text[j])) j++;
            } else if(j < text.size() && ispunct((unsigned char)text[j])){
                while(j < text.size() && ispunct((unsigned char)text[j])) j++;
            } else {
                j = i + 1;
            }

            std::string piece;
            for(size_t k = start; k < j; k++){
                if(text[k] == ' ') piece += "\xC4\xA0";
                else piece.push_back(text[k]);
            }
            auto found = lookup.find(piece);
            if(found == lookup.end()){
                printf("\"%s\" is not a single vocabulary entry, pass --ids instead\n", text.substr(start, j - start).c_str());
                return 1;
            }
            prompt.push_back(found->second);
            i = j;
        }
    }

    printf("prompt %zu tokens:", prompt.size());
    for(int id : prompt){
        if(id < 0 || id >= (int)vocab.size()){
            printf("\ntoken id %d is outside the vocabulary\n", id);
            return 1;
        }
        printf(" %d[%s]", id, shown(pieceBytes(vocab[id])).c_str());
    }
    printf("\n");

    std::vector<int> positions(prompt.size());
    std::iota(positions.begin(), positions.end(), 0);

    if(perplexity && scoreLast == 0) scoreLast = (int)prompt.size() - 1;
    if(scoreLast > 0){
        if(scoreLast > (int)prompt.size() - 1) scoreLast = (int)prompt.size() - 1;
        size_t first = prompt.size() - scoreLast;
        double nll = 0;
        int hits = 0;
        started = std::chrono::steady_clock::now();
        std::vector<int> head(prompt.begin(), prompt.begin() + first);
        std::vector<int> headPositions(positions.begin(), positions.begin() + first);
        std::vector<float> scores = model.forward(head, headPositions);
        printf("prefill of %zu tokens %.2f s\n", first, std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
        started = std::chrono::steady_clock::now();
        for(size_t i = first - 1; i + 1 < prompt.size(); i++){
            if(i >= first) scores = model.forward({prompt[i]}, {positions[i]});
            float top = *std::max_element(scores.begin(), scores.end());
            double total = 0;
            for(float v : scores) total += exp((double)v - top);
            int wanted = prompt[i + 1];
            double loss = log(total) + top - scores[wanted];
            int best = (int)(std::max_element(scores.begin(), scores.end()) - scores.begin());
            hits += best == wanted;
            nll += loss;
            printf("  %4zu %-16s nll %7.3f  argmax %s\n", i + 1, ("[" + shown(pieceBytes(vocab[wanted])) + "]").c_str(), loss, ("[" + shown(pieceBytes(vocab[best])) + "]").c_str());
        }
        double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        printf("%s: %d tokens scored, mean nll %.4f, perplexity %.3f, argmax hits %d, %.2f tokens/s\n", usePle ? "with ple" : "without ple", scoreLast, nll / scoreLast, exp(nll / scoreLast), hits, scoreLast / seconds);
        return 0;
    }

    started = std::chrono::steady_clock::now();
    std::vector<float> logits;
    if(stepwise){
        for(size_t i = 0; i < prompt.size(); i++) logits = model.forward({prompt[i]}, {positions[i]});
    } else {
        logits = model.forward(prompt, positions);
    }
    double prefillSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    if(logits.empty()){
        printf("forward failed\n");
        return 1;
    }
    printf("%s %.2f s, %.2f tokens/s\n", stepwise ? "stepwise prefill" : "prefill", prefillSeconds, prompt.size() / prefillSeconds);

    std::vector<int> order(logits.size());
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + 5, order.end(), [&](int l, int r){ return logits[l] > logits[r]; });
    double top = logits[order[0]], total = 0;
    for(float v : logits) total += exp((double)v - top);
    printf("top 5 for the first predicted token:\n");
    for(int k = 0; k < 5; k++){
        int id = order[k];
        printf("  %6d %-20s logit %9.5f  p %.5f\n", id, ("[" + shown(pieceBytes(vocab[id])) + "]").c_str(), logits[id], exp((double)logits[id] - top) / total);
    }

    std::vector<double> gdnMs, qsaMs;
    double headMs = 0;
    std::string continuation;
    std::vector<int> produced;
    printf("greedy:");
    fflush(stdout);
    started = std::chrono::steady_clock::now();
    int position = (int)prompt.size();
    for(int step = 0; step < count; step++){
        int next = (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
        produced.push_back(next);
        std::string piece = pieceBytes(vocab[next]);
        continuation += piece;
        printf(" %d[%s]", next, shown(piece).c_str());
        fflush(stdout);
        if(next == 248044 || next == 248046 || step + 1 == count) break;

        logits = model.forward({next}, {position++});
        for(size_t l = 0; l < model.layerMs.size(); l++){
            if(model.layers[l].attention) qsaMs.push_back(model.layerMs[l]);
            else gdnMs.push_back(model.layerMs[l]);
        }
        headMs += model.headMs;
    }
    double decodeSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    int steps = (int)produced.size() - 1;
    printf("\ncontinuation: \"%s\"\n", shown(continuation).c_str());
    if(steps > 0){
        auto average = [](const std::vector<double>& v){ return v.empty() ? 0.0 : std::accumulate(v.begin(), v.end(), 0.0) / v.size(); };
        printf("decode %d tokens in %.2f s, %.2f tokens/s\n", steps, decodeSeconds, steps / decodeSeconds);
        printf("per layer: gated deltanet %.1f ms, full attention %.1f ms, head %.1f ms per token\n", average(gdnMs), average(qsaMs), headMs / steps);
        printf("last token layer ms:");
        for(size_t l = 0; l < model.layerMs.size(); l++) printf(" %.0f", model.layerMs[l]);
        printf("\n");
    }
    printf("peak private memory %.2f GB, %.1f GB RAM free\n", privatePeakGb(), (double)ramAvailable() / (double)gib);
    return 0;
}
