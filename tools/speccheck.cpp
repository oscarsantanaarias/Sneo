#include "../src/serve.h"
#include <iostream>

struct Toy {
    std::vector<int> history;
    size_t saved = 0;
    int table[64][64];
    int forwards = 0;
    int verifies = 0;

    Toy() {
        std::mt19937 random(9);
        for(int a = 0; a < 64; a++){
            for(int b = 0; b < 64; b++) table[a][b] = (int)(random() % 24);
        }
    }

    std::vector<float> after() const {
        int a = history.size() >= 2 ? history[history.size() - 2] : 0;
        int b = history.empty() ? 0 : history.back();
        std::vector<float> logits(64);
        for(int v = 0; v < 64; v++) logits[v] = v == table[a % 64][b % 64] ? 6.0f : (float)((v * 7 + a + b) % 11) * 0.15f;
        return logits;
    }
};

GenerateResult runOnce(bool drafting, float temperature, uint64_t seed, Toy& toy) {
    GenerateRequest req;
    req.maxTokens = 400;
    req.temperature = temperature;
    req.topK = 0;
    req.topP = 1.0f;
    req.seed = seed;

    std::vector<int> prompt;
    for(int r = 0; r < 6; r++){
        for(int i = 0; i < 12; i++) prompt.push_back((i * 5 + r) % 24);
    }

    GenerateHooks hooks;
    hooks.render = [&](const GenerateRequest&){ return std::string("toy"); };
    hooks.encode = [&](const std::string&){ return prompt; };
    hooks.piece = [&](int id){ return std::to_string(id) + " "; };
    hooks.isStop = [&](int id){ return id == 63; };
    hooks.forward = [&](const std::vector<int>& ids, size_t, std::vector<float>& logits) {
        toy.forwards++;
        toy.history.insert(toy.history.end(), ids.begin(), ids.end());
        logits = toy.after();
        return true;
    };
    if(drafting){
        hooks.verifyLimit = 16;
        hooks.mark = [&](int){ toy.saved = toy.history.size(); return true; };
        hooks.rewind = [&]{ toy.history.resize(toy.saved); return true; };
        hooks.verify = [&](const std::vector<int>& ids, std::vector<float>& rows) {
            toy.verifies++;
            rows.clear();
            for(int id : ids){
                toy.history.push_back(id);
                std::vector<float> row = toy.after();
                rows.insert(rows.end(), row.begin(), row.end());
            }
            return true;
        };
    }
    return runGenerate(INVALID_SOCKET, req, hooks);
}

int main() {
    startSockets();
    Toy plain, fast;
    GenerateResult a = runOnce(false, 0.0f, 1, plain);
    GenerateResult b = runOnce(true, 0.0f, 1, fast);
    bool same = a.content == b.content && a.outputTokens == b.outputTokens;
    std::cout << "greedy: " << a.outputTokens << " tokens, identical with drafting: " << (same ? "yes" : "NO") << std::endl;
    std::cout << "drafted " << b.drafted << ", kept " << b.kept << ", model calls " << plain.forwards << " plain vs " << fast.forwards + fast.verifies << " with drafting" << std::endl;
    if(!same){
        std::cout << "plain: " << a.content.substr(0, 200) << std::endl;
        std::cout << "draft: " << b.content.substr(0, 200) << std::endl;
        return 1;
    }

    int histogramPlain[64] = {}, histogramFast[64] = {};
    uint64_t keptTotal = 0, draftedTotal = 0;
    for(uint64_t seed = 1; seed <= 300; seed++){
        Toy p, f;
        GenerateResult x = runOnce(false, 0.8f, seed, p);
        GenerateResult y = runOnce(true, 0.8f, seed + 100000, f);
        for(size_t i = 0; i + 1 < x.content.size(); i++){
            if(x.content[i] == ' ') continue;
        }
        for(int id : p.history) if(id >= 0 && id < 64) histogramPlain[id]++;
        for(int id : f.history) if(id >= 0 && id < 64) histogramFast[id]++;
        keptTotal += y.kept;
        draftedTotal += y.drafted;
    }
    double plainTotal = 0, fastTotal = 0, worst = 0;
    for(int v = 0; v < 64; v++){
        plainTotal += histogramPlain[v];
        fastTotal += histogramFast[v];
    }
    for(int v = 0; v < 64; v++) worst = std::max(worst, std::fabs(histogramPlain[v] / plainTotal - histogramFast[v] / fastTotal));
    std::cout << "sampling at temperature 0.8 over 300 runs: largest token frequency gap " << worst << " (should be small), drafts kept " << keptTotal << " of " << draftedTotal << std::endl;
    return 0;
}
