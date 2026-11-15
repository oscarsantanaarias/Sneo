#include "../src/serve.h"
#include <iostream>

int main() {
    OutputSplitter split;
    split.inThink = true;
    std::string reasoning, content;
    const char* pieces[] = {"let me ", "think</th", "ink>\n\nHello", " there <tool", "_call>\n<function=read_file>\n<parameter=path>\nsrc/a.js\n</parameter>\n</function>\n</tool_call>", " done"};
    for(const char* p : pieces) split.push(p, reasoning, content);
    split.finish(reasoning, content);
    std::cout << "reasoning=[" << reasoning << "]" << std::endl;
    std::cout << "content=[" << content << "]" << std::endl;
    std::cout << "tools=" << split.tools.size() << std::endl;
    for(size_t i = 0; i < split.tools.size(); i++) std::cout << writeJson(parseToolText(split.tools[i], (int)i)) << std::endl;
    std::cout << writeJson(parseToolText("{\"name\": \"list_files\", \"arguments\": {\"dir\": \".\"}}", 1)) << std::endl;

    GenerateRequest req;
    std::vector<float> logits = {0.1f, 3.0f, 2.9f, -1.0f, 0.5f};
    std::mt19937_64 random(3);
    int counts[5] = {};
    for(int i = 0; i < 2000; i++){
        std::vector<float> copy = logits;
        counts[sampleToken(copy, req, {}, random)]++;
    }
    std::cout << "samples: " << counts[0] << " " << counts[1] << " " << counts[2] << " " << counts[3] << " " << counts[4] << std::endl;
    req.temperature = 0;
    std::vector<float> copy = logits;
    std::cout << "greedy: " << sampleToken(copy, req, {}, random) << std::endl;
    return 0;
}
