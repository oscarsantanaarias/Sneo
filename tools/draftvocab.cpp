#include "../src/gguf.h"
#include "../src/tokenizer.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <numeric>

std::string readText(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    std::stringstream text;
    text << in.rdbuf();
    return text.str();
}

std::vector<std::filesystem::path> collect(const std::string& folder) {
    std::vector<std::filesystem::path> files;
    std::error_code code;
    if(std::filesystem::is_regular_file(folder, code)){
        files.push_back(folder);
        return files;
    }
    for(auto it = std::filesystem::recursive_directory_iterator(folder, std::filesystem::directory_options::skip_permission_denied, code); it != std::filesystem::recursive_directory_iterator(); it.increment(code)){
        if(code) break;
        if(!it->is_regular_file(code)) continue;
        std::string kind = it->path().extension().string();
        bool wanted = kind == ".md" || kind == ".txt" || kind == ".h" || kind == ".cpp" || kind == ".js" || kind == ".jsx" || kind == ".py" || kind == ".ts";
        if(wanted && it->file_size(code) < (8ULL << 20)) files.push_back(it->path());
    }
    std::sort(files.begin(), files.end());
    return files;
}

int main(int argc, char** argv) {
    if(argc < 4){
        printf("usage: draftvocab <output> <size> <folder or file>... [--test <folder or file>...]\n");
        return 1;
    }
    std::string output = argv[1];
    int size = atoi(argv[2]);
    ModelFile model;
    std::string error;
    Tokenizer words;
    if(!openModel("modelos\\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf", model, error) || !words.load(model, error)){
        printf("cannot read the model: %s\n", error.c_str());
        return 1;
    }
    std::vector<std::string> learn, test;
    bool testing = false;
    for(int i = 3; i < argc; i++){
        std::string arg = argv[i];
        if(arg == "--test"){
            testing = true;
            continue;
        }
        (testing ? test : learn).push_back(arg);
    }

    size_t vocab = 0;
    std::vector<uint64_t> counts;
    uint64_t seen = 0;
    for(const std::string& folder : learn){
        for(const auto& file : collect(folder)){
            std::vector<int> ids = words.encode(readText(file), false);
            for(int id : ids){
                if(id < 0) continue;
                if((size_t)id >= counts.size()) counts.resize((size_t)id + 1, 0);
                counts[id]++;
            }
            seen += ids.size();
        }
    }
    vocab = std::max(counts.size(), words.pieces.size());
    counts.resize(vocab, 0);
    std::vector<int> order(vocab);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b){ return counts[a] > counts[b]; });
    size = std::min(size, (int)vocab);
    size_t distinct = 0;
    for(uint64_t c : counts) distinct += c > 0;
    printf("learned from %llu tokens, %zu distinct of %zu\n", (unsigned long long)seen, distinct, vocab);

    std::vector<char> kept(vocab, 0);
    std::vector<int> chosen(order.begin(), order.begin() + size);
    std::sort(chosen.begin(), chosen.end());
    for(int id : chosen) kept[id] = 1;

    for(const std::string& folder : test){
        uint64_t total = 0, inside = 0;
        for(const auto& file : collect(folder)){
            std::vector<int> ids = words.encode(readText(file), false);
            for(int id : ids){
                total++;
                inside += id >= 0 && (size_t)id < vocab && kept[id];
            }
        }
        printf("coverage of %s: %.2f%% of %llu tokens\n", folder.c_str(), total ? 100.0 * inside / total : 0.0, (unsigned long long)total);
    }

    std::ofstream out(output, std::ios::binary);
    out.write((const char*)chosen.data(), chosen.size() * sizeof(int));
    printf("wrote %d token ids to %s\n", size, output.c_str());
    return 0;
}
