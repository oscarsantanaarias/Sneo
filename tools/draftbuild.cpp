#include "../src/draft.h"
#include <iostream>
#include <chrono>

int main(int argc, char** argv) {
    std::string folder = argc > 1 ? argv[1] : "modelos";
    int threads = argc > 2 ? atoi(argv[2]) : 6;
    int bits = argc > 3 ? atoi(argv[3]) : 2;
    std::string error;
    auto start = std::chrono::steady_clock::now();
    if(!buildDraft(draftSourcePath(folder), draftRuntimePath(folder, bits), threads, error, bits)){
        std::cerr << "draft build failed: " << error << std::endl;
        return 1;
    }
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    ModelFile check;
    if(!openModel(draftRuntimePath(folder, bits), check, error)){
        std::cerr << "the new draft file does not open: " << error << std::endl;
        return 1;
    }
    uint64_t bytes = 0;
    for(const TensorInfo& t : check.tensors) bytes += t.bytes;
    std::cout << "draft runtime file " << draftRuntimePath(folder, bits) << ": " << check.tensors.size() << " tensors, " << (double)bytes / 1e9 << " GB, built in " << seconds << " s" << std::endl;
    return 0;
}
