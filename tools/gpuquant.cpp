#include "../src/kernels.h"
#include "../src/gguf.h"
#include <iostream>
#include <random>
#include <chrono>
#include <set>
#include <cmath>

uint64_t upload(const void* data, uint64_t bytes) {
    uint64_t ptr = 0;
    if(cudaLink().alloc(&ptr, (size_t)bytes) != 0) return 0;
    cudaLink().copyUp(ptr, data, (size_t)bytes);
    return ptr;
}

int main(int argc, char* argv[]) {
    if(argc < 4){
        std::cout << "gpuquant <model.gguf> <tables.bin> <dump.bin>" << std::endl;
        return 1;
    }

    ModelFile model;
    std::string error;
    if(!openModel(argv[1], model, error)){
        std::cout << error << std::endl;
        return 1;
    }

    KernelModule module;
    if(!buildModule(quantKernelSource, module)){
        std::cout << "kernel build failed: " << module.log << std::endl;
        return 1;
    }

    FILE* tableFile = nullptr;
    fopen_s(&tableFile, argv[2], "rb");
    std::vector<char> tableData(256 * 8 + 512 * 8 + 1024 * 8 + 256 * 4 + 512 * 4 + 128 + 16);
    if(!tableFile || fread(tableData.data(), 1, tableData.size(), tableFile) != tableData.size()){
        std::cout << "tables missing" << std::endl;
        return 1;
    }
    fclose(tableFile);

    uint64_t base = upload(tableData.data(), tableData.size());
    GridTables tables = {base, base + 2048, base + 6144, base + 14336, base + 15360, base + 17408, base + 17536};

    FILE* dump = nullptr;
    fopen_s(&dump, argv[3], "wb");
    std::mt19937 random(11);
    std::uniform_real_distribution<float> spread(-1.0f, 1.0f);
    std::set<uint32_t> seen;

    for(const TensorInfo& t : model.tensors){
        if(seen.count(t.type) || t.dimCount < 2 || t.dims[0] % 32 != 0) continue;
        seen.insert(t.type);

        int cols = (int)t.dims[0];
        uint64_t totalRows = t.elements / cols;
        int rows = (int)std::min<uint64_t>(256, totalRows);
        uint64_t rowBytes = t.bytes / totalRows;
        std::vector<char> raw(rowBytes * rows);

        FILE* f = nullptr;
        fopen_s(&f, model.paths[t.file].c_str(), "rb");
        _fseeki64(f, (long long)t.offset, SEEK_SET);
        fread(raw.data(), 1, raw.size(), f);
        fclose(f);

        std::vector<float> input(cols), output(rows);
        for(float& v : input) v = spread(random);

        uint64_t w = upload(raw.data(), raw.size());
        uint64_t x = upload(input.data(), input.size() * 4);
        uint64_t y = 0;
        cudaLink().alloc(&y, output.size() * 4);
        if(!launchRows(module, (int)t.type, w, rowBytes, x, y, rows, cols, tables) || cudaLink().sync() != 0){
            std::cout << "launch failed for type " << t.type << std::endl;
            return 1;
        }
        cudaLink().copyDown(output.data(), y, output.size() * 4);

        PackedInput packed;
        std::vector<float> fast(rows);
        packInput(module, x, cols, packed);
        launchRowsQ8(module, (int)t.type, w, rowBytes, x, packed, y, rows, cols, tables);
        cudaLink().sync();
        cudaLink().copyDown(fast.data(), y, fast.size() * 4);
        double worst = 0, peak = 0;
        for(int r = 0; r < rows; r++){
            worst = std::max(worst, (double)std::fabs(fast[r] - output[r]));
            peak = std::max(peak, (double)std::fabs(output[r]));
        }
        std::cout << "  int8 path vs float path: max rel diff " << worst / (peak + 1e-12) << std::endl;

        cudaLink().release(w);
        cudaLink().release(x);
        cudaLink().release(y);

        uint32_t head[3] = {t.type, (uint32_t)rows, (uint32_t)cols};
        uint64_t rawLength = raw.size();
        uint32_t nameLength = (uint32_t)t.name.size();
        fwrite(head, 4, 3, dump);
        fwrite(&rawLength, 8, 1, dump);
        fwrite(&nameLength, 4, 1, dump);
        fwrite(t.name.data(), 1, nameLength, dump);
        fwrite(raw.data(), 1, raw.size(), dump);
        fwrite(input.data(), 4, input.size(), dump);
        fwrite(output.data(), 4, output.size(), dump);
        std::cout << "type " << t.type << " (" << typeInfo(t.type).name << ") " << t.name << " done" << std::endl;
    }
    fclose(dump);

    for(const TensorInfo& t : model.tensors){
        if(!t.expert || t.bytes > (600ULL << 20)) continue;
        int cols = (int)t.dims[0];
        uint64_t totalRows = t.elements / cols;
        uint64_t rowBytes = t.bytes / totalRows;
        std::vector<char> raw(t.bytes);
        FILE* f = nullptr;
        fopen_s(&f, model.paths[t.file].c_str(), "rb");
        _fseeki64(f, (long long)t.offset, SEEK_SET);
        fread(raw.data(), 1, raw.size(), f);
        fclose(f);

        std::vector<float> input(cols, 0.5f);
        uint64_t w = upload(raw.data(), raw.size());
        uint64_t x = upload(input.data(), input.size() * 4);
        uint64_t y = 0;
        cudaLink().alloc(&y, totalRows * 4);
        launchRows(module, (int)t.type, w, rowBytes, x, y, (int)totalRows, cols, tables);
        cudaLink().sync();
        auto t0 = std::chrono::steady_clock::now();
        for(int i = 0; i < 20; i++) launchRows(module, (int)t.type, w, rowBytes, x, y, (int)totalRows, cols, tables);
        cudaLink().sync();
        double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 20;
        PackedInput packed;
        packInput(module, x, cols, packed);
        launchRowsQ8(module, (int)t.type, w, rowBytes, x, packed, y, (int)totalRows, cols, tables);
        cudaLink().sync();
        auto t1 = std::chrono::steady_clock::now();
        for(int i = 0; i < 20; i++) launchRowsQ8(module, (int)t.type, w, rowBytes, x, packed, y, (int)totalRows, cols, tables);
        cudaLink().sync();
        double fastSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count() / 20;
        std::cout << "speed int8 " << typeInfo(t.type).name << " " << t.name << ": " << (double)t.bytes / fastSeconds / 1e9 << " GB/s, " << (double)t.elements / fastSeconds / 1e9 << " Gweights/s" << std::endl;
        std::cout << "speed " << typeInfo(t.type).name << " " << t.name << ": " << (double)t.bytes / seconds / 1e9 << " GB/s, " << (double)t.elements / seconds / 1e9 << " Gweights/s" << std::endl;
        cudaLink().release(w);
        cudaLink().release(x);
        cudaLink().release(y);
        static std::set<uint32_t> timed;
        timed.insert(t.type);
        if(timed.size() >= 4) break;
    }
    return 0;
}
