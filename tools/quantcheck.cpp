#include "../src/residency.h"
#include "../src/quant.h"
#include <chrono>

int main(int argc, char** argv) {
    if(argc < 3){
        printf("usage: quantcheck <model-00001-of-N.gguf> <dump.bin> [rows per tensor] [bench]\n");
        return 1;
    }

    ModelFile model;
    std::string error;
    if(!openModel(argv[1], model, error)){
        printf("%s\n", error.c_str());
        return 1;
    }

    int perTensor = argc > 3 ? atoi(argv[3]) : 6;
    bool bench = argc > 4 && std::string(argv[4]) == "bench";

    std::map<uint32_t, std::vector<size_t>> byType;
    for(size_t i = 0; i < model.tensors.size(); i++) byType[model.tensors[i].type].push_back(i);

    std::vector<HANDLE> files;
    for(const std::string& path : model.paths){
        HANDLE h = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_RANDOM_ACCESS, nullptr);
        if(h == INVALID_HANDLE_VALUE){
            printf("cannot open %s\n", path.c_str());
            return 1;
        }
        files.push_back(h);
    }

    FILE* out = nullptr;
    fopen_s(&out, argv[2], "wb");
    if(!out){
        printf("cannot write %s\n", argv[2]);
        return 1;
    }

    uint64_t seed = 0x9E3779B97F4A7C15ULL;
    uint64_t records = 0;

    for(auto& entry : byType){
        uint32_t type = entry.first;
        std::vector<size_t>& list = entry.second;

        if(!knownType(type)){
            printf("type %u is not supported (%zu tensors)\n", type, list.size());
            continue;
        }

        std::vector<size_t> picks = {list.front(), list[list.size() / 3], list[2 * list.size() / 3], list.back()};
        std::sort(picks.begin(), picks.end());
        picks.erase(std::unique(picks.begin(), picks.end()), picks.end());

        int samples = 0;
        std::string names;
        for(size_t index : picks){
            const TensorInfo& t = model.tensors[index];
            uint64_t cols = t.dims[0];
            uint64_t rows = t.elements / cols;
            uint64_t bytes = rowBytes(type, cols);
            names += (names.empty() ? "" : ", ") + t.name;

            for(int k = 0; k < perTensor; k++){
                seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
                uint64_t row = k == 0
                    ? 0
                    : k == 1
                        ? rows - 1
                        : (seed >> 17) % rows;

                std::vector<uint8_t> raw((size_t)bytes);
                std::vector<float> values((size_t)cols);
                if(!readRange(files[t.file], t.offset + row * bytes, (char*)raw.data(), bytes)){
                    printf("read failed on %s row %llu\n", t.name.c_str(), (unsigned long long)row);
                    return 1;
                }
                dequantRow(type, raw.data(), values.data(), (int64_t)cols);

                uint32_t head[3] = {type, (uint32_t)cols, (uint32_t)t.name.size()};
                fwrite(head, 4, 3, out);
                fwrite(t.name.data(), 1, t.name.size(), out);
                fwrite(&row, 8, 1, out);
                fwrite(raw.data(), 1, raw.size(), out);
                fwrite(values.data(), 4, values.size(), out);
                samples++;
                records++;
            }
        }

        printf("type %2u %-8s %4zu tensors, %3d rows from %s\n", type, type == 42 ? "Q2_0" : typeInfo(type).name, list.size(), samples, names.c_str());
    }

    fclose(out);
    printf("%llu rows written to %s\n", (unsigned long long)records, argv[2]);

    if(!bench) return 0;

    printf("\nsingle thread decode+dot speed per type\n");
    for(auto& entry : byType){
        uint32_t type = entry.first;
        if(!knownType(type) || type == 0) continue;
        const TensorInfo& t = model.tensors[entry.second.front()];
        uint64_t cols = t.dims[0];
        uint64_t rows = std::min<uint64_t>(t.elements / cols, 512);
        if(cols % 32 != 0 || rows < 4) continue;
        uint64_t bytes = rowBytes(type, cols);
        std::vector<uint8_t> raw((size_t)(bytes * rows));
        if(!readRange(files[t.file], t.offset, (char*)raw.data(), raw.size())) continue;
        std::vector<float> x((size_t)cols, 0.5f), row((size_t)cols);
        float sink = 0;
        auto start = std::chrono::steady_clock::now();
        int passes = 0;
        double seconds = 0;
        while(seconds < 0.3){
            for(uint64_t r = 0; r < rows; r++){
                dequantRow(type, raw.data() + r * bytes, row.data(), (int64_t)cols);
                sink += dot(row.data(), x.data(), (int)cols);
            }
            passes++;
            seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        }
        double elements = (double)passes * rows * cols;
        printf("type %2u %-8s %7.2f Gelem/s %7.2f GB/s  (%g)\n", type, type == 42 ? "Q2_0" : typeInfo(type).name, elements / seconds / 1e9, (double)passes * rows * bytes / seconds / 1e9, sink > 0 ? 1.0 : 0.0);
    }
    return 0;
}
