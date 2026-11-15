#include "../src/brain.h"
#include <dbghelp.h>
#include <iostream>
#include <chrono>
#include <numeric>
#include <random>

#pragma comment(lib, "dbghelp.lib")

LONG WINAPI report(EXCEPTION_POINTERS* info) {
    HANDLE process = GetCurrentProcess();
    HANDLE thread = GetCurrentThread();
    SymSetOptions(SYMOPT_LOAD_LINES | SYMOPT_UNDNAME);
    SymInitialize(process, nullptr, TRUE);
    CONTEXT context = *info->ContextRecord;
    STACKFRAME64 frame = {};
    frame.AddrPC.Offset = context.Rip;
    frame.AddrPC.Mode = AddrModeFlat;
    frame.AddrFrame.Offset = context.Rbp;
    frame.AddrFrame.Mode = AddrModeFlat;
    frame.AddrStack.Offset = context.Rsp;
    frame.AddrStack.Mode = AddrModeFlat;
    printf("crash %08lx at %p touching %p\n", info->ExceptionRecord->ExceptionCode, info->ExceptionRecord->ExceptionAddress, (void*)info->ExceptionRecord->ExceptionInformation[1]);
    for(int i = 0; i < 24 && StackWalk64(IMAGE_FILE_MACHINE_AMD64, process, thread, &frame, &context, nullptr, SymFunctionTableAccess64, SymGetModuleBase64, nullptr); i++){
        char buffer[sizeof(SYMBOL_INFO) + 256] = {};
        auto symbol = (SYMBOL_INFO*)buffer;
        symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
        symbol->MaxNameLen = 255;
        DWORD64 offset = 0;
        DWORD column = 0;
        IMAGEHLP_LINE64 line = {};
        line.SizeOfStruct = sizeof(line);
        bool named = SymFromAddr(process, frame.AddrPC.Offset, &offset, symbol) != 0;
        bool lined = SymGetLineFromAddr64(process, frame.AddrPC.Offset, &column, &line) != 0;
        printf("  %s line %lu %s\n", named ? symbol->Name : "?", lined ? line.LineNumber : 0UL, lined ? line.FileName : "");
    }
    fflush(stdout);
    return EXCEPTION_EXECUTE_HANDLER;
}

double freeGb() {
    MEMORYSTATUSEX memory = {};
    memory.dwLength = sizeof(memory);
    GlobalMemoryStatusEx(&memory);
    return memory.ullAvailPhys / 1073741824.0;
}

int main(int argc, char** argv) {
    if(argc < 2){
        std::cout << "fronttest model.gguf [layers] [rows] [vram gb] [ram gb] [stream 0/1]" << std::endl;
        return 1;
    }
    SetUnhandledExceptionFilter(report);
    std::string modelPath = argv[1];
    int end = argc > 2 ? atoi(argv[2]) : 14;
    int n = argc > 3 ? atoi(argv[3]) : 512;
    double vram = argc > 4 ? atof(argv[4]) : 7.9;
    double ram = argc > 5 ? atof(argv[5]) : 31.0;
    bool stream = argc > 6 && atoi(argv[6]) != 0;
    const double gib = 1073741824.0;

    ModelFile file;
    std::string error;
    if(!openModel(modelPath, file, error)){
        std::cout << "cannot read the model: " << error << std::endl;
        return 1;
    }

    GpuModel gpu;
    gpu.firstLayer = 0;
    gpu.endLayer = end;
    gpu.contextLimit = 65536;
    gpu.relayLimit = 4;
    gpu.tailSkip = 7;
    gpu.vramCap = (uint64_t)(vram * gib);
    gpu.hostBudget = (uint64_t)(ram * gib);
    gpu.streamPrefill = stream;
    gpu.superLimit = 2048;
    std::cout << "ram free before load " << freeGb() << " GB" << std::endl;
    auto start = std::chrono::steady_clock::now();
    if(!gpu.loadFile(file, modelPath, 16)){
        std::cout << "load failed: " << gpu.problem << std::endl;
        return 1;
    }
    std::cout << "loaded layers 0.." << end - 1 << " in " << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << " s, " << gpu.residentExperts << " experts per layer in vram, " << gpu.warmPerLayer << " pinned, warm budget " << gpu.warmBudget / gib << " GB, ram free " << freeGb() << " GB" << std::endl;

    for(int l : {0, end - 1}){
        GpuLayer& L = gpu.layers[l];
        std::cout << "layer " << l << " expert types gate " << L.expertGate.type << " up " << L.expertUp.type << " down " << L.expertDown.type << ", int8 cpu " << (hostDotType(L.expertGate.type) ? "gate " : "") << (hostDotType(L.expertUp.type) ? "up " : "") << (hostDotType(L.expertDown.type) ? "down" : "") << std::endl;
    }

    std::mt19937 random(7);
    std::normal_distribution<float> spread(0.0f, 0.02f);
    int at = 0;
    for(int rows : {1, 4, 16, 64, 256, n}){
        std::vector<int> positions(rows);
        for(int i = 0; i < rows; i++) positions[i] = at + i;
        std::vector<float> embeds((size_t)rows * hidden), ples((size_t)rows * hidden, 0.0f), out((size_t)rows * hcWidth);
        for(float& v : embeds) v = spread(random);
        std::cout << "sending " << rows << " rows at position " << at << std::endl;
        start = std::chrono::steady_clock::now();
        bool ok = gpu.forwardRows(rows, false, false, positions.data(), embeds.data(), ples.data(), out.data());
        std::cout << "  " << (ok ? "ok" : "failed: " + gpu.problem) << " in " << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << " s, ram free " << freeGb() << " GB" << std::endl;
        if(!ok) return 1;
        double total = 0;
        for(float v : out) total += fabs(v);
        std::cout << "  mean |out| " << total / out.size() << ", first " << out[0] << " " << out[1] << " " << out[2] << ", last row " << out[(size_t)(rows - 1) * hcWidth] << " " << out[(size_t)(rows - 1) * hcWidth + 1] << std::endl;
        at += rows;
    }
    return 0;
}
