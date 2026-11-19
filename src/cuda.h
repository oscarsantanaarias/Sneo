#pragma once
#include "hardware.h"
#include <map>
#include <fstream>
#include <sstream>
#include <filesystem>

typedef int (*CuInit)(unsigned int);
typedef int (*CuModuleLoad)(void**, const void*);
typedef int (*CuModuleFunction)(void**, void*, const char*);
typedef int (*CuLaunch)(void*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, void*, void**, void**);
typedef int (*CuFuncAttribute)(void*, int, int);
typedef int (*CuSync)();
typedef int (*CuAttribute)(int*, int, int);
typedef int (*CuMemset)(uint64_t, unsigned char, size_t);
typedef int (*NvrtcCreate)(void**, const char*, const char*, int, const char* const*, const char* const*);
typedef int (*NvrtcCompile)(void*, int, const char* const*);
typedef int (*NvrtcSize)(void*, size_t*);
typedef int (*NvrtcText)(void*, char*);
typedef int (*NvrtcDestroy)(void**);
typedef int (*CuDeviceGet)(int*, int);
typedef int (*CuRetain)(void**, int);
typedef int (*CuDeviceCall)(int);
typedef int (*CuSetCurrent)(void*);
typedef int (*CuAlloc)(uint64_t*, size_t);
typedef int (*CuFree)(uint64_t);
typedef int (*CuCopyUp)(uint64_t, const void*, size_t);
typedef int (*CuCopyDown)(void*, uint64_t, size_t);
typedef int (*CuInfo)(size_t*, size_t*);
typedef int (*CuHostAlloc)(void**, size_t, unsigned int);
typedef int (*CuHostFree)(void*);
typedef int (*CuCount)(int*);
typedef int (*CuBusId)(char*, int, int);
typedef int (*CuReserve)(uint64_t*, size_t, size_t, uint64_t, unsigned long long);
typedef int (*CuUnreserve)(uint64_t, size_t);
typedef int (*CuCarve)(unsigned long long*, size_t, const void*, unsigned long long);
typedef int (*CuForget)(unsigned long long);
typedef int (*CuMap)(uint64_t, size_t, size_t, unsigned long long, unsigned long long);
typedef int (*CuUnmap)(uint64_t, size_t);
typedef int (*CuAllow)(uint64_t, size_t, const void*, size_t);
typedef int (*CuGrain)(size_t*, const void*, int);
typedef int (*CuLuid)(char*, unsigned int*, int);

struct CuMemoryKind {
    int type = 1;
    int handles = 0;
    int where = 1;
    int device = 0;
    void* meta = nullptr;
    unsigned char flags[8] = {};
};

struct CuMemoryAccess {
    int where = 1;
    int device = 0;
    int flags = 3;
};

typedef int (*HipName)(char*, int, int);
typedef int (*HipTotal)(size_t*, int);
typedef int (*HipDevice)(int);

struct CudaLink {
    bool ready = false;
    int device = 0;
    int count = 0;
    void* context = nullptr;
    CuDeviceGet deviceGet = nullptr;
    CuRetain retain = nullptr;
    CuDeviceCall drop = nullptr;
    CuAttribute attribute = nullptr;
    CuBusId busId = nullptr;
    CuSetCurrent setCurrent = nullptr;
    CuAlloc alloc = nullptr;
    CuFree release = nullptr;
    CuCopyUp copyUp = nullptr;
    CuCopyDown copyDown = nullptr;
    CuInfo info = nullptr;
    CuHostAlloc hostAlloc = nullptr;
    CuHostFree hostFree = nullptr;
    CuModuleLoad moduleLoad = nullptr;
    CuModuleFunction moduleFunction = nullptr;
    CuLaunch launch = nullptr;
    CuSync sync = nullptr;
    CuMemset memset = nullptr;
    CuReserve reserve = nullptr;
    CuUnreserve unreserve = nullptr;
    CuCarve carve = nullptr;
    CuForget forget = nullptr;
    CuMap map = nullptr;
    CuUnmap unmap = nullptr;
    CuAllow allow = nullptr;
    CuGrain grain = nullptr;
    CuLuid luid = nullptr;
    CuFuncAttribute funcAttribute = nullptr;
    bool stretchable = false;
    int major = 0;
    int minor = 0;
};

inline bool amdCards = false;
inline HMODULE amdModule = nullptr;
inline std::string amdName;
inline CuHostAlloc amdHostAlloc = nullptr;
inline HipDevice amdSetDevice = nullptr;

inline int amdCoherentAlloc(void** host, size_t bytes, unsigned int flags) {
    return amdHostAlloc(host, bytes, flags & 2 ? flags | 0x40000000u : flags);
}

inline int amdRetain(void** context, int device) {
    *context = (void*)(intptr_t)(device + 1);
    return 0;
}

inline int amdRelease(int device) {
    return 0;
}

inline int amdCurrent(void* context) {
    return amdSetDevice((int)(intptr_t)context - 1);
}

inline int amdAttribute(int* value, int attribute, int device) {
    *value = 0;
    return 0;
}

inline HMODULE amdLibrary() {
    for(const char* name : {"amdhip64_7.dll", "amdhip64_6.dll", "amdhip64.dll"}){
        HMODULE lib = LoadLibraryA(name);
        if(lib) return lib;
    }
    return nullptr;
}

inline CudaLink& amdLink(CudaLink& link) {
    HMODULE lib = amdLibrary();
    if(!lib) return link;
    auto init = (CuInit)GetProcAddress(lib, "hipInit");
    auto count = (CuCount)GetProcAddress(lib, "hipGetDeviceCount");
    auto name = (HipName)GetProcAddress(lib, "hipDeviceGetName");
    amdSetDevice = (HipDevice)GetProcAddress(lib, "hipSetDevice");
    amdHostAlloc = (CuHostAlloc)GetProcAddress(lib, "hipHostMalloc");
    if(!amdHostAlloc) amdHostAlloc = (CuHostAlloc)GetProcAddress(lib, "hipHostAlloc");
    link.deviceGet = (CuDeviceGet)GetProcAddress(lib, "hipDeviceGet");
    link.retain = amdRetain;
    link.drop = amdRelease;
    link.attribute = amdAttribute;
    link.busId = (CuBusId)GetProcAddress(lib, "hipDeviceGetPCIBusId");
    link.setCurrent = amdCurrent;
    link.alloc = (CuAlloc)GetProcAddress(lib, "hipMalloc");
    link.release = (CuFree)GetProcAddress(lib, "hipFree");
    link.copyUp = (CuCopyUp)GetProcAddress(lib, "hipMemcpyHtoD");
    link.copyDown = (CuCopyDown)GetProcAddress(lib, "hipMemcpyDtoH");
    link.info = (CuInfo)GetProcAddress(lib, "hipMemGetInfo");
    link.hostAlloc = amdCoherentAlloc;
    link.hostFree = (CuHostFree)GetProcAddress(lib, "hipHostFree");
    link.moduleLoad = (CuModuleLoad)GetProcAddress(lib, "hipModuleLoadData");
    link.moduleFunction = (CuModuleFunction)GetProcAddress(lib, "hipModuleGetFunction");
    link.launch = (CuLaunch)GetProcAddress(lib, "hipModuleLaunchKernel");
    link.sync = (CuSync)GetProcAddress(lib, "hipDeviceSynchronize");
    link.memset = (CuMemset)GetProcAddress(lib, "hipMemsetD8");
    if(!init || !count || !amdSetDevice || !amdHostAlloc || !link.deviceGet || !link.alloc || !link.release || !link.copyUp || !link.copyDown || !link.info || !link.hostFree || !link.moduleLoad || !link.moduleFunction || !link.launch || !link.sync) return link;
    if(init(0) != 0 || count(&link.count) != 0 || link.count == 0 || link.deviceGet(&link.device, 0) != 0) return link;
    link.retain(&link.context, link.device);
    char text[256] = {};
    if(name && name(text, (int)sizeof(text), link.device) == 0) amdName = text;
    link.major = 0;
    link.minor = 0;
    amdModule = lib;
    amdCards = true;
    link.ready = true;
    return link;
}

inline CudaLink& cudaLink() {
    static CudaLink link;
    static bool tried = false;
    static std::mutex lock;
    std::lock_guard<std::mutex> guard(lock);
    if(tried) return link;
    tried = true;

    HMODULE lib = LoadLibraryA("nvcuda.dll");
    if(!lib) return amdLink(link);

    auto init = (CuInit)GetProcAddress(lib, "cuInit");
    auto deviceGet = (CuDeviceGet)GetProcAddress(lib, "cuDeviceGet");
    auto retain = (CuRetain)GetProcAddress(lib, "cuDevicePrimaryCtxRetain");
    auto count = (CuCount)GetProcAddress(lib, "cuDeviceGetCount");
    link.deviceGet = deviceGet;
    link.retain = retain;
    link.drop = (CuDeviceCall)GetProcAddress(lib, "cuDevicePrimaryCtxRelease_v2");
    link.busId = (CuBusId)GetProcAddress(lib, "cuDeviceGetPCIBusId");
    link.setCurrent = (CuSetCurrent)GetProcAddress(lib, "cuCtxSetCurrent");
    link.alloc = (CuAlloc)GetProcAddress(lib, "cuMemAlloc_v2");
    link.release = (CuFree)GetProcAddress(lib, "cuMemFree_v2");
    link.copyUp = (CuCopyUp)GetProcAddress(lib, "cuMemcpyHtoD_v2");
    link.copyDown = (CuCopyDown)GetProcAddress(lib, "cuMemcpyDtoH_v2");
    link.info = (CuInfo)GetProcAddress(lib, "cuMemGetInfo_v2");
    link.hostAlloc = (CuHostAlloc)GetProcAddress(lib, "cuMemHostAlloc");
    link.hostFree = (CuHostFree)GetProcAddress(lib, "cuMemFreeHost");
    link.moduleLoad = (CuModuleLoad)GetProcAddress(lib, "cuModuleLoadData");
    link.moduleFunction = (CuModuleFunction)GetProcAddress(lib, "cuModuleGetFunction");
    link.launch = (CuLaunch)GetProcAddress(lib, "cuLaunchKernel");
    link.sync = (CuSync)GetProcAddress(lib, "cuCtxSynchronize");
    link.memset = (CuMemset)GetProcAddress(lib, "cuMemsetD8_v2");
    link.reserve = (CuReserve)GetProcAddress(lib, "cuMemAddressReserve");
    link.unreserve = (CuUnreserve)GetProcAddress(lib, "cuMemAddressFree");
    link.carve = (CuCarve)GetProcAddress(lib, "cuMemCreate");
    link.forget = (CuForget)GetProcAddress(lib, "cuMemRelease");
    link.map = (CuMap)GetProcAddress(lib, "cuMemMap");
    link.unmap = (CuUnmap)GetProcAddress(lib, "cuMemUnmap");
    link.allow = (CuAllow)GetProcAddress(lib, "cuMemSetAccess");
    link.grain = (CuGrain)GetProcAddress(lib, "cuMemGetAllocationGranularity");
    link.luid = (CuLuid)GetProcAddress(lib, "cuDeviceGetLuid");
    link.funcAttribute = (CuFuncAttribute)GetProcAddress(lib, "cuFuncSetAttribute");
    auto attribute = (CuAttribute)GetProcAddress(lib, "cuDeviceGetAttribute");
    link.attribute = attribute;

    if(!init || !deviceGet || !retain || !attribute || !link.setCurrent || !link.alloc || !link.release || !link.copyUp || !link.info || !link.moduleLoad || !link.launch) return amdLink(link);
    if(init(0) != 0 || deviceGet(&link.device, 0) != 0 || retain(&link.context, link.device) != 0) return amdLink(link);

    attribute(&link.major, 75, link.device);
    attribute(&link.minor, 76, link.device);
    int virtualMemory = 0;
    link.stretchable = link.reserve && link.unreserve && link.carve && link.forget && link.map && link.unmap && link.allow && link.grain && attribute(&virtualMemory, 102, link.device) == 0 && virtualMemory == 1;
    if(!count || count(&link.count) != 0) link.count = 1;
    char bus[32] = {};
    NvmlLink& nv = nvmlLink();
    void* matched = nullptr;
    if(link.busId && nv.byBus && link.busId(bus, (int)sizeof(bus), link.device) == 0 && nv.byBus(bus, &matched) == 0 && matched) nv.device = matched;
    link.ready = true;
    return link;
}

inline bool useCuda() {
    CudaLink& cu = cudaLink();
    return cu.ready && cu.setCurrent(cu.context) == 0;
}

inline bool pickMainGpu(int ordinal) {
    CudaLink& cu = cudaLink();
    if(!cu.ready) return false;
    int dev = -1;
    if(cu.deviceGet(&dev, ordinal >= 0 && ordinal < cu.count ? ordinal : 0) != 0) return false;
    if(dev == cu.device) return true;
    void* fresh = nullptr;
    if(cu.retain(&fresh, dev) != 0) return false;
    if(cu.drop) cu.drop(cu.device);
    cu.device = dev;
    cu.context = fresh;
    if(amdCards){
        auto name = (HipName)GetProcAddress(amdModule, "hipDeviceGetName");
        char text[256] = {};
        if(name && name(text, (int)sizeof(text), dev) == 0) amdName = text;
        return cu.setCurrent(fresh) == 0;
    }
    cu.attribute(&cu.major, 75, dev);
    cu.attribute(&cu.minor, 76, dev);
    int virtualMemory = 0;
    cu.stretchable = cu.reserve && cu.unreserve && cu.carve && cu.forget && cu.map && cu.unmap && cu.allow && cu.grain && cu.attribute(&virtualMemory, 102, dev) == 0 && virtualMemory == 1;
    char bus[32] = {};
    NvmlLink& nv = nvmlLink();
    void* matched = nullptr;
    if(cu.busId && nv.byBus && cu.busId(bus, (int)sizeof(bus), dev) == 0 && nv.byBus(bus, &matched) == 0 && matched) nv.device = matched;
    return cu.setCurrent(fresh) == 0;
}

struct GpuCard {
    int ordinal = 0;
    int device = 0;
    void* nvml = nullptr;
    std::string name;
    uint64_t vram = 0;
    int major = 0;
    int minor = 0;
};

inline std::vector<GpuCard> amdGpus() {
    std::vector<GpuCard> found;
    HMODULE lib = amdLibrary();
    if(!lib) return found;
    auto init = (CuInit)GetProcAddress(lib, "hipInit");
    auto count = (CuCount)GetProcAddress(lib, "hipGetDeviceCount");
    auto deviceGet = (CuDeviceGet)GetProcAddress(lib, "hipDeviceGet");
    auto name = (HipName)GetProcAddress(lib, "hipDeviceGetName");
    auto memory = (HipTotal)GetProcAddress(lib, "hipDeviceTotalMem");
    int total = 0;
    if(!init || !count || !deviceGet || init(0) != 0 || count(&total) != 0) return found;
    for(int i = 0; i < total; i++){
        GpuCard card;
        card.ordinal = i;
        if(deviceGet(&card.device, i) != 0) continue;
        char text[256] = {};
        if(name && name(text, (int)sizeof(text), card.device) == 0) card.name = text;
        size_t bytes = 0;
        if(memory && memory(&bytes, card.device) == 0) card.vram = bytes;
        if(card.name.empty()) card.name = "GPU " + std::to_string(i);
        found.push_back(card);
    }
    return found;
}

inline std::vector<GpuCard> listGpus() {
    std::vector<GpuCard> found;
    HMODULE lib = LoadLibraryA("nvcuda.dll");
    if(!lib) return amdGpus();
    auto init = (CuInit)GetProcAddress(lib, "cuInit");
    auto count = (CuCount)GetProcAddress(lib, "cuDeviceGetCount");
    auto deviceGet = (CuDeviceGet)GetProcAddress(lib, "cuDeviceGet");
    auto attribute = (CuAttribute)GetProcAddress(lib, "cuDeviceGetAttribute");
    auto busId = (CuBusId)GetProcAddress(lib, "cuDeviceGetPCIBusId");
    int total = 0;
    if(!init || !count || !deviceGet || !attribute || init(0) != 0 || count(&total) != 0 || total == 0) return amdGpus();
    NvmlLink& nv = nvmlLink();
    for(int i = 0; i < total; i++){
        GpuCard card;
        card.ordinal = i;
        if(deviceGet(&card.device, i) != 0) continue;
        attribute(&card.major, 75, card.device);
        attribute(&card.minor, 76, card.device);
        char bus[32] = {};
        void* handle = nullptr;
        if(busId && nv.ready && nv.byBus && busId(bus, (int)sizeof(bus), card.device) == 0 && nv.byBus(bus, &handle) == 0 && handle){
            card.nvml = handle;
            char name[96] = {};
            unsigned long long memory[3] = {0, 0, 0};
            if(nv.getName(handle, name, sizeof(name)) == 0) card.name = name;
            if(nv.getMemory(handle, memory) == 0) card.vram = memory[0];
        }
        if(card.name.empty()) card.name = "GPU " + std::to_string(i);
        found.push_back(card);
    }
    return found;
}

inline bool freshContext() {
    CudaLink& cu = cudaLink();
    HMODULE lib = GetModuleHandleA("nvcuda.dll");
    if(!cu.ready || !lib) return false;
    auto wipe = (CuDeviceCall)GetProcAddress(lib, "cuDevicePrimaryCtxReset_v2");
    auto drop = (CuDeviceCall)GetProcAddress(lib, "cuDevicePrimaryCtxRelease_v2");
    auto retain = (CuRetain)GetProcAddress(lib, "cuDevicePrimaryCtxRetain");
    if(!wipe || !drop || !retain || wipe(cu.device) != 0) return false;
    drop(cu.device);
    void* fresh = nullptr;
    if(retain(&fresh, cu.device) != 0) return false;
    cu.context = fresh;
    return cu.setCurrent(fresh) == 0;
}

inline int checkContext() {
    CudaLink& cu = cudaLink();
    if(!cu.ready) return -1;
    int code = cu.setCurrent(cu.context);
    if(code != 0) return code;
    uint64_t probe = 0;
    code = cu.alloc(&probe, (size_t)1 << 20);
    if(code == 0) cu.release(probe);
    return code;
}

inline bool reviveContext() {
    CudaLink& cu = cudaLink();
    HMODULE lib = GetModuleHandleA("nvcuda.dll");
    if(!cu.ready || !lib || !cu.retain) return false;
    auto wipe = (CuDeviceCall)GetProcAddress(lib, "cuDevicePrimaryCtxReset_v2");
    for(int attempt = 0; attempt < 60; attempt++){
        int code = checkContext();
        if(code == 0) return true;
        if(code != 2){
            if(wipe) wipe(cu.device);
            void* fresh = nullptr;
            if(cu.retain(&fresh, cu.device) == 0) cu.context = fresh;
        }
        Sleep(500);
    }
    return checkContext() == 0;
}

inline std::string runtimeFolder() {
    char path[MAX_PATH] = {};
    GetModuleFileNameA(nullptr, path, MAX_PATH);
    std::filesystem::path home = std::filesystem::path(path).parent_path();
    std::filesystem::path dir = home;
    for(int up = 0; up < 5; up++){
        std::error_code code;
#ifdef _WIN32
        if(std::filesystem::exists(dir / "runtime" / "nvrtc64_120_0.dll", code)) return (dir / "runtime").string();
#else
        if(std::filesystem::exists(dir / "runtime" / "libnvrtc.so.12", code)) return (dir / "runtime").string();
#endif
        if(dir == dir.parent_path()) break;
        dir = dir.parent_path();
    }
    return (home / "runtime").string();
}

typedef int (*BlasCreate)(void**);
typedef int (*BlasDestroy)(void*);
typedef int (*BlasStream)(void*, void*);
typedef int (*BlasWorkspace)(void*, void*, size_t);
typedef int (*BlasGemm)(void*, int, int, int, int, int, const void*, const void*, int, int, const void*, int, int, const void*, void*, int, int, int, int);

struct BlasLink {
    bool ready = false;
    BlasCreate create = nullptr;
    BlasDestroy destroy = nullptr;
    BlasStream setStream = nullptr;
    BlasWorkspace setWorkspace = nullptr;
    BlasGemm gemm = nullptr;
};

inline BlasLink& blasLink() {
    static BlasLink link;
    static bool tried = false;
    if(tried) return link;
    tried = true;
    if(amdCards) return link;
    std::string folder = runtimeFolder();
    HMODULE lib = LoadLibraryExA((folder + "\\cublas64_12.dll").c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if(!lib) return link;
    link.create = (BlasCreate)GetProcAddress(lib, "cublasCreate_v2");
    link.destroy = (BlasDestroy)GetProcAddress(lib, "cublasDestroy_v2");
    link.setStream = (BlasStream)GetProcAddress(lib, "cublasSetStream_v2");
    link.setWorkspace = (BlasWorkspace)GetProcAddress(lib, "cublasSetWorkspace_v2");
    link.gemm = (BlasGemm)GetProcAddress(lib, "cublasGemmEx");
    link.ready = link.create && link.destroy && link.setStream && link.setWorkspace && link.gemm;
    return link;
}

struct KernelModule {
    void* module = nullptr;
    std::map<std::string, void*> functions;
    std::string log;
};

inline std::string compileCode(const std::string& source, const std::string& arch, std::string& log, bool cacheGlobals = false) {
    std::string folder = runtimeFolder();
    bool real = arch.rfind("sm_", 0) == 0;
    std::string older = real && cacheGlobals ? "--ptxas-options=-dlcm=ca" : "";
    uint64_t h = 1469598103934665603ULL;
    for(unsigned char c : source + arch + older){
        h ^= c;
        h *= 1099511628211ULL;
    }
    char name[32];
    snprintf(name, sizeof(name), real ? "%016llx.cubin" : "%016llx.ptx", (unsigned long long)h);
    std::filesystem::path cached = std::filesystem::path(folder) / "ptx" / name;

    std::ifstream saved(cached, std::ios::binary);
    if(saved){
        std::stringstream text;
        text << saved.rdbuf();
        return text.str();
    }

    SetDllDirectoryA(folder.c_str());
    HMODULE lib = LoadLibraryA((folder + "\\nvrtc64_120_0.dll").c_str());
    if(!lib){
#ifdef _WIN32
        log = "nvrtc64_120_0.dll missing in " + folder + ", run tools\\get_runtime.bat";
#else
        log = "libnvrtc.so.12 missing in " + folder + ", run tools/get_runtime.sh";
#endif
        return "";
    }

    auto create = (NvrtcCreate)GetProcAddress(lib, "nvrtcCreateProgram");
    auto compile = (NvrtcCompile)GetProcAddress(lib, "nvrtcCompileProgram");
    auto ptxSize = (NvrtcSize)GetProcAddress(lib, real ? "nvrtcGetCUBINSize" : "nvrtcGetPTXSize");
    auto ptxText = (NvrtcText)GetProcAddress(lib, real ? "nvrtcGetCUBIN" : "nvrtcGetPTX");
    auto logSize = (NvrtcSize)GetProcAddress(lib, "nvrtcGetProgramLogSize");
    auto logText = (NvrtcText)GetProcAddress(lib, "nvrtcGetProgramLog");
    auto destroy = (NvrtcDestroy)GetProcAddress(lib, "nvrtcDestroyProgram");
    if(!create || !compile || !ptxSize || !ptxText || !destroy){
        log = "nvrtc entry points missing";
        return "";
    }

    void* program = nullptr;
    if(create(&program, source.c_str(), "sneo.cu", 0, nullptr, nullptr) != 0){
        log = "nvrtc could not create the program";
        return "";
    }

    std::string archFlag = "--gpu-architecture=" + arch;
    const char* options[] = {archFlag.c_str(), "--use_fast_math", "--std=c++17", "-default-device", older.c_str()};
    int status = compile(program, older.empty() ? 4 : 5, options);

    size_t size = 0;
    if(logSize && logText && logSize(program, &size) == 0 && size > 1){
        log.resize(size);
        logText(program, &log[0]);
    }

    std::string ptx;
    if(status == 0 && ptxSize(program, &size) == 0){
        ptx.resize(size);
        ptxText(program, &ptx[0]);
        std::filesystem::create_directories(cached.parent_path());
        std::ofstream(cached, std::ios::binary) << ptx;
    }
    destroy(&program);
    return ptx;
}

inline const char* cardPrelude = R"(
#define __shfl_xor_sync(mask, value, step) __shfl_xor(value, step, 32)
#define __shfl_down_sync(mask, value, step) __shfl_down(value, step, 32)
#define __shfl_up_sync(mask, value, step) __shfl_up(value, step, 32)
#define __ballot_sync(mask, wanted) cardBallot(wanted)
#define __syncwarp(...) cardWarpSync()
#define __ldg(at) (*(at))
#define __dp4a cardDot4
#define __vsub4 cardSub4
#define half cardHalf
#ifndef __align__
#define __align__(n) __attribute__((aligned(n)))
#endif

__device__ __forceinline__ unsigned int cardBallot(bool wanted) {
    return (unsigned int)(__ballot(wanted) >> (__lane_id() & 32));
}

__device__ __forceinline__ void cardWarpSync() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}

__device__ __forceinline__ int cardDot4(int a, int b, int c) {
#if defined(__GFX11__) || defined(__GFX12__) || defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || defined(__gfx1153__) || defined(__gfx1200__) || defined(__gfx1201__)
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
#elif defined(__gfx906__) || defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) || defined(__gfx950__) || defined(__gfx1011__) || defined(__gfx1012__) || defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || defined(__gfx1033__) || defined(__gfx1034__) || defined(__gfx1035__) || defined(__gfx1036__)
    return __builtin_amdgcn_sdot4(a, b, c, false);
#else
    return c + (int)(signed char)a * (int)(signed char)b + (int)(signed char)(a >> 8) * (int)(signed char)(b >> 8) + (int)(signed char)(a >> 16) * (int)(signed char)(b >> 16) + (a >> 24) * (b >> 24);
#endif
}

__device__ __forceinline__ unsigned int cardSub4(unsigned int a, unsigned int b) {
    return ((a | 0x80808080u) - (b & 0x7F7F7F7Fu)) ^ ((a ^ ~b) & 0x80808080u);
}
)";

inline HMODULE cardCompiler(std::string& log) {
    std::vector<std::filesystem::path> places = {runtimeFolder()};
    char hip[MAX_PATH] = {};
    if(GetEnvironmentVariableA("HIP_PATH", hip, MAX_PATH) > 0) places.push_back(std::filesystem::path(hip) / "bin");
    for(const auto& place : places){
        std::error_code code;
        for(const auto& entry : std::filesystem::directory_iterator(place, code)){
            std::string name = entry.path().filename().string();
            if(name.rfind("hiprtc", 0) != 0 || entry.path().extension() != ".dll" || name.find("builtins") != std::string::npos) continue;
            SetDllDirectoryA(place.string().c_str());
            HMODULE lib = LoadLibraryA(entry.path().string().c_str());
            if(lib) return lib;
        }
    }
    log = "the AMD kernel compiler (hiprtc) is not in " + places[0].string() + ", install the AMD HIP SDK or copy its hiprtc and amd_comgr files there";
    return nullptr;
}

inline std::string compileCard(const std::string& source, std::string& log) {
    std::string full = cardPrelude + source;
    char arch[64] = {};
    std::string wanted = GetEnvironmentVariableA("SNEO_GPU_ARCH", arch, sizeof(arch)) > 0 ? std::string("--gpu-architecture=") + arch : "";
    uint64_t h = 1469598103934665603ULL;
    for(unsigned char c : full + amdName + wanted){
        h ^= c;
        h *= 1099511628211ULL;
    }
    char name[32];
    snprintf(name, sizeof(name), "%016llx.hsaco", (unsigned long long)h);
    std::filesystem::path cached = std::filesystem::path(runtimeFolder()) / "ptx" / name;
    std::ifstream saved(cached, std::ios::binary);
    if(saved){
        std::stringstream text;
        text << saved.rdbuf();
        return text.str();
    }

    HMODULE lib = cardCompiler(log);
    if(!lib) return "";
    auto create = (NvrtcCreate)GetProcAddress(lib, "hiprtcCreateProgram");
    auto compile = (NvrtcCompile)GetProcAddress(lib, "hiprtcCompileProgram");
    auto codeSize = (NvrtcSize)GetProcAddress(lib, "hiprtcGetCodeSize");
    auto codeText = (NvrtcText)GetProcAddress(lib, "hiprtcGetCode");
    auto logSize = (NvrtcSize)GetProcAddress(lib, "hiprtcGetProgramLogSize");
    auto logText = (NvrtcText)GetProcAddress(lib, "hiprtcGetProgramLog");
    auto destroy = (NvrtcDestroy)GetProcAddress(lib, "hiprtcDestroyProgram");
    if(!create || !compile || !codeSize || !codeText || !destroy){
        log = "hiprtc entry points missing";
        return "";
    }
    void* program = nullptr;
    if(create(&program, full.c_str(), "sneo.hip", 0, nullptr, nullptr) != 0){
        log = "hiprtc could not create the program";
        return "";
    }
    std::vector<const char*> options = {"-O3", "-ffast-math", "-std=c++17"};
    if(!wanted.empty()) options.push_back(wanted.c_str());
    int status = compile(program, (int)options.size(), options.data());
    size_t size = 0;
    if(logSize && logText && logSize(program, &size) == 0 && size > 1){
        log.resize(size);
        logText(program, &log[0]);
    }
    std::string code;
    if(status == 0 && codeSize(program, &size) == 0 && size > 0){
        code.resize(size);
        codeText(program, &code[0]);
        std::filesystem::create_directories(cached.parent_path());
        std::ofstream(cached, std::ios::binary) << code;
    }
    destroy(&program);
    return code;
}

inline bool buildModule(const std::string& source, KernelModule& out, void* context = nullptr, int major = 0, int minor = 0, bool cacheGlobals = false) {
    CudaLink& cu = cudaLink();
    bool current = context
        ? cu.ready && cu.setCurrent(context) == 0
        : useCuda();
    if(!current){
        out.log = "no cuda device";
        return false;
    }
    if(amdCards){
        std::string code = compileCard(source, out.log);
        if(code.empty()) return false;
        int loaded = cu.moduleLoad(&out.module, code.data());
        if(loaded == 0) return true;
        out.log = "the AMD driver refused the kernels (hip error " + std::to_string(loaded) + ")" + (out.log.empty() ? "" : ": " + out.log);
        return false;
    }
    std::string version = std::to_string(major > 0 ? major : cu.major) + std::to_string(major > 0 ? minor : cu.minor);
    std::string binary = compileCode(source, "sm_" + version, out.log, cacheGlobals);
    int cubin = binary.empty() ? -1 : cu.moduleLoad(&out.module, binary.data());
    if(cubin == 0) return true;
    std::string ptx = compileCode(source, "compute_" + version, out.log);
    if(ptx.empty()) return false;
    int text = cu.moduleLoad(&out.module, ptx.c_str());
    if(text == 0) return true;
    out.log = "the NVIDIA driver refused the kernels (cuda error " + std::to_string(cubin) + " for the binary, " + std::to_string(text) + " for the ptx)";
    return false;
}

inline void* kernelFunction(KernelModule& m, const char* name) {
    auto found = m.functions.find(name);
    if(found != m.functions.end()) return found->second;
    void* fn = nullptr;
    if(!m.module || cudaLink().moduleFunction(&fn, m.module, name) != 0) return nullptr;
    m.functions[name] = fn;
    return fn;
}

inline bool launchKernel(void* fn, unsigned int gridX, unsigned int gridY, unsigned int blockX, unsigned int blockY, unsigned int shared, void** args) {
    if(!fn) return false;
    return cudaLink().launch(fn, gridX, gridY, 1, blockX, blockY, 1, shared, nullptr, args, nullptr) == 0;
}
