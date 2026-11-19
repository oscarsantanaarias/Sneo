#pragma once
#include "platform.h"
#ifdef _WIN32
#include <pdh.h>
#include <winioctl.h>
#endif
#include <cstdint>
#include <cstring>
#include <string>
#include <mutex>
#include <atomic>
#include <filesystem>
#include <vector>
#include <algorithm>

#pragma comment(lib, "pdh.lib")

inline std::atomic<bool> stopGenerating{false};
#pragma comment(lib, "advapi32.lib")

// windows baja la velocidad de los programas que no tienen la ventana adelante: aca se le pide que no; linux no hace eso
inline void keepFullSpeed() {
#ifdef _WIN32
    PROCESS_POWER_THROTTLING_STATE state = {};
    state.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
    state.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
    state.StateMask = 0;
    SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &state, sizeof(state));
#endif
}

inline void keepThreadFullSpeed() {
#ifdef _WIN32
    THREAD_POWER_THROTTLING_STATE state = {};
    state.Version = THREAD_POWER_THROTTLING_CURRENT_VERSION;
    state.ControlMask = THREAD_POWER_THROTTLING_EXECUTION_SPEED;
    state.StateMask = 0;
    SetThreadInformation(GetCurrentThread(), ThreadPowerThrottling, &state, sizeof(state));
#endif
}

inline std::string missingCpuFeatures() {
    int info[4] = {};
    __cpuid(info, 0);
    int highest = info[0];
    __cpuid(info, 1);
    bool fma = (info[2] >> 12) & 1;
    bool saved = (info[2] >> 27) & 1;
    bool avx = (info[2] >> 28) & 1;
    bool half = (info[2] >> 29) & 1;
    bool enabled = saved && (_xgetbv(0) & 6) == 6;
    bool avx2 = false;
    if(highest >= 7){
        __cpuidex(info, 7, 0);
        avx2 = (info[1] >> 5) & 1;
    }
    std::string missing;
    if(!avx || !enabled) missing += " AVX";
    if(!avx2) missing += " AVX2";
    if(!fma) missing += " FMA";
    if(!half) missing += " F16C";
    return missing;
}

struct GpuStats {
    uint32_t present;
    char name[96];
    uint64_t vramTotal;
    uint64_t vramFree;
    uint32_t load;
    uint32_t memoryLoad;
    uint32_t temperature;
    uint32_t powerMilliwatts;
    uint32_t powerLimitMilliwatts;
    uint32_t clockMhz;
    uint32_t fanPercent;
    uint32_t pcieGen;
    uint32_t pcieWidth;
    uint32_t pcieTxKbs;
    uint32_t pcieRxKbs;
    uint32_t pad;
    uint64_t energyMillijoules;
};

struct HostStats {
    char nodeName[64];
    char cpuName[96];
    uint32_t cores;
    uint32_t threads;
    float cpuLoad;
    uint32_t driveKind;
    uint64_t ramTotal;
    uint64_t ramFree;
    uint64_t diskReadBytes;
    uint64_t diskWriteBytes;
    uint64_t diskFree;
    uint64_t diskTotal;
    uint64_t heldGpu;
    uint64_t heldRam;
    uint64_t loadTotal;
    uint64_t loadDone;
};

struct NodeStats {
    HostStats host;
    GpuStats gpu;
};

// los textos que llegan de otra pc siempre terminan dentro de su espacio
inline void sealStats(NodeStats& stats) {
    stats.host.nodeName[sizeof(stats.host.nodeName) - 1] = 0;
    stats.host.cpuName[sizeof(stats.host.cpuName) - 1] = 0;
    stats.gpu.name[sizeof(stats.gpu.name) - 1] = 0;
}

typedef int (*NvmlPlain)();
typedef int (*NvmlCount)(unsigned int*);
typedef int (*NvmlHandle)(unsigned int, void**);
typedef int (*NvmlName)(void*, char*, unsigned int);
typedef int (*NvmlMemory)(void*, unsigned long long*);
typedef int (*NvmlValue)(void*, unsigned int*);
typedef int (*NvmlPick)(void*, int, unsigned int*);
typedef int (*NvmlEnergy)(void*, uint64_t*);
typedef int (*NvmlByBus)(const char*, void**);

struct NvmlLink {
    bool ready = false;
    void* device = nullptr;
    NvmlCount count = nullptr;
    NvmlHandle byIndex = nullptr;
    NvmlByBus byBus = nullptr;
    NvmlName getName = nullptr;
    NvmlMemory getMemory = nullptr;
    NvmlValue getUtilization = nullptr;
    NvmlPick getTemperature = nullptr;
    NvmlValue getPower = nullptr;
    NvmlValue getPowerLimit = nullptr;
    NvmlPick getClock = nullptr;
    NvmlValue getFan = nullptr;
    NvmlValue getPcieGen = nullptr;
    NvmlValue getPcieWidth = nullptr;
    NvmlPick getPcieThroughput = nullptr;
    NvmlValue getDisplayActive = nullptr;
    NvmlEnergy getEnergy = nullptr;
};

inline NvmlLink& nvmlLink() {
    static NvmlLink link;
    static bool tried = false;
    if(tried) return link;
    tried = true;

    HMODULE lib = LoadLibraryA("nvml.dll");
    if(!lib) lib = LoadLibraryA("C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvml.dll");
    if(!lib) return link;

    auto init = (NvmlPlain)GetProcAddress(lib, "nvmlInit_v2");
    auto count = (NvmlCount)GetProcAddress(lib, "nvmlDeviceGetCount_v2");
    auto handle = (NvmlHandle)GetProcAddress(lib, "nvmlDeviceGetHandleByIndex_v2");
    if(!init || !count || !handle || init() != 0) return link;

    unsigned int total = 0;
    if(count(&total) != 0 || total == 0) return link;
    if(handle(0, &link.device) != 0) return link;

    link.count = count;
    link.byIndex = handle;
    link.byBus = (NvmlByBus)GetProcAddress(lib, "nvmlDeviceGetHandleByPciBusId_v2");
    link.getName = (NvmlName)GetProcAddress(lib, "nvmlDeviceGetName");
    link.getMemory = (NvmlMemory)GetProcAddress(lib, "nvmlDeviceGetMemoryInfo");
    link.getUtilization = (NvmlValue)GetProcAddress(lib, "nvmlDeviceGetUtilizationRates");
    link.getTemperature = (NvmlPick)GetProcAddress(lib, "nvmlDeviceGetTemperature");
    link.getPower = (NvmlValue)GetProcAddress(lib, "nvmlDeviceGetPowerUsage");
    link.getPowerLimit = (NvmlValue)GetProcAddress(lib, "nvmlDeviceGetEnforcedPowerLimit");
    link.getClock = (NvmlPick)GetProcAddress(lib, "nvmlDeviceGetClockInfo");
    link.getFan = (NvmlValue)GetProcAddress(lib, "nvmlDeviceGetFanSpeed");
    link.getPcieGen = (NvmlValue)GetProcAddress(lib, "nvmlDeviceGetCurrPcieLinkGeneration");
    link.getPcieWidth = (NvmlValue)GetProcAddress(lib, "nvmlDeviceGetCurrPcieLinkWidth");
    link.getPcieThroughput = (NvmlPick)GetProcAddress(lib, "nvmlDeviceGetPcieThroughput");
    link.getDisplayActive = (NvmlValue)GetProcAddress(lib, "nvmlDeviceGetDisplayActive");
    link.getEnergy = (NvmlEnergy)GetProcAddress(lib, "nvmlDeviceGetTotalEnergyConsumption");
    link.ready = link.getName && link.getMemory;
    return link;
}

inline int gpuShowsScreen() {
    NvmlLink& nv = nvmlLink();
    unsigned int active = 0;
    if(!nv.ready || !nv.getDisplayActive || nv.getDisplayActive(nv.device, &active) != 0) return -1;
    return active ? 1 : 0;
}

inline GpuStats readGpuOf(void* device) {
    GpuStats gpu = {};
    strcpy_s(gpu.name, "None");

    NvmlLink& nv = nvmlLink();
    if(!nv.ready || !device) return gpu;

    nv.getName(device, gpu.name, sizeof(gpu.name));
    unsigned long long memory[3] = {0, 0, 0};
    if(nv.getMemory(device, memory) == 0){
        gpu.vramTotal = memory[0];
        gpu.vramFree = memory[1];
        gpu.present = 1;
    }

    unsigned int rates[2] = {0, 0};
    if(nv.getUtilization && nv.getUtilization(device, rates) == 0){
        gpu.load = rates[0];
        gpu.memoryLoad = rates[1];
    }

    if(nv.getTemperature) nv.getTemperature(device, 0, &gpu.temperature);
    if(nv.getPower) nv.getPower(device, &gpu.powerMilliwatts);
    if(nv.getPowerLimit) nv.getPowerLimit(device, &gpu.powerLimitMilliwatts);
    if(nv.getClock) nv.getClock(device, 0, &gpu.clockMhz);
    if(nv.getFan) nv.getFan(device, &gpu.fanPercent);
    if(nv.getPcieGen) nv.getPcieGen(device, &gpu.pcieGen);
    if(nv.getPcieWidth) nv.getPcieWidth(device, &gpu.pcieWidth);
    if(nv.getPcieThroughput){
        nv.getPcieThroughput(device, 0, &gpu.pcieTxKbs);
        nv.getPcieThroughput(device, 1, &gpu.pcieRxKbs);
    }
    if(nv.getEnergy) nv.getEnergy(device, &gpu.energyMillijoules);
    return gpu;
}

inline GpuStats readGpu() {
    return readGpuOf(nvmlLink().device);
}

inline std::vector<std::pair<void*, GpuStats>> readOtherGpus() {
    std::vector<std::pair<void*, GpuStats>> found;
    NvmlLink& nv = nvmlLink();
    unsigned int total = 0;
    if(!nv.ready || !nv.count || !nv.byIndex || nv.count(&total) != 0) return found;
    for(unsigned int i = 0; i < total; i++){
        void* device = nullptr;
        if(nv.byIndex(i, &device) != 0 || !device || device == nv.device) continue;
        found.push_back({device, readGpuOf(device)});
    }
    return found;
}

inline uint64_t fileTimeValue(FILETIME t) {
    return ((uint64_t)t.dwHighDateTime << 32) | t.dwLowDateTime;
}

enum DriveKind : uint32_t { DriveUnknown, DriveHdd, DriveSsd, DriveNvme, DriveUsb };

#ifndef _WIN32
// en linux el disco de la carpeta sale de /sys: el dispositivo del archivo, nvme por el nombre y disco con aguja por rotational
inline uint32_t readDriveKind(const std::string& folder) {
    struct stat st;
    if(stat(folder.empty() ? "." : folder.c_str(), &st) != 0) return DriveUnknown;
    std::string link = "/sys/dev/block/" + std::to_string(major(st.st_dev)) + ":" + std::to_string(minor(st.st_dev));
    char real[PATH_MAX] = {};
    if(!realpath(link.c_str(), real)) return DriveUnknown;
    std::filesystem::path device(real);
    std::error_code e;
    if(!std::filesystem::exists(device / "queue", e)) device = device.parent_path();
    std::string name = device.filename().string();
    if(name.rfind("nvme", 0) == 0) return DriveNvme;
    if(std::string(real).find("/usb") != std::string::npos) return DriveUsb;
    FILE* f = fopen((device / "queue" / "rotational").string().c_str(), "r");
    if(!f) return DriveUnknown;
    int spinning = -1;
    if(fscanf(f, "%d", &spinning) != 1) spinning = -1;
    fclose(f);
    return spinning < 0 ? DriveUnknown : spinning ? DriveHdd : DriveSsd;
}
#else
inline uint32_t readDriveKind(const std::string& folder) {
    char full[MAX_PATH] = {};
    char root[MAX_PATH] = {};
    char volume[MAX_PATH] = {};
    if(!GetFullPathNameA(folder.empty() ? "." : folder.c_str(), MAX_PATH, full, nullptr)) return DriveUnknown;
    if(!GetVolumePathNameA(full, root, MAX_PATH)) return DriveUnknown;
    if(!GetVolumeNameForVolumeMountPointA(root, volume, MAX_PATH)) return DriveUnknown;

    size_t length = strlen(volume);
    if(length && volume[length - 1] == '\\') volume[length - 1] = 0;

    HANDLE device = CreateFileA(volume, 0, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, 0, nullptr);
    if(device == INVALID_HANDLE_VALUE) return DriveUnknown;

    STORAGE_PROPERTY_QUERY query = {};
    query.PropertyId = StorageDeviceProperty;
    query.QueryType = PropertyStandardQuery;
    BYTE describe[1024] = {};
    DWORD got = 0;
    STORAGE_BUS_TYPE bus = BusTypeUnknown;
    if(DeviceIoControl(device, IOCTL_STORAGE_QUERY_PROPERTY, &query, sizeof(query), describe, sizeof(describe), &got, nullptr) && got >= sizeof(STORAGE_DEVICE_DESCRIPTOR)){
        bus = ((STORAGE_DEVICE_DESCRIPTOR*)describe)->BusType;
    }

    query.PropertyId = StorageDeviceSeekPenaltyProperty;
    DEVICE_SEEK_PENALTY_DESCRIPTOR seek = {};
    bool seekKnown = DeviceIoControl(device, IOCTL_STORAGE_QUERY_PROPERTY, &query, sizeof(query), &seek, sizeof(seek), &got, nullptr) && got >= sizeof(seek);
    CloseHandle(device);

    if(bus == BusTypeNvme) return DriveNvme;
    if(bus == BusTypeUsb) return DriveUsb;
    if(!seekKnown) return DriveUnknown;
    return seek.IncursSeekPenalty
        ? DriveHdd
        : DriveSsd;
}
#endif

inline const char* driveName(uint32_t kind) {
    static const char* names[] = {"unknown", "hdd", "ssd", "nvme", "usb"};
    return kind < 5 ? names[kind] : names[0];
}

inline std::vector<GROUP_AFFINITY> coresFor(const std::string& plan) {
    if(plan == "all") return {};
    DWORD bytes = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &bytes);
    std::string buffer(bytes, '\0');
    if(!bytes || !GetLogicalProcessorInformationEx(RelationProcessorCore, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)buffer.data(), &bytes)) return {};
    BYTE top = 0, bottom = 255;
    for(DWORD at = 0; at < bytes;){
        auto item = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)(buffer.data() + at);
        top = std::max(top, item->Processor.EfficiencyClass);
        bottom = std::min(bottom, item->Processor.EfficiencyClass);
        at += item->Size;
    }
    if(plan == "auto" && top == bottom) return {};
    std::vector<GROUP_AFFINITY> first, second;
    for(DWORD at = 0; at < bytes;){
        auto item = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)(buffer.data() + at);
        at += item->Size;
        if(item->Processor.EfficiencyClass != top) continue;
        GROUP_AFFINITY whole = item->Processor.GroupMask[0];
        int seen = 0;
        for(int bit = 0; bit < 64; bit++){
            if(!(whole.Mask & (1ULL << bit))) continue;
            GROUP_AFFINITY one = {};
            one.Group = whole.Group;
            one.Mask = 1ULL << bit;
            (seen++ == 0 ? first : second).push_back(one);
        }
    }
    if(plan == "performance-ht") first.insert(first.end(), second.begin(), second.end());
    return first;
}

inline HostStats readHost(const std::string& modelFolder) {
    static std::mutex lock;
    static bool first = true;
    static HostStats fixed = {};
    static uint64_t lastIdle = 0, lastKernel = 0, lastUser = 0;
#ifdef _WIN32
    static PDH_HQUERY query = nullptr;
    static PDH_HCOUNTER readCounter = nullptr, writeCounter = nullptr;
#endif
    static std::string counterDrive;
    static std::string driveFolder;
    static uint32_t driveKind = DriveUnknown;
    static uint64_t driveAt = 0;

    std::lock_guard<std::mutex> guard(lock);

    if(first){
        first = false;
        DWORD size = sizeof(fixed.nodeName);
        GetComputerNameA(fixed.nodeName, &size);

#ifdef _WIN32
        DWORD nameSize = sizeof(fixed.cpuName);
        RegGetValueA(HKEY_LOCAL_MACHINE, "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", "ProcessorNameString", RRF_RT_REG_SZ, nullptr, fixed.cpuName, &nameSize);
#else
        // en linux el nombre y los nucleos fisicos salen de /proc/cpuinfo: pares distintos de paquete y nucleo
        if(FILE* info = fopen("/proc/cpuinfo", "r")){
            char line[512];
            int package = 0;
            std::vector<std::pair<int, int>> seen;
            while(fgets(line, sizeof(line), info)){
                if(!fixed.cpuName[0] && strncmp(line, "model name", 10) == 0){
                    const char* value = strchr(line, ':');
                    if(value) snprintf(fixed.cpuName, sizeof(fixed.cpuName), "%s", value + 2);
                    size_t end = strlen(fixed.cpuName);
                    if(end && fixed.cpuName[end - 1] == '\n') fixed.cpuName[end - 1] = 0;
                }
                int value = 0;
                if(sscanf(line, "physical id : %d", &value) == 1) package = value;
                if(sscanf(line, "core id : %d", &value) == 1 && std::find(seen.begin(), seen.end(), std::make_pair(package, value)) == seen.end()) seen.push_back({package, value});
            }
            fclose(info);
            fixed.cores = (uint32_t)seen.size();
        }
#endif

        DWORD bytes = 0;
        GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &bytes);
        std::string buffer(bytes, '\0');
        if(GetLogicalProcessorInformationEx(RelationProcessorCore, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)buffer.data(), &bytes)){
            DWORD offset = 0;
            while(offset < bytes){
                auto item = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)(buffer.data() + offset);
                fixed.cores++;
                offset += item->Size;
            }
        }
        fixed.threads = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);

#ifdef _WIN32
        PdhOpenQueryA(nullptr, 0, &query);
#endif

        FILETIME idle, kernel, user;
        GetSystemTimes(&idle, &kernel, &user);
        lastIdle = fileTimeValue(idle);
        lastKernel = fileTimeValue(kernel);
        lastUser = fileTimeValue(user);
    }

    HostStats host = fixed;

    FILETIME idle, kernel, user;
    if(GetSystemTimes(&idle, &kernel, &user)){
        uint64_t idleNow = fileTimeValue(idle);
        uint64_t kernelNow = fileTimeValue(kernel);
        uint64_t userNow = fileTimeValue(user);
        uint64_t busy = (kernelNow - lastKernel) + (userNow - lastUser);
        uint64_t idleSpan = idleNow - lastIdle;
        host.cpuLoad = busy > 0 ? (float)(busy - idleSpan) * 100.0f / (float)busy : 0.0f;
        lastIdle = idleNow;
        lastKernel = kernelNow;
        lastUser = userNow;
    }

    MEMORYSTATUSEX memory = {};
    memory.dwLength = sizeof(memory);
    if(GlobalMemoryStatusEx(&memory)){
        host.ramTotal = memory.ullTotalPhys;
        host.ramFree = memory.ullAvailPhys;
    }

#ifdef _WIN32
    std::string drive = std::filesystem::absolute(modelFolder.empty() ? "." : modelFolder).root_name().string();
    if(query && drive != counterDrive){
        counterDrive = drive;
        if(readCounter) PdhRemoveCounter(readCounter);
        if(writeCounter) PdhRemoveCounter(writeCounter);
        readCounter = writeCounter = nullptr;
        std::string scope = drive.size() == 2 && drive[1] == ':' ? "\\LogicalDisk(" + drive + ")" : "\\PhysicalDisk(_Total)";
        PdhAddEnglishCounterA(query, (scope + "\\Disk Read Bytes/sec").c_str(), 0, &readCounter);
        PdhAddEnglishCounterA(query, (scope + "\\Disk Write Bytes/sec").c_str(), 0, &writeCounter);
        PdhCollectQueryData(query);
    }

    if(query && PdhCollectQueryData(query) == ERROR_SUCCESS){
        PDH_FMT_COUNTERVALUE value = {};
        if(readCounter && PdhGetFormattedCounterValue(readCounter, PDH_FMT_LARGE, nullptr, &value) == ERROR_SUCCESS){
            host.diskReadBytes = (uint64_t)value.largeValue;
        }
        if(writeCounter && PdhGetFormattedCounterValue(writeCounter, PDH_FMT_LARGE, nullptr, &value) == ERROR_SUCCESS){
            host.diskWriteBytes = (uint64_t)value.largeValue;
        }
    }
#endif

    uint64_t tick = GetTickCount64();
    if(driveAt == 0 || modelFolder != driveFolder || tick - driveAt > 60000){
        driveFolder = modelFolder;
        driveKind = readDriveKind(modelFolder);
        driveAt = tick;
    }
    host.driveKind = driveKind;

    ULARGE_INTEGER freeBytes = {}, totalBytes = {};
    if(GetDiskFreeSpaceExA(modelFolder.empty() ? "." : modelFolder.c_str(), &freeBytes, &totalBytes, nullptr)){
        host.diskFree = freeBytes.QuadPart;
        host.diskTotal = totalBytes.QuadPart;
    }

    return host;
}

inline NodeStats readNode(const std::string& modelFolder) {
    NodeStats stats = {};
    stats.host = readHost(modelFolder);
    stats.gpu = readGpu();
    return stats;
}
