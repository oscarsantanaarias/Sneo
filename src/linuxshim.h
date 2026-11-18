#pragma once
// las llamadas de windows que usa el motor hechas con linux: asi el resto del codigo no cambia
// lo que en linux no tiene sentido (presupuesto de video de wddm, contadores pdh) contesta que no esta y el motor sigue sin eso
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <strings.h>
#include <climits>
#include <algorithm>
#include <cerrno>
#include <string>
#include <map>
#include <mutex>
#include <thread>
#include <fcntl.h>
#include <unistd.h>
#include <dlfcn.h>
#include <sched.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/sysmacros.h>
#include <signal.h>
#include <sys/wait.h>
#include <sys/statvfs.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <cpuid.h>
#include <immintrin.h>

typedef uint32_t DWORD;
typedef int BOOL;
typedef uint8_t BYTE;
typedef uint16_t WORD;
typedef unsigned long ULONG;
typedef long LONG;
typedef size_t SIZE_T;
typedef uintptr_t ULONG_PTR;
typedef uintptr_t DWORD_PTR;
typedef uint64_t ULONGLONG;
typedef int64_t LONGLONG;
typedef void* PVOID;
typedef void* LPVOID;
typedef void* HANDLE;
typedef void* HMODULE;
typedef void* FARPROC;
typedef int SOCKET;
typedef const char* LPCSTR;
typedef char* LPSTR;
typedef BYTE BOOLEAN;
typedef WORD KAFFINITY_WORD;
typedef uint64_t KAFFINITY;

#ifndef TRUE
#define TRUE 1
#define FALSE 0
#endif
#define MAX_PATH 4096
#define WINAPI
#define INVALID_HANDLE_VALUE ((HANDLE)(intptr_t)-1)
#define INVALID_SOCKET (-1)
#define SOCKET_ERROR (-1)
#define INFINITE 0xFFFFFFFFu
#define ERROR_SUCCESS 0
#define ERROR_INSUFFICIENT_BUFFER 122
#define ERROR_BUFFER_OVERFLOW 111
#define ERROR_NOT_LOCKED 158
#define ERROR_IO_PENDING 997

typedef union _LARGE_INTEGER {
    struct { DWORD LowPart; LONG HighPart; };
    LONGLONG QuadPart;
} LARGE_INTEGER;

typedef union _ULARGE_INTEGER {
    struct { DWORD LowPart; DWORD HighPart; };
    ULONGLONG QuadPart;
} ULARGE_INTEGER;

// ---- tiempo, entorno y proceso ----

inline ULONGLONG GetTickCount64() {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (ULONGLONG)t.tv_sec * 1000ULL + (ULONGLONG)t.tv_nsec / 1000000ULL;
}

inline void Sleep(DWORD ms) { usleep((useconds_t)ms * 1000); }
inline HANDLE GetCurrentProcess() { return (HANDLE)(intptr_t)-1; }
inline HANDLE GetCurrentThread() { return (HANDLE)(intptr_t)-2; }
inline DWORD GetCurrentProcessId() { return (DWORD)getpid(); }
inline DWORD GetCurrentThreadId() { return (DWORD)gettid(); }
inline DWORD GetLastError() { return (DWORD)errno; }
inline int WSAGetLastError() { return errno; }
inline DWORD GetActiveProcessorCount(WORD) { long n = sysconf(_SC_NPROCESSORS_ONLN); return n > 0 ? (DWORD)n : 1; }
#define ALL_PROCESSOR_GROUPS 0xFFFF

inline DWORD GetEnvironmentVariableA(const char* name, char* out, DWORD size) {
    const char* v = getenv(name);
    if(!v) return 0;
    size_t len = strlen(v);
    if(!out || size <= len) return (DWORD)(len + 1);
    memcpy(out, v, len + 1);
    return (DWORD)len;
}
inline BOOL SetEnvironmentVariableA(const char* name, const char* value) {
    return (value ? setenv(name, value, 1) : unsetenv(name)) == 0;
}

inline DWORD GetModuleFileNameA(HMODULE, char* out, DWORD size) {
    ssize_t n = readlink("/proc/self/exe", out, size ? size - 1 : 0);
    if(n < 0) return 0;
    out[n] = 0;
    return (DWORD)n;
}

inline BOOL GetComputerNameA(char* out, DWORD* size) {
    if(gethostname(out, *size) != 0) return FALSE;
    out[*size - 1] = 0;
    *size = (DWORD)strlen(out);
    return TRUE;
}

inline BOOL QueryPerformanceFrequency(LARGE_INTEGER* f) { f->QuadPart = 1000000000LL; return TRUE; }
inline BOOL QueryPerformanceCounter(LARGE_INTEGER* c) {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    c->QuadPart = (LONGLONG)t.tv_sec * 1000000000LL + t.tv_nsec;
    return TRUE;
}

typedef struct _FILETIME { DWORD dwLowDateTime; DWORD dwHighDateTime; } FILETIME;
// tiempos de la cpu de todo el sistema desde /proc/stat en unidades de 100 ns como en windows
inline BOOL GetSystemTimes(FILETIME* idle, FILETIME* kernel, FILETIME* user) {
    FILE* f = fopen("/proc/stat", "r");
    if(!f) return FALSE;
    unsigned long long u = 0, n = 0, s = 0, i = 0, w = 0, q = 0, sq = 0;
    int got = fscanf(f, "cpu %llu %llu %llu %llu %llu %llu %llu", &u, &n, &s, &i, &w, &q, &sq);
    fclose(f);
    if(got < 4) return FALSE;
    long hz = sysconf(_SC_CLK_TCK);
    auto put = [&](FILETIME* t, unsigned long long ticks){
        if(!t) return;
        unsigned long long v = ticks * (10000000ULL / (unsigned long long)(hz > 0 ? hz : 100));
        t->dwLowDateTime = (DWORD)v;
        t->dwHighDateTime = (DWORD)(v >> 32);
    };
    put(idle, i + w);
    put(kernel, s + q + sq + i + w);
    put(user, u + n);
    return TRUE;
}

// terminar ya sin pasar por los destructores, como exitprocess con el motor a mitad de cosas
inline void ExitProcess(unsigned code) { fflush(nullptr); _exit((int)code); }
inline BOOL SetConsoleOutputCP(unsigned) { return TRUE; }
#define THREAD_PRIORITY_HIGHEST 2
#define THREAD_PRIORITY_ABOVE_NORMAL 1
#define THREAD_MODE_BACKGROUND_BEGIN 0x00010000
#define THREAD_MODE_BACKGROUND_END 0x00020000
// en linux cada hilo tiene su propio nice: el de fondo baja a 10 y subir prioridad pide permisos que no hay asi que se deja
inline BOOL SetThreadPriority(HANDLE, int level) {
    if(level == THREAD_MODE_BACKGROUND_BEGIN) return setpriority(PRIO_PROCESS, (id_t)syscall(SYS_gettid), 10) == 0;
    if(level == THREAD_MODE_BACKGROUND_END) return setpriority(PRIO_PROCESS, (id_t)syscall(SYS_gettid), 0) == 0;
    return TRUE;
}
#define CP_UTF8 65001

// ---- memoria ----

typedef struct _MEMORYSTATUSEX {
    DWORD dwLength;
    DWORD dwMemoryLoad;
    ULONGLONG ullTotalPhys;
    ULONGLONG ullAvailPhys;
    ULONGLONG ullTotalPageFile;
    ULONGLONG ullAvailPageFile;
    ULONGLONG ullTotalVirtual;
    ULONGLONG ullAvailVirtual;
    ULONGLONG ullAvailExtendedVirtual;
} MEMORYSTATUSEX;

// ram total y disponible desde /proc/meminfo: MemAvailable ya cuenta la cache que se puede soltar como el disponible de windows
inline BOOL GlobalMemoryStatusEx(MEMORYSTATUSEX* m) {
    FILE* f = fopen("/proc/meminfo", "r");
    if(!f) return FALSE;
    char key[64];
    unsigned long long kb = 0, total = 0, avail = 0, swapTotal = 0, swapFree = 0;
    while(fscanf(f, "%63s %llu kB\n", key, &kb) == 2){
        if(!strcmp(key, "MemTotal:")) total = kb;
        else if(!strcmp(key, "MemAvailable:")) avail = kb;
        else if(!strcmp(key, "SwapTotal:")) swapTotal = kb;
        else if(!strcmp(key, "SwapFree:")) swapFree = kb;
    }
    fclose(f);
    m->ullTotalPhys = total << 10;
    m->ullAvailPhys = avail << 10;
    m->ullTotalPageFile = (total + swapTotal) << 10;
    m->ullAvailPageFile = (avail + swapFree) << 10;
    m->ullTotalVirtual = m->ullAvailVirtual = 1ULL << 47;
    m->ullAvailExtendedVirtual = 0;
    m->dwMemoryLoad = total ? (DWORD)(100 - avail * 100 / total) : 0;
    return TRUE;
}

typedef struct _PROCESS_MEMORY_COUNTERS {
    DWORD cb;
    DWORD PageFaultCount;
    SIZE_T PeakWorkingSetSize;
    SIZE_T WorkingSetSize;
    SIZE_T QuotaPeakPagedPoolUsage;
    SIZE_T QuotaPagedPoolUsage;
    SIZE_T QuotaPeakNonPagedPoolUsage;
    SIZE_T QuotaNonPagedPoolUsage;
    SIZE_T PagefileUsage;
    SIZE_T PeakPagefileUsage;
} PROCESS_MEMORY_COUNTERS;
typedef PROCESS_MEMORY_COUNTERS PROCESS_MEMORY_COUNTERS_EX;

inline BOOL GetProcessMemoryInfo(HANDLE, PROCESS_MEMORY_COUNTERS* c, DWORD) {
    memset(c, 0, sizeof(*c));
    FILE* f = fopen("/proc/self/statm", "r");
    if(!f) return FALSE;
    unsigned long long size = 0, resident = 0;
    int got = fscanf(f, "%llu %llu", &size, &resident);
    fclose(f);
    if(got != 2) return FALSE;
    long page = sysconf(_SC_PAGESIZE);
    c->WorkingSetSize = (SIZE_T)(resident * page);
    c->PeakWorkingSetSize = c->WorkingSetSize;
    c->PagefileUsage = (SIZE_T)(size * page);
    return TRUE;
}

typedef struct _IO_COUNTERS {
    ULONGLONG ReadOperationCount, WriteOperationCount, OtherOperationCount;
    ULONGLONG ReadTransferCount, WriteTransferCount, OtherTransferCount;
} IO_COUNTERS;
inline BOOL GetProcessIoCounters(HANDLE, IO_COUNTERS* io) {
    memset(io, 0, sizeof(*io));
    FILE* f = fopen("/proc/self/io", "r");
    if(!f) return FALSE;
    char key[64];
    unsigned long long v = 0;
    while(fscanf(f, "%63s %llu\n", key, &v) == 2){
        if(!strcmp(key, "read_bytes:")) io->ReadTransferCount = v;
        else if(!strcmp(key, "write_bytes:")) io->WriteTransferCount = v;
        else if(!strcmp(key, "syscr:")) io->ReadOperationCount = v;
        else if(!strcmp(key, "syscw:")) io->WriteOperationCount = v;
    }
    fclose(f);
    return TRUE;
}

#define MEM_COMMIT 0x1000
#define MEM_RESERVE 0x2000
#define MEM_RELEASE 0x8000
#define MEM_LARGE_PAGES 0x20000000
#define PAGE_READWRITE 0x04
#define PAGE_READONLY 0x02

// virtualalloc con mmap: el tamano se guarda para soltarlo despues con virtualfree como en windows
inline std::map<void*, size_t>& shimAllocations() { static std::map<void*, size_t> list; return list; }
inline std::mutex& shimAllocLock() { static std::mutex lock; return lock; }
inline LPVOID VirtualAlloc(LPVOID, SIZE_T size, DWORD, DWORD) {
    void* p = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if(p == MAP_FAILED) return nullptr;
    std::lock_guard<std::mutex> guard(shimAllocLock());
    shimAllocations()[p] = size;
    return p;
}
inline BOOL VirtualFree(LPVOID p, SIZE_T, DWORD) {
    if(!p) return FALSE;
    size_t size = 0;
    {
        std::lock_guard<std::mutex> guard(shimAllocLock());
        auto it = shimAllocations().find(p);
        if(it == shimAllocations().end()) return FALSE;
        size = it->second;
        shimAllocations().erase(it);
    }
    return munmap(p, size) == 0;
}

inline uintptr_t shimPageDown(const void* p) { return (uintptr_t)p & ~(uintptr_t)(sysconf(_SC_PAGESIZE) - 1); }
inline size_t shimPageSpan(const void* p, size_t bytes) { return (size_t)((uintptr_t)p + bytes - shimPageDown(p)); }

inline BOOL VirtualLock(LPVOID p, SIZE_T bytes) { return mlock((void*)shimPageDown(p), shimPageSpan(p, bytes)) == 0; }
// en windows soltar algo que no estaba fijado lo saca del conjunto de trabajo: en linux se marca frio para que se suelte primero
inline BOOL VirtualUnlock(LPVOID p, SIZE_T bytes) {
    void* at = (void*)shimPageDown(p);
    size_t span = shimPageSpan(p, bytes);
    munlock(at, span);
#ifdef MADV_COLD
    madvise(at, span, MADV_COLD);
#endif
    errno = ERROR_NOT_LOCKED;
    return FALSE;
}

typedef struct _WIN32_MEMORY_RANGE_ENTRY {
    PVOID VirtualAddress;
    SIZE_T NumberOfBytes;
} WIN32_MEMORY_RANGE_ENTRY, *PWIN32_MEMORY_RANGE_ENTRY;

// pedir paginas al disco antes de leerlas: madvise willneed lanza la lectura y vuelve enseguida como en windows
inline BOOL PrefetchVirtualMemory(HANDLE, ULONG_PTR count, WIN32_MEMORY_RANGE_ENTRY* ranges, ULONG) {
    for(ULONG_PTR i = 0; i < count; i++){
        if(!ranges[i].VirtualAddress || !ranges[i].NumberOfBytes) continue;
        madvise((void*)shimPageDown(ranges[i].VirtualAddress), shimPageSpan(ranges[i].VirtualAddress, ranges[i].NumberOfBytes), MADV_WILLNEED);
    }
    return TRUE;
}

#define QUOTA_LIMITS_HARDWS_MIN_ENABLE 0x1
#define QUOTA_LIMITS_HARDWS_MIN_DISABLE 0x2
#define QUOTA_LIMITS_HARDWS_MAX_ENABLE 0x4
#define QUOTA_LIMITS_HARDWS_MAX_DISABLE 0x8
// linux no tiene tope de conjunto de trabajo por proceso: se sube el limite de memoria fijada si se puede
inline BOOL SetProcessWorkingSetSizeEx(HANDLE, SIZE_T low, SIZE_T, DWORD) {
    if(low && low != (SIZE_T)-1){
        rlimit r;
        if(getrlimit(RLIMIT_MEMLOCK, &r) == 0 && r.rlim_cur != RLIM_INFINITY && r.rlim_cur < low){
            r.rlim_cur = std::min<rlim_t>(r.rlim_max, (rlim_t)low);
            setrlimit(RLIMIT_MEMLOCK, &r);
        }
    }
    return TRUE;
}

// ---- nucleos ----

typedef struct _GROUP_AFFINITY {
    KAFFINITY Mask;
    WORD Group;
    WORD Reserved[3];
} GROUP_AFFINITY;

inline BOOL SetThreadGroupAffinity(HANDLE, const GROUP_AFFINITY* place, GROUP_AFFINITY*) {
    if(!place || !place->Mask) return FALSE;
    cpu_set_t set;
    CPU_ZERO(&set);
    for(int bit = 0; bit < 64; bit++){
        if(place->Mask & (1ULL << bit)) CPU_SET(place->Group * 64 + bit, &set);
    }
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
}

// la lista de nucleos por eficiencia no esta en linux de la misma forma: se contesta vacio y el motor usa todos
typedef enum _LOGICAL_PROCESSOR_RELATIONSHIP { RelationProcessorCore = 0, RelationAll = 0xffff } LOGICAL_PROCESSOR_RELATIONSHIP;
typedef struct _PROCESSOR_RELATIONSHIP {
    BYTE Flags;
    BYTE EfficiencyClass;
    BYTE Reserved[20];
    WORD GroupCount;
    GROUP_AFFINITY GroupMask[1];
} PROCESSOR_RELATIONSHIP;
typedef struct _SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX {
    LOGICAL_PROCESSOR_RELATIONSHIP Relationship;
    DWORD Size;
    PROCESSOR_RELATIONSHIP Processor;
} SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, *PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX;
inline BOOL GetLogicalProcessorInformationEx(LOGICAL_PROCESSOR_RELATIONSHIP, PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, DWORD* bytes) {
    if(bytes) *bytes = 0;
    errno = ERROR_INSUFFICIENT_BUFFER;
    return FALSE;
}

// gcc ya tiene __cpuid como macro de 5 valores y las versiones nuevas traen su propio __cpuidex: se tapan los dos con la forma de windows
inline void shimCpuid(int info[4], int leaf, int sub) {
    unsigned a = 0, b = 0, c = 0, d = 0;
    __cpuid_count((unsigned)leaf, (unsigned)sub, a, b, c, d);
    info[0] = (int)a; info[1] = (int)b; info[2] = (int)c; info[3] = (int)d;
}
#undef __cpuid
#undef __cpuidex
#define __cpuidex(info, leaf, sub) shimCpuid(info, leaf, sub)
#define __cpuid(info, leaf) shimCpuid(info, leaf, 0)

// ---- bibliotecas ----

// los nombres de windows de las bibliotecas de nvidia y amd pasan a los de linux
inline std::string shimLibraryName(const char* name) {
    std::string path = name ? name : "";
    size_t cut = path.find_last_of("\\/");
    std::string base = cut == std::string::npos ? path : path.substr(cut + 1);
    std::string folder = cut == std::string::npos ? "" : path.substr(0, cut + 1);
    for(char& ch : folder) if(ch == '\\') ch = '/';
    struct Pair { const char* from; const char* to; };
    static const Pair names[] = {
        {"nvcuda.dll", "libcuda.so.1"},
        {"nvml.dll", "libnvidia-ml.so.1"},
        {"cublas64_12.dll", "libcublas.so.12"},
        {"cublasLt64_12.dll", "libcublasLt.so.12"},
        {"nvrtc64_120_0.dll", "libnvrtc.so.12"},
        {"nvrtc-builtins64_129.dll", "libnvrtc-builtins.so.12.9"},
        {"amdhip64_6.dll", "libamdhip64.so.6"},
        {"amdhip64.dll", "libamdhip64.so"},
        {"hiprtc0601.dll", "libhiprtc.so.6"},
    };
    for(const Pair& p : names){
        if(strcasecmp(base.c_str(), p.from) == 0) return folder.empty() ? p.to : folder + p.to;
    }
    return folder + base;
}

inline HMODULE LoadLibraryA(const char* name) {
    std::string real = shimLibraryName(name);
    void* lib = dlopen(real.c_str(), RTLD_NOW | RTLD_LOCAL);
    // si no esta en la carpeta pedida se busca por nombre en las del sistema
    if(!lib && real.find('/') != std::string::npos) lib = dlopen(real.substr(real.find_last_of('/') + 1).c_str(), RTLD_NOW | RTLD_LOCAL);
    return lib;
}
#define LOAD_WITH_ALTERED_SEARCH_PATH 0x8
#define LOAD_LIBRARY_SEARCH_DEFAULT_DIRS 0x1000
inline HMODULE LoadLibraryExA(const char* name, HANDLE, DWORD) { return LoadLibraryA(name); }
inline HMODULE GetModuleHandleA(const char* name) {
    if(!name) return dlopen(nullptr, RTLD_NOW);
    std::string real = shimLibraryName(name);
    return dlopen(real.c_str(), RTLD_NOW | RTLD_NOLOAD);
}
inline FARPROC GetProcAddress(HMODULE lib, const char* name) { return lib ? dlsym(lib, name) : nullptr; }
inline BOOL FreeLibrary(HMODULE lib) { return lib ? dlclose(lib) == 0 : FALSE; }
inline BOOL SetDllDirectoryA(const char*) { return TRUE; }

// ---- archivos ----

#define GENERIC_READ 0x80000000u
#define GENERIC_WRITE 0x40000000u
#define FILE_SHARE_READ 0x1
#define FILE_SHARE_WRITE 0x2
#define FILE_SHARE_DELETE 0x4
#define CREATE_NEW 1
#define CREATE_ALWAYS 2
#define OPEN_EXISTING 3
#define OPEN_ALWAYS 4
#define FILE_ATTRIBUTE_NORMAL 0x80
#define FILE_FLAG_WRITE_THROUGH 0x80000000u
#define FILE_FLAG_OVERLAPPED 0x40000000u
#define FILE_FLAG_NO_BUFFERING 0x20000000u
#define FILE_FLAG_RANDOM_ACCESS 0x10000000u
#define FILE_FLAG_SEQUENTIAL_SCAN 0x08000000u
#define INVALID_FILE_ATTRIBUTES 0xFFFFFFFFu
#define FILE_ATTRIBUTE_DIRECTORY 0x10
#define PAGE_READONLY_MAP 0x02
#define FILE_MAP_READ 0x4

inline int shimFd(HANDLE h) { return (int)(intptr_t)h; }

inline HANDLE CreateFileA(const char* path, DWORD access, DWORD, void*, DWORD create, DWORD flags, HANDLE) {
    int mode = (access & GENERIC_WRITE) ? ((access & GENERIC_READ) ? O_RDWR : O_WRONLY) : O_RDONLY;
    if(create == CREATE_ALWAYS) mode |= O_CREAT | O_TRUNC;
    else if(create == CREATE_NEW) mode |= O_CREAT | O_EXCL;
    else if(create == OPEN_ALWAYS) mode |= O_CREAT;
    std::string p = path;
    for(char& ch : p) if(ch == '\\') ch = '/';
    int fd = open(p.c_str(), mode | O_CLOEXEC, 0644);
    if(fd < 0) return INVALID_HANDLE_VALUE;
    if(flags & FILE_FLAG_RANDOM_ACCESS) posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM);
    if(flags & FILE_FLAG_SEQUENTIAL_SCAN) posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
    return (HANDLE)(intptr_t)fd;
}

// lectura con posicion de windows: aca se hace en el momento con pread y el resultado queda en la estructura
typedef struct _OVERLAPPED {
    ULONG_PTR Internal;
    ULONG_PTR InternalHigh;
    union {
        struct { DWORD Offset; DWORD OffsetHigh; };
        PVOID Pointer;
    };
    HANDLE hEvent;
} OVERLAPPED, *LPOVERLAPPED;

inline BOOL ReadFile(HANDLE h, void* out, DWORD bytes, DWORD* got, OVERLAPPED* at) {
    int fd = shimFd(h);
    ssize_t n;
    if(at){
        off_t offset = (off_t)(((uint64_t)at->OffsetHigh << 32) | at->Offset);
        size_t done = 0;
        n = 0;
        while(done < bytes){
            ssize_t r = pread(fd, (char*)out + done, bytes - done, offset + (off_t)done);
            if(r <= 0){ if(r < 0) n = -1; break; }
            done += (size_t)r;
        }
        if(n >= 0) n = (ssize_t)done;
        at->Internal = n < 0 ? (ULONG_PTR)errno : 0;
        at->InternalHigh = n < 0 ? 0 : (ULONG_PTR)n;
    } else n = read(fd, out, bytes);
    if(got) *got = n < 0 ? 0 : (DWORD)n;
    return n >= 0;
}
inline BOOL WriteFile(HANDLE h, const void* data, DWORD bytes, DWORD* put, OVERLAPPED* at) {
    int fd = shimFd(h);
    ssize_t n;
    if(at){
        off_t offset = (off_t)(((uint64_t)at->OffsetHigh << 32) | at->Offset);
        n = pwrite(fd, data, bytes, offset);
        at->InternalHigh = n < 0 ? 0 : (ULONG_PTR)n;
    } else n = write(fd, data, bytes);
    if(put) *put = n < 0 ? 0 : (DWORD)n;
    return n >= 0;
}
inline BOOL GetOverlappedResult(HANDLE, OVERLAPPED* at, DWORD* got, BOOL) {
    if(got) *got = (DWORD)at->InternalHigh;
    return at->Internal == 0;
}
inline HANDLE CreateEventA(void*, BOOL, BOOL, const char*) { return (HANDLE)(intptr_t)1; }
inline BOOL ResetEvent(HANDLE) { return TRUE; }
inline BOOL SetEvent(HANDLE) { return TRUE; }
#define MOVEFILE_REPLACE_EXISTING 0x1
#define MOVEFILE_WRITE_THROUGH 0x8
inline std::string shimPath(const char* path) { std::string p = path; for(char& ch : p) if(ch == (char)92) ch = (char)47; return p; }
inline BOOL DeleteFileA(const char* path) { return unlink(shimPath(path).c_str()) == 0; }
// rename en linux ya reemplaza el destino como movefileex con replace_existing
inline BOOL MoveFileExA(const char* from, const char* to, DWORD) { return rename(shimPath(from).c_str(), shimPath(to).c_str()) == 0; }
inline BOOL MoveFileA(const char* from, const char* to) {
    struct stat st;
    if(stat(shimPath(to).c_str(), &st) == 0) return FALSE;
    return rename(shimPath(from).c_str(), shimPath(to).c_str()) == 0;
}
inline BOOL GetFileSizeEx(HANDLE h, LARGE_INTEGER* size) {
    struct stat st;
    if(fstat(shimFd(h), &st) != 0) return FALSE;
    size->QuadPart = (LONGLONG)st.st_size;
    return TRUE;
}
inline BOOL FlushFileBuffers(HANDLE h) { return fsync(shimFd(h)) == 0; }

// el archivo mapeado guarda el descriptor y el tamano para que mapviewoffile y unmapviewoffile funcionen como en windows
struct ShimMapping { int fd; uint64_t size; };
inline std::map<void*, size_t>& shimViews() { static std::map<void*, size_t> list; return list; }
inline std::mutex& shimViewLock() { static std::mutex lock; return lock; }
inline HANDLE CreateFileMappingA(HANDLE file, void*, DWORD, DWORD, DWORD, const char*) {
    struct stat st;
    if(fstat(shimFd(file), &st) != 0 || st.st_size == 0) return nullptr;
    return new ShimMapping{shimFd(file), (uint64_t)st.st_size};
}
inline LPVOID MapViewOfFile(HANDLE mapping, DWORD, DWORD offsetHigh, DWORD offsetLow, SIZE_T bytes) {
    ShimMapping* m = (ShimMapping*)mapping;
    if(!m) return nullptr;
    uint64_t offset = ((uint64_t)offsetHigh << 32) | offsetLow;
    size_t size = bytes ? bytes : (size_t)(m->size - offset);
    void* p = mmap(nullptr, size, PROT_READ, MAP_SHARED, m->fd, (off_t)offset);
    if(p == MAP_FAILED) return nullptr;
    std::lock_guard<std::mutex> guard(shimViewLock());
    shimViews()[p] = size;
    return p;
}
inline BOOL UnmapViewOfFile(const void* p) {
    size_t size = 0;
    {
        std::lock_guard<std::mutex> guard(shimViewLock());
        auto it = shimViews().find((void*)p);
        if(it == shimViews().end()) return FALSE;
        size = it->second;
        shimViews().erase(it);
    }
    return munmap((void*)p, size) == 0;
}

// closehandle cierra descriptores y libera los mapeos y eventos falsos
inline BOOL CloseHandle(HANDLE h) {
    if(!h || h == INVALID_HANDLE_VALUE || h == (HANDLE)(intptr_t)1) return TRUE;
    intptr_t v = (intptr_t)h;
    if(v > 0 && v < (1 << 20)) return close((int)v) == 0;
    delete (ShimMapping*)h;
    return TRUE;
}

inline DWORD GetFileAttributesA(const char* path) {
    struct stat st;
    std::string p = path;
    for(char& ch : p) if(ch == '\\') ch = '/';
    if(stat(p.c_str(), &st) != 0) return INVALID_FILE_ATTRIBUTES;
    return S_ISDIR(st.st_mode) ? FILE_ATTRIBUTE_DIRECTORY : FILE_ATTRIBUTE_NORMAL;
}
inline DWORD GetFullPathNameA(const char* path, DWORD size, char* out, char** part) {
    char real[PATH_MAX];
    if(!realpath(path, real)) snprintf(real, sizeof(real), "%s", path);
    size_t len = strlen(real);
    if(size <= len) return (DWORD)(len + 1);
    memcpy(out, real, len + 1);
    if(part) *part = nullptr;
    return (DWORD)len;
}
inline BOOL GetDiskFreeSpaceExA(const char* path, ULARGE_INTEGER* freeForCaller, ULARGE_INTEGER* total, ULARGE_INTEGER* totalFree) {
    struct statvfs v;
    std::string p = path;
    for(char& ch : p) if(ch == '\\') ch = '/';
    if(statvfs(p.c_str(), &v) != 0) return FALSE;
    if(freeForCaller) freeForCaller->QuadPart = (ULONGLONG)v.f_bavail * v.f_frsize;
    if(total) total->QuadPart = (ULONGLONG)v.f_blocks * v.f_frsize;
    if(totalFree) totalFree->QuadPart = (ULONGLONG)v.f_bfree * v.f_frsize;
    return TRUE;
}

// ---- red ----

typedef struct WSAData { WORD wVersion; WORD wHighVersion; char szDescription[257]; char szSystemStatus[129]; } WSADATA;
inline int WSAStartup(WORD, WSADATA*) { signal(SIGPIPE, SIG_IGN); return 0; }
inline int WSACleanup() { return 0; }
#define MAKEWORD(a, b) ((WORD)(((BYTE)(a)) | ((WORD)((BYTE)(b))) << 8))
inline int closesocket(SOCKET s) { return close(s); }
inline int ioctlsocket(SOCKET s, long cmd, unsigned long* arg) { int v = (int)*arg; return ioctl(s, cmd, &v); }
#define WSAEWOULDBLOCK EWOULDBLOCK
#define WSAEINTR EINTR
#define WSAECONNRESET ECONNRESET
#define WSAETIMEDOUT ETIMEDOUT
#define SD_SEND SHUT_WR
#define SD_RECEIVE SHUT_RD
#define SD_BOTH SHUT_RDWR

// ---- texto y archivos de c ----

inline int fopen_s(FILE** f, const char* path, const char* mode) {
    std::string p = path;
    for(char& ch : p) if(ch == '\\') ch = '/';
    *f = fopen(p.c_str(), mode);
    return *f ? 0 : errno;
}
inline int strcpy_s(char* out, size_t size, const char* in) {
    if(!out || !size) return EINVAL;
    snprintf(out, size, "%s", in ? in : "");
    return 0;
}
#define _fseeki64 fseeko
#define _ftelli64 ftello
// xgetbv sin pedir la opcion xsave a gcc
inline uint64_t shimXgetbv(unsigned index) {
    unsigned low = 0, high = 0;
    __asm__ volatile("xgetbv" : "=a"(low), "=d"(high) : "c"(index));
    return ((uint64_t)high << 32) | low;
}
#undef _xgetbv
#define _xgetbv(index) shimXgetbv(index)
#define _popen popen
#define _pclose pclose
#define _stricmp strcasecmp
#define _strnicmp strncasecmp
#define __forceinline inline __attribute__((always_inline))
