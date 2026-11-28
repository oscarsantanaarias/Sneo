#pragma once
#include "kernels.h"

typedef int (*CuCopyUpAsync)(uint64_t, const void*, size_t, void*);
typedef int (*CuCopyDownAsync)(void*, uint64_t, size_t, void*);
typedef int (*CuEventCreate)(void**, unsigned int);
typedef int (*CuEventRecord)(void*, void*);
typedef int (*CuEventSync)(void*);
typedef int (*CuEventDestroy)(void*);
typedef int (*CuStreamCreate)(void**, unsigned int);
typedef int (*CuStreamDestroy)(void*);
typedef int (*CuBeginCapture)(void*, int);
typedef int (*CuEndCapture)(void*, void**);
typedef int (*CuInstantiate)(void**, void*, unsigned long long);
typedef int (*CuGraphLaunch)(void*, void*);
typedef int (*CuGraphDestroy)(void*);
typedef int (*CuEventElapsed)(float*, void*, void*);
typedef int (*CuHostDevice)(uint64_t*, void*, unsigned int);
typedef int (*CuStreamQuery)(void*);
typedef int (*CuCopyAcross)(uint64_t, uint64_t, size_t, void*);
typedef int (*CuStreamWait)(void*, void*, unsigned int);
typedef int (*CuStreamSync)(void*);
typedef int (*CuEventQuery)(void*);

struct CopyPlane {
    size_t srcX = 0, srcY = 0;
    unsigned int srcKind = 0;
    const void* srcHost = nullptr;
    uint64_t srcDevice = 0;
    void* srcArray = nullptr;
    size_t srcPitch = 0;
    size_t dstX = 0, dstY = 0;
    unsigned int dstKind = 0;
    void* dstHost = nullptr;
    uint64_t dstDevice = 0;
    void* dstArray = nullptr;
    size_t dstPitch = 0;
    size_t width = 0, height = 0;
};
typedef int (*CuCopyPlane)(const CopyPlane*, void*);

struct CudaQueue {
    bool ready = false;
    CuCopyUpAsync copyUp = nullptr;
    CuCopyDownAsync copyDown = nullptr;
    CuEventCreate eventCreate = nullptr;
    CuEventRecord eventRecord = nullptr;
    CuEventSync eventSync = nullptr;
    CuEventDestroy eventDestroy = nullptr;
    CuStreamCreate streamCreate = nullptr;
    CuStreamDestroy streamDestroy = nullptr;
    CuBeginCapture beginCapture = nullptr;
    CuEndCapture endCapture = nullptr;
    CuInstantiate instantiate = nullptr;
    CuGraphLaunch graphLaunch = nullptr;
    CuGraphDestroy graphDestroy = nullptr;
    CuGraphDestroy execDestroy = nullptr;
    CuEventElapsed eventElapsed = nullptr;
    CuHostDevice hostDevice = nullptr;
    CuStreamQuery query = nullptr;
    CuCopyAcross copyAcross = nullptr;
    CuStreamWait streamWait = nullptr;
    CuEventQuery eventQuery = nullptr;
    CuStreamSync streamSync = nullptr;
    CuCopyPlane copyPlane = nullptr;

    int planeUp(uint64_t to, size_t toPitch, const void* from, size_t fromPitch, size_t width, size_t height, void* stream) {
        CopyPlane plane;
        plane.srcKind = 1;
        plane.srcHost = from;
        plane.srcPitch = fromPitch;
        plane.dstKind = 2;
        plane.dstDevice = to;
        plane.dstPitch = toPitch;
        plane.width = width;
        plane.height = height;
        return copyPlane(&plane, stream);
    }
};

inline CudaQueue& cudaQueue() {
    static CudaQueue queue;
    static bool tried = false;
    static std::mutex lock;
    std::lock_guard<std::mutex> guard(lock);
    if(tried) return queue;
    tried = true;

    cudaLink();
    HMODULE lib = amdCards ? amdModule : GetModuleHandleA("nvcuda.dll");
    if(!lib) return queue;
    if(amdCards){
        queue.copyUp = (CuCopyUpAsync)GetProcAddress(lib, "hipMemcpyHtoDAsync");
        queue.copyDown = (CuCopyDownAsync)GetProcAddress(lib, "hipMemcpyDtoHAsync");
        queue.eventCreate = (CuEventCreate)GetProcAddress(lib, "hipEventCreateWithFlags");
        queue.eventRecord = (CuEventRecord)GetProcAddress(lib, "hipEventRecord");
        queue.eventSync = (CuEventSync)GetProcAddress(lib, "hipEventSynchronize");
        queue.eventDestroy = (CuEventDestroy)GetProcAddress(lib, "hipEventDestroy");
        queue.streamCreate = (CuStreamCreate)GetProcAddress(lib, "hipStreamCreateWithFlags");
        queue.streamDestroy = (CuStreamDestroy)GetProcAddress(lib, "hipStreamDestroy");
        queue.beginCapture = (CuBeginCapture)GetProcAddress(lib, "hipStreamBeginCapture");
        queue.endCapture = (CuEndCapture)GetProcAddress(lib, "hipStreamEndCapture");
        queue.instantiate = (CuInstantiate)GetProcAddress(lib, "hipGraphInstantiateWithFlags");
        queue.graphLaunch = (CuGraphLaunch)GetProcAddress(lib, "hipGraphLaunch");
        queue.graphDestroy = (CuGraphDestroy)GetProcAddress(lib, "hipGraphDestroy");
        queue.execDestroy = (CuGraphDestroy)GetProcAddress(lib, "hipGraphExecDestroy");
        queue.eventElapsed = (CuEventElapsed)GetProcAddress(lib, "hipEventElapsedTime");
        queue.hostDevice = (CuHostDevice)GetProcAddress(lib, "hipHostGetDevicePointer");
        queue.query = (CuStreamQuery)GetProcAddress(lib, "hipStreamQuery");
        queue.copyAcross = (CuCopyAcross)GetProcAddress(lib, "hipMemcpyDtoDAsync");
        queue.streamWait = (CuStreamWait)GetProcAddress(lib, "hipStreamWaitEvent");
        queue.eventQuery = (CuEventQuery)GetProcAddress(lib, "hipEventQuery");
        queue.streamSync = (CuStreamSync)GetProcAddress(lib, "hipStreamSynchronize");
        queue.ready = queue.copyUp && queue.copyDown && queue.eventCreate && queue.eventRecord && queue.eventSync && queue.eventDestroy && queue.streamCreate && queue.streamDestroy && queue.beginCapture && queue.endCapture && queue.instantiate && queue.graphLaunch && queue.graphDestroy && queue.execDestroy && queue.eventElapsed && queue.hostDevice && queue.query && queue.copyAcross && queue.streamWait && queue.eventQuery;
        return queue;
    }
    queue.copyUp = (CuCopyUpAsync)GetProcAddress(lib, "cuMemcpyHtoDAsync_v2");
    queue.copyDown = (CuCopyDownAsync)GetProcAddress(lib, "cuMemcpyDtoHAsync_v2");
    queue.eventCreate = (CuEventCreate)GetProcAddress(lib, "cuEventCreate");
    queue.eventRecord = (CuEventRecord)GetProcAddress(lib, "cuEventRecord");
    queue.eventSync = (CuEventSync)GetProcAddress(lib, "cuEventSynchronize");
    queue.eventDestroy = (CuEventDestroy)GetProcAddress(lib, "cuEventDestroy_v2");
    queue.streamCreate = (CuStreamCreate)GetProcAddress(lib, "cuStreamCreate");
    queue.streamDestroy = (CuStreamDestroy)GetProcAddress(lib, "cuStreamDestroy_v2");
    queue.beginCapture = (CuBeginCapture)GetProcAddress(lib, "cuStreamBeginCapture_v2");
    queue.endCapture = (CuEndCapture)GetProcAddress(lib, "cuStreamEndCapture");
    queue.instantiate = (CuInstantiate)GetProcAddress(lib, "cuGraphInstantiateWithFlags");
    queue.graphLaunch = (CuGraphLaunch)GetProcAddress(lib, "cuGraphLaunch");
    queue.graphDestroy = (CuGraphDestroy)GetProcAddress(lib, "cuGraphDestroy");
    queue.execDestroy = (CuGraphDestroy)GetProcAddress(lib, "cuGraphExecDestroy");
    queue.eventElapsed = (CuEventElapsed)GetProcAddress(lib, "cuEventElapsedTime");
    queue.hostDevice = (CuHostDevice)GetProcAddress(lib, "cuMemHostGetDevicePointer_v2");
    queue.query = (CuStreamQuery)GetProcAddress(lib, "cuStreamQuery");
    queue.copyAcross = (CuCopyAcross)GetProcAddress(lib, "cuMemcpyDtoDAsync_v2");
    queue.streamWait = (CuStreamWait)GetProcAddress(lib, "cuStreamWaitEvent");
    queue.eventQuery = (CuEventQuery)GetProcAddress(lib, "cuEventQuery");
    queue.streamSync = (CuStreamSync)GetProcAddress(lib, "cuStreamSynchronize");
    queue.copyPlane = (CuCopyPlane)GetProcAddress(lib, "cuMemcpy2DAsync_v2");
    queue.ready = queue.copyUp && queue.copyDown && queue.eventCreate && queue.eventRecord && queue.eventSync && queue.eventDestroy && queue.streamCreate && queue.streamDestroy && queue.beginCapture && queue.endCapture && queue.instantiate && queue.graphLaunch && queue.graphDestroy && queue.execDestroy && queue.eventElapsed && queue.hostDevice && queue.query && queue.copyAcross && queue.streamWait && queue.eventQuery;
    return queue;
}

const char* modelKernelSource = R"(
#ifndef EXACT_MIX
#define EXACT_MIX 0
#endif
__device__ __forceinline__ float gateOf(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ u16 toHalf(float v) {
#ifdef __HIP__
    return __builtin_bit_cast(u16, (_Float16)v);
#else
    u16 out;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(out) : "f"(v));
    return out;
#endif
}

__device__ float blockSum(float v, float* scratch) {
    for(int step = 16; step > 0; step >>= 1) v += __shfl_xor_sync(0xffffffff, v, step);
    __syncthreads();
    if((threadIdx.x & 31) == 0) scratch[threadIdx.x >> 5] = v;
    __syncthreads();
    float total = 0.0f;
    for(int w = 0; w < (int)(blockDim.x >> 5); w++) total += scratch[w];
    return total;
}

__device__ float blockMax(float v, float* scratch) {
    for(int step = 16; step > 0; step >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, step));
    __syncthreads();
    if((threadIdx.x & 31) == 0) scratch[threadIdx.x >> 5] = v;
    __syncthreads();
    float best = scratch[0];
    for(int w = 1; w < (int)(blockDim.x >> 5); w++) best = fmaxf(best, scratch[w]);
    return best;
}

__device__ int blockScan(int v, int& total, int* scratch) {
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int x = v;
    for(int step = 1; step < 32; step <<= 1){
        int y = __shfl_up_sync(0xffffffff, x, step);
        if(lane >= step) x += y;
    }
    __syncthreads();
    if(lane == 31) scratch[warp] = x;
    __syncthreads();
    int before = 0;
    total = 0;
    for(int w = 0; w < (int)(blockDim.x >> 5); w++){
        if(w < warp) before += scratch[w];
        total += scratch[w];
    }
    return before + x - v;
}

__device__ __forceinline__ int table16(u32 t0, u32 t1, u32 t2, u32 t3, u32 nibbles) {
    u32 x = nibbles | (nibbles >> 4);
    u32 selector = ((x & 0xFFu) | ((x >> 8) & 0xFF00u)) & 0x7777u;
    u32 lo = __byte_perm(t0, t1, selector);
    u32 hi = __byte_perm(t2, t3, selector);
    u32 mask = ((nibbles >> 3) & 0x01010101u) * 0xFFu;
    return (int)((lo & ~mask) | (hi & mask));
}

__device__ __forceinline__ void unpackGroup(int type, const u8* block, int g, int* w, float& low, float& high, float& offset, const GridTables& t) {
    offset = 0.0f;
    if(type == 11){
        float d = half(word(block + 108));
        int halfIdx = g >> 2, l = g & 3, shift = 2 * l;
        const u8* qp = block + 32 + 32 * halfIdx;
        #pragma unroll
        for(int i = 0; i < 8; i++){
            u32 qv = (load32(qp + 4 * i) >> shift) & 0x03030303u;
            u32 hbits = (load32(block + 4 * i) >> g) & 0x01010101u;
            u32 sub = __vsub4(0x01010101u, hbits) * 4u;
            w[i] = (int)__vsub4(qv, sub);
        }
        u32 a0 = load32(block + 96), a1 = load32(block + 100), a2 = load32(block + 104);
        u32 uw = g < 2 ? ((a0 & 0x0f0f0f0fu) | ((a2 & 0x03030303u) << 4))
               : g < 4 ? ((a1 & 0x0f0f0f0fu) | (((a2 >> 2) & 0x03030303u) << 4))
               : g < 6 ? (((a0 >> 4) & 0x0f0f0f0fu) | (((a2 >> 4) & 0x03030303u) << 4))
               : (((a1 >> 4) & 0x0f0f0f0fu) | (((a2 >> 6) & 0x03030303u) << 4));
        int sshift = 16 * (g & 1);
        low = d * (float)((int)((uw >> sshift) & 255u) - 32);
        high = d * (float)((int)((uw >> (sshift + 8)) & 255u) - 32);
        return;
    }

    if(type == 12 || type == 13){
        float d = half(word(block));
        float dmin = half(word(block + 2));
        int sc, m;
        scaleMin(g, block + 4, sc, m);
        const u8* qs = (type == 12 ? block + 16 : block + 48) + 32 * (g / 2);
        const u8* qh = block + 16;
        int shift = (g & 1) * 4;
        if(((u64)block & 15) == 0){
            uint4 a = *(const uint4*)qs;
            uint4 b = *(const uint4*)(qs + 16);
            u32 bits[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
            u32 top[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            if(type == 13){
                uint4 c = *(const uint4*)qh;
                uint4 e = *(const uint4*)(qh + 16);
                top[0] = c.x;
                top[1] = c.y;
                top[2] = c.z;
                top[3] = c.w;
                top[4] = e.x;
                top[5] = e.y;
                top[6] = e.z;
                top[7] = e.w;
            }
            #pragma unroll
            for(int i = 0; i < 8; i++){
                u32 v = (bits[i] >> shift) & 0x0F0F0F0Fu;
                if(type == 13) v |= ((top[i] >> g) & 0x01010101u) << 4;
                w[i] = (int)v;
            }
        } else {
            #pragma unroll
            for(int i = 0; i < 8; i++){
                u32 v = (load32(qs + 4 * i) >> shift) & 0x0F0F0F0Fu;
                if(type == 13) v |= ((load32(qh + 4 * i) >> g) & 0x01010101u) << 4;
                w[i] = (int)v;
            }
        }
        low = d * sc;
        high = low;
        offset = dmin * m;
        return;
    }

    if(type == 14){
        const u8* ql = block + 64 * (g / 4);
        const u8* qh = block + 128 + 32 * (g / 4);
        const i8* sc = (const i8*)(block + 192) + 8 * (g / 4);
        float d = half(word(block + 208));
        int k = g % 4;
        const u8* lowBytes = ql + (k & 1) * 32;
        int lowShift = (k >> 1) * 4;
        if(((u64)block & 1) == 0){
            const u32* lowWords = (const u32*)((u64)lowBytes & ~3ull);
            const u32* highWords = (const u32*)((u64)qh & ~3ull);
            int lowAlign = (int)((u64)lowBytes & 3) * 8;
            int highAlign = (int)((u64)qh & 3) * 8;
            u32 lw[9], hw[9];
            #pragma unroll
            for(int j = 0; j < 9; j++){
                lw[j] = lowWords[j];
                hw[j] = highWords[j];
            }
            #pragma unroll
            for(int i = 0; i < 8; i++){
                u32 lo = (__funnelshift_r(lw[i], lw[i + 1], lowAlign) >> lowShift) & 0x0F0F0F0Fu;
                u32 hi = ((__funnelshift_r(hw[i], hw[i + 1], highAlign) >> (2 * k)) & 0x03030303u) << 4;
                w[i] = (int)__vsub4(lo | hi, 0x20202020u);
            }
        } else {
            #pragma unroll
            for(int i = 0; i < 8; i++){
                u32 lo = (load32(lowBytes + 4 * i) >> lowShift) & 0x0F0F0F0Fu;
                u32 hi = ((load32(qh + 4 * i) >> (2 * k)) & 0x03030303u) << 4;
                w[i] = (int)__vsub4(lo | hi, 0x20202020u);
            }
        }
        low = d * (float)sc[2 * k];
        high = d * (float)sc[2 * k + 1];
        return;
    }

    if(type == 39){
        #pragma unroll
        for(int i = 0; i < 4; i++){
            u32 raw = load32(block + 1 + 4 * i);
            w[i] = fp4Bytes(raw & 0x0F0F0F0Fu);
            w[i + 4] = fp4Bytes((raw >> 4) & 0x0F0F0F0Fu);
        }
        low = mxScale(block[0]);
        high = low;
        return;
    }

    // nvfp4 tiene dos escalas por grupo de 32: low para los primeros 16 y high para los otros 16
    if(type == 40){
        #pragma unroll
        for(int s = 0; s < 2; s++){
            const u8* qs = block + 4 + 8 * (2 * g + s);
            u32 a = load32(qs), b = load32(qs + 4);
            w[4 * s] = fp4Bytes(a & 0x0F0F0F0Fu);
            w[4 * s + 1] = fp4Bytes(b & 0x0F0F0F0Fu);
            w[4 * s + 2] = fp4Bytes((a >> 4) & 0x0F0F0F0Fu);
            w[4 * s + 3] = fp4Bytes((b >> 4) & 0x0F0F0F0Fu);
        }
        low = nvScale(block[2 * g]);
        high = nvScale(block[2 * g + 1]);
        return;
    }

    if(type == 20 || type == 23){
        float d = half(word(block));
        const u8* qs = type == 20 ? block + 2 : block + 8 + 16 * g;
        if(type == 23){
            int lo = (block[4 + g / 2] >> (4 * (g % 2))) & 15;
            int hi = (word(block + 2) >> (2 * g)) & 3;
            d *= (float)((lo | (hi << 4)) - 32);
        }
        u32 raw[4];
        if(((u64)qs & 3) == 0){
            #pragma unroll
            for(int i = 0; i < 4; i++) raw[i] = ((const u32*)qs)[i];
        } else if(((u64)qs & 1) == 0){
            const u32* words = (const u32*)((u64)qs & ~3ull);
            u32 ww[5];
            #pragma unroll
            for(int j = 0; j < 5; j++) ww[j] = words[j];
            #pragma unroll
            for(int i = 0; i < 4; i++) raw[i] = __funnelshift_r(ww[i], ww[i + 1], 16);
        } else {
            #pragma unroll
            for(int i = 0; i < 4; i++) raw[i] = load32(qs + 4 * i);
        }
        const u32* table = (const u32*)t.iq4nl;
        u32 t0 = __ldg(table), t1 = __ldg(table + 1), t2 = __ldg(table + 2), t3 = __ldg(table + 3);
        #pragma unroll
        for(int i = 0; i < 4; i++){
            w[i] = table16(t0, t1, t2, t3, raw[i] & 0x0F0F0F0Fu);
            w[i + 4] = table16(t0, t1, t2, t3, (raw[i] >> 4) & 0x0F0F0F0Fu);
        }
        low = d;
        high = d;
        return;
    }

    if(type == 42){
        float d = half(word(block));
        const u8* qs = block + 2 + 8 * g;
        #pragma unroll
        for(int i = 0; i < 8; i++){
            u32 b = qs[i];
            u32 v = (b & 3) | (((b >> 2) & 3) << 8) | (((b >> 4) & 3) << 16) | (((b >> 6) & 3) << 24);
            w[i] = (int)__vsub4(v, 0x01010101u);
        }
        low = d;
        high = d;
        return;
    }

    if(type == 16){
        float d = half(word(block));
        const u8* qs = block + 2 + 8 * g;
        u32 aux = load32(qs + 4);
        float dl = d * (0.5f + (float)(aux >> 28)) * 0.25f;
        #pragma unroll
        for(int l = 0; l < 4; l++){
            u64 grid = __ldg(t.iq2xxs + qs[l]);
            u32 signs = __ldg(t.signs + ((aux >> (7 * l)) & 127));
            w[2 * l] = signed4((u32)grid, signs);
            w[2 * l + 1] = signed4((u32)(grid >> 32), signs >> 4);
        }
        low = dl;
        high = dl;
        return;
    }

    if(type == 17 || type == 22){
        float d = half(word(block));
        u8 scale = type == 17 ? block[66 + g] : block[74 + g];
        #pragma unroll
        for(int l = 0; l < 4; l++){
            u64 grid;
            u32 signs;
            if(type == 17){
                u16 code = word(block + 2 + 8 * g + 2 * l);
                grid = __ldg(t.iq2xs + (code & 511));
                signs = __ldg(t.signs + (code >> 9));
            } else {
                int index = block[2 + 4 * g + l] | ((block[66 + g] << (8 - 2 * l)) & 0x300);
                grid = __ldg(t.iq2s + index);
                signs = block[34 + 4 * g + l];
            }
            w[2 * l] = signed4((u32)grid, signs);
            w[2 * l + 1] = signed4((u32)(grid >> 32), signs >> 4);
        }
        low = d * 0.25f * (0.5f + (float)(scale & 15));
        high = d * 0.25f * (0.5f + (float)(scale >> 4));
        return;
    }

    if(type == 18){
        float d = half(word(block));
        const u8* qs = block + 2 + 8 * g;
        u32 aux = load32(block + 66 + 4 * g);
        float dl = d * (0.5f + (float)(aux >> 28)) * 0.5f;
        #pragma unroll
        for(int l = 0; l < 4; l++){
            u32 signs = __ldg(t.signs + ((aux >> (7 * l)) & 127));
            w[2 * l] = signed4(__ldg(t.iq3xxs + qs[2 * l]), signs);
            w[2 * l + 1] = signed4(__ldg(t.iq3xxs + qs[2 * l + 1]), signs >> 4);
        }
        low = dl;
        high = dl;
        return;
    }

    if(type == 21){
        float d = half(word(block));
        const u8* qs = block + 2 + 8 * g;
        u8 qh = block[66 + g];
        const u8* signs = block + 74 + 4 * g;
        float dl = d * (float)(1 + 2 * ((block[106 + g / 2] >> (4 * (g % 2))) & 15));
        #pragma unroll
        for(int l = 0; l < 4; l++){
            u32 first = __ldg(t.iq3s + (qs[2 * l] | ((qh << (8 - 2 * l)) & 256)));
            u32 second = __ldg(t.iq3s + (qs[2 * l + 1] | ((qh << (7 - 2 * l)) & 256)));
            w[2 * l] = signed4(first, signs[l]);
            w[2 * l + 1] = signed4(second, (u32)signs[l] >> 4);
        }
        low = dl;
        high = dl;
        return;
    }

    // iq1_m sale entero exacto porque la tabla va por 8 y el corrimiento de un octavo es sumar o restar 1
    if(type == 29){
        u16 bits = (u16)((word(block + 48) >> 12) | ((word(block + 50) >> 8) & 0x00F0) | ((word(block + 52) >> 4) & 0x0F00) | (word(block + 54) & 0xF000));
        float d = half(bits) * 0.125f;
        int sc = word(block + 48 + 2 * (g / 2)) >> (6 * (g % 2));
        #pragma unroll
        for(int l = 0; l < 4; l++){
            int h = block[32 + 2 * g + l / 2] >> (4 * (l % 2));
            u64 grid = __ldg(t.iq1s + (block[4 * g + l] | ((h & 7) << 8)));
            u32 drop = h & 8 ? 0x01010101u : 0xFFFFFFFFu;
            w[2 * l] = (int)__vsub4((u32)grid, drop);
            w[2 * l + 1] = (int)__vsub4((u32)(grid >> 32), drop);
        }
        low = d * (float)(2 * (sc & 7) + 1);
        high = d * (float)(2 * ((sc >> 3) & 7) + 1);
        return;
    }

    if(type == 8){
        float d = half(word(block));
        #pragma unroll
        for(int i = 0; i < 8; i++) w[i] = (int)load32(block + 2 + 4 * i);
        low = d;
        high = d;
        return;
    }

    if(type == 7){
        float d = half(word(block));
        u32 qh = load32(block + 4);
        #pragma unroll
        for(int i = 0; i < 4; i++){
            u32 raw = load32(block + 8 + 4 * i);
            u32 lowBits = (qh >> (4 * i)) & 15;
            u32 highBits = (qh >> (16 + 4 * i)) & 15;
            u32 lowTop = ((lowBits & 1) << 4) | ((lowBits & 2) << 11) | ((lowBits & 4) << 18) | ((lowBits & 8) << 25);
            u32 highTop = ((highBits & 1) << 4) | ((highBits & 2) << 11) | ((highBits & 4) << 18) | ((highBits & 8) << 25);
            w[i] = (int)((raw & 0x0F0F0F0Fu) | lowTop);
            w[i + 4] = (int)(((raw >> 4) & 0x0F0F0F0Fu) | highTop);
        }
        low = d;
        high = d;
        offset = -half(word(block + 2));
        return;
    }

    #pragma unroll
    for(int i = 0; i < 8; i++) w[i] = 0;
    low = 0.0f;
    high = 0.0f;
}

__device__ __forceinline__ void tensorStep(int* c, int a0, int a1, int b) {
#if __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};" : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3]) : "r"(a0), "r"(a1), "r"(b), "r"(0), "r"(0), "r"(0), "r"(0));
#else
    c[0] = c[1] = c[2] = c[3] = 0;
#endif
}

extern "C" __global__ void mmaRows(int type, const u8* weights, u64 rowBytes, u64 expertBytes, const int* groupSlot, const int* groupStart, const int* groupCount, const int* inputIndex, const i8* packed, const float* scales, const int* sums, const i8* fine, const float* fineScales, const int* fineSums, float* output, int rows, int cols, int count, int tiles, const int* tokenOf, const float* weightOf, unsigned long long* gather, GridTables t) {
    __shared__ __align__(16) i8 wq[64][48];
    __shared__ __align__(16) i8 xq[64][48];
    __shared__ __align__(16) i8 xf[64][48];
    __shared__ float lowS[64], highS[64], offS[64];
    __shared__ float xs[64], xsum[64], fs[64], fsum[64];
    __shared__ int source[64];

    int group = blockIdx.y / tiles, tile = blockIdx.y % tiles;
    int slot = groupSlot ? groupSlot[group] : 0;
    int start = groupStart ? groupStart[group] : 0;
    int total = groupCount ? groupCount[group] : count;
    int first = tile * 64;
    if(first >= total) return;
    int rowBase = blockIdx.x * 64;
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2, quad = lane & 3;
    if(tid < 64){
        int j = first + tid;
        source[tid] = j >= total
            ? -1
            : inputIndex
                ? inputIndex[start + j]
                : start + j;
    }
    __syncthreads();

    int groups = cols / 32;
    int perBlock = groupsPerBlock(type);
    int bytes = blockBytes(type);
    const u8* base = weights + expertBytes * slot;
    float acc[8][4];
    #pragma unroll
    for(int nt = 0; nt < 8; nt++){
        #pragma unroll
        for(int e = 0; e < 4; e++) acc[nt][e] = 0.0f;
    }

    for(int u = 0; u < groups; u++){
        if(tid < 64){
            int row = rowBase + tid;
            int w[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            float low = 0.0f, high = 0.0f, shift = 0.0f;
            if(row < rows) unpackGroup(type, base + rowBytes * row + (u64)(u / perBlock) * bytes, u % perBlock, w, low, high, shift, t);
            int4* into = (int4*)wq[tid];
            into[0] = make_int4(w[0], w[1], w[2], w[3]);
            into[1] = make_int4(w[4], w[5], w[6], w[7]);
            lowS[tid] = low;
            highS[tid] = high;
            offS[tid] = shift;
        } else {
            int k = tid - 64;
            int src = source[k];
            int4 zero = make_int4(0, 0, 0, 0);
            int4* intoQ = (int4*)xq[k];
            int4* intoF = (int4*)xf[k];
            if(src >= 0){
                u64 at = (u64)src * groups + u;
                const int4* q = (const int4*)(packed + at * 32);
                intoQ[0] = q[0];
                intoQ[1] = q[1];
                xs[k] = scales[at];
                xsum[k] = (float)sums[at];
                if(fine){
                    const int4* r = (const int4*)(fine + at * 32);
                    intoF[0] = r[0];
                    intoF[1] = r[1];
                    fs[k] = fineScales[at];
                    fsum[k] = (float)fineSums[at];
                } else {
                    intoF[0] = zero;
                    intoF[1] = zero;
                    fs[k] = 0.0f;
                    fsum[k] = 0.0f;
                }
            } else {
                intoQ[0] = zero;
                intoQ[1] = zero;
                intoF[0] = zero;
                intoF[1] = zero;
                xs[k] = 0.0f;
                xsum[k] = 0.0f;
                fs[k] = 0.0f;
                fsum[k] = 0.0f;
            }
        }
        __syncthreads();

        int r0 = warp * 16 + gid;
        int a0 = *(const int*)&wq[r0][quad * 4];
        int a1 = *(const int*)&wq[r0 + 8][quad * 4];
        int a2 = *(const int*)&wq[r0][16 + quad * 4];
        int a3 = *(const int*)&wq[r0 + 8][16 + quad * 4];
        float lowA = lowS[r0], highA = highS[r0], offA = offS[r0];
        float lowB = lowS[r0 + 8], highB = highS[r0 + 8], offB = offS[r0 + 8];
        #pragma unroll
        for(int nt = 0; nt < 8; nt++){
            int tok = nt * 8 + gid;
            int qL[4], qH[4], fL[4], fH[4];
            tensorStep(qL, a0, a1, *(const int*)&xq[tok][quad * 4]);
            tensorStep(qH, a2, a3, *(const int*)&xq[tok][16 + quad * 4]);
            tensorStep(fL, a0, a1, *(const int*)&xf[tok][quad * 4]);
            tensorStep(fH, a2, a3, *(const int*)&xf[tok][16 + quad * 4]);
            #pragma unroll
            for(int e = 0; e < 4; e++){
                int tj = nt * 8 + quad * 2 + (e & 1);
                float low = e < 2 ? lowA : lowB;
                float high = e < 2 ? highA : highB;
                float off = e < 2 ? offA : offB;
                acc[nt][e] += xs[tj] * (low * (float)qL[e] + high * (float)qH[e] - off * xsum[tj]) + fs[tj] * (low * (float)fL[e] + high * (float)fH[e] - off * fsum[tj]);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for(int nt = 0; nt < 8; nt++){
        #pragma unroll
        for(int e = 0; e < 4; e++){
            int ri = rowBase + warp * 16 + gid + (e >> 1) * 8;
            int j = first + nt * 8 + quad * 2 + (e & 1);
            if(ri >= rows || j >= total) continue;
            if(gather){
                int a = start + j;
                if(EXACT_MIX) atomicAdd(gather + (u64)tokenOf[a] * rows + ri, (unsigned long long)__float2ll_rn(weightOf[a] * acc[nt][e] * 4294967296.0f));
                else atomicAdd((float*)gather + (u64)tokenOf[a] * rows + ri, weightOf[a] * acc[nt][e]);
                continue;
            }
            output[(u64)(start + j) * rows + ri] = acc[nt][e];
        }
    }
}

__device__ __forceinline__ float exactFloat(int v) {
    return __int_as_float(v + 0x4B400000) - 12582912.0f;
}

template<int TYPE, bool FINE, bool WIDE>
__device__ __forceinline__ void dotRowsBody(const u8* weights, u64 rowBytes, u64 expertBytes, const int* groupSlot, const int* groupStart, const int* groupCount, const int* inputIndex, const i8* packed, const float* scales, const int* sums, const i8* fine, const float* fineScales, const int* fineSums, float* output, int rows, int cols, int count, int tiles, const int* tokenOf, const float* weightOf, unsigned long long* gather, GridTables t) {
    constexpr bool twin = TYPE == 14 || TYPE == 17 || TYPE == 22 || TYPE == 29 || TYPE == 40;
    constexpr bool shifted = TYPE == 12 || TYPE == 13 || TYPE == 7;
    constexpr int TOK = WIDE ? 128 : 64;
    constexpr int THREADS = WIDE ? 256 : 128;
    constexpr int STAGE = WIDE && !FINE ? 4 : 2;
    __shared__ __align__(16) int wq[STAGE][8][64];
    __shared__ __align__(16) int xq[STAGE][TOK][8];
    __shared__ __align__(16) int xf[FINE ? STAGE : 1][FINE ? TOK : 1][8];
    __shared__ __align__(16) float lowS[STAGE][64];
    __shared__ __align__(16) float highS[STAGE][64];
    __shared__ __align__(16) float offS[STAGE][64];
    __shared__ __align__(16) float xs[STAGE][TOK];
    __shared__ __align__(16) float xsum[STAGE][TOK];
    __shared__ __align__(16) float fs[FINE ? STAGE : 1][FINE ? TOK : 1];
    __shared__ __align__(16) float fsum[FINE ? STAGE : 1][FINE ? TOK : 1];
    __shared__ int source[TOK];

    int group = blockIdx.y / tiles, tile = blockIdx.y % tiles;
    int slot = groupSlot ? groupSlot[group] : 0;
    int start = groupStart ? groupStart[group] : 0;
    int total = groupCount ? groupCount[group] : count;
    int first = tile * TOK;
    if(first >= total) return;
    int rowBase = blockIdx.x * 64;
    int tid = threadIdx.x;
    if(tid < TOK){
        int j = first + tid;
        source[tid] = j >= total
            ? -1
            : inputIndex
                ? inputIndex[start + j]
                : start + j;
    }
    __syncthreads();

    int groups = cols / 32;
    const int perBlock = groupsPerBlock(TYPE);
    const int bytes = blockBytes(TYPE);
    const u8* base = weights + expertBytes * slot;
    int quad = tid & 15, band = tid >> 4;
    int r0 = quad * 4;
    int valid = min(TOK, total - first);
    float acc[4][8];
    #pragma unroll
    for(int r = 0; r < 4; r++){
        #pragma unroll
        for(int k = 0; k < 8; k++) acc[r][k] = 0.0f;
    }

    for(int u = 0; u < groups; u += STAGE){
        #pragma unroll
        for(int job = tid; job < 64 * STAGE; job += THREADS){
            int h = job >> 6, lane = job & 63;
            int g = u + h;
            int row = rowBase + lane;
            int w[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            float low = 0.0f, high = 0.0f, shift = 0.0f;
            if(row < rows && g < groups) unpackGroup(TYPE, base + rowBytes * row + (u64)(g / perBlock) * bytes, g % perBlock, w, low, high, shift, t);
            #pragma unroll
            for(int i = 0; i < 8; i++) wq[h][i][lane] = w[i];
            lowS[h][lane] = low;
            highS[h][lane] = high;
            offS[h][lane] = shift;
        }
        #pragma unroll
        for(int job = tid; job < TOK * STAGE; job += THREADS){
            int h = job / TOK, lane = job % TOK;
            int g = u + h;
            int src = source[lane];
            int4 zero = make_int4(0, 0, 0, 0);
            if(src >= 0 && g < groups){
                u64 at = (u64)src * groups + g;
                const int4* q = (const int4*)(packed + at * 32);
                *(int4*)&xq[h][lane][0] = q[0];
                *(int4*)&xq[h][lane][4] = q[1];
                xs[h][lane] = scales[at];
                xsum[h][lane] = (float)sums[at];
                if constexpr(FINE){
                    const int4* r = (const int4*)(fine + at * 32);
                    *(int4*)&xf[h][lane][0] = r[0];
                    *(int4*)&xf[h][lane][4] = r[1];
                    fs[h][lane] = fineScales[at];
                    fsum[h][lane] = (float)fineSums[at];
                }
            } else {
                *(int4*)&xq[h][lane][0] = zero;
                *(int4*)&xq[h][lane][4] = zero;
                xs[h][lane] = 0.0f;
                xsum[h][lane] = 0.0f;
                if constexpr(FINE){
                    *(int4*)&xf[h][lane][0] = zero;
                    *(int4*)&xf[h][lane][4] = zero;
                    fs[h][lane] = 0.0f;
                    fsum[h][lane] = 0.0f;
                }
            }
        }
        __syncthreads();

        #pragma unroll 1
        for(int h = 0; h < STAGE && band * 8 < valid; h++){
            if(u + h >= groups) break;
            int w[4][8];
            #pragma unroll
            for(int i = 0; i < 8; i++){
                int4 v = *(const int4*)&wq[h][i][r0];
                w[0][i] = v.x;
                w[1][i] = v.y;
                w[2][i] = v.z;
                w[3][i] = v.w;
            }
            float4 lo4 = *(const float4*)&lowS[h][r0];
            float lowR[4] = {lo4.x, lo4.y, lo4.z, lo4.w};
            float highR[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float offR[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            if constexpr(twin){
                float4 hi4 = *(const float4*)&highS[h][r0];
                highR[0] = hi4.x; highR[1] = hi4.y; highR[2] = hi4.z; highR[3] = hi4.w;
            }
            if constexpr(shifted){
                float4 of4 = *(const float4*)&offS[h][r0];
                offR[0] = of4.x; offR[1] = of4.y; offR[2] = of4.z; offR[3] = of4.w;
            }
            float sxK[8], ssK[8];
            {
                float4 a = *(const float4*)&xs[h][band * 8], b = *(const float4*)&xs[h][band * 8 + 4];
                sxK[0] = a.x; sxK[1] = a.y; sxK[2] = a.z; sxK[3] = a.w; sxK[4] = b.x; sxK[5] = b.y; sxK[6] = b.z; sxK[7] = b.w;
                if constexpr(shifted){
                    float4 c = *(const float4*)&xsum[h][band * 8], d = *(const float4*)&xsum[h][band * 8 + 4];
                    ssK[0] = c.x * a.x; ssK[1] = c.y * a.y; ssK[2] = c.z * a.z; ssK[3] = c.w * a.w; ssK[4] = d.x * b.x; ssK[5] = d.y * b.y; ssK[6] = d.z * b.z; ssK[7] = d.w * b.w;
                }
            }
            #pragma unroll
            for(int k = 0; k < 8; k++){
                int tok = band * 8 + k;
                int4 xa = *(const int4*)&xq[h][tok][0], xb = *(const int4*)&xq[h][tok][4];
                int x[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
                #pragma unroll
                for(int r = 0; r < 4; r++){
                    float base = shifted
                        ? fmaf(-offR[r], ssK[k], acc[r][k])
                        : acc[r][k];
                    if constexpr(!twin){
                        int q = 0;
                        #pragma unroll
                        for(int i = 0; i < 8; i++) q = __dp4a(w[r][i], x[i], q);
                        acc[r][k] = fmaf(sxK[k] * lowR[r], exactFloat(q), base);
                    } else {
                        int qa = 0, qb = 0;
                        #pragma unroll
                        for(int i = 0; i < 4; i++){
                            qa = __dp4a(w[r][i], x[i], qa);
                            qb = __dp4a(w[r][i + 4], x[i + 4], qb);
                        }
                        acc[r][k] = fmaf(sxK[k], fmaf(lowR[r], exactFloat(qa), highR[r] * exactFloat(qb)), base);
                    }
                }
                if constexpr(FINE){
                    int4 ya = *(const int4*)&xf[h][tok][0], yb = *(const int4*)&xf[h][tok][4];
                    int y[8] = {ya.x, ya.y, ya.z, ya.w, yb.x, yb.y, yb.z, yb.w};
                    float sf = fs[h][tok], sfs = fsum[h][tok] * fs[h][tok];
                    #pragma unroll
                    for(int r = 0; r < 4; r++){
                        float base = shifted
                            ? fmaf(-offR[r], sfs, acc[r][k])
                            : acc[r][k];
                        if constexpr(!twin){
                            int f = 0;
                            #pragma unroll
                            for(int i = 0; i < 8; i++) f = __dp4a(w[r][i], y[i], f);
                            acc[r][k] = fmaf(sf * lowR[r], exactFloat(f), base);
                        } else {
                            int fa = 0, fb = 0;
                            #pragma unroll
                            for(int i = 0; i < 4; i++){
                                fa = __dp4a(w[r][i], y[i], fa);
                                fb = __dp4a(w[r][i + 4], y[i + 4], fb);
                            }
                            acc[r][k] = fmaf(sf, fmaf(lowR[r], exactFloat(fa), highR[r] * exactFloat(fb)), base);
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for(int r = 0; r < 4; r++){
        int ri = rowBase + r0 + r;
        #pragma unroll
        for(int k = 0; k < 8; k++){
            int j = first + band * 8 + k;
            if(ri >= rows || j >= total) continue;
            if(gather){
                int a = start + j;
                if(EXACT_MIX) atomicAdd(gather + (u64)tokenOf[a] * rows + ri, (unsigned long long)__float2ll_rn(weightOf[a] * acc[r][k] * 4294967296.0f));
                else atomicAdd((float*)gather + (u64)tokenOf[a] * rows + ri, weightOf[a] * acc[r][k]);
                continue;
            }
            output[(u64)(start + j) * rows + ri] = acc[r][k];
        }
    }
}

#define DOT_ROWS(T) extern "C" __global__ void __launch_bounds__(128, 4) dotRows##T(int type, const u8* weights, u64 rowBytes, u64 expertBytes, const int* groupSlot, const int* groupStart, const int* groupCount, const int* inputIndex, const i8* packed, const float* scales, const int* sums, const i8* fine, const float* fineScales, const int* fineSums, float* output, int rows, int cols, int count, int tiles, const int* tokenOf, const float* weightOf, unsigned long long* gather, GridTables t) { dotRowsBody<T, true, false>(weights, rowBytes, expertBytes, groupSlot, groupStart, groupCount, inputIndex, packed, scales, sums, fine, fineScales, fineSums, output, rows, cols, count, tiles, tokenOf, weightOf, gather, t); } \
extern "C" __global__ void __launch_bounds__(128, 4) dotRows##T##c(int type, const u8* weights, u64 rowBytes, u64 expertBytes, const int* groupSlot, const int* groupStart, const int* groupCount, const int* inputIndex, const i8* packed, const float* scales, const int* sums, const i8* fine, const float* fineScales, const int* fineSums, float* output, int rows, int cols, int count, int tiles, const int* tokenOf, const float* weightOf, unsigned long long* gather, GridTables t) { dotRowsBody<T, false, false>(weights, rowBytes, expertBytes, groupSlot, groupStart, groupCount, inputIndex, packed, scales, sums, fine, fineScales, fineSums, output, rows, cols, count, tiles, tokenOf, weightOf, gather, t); } \
extern "C" __global__ void __launch_bounds__(256, 2) dotRowsW##T(int type, const u8* weights, u64 rowBytes, u64 expertBytes, const int* groupSlot, const int* groupStart, const int* groupCount, const int* inputIndex, const i8* packed, const float* scales, const int* sums, const i8* fine, const float* fineScales, const int* fineSums, float* output, int rows, int cols, int count, int tiles, const int* tokenOf, const float* weightOf, unsigned long long* gather, GridTables t) { dotRowsBody<T, true, true>(weights, rowBytes, expertBytes, groupSlot, groupStart, groupCount, inputIndex, packed, scales, sums, fine, fineScales, fineSums, output, rows, cols, count, tiles, tokenOf, weightOf, gather, t); } \
extern "C" __global__ void __launch_bounds__(256, 2) dotRowsW##T##c(int type, const u8* weights, u64 rowBytes, u64 expertBytes, const int* groupSlot, const int* groupStart, const int* groupCount, const int* inputIndex, const i8* packed, const float* scales, const int* sums, const i8* fine, const float* fineScales, const int* fineSums, float* output, int rows, int cols, int count, int tiles, const int* tokenOf, const float* weightOf, unsigned long long* gather, GridTables t) { dotRowsBody<T, false, true>(weights, rowBytes, expertBytes, groupSlot, groupStart, groupCount, inputIndex, packed, scales, sums, fine, fineScales, fineSums, output, rows, cols, count, tiles, tokenOf, weightOf, gather, t); }
DOT_ROWS(7)
DOT_ROWS(8)
DOT_ROWS(11)
DOT_ROWS(12)
DOT_ROWS(13)
DOT_ROWS(14)
DOT_ROWS(16)
DOT_ROWS(17)
DOT_ROWS(18)
DOT_ROWS(20)
DOT_ROWS(21)
DOT_ROWS(22)
DOT_ROWS(23)
DOT_ROWS(29)
DOT_ROWS(39)
DOT_ROWS(40)
DOT_ROWS(42)

__device__ __forceinline__ void fullStep(int* c, int a0, int a1, int a2, int a3, int b0, int b1) {
#if __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};" : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(0), "r"(0), "r"(0), "r"(0));
#else
    c[0] = c[1] = c[2] = c[3] = 0;
#endif
}


__device__ __forceinline__ void floatStep(float* c, int a0, int a1, int a2, int a3, int b0, int b1) {
    int q[4];
#if __CUDA_ARCH__ >= 800
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};" : "=r"(q[0]), "=r"(q[1]), "=r"(q[2]), "=r"(q[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(0x4B400000));
#else
    q[0] = q[1] = q[2] = q[3] = 0x4B400000;
#endif
    #pragma unroll
    for(int i = 0; i < 4; i++) c[i] = __int_as_float(q[i]) - 12582912.0f;
}

__device__ __forceinline__ void floatHalfStep(float* c, int a0, int a1, int b) {
    int q[4];
#if __CUDA_ARCH__ >= 800
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%7};" : "=r"(q[0]), "=r"(q[1]), "=r"(q[2]), "=r"(q[3]) : "r"(a0), "r"(a1), "r"(b), "r"(0x4B400000));
#else
    q[0] = q[1] = q[2] = q[3] = 0x4B400000;
#endif
    #pragma unroll
    for(int i = 0; i < 4; i++) c[i] = __int_as_float(q[i]) - 12582912.0f;
}

template<int STAGE, int TOK = 64>
struct DeepSpace {
    i8 wq[64][STAGE * 32 + 16];
    i8 xq[TOK][STAGE * 32 + 16];
    float4 rowMarks[STAGE][64];
    float2 marks[STAGE][TOK];
    int source[TOK];
};

template<int STAGE, int TOK = 64>
struct DeepFine {
    i8 xf[TOK][STAGE * 32 + 16];
    float xb[STAGE][TOK];
};

template<bool SPLIT, bool FINE, int STAGE, int TOK, bool SHIFT>
__device__ __forceinline__ void deepRows(DeepSpace<STAGE, TOK>& s, DeepFine<STAGE, TOK>* f, int type, const u8* weights, u64 rowBytes, u64 expertBytes, const int* groupSlot, const int* groupStart, const int* groupCount, const int* inputIndex, const i8* packed, const float* scales, const int* sums, const i8* fine, const float* fineScales, const int* fineSums, float* output, int rows, int cols, int count, int tiles, const int* tokenOf, const float* weightOf, unsigned long long* gather, GridTables t) {
    int group = blockIdx.y / tiles, tile = blockIdx.y % tiles;
    int slot = groupSlot ? groupSlot[group] : 0;
    int start = groupStart ? groupStart[group] : 0;
    int total = groupCount ? groupCount[group] : count;
    int first = tile * TOK;
    if(first >= total) return;
    int rowBase = blockIdx.x * 64;
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2, quad = lane & 3;
    int band = (warp >> 2) * 64;
    if(tid < TOK){
        int j = first + tid;
        s.source[tid] = j >= total
            ? -1
            : inputIndex
                ? inputIndex[start + j]
                : start + j;
    }
    __syncthreads();

    int groups = cols / 32;
    int perBlock = groupsPerBlock(type);
    int bytes = blockBytes(type);
    const u8* base = weights + expertBytes * slot;
    float acc[8][4];
    #pragma unroll
    for(int nt = 0; nt < 8; nt++){
        #pragma unroll
        for(int e = 0; e < 4; e++) acc[nt][e] = 0.0f;
    }

    for(int u = 0; u < groups; u += STAGE){
        #pragma unroll
        for(int job = tid; job < 64 * STAGE; job += 2 * TOK){
            int r = job / STAGE, h = job % STAGE;
            int g = u + h;
            int row = rowBase + r;
            int w[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            float low = 0.0f, high = 0.0f, shift = 0.0f;
            if(row < rows && g < groups) unpackGroup(type, base + rowBytes * row + (u64)(g / perBlock) * bytes, g % perBlock, w, low, high, shift, t);
            int4* into = (int4*)&s.wq[r][h * 32];
            into[0] = make_int4(w[0], w[1], w[2], w[3]);
            into[1] = make_int4(w[4], w[5], w[6], w[7]);
            s.rowMarks[h][r] = make_float4(low, high, shift, 0.0f);
        }
        #pragma unroll
        for(int job = tid; job < TOK * STAGE; job += 2 * TOK){
            int k = job / STAGE, h = job % STAGE;
            int g = u + h;
            int src = s.source[k];
            int4 zero = make_int4(0, 0, 0, 0);
            int4* intoQ = (int4*)&s.xq[k][h * 32];
            if(src < 0 || g >= groups){
                intoQ[0] = zero;
                intoQ[1] = zero;
                s.marks[h][k] = make_float2(0.0f, 0.0f);
                if(FINE){
                    int4* intoF = (int4*)&f->xf[k][h * 32];
                    intoF[0] = zero;
                    intoF[1] = zero;
                    f->xb[h][k] = 0.0f;
                }
                continue;
            }
            u64 at = (u64)src * groups + g;
            const int4* qv = (const int4*)(packed + at * 32);
            intoQ[0] = qv[0];
            intoQ[1] = qv[1];
            float a = scales[at];
            float c = a * (float)sums[at];
            if(FINE){
                const int4* rv = (const int4*)(fine + at * 32);
                int4* intoF = (int4*)&f->xf[k][h * 32];
                intoF[0] = rv[0];
                intoF[1] = rv[1];
                float b = fineScales[at];
                f->xb[h][k] = b;
                c += b * (float)fineSums[at];
            }
            s.marks[h][k] = make_float2(a, c);
        }
        __syncthreads();

        int r0 = (warp & 3) * 16 + gid;
        #pragma unroll
        for(int h = 0; h < STAGE; h++){
            int a0 = *(const int*)&s.wq[r0][h * 32 + quad * 4];
            int a1 = *(const int*)&s.wq[r0 + 8][h * 32 + quad * 4];
            int a2 = *(const int*)&s.wq[r0][h * 32 + 16 + quad * 4];
            int a3 = *(const int*)&s.wq[r0 + 8][h * 32 + 16 + quad * 4];
            float4 rowA = s.rowMarks[h][r0], rowB = s.rowMarks[h][r0 + 8];
            float lowA = rowA.x, highA = rowA.y, offA = rowA.z;
            float lowB = rowB.x, highB = rowB.y, offB = rowB.z;
            #pragma unroll
            for(int nt = 0; nt < 8; nt++){
                int tok = band + nt * 8 + gid;
                int t0 = band + nt * 8 + quad * 2;
                float4 pair = *(const float4*)&s.marks[h][t0];
                float sa0 = pair.x, sc0 = pair.y, sa1 = pair.z, sc1 = pair.w;
                int b0 = *(const int*)&s.xq[tok][h * 32 + quad * 4];
                int b1 = *(const int*)&s.xq[tok][h * 32 + 16 + quad * 4];
                if(!SPLIT){
                    float q[4];
                    floatStep(q, a0, a1, a2, a3, b0, b1);
                    float v0 = sa0 * q[0], v1 = sa1 * q[1], v2 = sa0 * q[2], v3 = sa1 * q[3];
                    if(FINE){
                        float sb0 = f->xb[h][t0], sb1 = f->xb[h][t0 + 1];
                        float r[4];
                        floatStep(r, a0, a1, a2, a3, *(const int*)&f->xf[tok][h * 32 + quad * 4], *(const int*)&f->xf[tok][h * 32 + 16 + quad * 4]);
                        v0 += sb0 * r[0];
                        v1 += sb1 * r[1];
                        v2 += sb0 * r[2];
                        v3 += sb1 * r[3];
                    }
                    if constexpr(SHIFT){
                        acc[nt][0] += lowA * v0 - offA * sc0;
                        acc[nt][1] += lowA * v1 - offA * sc1;
                        acc[nt][2] += lowB * v2 - offB * sc0;
                        acc[nt][3] += lowB * v3 - offB * sc1;
                        continue;
                    }
                    acc[nt][0] = fmaf(lowA, v0, acc[nt][0]);
                    acc[nt][1] = fmaf(lowA, v1, acc[nt][1]);
                    acc[nt][2] = fmaf(lowB, v2, acc[nt][2]);
                    acc[nt][3] = fmaf(lowB, v3, acc[nt][3]);
                    continue;
                }
                float qL[4], qH[4];
                floatHalfStep(qL, a0, a1, b0);
                floatHalfStep(qH, a2, a3, b1);
                if constexpr(!FINE && !SHIFT){
                    acc[nt][0] = fmaf(sa0, fmaf(highA, qH[0], lowA * qL[0]), acc[nt][0]);
                    acc[nt][1] = fmaf(sa1, fmaf(highA, qH[1], lowA * qL[1]), acc[nt][1]);
                    acc[nt][2] = fmaf(sa0, fmaf(highB, qH[2], lowB * qL[2]), acc[nt][2]);
                    acc[nt][3] = fmaf(sa1, fmaf(highB, qH[3], lowB * qL[3]), acc[nt][3]);
                    continue;
                }
                float v0 = sa0 * (lowA * qL[0] + highA * qH[0]);
                float v1 = sa1 * (lowA * qL[1] + highA * qH[1]);
                float v2 = sa0 * (lowB * qL[2] + highB * qH[2]);
                float v3 = sa1 * (lowB * qL[3] + highB * qH[3]);
                if(FINE){
                    float sb0 = f->xb[h][t0], sb1 = f->xb[h][t0 + 1];
                    float fL[4], fH[4];
                    floatHalfStep(fL, a0, a1, *(const int*)&f->xf[tok][h * 32 + quad * 4]);
                    floatHalfStep(fH, a2, a3, *(const int*)&f->xf[tok][h * 32 + 16 + quad * 4]);
                    v0 += sb0 * (lowA * fL[0] + highA * fH[0]);
                    v1 += sb1 * (lowA * fL[1] + highA * fH[1]);
                    v2 += sb0 * (lowB * fL[2] + highB * fH[2]);
                    v3 += sb1 * (lowB * fL[3] + highB * fH[3]);
                }
                acc[nt][0] += v0 - offA * sc0;
                acc[nt][1] += v1 - offA * sc1;
                acc[nt][2] += v2 - offB * sc0;
                acc[nt][3] += v3 - offB * sc1;
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for(int nt = 0; nt < 8; nt++){
        #pragma unroll
        for(int e = 0; e < 4; e++){
            int ri = rowBase + (warp & 3) * 16 + gid + (e >> 1) * 8;
            int j = first + band + nt * 8 + quad * 2 + (e & 1);
            if(ri >= rows || j >= total) continue;
            if(gather){
                int a = start + j;
                if(EXACT_MIX) atomicAdd(gather + (u64)tokenOf[a] * rows + ri, (unsigned long long)__float2ll_rn(weightOf[a] * acc[nt][e] * 4294967296.0f));
                else atomicAdd((float*)gather + (u64)tokenOf[a] * rows + ri, weightOf[a] * acc[nt][e]);
                continue;
            }
            output[(u64)(start + j) * rows + ri] = acc[nt][e];
        }
    }
}

#define DEEP_ARGS int type, const u8* weights, u64 rowBytes, u64 expertBytes, const int* groupSlot, const int* groupStart, const int* groupCount, const int* inputIndex, const i8* packed, const float* scales, const int* sums, const i8* fine, const float* fineScales, const int* fineSums, float* output, int rows, int cols, int count, int tiles, const int* tokenOf, const float* weightOf, unsigned long long* gather, GridTables t
#define DEEP_PASS type, weights, rowBytes, expertBytes, groupSlot, groupStart, groupCount, inputIndex, packed, scales, sums, fine, fineScales, fineSums, output, rows, cols, count, tiles, tokenOf, weightOf, gather, t


template<int STAGE>
struct DeepTall {
    i8 wq[128][STAGE * 32 + 16];
    i8 xq[128][STAGE * 32 + 16];
    float low[STAGE][128];
    float high[STAGE][128];
    float shift[STAGE][128];
    float2 marks[STAGE][128];
    int source[128];
};

template<bool SPLIT, int STAGE, bool SHIFT>
__device__ __forceinline__ void deepTall(DeepTall<STAGE>& s, int type, const u8* weights, u64 rowBytes, u64 expertBytes, const int* groupSlot, const int* groupStart, const int* groupCount, const int* inputIndex, const i8* packed, const float* scales, const int* sums, float* output, int rows, int cols, int count, int tiles, const int* tokenOf, const float* weightOf, unsigned long long* gather, GridTables t) {
    int group = blockIdx.y / tiles, tile = blockIdx.y % tiles;
    int slot = groupSlot ? groupSlot[group] : 0;
    int start = groupStart ? groupStart[group] : 0;
    int total = groupCount ? groupCount[group] : count;
    int first = tile * 128;
    if(first >= total) return;
    int rowBase = blockIdx.x * 128;
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2, quad = lane & 3;
    if(tid < 128){
        int j = first + tid;
        s.source[tid] = j >= total
            ? -1
            : inputIndex
                ? inputIndex[start + j]
                : start + j;
    }
    __syncthreads();

    int groups = cols / 32;
    int perBlock = groupsPerBlock(type);
    int bytes = blockBytes(type);
    const u8* base = weights + expertBytes * slot;
    int live = min(16, (total - first + 7) / 8);
    float acc[16][4];
    #pragma unroll
    for(int nt = 0; nt < 16; nt++){
        #pragma unroll
        for(int e = 0; e < 4; e++) acc[nt][e] = 0.0f;
    }

    for(int u = 0; u < groups; u += STAGE){
        #pragma unroll
        for(int job = tid; job < 128 * STAGE; job += 256){
            int r = job / STAGE, h = job % STAGE;
            int g = u + h;
            int row = rowBase + r;
            int w[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            float lo = 0.0f, hi = 0.0f, sh = 0.0f;
            if(row < rows && g < groups) unpackGroup(type, base + rowBytes * row + (u64)(g / perBlock) * bytes, g % perBlock, w, lo, hi, sh, t);
            int4* into = (int4*)&s.wq[r][h * 32];
            into[0] = make_int4(w[0], w[1], w[2], w[3]);
            into[1] = make_int4(w[4], w[5], w[6], w[7]);
            s.low[h][r] = lo;
            s.high[h][r] = hi;
            s.shift[h][r] = sh;
        }
        #pragma unroll
        for(int job = tid; job < 128 * STAGE; job += 256){
            int k = job / STAGE, h = job % STAGE;
            int g = u + h;
            int src = s.source[k];
            int4* intoQ = (int4*)&s.xq[k][h * 32];
            if(src < 0 || g >= groups){
                int4 zero = make_int4(0, 0, 0, 0);
                intoQ[0] = zero;
                intoQ[1] = zero;
                s.marks[h][k] = make_float2(0.0f, 0.0f);
                continue;
            }
            u64 at = (u64)src * groups + g;
            const int4* qv = (const int4*)(packed + at * 32);
            intoQ[0] = qv[0];
            intoQ[1] = qv[1];
            float a = scales[at];
            s.marks[h][k] = make_float2(a, a * (float)sums[at]);
        }
        __syncthreads();

        int r0 = warp * 16 + gid;
        #pragma unroll
        for(int h = 0; h < STAGE; h++){
            int a0 = *(const int*)&s.wq[r0][h * 32 + quad * 4];
            int a1 = *(const int*)&s.wq[r0 + 8][h * 32 + quad * 4];
            int a2 = *(const int*)&s.wq[r0][h * 32 + 16 + quad * 4];
            int a3 = *(const int*)&s.wq[r0 + 8][h * 32 + 16 + quad * 4];
            float lowA = s.low[h][r0], highA = s.high[h][r0], offA = s.shift[h][r0];
            float lowB = s.low[h][r0 + 8], highB = s.high[h][r0 + 8], offB = s.shift[h][r0 + 8];
            #pragma unroll
            for(int nt = 0; nt < 16; nt++){
                if(nt >= live) break;
                int tok = nt * 8 + gid;
                int t0 = nt * 8 + quad * 2;
                float4 pair = *(const float4*)&s.marks[h][t0];
                float sa0 = pair.x, sc0 = pair.y, sa1 = pair.z, sc1 = pair.w;
                int b0 = *(const int*)&s.xq[tok][h * 32 + quad * 4];
                int b1 = *(const int*)&s.xq[tok][h * 32 + 16 + quad * 4];
                if(!SPLIT){
                    float q[4];
                    floatStep(q, a0, a1, a2, a3, b0, b1);
                    float v0 = sa0 * q[0], v1 = sa1 * q[1], v2 = sa0 * q[2], v3 = sa1 * q[3];
                    if constexpr(SHIFT){
                        acc[nt][0] += lowA * v0 - offA * sc0;
                        acc[nt][1] += lowA * v1 - offA * sc1;
                        acc[nt][2] += lowB * v2 - offB * sc0;
                        acc[nt][3] += lowB * v3 - offB * sc1;
                        continue;
                    }
                    acc[nt][0] = fmaf(lowA, v0, acc[nt][0]);
                    acc[nt][1] = fmaf(lowA, v1, acc[nt][1]);
                    acc[nt][2] = fmaf(lowB, v2, acc[nt][2]);
                    acc[nt][3] = fmaf(lowB, v3, acc[nt][3]);
                    continue;
                }
                float qL[4], qH[4];
                floatHalfStep(qL, a0, a1, b0);
                floatHalfStep(qH, a2, a3, b1);
                acc[nt][0] = fmaf(sa0, fmaf(highA, qH[0], lowA * qL[0]), acc[nt][0]);
                acc[nt][1] = fmaf(sa1, fmaf(highA, qH[1], lowA * qL[1]), acc[nt][1]);
                acc[nt][2] = fmaf(sa0, fmaf(highB, qH[2], lowB * qL[2]), acc[nt][2]);
                acc[nt][3] = fmaf(sa1, fmaf(highB, qH[3], lowB * qL[3]), acc[nt][3]);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for(int nt = 0; nt < 16; nt++){
        #pragma unroll
        for(int e = 0; e < 4; e++){
            int ri = rowBase + warp * 16 + gid + (e >> 1) * 8;
            int j = first + nt * 8 + quad * 2 + (e & 1);
            if(ri >= rows || j >= total) continue;
            if(gather){
                int a = start + j;
                if(EXACT_MIX) atomicAdd(gather + (u64)tokenOf[a] * rows + ri, (unsigned long long)__float2ll_rn(weightOf[a] * acc[nt][e] * 4294967296.0f));
                else atomicAdd((float*)gather + (u64)tokenOf[a] * rows + ri, weightOf[a] * acc[nt][e]);
                continue;
            }
            output[(u64)(start + j) * rows + ri] = acc[nt][e];
        }
    }
}

#define TALL_PASS type, weights, rowBytes, expertBytes, groupSlot, groupStart, groupCount, inputIndex, packed, scales, sums, output, rows, cols, count, tiles, tokenOf, weightOf, gather, t

extern "C" __global__ void __launch_bounds__(256) mmaDeepLeanX(DEEP_ARGS) {
    __shared__ __align__(16) DeepTall<4> s;
    if(type == 7 || type == 12 || type == 13) deepTall<false, 4, true>(s, TALL_PASS);
    else deepTall<false, 4, false>(s, TALL_PASS);
}

extern "C" __global__ void __launch_bounds__(256) mmaDeepSplitLeanX(DEEP_ARGS) {
    __shared__ __align__(16) DeepTall<4> s;
    deepTall<true, 4, false>(s, TALL_PASS);
}

extern "C" __global__ void __launch_bounds__(128) mmaDeep(DEEP_ARGS) {
    __shared__ __align__(16) DeepSpace<4> s;
    __shared__ __align__(16) DeepFine<4> f;
    if(type == 7 || type == 12 || type == 13) deepRows<false, true, 4, 64, true>(s, &f, DEEP_PASS);
    else deepRows<false, true, 4, 64, false>(s, &f, DEEP_PASS);
}

extern "C" __global__ void __launch_bounds__(128) mmaDeepSplit(DEEP_ARGS) {
    __shared__ __align__(16) DeepSpace<4> s;
    __shared__ __align__(16) DeepFine<4> f;
    deepRows<true, true, 4, 64, false>(s, &f, DEEP_PASS);
}

extern "C" __global__ void __launch_bounds__(128) mmaDeepLean(DEEP_ARGS) {
    __shared__ __align__(16) DeepSpace<4> s;
    if(type == 7 || type == 12 || type == 13) deepRows<false, false, 4, 64, true>(s, nullptr, DEEP_PASS);
    else deepRows<false, false, 4, 64, false>(s, nullptr, DEEP_PASS);
}

extern "C" __global__ void __launch_bounds__(128) mmaDeepSplitLean(DEEP_ARGS) {
    __shared__ __align__(16) DeepSpace<8> s;
    deepRows<true, false, 8, 64, false>(s, nullptr, DEEP_PASS);
}

extern "C" __global__ void __launch_bounds__(256) mmaDeepW(DEEP_ARGS) {
    __shared__ __align__(16) DeepSpace<2, 128> s;
    __shared__ __align__(16) DeepFine<2, 128> f;
    if(type == 7 || type == 12 || type == 13) deepRows<false, true, 2, 128, true>(s, &f, DEEP_PASS);
    else deepRows<false, true, 2, 128, false>(s, &f, DEEP_PASS);
}

extern "C" __global__ void __launch_bounds__(256) mmaDeepSplitW(DEEP_ARGS) {
    __shared__ __align__(16) DeepSpace<2, 128> s;
    __shared__ __align__(16) DeepFine<2, 128> f;
    deepRows<true, true, 2, 128, false>(s, &f, DEEP_PASS);
}

extern "C" __global__ void __launch_bounds__(256) mmaDeepLeanW(DEEP_ARGS) {
    __shared__ __align__(16) DeepSpace<4, 128> s;
    if(type == 7 || type == 12 || type == 13) deepRows<false, false, 4, 128, true>(s, nullptr, DEEP_PASS);
    else deepRows<false, false, 4, 128, false>(s, nullptr, DEEP_PASS);
}

extern "C" __global__ void __launch_bounds__(256) mmaDeepSplitLeanW(DEEP_ARGS) {
    __shared__ __align__(16) DeepSpace<4, 128> s;
    deepRows<true, false, 4, 128, false>(s, nullptr, DEEP_PASS);
}

extern "C" __global__ void mmaWide(int type, const u8* weights, u64 rowBytes, u64 expertBytes, const int* groupSlot, const int* groupStart, const int* groupCount, const int* inputIndex, const i8* packed, const float* scales, const int* sums, const i8* fine, const float* fineScales, const int* fineSums, float* output, int rows, int cols, int count, int tiles, const int* tokenOf, const float* weightOf, unsigned long long* gather, GridTables t) {
    __shared__ __align__(16) i8 wq[64][48];
    __shared__ __align__(16) i8 xq[128][48];
    __shared__ __align__(16) i8 xf[128][48];
    __shared__ float lowS[64], highS[64], offS[64];
    __shared__ float xa[128], xb[128], xc[128], xd[128];
    __shared__ int source[128];

    int group = blockIdx.y / tiles, tile = blockIdx.y % tiles;
    int slot = groupSlot ? groupSlot[group] : 0;
    int start = groupStart ? groupStart[group] : 0;
    int total = groupCount ? groupCount[group] : count;
    int first = tile * 128;
    if(first >= total) return;
    int rowBase = blockIdx.x * 64;
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2, quad = lane & 3;
    int warpRow = (warp & 3) * 16, warpTok = (warp >> 2) * 64;
    bool uniform = type != 14 && type != 17 && type != 22 && type != 29 && type != 40;
    if(tid < 128){
        int j = first + tid;
        source[tid] = j >= total
            ? -1
            : inputIndex
                ? inputIndex[start + j]
                : start + j;
    }
    __syncthreads();

    int groups = cols / 32;
    int perBlock = groupsPerBlock(type);
    int bytes = blockBytes(type);
    const u8* base = weights + expertBytes * slot;
    float acc[8][4];
    #pragma unroll
    for(int nt = 0; nt < 8; nt++){
        #pragma unroll
        for(int e = 0; e < 4; e++) acc[nt][e] = 0.0f;
    }

    for(int u = 0; u < groups; u++){
        if(tid < 64){
            int row = rowBase + tid;
            int w[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            float low = 0.0f, high = 0.0f, shift = 0.0f;
            if(row < rows) unpackGroup(type, base + rowBytes * row + (u64)(u / perBlock) * bytes, u % perBlock, w, low, high, shift, t);
            int4* into = (int4*)wq[tid];
            into[0] = make_int4(w[0], w[1], w[2], w[3]);
            into[1] = make_int4(w[4], w[5], w[6], w[7]);
            lowS[tid] = low;
            highS[tid] = high;
            offS[tid] = shift;
        } else if(tid < 192){
            int k = tid - 64;
            int src = source[k];
            int4 zero = make_int4(0, 0, 0, 0);
            int4* intoQ = (int4*)xq[k];
            int4* intoF = (int4*)xf[k];
            if(src >= 0){
                u64 at = (u64)src * groups + u;
                const int4* qv = (const int4*)(packed + at * 32);
                intoQ[0] = qv[0];
                intoQ[1] = qv[1];
                float a = scales[at];
                float qs = (float)sums[at];
                float b = 0.0f, fs = 0.0f;
                if(fine){
                    const int4* rv = (const int4*)(fine + at * 32);
                    intoF[0] = rv[0];
                    intoF[1] = rv[1];
                    b = fineScales[at];
                    fs = (float)fineSums[at];
                } else {
                    intoF[0] = zero;
                    intoF[1] = zero;
                }
                xa[k] = a;
                xb[k] = b;
                xc[k] = a * qs + b * fs;
                xd[k] = 0.0f;
            } else {
                intoQ[0] = zero;
                intoQ[1] = zero;
                intoF[0] = zero;
                intoF[1] = zero;
                xa[k] = 0.0f;
                xb[k] = 0.0f;
                xc[k] = 0.0f;
                xd[k] = 0.0f;
            }
        }
        __syncthreads();

        int r0 = warpRow + gid;
        int a0 = *(const int*)&wq[r0][quad * 4];
        int a1 = *(const int*)&wq[r0 + 8][quad * 4];
        int a2 = *(const int*)&wq[r0][16 + quad * 4];
        int a3 = *(const int*)&wq[r0 + 8][16 + quad * 4];
        float lowA = lowS[r0], highA = highS[r0], offA = offS[r0];
        float lowB = lowS[r0 + 8], highB = highS[r0 + 8], offB = offS[r0 + 8];
        #pragma unroll
        for(int nt = 0; nt < 8; nt++){
            int tok = warpTok + nt * 8 + gid;
            int t0 = warpTok + nt * 8 + quad * 2;
            float sa0 = xa[t0], sa1 = xa[t0 + 1];
            float sb0 = xb[t0], sb1 = xb[t0 + 1];
            float sc0 = xc[t0], sc1 = xc[t0 + 1];
            if(uniform){
                int q[4], f[4];
                fullStep(q, a0, a1, a2, a3, *(const int*)&xq[tok][quad * 4], *(const int*)&xq[tok][16 + quad * 4]);
                fullStep(f, a0, a1, a2, a3, *(const int*)&xf[tok][quad * 4], *(const int*)&xf[tok][16 + quad * 4]);
                acc[nt][0] += lowA * (sa0 * (float)q[0] + sb0 * (float)f[0]) - offA * sc0;
                acc[nt][1] += lowA * (sa1 * (float)q[1] + sb1 * (float)f[1]) - offA * sc1;
                acc[nt][2] += lowB * (sa0 * (float)q[2] + sb0 * (float)f[2]) - offB * sc0;
                acc[nt][3] += lowB * (sa1 * (float)q[3] + sb1 * (float)f[3]) - offB * sc1;
                continue;
            }
            int qL[4], qH[4], fL[4], fH[4];
            tensorStep(qL, a0, a1, *(const int*)&xq[tok][quad * 4]);
            tensorStep(qH, a2, a3, *(const int*)&xq[tok][16 + quad * 4]);
            tensorStep(fL, a0, a1, *(const int*)&xf[tok][quad * 4]);
            tensorStep(fH, a2, a3, *(const int*)&xf[tok][16 + quad * 4]);
            acc[nt][0] += sa0 * (lowA * (float)qL[0] + highA * (float)qH[0]) + sb0 * (lowA * (float)fL[0] + highA * (float)fH[0]) - offA * sc0;
            acc[nt][1] += sa1 * (lowA * (float)qL[1] + highA * (float)qH[1]) + sb1 * (lowA * (float)fL[1] + highA * (float)fH[1]) - offA * sc1;
            acc[nt][2] += sa0 * (lowB * (float)qL[2] + highB * (float)qH[2]) + sb0 * (lowB * (float)fL[2] + highB * (float)fH[2]) - offB * sc0;
            acc[nt][3] += sa1 * (lowB * (float)qL[3] + highB * (float)qH[3]) + sb1 * (lowB * (float)fL[3] + highB * (float)fH[3]) - offB * sc1;
        }
        __syncthreads();
    }

    #pragma unroll
    for(int nt = 0; nt < 8; nt++){
        #pragma unroll
        for(int e = 0; e < 4; e++){
            int ri = rowBase + warpRow + gid + (e >> 1) * 8;
            int j = first + warpTok + nt * 8 + quad * 2 + (e & 1);
            if(ri >= rows || j >= total) continue;
            if(gather){
                int a = start + j;
                if(EXACT_MIX) atomicAdd(gather + (u64)tokenOf[a] * rows + ri, (unsigned long long)__float2ll_rn(weightOf[a] * acc[nt][e] * 4294967296.0f));
                else atomicAdd((float*)gather + (u64)tokenOf[a] * rows + ri, weightOf[a] * acc[nt][e]);
                continue;
            }
            output[(u64)(start + j) * rows + ri] = acc[nt][e];
        }
    }
}

extern "C" __global__ void rowsStaged(int type, const u8* weights, u64 rowBytes, u64 expertBytes, const int* groupSlot, const int* groupCount, int inputStride, const i8* packed, const float* scales, const int* sums, const i8* fine, const float* fineScales, const int* fineSums, float* output, int rows, int cols, int stageBytes, GridTables t) {
    extern __shared__ __align__(16) u8 stage[];
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp;
    int group = blockIdx.y;
    if(row >= rows) return;
    if(groupCount && groupCount[group] == 0) return;
    int slot = groupSlot ? groupSlot[group] : 0;
    const u8* line = weights + expertBytes * slot + rowBytes * row;
    u64 start = (u64)line & ~15ull;
    int offset = (int)((u64)line - start);
    int span = (offset + (int)rowBytes + 15) >> 4;
    u8* mine = stage + warp * stageBytes;
    const uint4* from = (const uint4*)start;
    uint4* into = (uint4*)mine;
    for(int i = lane; i < span; i += 32) into[i] = __ldg(from + i);
    __syncwarp();

    int groups = cols / 32;
    int perBlock = groupsPerBlock(type);
    int bytes = blockBytes(type);
    u64 base = (u64)group * inputStride / 32;
    const u8* line0 = mine + offset;
    float acc = 0.0f;
    for(int u = lane; u < groups; u += 32){
        int w[8];
        float low, high, shift;
        unpackGroup(type, line0 + (u / perBlock) * bytes, u % perBlock, w, low, high, shift, t);
        u64 at = base + u;
        const int4* q = (const int4*)(packed + at * 32);
        int4 a = __ldg(q);
        int4 b = __ldg(q + 1);
        int lowSum = __dp4a(w[3], a.w, __dp4a(w[2], a.z, __dp4a(w[1], a.y, __dp4a(w[0], a.x, 0))));
        int highSum = __dp4a(w[7], b.w, __dp4a(w[6], b.z, __dp4a(w[5], b.y, __dp4a(w[4], b.x, 0))));
        float part = __ldg(scales + at) * (low * (float)lowSum + high * (float)highSum - shift * (float)__ldg(sums + at));
        if(fine){
            const int4* r = (const int4*)(fine + at * 32);
            int4 c = __ldg(r);
            int4 e = __ldg(r + 1);
            int lowFine = __dp4a(w[3], c.w, __dp4a(w[2], c.z, __dp4a(w[1], c.y, __dp4a(w[0], c.x, 0))));
            int highFine = __dp4a(w[7], e.w, __dp4a(w[6], e.z, __dp4a(w[5], e.y, __dp4a(w[4], e.x, 0))));
            part += __ldg(fineScales + at) * (low * (float)lowFine + high * (float)highFine - shift * (float)__ldg(fineSums + at));
        }
        acc += part;
    }
    for(int step = 16; step > 0; step >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, step);
    if(lane == 0) output[(u64)group * rows + row] = acc;
}

extern "C" __global__ void rowsStagedSet(StagedSet set, int inputStride, const i8* packed, const float* scales, const int* sums, const i8* fine, const float* fineScales, const int* fineSums, int rows, int cols, int stageBytes, GridTables t) {
    extern __shared__ __align__(16) u8 stage[];
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp;
    int which = set.count - 1 - (int)(blockIdx.y / set.per), group = blockIdx.y % set.per;
    if(row >= rows || which < 0) return;
    const int* used = (const int*)set.used[which];
    if(used && used[group] == 0) return;
    const int* slots = (const int*)set.slot[which];
    int slot = slots ? slots[group] : 0;
    int type = set.type[which];
    u64 rowBytes = set.rowBytes[which];
    const u8* line = (const u8*)set.weights[which] + set.expertBytes[which] * slot + rowBytes * row;
    u64 start = (u64)line & ~15ull;
    int offset = (int)((u64)line - start);
    int span = (offset + (int)rowBytes + 15) >> 4;
    u8* mine = stage + warp * stageBytes;
    const uint4* from = (const uint4*)start;
    uint4* into = (uint4*)mine;
    for(int i = lane; i < span; i += 32) into[i] = __ldg(from + i);
    __syncwarp();

    int groups = cols / 32;
    int perBlock = groupsPerBlock(type);
    int bytes = blockBytes(type);
    const int* inputs = (const int*)set.input[which];
    u64 base = (u64)(inputs ? inputs[group] : group + set.shift[which]) * inputStride / 32;
    const u8* line0 = mine + offset;
    float acc = 0.0f;
    for(int u = lane; u < groups; u += 32){
        int w[8];
        float low, high, shift;
        unpackGroup(type, line0 + (u / perBlock) * bytes, u % perBlock, w, low, high, shift, t);
        u64 at = base + u;
        const int4* q = (const int4*)(packed + at * 32);
        int4 a = __ldg(q);
        int4 b = __ldg(q + 1);
        int lowSum = __dp4a(w[3], a.w, __dp4a(w[2], a.z, __dp4a(w[1], a.y, __dp4a(w[0], a.x, 0))));
        int highSum = __dp4a(w[7], b.w, __dp4a(w[6], b.z, __dp4a(w[5], b.y, __dp4a(w[4], b.x, 0))));
        float part = __ldg(scales + at) * (low * (float)lowSum + high * (float)highSum - shift * (float)__ldg(sums + at));
        if(fine){
            const int4* r = (const int4*)(fine + at * 32);
            int4 c = __ldg(r);
            int4 e = __ldg(r + 1);
            int lowFine = __dp4a(w[3], c.w, __dp4a(w[2], c.z, __dp4a(w[1], c.y, __dp4a(w[0], c.x, 0))));
            int highFine = __dp4a(w[7], e.w, __dp4a(w[6], e.z, __dp4a(w[5], e.y, __dp4a(w[4], e.x, 0))));
            part += __ldg(fineScales + at) * (low * (float)lowFine + high * (float)highFine - shift * (float)__ldg(fineSums + at));
        }
        acc += part;
    }
    for(int step = 16; step > 0; step >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, step);
    if(lane == 0) ((float*)set.output[which])[(u64)group * rows + row] = acc;
}

template<int R, int T>
__device__ __forceinline__ void manyRows(const StagedSet& set, int inputStride, const i8* packed, const float* scales, const int* sums, const i8* fine, const float* fineScales, const int* fineSums, int rows, int cols, int stageBytes, const GridTables& t) {
    extern __shared__ __align__(16) u8 stage[];
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int first = (blockIdx.x * 8 + warp) * R;
    int which = set.count - 1 - (int)(blockIdx.y / set.per), group = blockIdx.y % set.per;
    if(first >= rows || which < 0) return;
    const int* used = (const int*)set.used[which];
    if(used && used[group] == 0) return;
    const int* slots = (const int*)set.slot[which];
    int slot = slots ? slots[group] : 0;
    int type = T >= 0 ? T : set.type[which];
    u64 rowBytes = set.rowBytes[which];
    int count = min(R, rows - first);
    const u8* line = (const u8*)set.weights[which] + set.expertBytes[which] * slot + rowBytes * first;
    u64 start = (u64)line & ~15ull;
    int offset = (int)((u64)line - start);
    int span = (offset + (int)rowBytes * count + 15) >> 4;
    u8* mine = stage + warp * stageBytes;
    const uint4* from = (const uint4*)start;
    uint4* into = (uint4*)mine;
    for(int i = lane; i < span; i += 32) into[i] = __ldg(from + i);
    __syncwarp();

    int groups = cols / 32;
    int perBlock = groupsPerBlock(type);
    int bytes = blockBytes(type);
    const int* inputs = (const int*)set.input[which];
    u64 base = (u64)(inputs ? inputs[group] : group + set.shift[which]) * inputStride / 32;
    const u8* line0 = mine + offset;
    float acc[R];
    #pragma unroll
    for(int r = 0; r < R; r++) acc[r] = 0.0f;
    for(int u = lane; u < groups; u += 32){
        u64 at = base + u;
        const int4* q = (const int4*)(packed + at * 32);
        int4 a = __ldg(q);
        int4 b = __ldg(q + 1);
        float scale = __ldg(scales + at);
        float total = (float)__ldg(sums + at);
        int4 c = make_int4(0, 0, 0, 0), e = c;
        float fineScale = 0.0f, fineTotal = 0.0f;
        if(fine){
            const int4* p = (const int4*)(fine + at * 32);
            c = __ldg(p);
            e = __ldg(p + 1);
            fineScale = __ldg(fineScales + at);
            fineTotal = (float)__ldg(fineSums + at);
        }
        const u8* cell = line0 + (u / perBlock) * bytes;
        int g = u % perBlock;
        #pragma unroll
        for(int r = 0; r < R; r++){
            if(r >= count) break;
            int w[8];
            float low, high, shift;
            unpackGroup(type, cell + (u64)r * rowBytes, g, w, low, high, shift, t);
            int lowSum = __dp4a(w[3], a.w, __dp4a(w[2], a.z, __dp4a(w[1], a.y, __dp4a(w[0], a.x, 0))));
            int highSum = __dp4a(w[7], b.w, __dp4a(w[6], b.z, __dp4a(w[5], b.y, __dp4a(w[4], b.x, 0))));
            float part = scale * (low * (float)lowSum + high * (float)highSum - shift * total);
            if(fine){
                int lowFine = __dp4a(w[3], c.w, __dp4a(w[2], c.z, __dp4a(w[1], c.y, __dp4a(w[0], c.x, 0))));
                int highFine = __dp4a(w[7], e.w, __dp4a(w[6], e.z, __dp4a(w[5], e.y, __dp4a(w[4], e.x, 0))));
                part += fineScale * (low * (float)lowFine + high * (float)highFine - shift * fineTotal);
            }
            acc[r] += part;
        }
    }
    #pragma unroll
    for(int r = 0; r < R; r++){
        for(int step = 16; step > 0; step >>= 1) acc[r] += __shfl_xor_sync(0xffffffff, acc[r], step);
    }
    float* output = (float*)set.output[which] + (u64)group * rows + first;
    #pragma unroll
    for(int r = 0; r < R; r++){
        if(lane == r && r < count) output[r] = acc[r];
    }
}

#define MANY_ROWS(NAME, R, T) extern "C" __global__ void NAME(StagedSet set, int inputStride, const i8* packed, const float* scales, const int* sums, const i8* fine, const float* fineScales, const int* fineSums, int rows, int cols, int stageBytes, GridTables t) { manyRows<R, T>(set, inputStride, packed, scales, sums, fine, fineScales, fineSums, rows, cols, stageBytes, t); }
MANY_ROWS(rowsStagedSet2, 2, -1)
MANY_ROWS(rowsStagedSet4, 4, -1)
MANY_ROWS(rowsStagedSet4t7, 4, 7)
MANY_ROWS(rowsStagedSet4t8, 4, 8)
MANY_ROWS(rowsStagedSet4t12, 4, 12)
MANY_ROWS(rowsStagedSet4t13, 4, 13)
MANY_ROWS(rowsStagedSet4t16, 4, 16)
MANY_ROWS(rowsStagedSet4t17, 4, 17)
MANY_ROWS(rowsStagedSet4t18, 4, 18)
MANY_ROWS(rowsStagedSet4t20, 4, 20)
MANY_ROWS(rowsStagedSet4t21, 4, 21)
MANY_ROWS(rowsStagedSet4t22, 4, 22)
MANY_ROWS(rowsStagedSet4t23, 4, 23)
MANY_ROWS(rowsStagedSet4t42, 4, 42)

template<int R, int T>
__device__ __forceinline__ void teamRows(const StagedSet& set, int inputStride, const i8* packed, const float* scales, const int* sums, const i8* fine, const float* fineScales, const int* fineSums, int rows, int cols, int stageBytes, const GridTables& t) {
    extern __shared__ __align__(16) u8 stage[];
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int first = (blockIdx.x * 8 + warp) * R;
    int which = set.count - 1 - (int)(blockIdx.y / set.per), seat = blockIdx.y % set.per;
    if(first >= rows || which < 0) return;
    int4 crew = __ldg((const int4*)set.team[which] + seat);
    if(crew.x < 0) return;
    int size = crew.y < 0
        ? 1
        : crew.z < 0
            ? 2
            : crew.w < 0
                ? 3
                : 4;
    int pair[4] = {crew.x, crew.y, crew.z, crew.w};
    int slot = ((const int*)set.slot[which])[crew.x];
    int type = T >= 0 ? T : set.type[which];
    u64 rowBytes = set.rowBytes[which];
    int count = min(R, rows - first);
    const u8* line = (const u8*)set.weights[which] + set.expertBytes[which] * slot + rowBytes * first;
    u64 start = (u64)line & ~15ull;
    int offset = (int)((u64)line - start);
    int span = (offset + (int)rowBytes * count + 15) >> 4;
    u8* mine = stage + warp * stageBytes;
    const uint4* from = (const uint4*)start;
    uint4* into = (uint4*)mine;
    for(int i = lane; i < span; i += 32) into[i] = __ldg(from + i);
    __syncwarp();

    int groups = cols / 32;
    int perBlock = groupsPerBlock(type);
    int bytes = blockBytes(type);
    const int* inputs = (const int*)set.input[which];
    u64 base[4];
    #pragma unroll
    for(int m = 0; m < 4; m++) base[m] = m < size ? (u64)(inputs ? inputs[pair[m]] : pair[m] + set.shift[which]) * inputStride / 32 : 0;
    const u8* line0 = mine + offset;
    float acc[4][R];
    #pragma unroll
    for(int m = 0; m < 4; m++){
        #pragma unroll
        for(int r = 0; r < R; r++) acc[m][r] = 0.0f;
    }
    for(int u = lane; u < groups; u += 32){
        const u8* cell = line0 + (u / perBlock) * bytes;
        int g = u % perBlock;
        int w[R][8];
        float low[R], high[R], shift[R];
        #pragma unroll
        for(int r = 0; r < R; r++){
            if(r >= count) break;
            unpackGroup(type, cell + (u64)r * rowBytes, g, w[r], low[r], high[r], shift[r], t);
        }
        #pragma unroll
        for(int m = 0; m < 4; m++){
            if(m >= size) break;
            u64 at = base[m] + u;
            const int4* q = (const int4*)(packed + at * 32);
            int4 a = __ldg(q);
            int4 b = __ldg(q + 1);
            float scale = __ldg(scales + at);
            float total = (float)__ldg(sums + at);
            int4 c = make_int4(0, 0, 0, 0), e = c;
            float fineScale = 0.0f, fineTotal = 0.0f;
            if(fine){
                const int4* p = (const int4*)(fine + at * 32);
                c = __ldg(p);
                e = __ldg(p + 1);
                fineScale = __ldg(fineScales + at);
                fineTotal = (float)__ldg(fineSums + at);
            }
            #pragma unroll
            for(int r = 0; r < R; r++){
                if(r >= count) break;
                int lowSum = __dp4a(w[r][3], a.w, __dp4a(w[r][2], a.z, __dp4a(w[r][1], a.y, __dp4a(w[r][0], a.x, 0))));
                int highSum = __dp4a(w[r][7], b.w, __dp4a(w[r][6], b.z, __dp4a(w[r][5], b.y, __dp4a(w[r][4], b.x, 0))));
                float part = scale * (low[r] * (float)lowSum + high[r] * (float)highSum - shift[r] * total);
                if(fine){
                    int lowFine = __dp4a(w[r][3], c.w, __dp4a(w[r][2], c.z, __dp4a(w[r][1], c.y, __dp4a(w[r][0], c.x, 0))));
                    int highFine = __dp4a(w[r][7], e.w, __dp4a(w[r][6], e.z, __dp4a(w[r][5], e.y, __dp4a(w[r][4], e.x, 0))));
                    part += fineScale * (low[r] * (float)lowFine + high[r] * (float)highFine - shift[r] * fineTotal);
                }
                acc[m][r] += part;
            }
        }
    }
    #pragma unroll
    for(int m = 0; m < 4; m++){
        if(m >= size) break;
        #pragma unroll
        for(int r = 0; r < R; r++){
            for(int step = 16; step > 0; step >>= 1) acc[m][r] += __shfl_xor_sync(0xffffffff, acc[m][r], step);
        }
        float* output = (float*)set.output[which] + (u64)pair[m] * rows + first;
        #pragma unroll
        for(int r = 0; r < R; r++){
            if(lane == r && r < count) output[r] = acc[m][r];
        }
    }
}

#define TEAM_ROWS(NAME, R, T, B) extern "C" __global__ void __launch_bounds__(256, B) NAME(StagedSet set, int inputStride, const i8* packed, const float* scales, const int* sums, const i8* fine, const float* fineScales, const int* fineSums, int rows, int cols, int stageBytes, GridTables t) { teamRows<R, T>(set, inputStride, packed, scales, sums, fine, fineScales, fineSums, rows, cols, stageBytes, t); }
TEAM_ROWS(rowsTeam1, 1, -1, 1)
TEAM_ROWS(rowsTeam2, 2, -1, 1)
TEAM_ROWS(rowsTeam4, 4, -1, 1)
TEAM_ROWS(rowsTeamDown1, 1, -1, 1)
TEAM_ROWS(rowsTeamDown2, 2, -1, 1)
TEAM_ROWS(rowsTeamDown4, 4, -1, 1)
TEAM_ROWS(rowsTeam4t7, 4, 7, 1)
TEAM_ROWS(rowsTeam4t8, 4, 8, 1)
TEAM_ROWS(rowsTeam4t12, 4, 12, 1)
TEAM_ROWS(rowsTeam4t13, 4, 13, 1)
TEAM_ROWS(rowsTeam4t16, 4, 16, 1)
TEAM_ROWS(rowsTeam4t17, 4, 17, 1)
TEAM_ROWS(rowsTeam4t18, 4, 18, 1)
TEAM_ROWS(rowsTeam4t20, 4, 20, 1)
TEAM_ROWS(rowsTeam4t21, 4, 21, 1)
TEAM_ROWS(rowsTeam4t22, 4, 22, 1)
TEAM_ROWS(rowsTeam4t23, 4, 23, 1)
TEAM_ROWS(rowsTeam4t42, 4, 42, 1)
TEAM_ROWS(rowsTeamDown4t7, 4, 7, 3)
TEAM_ROWS(rowsTeamDown4t8, 4, 8, 3)
TEAM_ROWS(rowsTeamDown4t12, 4, 12, 3)
TEAM_ROWS(rowsTeamDown4t13, 4, 13, 3)
TEAM_ROWS(rowsTeamDown4t16, 4, 16, 3)
TEAM_ROWS(rowsTeamDown4t17, 4, 17, 3)
TEAM_ROWS(rowsTeamDown4t18, 4, 18, 3)
TEAM_ROWS(rowsTeamDown4t20, 4, 20, 3)
TEAM_ROWS(rowsTeamDown4t21, 4, 21, 3)
TEAM_ROWS(rowsTeamDown4t22, 4, 22, 3)
TEAM_ROWS(rowsTeamDown4t23, 4, 23, 3)
TEAM_ROWS(rowsTeamDown4t42, 4, 42, 3)

extern "C" __global__ void rowsMulti(RowSet set, const i8* packed, const float* scales, const int* sums, const i8* fine, const float* fineScales, const int* fineSums, int cols, int stageBytes, GridTables t) {
    extern __shared__ __align__(16) u8 stage[];
    int block = blockIdx.x;
    int which = 0;
    while(which + 1 < set.count && block >= set.start[which + 1]) which++;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = (block - set.start[which]) * 8 + warp;
    if(row >= set.rows[which]) return;
    int type = set.type[which];
    const u8* line = (const u8*)set.weights[which] + set.stride[which] * row;
    u64 start = (u64)line & ~15ull;
    int offset = (int)((u64)line - start);
    int span = (offset + (int)set.stride[which] + 15) >> 4;
    u8* mine = stage + warp * stageBytes;
    const uint4* from = (const uint4*)start;
    uint4* into = (uint4*)mine;
    for(int i = lane; i < span; i += 32) into[i] = __ldg(from + i);
    __syncwarp();

    int groups = cols / 32;
    int perBlock = groupsPerBlock(type);
    int bytes = blockBytes(type);
    const u8* line0 = mine + offset;
    float acc = 0.0f;
    for(int u = lane; u < groups; u += 32){
        int w[8];
        float low, high, shift;
        unpackGroup(type, line0 + (u / perBlock) * bytes, u % perBlock, w, low, high, shift, t);
        const int4* q = (const int4*)(packed + (u64)u * 32);
        int4 a = __ldg(q);
        int4 b = __ldg(q + 1);
        int lowSum = __dp4a(w[3], a.w, __dp4a(w[2], a.z, __dp4a(w[1], a.y, __dp4a(w[0], a.x, 0))));
        int highSum = __dp4a(w[7], b.w, __dp4a(w[6], b.z, __dp4a(w[5], b.y, __dp4a(w[4], b.x, 0))));
        float part = __ldg(scales + u) * (low * (float)lowSum + high * (float)highSum - shift * (float)__ldg(sums + u));
        if(fine){
            const int4* r = (const int4*)(fine + (u64)u * 32);
            int4 c = __ldg(r);
            int4 e = __ldg(r + 1);
            int lowFine = __dp4a(w[3], c.w, __dp4a(w[2], c.z, __dp4a(w[1], c.y, __dp4a(w[0], c.x, 0))));
            int highFine = __dp4a(w[7], e.w, __dp4a(w[6], e.z, __dp4a(w[5], e.y, __dp4a(w[4], e.x, 0))));
            part += __ldg(fineScales + u) * (low * (float)lowFine + high * (float)highFine - shift * (float)__ldg(fineSums + u));
        }
        acc += part;
    }
    for(int step = 16; step > 0; step >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, step);
    if(lane == 0) ((float*)set.output[which])[row] = acc;
}

template<int N>
__device__ __forceinline__ void batchRows(int type, const u8* weights, u64 rowBytes, const i8* packed, const float* scales, const int* sums, const i8* fine, const float* fineScales, const int* fineSums, float* output, int rows, int cols, int count, int stageBytes, const GridTables& t) {
    extern __shared__ __align__(16) u8 stage[];
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp;
    if(row >= rows) return;
    const u8* line = weights + rowBytes * row;
    u64 start = (u64)line & ~15ull;
    int offset = (int)((u64)line - start);
    int span = (offset + (int)rowBytes + 15) >> 4;
    u8* mine = stage + warp * stageBytes;
    const uint4* from = (const uint4*)start;
    uint4* into = (uint4*)mine;
    for(int i = lane; i < span; i += 32) into[i] = __ldg(from + i);
    __syncwarp();

    int groups = cols / 32;
    int perBlock = groupsPerBlock(type);
    int bytes = blockBytes(type);
    const u8* line0 = mine + offset;
    float acc[N];
    #pragma unroll
    for(int k = 0; k < N; k++) acc[k] = 0.0f;
    for(int u = lane; u < groups; u += 32){
        int w[8];
        float low, high, shift;
        unpackGroup(type, line0 + (u / perBlock) * bytes, u % perBlock, w, low, high, shift, t);
        #pragma unroll
        for(int k = 0; k < N; k++){
            if(k >= count) continue;
            u64 at = (u64)k * groups + u;
            const int4* q = (const int4*)(packed + at * 32);
            int4 a = __ldg(q);
            int4 b = __ldg(q + 1);
            int lowSum = __dp4a(w[3], a.w, __dp4a(w[2], a.z, __dp4a(w[1], a.y, __dp4a(w[0], a.x, 0))));
            int highSum = __dp4a(w[7], b.w, __dp4a(w[6], b.z, __dp4a(w[5], b.y, __dp4a(w[4], b.x, 0))));
            float part = __ldg(scales + at) * (low * (float)lowSum + high * (float)highSum - shift * (float)__ldg(sums + at));
            if(fine){
                const int4* r = (const int4*)(fine + at * 32);
                int4 c = __ldg(r);
                int4 e = __ldg(r + 1);
                int lowFine = __dp4a(w[3], c.w, __dp4a(w[2], c.z, __dp4a(w[1], c.y, __dp4a(w[0], c.x, 0))));
                int highFine = __dp4a(w[7], e.w, __dp4a(w[6], e.z, __dp4a(w[5], e.y, __dp4a(w[4], e.x, 0))));
                part += __ldg(fineScales + at) * (low * (float)lowFine + high * (float)highFine - shift * (float)__ldg(fineSums + at));
            }
            acc[k] += part;
        }
    }
    #pragma unroll
    for(int k = 0; k < N; k++){
        if(k >= count) continue;
        float v = acc[k];
        for(int step = 16; step > 0; step >>= 1) v += __shfl_xor_sync(0xffffffff, v, step);
        if(lane == 0) output[(u64)k * rows + row] = v;
    }
}

extern "C" __global__ void rowsBatch(int type, const u8* weights, u64 rowBytes, const i8* packed, const float* scales, const int* sums, const i8* fine, const float* fineScales, const int* fineSums, float* output, int rows, int cols, int count, int stageBytes, GridTables t) {
    batchRows<16>(type, weights, rowBytes, packed, scales, sums, fine, fineScales, fineSums, output, rows, cols, count, stageBytes, t);
}

extern "C" __global__ void rowsBatch2(int type, const u8* weights, u64 rowBytes, const i8* packed, const float* scales, const int* sums, const i8* fine, const float* fineScales, const int* fineSums, float* output, int rows, int cols, int count, int stageBytes, GridTables t) {
    batchRows<2>(type, weights, rowBytes, packed, scales, sums, fine, fineScales, fineSums, output, rows, cols, count, stageBytes, t);
}

extern "C" __global__ void rowsBatch3(int type, const u8* weights, u64 rowBytes, const i8* packed, const float* scales, const int* sums, const i8* fine, const float* fineScales, const int* fineSums, float* output, int rows, int cols, int count, int stageBytes, GridTables t) {
    batchRows<3>(type, weights, rowBytes, packed, scales, sums, fine, fineScales, fineSums, output, rows, cols, count, stageBytes, t);
}

extern "C" __global__ void rowsBatch4(int type, const u8* weights, u64 rowBytes, const i8* packed, const float* scales, const int* sums, const i8* fine, const float* fineScales, const int* fineSums, float* output, int rows, int cols, int count, int stageBytes, GridTables t) {
    batchRows<4>(type, weights, rowBytes, packed, scales, sums, fine, fineScales, fineSums, output, rows, cols, count, stageBytes, t);
}

template<int TYPE, int N, bool DIRECT>
__device__ __forceinline__ void typedRows(const u8* weights, u64 rowBytes, const i8* packed, const float* scales, const int* sums, const i8* fine, const float* fineScales, const int* fineSums, float* output, int rows, int cols, int stageBytes, const GridTables& t) {
    extern __shared__ __align__(16) u8 stage[];
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp;
    if(row >= rows) return;
    const u8* line = weights + rowBytes * row;
    const u8* line0 = line;
    if(!DIRECT){
        u64 start = (u64)line & ~15ull;
        int offset = (int)((u64)line - start);
        int span = (offset + (int)rowBytes + 15) >> 4;
        u8* mine = stage + warp * stageBytes;
        const uint4* from = (const uint4*)start;
        uint4* into = (uint4*)mine;
        for(int i = lane; i < span; i += 32) into[i] = __ldg(from + i);
        __syncwarp();
        line0 = mine + offset;
    }

    int groups = cols / 32;
    const int perBlock = groupsPerBlock(TYPE);
    const int bytes = blockBytes(TYPE);
    float acc[N];
    #pragma unroll
    for(int k = 0; k < N; k++) acc[k] = 0.0f;
    for(int u = lane; u < groups; u += 32){
        int w[8];
        float low, high, shift;
        unpackGroup(TYPE, line0 + (u / perBlock) * bytes, u % perBlock, w, low, high, shift, t);
        #pragma unroll
        for(int k = 0; k < N; k++){
            u64 at = (u64)k * groups + u;
            const int4* q = (const int4*)(packed + at * 32);
            int4 a = __ldg(q);
            int4 b = __ldg(q + 1);
            int lowSum = __dp4a(w[3], a.w, __dp4a(w[2], a.z, __dp4a(w[1], a.y, __dp4a(w[0], a.x, 0))));
            int highSum = __dp4a(w[7], b.w, __dp4a(w[6], b.z, __dp4a(w[5], b.y, __dp4a(w[4], b.x, 0))));
            float part = __ldg(scales + at) * (low * (float)lowSum + high * (float)highSum - shift * (float)__ldg(sums + at));
            if(fine){
                const int4* r = (const int4*)(fine + at * 32);
                int4 c = __ldg(r);
                int4 e = __ldg(r + 1);
                int lowFine = __dp4a(w[3], c.w, __dp4a(w[2], c.z, __dp4a(w[1], c.y, __dp4a(w[0], c.x, 0))));
                int highFine = __dp4a(w[7], e.w, __dp4a(w[6], e.z, __dp4a(w[5], e.y, __dp4a(w[4], e.x, 0))));
                part += __ldg(fineScales + at) * (low * (float)lowFine + high * (float)highFine - shift * (float)__ldg(fineSums + at));
            }
            acc[k] += part;
        }
    }
    #pragma unroll
    for(int k = 0; k < N; k++){
        float v = acc[k];
        for(int step = 16; step > 0; step >>= 1) v += __shfl_xor_sync(0xffffffff, v, step);
        if(lane == 0) output[(u64)k * rows + row] = v;
    }
}

#define TYPED_ROWS(T, N) extern "C" __global__ void rowsType##T##x##N(const u8* weights, u64 rowBytes, const i8* packed, const float* scales, const int* sums, const i8* fine, const float* fineScales, const int* fineSums, float* output, int rows, int cols, int stageBytes, GridTables t) { typedRows<T, N, false>(weights, rowBytes, packed, scales, sums, fine, fineScales, fineSums, output, rows, cols, stageBytes, t); } extern "C" __global__ void rowsDirect##T##x##N(const u8* weights, u64 rowBytes, const i8* packed, const float* scales, const int* sums, const i8* fine, const float* fineScales, const int* fineSums, float* output, int rows, int cols, int stageBytes, GridTables t) { typedRows<T, N, true>(weights, rowBytes, packed, scales, sums, fine, fineScales, fineSums, output, rows, cols, stageBytes, t); }
#define TYPED_COUNTS(T) TYPED_ROWS(T, 1) TYPED_ROWS(T, 2) TYPED_ROWS(T, 3) TYPED_ROWS(T, 4) TYPED_ROWS(T, 5) TYPED_ROWS(T, 6) TYPED_ROWS(T, 7) TYPED_ROWS(T, 8) TYPED_ROWS(T, 9)
TYPED_COUNTS(11)
TYPED_COUNTS(12)
TYPED_COUNTS(13)
TYPED_COUNTS(14)
TYPED_COUNTS(20)
TYPED_COUNTS(21)
TYPED_COUNTS(23)
TYPED_COUNTS(42)
TYPED_COUNTS(7)
TYPED_COUNTS(8)

template<int TYPE, int N>
__device__ __forceinline__ void pairRows(const u8* weights, u64 rowBytes, const i8* packed, const float* scales, const int* sums, const i8* fine, const float* fineScales, const int* fineSums, float* output, int rows, int cols, const GridTables& t) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * 16 + warp * 2;
    if(row >= rows) return;
    bool second = row + 1 < rows;
    const u8* first = weights + rowBytes * row;
    const u8* next = second
        ? first + rowBytes
        : first;
    int groups = cols / 32;
    const int perBlock = groupsPerBlock(TYPE);
    const int bytes = blockBytes(TYPE);
    float acc[N], other[N];
    #pragma unroll
    for(int k = 0; k < N; k++){
        acc[k] = 0.0f;
        other[k] = 0.0f;
    }
    for(int u = lane; u < groups; u += 32){
        int w[8], v[8];
        float low, high, shift, low2, high2, shift2;
        unpackGroup(TYPE, first + (u / perBlock) * bytes, u % perBlock, w, low, high, shift, t);
        unpackGroup(TYPE, next + (u / perBlock) * bytes, u % perBlock, v, low2, high2, shift2, t);
        #pragma unroll
        for(int k = 0; k < N; k++){
            u64 at = (u64)k * groups + u;
            const int4* q = (const int4*)(packed + at * 32);
            int4 a = __ldg(q);
            int4 b = __ldg(q + 1);
            float scale = __ldg(scales + at);
            float total = shift != 0.0f || shift2 != 0.0f ? (float)__ldg(sums + at) : 0.0f;
            int lowSum = __dp4a(w[3], a.w, __dp4a(w[2], a.z, __dp4a(w[1], a.y, __dp4a(w[0], a.x, 0))));
            int highSum = __dp4a(w[7], b.w, __dp4a(w[6], b.z, __dp4a(w[5], b.y, __dp4a(w[4], b.x, 0))));
            int lowSum2 = __dp4a(v[3], a.w, __dp4a(v[2], a.z, __dp4a(v[1], a.y, __dp4a(v[0], a.x, 0))));
            int highSum2 = __dp4a(v[7], b.w, __dp4a(v[6], b.z, __dp4a(v[5], b.y, __dp4a(v[4], b.x, 0))));
            float part = scale * (low * (float)lowSum + high * (float)highSum - shift * total);
            float part2 = scale * (low2 * (float)lowSum2 + high2 * (float)highSum2 - shift2 * total);
            if(fine){
                const int4* r = (const int4*)(fine + at * 32);
                int4 c = __ldg(r);
                int4 e = __ldg(r + 1);
                float fineScale = __ldg(fineScales + at);
                float fineTotal = shift != 0.0f || shift2 != 0.0f ? (float)__ldg(fineSums + at) : 0.0f;
                int lowFine = __dp4a(w[3], c.w, __dp4a(w[2], c.z, __dp4a(w[1], c.y, __dp4a(w[0], c.x, 0))));
                int highFine = __dp4a(w[7], e.w, __dp4a(w[6], e.z, __dp4a(w[5], e.y, __dp4a(w[4], e.x, 0))));
                int lowFine2 = __dp4a(v[3], c.w, __dp4a(v[2], c.z, __dp4a(v[1], c.y, __dp4a(v[0], c.x, 0))));
                int highFine2 = __dp4a(v[7], e.w, __dp4a(v[6], e.z, __dp4a(v[5], e.y, __dp4a(v[4], e.x, 0))));
                part += fineScale * (low * (float)lowFine + high * (float)highFine - shift * fineTotal);
                part2 += fineScale * (low2 * (float)lowFine2 + high2 * (float)highFine2 - shift2 * fineTotal);
            }
            acc[k] += part;
            other[k] += part2;
        }
    }
    #pragma unroll
    for(int k = 0; k < N; k++){
        float x = acc[k];
        float y = other[k];
        for(int step = 16; step > 0; step >>= 1){
            x += __shfl_xor_sync(0xffffffff, x, step);
            y += __shfl_xor_sync(0xffffffff, y, step);
        }
        if(lane == 0){
            output[(u64)k * rows + row] = x;
            if(second) output[(u64)k * rows + row + 1] = y;
        }
    }
}

#define PAIR_ROWS(T, N) extern "C" __global__ void rowsPair##T##x##N(const u8* weights, u64 rowBytes, const i8* packed, const float* scales, const int* sums, const i8* fine, const float* fineScales, const int* fineSums, float* output, int rows, int cols, int stageBytes, GridTables t) { pairRows<T, N>(weights, rowBytes, packed, scales, sums, fine, fineScales, fineSums, output, rows, cols, t); }
#define PAIR_COUNTS(T) PAIR_ROWS(T, 1) PAIR_ROWS(T, 2) PAIR_ROWS(T, 3) PAIR_ROWS(T, 4) PAIR_ROWS(T, 5) PAIR_ROWS(T, 6) PAIR_ROWS(T, 7) PAIR_ROWS(T, 8) PAIR_ROWS(T, 9)
PAIR_COUNTS(11)
PAIR_COUNTS(12)
PAIR_COUNTS(13)
PAIR_COUNTS(14)
PAIR_COUNTS(20)
PAIR_COUNTS(21)
PAIR_COUNTS(23)
PAIR_COUNTS(42)
PAIR_COUNTS(7)
PAIR_COUNTS(8)

__device__ __forceinline__ u32 pairLoad(const u8* p) {
    return (u32)*(const u16*)p | ((u32)*(const u16*)(p + 2) << 16);
}

// las dos piezas de 4 pesos que un hilo pone en el fragmento de los nucleos tensoriales: k 4i..4i+3 y 16+4i..16+4i+3
// para los tipos comunes se lee solo eso, los demas pasan por el desempaque entero
template<int TYPE>
__device__ __forceinline__ void groupPiece(const u8* block, int g, int i, int& wa, int& wb, float& low, float& high, float& offset, const GridTables& t) {
    offset = 0.0f;
    if(TYPE == 12 || TYPE == 13){
        float d = half(word(block));
        float dmin = half(word(block + 2));
        int sc, m;
        scaleMin(g, block + 4, sc, m);
        const u8* qs = (TYPE == 12 ? block + 16 : block + 48) + 32 * (g / 2);
        int shift = (g & 1) * 4;
        bool even = ((u64)block & 3) == 0;
        u32 a = even ? *(const u32*)(qs + 4 * i) : load32(qs + 4 * i);
        u32 b = even ? *(const u32*)(qs + 16 + 4 * i) : load32(qs + 16 + 4 * i);
        u32 va = (a >> shift) & 0x0F0F0F0Fu, vb = (b >> shift) & 0x0F0F0F0Fu;
        if(TYPE == 13){
            const u8* qh = block + 16;
            va |= ((load32(qh + 4 * i) >> g) & 0x01010101u) << 4;
            vb |= ((load32(qh + 16 + 4 * i) >> g) & 0x01010101u) << 4;
        }
        wa = (int)va;
        wb = (int)vb;
        low = d * sc;
        high = low;
        offset = dmin * m;
        return;
    }
    if(TYPE == 14){
        const u8* ql = block + 64 * (g / 4);
        const u8* qh = block + 128 + 32 * (g / 4);
        const i8* sc = (const i8*)(block + 192) + 8 * (g / 4);
        float d = half(word(block + 208));
        int k = g % 4;
        const u8* lowBytes = ql + (k & 1) * 32;
        int lowShift = (k >> 1) * 4;
        u32 la = (pairLoad(lowBytes + 4 * i) >> lowShift) & 0x0F0F0F0Fu;
        u32 lb = (pairLoad(lowBytes + 16 + 4 * i) >> lowShift) & 0x0F0F0F0Fu;
        u32 ha = ((pairLoad(qh + 4 * i) >> (2 * k)) & 0x03030303u) << 4;
        u32 hb = ((pairLoad(qh + 16 + 4 * i) >> (2 * k)) & 0x03030303u) << 4;
        wa = (int)__vsub4(la | ha, 0x20202020u);
        wb = (int)__vsub4(lb | hb, 0x20202020u);
        low = d * (float)sc[2 * k];
        high = d * (float)sc[2 * k + 1];
        return;
    }
    if(TYPE == 8){
        float d = half(word(block));
        wa = (int)pairLoad(block + 2 + 4 * i);
        wb = (int)pairLoad(block + 18 + 4 * i);
        low = d;
        high = d;
        return;
    }
    int w[8];
    unpackGroup(TYPE, block, g, w, low, high, offset, t);
    wa = w[i];
    wb = w[i + 4];
}

// pocas filas de tokens (2 a 16) con los nucleos tensoriales: un bloque hace 16 filas de pesos y sus 8 warps se reparten
// los grupos de 32; cada fragmento de tokens se lee una vez para las 16 filas, en vez de una vez por fila como en los
// kernels de dp4a que con 4 u 8 tokens leian mas tokens que pesos
template<int TYPE, int TILES>
__device__ __forceinline__ void fewTensor(const u8* weights, u64 rowBytes, const i8* packed, const float* scales, const int* sums, const i8* fine, const float* fineScales, const int* fineSums, float* output, int rows, int cols, int count, const GridTables& t) {
    // en cada paso un warp desempaca 16 filas por 4 grupos con lecturas seguidas (cada hilo un grupo entero, como los kernels
    // de dp4a) y deja los pesos en memoria compartida; de ahi cada hilo toma su pedazo para los nucleos tensoriales.
    // leyendo los pedazos directo cada instruccion tocaba 8 filas distintas y el kernel no pasaba de 350 GB/s
    __shared__ float parts[8][32][TILES * 4];
    __shared__ __align__(16) i8 decoded[8][16][144];
    __shared__ float scaled[8][16][4][3];
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, quad = lane & 3;
    int warps = blockDim.x >> 5;
    int base = blockIdx.x * 16;
    int r0 = base + gid, r1 = base + gid + 8;
    bool has0 = r0 < rows, has1 = r1 < rows;
    int groups = cols / 32;
    const int perBlock = groupsPerBlock(TYPE);
    const int bytes = blockBytes(TYPE);
    float acc[TILES][4];
    #pragma unroll
    for(int nt = 0; nt < TILES; nt++){
        #pragma unroll
        for(int e = 0; e < 4; e++) acc[nt][e] = 0.0f;
    }
    int steps = (groups + 3) / 4;
    for(int s = warp; s < steps; s += warps){
        int first = s * 4;
        #pragma unroll
        for(int it = 0; it < 2; it++){
            int pair = it * 32 + lane;
            int r = pair >> 2, gl = pair & 3;
            int u = first + gl, row = base + r;
            int w[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            float low = 0.0f, high = 0.0f, shift = 0.0f;
            if(u < groups && row < rows) unpackGroup(TYPE, weights + rowBytes * (u64)row + (u64)(u / perBlock) * bytes, u % perBlock, w, low, high, shift, t);
            int4* into = (int4*)(decoded[warp][r] + gl * 32);
            into[0] = make_int4(w[0], w[1], w[2], w[3]);
            into[1] = make_int4(w[4], w[5], w[6], w[7]);
            scaled[warp][r][gl][0] = low;
            scaled[warp][r][gl][1] = high;
            scaled[warp][r][gl][2] = shift;
        }
        __syncwarp();
        int stepGroups = min(4, groups - first);
        for(int gl = 0; gl < stepGroups; gl++){
        int u = first + gl;
        int a0 = *(const int*)(decoded[warp][gid] + gl * 32 + quad * 4);
        int b0 = *(const int*)(decoded[warp][gid] + gl * 32 + 16 + quad * 4);
        int a1 = *(const int*)(decoded[warp][gid + 8] + gl * 32 + quad * 4);
        int b1 = *(const int*)(decoded[warp][gid + 8] + gl * 32 + 16 + quad * 4);
        float low0 = scaled[warp][gid][gl][0], high0 = scaled[warp][gid][gl][1], shift0 = scaled[warp][gid][gl][2];
        float low1 = scaled[warp][gid + 8][gl][0], high1 = scaled[warp][gid + 8][gl][1], shift1 = scaled[warp][gid + 8][gl][2];
        #pragma unroll
        for(int nt = 0; nt < TILES; nt++){
            int token = nt * 8 + gid;
            int x0 = 0, x1 = 0, f0 = 0, f1 = 0;
            if(token < count){
                const int* q = (const int*)(packed + ((u64)token * groups + u) * 32);
                x0 = __ldg(q + quad);
                x1 = __ldg(q + 4 + quad);
                if(fine){
                    const int* r = (const int*)(fine + ((u64)token * groups + u) * 32);
                    f0 = __ldg(r + quad);
                    f1 = __ldg(r + 4 + quad);
                }
            }
            int ta = nt * 8 + quad * 2, tb = ta + 1;
            float sa = 0.0f, sb = 0.0f, za = 0.0f, zb = 0.0f;
            if(ta < count){
                u64 at = (u64)ta * groups + u;
                sa = __ldg(scales + at);
                za = (float)__ldg(sums + at);
            }
            if(tb < count){
                u64 at = (u64)tb * groups + u;
                sb = __ldg(scales + at);
                zb = (float)__ldg(sums + at);
            }
            int c[4], e[4];
            tensorStep(c, a0, a1, x0);
            tensorStep(e, b0, b1, x1);
            acc[nt][0] += sa * (low0 * (float)c[0] + high0 * (float)e[0] - shift0 * za);
            acc[nt][1] += sb * (low0 * (float)c[1] + high0 * (float)e[1] - shift0 * zb);
            acc[nt][2] += sa * (low1 * (float)c[2] + high1 * (float)e[2] - shift1 * za);
            acc[nt][3] += sb * (low1 * (float)c[3] + high1 * (float)e[3] - shift1 * zb);
            if(fine){
                float fa = 0.0f, fb = 0.0f, ya = 0.0f, yb = 0.0f;
                if(ta < count){
                    u64 at = (u64)ta * groups + u;
                    fa = __ldg(fineScales + at);
                    ya = (float)__ldg(fineSums + at);
                }
                if(tb < count){
                    u64 at = (u64)tb * groups + u;
                    fb = __ldg(fineScales + at);
                    yb = (float)__ldg(fineSums + at);
                }
                tensorStep(c, a0, a1, f0);
                tensorStep(e, b0, b1, f1);
                acc[nt][0] += fa * (low0 * (float)c[0] + high0 * (float)e[0] - shift0 * ya);
                acc[nt][1] += fb * (low0 * (float)c[1] + high0 * (float)e[1] - shift0 * yb);
                acc[nt][2] += fa * (low1 * (float)c[2] + high1 * (float)e[2] - shift1 * ya);
                acc[nt][3] += fb * (low1 * (float)c[3] + high1 * (float)e[3] - shift1 * yb);
            }
        }
        }
        __syncwarp();
    }
    #pragma unroll
    for(int nt = 0; nt < TILES; nt++){
        #pragma unroll
        for(int e = 0; e < 4; e++) parts[warp][lane][nt * 4 + e] = acc[nt][e];
    }
    __syncthreads();
    if(warp != 0) return;
    #pragma unroll
    for(int nt = 0; nt < TILES; nt++){
        float total[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for(int w = 0; w < warps; w++){
            #pragma unroll
            for(int e = 0; e < 4; e++) total[e] += parts[w][lane][nt * 4 + e];
        }
        int ta = nt * 8 + quad * 2, tb = ta + 1;
        if(has0 && ta < count) output[(u64)ta * rows + r0] = total[0];
        if(has0 && tb < count) output[(u64)tb * rows + r0] = total[1];
        if(has1 && ta < count) output[(u64)ta * rows + r1] = total[2];
        if(has1 && tb < count) output[(u64)tb * rows + r1] = total[3];
    }
}

#define FEW_TENSOR(T, TILES) extern "C" __global__ void __launch_bounds__(256) rowsMma##T##t##TILES(const u8* weights, u64 rowBytes, const i8* packed, const float* scales, const int* sums, const i8* fine, const float* fineScales, const int* fineSums, float* output, int rows, int cols, int count, GridTables t) { fewTensor<T, TILES>(weights, rowBytes, packed, scales, sums, fine, fineScales, fineSums, output, rows, cols, count, t); }
#define FEW_TENSORS(T) FEW_TENSOR(T, 1) FEW_TENSOR(T, 2)
FEW_TENSORS(11)
FEW_TENSORS(12)
FEW_TENSORS(13)
FEW_TENSORS(14)
FEW_TENSORS(20)
FEW_TENSORS(21)
FEW_TENSORS(23)
FEW_TENSORS(42)
FEW_TENSORS(7)
FEW_TENSORS(8)

__device__ __forceinline__ void fineGroup(float v, int group, i8* packed, float* scales, int* sums, i8* fine, float* fineScales, int* fineSums);

extern "C" __global__ void quantizeFine(const float* input, int count, i8* packed, float* scales, int* sums, i8* fine, float* fineScales, int* fineSums) {
    int group = blockIdx.x * blockDim.y + threadIdx.y;
    if(group * 32 >= count) return;
    fineGroup(input[group * 32 + threadIdx.x], group, packed, scales, sums, fine, fineScales, fineSums);
}

extern "C" __global__ void quantizeCopy(const float* input, float* copy, int count, i8* packed, float* scales, int* sums, i8* fine, float* fineScales, int* fineSums) {
    int group = blockIdx.x * blockDim.y + threadIdx.y;
    if(group * 32 >= count) return;
    float v = input[group * 32 + threadIdx.x];
    copy[group * 32 + threadIdx.x] = v;
    fineGroup(v, group, packed, scales, sums, fine, fineScales, fineSums);
}

__device__ __forceinline__ void packRow(const float* x, int cols, i8* packed, float* scales, int* sums) {
    int lane = threadIdx.x;
    for(int g = threadIdx.y; g < cols / 32; g += blockDim.y){
        float v = x[g * 32 + lane];
        float peak = fabsf(v);
        for(int step = 16; step > 0; step >>= 1) peak = fmaxf(peak, __shfl_xor_sync(0xffffffff, peak, step));
        float d = peak / 127.0f;
        int q = d > 0.0f ? __float2int_rn(v / d) : 0;
        packed[g * 32 + lane] = (i8)q;
        int total = q;
        for(int step = 16; step > 0; step >>= 1) total += __shfl_xor_sync(0xffffffff, total, step);
        if(lane == 0){
            scales[g] = d;
            sums[g] = total;
        }
    }
    __syncthreads();
}

#define HELP_TABLES 3232

__device__ __forceinline__ GridTables sharedTables(const GridTables& t, u8* room) {
    int tid = threadIdx.y * 32 + threadIdx.x;
    u32* into = (u32*)room;
    for(int i = tid; i < 256; i += 256) into[i] = t.iq3xxs[i];
    for(int i = tid; i < 512; i += 256) into[256 + i] = t.iq3s[i];
    for(int i = tid; i < 32; i += 256) into[768 + i] = ((const u32*)t.signs)[i];
    for(int i = tid; i < 4; i += 256) into[800 + i] = ((const u32*)t.iq4nl)[i];
    GridTables s = t;
    s.iq3xxs = into;
    s.iq3s = into + 256;
    s.signs = (const u8*)(into + 768);
    s.iq4nl = (const i8*)(into + 800);
    return s;
}

__device__ __forceinline__ const u8* stageRow(const u8* line, u64 rowBytes, u8* mine) {
    u64 start = (u64)line & ~15ull;
    int offset = (int)((u64)line - start);
    int span = (offset + (int)rowBytes + 15) >> 4;
    const uint4* from = (const uint4*)start;
    uint4* into = (uint4*)mine;
    for(int i = threadIdx.x; i < span; i += 32) into[i] = __ldg(from + i);
    __syncwarp();
    return mine + offset;
}

extern "C" __global__ void helperGateUp(int gateType, int upType, const u8* gateWeights, const u8* upWeights, u64 gateRow, u64 upRow, u64 gateExpert, u64 upExpert, const int* experts, const float* input, const i8* packedIn, const float* scalesIn, const int* sumsIn, float* hidden, int rows, int cols, int stage, GridTables t) {
    extern __shared__ __align__(16) u8 room[];
    GridTables s = sharedTables(t, room);
    u8* stages = room + HELP_TABLES;
    int slot = blockIdx.y;
    const float* x = input + (u64)slot * cols;
    const i8* packed = packedIn + (u64)slot * cols;
    const float* scales = scalesIn + (u64)slot * (cols / 32);
    const int* sums = sumsIn + (u64)slot * (cols / 32);
    __syncthreads();
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if(row >= rows) return;
    int expert = experts[slot];
    u8* mine = stages + threadIdx.y * 2 * stage;
    const u8* gateLine = stageRow(gateWeights + gateExpert * expert + gateRow * row, gateRow, mine);
    const u8* upLine = stageRow(upWeights + upExpert * expert + upRow * row, upRow, mine + stage);
    float g = rowDotQ8(gateType, gateLine, cols, x, packed, scales, sums, s);
    float u = rowDotQ8(upType, upLine, cols, x, packed, scales, sums, s);
    if(threadIdx.x == 0) hidden[(u64)rows * slot + row] = g * gateOf(g) * u;
}

extern "C" __global__ void helperDown(int type, const u8* weights, u64 rowBytes, u64 expertBytes, const int* experts, const float* hidden, const i8* packedIn, const float* scalesIn, const int* sumsIn, float* output, int rows, int cols, int stage, GridTables t) {
    extern __shared__ __align__(16) u8 room[];
    GridTables s = sharedTables(t, room);
    u8* stages = room + HELP_TABLES;
    int slot = blockIdx.y;
    const float* x = hidden + (u64)slot * cols;
    const i8* packed = packedIn + (u64)slot * cols;
    const float* scales = scalesIn + (u64)slot * (cols / 32);
    const int* sums = sumsIn + (u64)slot * (cols / 32);
    __syncthreads();
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if(row >= rows) return;
    const u8* line = stageRow(weights + expertBytes * experts[slot] + rowBytes * row, rowBytes, stages + threadIdx.y * stage);
    float v = rowDotQ8(type, line, cols, x, packed, scales, sums, s);
    if(threadIdx.x == 0) output[(u64)rows * slot + row] = v;
}

extern "C" __global__ void addNormPack(float* R, const float* y, const float* weight, float* out, i8* packed, float* scales, int* sums, i8* fine, float* fineScales, int* fineSums) {
    __shared__ float scratch[32];
    int t = blockIdx.x, lane = threadIdx.x, warp = threadIdx.y, tid = warp * 32 + lane;
    float* r = R + (u64)t * HIDDEN;
    float v[(HIDDEN + 1023) / 1024], w[(HIDDEN + 1023) / 1024];
    #pragma unroll
    for(int i = 0; i < (HIDDEN + 1023) / 1024; i++){
        int d = i * 1024 + tid;
        v[i] = 0.0f;
        w[i] = 0.0f;
        if(d >= HIDDEN) continue;
        v[i] = y ? r[d] + y[(u64)t * HIDDEN + d] : r[d];
        w[i] = weight[d];
    }
    float sum = 0.0f;
    #pragma unroll
    for(int i = 0; i < (HIDDEN + 1023) / 1024; i++){
        int d = i * 1024 + tid;
        if(d >= HIDDEN) continue;
        if(y) r[d] = v[i];
        sum += v[i] * v[i];
    }
    for(int step = 16; step > 0; step >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, step);
    if(lane == 0) scratch[warp] = sum;
    __syncthreads();
    float total = 0.0f;
    for(int k = 0; k < 32; k++) total += scratch[k];
    float inv = rsqrtf(total / (float)HIDDEN + 1e-6f);
    #pragma unroll
    for(int i = 0; i < (HIDDEN + 1023) / 1024; i++){
        int d = i * 1024 + tid;
        if(d >= HIDDEN) continue;
        float x = v[i] * inv * w[i];
        out[(u64)t * HIDDEN + d] = x;
        int group = t * (HIDDEN / 32) + d / 32;
        if(fine){
            fineGroup(x, group, packed, scales, sums, fine, fineScales, fineSums);
            continue;
        }
        float peak = fabsf(x);
        for(int step = 16; step > 0; step >>= 1) peak = fmaxf(peak, __shfl_xor_sync(0xffffffff, peak, step));
        float scale = peak / 127.0f;
        int q = scale > 0.0f ? __float2int_rn(x / scale) : 0;
        packed[group * 32 + lane] = (i8)q;
        int count = q;
        for(int step = 16; step > 0; step >>= 1) count += __shfl_xor_sync(0xffffffff, count, step);
        if(lane == 0){
            scales[group] = scale;
            sums[group] = count;
        }
    }
}

extern "C" __global__ void swigluFine(const float* gate, const float* up, float* out, int count, i8* packed, float* scales, int* sums, i8* fine, float* fineScales, int* fineSums) {
    int group = blockIdx.x * blockDim.y + threadIdx.y;
    if(group * 32 >= count) return;
    int i = group * 32 + threadIdx.x;
    float g = gate[i];
    float v = g * gateOf(g) * up[i];
    out[i] = v;
    fineGroup(v, group, packed, scales, sums, fine, fineScales, fineSums);
}

__device__ __forceinline__ void fineGroup(float v, int group, i8* packed, float* scales, int* sums, i8* fine, float* fineScales, int* fineSums) {
    float peak = fabsf(v);
    for(int step = 16; step > 0; step >>= 1) peak = fmaxf(peak, __shfl_xor_sync(0xffffffff, peak, step));
    float d = peak / 127.0f;
    int q = d > 0.0f ? __float2int_rn(v / d) : 0;
    float rest = v - d * (float)q;
    float restPeak = fabsf(rest);
    for(int step = 16; step > 0; step >>= 1) restPeak = fmaxf(restPeak, __shfl_xor_sync(0xffffffff, restPeak, step));
    float e = restPeak / 127.0f;
    int r = e > 0.0f ? __float2int_rn(rest / e) : 0;
    packed[group * 32 + threadIdx.x] = (i8)q;
    fine[group * 32 + threadIdx.x] = (i8)r;
    int total = q, totalFine = r;
    for(int step = 16; step > 0; step >>= 1){
        total += __shfl_xor_sync(0xffffffff, total, step);
        totalFine += __shfl_xor_sync(0xffffffff, totalFine, step);
    }
    if(threadIdx.x == 0){
        scales[group] = d;
        sums[group] = total;
        fineScales[group] = e;
        fineSums[group] = totalFine;
    }
}

__device__ __forceinline__ u32 pairBf16(float a, float b) {
    u32 out = 0;
#if __CUDA_ARCH__ >= 800
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(out) : "f"(b), "f"(a));
#endif
    return out;
}

__device__ __forceinline__ void halfStep(float* c, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
#if __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

extern "C" __global__ void splitBf16(const float* in, int inStride, u32* high, u32* low, int rows, int cols) {
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    int pairs = cols / 2;
    if(i >= (u64)rows * pairs) return;
    int r = (int)(i / pairs), c = (int)(i % pairs);
    float2 v = *(const float2*)(in + (u64)r * inStride + 2 * c);
    u32 big = pairBf16(v.x, v.y);
    high[i] = big;
    low[i] = pairBf16(v.x - __uint_as_float(big << 16), v.y - __uint_as_float(big & 0xFFFF0000u));
}

extern "C" __global__ void gateTake(const float* both, float* low, float* injected, int rows, int extra, int count) {
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    int width = rows + extra;
    if(i >= (u64)width * count) return;
    u64 t = i / width;
    int r = (int)(i % width);
    float v = both[i];
    if(r >= rows){
        injected[t * extra + r - rows] = v;
        return;
    }
    float g = v * 0.25f;
    low[t * rows + r] = g * gateOf(g);
}

extern "C" __global__ void gateRows(float* data, int stride, int rows, int count) {
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= (u64)rows * count) return;
    float* at = data + (i / rows) * stride + i % rows;
    float v = *at * 0.25f;
    *at = v * gateOf(v);
}

__device__ __forceinline__ u32 pairHalf(float a, float b) {
    u32 out = 0;
#if __CUDA_ARCH__ >= 800
    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(out) : "f"(b), "f"(a));
#endif
    return out;
}

extern "C" __global__ void halfUnpack(int type, const u8* weights, u64 rowBytes, u32* out, int rows, int cols, GridTables t) {
    int groups = cols / 32;
    u64 job = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if(job >= (u64)rows * groups) return;
    int row = (int)(job / groups), g = (int)(job % groups);
    int perBlock = groupsPerBlock(type);
    int w[8];
    float low, high, shift;
    unpackGroup(type, weights + rowBytes * row + (u64)(g / perBlock) * blockBytes(type), g % perBlock, w, low, high, shift, t);
    uint4* into = (uint4*)(out + (u64)row * (cols / 2) + 16 * g);
    u32 pairs[16];
    #pragma unroll
    for(int i = 0; i < 8; i++){
        float scale = i < 4 ? low : high;
        float v0 = scale * (float)(i8)(w[i] & 255) - shift;
        float v1 = scale * (float)(i8)((w[i] >> 8) & 255) - shift;
        float v2 = scale * (float)(i8)((w[i] >> 16) & 255) - shift;
        float v3 = scale * (float)(i8)((w[i] >> 24) & 255) - shift;
        pairs[2 * i] = pairHalf(v0, v1);
        pairs[2 * i + 1] = pairHalf(v2, v3);
    }
    #pragma unroll
    for(int i = 0; i < 4; i++) into[i] = make_uint4(pairs[4 * i], pairs[4 * i + 1], pairs[4 * i + 2], pairs[4 * i + 3]);
}

extern "C" __global__ void bf16ToHalf(const u32* in, u32* out, u64 pairs) {
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= pairs) return;
    u32 v = in[i];
    out[i] = pairHalf(fminf(fmaxf(__uint_as_float(v << 16), -65504.0f), 65504.0f), fminf(fmaxf(__uint_as_float(v & 0xFFFF0000u), -65504.0f), 65504.0f));
}

extern "C" __global__ void halfInput(const float* in, u32* out, u64 pairs) {
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= pairs) return;
    float2 v = *(const float2*)(in + 2 * i);
    out[i] = pairHalf(fminf(fmaxf(v.x, -65504.0f), 65504.0f), fminf(fmaxf(v.y, -65504.0f), 65504.0f));
}

extern "C" __global__ void plainTensor(const u8* weights, u64 rowBytes, const float* input, int inputStride, float* output, int outputStride, int rows, int cols, int count, int splits, int mode) {
    __shared__ u32 wt[64][20];
    __shared__ u32 xh[64][20];
    __shared__ u32 xl[64][20];
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2, quad = lane & 3;
    int rowBase = blockIdx.x * 64;
    int tile = blockIdx.y / splits, split = blockIdx.y % splits;
    int first = tile * 64;
    int span = (cols / 32 + splits - 1) / splits * 32;
    int k1 = min(cols, (split + 1) * span);
    float acc[8][4];
    #pragma unroll
    for(int nt = 0; nt < 8; nt++){
        #pragma unroll
        for(int e = 0; e < 4; e++) acc[nt][e] = 0.0f;
    }

    for(int k0 = split * span; k0 < k1; k0 += 32){
        for(int i = tid; i < 64 * 16; i += blockDim.x){
            int r = i >> 4, pair = i & 15;
            int row = rowBase + r;
            wt[r][pair] = row < rows ? *(const u32*)(weights + rowBytes * row + 4 * (k0 / 2 + pair)) : 0u;
        }
        for(int i = tid; i < 64 * 16; i += blockDim.x){
            int t = i >> 4, pair = i & 15;
            int token = first + t;
            float2 v = token < count ? *(const float2*)(input + (u64)token * inputStride + k0 + 2 * pair) : make_float2(0.0f, 0.0f);
            u32 big = pairBf16(v.x, v.y);
            float bx = __uint_as_float(big << 16);
            float by = __uint_as_float(big & 0xFFFF0000u);
            xh[t][pair] = big;
            xl[t][pair] = pairBf16(v.x - bx, v.y - by);
        }
        __syncthreads();

        int r0 = warp * 16 + gid;
        #pragma unroll
        for(int ks = 0; ks < 16; ks += 8){
            u32 a0 = wt[r0][ks + quad];
            u32 a1 = wt[r0 + 8][ks + quad];
            u32 a2 = wt[r0][ks + quad + 4];
            u32 a3 = wt[r0 + 8][ks + quad + 4];
            #pragma unroll
            for(int nt = 0; nt < 8; nt++){
                int tok = nt * 8 + gid;
                halfStep(acc[nt], a0, a1, a2, a3, xh[tok][ks + quad], xh[tok][ks + quad + 4]);
                halfStep(acc[nt], a0, a1, a2, a3, xl[tok][ks + quad], xl[tok][ks + quad + 4]);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for(int nt = 0; nt < 8; nt++){
        #pragma unroll
        for(int e = 0; e < 4; e++){
            int ri = rowBase + warp * 16 + gid + (e >> 1) * 8;
            int token = first + nt * 8 + quad * 2 + (e & 1);
            if(ri >= rows || token >= count) continue;
            float* at = output + (u64)token * outputStride + ri;
            if(splits > 1){
                output[((u64)split * count + token) * outputStride + ri] = acc[nt][e];
                continue;
            }
            float v = acc[nt][e];
            if(mode == 1){
                v *= 0.25f;
                v = v * gateOf(v);
            }
            *at = v;
        }
    }
}

extern "C" __global__ void plainTile(const u8* weights, u64 rowBytes, const float* input, int inputStride, float* output, int outputStride, int rows, int cols, int count, int splits, int mode) {
    __shared__ __align__(16) float wt[32][68];
    __shared__ __align__(16) float xt[32][68];
    int tid = threadIdx.x;
    int rowBase = blockIdx.x * 64;
    int tile = blockIdx.y / splits, split = blockIdx.y % splits;
    int first = tile * 64;
    int span = (cols / 32 + splits - 1) / splits * 32;
    int k0 = split * span;
    int k1 = min(cols, k0 + span);
    int tx = tid & 15, ty = tid >> 4;
    float acc[4][4];
    #pragma unroll
    for(int i = 0; i < 4; i++){
        #pragma unroll
        for(int j = 0; j < 4; j++) acc[i][j] = 0.0f;
    }
    int loadRow = tid >> 2, loadPart = tid & 3;
    for(int k = k0; k < k1; k += 32){
        int wr = rowBase + loadRow;
        uint4 raw = wr < rows
            ? *(const uint4*)(weights + rowBytes * wr + (u64)(k + loadPart * 8) * 2)
            : make_uint4(0, 0, 0, 0);
        u32 words[4] = {raw.x, raw.y, raw.z, raw.w};
        #pragma unroll
        for(int q = 0; q < 4; q++){
            wt[loadPart * 8 + 2 * q][loadRow] = __int_as_float(words[q] << 16);
            wt[loadPart * 8 + 2 * q + 1][loadRow] = __int_as_float(words[q] & 0xFFFF0000u);
        }
        int xr = first + loadRow;
        float4 a = make_float4(0.0f, 0.0f, 0.0f, 0.0f), b = a;
        if(xr < count){
            const float4* src = (const float4*)(input + (u64)xr * inputStride + k + loadPart * 8);
            a = src[0];
            b = src[1];
        }
        xt[loadPart * 8][loadRow] = a.x;
        xt[loadPart * 8 + 1][loadRow] = a.y;
        xt[loadPart * 8 + 2][loadRow] = a.z;
        xt[loadPart * 8 + 3][loadRow] = a.w;
        xt[loadPart * 8 + 4][loadRow] = b.x;
        xt[loadPart * 8 + 5][loadRow] = b.y;
        xt[loadPart * 8 + 6][loadRow] = b.z;
        xt[loadPart * 8 + 7][loadRow] = b.w;
        __syncthreads();
        #pragma unroll 8
        for(int c = 0; c < 32; c++){
            float4 w = *(const float4*)&wt[c][tx * 4];
            float4 x = *(const float4*)&xt[c][ty * 4];
            acc[0][0] = fmaf(w.x, x.x, acc[0][0]); acc[0][1] = fmaf(w.x, x.y, acc[0][1]); acc[0][2] = fmaf(w.x, x.z, acc[0][2]); acc[0][3] = fmaf(w.x, x.w, acc[0][3]);
            acc[1][0] = fmaf(w.y, x.x, acc[1][0]); acc[1][1] = fmaf(w.y, x.y, acc[1][1]); acc[1][2] = fmaf(w.y, x.z, acc[1][2]); acc[1][3] = fmaf(w.y, x.w, acc[1][3]);
            acc[2][0] = fmaf(w.z, x.x, acc[2][0]); acc[2][1] = fmaf(w.z, x.y, acc[2][1]); acc[2][2] = fmaf(w.z, x.z, acc[2][2]); acc[2][3] = fmaf(w.z, x.w, acc[2][3]);
            acc[3][0] = fmaf(w.w, x.x, acc[3][0]); acc[3][1] = fmaf(w.w, x.y, acc[3][1]); acc[3][2] = fmaf(w.w, x.z, acc[3][2]); acc[3][3] = fmaf(w.w, x.w, acc[3][3]);
        }
        __syncthreads();
    }
    #pragma unroll
    for(int i = 0; i < 4; i++){
        int ri = rowBase + tx * 4 + i;
        #pragma unroll
        for(int j = 0; j < 4; j++){
            int token = first + ty * 4 + j;
            if(ri >= rows || token >= count) continue;
            if(splits > 1){
                output[((u64)split * count + token) * outputStride + ri] = acc[i][j];
                continue;
            }
            float v = acc[i][j];
            if(mode == 1){
                v *= 0.25f;
                v = v * gateOf(v);
            }
            output[(u64)token * outputStride + ri] = v;
        }
    }
}

extern "C" __global__ void mergeSplits(const float* partial, float* output, int outputStride, int rows, int count, int splits, int mode) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= rows * count) return;
    int t = i / rows, r = i % rows;
    float v = 0.0f;
    for(int s = 0; s < splits; s++) v += partial[((u64)s * count + t) * rows + r];
    if(mode == 1){
        v *= 0.25f;
        v = v * gateOf(v);
    }
    output[(u64)t * outputStride + r] = v;
}

extern "C" __global__ void settleRows(float* output, int outputStride, int rows, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= rows * count) return;
    float* at = output + (u64)(i / rows) * outputStride + i % rows;
    float v = *at * 0.25f;
    *at = v * gateOf(v);
}

extern "C" __global__ void clearRows(float* output, int outputStride, int rows, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= rows * count) return;
    output[(u64)(i / rows) * outputStride + i % rows] = 0.0f;
}

extern "C" __global__ void plainRows(int type, const u8* weights, u64 rowBytes, const float* input, int inputStride, float* output, int outputStride, int rows, int cols, int count, int split, int mode) {
    __shared__ float partial[8][8];
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * (8 / split) + warp / split;
    int part = warp % split;
    bool valid = row < rows;
    const uint4* line = (const uint4*)(weights + rowBytes * (valid ? row : 0));
    int chunks = cols / 8;

    if(count == 1){
        float sum = 0.0f;
        if(valid){
            #pragma unroll 4
            for(int c = part * 32 + lane; c < chunks; c += split * 32){
                uint4 raw = __ldg(line + c);
                const float4* xs = (const float4*)(input + 8 * c);
                float4 a = __ldg(xs);
                float4 b = __ldg(xs + 1);
                float w0 = type == 30 ? __int_as_float(raw.x << 16) : half((u16)(raw.x & 0xFFFF));
                float w1 = type == 30 ? __int_as_float(raw.x & 0xFFFF0000u) : half((u16)(raw.x >> 16));
                float w2 = type == 30 ? __int_as_float(raw.y << 16) : half((u16)(raw.y & 0xFFFF));
                float w3 = type == 30 ? __int_as_float(raw.y & 0xFFFF0000u) : half((u16)(raw.y >> 16));
                float w4 = type == 30 ? __int_as_float(raw.z << 16) : half((u16)(raw.z & 0xFFFF));
                float w5 = type == 30 ? __int_as_float(raw.z & 0xFFFF0000u) : half((u16)(raw.z >> 16));
                float w6 = type == 30 ? __int_as_float(raw.w << 16) : half((u16)(raw.w & 0xFFFF));
                float w7 = type == 30 ? __int_as_float(raw.w & 0xFFFF0000u) : half((u16)(raw.w >> 16));
                sum += w0 * a.x + w1 * a.y + w2 * a.z + w3 * a.w + w4 * b.x + w5 * b.y + w6 * b.z + w7 * b.w;
            }
        }
        for(int step = 16; step > 0; step >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, step);
        if(lane == 0) partial[warp][0] = sum;
        __syncthreads();
        if(!valid || part != 0 || lane != 0) return;
        float v = 0.0f;
        for(int s = 0; s < split; s++) v += partial[warp + s][0];
        if(mode == 1){
            v *= 0.25f;
            v = v * gateOf(v);
        }
        output[row] = v;
        return;
    }

    for(int first = 0; first < count; first += 8){
        float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        if(valid){
            for(int c = part * 32 + lane; c < chunks; c += split * 32){
                uint4 raw = line[c];
                u32 words[4] = {raw.x, raw.y, raw.z, raw.w};
                float w[8];
                #pragma unroll
                for(int k = 0; k < 4; k++){
                    w[2 * k] = type == 30
                        ? __int_as_float(words[k] << 16)
                        : half((u16)(words[k] & 0xFFFF));
                    w[2 * k + 1] = type == 30
                        ? __int_as_float(words[k] & 0xFFFF0000u)
                        : half((u16)(words[k] >> 16));
                }
                #pragma unroll
                for(int j = 0; j < 8; j++){
                    if(first + j >= count) continue;
                    const float4* x = (const float4*)(input + (u64)(first + j) * inputStride + 8 * c);
                    float4 a = x[0];
                    float4 b = x[1];
                    acc[j] += w[0] * a.x + w[1] * a.y + w[2] * a.z + w[3] * a.w + w[4] * b.x + w[5] * b.y + w[6] * b.z + w[7] * b.w;
                }
            }
        }

        #pragma unroll
        for(int j = 0; j < 8; j++){
            for(int step = 16; step > 0; step >>= 1) acc[j] += __shfl_xor_sync(0xffffffff, acc[j], step);
        }
        if(lane == 0){
            #pragma unroll
            for(int j = 0; j < 8; j++) partial[warp][j] = acc[j];
        }
        __syncthreads();
        if(valid && part == 0 && lane < 8 && first + lane < count){
            float v = 0.0f;
            for(int s = 0; s < split; s++) v += partial[warp + s][lane];
            if(mode == 1){
                v *= 0.25f;
                v = v * gateOf(v);
            }
            output[(u64)(first + lane) * outputStride + row] = v;
        }
        __syncthreads();
    }
}

template<int N>
__device__ __forceinline__ void plainFew(int type, const u8* weights, u64 rowBytes, const float* input, int inputStride, float* output, int outputStride, int rows, int cols, int split, int mode) {
    __shared__ float partial[8][N];
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * (8 / split) + warp / split;
    int part = warp % split;
    bool valid = row < rows;
    const uint4* line = (const uint4*)(weights + rowBytes * (valid ? row : 0));
    int chunks = cols / 8;
    float acc[N];
    #pragma unroll
    for(int j = 0; j < N; j++) acc[j] = 0.0f;
    if(valid){
        for(int c = part * 32 + lane; c < chunks; c += split * 32){
            uint4 raw = __ldg(line + c);
            u32 words[4] = {raw.x, raw.y, raw.z, raw.w};
            float w[8];
            #pragma unroll
            for(int k = 0; k < 4; k++){
                w[2 * k] = type == 30
                    ? __int_as_float(words[k] << 16)
                    : half((u16)(words[k] & 0xFFFF));
                w[2 * k + 1] = type == 30
                    ? __int_as_float(words[k] & 0xFFFF0000u)
                    : half((u16)(words[k] >> 16));
            }
            #pragma unroll
            for(int j = 0; j < N; j++){
                const float4* x = (const float4*)(input + (u64)j * inputStride + 8 * c);
                float4 a = __ldg(x);
                float4 b = __ldg(x + 1);
                acc[j] += w[0] * a.x + w[1] * a.y + w[2] * a.z + w[3] * a.w + w[4] * b.x + w[5] * b.y + w[6] * b.z + w[7] * b.w;
            }
        }
    }
    #pragma unroll
    for(int j = 0; j < N; j++){
        for(int step = 16; step > 0; step >>= 1) acc[j] += __shfl_xor_sync(0xffffffff, acc[j], step);
    }
    if(lane == 0){
        #pragma unroll
        for(int j = 0; j < N; j++) partial[warp][j] = acc[j];
    }
    __syncthreads();
    if(!valid || part != 0 || lane >= N) return;
    float v = 0.0f;
    for(int s = 0; s < split; s++) v += partial[warp + s][lane];
    if(mode == 1){
        v *= 0.25f;
        v = v * gateOf(v);
    }
    output[(u64)lane * outputStride + row] = v;
}

#define PLAIN_FEW(N) extern "C" __global__ void plainRows##N(int type, const u8* weights, u64 rowBytes, const float* input, int inputStride, float* output, int outputStride, int rows, int cols, int count, int split, int mode) { plainFew<N>(type, weights, rowBytes, input, inputStride, output, outputStride, rows, cols, split, mode); }
template<int N>
__device__ __forceinline__ void pairFew(const u8* first, const u8* second, u64 firstBytes, u64 secondBytes, int firstRows, int rows, const float* input, int inputStride, float* firstOut, int firstStride, float* secondOut, int secondStride, int cols, int gated) {
    __shared__ float partial[8][N];
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x;
    bool late = row >= firstRows;
    const uint4* line = (const uint4*)(late ? second + secondBytes * (row - firstRows) : first + firstBytes * row);
    int chunks = cols / 8;
    float acc[N];
    #pragma unroll
    for(int j = 0; j < N; j++) acc[j] = 0.0f;
    for(int c = warp * 32 + lane; c < chunks; c += 256){
        uint4 raw = __ldg(line + c);
        float w[8] = {__int_as_float(raw.x << 16), __int_as_float(raw.x & 0xFFFF0000u), __int_as_float(raw.y << 16), __int_as_float(raw.y & 0xFFFF0000u), __int_as_float(raw.z << 16), __int_as_float(raw.z & 0xFFFF0000u), __int_as_float(raw.w << 16), __int_as_float(raw.w & 0xFFFF0000u)};
        #pragma unroll
        for(int j = 0; j < N; j++){
            const float4* x = (const float4*)(input + (u64)j * inputStride + 8 * c);
            float4 a = __ldg(x);
            float4 b = __ldg(x + 1);
            acc[j] += w[0] * a.x + w[1] * a.y + w[2] * a.z + w[3] * a.w + w[4] * b.x + w[5] * b.y + w[6] * b.z + w[7] * b.w;
        }
    }
    #pragma unroll
    for(int j = 0; j < N; j++){
        for(int step = 16; step > 0; step >>= 1) acc[j] += __shfl_xor_sync(0xffffffff, acc[j], step);
    }
    if(lane == 0){
        #pragma unroll
        for(int j = 0; j < N; j++) partial[warp][j] = acc[j];
    }
    __syncthreads();
    if(row >= rows || threadIdx.x >= N) return;
    float v = 0.0f;
    for(int s = 0; s < 8; s++) v += partial[s][threadIdx.x];
    if(late){
        secondOut[(u64)threadIdx.x * secondStride + row - firstRows] = v;
        return;
    }
    if(gated){
        v *= 0.25f;
        v = v * gateOf(v);
    }
    firstOut[(u64)threadIdx.x * firstStride + row] = v;
}

#define PAIR_FEW(N) extern "C" __global__ void plainPair##N(const u8* first, const u8* second, u64 firstBytes, u64 secondBytes, int firstRows, int rows, const float* input, int inputStride, float* firstOut, int firstStride, float* secondOut, int secondStride, int cols, int gated) { pairFew<N>(first, second, firstBytes, secondBytes, firstRows, rows, input, inputStride, firstOut, firstStride, secondOut, secondStride, cols, gated); }
PAIR_FEW(1)
PAIR_FEW(2)
PAIR_FEW(3)
PAIR_FEW(4)
PAIR_FEW(5)
PAIR_FEW(6)
PAIR_FEW(7)
PAIR_FEW(8)
PAIR_FEW(9)

template<int N>
__device__ __forceinline__ void upMix(const u8* weights, u64 rowBytes, const float* lo, const float* xn, float* x) {
    __shared__ float low[N][HCR];
    for(int i = threadIdx.x; i < N * HCR; i += blockDim.x) low[i / HCR][i % HCR] = lo[i];
    __syncthreads();
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int d = blockIdx.x * 8 + warp;
    int c = lane >> 3, part = lane & 7;
    bool valid = d < HIDDEN;
    const uint4* line = (const uint4*)(weights + rowBytes * (u64)(c * HIDDEN + (valid ? d : 0)));
    float acc[N];
    #pragma unroll
    for(int j = 0; j < N; j++) acc[j] = 0.0f;
    if(valid){
        for(int k = part; k < HCR / 8; k += 8){
            uint4 raw = __ldg(line + k);
            float w[8] = {__int_as_float(raw.x << 16), __int_as_float(raw.x & 0xFFFF0000u), __int_as_float(raw.y << 16), __int_as_float(raw.y & 0xFFFF0000u), __int_as_float(raw.z << 16), __int_as_float(raw.z & 0xFFFF0000u), __int_as_float(raw.w << 16), __int_as_float(raw.w & 0xFFFF0000u)};
            #pragma unroll
            for(int j = 0; j < N; j++){
                const float* l = low[j] + 8 * k;
                acc[j] += w[0] * l[0] + w[1] * l[1] + w[2] * l[2] + w[3] * l[3] + w[4] * l[4] + w[5] * l[5] + w[6] * l[6] + w[7] * l[7];
            }
        }
    }
    #pragma unroll
    for(int j = 0; j < N; j++){
        float v = acc[j];
        for(int step = 4; step > 0; step >>= 1) v += __shfl_xor_sync(0xffffffff, v, step);
        v = part == 0 && valid ? xn[(u64)j * HCW + c * HIDDEN + d] * gateOf(v) : 0.0f;
        v += __shfl_xor_sync(0xffffffff, v, 8);
        v += __shfl_xor_sync(0xffffffff, v, 16);
        if(lane == 0 && valid) x[(u64)j * HIDDEN + d] = (1.0f / HC) * v;
    }
}

#define UP_MIX(N) extern "C" __global__ void hcUpMix##N(const u8* weights, u64 rowBytes, const float* lo, const float* xn, float* x) { upMix<N>(weights, rowBytes, lo, xn, x); }
UP_MIX(1)
UP_MIX(2)
UP_MIX(3)
UP_MIX(4)
UP_MIX(5)
UP_MIX(6)
UP_MIX(7)
UP_MIX(8)
UP_MIX(9)

PLAIN_FEW(2)
PLAIN_FEW(3)
PLAIN_FEW(4)
PLAIN_FEW(5)
PLAIN_FEW(6)
PLAIN_FEW(7)
PLAIN_FEW(8)
PLAIN_FEW(9)

extern "C" __global__ void copyWords(const u32* from, u32* into, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < count) into[i] = from[i];
}

extern "C" __global__ void placeRope(float* ropeCos, float* ropeSin, const int* places, int first, int listed, int shift, double theta, int highSpan, int wideSpan) {
    int cell = first + blockIdx.x, i = threadIdx.x;
    int t = cell - shift, h = t, w = t;
    if(blockIdx.x < listed){
        t = places[blockIdx.x * 3];
        h = places[blockIdx.x * 3 + 1];
        w = places[blockIdx.x * 3 + 2];
    }
    int value = i % 3 == 1 && i < 3 * highSpan
        ? h
        : i % 3 == 2 && i < 3 * wideSpan
            ? w
            : t;
    double angle = (double)value * pow(theta, -2.0 * i / 64.0);
    ropeCos[(u64)cell * 32 + i] = (float)cos(angle);
    ropeSin[(u64)cell * 32 + i] = (float)sin(angle);
}

extern "C" __global__ void planExperts(const int* ids, const int* slots, const int* warmSlots, int* assign, int* groupSlot, int* groupCount, int* warmSlot, int* warmCount, int* spillTotal, int share, float* weights, int tail) {
    int lane = threadIdx.x;
    int e = lane < KUSED ? ids[lane] : 0;
    int slot = lane < KUSED ? slots[e] : -1;
    bool kept = lane < KUSED && slot >= 0;
    bool skip = tail > 0 && lane < KUSED && lane >= tail && !kept;
    bool live = lane < KUSED && !skip;
    int near = live && slot < 0 && warmSlots ? warmSlots[e] : -1;
    if(tail > 0){
        float w = lane < KUSED ? weights[lane] : 0.0f;
        float total = live ? w : 0.0f;
        for(int step = 16; step > 0; step >>= 1) total += __shfl_xor_sync(0xffffffff, total, step);
        if(lane < KUSED) weights[lane] = live && total > 0.0f ? w / total : 0.0f;
    }
    unsigned int below = (1u << lane) - 1u;
    unsigned int candidate = __ballot_sync(0xffffffff, live && !kept && near >= 0);
    int missed = __popc(__ballot_sync(0xffffffff, live && !kept));
    int allowed = missed * share / 8;
    bool warm = ((candidate >> lane) & 1u) && __popc(candidate & below) < allowed;
    unsigned int resident = __ballot_sync(0xffffffff, kept);
    unsigned int pinned = __ballot_sync(0xffffffff, warm);
    unsigned int spilled = __ballot_sync(0xffffffff, live && !kept && !warm);
    int total = __popc(resident);
    int warmTotal = __popc(pinned);
    if(kept){
        int g = __popc(resident & below);
        assign[lane] = g;
        groupSlot[g] = slot;
    } else if(warm){
        int g = __popc(pinned & below);
        assign[lane] = KUSED + g;
        warmSlot[g] = near;
    } else if(live){
        assign[lane] = 2 * KUSED + __popc(spilled & below);
    } else if(lane < KUSED){
        assign[lane] = 0;
    }
    if(lane < KUSED){
        groupCount[lane] = lane < total ? 1 : 0;
        warmCount[lane] = lane < warmTotal ? 1 : 0;
    }
    if(lane == 0) *spillTotal = __popc(spilled);
}

// el borrador puede elegir entre mas expertos que el modelo: los arreglos por experto toman el mayor
#ifndef PLAN_EXPERTS
#define PLAN_EXPERTS EXPERTS
#endif
extern "C" __global__ void planBatch(const int* ids, const int* slots, const int* warmSlots, int tokens, int share, int* assign, int* groupSlot, int* groupInput, int* groupCount, int* warmSlot, int* warmCount, int* warmInput, int* spillTotal, float* weights, int tail, int* teams, int* warmTeams) {
    int lane = threadIdx.x;
    unsigned int below = (1u << lane) - 1u;
    int residentSoFar = 0, warmSoFar = 0, spillSoFar = 0;
    __shared__ int home[PLAN_EXPERTS];
    __shared__ int keptFill[128 * KUSED];
    __shared__ int warmFill[128 * KUSED];
    int keptSeats = 0, warmSeats = 0;
    if(teams){
        for(int i = lane; i < PLAN_EXPERTS; i += 32) home[i] = -1;
        for(int i = lane; i < 4 * KUSED * tokens; i += 32){
            teams[i] = -1;
            warmTeams[i] = -1;
        }
        __syncwarp();
    }
    __shared__ int side[PLAN_EXPERTS];
    for(int i = lane; i < PLAN_EXPERTS; i += 32) side[i] = -1;
    int misses = 0;
    for(int t = 0; t < tokens; t++){
        int e = lane < KUSED ? ids[t * KUSED + lane] : 0;
        bool kept = lane < KUSED && slots[e] >= 0;
        misses += __popc(__ballot_sync(0xffffffff, lane < KUSED && !kept && !(tail > 0 && lane >= tail)));
    }
    int wanted = misses > 1 ? (misses - 1) * 3 * share / 32 : 0;
    __syncwarp();
    for(int t = 0; t < tokens; t++){
        int r = t * KUSED + lane;
        int e = lane < KUSED ? ids[r] : 0;
        int slot = lane < KUSED ? slots[e] : -1;
        bool kept = lane < KUSED && slot >= 0;
        bool live = lane < KUSED && !(tail > 0 && lane >= tail && !kept);
        int near = live && slot < 0 && warmSlots ? warmSlots[e] : -1;
        if(tail > 0){
            float w = lane < KUSED ? weights[r] : 0.0f;
            float total = live ? w : 0.0f;
            for(int step = 16; step > 0; step >>= 1) total += __shfl_xor_sync(0xffffffff, total, step);
            if(lane < KUSED) weights[r] = live && total > 0.0f ? w / total : 0.0f;
        }
        bool lost = live && !kept;
        int known = lost ? side[e] : -1;
        unsigned int fresh = __ballot_sync(0xffffffff, lost && near >= 0 && known < 0);
        bool warm = known == 1 || (((fresh >> lane) & 1u) && __popc(fresh & below) < wanted - warmSoFar);
        if(lost) side[e] = warm ? 1 : 0;
        __syncwarp();
        unsigned int resident = __ballot_sync(0xffffffff, kept);
        unsigned int pinned = __ballot_sync(0xffffffff, warm);
        unsigned int spilled = __ballot_sync(0xffffffff, live && !kept && !warm);
        if(kept){
            int g = residentSoFar + __popc(resident & below);
            assign[r] = g;
            groupSlot[g] = slot;
            groupInput[g] = t;
        } else if(warm){
            int g = warmSoFar + __popc(pinned & below);
            assign[r] = KUSED * tokens + g;
            warmSlot[g] = near;
            warmInput[g] = t;
        } else if(live){
            assign[r] = 2 * KUSED * tokens + spillSoFar + __popc(spilled & below);
        } else if(lane < KUSED){
            assign[r] = 0;
        }
        if(teams){
            bool joins = kept || warm;
            int g = kept
                ? residentSoFar + __popc(resident & below)
                : warmSoFar + __popc(pinned & below);
            int* fill = warm ? warmFill : keptFill;
            int seat = joins ? home[e] : -1;
            bool fresh = joins && (seat < 0 || fill[seat] == 4);
            unsigned int newKept = __ballot_sync(0xffffffff, fresh && kept);
            unsigned int newWarm = __ballot_sync(0xffffffff, fresh && warm);
            if(fresh){
                seat = kept
                    ? keptSeats + __popc(newKept & below)
                    : warmSeats + __popc(newWarm & below);
                home[e] = seat;
                fill[seat] = 0;
            }
            if(joins){
                (warm ? warmTeams : teams)[4 * seat + fill[seat]] = g;
                fill[seat]++;
            }
            keptSeats += __popc(newKept);
            warmSeats += __popc(newWarm);
            __syncwarp();
        }
        residentSoFar += __popc(resident);
        warmSoFar += __popc(pinned);
        spillSoFar += __popc(spilled);
    }
    for(int g = lane; g < KUSED * tokens; g += 32){
        groupCount[g] = g < residentSoFar ? 1 : 0;
        warmCount[g] = g < warmSoFar ? 1 : 0;
    }
    if(lane == 0) *spillTotal = spillSoFar;
}

extern "C" __global__ void rowsRouted(int type, const u8* weights, u64 rowBytes, u64 expertBytes, const int* groupSlot, const int* groupCount, const int* groupInput, const i8* packed, const float* scales, const int* sums, const i8* fine, const float* fineScales, const int* fineSums, float* output, int rows, int cols, int stageBytes, GridTables t) {
    extern __shared__ __align__(16) u8 stage[];
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp;
    int group = blockIdx.y;
    if(row >= rows) return;
    if(groupCount[group] == 0) return;
    const u8* line = weights + expertBytes * groupSlot[group] + rowBytes * row;
    u64 start = (u64)line & ~15ull;
    int offset = (int)((u64)line - start);
    int span = (offset + (int)rowBytes + 15) >> 4;
    u8* mine = stage + warp * stageBytes;
    const uint4* from = (const uint4*)start;
    uint4* into = (uint4*)mine;
    for(int i = lane; i < span; i += 32) into[i] = __ldg(from + i);
    __syncwarp();

    int groups = cols / 32;
    int perBlock = groupsPerBlock(type);
    int bytes = blockBytes(type);
    u64 base = (u64)groupInput[group] * groups;
    const u8* line0 = mine + offset;
    float acc = 0.0f;
    for(int u = lane; u < groups; u += 32){
        int w[8];
        float low, high, shift;
        unpackGroup(type, line0 + (u / perBlock) * bytes, u % perBlock, w, low, high, shift, t);
        u64 at = base + u;
        const int4* q = (const int4*)(packed + at * 32);
        int4 a = __ldg(q);
        int4 b = __ldg(q + 1);
        int lowSum = __dp4a(w[3], a.w, __dp4a(w[2], a.z, __dp4a(w[1], a.y, __dp4a(w[0], a.x, 0))));
        int highSum = __dp4a(w[7], b.w, __dp4a(w[6], b.z, __dp4a(w[5], b.y, __dp4a(w[4], b.x, 0))));
        float part = __ldg(scales + at) * (low * (float)lowSum + high * (float)highSum - shift * (float)__ldg(sums + at));
        if(fine){
            const int4* r = (const int4*)(fine + at * 32);
            int4 c = __ldg(r);
            int4 e = __ldg(r + 1);
            int lowFine = __dp4a(w[3], c.w, __dp4a(w[2], c.z, __dp4a(w[1], c.y, __dp4a(w[0], c.x, 0))));
            int highFine = __dp4a(w[7], e.w, __dp4a(w[6], e.z, __dp4a(w[5], e.y, __dp4a(w[4], e.x, 0))));
            part += __ldg(fineScales + at) * (low * (float)lowFine + high * (float)highFine - shift * (float)__ldg(fineSums + at));
        }
        acc += part;
    }
    for(int step = 16; step > 0; step >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, step);
    if(lane == 0) output[(u64)group * rows + row] = acc;
}

extern "C" __global__ void routePlan(const float* logits, int* ids, float* weights, int* idsOut, const int* slots, const int* warmSlots, int* assign, int* groupSlot, int* groupCount, int* warmSlot, int* warmCount, int* spillTotal, int share, int tail, volatile unsigned int* flag, const volatile unsigned int* ticket, int layer) {
    int lane = threadIdx.x;
    float v[EXPERTS / 32];
    #pragma unroll
    for(int j = 0; j < EXPERTS / 32; j++) v[j] = logits[lane + 32 * j];
    int e = 0;
    float mine = 0.0f, top = 0.0f;
    for(int k = 0; k < KUSED; k++){
        float best = -3.0e38f;
        int index = 1 << 30;
        #pragma unroll
        for(int j = 0; j < EXPERTS / 32; j++){
            if(v[j] > best){
                best = v[j];
                index = lane + 32 * j;
            }
        }
        for(int step = 16; step > 0; step >>= 1){
            float ov = __shfl_xor_sync(0xffffffff, best, step);
            int oi = __shfl_xor_sync(0xffffffff, index, step);
            if(ov > best || (ov == best && oi < index)){
                best = ov;
                index = oi;
            }
        }
        if(index >= EXPERTS) index = k;
        if(k == 0) top = best;
        if(lane == k){
            e = index;
            mine = best;
        }
        #pragma unroll
        for(int j = 0; j < EXPERTS / 32; j++){
            if(index == lane + 32 * j) v[j] = -3.0e38f;
        }
    }
    float prob = lane < KUSED ? __expf(mine - top) : 0.0f;
    float sum = prob;
    for(int step = 16; step > 0; step >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, step);
    float w = prob / sum;

    int slot = lane < KUSED ? slots[e] : -1;
    bool kept = lane < KUSED && slot >= 0;
    bool skip = tail > 0 && lane < KUSED && lane >= tail && !kept;
    bool live = lane < KUSED && !skip;
    int near = live && slot < 0 && warmSlots ? warmSlots[e] : -1;
    if(tail > 0){
        float total = live ? w : 0.0f;
        for(int step = 16; step > 0; step >>= 1) total += __shfl_xor_sync(0xffffffff, total, step);
        w = live && total > 0.0f ? w / total : 0.0f;
    }
    if(lane < KUSED){
        ids[lane] = e;
        weights[lane] = w;
        idsOut[lane] = e;
    }
    unsigned int below = (1u << lane) - 1u;
    unsigned int candidate = __ballot_sync(0xffffffff, live && !kept && near >= 0);
    int missed = __popc(__ballot_sync(0xffffffff, live && !kept));
    int allowed = missed * share / 8;
    bool warm = ((candidate >> lane) & 1u) && __popc(candidate & below) < allowed;
    unsigned int resident = __ballot_sync(0xffffffff, kept);
    unsigned int pinned = __ballot_sync(0xffffffff, warm);
    unsigned int spilled = __ballot_sync(0xffffffff, live && !kept && !warm);
    int total = __popc(resident);
    int warmTotal = __popc(pinned);
    if(kept){
        int g = __popc(resident & below);
        assign[lane] = g;
        groupSlot[g] = slot;
    } else if(warm){
        int g = __popc(pinned & below);
        assign[lane] = KUSED + g;
        warmSlot[g] = near;
    } else if(live){
        assign[lane] = 2 * KUSED + __popc(spilled & below);
    } else if(lane < KUSED){
        assign[lane] = 0;
    }
    if(lane < KUSED){
        groupCount[lane] = lane < total ? 1 : 0;
        warmCount[lane] = lane < warmTotal ? 1 : 0;
    }
    if(lane == 0) *spillTotal = __popc(spilled);
    if(!flag) return;
    __threadfence_system();
    __syncwarp();
    if(lane != 0) return;
    *flag = *ticket * 64u + (unsigned int)layer + 1u;
    __threadfence_system();
}

extern "C" __global__ void routePlanBatch(const float* logits, int* ids, float* weights, int* idsOut, const int* slots, const int* warmSlots, int tokens, int share, int* assign, int* groupSlot, int* groupInput, int* groupCount, int* warmSlot, int* warmCount, int* warmInput, int* spillTotal, int tail, volatile unsigned int* flag, const volatile unsigned int* ticket, int layer, int* teams, int* warmTeams) {
    __shared__ int pickedId[16 * KUSED];
    __shared__ float pickedWeight[16 * KUSED];
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if(warp < tokens){
        const float* row = logits + (u64)warp * EXPERTS;
        float v[EXPERTS / 32];
        #pragma unroll
        for(int j = 0; j < EXPERTS / 32; j++) v[j] = row[lane + 32 * j];
        int e = 0;
        float mine = 0.0f, top = 0.0f;
        for(int k = 0; k < KUSED; k++){
            float best = -3.0e38f;
            int index = 1 << 30;
            #pragma unroll
            for(int j = 0; j < EXPERTS / 32; j++){
                if(v[j] > best){
                    best = v[j];
                    index = lane + 32 * j;
                }
            }
            for(int step = 16; step > 0; step >>= 1){
                float ov = __shfl_xor_sync(0xffffffff, best, step);
                int oi = __shfl_xor_sync(0xffffffff, index, step);
                if(ov > best || (ov == best && oi < index)){
                    best = ov;
                    index = oi;
                }
            }
            if(index >= EXPERTS) index = k;
            if(k == 0) top = best;
            if(lane == k){
                e = index;
                mine = best;
            }
            #pragma unroll
            for(int j = 0; j < EXPERTS / 32; j++){
                if(index == lane + 32 * j) v[j] = -3.0e38f;
            }
        }
        float prob = lane < KUSED ? __expf(mine - top) : 0.0f;
        float sum = prob;
        for(int step = 16; step > 0; step >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, step);
        if(lane < KUSED){
            pickedId[warp * KUSED + lane] = e;
            pickedWeight[warp * KUSED + lane] = prob / sum;
        }
    }
    __syncthreads();
    if(warp != 0) return;
    unsigned int below = (1u << lane) - 1u;
    int residentSoFar = 0, warmSoFar = 0, spillSoFar = 0;
    __shared__ int home[EXPERTS];
    __shared__ int keptFill[16 * KUSED];
    __shared__ int warmFill[16 * KUSED];
    int keptSeats = 0, warmSeats = 0;
    if(teams){
        for(int i = lane; i < EXPERTS; i += 32) home[i] = -1;
        for(int i = lane; i < 4 * KUSED * tokens; i += 32){
            teams[i] = -1;
            warmTeams[i] = -1;
        }
        __syncwarp();
    }
    __shared__ int side[EXPERTS];
    for(int i = lane; i < EXPERTS; i += 32) side[i] = -1;
    int misses = 0;
    for(int t = 0; t < tokens; t++){
        int e = lane < KUSED ? pickedId[t * KUSED + lane] : 0;
        bool kept = lane < KUSED && slots[e] >= 0;
        misses += __popc(__ballot_sync(0xffffffff, lane < KUSED && !kept && !(tail > 0 && lane >= tail)));
    }
    int wanted = misses > 1 ? (misses - 1) * 3 * share / 32 : 0;
    __syncwarp();
    for(int t = 0; t < tokens; t++){
        int r = t * KUSED + lane;
        int e = lane < KUSED ? pickedId[r] : 0;
        float w = lane < KUSED ? pickedWeight[r] : 0.0f;
        int slot = lane < KUSED ? slots[e] : -1;
        bool kept = lane < KUSED && slot >= 0;
        bool live = lane < KUSED && !(tail > 0 && lane >= tail && !kept);
        int near = live && slot < 0 && warmSlots ? warmSlots[e] : -1;
        if(tail > 0){
            float total = live ? w : 0.0f;
            for(int step = 16; step > 0; step >>= 1) total += __shfl_xor_sync(0xffffffff, total, step);
            w = live && total > 0.0f ? w / total : 0.0f;
        }
        if(lane < KUSED){
            ids[r] = e;
            weights[r] = w;
            idsOut[r] = e;
        }
        bool lost = live && !kept;
        int known = lost ? side[e] : -1;
        unsigned int fresh = __ballot_sync(0xffffffff, lost && near >= 0 && known < 0);
        bool warm = known == 1 || (((fresh >> lane) & 1u) && __popc(fresh & below) < wanted - warmSoFar);
        if(lost) side[e] = warm ? 1 : 0;
        __syncwarp();
        unsigned int resident = __ballot_sync(0xffffffff, kept);
        unsigned int pinned = __ballot_sync(0xffffffff, warm);
        unsigned int spilled = __ballot_sync(0xffffffff, live && !kept && !warm);
        if(kept){
            int g = residentSoFar + __popc(resident & below);
            assign[r] = g;
            groupSlot[g] = slot;
            groupInput[g] = t;
        } else if(warm){
            int g = warmSoFar + __popc(pinned & below);
            assign[r] = KUSED * tokens + g;
            warmSlot[g] = near;
            warmInput[g] = t;
        } else if(live){
            assign[r] = 2 * KUSED * tokens + spillSoFar + __popc(spilled & below);
        } else if(lane < KUSED){
            assign[r] = 0;
        }
        if(teams){
            bool joins = kept || warm;
            int g = kept
                ? residentSoFar + __popc(resident & below)
                : warmSoFar + __popc(pinned & below);
            int* fill = warm ? warmFill : keptFill;
            int seat = joins ? home[e] : -1;
            bool fresh = joins && (seat < 0 || fill[seat] == 4);
            unsigned int newKept = __ballot_sync(0xffffffff, fresh && kept);
            unsigned int newWarm = __ballot_sync(0xffffffff, fresh && warm);
            if(fresh){
                seat = kept
                    ? keptSeats + __popc(newKept & below)
                    : warmSeats + __popc(newWarm & below);
                home[e] = seat;
                fill[seat] = 0;
            }
            if(joins){
                (warm ? warmTeams : teams)[4 * seat + fill[seat]] = g;
                fill[seat]++;
            }
            keptSeats += __popc(newKept);
            warmSeats += __popc(newWarm);
            __syncwarp();
        }
        residentSoFar += __popc(resident);
        warmSoFar += __popc(pinned);
        spillSoFar += __popc(spilled);
    }
    for(int g = lane; g < KUSED * tokens; g += 32){
        groupCount[g] = g < residentSoFar ? 1 : 0;
        warmCount[g] = g < warmSoFar ? 1 : 0;
    }
    if(lane == 0) *spillTotal = spillSoFar;
    if(!flag) return;
    __threadfence_system();
    __syncwarp();
    if(lane != 0) return;
    *flag = *ticket * 64u + (unsigned int)layer + 1u;
    __threadfence_system();
}

extern "C" __global__ void tellHost(volatile unsigned int* flag, const volatile unsigned int* ticket, int layer) {
    __threadfence_system();
    *flag = *ticket * 64u + (unsigned int)layer + 1u;
    __threadfence_system();
}

__device__ __forceinline__ unsigned long long globalNanos() {
#ifdef __HIP__
    return (unsigned long long)wall_clock64() * 10ull;
#else
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
#endif
}

extern "C" __global__ void stampTime(unsigned long long* at) {
    *(volatile unsigned long long*)at = globalNanos();
    __threadfence_system();
}

extern "C" __global__ void spinFor(unsigned long long nanos) {
    unsigned long long begin = globalNanos();
    while(globalNanos() - begin < nanos){
    }
}

extern "C" __global__ void awaitHost(const volatile unsigned int* flag, const volatile unsigned int* ticket, int layer, const int* spillTotal, volatile unsigned int* alarm, const volatile unsigned int* working) {
    if(*spillTotal == 0) return;
    unsigned int want = *ticket * 64u + (unsigned int)layer + 1u;
    unsigned long long begin = globalNanos();
    while(*flag < want){
        unsigned long long limit = *working >= want ? 120000000000ull : 1500000000ull;
        if(globalNanos() - begin > limit){
            if(*alarm == 0u) *alarm = want;
            __threadfence_system();
            return;
        }
#if __CUDA_ARCH__ >= 700
        __nanosleep(256);
#endif
    }
    __threadfence_system();
}

extern "C" __global__ void spreadRows(const float* embed, float* R, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= total) return;
    R[i] = embed[(i / HCW) * HIDDEN + i % HIDDEN];
}

extern "C" __global__ void addSpread(const float* streams, const float* shared, float* R, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= total) return;
    R[i] = streams[i] + shared[(i / HCW) * HIDDEN + i % HIDDEN];
}

extern "C" __global__ void streamNorm(const float* in, const float* weight, float* out) {
    __shared__ float scratch[32];
    const float* x = in + (u64)blockIdx.x * HIDDEN;
    float* y = out + (u64)blockIdx.x * HIDDEN;
    const float* w = weight + (blockIdx.x % HC) * HIDDEN;
    float sum = 0.0f;
    for(int d = threadIdx.x; d < HIDDEN; d += blockDim.x) sum += x[d] * x[d];
    float inv = rsqrtf(blockSum(sum, scratch) / (float)HIDDEN + 1e-6f);
    for(int d = threadIdx.x; d < HIDDEN; d += blockDim.x) y[d] = x[d] * inv * w[d];
}

extern "C" __global__ void streamNormHalf(const float* in, const float* weight, float* out, u32* half) {
    __shared__ float scratch[32];
    const float* x = in + (u64)blockIdx.x * HIDDEN;
    float* y = out + (u64)blockIdx.x * HIDDEN;
    const float* w = weight + (blockIdx.x % HC) * HIDDEN;
    float sum = 0.0f;
    for(int d = threadIdx.x; d < HIDDEN; d += blockDim.x) sum += x[d] * x[d];
    float inv = rsqrtf(blockSum(sum, scratch) / (float)HIDDEN + 1e-6f);
    u64 base = (u64)blockIdx.x * (HIDDEN / 2);
    for(int p = threadIdx.x; p < HIDDEN / 2; p += blockDim.x){
        float a = x[2 * p] * inv * w[2 * p], b = x[2 * p + 1] * inv * w[2 * p + 1];
        y[2 * p] = a;
        y[2 * p + 1] = b;
        half[base + p] = pairHalf(fminf(fmaxf(a, -65504.0f), 65504.0f), fminf(fmaxf(b, -65504.0f), 65504.0f));
    }
}

#if HIDDEN % 128 == 0
extern "C" __global__ void __launch_bounds__(256) writeNormHalf(float* R, const float* y, const float* inj, const float* weight, float* out, u32* half, int rows) {
    int row = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if(row >= rows) return;
    int t = row / HC, c = row % HC;
    float4* r = (float4*)(R + (u64)row * HIDDEN);
    const float4* add = (const float4*)(y + (u64)t * HIDDEN);
    float scale = y ? 2.0f * gateOf(inj[t * HC + c] * (1.0f / HC)) : 0.0f;
    float4 v[HIDDEN / 128];
    float sum = 0.0f;
    #pragma unroll
    for(int k = 0; k < HIDDEN / 128; k++){
        int p = lane + 32 * k;
        float4 a = r[p];
        if(y){
            float4 b = add[p];
            a.x += b.x * scale;
            a.y += b.y * scale;
            a.z += b.z * scale;
            a.w += b.w * scale;
            r[p] = a;
        }
        v[k] = a;
        sum += a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w;
    }
    for(int step = 16; step > 0; step >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, step);
    float inv = rsqrtf(sum / (float)HIDDEN + 1e-6f);
    const float4* w = (const float4*)(weight + c * HIDDEN);
    float4* o = (float4*)(out + (u64)row * HIDDEN);
    uint2* h = (uint2*)(half + (u64)row * (HIDDEN / 2));
    #pragma unroll
    for(int k = 0; k < HIDDEN / 128; k++){
        int p = lane + 32 * k;
        float4 ww = w[p];
        float a = v[k].x * inv * ww.x, b = v[k].y * inv * ww.y, e = v[k].z * inv * ww.z, f = v[k].w * inv * ww.w;
        if(out) o[p] = make_float4(a, b, e, f);
        h[p] = make_uint2(pairHalf(fminf(fmaxf(a, -65504.0f), 65504.0f), fminf(fmaxf(b, -65504.0f), 65504.0f)), pairHalf(fminf(fmaxf(e, -65504.0f), 65504.0f), fminf(fmaxf(f, -65504.0f), 65504.0f)));
    }
}
#endif

extern "C" __global__ void streamNormSplit(const float* in, const float* weight, float* out, u32* high, u32* low) {
    __shared__ float scratch[32];
    const float* x = in + (u64)blockIdx.x * HIDDEN;
    float* y = out + (u64)blockIdx.x * HIDDEN;
    const float* w = weight + (blockIdx.x % HC) * HIDDEN;
    float sum = 0.0f;
    for(int d = threadIdx.x; d < HIDDEN; d += blockDim.x) sum += x[d] * x[d];
    float inv = rsqrtf(blockSum(sum, scratch) / (float)HIDDEN + 1e-6f);
    u64 base = (u64)blockIdx.x * (HIDDEN / 2);
    for(int p = threadIdx.x; p < HIDDEN / 2; p += blockDim.x){
        float a = x[2 * p] * inv * w[2 * p], b = x[2 * p + 1] * inv * w[2 * p + 1];
        y[2 * p] = a;
        y[2 * p + 1] = b;
        u32 big = pairBf16(a, b);
        high[base + p] = big;
        low[base + p] = pairBf16(a - __uint_as_float(big << 16), b - __uint_as_float(big & 0xFFFF0000u));
    }
}

extern "C" __global__ void mixStreams(const float* xn, const float* gate, float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n * HIDDEN) return;
    u64 at = (u64)(i / HIDDEN) * HCW + i % HIDDEN;
    float sum = 0.0f;
    #pragma unroll
    for(int c = 0; c < HC; c++) sum += xn[at + c * HIDDEN] * gateOf(gate[at + c * HIDDEN]);
    x[i] = (1.0f / HC) * sum;
}

__device__ __forceinline__ void copyAsync(void* to, const void* from, bool valid) {
#if __CUDA_ARCH__ >= 800
    u32 at;
    asm("{.reg .u64 a; cvta.to.shared.u64 a, %1; cvt.u32.u64 %0, a;}" : "=r"(at) : "l"(to));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(at), "l"(from), "r"(valid ? 16 : 0));
#else
    *(uint4*)to = valid ? *(const uint4*)from : make_uint4(0, 0, 0, 0);
#endif
}

__device__ __forceinline__ void copyCommit() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;");
#endif
}

__device__ __forceinline__ void copyWaitOne() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 1;");
#endif
}

__device__ __forceinline__ void copyWaitAll() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;");
#endif
}

#if HC == 4 && HIDDEN % 64 == 0
struct MixSpace {
    u16 w[256][40];
    u16 v[64][40];
};

extern "C" __global__ void __launch_bounds__(256) hcMixUp(const u8* weights, u64 rowBytes, const float* lo, const u16* xn, float* x, int n, int rank) {
    __shared__ __align__(16) MixSpace s;
    int dBase = blockIdx.x * 64, tBase = blockIdx.y * 64;
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2, tig = lane & 3;
    int live = min(8, (n - tBase + 7) / 8);
    float acc[2][8][4];
    #pragma unroll
    for(int m = 0; m < 2; m++){
        #pragma unroll
        for(int nt = 0; nt < 8; nt++) acc[m][nt][0] = acc[m][nt][1] = acc[m][nt][2] = acc[m][nt][3] = 0.0f;
    }
    for(int k0 = 0; k0 < rank; k0 += 32){
        #pragma unroll
        for(int j = 0; j < 4; j++){
            int c = tid + 256 * j, rho = c >> 2, piece = c & 3;
            u64 row = (u64)(rho >> 6) * HIDDEN + dBase + (rho & 63);
            copyAsync(&s.w[rho][piece * 8], weights + row * rowBytes + (u64)k0 * 2 + piece * 16, true);
        }
        copyCommit();
        int token = tid >> 2, part = (tid & 3) * 8, t = tBase + token;
        uint4 pairs = make_uint4(0, 0, 0, 0);
        if(t < n){
            const float4* from = (const float4*)(lo + (u64)t * rank + k0 + part);
            float4 a = from[0], b = from[1];
            pairs = make_uint4(pairBf16(a.x, a.y), pairBf16(a.z, a.w), pairBf16(b.x, b.y), pairBf16(b.z, b.w));
        }
        *(uint4*)&s.v[token][part] = pairs;
        copyWaitAll();
        __syncthreads();
        #pragma unroll
        for(int ks = 0; ks < 2; ks++){
            int k = ks * 16 + tig * 2;
            u32 a[2][4];
            #pragma unroll
            for(int m = 0; m < 2; m++){
                int r0 = m * 128 + warp * 8 + gid, r1 = r0 + 64;
                a[m][0] = *(const u32*)&s.w[r0][k];
                a[m][1] = *(const u32*)&s.w[r1][k];
                a[m][2] = *(const u32*)&s.w[r0][k + 8];
                a[m][3] = *(const u32*)&s.w[r1][k + 8];
            }
            #pragma unroll
            for(int nt = 0; nt < 8; nt++){
                if(nt >= live) break;
                u32 b0 = *(const u32*)&s.v[nt * 8 + gid][k];
                u32 b1 = *(const u32*)&s.v[nt * 8 + gid][k + 8];
                halfStep(acc[0][nt], a[0][0], a[0][1], a[0][2], a[0][3], b0, b1);
                halfStep(acc[1][nt], a[1][0], a[1][1], a[1][2], a[1][3], b0, b1);
            }
        }
        __syncthreads();
    }
    int d = dBase + warp * 8 + gid;
    #pragma unroll
    for(int nt = 0; nt < 8; nt++){
        #pragma unroll
        for(int e = 0; e < 2; e++){
            int t = tBase + nt * 8 + tig * 2 + e;
            if(t >= n) continue;
            const u16* at = xn + (u64)t * HCW + d;
            float sum = 0.0f;
            sum += half(at[0]) * gateOf(acc[0][nt][e]);
            sum += half(at[HIDDEN]) * gateOf(acc[0][nt][2 + e]);
            sum += half(at[2 * HIDDEN]) * gateOf(acc[1][nt][e]);
            sum += half(at[3 * HIDDEN]) * gateOf(acc[1][nt][2 + e]);
            x[(u64)t * HIDDEN + d] = (1.0f / HC) * sum;
        }
    }
}
#endif

extern "C" __global__ void writeBack(float* R, const float* y, const float* inj, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n * HIDDEN) return;
    int t = i / HIDDEN;
    u64 at = (u64)t * HCW + i % HIDDEN;
    float v = y[i];
#if HC == 1
    R[at] += v;
#else
    #pragma unroll
    for(int c = 0; c < HC; c++) R[at + c * HIDDEN] += v * 2.0f * gateOf(inj[t * HC + c] * (1.0f / HC));
#endif
}

extern "C" __global__ void writeExperts(float* R, const float* shared, const float* scalar, const float* expertOut, const float* spill, int spillBase, const float* weights, const int* assign, const float* inj, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n * HIDDEN) return;
    int t = i / HIDDEN, d = i % HIDDEN;
    float v = gateOf(scalar[t]) * shared[i];
    #pragma unroll
    for(int k = 0; k < KUSED; k++){
        int a = assign[t * KUSED + k];
        float e = a >= spillBase
            ? spill[(u64)(a - spillBase) * HIDDEN + d]
            : expertOut[(u64)a * HIDDEN + d];
        v += weights[t * KUSED + k] * e;
    }
    u64 at = (u64)t * HCW + d;
#if HC == 1
    R[at] += v;
#else
    #pragma unroll
    for(int c = 0; c < HC; c++) R[at + c * HIDDEN] += v * 2.0f * gateOf(inj[t * HC + c] * (1.0f / HC));
#endif
}

// nvfp4 con escala por experto: up y down son lineales y su escala va al peso de cada ruta
extern "C" __global__ void routeScale(const int* ids, float* weights, const float* factor, int routes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < routes) weights[i] *= factor[ids[i]];
}

// la escala de gate va antes del silu: cada ruta escala su fila de la gpu (las de la cpu se escalan alla)
extern "C" __global__ void gateScaleRoutes(float* gate, const int* assign, const int* ids, const float* scales, int spillBase, int width) {
    int r = blockIdx.x;
    int a = assign[r];
    if(a < 0 || a >= spillBase) return;
    float s = scales[ids[r]];
    for(int i = threadIdx.x; i < width; i += blockDim.x) gate[(u64)a * width + i] *= s;
}

extern "C" __global__ void scaleRows(float* rows, const float* scale, int width) {
    float s = scale[blockIdx.x];
    for(int i = threadIdx.x; i < width; i += blockDim.x) rows[(u64)blockIdx.x * width + i] *= s;
}

extern "C" __global__ void writeMixed(float* R, const float* shared, const float* scalar, const long long* moe, const float* inj, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n * HIDDEN) return;
    int t = i / HIDDEN, d = i % HIDDEN;
    float v = gateOf(scalar[t]) * shared[i] + (EXACT_MIX ? (float)((double)moe[i] * 2.3283064365386963e-10) : ((const float*)moe)[i]);
    u64 at = (u64)t * HCW + d;
#if HC == 1
    R[at] += v;
#else
    #pragma unroll
    for(int c = 0; c < HC; c++) R[at + c * HIDDEN] += v * 2.0f * gateOf(inj[t * HC + c] * (1.0f / HC));
#endif
}

extern "C" __global__ void addRows(long long* moe, const float* rows, const int* tokenOf, const float* weightOf, int count) {
    int r = blockIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if(r >= count || d >= HIDDEN) return;
    float v = weightOf[r] * rows[(u64)r * HIDDEN + d];
    u64 at = (u64)tokenOf[r] * HIDDEN + d;
    if(EXACT_MIX) atomicAdd((unsigned long long*)moe + at, (unsigned long long)__float2ll_rn(v * 4294967296.0f));
    else atomicAdd((float*)moe + at, v);
}

extern "C" __global__ void clearWords(u32* into, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < count) into[i] = 0u;
}

extern "C" __global__ void swiglu(const float* gate, const float* up, float* out, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= total) return;
    float g = gate[i];
    out[i] = g * gateOf(g) * up[i];
}

extern "C" __global__ void pickTop(const float* logits, int count, unsigned int* out) {
    __shared__ float values[32];
    __shared__ int places[32];
    __shared__ float scratch[32];
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float best = -3.0e38f;
    int where = 0;
    for(int i = threadIdx.x; i < count; i += blockDim.x){
        float v = logits[i];
        if(v > best){
            best = v;
            where = i;
        }
    }
    for(int step = 16; step > 0; step >>= 1){
        float other = __shfl_xor_sync(0xffffffff, best, step);
        int place = __shfl_xor_sync(0xffffffff, where, step);
        if(other > best || (other == best && place < where)){
            best = other;
            where = place;
        }
    }
    if(lane == 0){
        values[warp] = best;
        places[warp] = where;
    }
    __syncthreads();
    if(warp == 0){
        best = lane < (int)(blockDim.x >> 5) ? values[lane] : -3.0e38f;
        where = lane < (int)(blockDim.x >> 5) ? places[lane] : 0;
        for(int step = 16; step > 0; step >>= 1){
            float other = __shfl_xor_sync(0xffffffff, best, step);
            int place = __shfl_xor_sync(0xffffffff, where, step);
            if(other > best || (other == best && place < where)){
                best = other;
                where = place;
            }
        }
        if(lane == 0){
            values[0] = best;
            places[0] = where;
        }
    }
    __syncthreads();
    float top = values[0];
    int chosen = places[0];
    float sum = 0.0f;
    for(int i = threadIdx.x; i < count; i += blockDim.x) sum += __expf(logits[i] - top);
    sum = blockSum(sum, scratch);
    if(threadIdx.x == 0){
        out[0] = (unsigned int)chosen;
        out[1] = __float_as_uint(1.0f / sum);
    }
}

extern "C" __global__ void routeTop(const float* logits, int* ids, float* weights) {
    __shared__ float bestValue[32];
    __shared__ int bestIndex[32];
    __shared__ float pickedValue[KUSED];
    __shared__ int pickedIndex[KUSED];
    int t = blockIdx.x, e = threadIdx.x, warp = e >> 5, lane = e & 31, warps = (int)(blockDim.x >> 5);
    float mine = logits[(u64)t * blockDim.x + e];
    bool taken = false;
    for(int k = 0; k < KUSED; k++){
        float v = taken ? -3.0e38f : mine;
        int index = e;
        for(int step = 16; step > 0; step >>= 1){
            float ov = __shfl_xor_sync(0xffffffff, v, step);
            int oi = __shfl_xor_sync(0xffffffff, index, step);
            if(ov > v || (ov == v && oi < index)){
                v = ov;
                index = oi;
            }
        }
        if(lane == 0){
            bestValue[warp] = v;
            bestIndex[warp] = index;
        }
        __syncthreads();
        if(e == 0){
            float bv = bestValue[0];
            int bi = bestIndex[0];
            for(int w = 1; w < warps; w++){
                if(bestValue[w] > bv || (bestValue[w] == bv && bestIndex[w] < bi)){
                    bv = bestValue[w];
                    bi = bestIndex[w];
                }
            }
            pickedValue[k] = bv;
            pickedIndex[k] = bi;
        }
        __syncthreads();
        if(e == pickedIndex[k]) taken = true;
    }
    if(e != 0) return;
    float top = pickedValue[0];
    float prob[KUSED];
    float sum = 0.0f;
    for(int k = 0; k < KUSED; k++){
        prob[k] = __expf(pickedValue[k] - top);
        sum += prob[k];
    }
    for(int k = 0; k < KUSED; k++){
        ids[t * KUSED + k] = pickedIndex[k];
        weights[t * KUSED + k] = prob[k] / sum;
    }
}

#if EXPERTS % 32 == 0 && EXPERTS <= 1024
extern "C" __global__ void __launch_bounds__(256) routeWarp(const float* logits, int* ids, float* weights, int n) {
    int t = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if(t >= n) return;
    float mine[EXPERTS / 32];
    #pragma unroll
    for(int i = 0; i < EXPERTS / 32; i++) mine[i] = logits[(u64)t * EXPERTS + i * 32 + lane];
    unsigned int taken = 0;
    float pickedValue[KUSED];
    int pickedIndex[KUSED];
    #pragma unroll
    for(int k = 0; k < KUSED; k++){
        float v = -3.0e38f;
        int index = lane;
        bool first = true;
        #pragma unroll
        for(int i = 0; i < EXPERTS / 32; i++){
            float x = (taken >> i) & 1u ? -3.0e38f : mine[i];
            if(first || x > v){
                v = x;
                index = i * 32 + lane;
                first = false;
            }
        }
        for(int step = 16; step > 0; step >>= 1){
            float ov = __shfl_xor_sync(0xffffffff, v, step);
            int oi = __shfl_xor_sync(0xffffffff, index, step);
            if(ov > v || (ov == v && oi < index)){
                v = ov;
                index = oi;
            }
        }
        pickedValue[k] = v;
        pickedIndex[k] = index;
        if((index & 31) == lane) taken |= 1u << (index >> 5);
    }
    if(lane != 0) return;
    float top = pickedValue[0];
    float prob[KUSED];
    float sum = 0.0f;
    for(int k = 0; k < KUSED; k++){
        prob[k] = __expf(pickedValue[k] - top);
        sum += prob[k];
    }
    for(int k = 0; k < KUSED; k++){
        ids[t * KUSED + k] = pickedIndex[k];
        weights[t * KUSED + k] = prob[k] / sum;
    }
}
#endif

extern "C" __global__ void gdnConv(const float* qkv, const float* history, const float* weight, float* u) {
    __shared__ float scratch[32];
    int head = blockIdx.x, t = blockIdx.y;
    int c = head * 128 + threadIdx.x;
    const float* w = weight + c * 4;
    float s = 0.0f;
    #pragma unroll
    for(int k = 0; k < 4; k++){
        int when = t - 3 + k;
        float v = when >= 0
            ? qkv[(u64)when * CONVW + c]
            : history[(when + 3) * CONVW + c];
        s += w[k] * v;
    }
    float o = s * gateOf(s);
    if(head < 2 * KHEADS) o *= rsqrtf(blockSum(o * o, scratch) + 1e-6f);
    u[(u64)t * CONVW + c] = o;
}

extern "C" __global__ void gdnHistory(const float* qkv, float* history, int n) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if(c >= CONVW) return;
    for(int k = 0; k < 3; k++){
        int when = n - 3 + k;
        history[k * CONVW + c] = when >= 0
            ? qkv[(u64)when * CONVW + c]
            : history[(when + 3) * CONVW + c];
    }
}

#define SCAN_TOKENS 16

extern "C" __global__ void gdnScan(const float* u, const float* alpha, const float* beta, const float* dtBias, const float* decay, float* state, float* out, int n, int keep) {
    __shared__ float qs[SCAN_TOKENS][4][33];
    __shared__ float ks[SCAN_TOKENS][4][33];
    __shared__ float vs[SCAN_TOKENS][32];
    __shared__ float gs[SCAN_TOKENS];
    __shared__ float bs[SCAN_TOKENS];
    int h = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    int part = lane & 3, col = warp * 8 + (lane >> 2), j = blockIdx.x * 32 + col;
    int kh = h % KHEADS;
    float* S = state + (u64)h * 16384;
    float s[32];
    #pragma unroll
    for(int i = 0; i < 32; i++) s[i] = S[(part * 32 + i) * 128 + j];
    float bias = dtBias[h];
    float rate = decay[h];

    for(int first = 0; first < n; first += SCAN_TOKENS){
        int count = min(SCAN_TOKENS, n - first);
        __syncthreads();
        float q[SCAN_TOKENS], key[SCAN_TOKENS], v[SCAN_TOKENS / 4];
        #pragma unroll
        for(int t = 0; t < SCAN_TOKENS; t++){
            const float* row = u + (u64)(first + t) * CONVW;
            q[t] = t < count ? row[kh * 128 + tid] : 0.0f;
            key[t] = t < count ? row[KHEADS * 128 + kh * 128 + tid] : 0.0f;
        }
        #pragma unroll
        for(int r = 0; r < SCAN_TOKENS / 4; r++){
            int t = r * 4 + (tid >> 5);
            v[r] = t < count ? u[(u64)(first + t) * CONVW + 2 * KHEADS * 128 + h * 128 + blockIdx.x * 32 + lane] : 0.0f;
        }
        if(tid < count){
            float raw = alpha[(first + tid) * VHEADS + h] + bias;
            float soft = raw > 20.0f ? raw : log1pf(__expf(raw));
            gs[tid] = __expf(soft * rate);
            bs[tid] = gateOf(beta[(first + tid) * VHEADS + h]);
        }
        #pragma unroll
        for(int t = 0; t < SCAN_TOKENS; t++){
            qs[t][tid >> 5][tid & 31] = q[t];
            ks[t][tid >> 5][tid & 31] = key[t];
        }
        #pragma unroll
        for(int r = 0; r < SCAN_TOKENS / 4; r++) vs[r * 4 + (tid >> 5)][lane] = v[r];
        __syncthreads();

        for(int t = 0; t < count; t++){
            float g = gs[t], b = bs[t];
            float kv[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            #pragma unroll
            for(int i = 0; i < 32; i++) kv[i & 3] += s[i] * ks[t][part][i];
            float dot = kv[0] + kv[1] + kv[2] + kv[3];
            dot += __shfl_xor_sync(0xffffffff, dot, 1);
            dot += __shfl_xor_sync(0xffffffff, dot, 2);
            float delta = (vs[t][col] - g * dot) * b;

            float o[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            #pragma unroll
            for(int i = 0; i < 32; i++){
                s[i] = g * s[i] + ks[t][part][i] * delta;
                o[i & 3] += s[i] * qs[t][part][i];
            }
            float sum = o[0] + o[1] + o[2] + o[3];
            sum += __shfl_xor_sync(0xffffffff, sum, 1);
            sum += __shfl_xor_sync(0xffffffff, sum, 2);
            if(part == 0) out[(u64)(first + t) * INNER + h * 128 + j] = sum * 0.08838834764831845f;
        }
    }

    if(!keep) return;
    #pragma unroll
    for(int i = 0; i < 32; i++) S[(part * 32 + i) * 128 + j] = s[i];
}

#if VHEADS == 3 * KHEADS
extern "C" __global__ void __launch_bounds__(64) gdnScanGroup(const float* u, const float* alpha, const float* beta, const float* dtBias, const float* decay, float* state, float* out, int n, int keep) {
    __shared__ float qs[SCAN_TOKENS][4][33];
    __shared__ float ks[SCAN_TOKENS][4][33];
    __shared__ float vs[3][SCAN_TOKENS][16];
    __shared__ float gs[3][SCAN_TOKENS];
    __shared__ float bs[3][SCAN_TOKENS];
    int kh = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    int part = lane & 3, col = warp * 8 + (lane >> 2), j = blockIdx.x * 16 + col;
    float s[3][32];
    #pragma unroll
    for(int z = 0; z < 3; z++){
        const float* S = state + (u64)(kh + z * KHEADS) * 16384;
        #pragma unroll
        for(int i = 0; i < 32; i++) s[z][i] = S[(part * 32 + i) * 128 + j];
    }

    for(int first = 0; first < n; first += SCAN_TOKENS){
        int count = min(SCAN_TOKENS, n - first);
        __syncthreads();
        float q[2][SCAN_TOKENS], key[2][SCAN_TOKENS];
        #pragma unroll
        for(int p = 0; p < 2; p++){
            #pragma unroll
            for(int t = 0; t < SCAN_TOKENS; t++){
                const float* row = u + (u64)(first + t) * CONVW;
                q[p][t] = t < count ? row[kh * 128 + p * 64 + tid] : 0.0f;
                key[p][t] = t < count ? row[KHEADS * 128 + kh * 128 + p * 64 + tid] : 0.0f;
            }
        }
        #pragma unroll
        for(int r = 0; r < 3 * SCAN_TOKENS / 4; r++){
            int at = r * 64 + tid;
            int z = at / (SCAN_TOKENS * 16), t = (at / 16) % SCAN_TOKENS, c = at & 15;
            vs[z][t][c] = t < count ? u[(u64)(first + t) * CONVW + 2 * KHEADS * 128 + (kh + z * KHEADS) * 128 + blockIdx.x * 16 + c] : 0.0f;
        }
        if(tid < 3 * SCAN_TOKENS){
            int z = tid / SCAN_TOKENS, t = tid % SCAN_TOKENS, h = kh + z * KHEADS;
            if(t < count){
                float raw = alpha[(first + t) * VHEADS + h] + dtBias[h];
                float soft = raw > 20.0f ? raw : log1pf(__expf(raw));
                gs[z][t] = __expf(soft * decay[h]);
                bs[z][t] = gateOf(beta[(first + t) * VHEADS + h]);
            }
        }
        #pragma unroll
        for(int p = 0; p < 2; p++){
            int d = p * 64 + tid;
            #pragma unroll
            for(int t = 0; t < SCAN_TOKENS; t++){
                qs[t][d >> 5][d & 31] = q[p][t];
                ks[t][d >> 5][d & 31] = key[p][t];
            }
        }
        __syncthreads();

        for(int t = 0; t < count; t++){
            float kk[32], qq[32];
            #pragma unroll
            for(int i = 0; i < 32; i++){
                kk[i] = ks[t][part][i];
                qq[i] = qs[t][part][i];
            }
            float kv[3][4];
            #pragma unroll
            for(int z = 0; z < 3; z++) kv[z][0] = kv[z][1] = kv[z][2] = kv[z][3] = 0.0f;
            #pragma unroll
            for(int i = 0; i < 32; i++){
                #pragma unroll
                for(int z = 0; z < 3; z++) kv[z][i & 3] += s[z][i] * kk[i];
            }
            float dot[3], delta[3], g[3];
            #pragma unroll
            for(int z = 0; z < 3; z++) dot[z] = kv[z][0] + kv[z][1] + kv[z][2] + kv[z][3];
            #pragma unroll
            for(int z = 0; z < 3; z++) dot[z] += __shfl_xor_sync(0xffffffff, dot[z], 1);
            #pragma unroll
            for(int z = 0; z < 3; z++) dot[z] += __shfl_xor_sync(0xffffffff, dot[z], 2);
            #pragma unroll
            for(int z = 0; z < 3; z++){
                g[z] = gs[z][t];
                delta[z] = (vs[z][t][col] - g[z] * dot[z]) * bs[z][t];
            }
            float o[3][4];
            #pragma unroll
            for(int z = 0; z < 3; z++) o[z][0] = o[z][1] = o[z][2] = o[z][3] = 0.0f;
            #pragma unroll
            for(int i = 0; i < 32; i++){
                #pragma unroll
                for(int z = 0; z < 3; z++){
                    s[z][i] = g[z] * s[z][i] + kk[i] * delta[z];
                    o[z][i & 3] += s[z][i] * qq[i];
                }
            }
            float sum[3];
            #pragma unroll
            for(int z = 0; z < 3; z++) sum[z] = o[z][0] + o[z][1] + o[z][2] + o[z][3];
            #pragma unroll
            for(int z = 0; z < 3; z++) sum[z] += __shfl_xor_sync(0xffffffff, sum[z], 1);
            #pragma unroll
            for(int z = 0; z < 3; z++) sum[z] += __shfl_xor_sync(0xffffffff, sum[z], 2);
            if(part == 0){
                #pragma unroll
                for(int z = 0; z < 3; z++) out[(u64)(first + t) * INNER + (kh + z * KHEADS) * 128 + j] = sum[z] * 0.08838834764831845f;
            }
        }
    }

    if(!keep) return;
    #pragma unroll
    for(int z = 0; z < 3; z++){
        float* S = state + (u64)(kh + z * KHEADS) * 16384;
        #pragma unroll
        for(int i = 0; i < 32; i++) S[(part * 32 + i) * 128 + j] = s[z][i];
    }
}
#endif

extern "C" __global__ void gdnNorm(float* o, const float* z, const float* weight) {
    __shared__ float scratch[32];
    int h = blockIdx.x, t = blockIdx.y, j = threadIdx.x;
    u64 at = (u64)t * INNER + h * 128 + j;
    float v = o[at];
    float inv = rsqrtf(blockSum(v * v, scratch) / 128.0f + 1e-6f);
    float g = z[at];
#if SILU_NORM
    o[at] = v * inv * weight[j] * g * gateOf(g);
#else
    o[at] = v * inv * weight[j] * gateOf(g);
#endif
}

extern "C" __global__ void attnPrepare(float* q, const float* k, const float* v, float* iq, const float* ik, const float* queryNorm, const float* keyNorm, const float* indexNorm, i8* keys, float* keyScales, i8* values, float* valueScales, i8* indexRaw, float* indexScales, const float* ropeCos, const float* ropeSin, const int* positions) {
    __shared__ float buffer[256];
    __shared__ float scratch[32];
    int t = blockIdx.y, seg = blockIdx.x, i = threadIdx.x;
    int p = positions[t];
    if(seg == QHEADS + KVHEADS + IHEADS){
        for(int d = i; d < KVW; d += blockDim.x){
            float x = v[(u64)t * KVW + d];
            float peak = fabsf(x);
            for(int step = 16; step > 0; step >>= 1) peak = fmaxf(peak, __shfl_xor_sync(0xffffffff, peak, step));
            float inv = peak > 0.0f ? 127.0f / peak : 0.0f;
            values[(u64)p * KVW + d] = (i8)__float2int_rn(x * inv);
            if((d & 31) == 0) valueScales[(u64)p * (KVW / 32) + d / 32] = peak / 127.0f;
        }
        if(IHEADS > 0 && i < 128){
            float x = ik[(u64)t * 128 + i];
            float peak = fabsf(x);
            for(int step = 16; step > 0; step >>= 1) peak = fmaxf(peak, __shfl_xor_sync(0xffffffff, peak, step));
            float inv = peak > 0.0f ? 127.0f / peak : 0.0f;
            indexRaw[(u64)p * 128 + i] = (i8)__float2int_rn(x * inv);
            if((i & 31) == 0) indexScales[(u64)p * 4 + i / 32] = peak / 127.0f;
        }
        return;
    }

    float* src = seg < QHEADS
        ? q + (u64)t * (QHEADS * 512) + seg * 512
        : seg < QHEADS + KVHEADS
            ? (float*)k + (u64)t * KVW + (seg - QHEADS) * 256
            : iq + ((u64)t * IHEADS + seg - QHEADS - KVHEADS) * 128;
    const float* weight = seg < QHEADS
        ? queryNorm
        : seg < QHEADS + KVHEADS
            ? keyNorm
            : indexNorm;
    int width = seg < QHEADS + KVHEADS ? 256 : 128;

    float val = i < width ? src[i] : 0.0f;
    float inv = rsqrtf(blockSum(val * val, scratch) / (float)width + 1e-6f);
    buffer[i] = i < width ? val * inv * weight[i] : 0.0f;
    __syncthreads();
    if(i < 32){
        float c = ropeCos[p * 32 + i];
        float s = ropeSin[p * 32 + i];
        float a = buffer[i];
        float b = buffer[i + 32];
        buffer[i] = a * c - b * s;
        buffer[i + 32] = a * s + b * c;
    }
    __syncthreads();
    if(i >= width) return;
    if(seg >= QHEADS && seg < QHEADS + KVHEADS){
        float x = buffer[i];
        float peak = fabsf(x);
        for(int step = 16; step > 0; step >>= 1) peak = fmaxf(peak, __shfl_xor_sync(0xffffffff, peak, step));
        float inv = peak > 0.0f ? 127.0f / peak : 0.0f;
        int d = (seg - QHEADS) * 256 + i;
        keys[(u64)p * KVW + d] = (i8)__float2int_rn(x * inv);
        if((i & 31) == 0) keyScales[(u64)p * (KVW / 32) + d / 32] = peak / 127.0f;
        return;
    }
    src[i] = buffer[i];
}

extern "C" __global__ void poolIndex(const i8* indexRaw, const float* indexScales, float* pooled, const float* norm, const float* ropeCos, const float* ropeSin, const int* positions) {
    __shared__ float buffer[128];
    __shared__ float scratch[32];
    int t = blockIdx.x, i = threadIdx.x;
    int p = positions[t];
    if((p + 1) % 4 != 0) return;
    int block = (p + 1) / 4 - 1;
    float sum = 0.0f;
    for(int c = 0; c < 4; c++) sum += (float)indexRaw[(u64)(4 * block + c) * 128 + i] * indexScales[(u64)(4 * block + c) * 4 + i / 32];
    float val = sum * 0.25f;
    float inv = rsqrtf(blockSum(val * val, scratch) / 128.0f + 1e-6f);
    buffer[i] = val * inv * norm[i];
    __syncthreads();
    if(i < 32){
        int at = 4 * block;
        float c = ropeCos[at * 32 + i];
        float s = ropeSin[at * 32 + i];
        float a = buffer[i];
        float b = buffer[i + 32];
        buffer[i] = a * c - b * s;
        buffer[i + 32] = a * s + b * c;
    }
    __syncthreads();
    pooled[(u64)block * 128 + i] = buffer[i];
}

extern "C" __global__ void indexScores(const float* iq, const float* pooled, float* scores, const int* positions, int budget, int stride) {
    int t = blockIdx.y;
    int bl = blockIdx.x * blockDim.y + threadIdx.y;
    int p = positions[t];
    if(p + 1 <= budget || bl >= (p + 1) / 4) return;
    float4 key = ((const float4*)(pooled + (u64)bl * 128))[threadIdx.x];
    float score = 0.0f;
    for(int h = 0; h < 4; h++){
        float4 query = ((const float4*)(iq + ((u64)t * 4 + h) * 128))[threadIdx.x];
        float d = query.x * key.x + query.y * key.y + query.z * key.z + query.w * key.w;
        for(int step = 16; step > 0; step >>= 1) d += __shfl_xor_sync(0xffffffff, d, step);
        if(d > 0.0f) score += d;
    }
    if(threadIdx.x == 0) scores[(u64)t * stride + bl] = score;
}

extern "C" __global__ void __launch_bounds__(256) indexTiles(const float* iq, const float* pooled, float* scores, const int* positions, int budget, int stride, int n) {
    __shared__ __align__(16) float qs[16][132];
    __shared__ __align__(16) float ks[16][132];
    __shared__ int reach[32];
    int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    int t0 = blockIdx.y * 32, b0 = blockIdx.x * 128;
    if(tid < 32){
        int t = t0 + tid;
        int p = t < n ? positions[t] : -1;
        reach[tid] = p + 1 > budget ? (p + 1) / 4 : 0;
    }
    __syncthreads();
    int most = 0;
    for(int i = 0; i < 32; i++) most = max(most, reach[i]);
    if(b0 >= most) return;

    float acc[8][8];
    #pragma unroll
    for(int i = 0; i < 8; i++){
        #pragma unroll
        for(int j = 0; j < 8; j++) acc[i][j] = 0.0f;
    }
    for(int d0 = 0; d0 < 128; d0 += 16){
        #pragma unroll
        for(int r = 0; r < 2; r++){
            int at = r * 256 + tid, row = at >> 2, piece = at & 3;
            int t = t0 + (row >> 2);
            float4 v = t < n ? *(const float4*)(iq + (u64)t * 512 + (row & 3) * 128 + d0 + piece * 4) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            qs[piece * 4][row] = v.x;
            qs[piece * 4 + 1][row] = v.y;
            qs[piece * 4 + 2][row] = v.z;
            qs[piece * 4 + 3][row] = v.w;
            int b = b0 + row;
            float4 w = b < most ? *(const float4*)(pooled + (u64)b * 128 + d0 + piece * 4) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            ks[piece * 4][row] = w.x;
            ks[piece * 4 + 1][row] = w.y;
            ks[piece * 4 + 2][row] = w.z;
            ks[piece * 4 + 3][row] = w.w;
        }
        __syncthreads();
        #pragma unroll
        for(int k = 0; k < 16; k++){
            float4 a0 = *(const float4*)&qs[k][ty * 8], a1 = *(const float4*)&qs[k][ty * 8 + 4];
            float4 c0 = *(const float4*)&ks[k][tx * 4], c1 = *(const float4*)&ks[k][64 + tx * 4];
            float a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            float c[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
            #pragma unroll
            for(int i = 0; i < 8; i++){
                #pragma unroll
                for(int j = 0; j < 8; j++) acc[i][j] += a[i] * c[j];
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for(int s = 0; s < 2; s++){
        int local = ty * 2 + s, t = t0 + local;
        if(t >= n) continue;
        int limit = reach[local];
        float* row = scores + (u64)t * stride;
        #pragma unroll
        for(int j = 0; j < 8; j++){
            int b = b0 + (j < 4 ? tx * 4 + j : 64 + tx * 4 + j - 4);
            if(b >= limit) continue;
            float score = 0.0f;
            #pragma unroll
            for(int h = 0; h < 4; h++){
                float d = acc[s * 4 + h][j];
                if(d > 0.0f) score += d;
            }
            row[b] = score;
        }
    }
}

extern "C" __global__ void selectCells(const float* scores, int* cells, const int* positions, int budget, int stride) {
    __shared__ int hist[256];
    __shared__ int scratch[32];
    __shared__ u32 chosenPrefix;
    __shared__ int chosenRemain;
    int t = blockIdx.x, tid = threadIdx.x;
    int p = positions[t];
    if(p + 1 <= budget) return;
    int blocks = (p + 1) / 4;
    int tail = p + 1 - 4 * blocks;
    int need = budget - tail;
    int wanted = (need + 3) / 4;
    int last = need - 4 * (wanted - 1);
    const float* row = scores + (u64)t * stride;
    int* out = cells + (u64)t * budget;
    if(tid < tail) out[tid] = 4 * blocks + tid;

    u32 prefix = 0, mask = 0;
    int remain = wanted;
    for(int shift = 24; shift >= 0; shift -= 8){
        for(int b = tid; b < 256; b += blockDim.x) hist[b] = 0;
        __syncthreads();
        for(int b = tid; b < blocks; b += blockDim.x){
            u32 bits = __float_as_uint(row[b]);
            if((bits & mask) == prefix) atomicAdd(&hist[(bits >> shift) & 255], 1);
        }
        __syncthreads();
        if(tid < 32){
            int part[8], sum = 0;
            #pragma unroll
            for(int k = 0; k < 8; k++){
                part[k] = hist[255 - (tid * 8 + k)];
                sum += part[k];
            }
            int upto = sum;
            for(int step = 1; step < 32; step <<= 1){
                int other = __shfl_up_sync(0xffffffff, upto, step);
                if(tid >= step) upto += other;
            }
            int seen = upto - sum;
            if(seen < remain && remain <= upto){
                #pragma unroll
                for(int k = 0; k < 8; k++){
                    if(seen + part[k] >= remain){
                        chosenPrefix = prefix | ((u32)(255 - (tid * 8 + k)) << shift);
                        chosenRemain = remain - seen;
                        break;
                    }
                    seen += part[k];
                }
            }
        }
        __syncthreads();
        prefix = chosenPrefix;
        remain = chosenRemain;
        mask |= 255u << shift;
    }

    int written = 0, ties = 0;
    for(int start = 0; start < blocks; start += blockDim.x){
        int b = start + tid;
        u32 bits = b < blocks ? __float_as_uint(row[b]) : 0u;
        bool tie = b < blocks && bits == prefix;
        bool greater = b < blocks && bits > prefix;
        int tieTotal = 0;
        int tieRank = ties + blockScan(tie ? 1 : 0, tieTotal, scratch);
        bool partial = tie && tieRank == remain - 1;
        bool full = greater || (tie && tieRank < remain - 1);
        int fullTotal = 0;
        int at = written + blockScan(full ? 1 : 0, fullTotal, scratch);
        if(full){
            for(int c = 0; c < 4; c++) out[tail + 4 * at + c] = 4 * b + c;
        }
        if(partial){
            for(int c = 0; c < last; c++) out[tail + 4 * (wanted - 1) + c] = 4 * b + c;
        }
        written += fullTotal;
        ties += tieTotal;
    }
}

extern "C" __global__ void attention(const float* q, const i8* keys, const float* keyScales, const i8* values, const float* valueScales, const int* cells, const int* positions, float* out, int budget) {
    __shared__ float query[256];
    __shared__ float score[4096];
    __shared__ float merge[8][256];
    __shared__ float scratch[32];
    int t = blockIdx.y, h = blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    int kvh = h / (QHEADS / KVHEADS);
    int p = positions[t];
    const float* qh = q + (u64)t * (QHEADS * 512) + h * 512;
    query[tid] = qh[tid];
    bool all = p + 1 <= budget;
    int count = all ? p + 1 : budget;
    const int* list = cells + (u64)t * budget;
    __syncthreads();

    float qv[8];
    #pragma unroll
    for(int k = 0; k < 8; k++) qv[k] = query[lane * 8 + k];
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float runBest = -3.0e38f;
    float total = 0.0f;
    for(int from = 0; from < count; from += 4096){
        int width = min(4096, count - from);
        for(int i = warp; i < width; i += 8){
            int cell = all ? from + i : list[from + i];
            uint2 raw = *(const uint2*)(keys + (u64)cell * KVW + kvh * 256 + lane * 8);
            u32 words[2] = {raw.x, raw.y};
            float d = 0.0f;
            #pragma unroll
            for(int k = 0; k < 2; k++){
                #pragma unroll
                for(int b = 0; b < 4; b++) d += qv[4 * k + b] * (float)(i8)(words[k] >> (8 * b));
            }
            d *= keyScales[(u64)cell * (KVW / 32) + kvh * 8 + (lane >> 2)];
            for(int step = 16; step > 0; step >>= 1) d += __shfl_xor_sync(0xffffffff, d, step);
            if(lane == 0) score[i] = d * 0.0625f;
        }
        __syncthreads();

        float best = runBest;
        for(int i = tid; i < width; i += blockDim.x) best = fmaxf(best, score[i]);
        best = blockMax(best, scratch);
        float shrink = __expf(runBest - best);
        float sum = 0.0f;
        for(int i = tid; i < width; i += blockDim.x){
            float e = __expf(score[i] - best);
            score[i] = e;
            sum += e;
        }
        total = total * shrink + blockSum(sum, scratch);
        runBest = best;
        __syncthreads();

        #pragma unroll
        for(int k = 0; k < 8; k++) acc[k] *= shrink;
        for(int i = warp; i < width; i += 8){
            int cell = all ? from + i : list[from + i];
            uint2 raw = *(const uint2*)(values + (u64)cell * KVW + kvh * 256 + lane * 8);
            u32 words[2] = {raw.x, raw.y};
            float w = score[i] * valueScales[(u64)cell * (KVW / 32) + kvh * 8 + (lane >> 2)];
            #pragma unroll
            for(int k = 0; k < 2; k++){
                #pragma unroll
                for(int b = 0; b < 4; b++) acc[4 * k + b] += w * (float)(i8)(words[k] >> (8 * b));
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int k = 0; k < 8; k++) merge[warp][lane * 8 + k] = acc[k];
    __syncthreads();
    float o = 0.0f;
    for(int w = 0; w < 8; w++) o += merge[w][tid];
    out[(u64)t * (QHEADS * 256) + h * 256 + tid] = o / total * gateOf(qh[256 + tid]);
}

#if KVHEADS > 0 && QHEADS <= 24 * KVHEADS
#define QPERKV (QHEADS / KVHEADS)
#define TAKE 128
extern "C" __global__ void __launch_bounds__(256) attentionGroup(const float* q, const i8* keys, const float* keyScales, const i8* values, const float* valueScales, const int* cells, const int* positions, float* out, int budget) {
    __shared__ __align__(16) float query[QPERKV][2][132];
    __shared__ float score[QPERKV][TAKE];
    __shared__ float best[QPERKV], total[QPERKV], shrink[QPERKV];
    __shared__ int place[TAKE];
    int t = blockIdx.y, kvh = blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    int p = positions[t];
    const float* qt = q + (u64)t * (QHEADS * 512) + (u64)kvh * QPERKV * 512;
    for(int i = tid; i < QPERKV * 256; i += 256){
        int h = i >> 8, d = i & 255;
        query[h][d >> 7][d & 127] = qt[h * 512 + d];
    }
    if(tid < QPERKV){
        best[tid] = -3.0e38f;
        total[tid] = 0.0f;
    }
    bool all = p + 1 <= budget;
    int count = all ? p + 1 : budget;
    const int* list = cells + (u64)t * budget;
    float acc[QPERKV];
    #pragma unroll
    for(int h = 0; h < QPERKV; h++) acc[h] = 0.0f;
    int half = tid & 1, mine = tid >> 1;
    __syncthreads();

    for(int from = 0; from < count; from += TAKE){
        int width = min(TAKE, count - from);
        if(tid < width) place[tid] = all ? from + tid : list[from + tid];
        __syncthreads();

        int cell = mine < width ? place[mine] : place[0];
        const uint4* row = (const uint4*)(keys + (u64)cell * KVW + kvh * 256 + half * 128);
        const float* scale = keyScales + (u64)cell * (KVW / 32) + kvh * 8 + half * 4;
        float dot[QPERKV];
        #pragma unroll
        for(int h = 0; h < QPERKV; h++) dot[h] = 0.0f;
        #pragma unroll
        for(int g = 0; g < 4; g++){
            uint4 a = row[2 * g], b = row[2 * g + 1];
            u32 words[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
            float part[QPERKV];
            #pragma unroll
            for(int h = 0; h < QPERKV; h++) part[h] = 0.0f;
            #pragma unroll
            for(int w = 0; w < 8; w++){
                float k0 = (float)(i8)(words[w] & 255), k1 = (float)(i8)((words[w] >> 8) & 255), k2 = (float)(i8)((words[w] >> 16) & 255), k3 = (float)(i8)(words[w] >> 24);
                #pragma unroll
                for(int h = 0; h < QPERKV; h++){
                    float4 qv = *(const float4*)&query[h][half][g * 32 + w * 4];
                    part[h] += qv.x * k0 + qv.y * k1 + qv.z * k2 + qv.w * k3;
                }
            }
            float s = scale[g];
            #pragma unroll
            for(int h = 0; h < QPERKV; h++) dot[h] += part[h] * s;
        }
        #pragma unroll
        for(int h = 0; h < QPERKV; h++){
            float v = dot[h] + __shfl_xor_sync(0xffffffff, dot[h], 1);
            if(half == 0 && mine < width) score[h][mine] = v * 0.0625f;
        }
        __syncthreads();

        for(int h = warp; h < QPERKV; h += 8){
            float m = -3.0e38f;
            for(int i = lane; i < width; i += 32) m = fmaxf(m, score[h][i]);
            for(int step = 16; step > 0; step >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, step));
            float top = fmaxf(best[h], m);
            float sum = 0.0f;
            for(int i = lane; i < width; i += 32){
                float e = __expf(score[h][i] - top);
                score[h][i] = e;
                sum += e;
            }
            for(int step = 16; step > 0; step >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, step);
            if(lane == 0){
                float k = __expf(best[h] - top);
                shrink[h] = k;
                total[h] = total[h] * k + sum;
                best[h] = top;
            }
        }
        __syncthreads();

        #pragma unroll
        for(int h = 0; h < QPERKV; h++) acc[h] *= shrink[h];
        const float* vscale = valueScales + kvh * 8 + (tid >> 5);
        const i8* vbase = values + kvh * 256 + tid;
        #pragma unroll 4
        for(int i = 0; i < width; i++){
            int c = place[i];
            float v = (float)__ldg(vbase + (u64)c * KVW) * __ldg(vscale + (u64)c * (KVW / 32));
            #pragma unroll
            for(int h = 0; h < QPERKV; h++) acc[h] += score[h][i] * v;
        }
        __syncthreads();
    }
    #pragma unroll
    for(int h = 0; h < QPERKV; h++) out[(u64)t * (QHEADS * 256) + (kvh * QPERKV + h) * 256 + tid] = acc[h] / total[h] * gateOf(qt[h * 512 + 256 + tid]);
}
#endif

#if KVHEADS > 0 && QHEADS <= 16 * KVHEADS
#define MROWS (QHEADS / KVHEADS)
__device__ __forceinline__ void halfMma(float* c, const u32* a, u32 b0, u32 b1) {
#if __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ u32 codesHalf(u32 x) {
    u32 y = (((x & 0xffu) | ((x & 0xff00u) << 8)) ^ 0x00800080u) | 0x64006400u;
    u32 out = 0;
#if __CUDA_ARCH__ >= 800
    asm("sub.f16x2 %0, %1, %2;" : "=r"(out) : "r"(y), "r"(0x64806480u));
#endif
    return out;
}

__device__ __forceinline__ void splitHalf(float a, float b, u32& high, u32& low) {
    high = pairHalf(a, b);
    float ha = 0.0f, hb = 0.0f;
#if __CUDA_ARCH__ >= 800
    asm("{.reg .b16 l, h; mov.b32 {l, h}, %2; cvt.f32.f16 %0, l; cvt.f32.f16 %1, h;}" : "=f"(ha), "=f"(hb) : "r"(high));
#endif
    low = pairHalf(a - ha, b - hb);
}

__device__ __forceinline__ int swizzle(int cell, int byte) {
    return cell * 64 + ((((byte >> 4) ^ (cell >> 1)) & 3) << 4) + (byte & 15);
}


template<bool SIGNED>
__device__ __forceinline__ void wideStep(float* c, const u32* hi, const u32* lo, u32 b0, u32 b1) {
    int h[4], l[4];
#if __CUDA_ARCH__ >= 800
    if(SIGNED) asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};" : "=r"(h[0]), "=r"(h[1]), "=r"(h[2]), "=r"(h[3]) : "r"(hi[0]), "r"(hi[1]), "r"(hi[2]), "r"(hi[3]), "r"(b0), "r"(b1), "r"(0x4B400000));
    else asm("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};" : "=r"(h[0]), "=r"(h[1]), "=r"(h[2]), "=r"(h[3]) : "r"(hi[0]), "r"(hi[1]), "r"(hi[2]), "r"(hi[3]), "r"(b0), "r"(b1), "r"(0x4B400000));
    asm("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};" : "=r"(l[0]), "=r"(l[1]), "=r"(l[2]), "=r"(l[3]) : "r"(lo[0]), "r"(lo[1]), "r"(lo[2]), "r"(lo[3]), "r"(b0), "r"(b1), "r"(0x4B400000));
#else
    h[0] = h[1] = h[2] = h[3] = l[0] = l[1] = l[2] = l[3] = 0x4B400000;
#endif
    #pragma unroll
    for(int i = 0; i < 4; i++) c[i] = (__int_as_float(h[i]) - 12582912.0f) * 256.0f + (__int_as_float(l[i]) - 12582912.0f);
}

__device__ __forceinline__ void packWide(u32& hi, u32& lo, float a, float b, float c, float d, u32 order) {
    u32 x = __byte_perm(__float_as_uint(a), __float_as_uint(b), order);
    u32 y = __byte_perm(__float_as_uint(c), __float_as_uint(d), order);
    lo = __byte_perm(x, y, 0x5410);
    hi = __byte_perm(x, y, 0x7632);
}

__device__ __forceinline__ void turnBytes(u32* out, u32 w0, u32 w1, u32 w2, u32 w3) {
    u32 x0 = __byte_perm(w0, w1, 0x5140), x1 = __byte_perm(w0, w1, 0x7362);
    u32 y0 = __byte_perm(w2, w3, 0x5140), y1 = __byte_perm(w2, w3, 0x7362);
    out[0] = __byte_perm(x0, y0, 0x5410);
    out[1] = __byte_perm(x0, y0, 0x7632);
    out[2] = __byte_perm(x1, y1, 0x5410);
    out[3] = __byte_perm(x1, y1, 0x7632);
}

__device__ __forceinline__ int partAt(int r, int c) {
    return r * 32 + ((((c >> 2) ^ (r << 1)) & 7) << 2) + (c & 3);
}

__device__ __forceinline__ int probAt(int r, int c) {
    return r * 32 + ((((c >> 2) ^ ((r & 1) << 2)) & 7) << 2) + (c & 3);
}

extern "C" __global__ void __launch_bounds__(128) attentionMma(const float* q, const i8* keys, const float* keyScales, const i8* values, const float* valueScales, const int* cells, const int* positions, float* out, int budget) {
    __shared__ __align__(16) i8 rows[2][4][2][32 * 64];
    __shared__ float scales[2][4][4][32];
    __shared__ float part[4][16][33];
    __shared__ float prob[16][33];
    __shared__ float best[16], total[16], shrink[16], peak[4];
    int t = blockIdx.y, kvh = blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    int gid = lane >> 2, tig = lane & 3, dim0 = warp * 64;
    int p = positions[t];
    bool all = p + 1 <= budget;
    int count = all ? p + 1 : budget;
    const int* list = cells + (u64)t * budget;
    const float* qt = q + (u64)t * (QHEADS * 512) + (u64)kvh * MROWS * 512;

    float top = 0.0f;
    for(int i = tid; i < MROWS * 256; i += 128) top = fmaxf(top, fabsf(qt[(i >> 8) * 512 + (i & 255)]));
    for(int step = 16; step > 0; step >>= 1) top = fmaxf(top, __shfl_xor_sync(0xffffffff, top, step));
    if(lane == 0) peak[warp] = top;
    if(tid < 16){
        best[tid] = -3.0e38f;
        total[tid] = 0.0f;
    }
    __syncthreads();
    top = fmaxf(fmaxf(peak[0], peak[1]), fmaxf(peak[2], peak[3]));
    int power = 0;
    if(top > 0.0f) frexpf(top, &power);
    float up = ldexpf(1.0f, 14 - power), down = ldexpf(0.0625f * 1.4426950408889634f, power - 14);
    u32 qh[4][4], ql[4][4];
    #pragma unroll
    for(int kk = 0; kk < 4; kk++){
        #pragma unroll
        for(int r = 0; r < 4; r++){
            int row = gid + (r & 1) * 8, col = dim0 + kk * 16 + 2 * tig + (r >> 1) * 8;
            float2 x = row < MROWS ? *(const float2*)(qt + row * 512 + col) : make_float2(0.0f, 0.0f);
            splitHalf(x.x * up, x.y * up, qh[kk][r], ql[kk][r]);
        }
    }

    int chunks = (count + 31) / 32;
    float k0, k1, v0, v1;
    auto cellAt = [&](int c) {
        return c < count
            ? all
                ? c
                : list[c]
            : -1;
    };
    auto issue = [&](int cell, int st) {
        #pragma unroll
        for(int j = 0; j < 4; j++){
            int at = lane + 32 * j, c = at >> 2, piece = at & 3;
            int from = __shfl_sync(0xffffffff, cell, c);
            u64 offset = from >= 0 ? (u64)from * KVW + kvh * 256 + dim0 + piece * 16 : 0;
            copyAsync(&rows[st][warp][0][swizzle(c, piece * 16)], keys + offset, from >= 0);
            copyAsync(&rows[st][warp][1][swizzle(c, piece * 16)], values + offset, from >= 0);
        }
        u64 scaleAt = (u64)max(cell, 0) * (KVW / 32) + kvh * 8 + warp * 2;
        k0 = cell >= 0 ? keyScales[scaleAt] : 0.0f;
        k1 = cell >= 0 ? keyScales[scaleAt + 1] : 0.0f;
        v0 = cell >= 0 ? valueScales[scaleAt] : 0.0f;
        v1 = cell >= 0 ? valueScales[scaleAt + 1] : 0.0f;
    };
    issue(cellAt(lane), 0);
    int nextCell = cellAt(32 + lane);
    copyCommit();
    scales[0][warp][0][lane] = k0;
    scales[0][warp][1][lane] = k1;
    scales[0][warp][2][lane] = v0;
    scales[0][warp][3][lane] = v1;

    float acc[8][4];
    #pragma unroll
    for(int j = 0; j < 8; j++) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.0f;

    for(int ci = 0; ci < chunks; ci++){
        int st = ci & 1, nst = st ^ 1, c0 = ci * 32;
        bool more = ci + 1 < chunks;
        if(more) issue(nextCell, nst);
        nextCell = cellAt(c0 + 64 + lane);
        copyCommit();
        copyWaitOne();
        __syncwarp();
        const i8* K = rows[st][warp][0];
        const i8* V = rows[st][warp][1];
        #pragma unroll
        for(int nt = 0; nt < 4; nt++){
            float low[4] = {0.0f, 0.0f, 0.0f, 0.0f}, high[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            int c = nt * 8 + gid;
            #pragma unroll
            for(int kk = 0; kk < 4; kk++){
                u32 b0 = codesHalf(*(const u16*)(K + swizzle(c, kk * 16 + 2 * tig)));
                u32 b1 = codesHalf(*(const u16*)(K + swizzle(c, kk * 16 + 2 * tig + 8)));
                if(kk < 2){
                    halfMma(low, qh[kk], b0, b1);
                    halfMma(low, ql[kk], b0, b1);
                }
                else{
                    halfMma(high, qh[kk], b0, b1);
                    halfMma(high, ql[kk], b0, b1);
                }
            }
            int e = nt * 8 + 2 * tig;
            float a0 = scales[st][warp][0][e], a1 = scales[st][warp][0][e + 1];
            float b0 = scales[st][warp][1][e], b1 = scales[st][warp][1][e + 1];
            part[warp][gid][e] = low[0] * a0 + high[0] * b0;
            part[warp][gid][e + 1] = low[1] * a1 + high[1] * b1;
            part[warp][gid + 8][e] = low[2] * a0 + high[2] * b0;
            part[warp][gid + 8][e + 1] = low[3] * a1 + high[3] * b1;
        }
        __syncthreads();

        {
            int r = tid >> 3, sub = tid & 7;
            float x[4], m = -3.0e38f;
            #pragma unroll
            for(int j = 0; j < 4; j++){
                int c = sub * 4 + j;
                x[j] = c0 + c < count
                    ? (((part[0][r][c] + part[1][r][c]) + part[2][r][c]) + part[3][r][c]) * down
                    : -3.0e38f;
                m = fmaxf(m, x[j]);
            }
            for(int step = 1; step < 8; step <<= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, step));
            float old = best[r], fresh = fmaxf(old, m);
            float sum = 0.0f;
            #pragma unroll
            for(int j = 0; j < 4; j++){
                float e = exp2f(x[j] - fresh);
                prob[r][sub * 4 + j] = e;
                sum += e;
            }
            for(int step = 1; step < 8; step <<= 1) sum += __shfl_xor_sync(0xffffffff, sum, step);
            __syncwarp();
            if(sub == 0){
                float k = exp2f(old - fresh);
                shrink[r] = k;
                total[r] = total[r] * k + sum;
                best[r] = fresh;
            }
        }
        __syncthreads();

        float vmax = fmaxf(scales[st][warp][2][lane], scales[st][warp][3][lane]);
        for(int step = 16; step > 0; step >>= 1) vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffff, vmax, step));
        float vup = vmax > 0.0f ? 16384.0f / vmax : 0.0f, vdown = vmax * (1.0f / 16384.0f);
        float sum[8][4];
        #pragma unroll
        for(int j = 0; j < 8; j++) sum[j][0] = sum[j][1] = sum[j][2] = sum[j][3] = 0.0f;
        #pragma unroll
        for(int ks = 0; ks < 2; ks++){
            int ca = ks * 16 + 2 * tig, cb = ca + 8;
            float p00 = prob[gid][ca], p01 = prob[gid][ca + 1], p10 = prob[gid + 8][ca], p11 = prob[gid + 8][ca + 1];
            float p02 = prob[gid][cb], p03 = prob[gid][cb + 1], p12 = prob[gid + 8][cb], p13 = prob[gid + 8][cb + 1];
            #pragma unroll
            for(int g = 0; g < 2; g++){
                const float* vs = scales[st][warp][2 + g];
                float w0 = vs[ca] * vup, w1 = vs[ca + 1] * vup, w2 = vs[cb] * vup, w3 = vs[cb + 1] * vup;
                u32 hi[4], lo[4];
                splitHalf(p00 * w0, p01 * w1, hi[0], lo[0]);
                splitHalf(p10 * w0, p11 * w1, hi[1], lo[1]);
                splitHalf(p02 * w2, p03 * w3, hi[2], lo[2]);
                splitHalf(p12 * w2, p13 * w3, hi[3], lo[3]);
                #pragma unroll
                for(int j = 0; j < 4; j++){
                    int d = (g * 4 + j) * 8 + gid;
                    u32 x0 = (u32)(u8)V[swizzle(ca, d)] | ((u32)(u8)V[swizzle(ca + 1, d)] << 8);
                    u32 x1 = (u32)(u8)V[swizzle(cb, d)] | ((u32)(u8)V[swizzle(cb + 1, d)] << 8);
                    u32 b0 = codesHalf(x0), b1 = codesHalf(x1);
                    halfMma(sum[g * 4 + j], hi, b0, b1);
                    halfMma(sum[g * 4 + j], lo, b0, b1);
                }
            }
        }
        float a0 = shrink[gid], a1 = shrink[gid + 8];
        #pragma unroll
        for(int j = 0; j < 8; j++){
            acc[j][0] = acc[j][0] * a0 + sum[j][0] * vdown;
            acc[j][1] = acc[j][1] * a0 + sum[j][1] * vdown;
            acc[j][2] = acc[j][2] * a1 + sum[j][2] * vdown;
            acc[j][3] = acc[j][3] * a1 + sum[j][3] * vdown;
        }
        if(more){
            scales[nst][warp][0][lane] = k0;
            scales[nst][warp][1][lane] = k1;
            scales[nst][warp][2][lane] = v0;
            scales[nst][warp][3][lane] = v1;
        }
        __syncwarp();
    }

    float i0 = 1.0f / total[gid], i1 = 1.0f / total[gid + 8];
    float* ot = out + (u64)t * (QHEADS * 256) + (u64)kvh * MROWS * 256;
    #pragma unroll
    for(int j = 0; j < 8; j++){
        int d = dim0 + j * 8 + 2 * tig;
        if(gid < MROWS){
            ot[gid * 256 + d] = acc[j][0] * i0 * gateOf(qt[gid * 512 + 256 + d]);
            ot[gid * 256 + d + 1] = acc[j][1] * i0 * gateOf(qt[gid * 512 + 256 + d + 1]);
        }
        if(gid + 8 < MROWS){
            ot[(gid + 8) * 256 + d] = acc[j][2] * i1 * gateOf(qt[(gid + 8) * 512 + 256 + d]);
            ot[(gid + 8) * 256 + d + 1] = acc[j][3] * i1 * gateOf(qt[(gid + 8) * 512 + 256 + d + 1]);
        }
    }
}

extern "C" __global__ void __launch_bounds__(128, 3) attentionInt(const float* q, const i8* keys, const float* keyScales, const i8* values, const float* valueScales, const int* cells, const int* positions, float* out, int budget) {
    __shared__ __align__(16) i8 keyRows[4][32 * 64];
    __shared__ __align__(16) i8 valueRows[4][32 * 64];
    __shared__ __align__(16) float scales[4][4][32];
    __shared__ __align__(16) float part[4][16 * 32];
    __shared__ __align__(16) float prob[16 * 32];
    __shared__ float best[16], total[16], shrink[16], rowTop[16], rowDown[16], rowPeak[4][16];
    int t = blockIdx.y, kvh = blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    int gid = lane >> 2, tig = lane & 3, dim0 = warp * 64;
    int p = positions[t];
    bool all = p + 1 <= budget;
    int count = all ? p + 1 : budget;
    const int* list = cells + (u64)t * budget;
    const float* qt = q + (u64)t * (QHEADS * 512) + (u64)kvh * MROWS * 512;

    float4 qv[2][2][2];
    float peak[2] = {0.0f, 0.0f};
    #pragma unroll
    for(int h = 0; h < 2; h++){
        int row = gid + 8 * h;
        #pragma unroll
        for(int g = 0; g < 2; g++){
            #pragma unroll
            for(int s = 0; s < 2; s++){
                float4 v = row < MROWS ? *(const float4*)(qt + row * 512 + dim0 + g * 32 + s * 16 + tig * 4) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                qv[h][g][s] = v;
                peak[h] = fmaxf(peak[h], fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
            }
        }
        peak[h] = fmaxf(peak[h], __shfl_xor_sync(0xffffffff, peak[h], 1));
        peak[h] = fmaxf(peak[h], __shfl_xor_sync(0xffffffff, peak[h], 2));
    }
    if(tig == 0){
        rowPeak[warp][gid] = peak[0];
        rowPeak[warp][gid + 8] = peak[1];
    }
    if(tid < 16){
        best[tid] = -3.0e38f;
        total[tid] = 0.0f;
    }
    __syncthreads();
    if(tid < 16) rowDown[tid] = fmaxf(fmaxf(rowPeak[0][tid], rowPeak[1][tid]), fmaxf(rowPeak[2][tid], rowPeak[3][tid])) * (0.0625f * 1.4426950408889634f / 32767.0f);
    u32 qhi[2][4], qlo[2][4];
    #pragma unroll
    for(int h = 0; h < 2; h++){
        int row = gid + 8 * h;
        float top = fmaxf(fmaxf(rowPeak[0][row], rowPeak[1][row]), fmaxf(rowPeak[2][row], rowPeak[3][row]));
        float up = top > 0.0f ? 32767.0f / top : 0.0f;
        #pragma unroll
        for(int g = 0; g < 2; g++){
            #pragma unroll
            for(int s = 0; s < 2; s++){
                float4 v = qv[h][g][s];
                packWide(qhi[g][h + 2 * s], qlo[g][h + 2 * s], __fmaf_rn(v.x, up, 12582912.0f), __fmaf_rn(v.y, up, 12582912.0f), __fmaf_rn(v.z, up, 12582912.0f), __fmaf_rn(v.w, up, 12582912.0f), 0x5140);
            }
        }
    }

    int chunks = (count + 31) / 32;
    float k0, k1, v0, v1;
    auto cellAt = [&](int c) {
        return c < count
            ? all
                ? c
                : list[c]
            : -1;
    };
    auto issue = [&](i8* to, const i8* from, const float* marks, int cell, float& s0, float& s1) {
        #pragma unroll
        for(int j = 0; j < 4; j++){
            int at = lane + 32 * j, c = at >> 2, piece = at & 3;
            int source = __shfl_sync(0xffffffff, cell, c);
            u64 offset = source >= 0 ? (u64)source * KVW + kvh * 256 + dim0 + piece * 16 : 0;
            copyAsync(&to[swizzle(c, piece * 16)], from + offset, source >= 0);
        }
        u64 scaleAt = (u64)max(cell, 0) * (KVW / 32) + kvh * 8 + warp * 2;
        s0 = cell >= 0 ? marks[scaleAt] : 0.0f;
        s1 = cell >= 0 ? marks[scaleAt + 1] : 0.0f;
    };
    int cell = cellAt(lane), nextCell = cellAt(32 + lane);
    issue(keyRows[warp], keys, keyScales, cell, k0, k1);
    copyCommit();

    float acc[2][4][4];
    #pragma unroll
    for(int g = 0; g < 2; g++){
        #pragma unroll
        for(int j = 0; j < 4; j++) acc[g][j][0] = acc[g][j][1] = acc[g][j][2] = acc[g][j][3] = 0.0f;
    }
    int flip = tig >> 1;
    u32 order = flip ? 0x1504 : 0x5140;

    for(int ci = 0; ci < chunks; ci++){
        int c0 = ci * 32;
        bool more = ci + 1 < chunks;
        issue(valueRows[warp], values, valueScales, cell, v0, v1);
        copyCommit();
        copyWaitOne();
        scales[warp][0][lane] = k0;
        scales[warp][1][lane] = k1;
        __syncwarp();
        const i8* K = keyRows[warp];
        const i8* V = valueRows[warp];
        float* mine = part[warp];
        #pragma unroll
        for(int nt = 0; nt < 4; nt++){
            int c = nt * 8 + gid;
            float z[2][4];
            #pragma unroll
            for(int g = 0; g < 2; g++){
                u32 b0 = *(const u32*)(K + swizzle(c, g * 32 + tig * 4));
                u32 b1 = *(const u32*)(K + swizzle(c, g * 32 + 16 + tig * 4));
                wideStep<true>(z[g], qhi[g], qlo[g], b0, b1);
            }
            int e = nt * 8 + 2 * tig;
            float2 sa = *(const float2*)&scales[warp][0][e], sb = *(const float2*)&scales[warp][1][e];
            *(float2*)&mine[partAt(gid, e)] = make_float2(z[0][0] * sa.x + z[1][0] * sb.x, z[0][1] * sa.y + z[1][1] * sb.y);
            *(float2*)&mine[partAt(gid + 8, e)] = make_float2(z[0][2] * sa.x + z[1][2] * sb.x, z[0][3] * sa.y + z[1][3] * sb.y);
        }
        __syncthreads();

        {
            int r = tid >> 3, sub = tid & 7;
            float dn = rowDown[r];
            float4 s0 = *(const float4*)&part[0][partAt(r, sub * 4)];
            float4 s1 = *(const float4*)&part[1][partAt(r, sub * 4)];
            float4 s2 = *(const float4*)&part[2][partAt(r, sub * 4)];
            float4 s3 = *(const float4*)&part[3][partAt(r, sub * 4)];
            float y[4] = {((s0.x + s1.x) + s2.x) + s3.x, ((s0.y + s1.y) + s2.y) + s3.y, ((s0.z + s1.z) + s2.z) + s3.z, ((s0.w + s1.w) + s2.w) + s3.w};
            float x[4], m = -3.0e38f;
            #pragma unroll
            for(int j = 0; j < 4; j++){
                x[j] = c0 + sub * 4 + j < count
                    ? y[j] * dn
                    : -3.0e38f;
                m = fmaxf(m, x[j]);
            }
            for(int step = 1; step < 8; step <<= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, step));
            float old = best[r], fresh = fmaxf(old, m);
            float e[4], sum = 0.0f;
            #pragma unroll
            for(int j = 0; j < 4; j++){
                e[j] = exp2f(x[j] - fresh);
                sum += e[j];
            }
            *(float4*)&prob[probAt(r, sub * 4)] = make_float4(e[0], e[1], e[2], e[3]);
            for(int step = 1; step < 8; step <<= 1) sum += __shfl_xor_sync(0xffffffff, sum, step);
            __syncwarp();
            if(sub == 0){
                float k = exp2f(old - fresh);
                shrink[r] = k;
                rowTop[r] = exp2f(m - fresh);
                total[r] = total[r] * k + sum;
                best[r] = fresh;
            }
        }
        __syncthreads();

        if(more) issue(keyRows[warp], keys, keyScales, nextCell, k0, k1);
        copyCommit();
        copyWaitOne();
        scales[warp][2][lane] = v0;
        scales[warp][3][lane] = v1;
        __syncwarp();
        float vmax = fmaxf(v0, v1);
        for(int step = 16; step > 0; step >>= 1) vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffff, vmax, step));
        float t0 = rowTop[gid] * vmax, t1 = rowTop[gid + 8] * vmax;
        float u0 = t0 > 1.0e-30f ? __fdividef(65535.0f, t0) : 0.0f, u1 = t1 > 1.0e-30f ? __fdividef(65535.0f, t1) : 0.0f;
        float d0 = t0 * (1.0f / 65535.0f), d1 = t1 * (1.0f / 65535.0f);
        float4 pa = *(const float4*)&prob[probAt(gid, tig * 4)];
        float4 pb = *(const float4*)&prob[probAt(gid + 8, tig * 4)];
        float4 pc = *(const float4*)&prob[probAt(gid, 16 + tig * 4)];
        float4 pd = *(const float4*)&prob[probAt(gid + 8, 16 + tig * 4)];
        pa = make_float4(pa.x * u0, pa.y * u0, pa.z * u0, pa.w * u0);
        pc = make_float4(pc.x * u0, pc.y * u0, pc.z * u0, pc.w * u0);
        pb = make_float4(pb.x * u1, pb.y * u1, pb.z * u1, pb.w * u1);
        pd = make_float4(pd.x * u1, pd.y * u1, pd.z * u1, pd.w * u1);
        float a0 = shrink[gid], a1 = shrink[gid + 8];
        #pragma unroll
        for(int g = 0; g < 2; g++){
            float4 wa = *(const float4*)&scales[warp][2 + g][tig * 4];
            float4 wb = *(const float4*)&scales[warp][2 + g][16 + tig * 4];
            u32 hi[4], lo[4];
            packWide(hi[0], lo[0], __fmaf_rn(pa.x, wa.x, 12582912.0f), __fmaf_rn(pa.y, wa.y, 12582912.0f), __fmaf_rn(pa.z, wa.z, 12582912.0f), __fmaf_rn(pa.w, wa.w, 12582912.0f), order);
            packWide(hi[1], lo[1], __fmaf_rn(pb.x, wa.x, 12582912.0f), __fmaf_rn(pb.y, wa.y, 12582912.0f), __fmaf_rn(pb.z, wa.z, 12582912.0f), __fmaf_rn(pb.w, wa.w, 12582912.0f), order);
            packWide(hi[2], lo[2], __fmaf_rn(pc.x, wb.x, 12582912.0f), __fmaf_rn(pc.y, wb.y, 12582912.0f), __fmaf_rn(pc.z, wb.z, 12582912.0f), __fmaf_rn(pc.w, wb.w, 12582912.0f), order);
            packWide(hi[3], lo[3], __fmaf_rn(pd.x, wb.x, 12582912.0f), __fmaf_rn(pd.y, wb.y, 12582912.0f), __fmaf_rn(pd.z, wb.z, 12582912.0f), __fmaf_rn(pd.w, wb.w, 12582912.0f), order);
            u32 w[4], u[4];
            #pragma unroll
            for(int i = 0; i < 4; i++){
                w[i] = *(const u32*)(V + swizzle(tig * 4 + (i ^ flip), g * 32 + gid * 4));
                u[i] = *(const u32*)(V + swizzle(16 + tig * 4 + (i ^ flip), g * 32 + gid * 4));
            }
            u32 bw[4], bu[4];
            turnBytes(bw, w[0], w[1], w[2], w[3]);
            turnBytes(bu, u[0], u[1], u[2], u[3]);
            #pragma unroll
            for(int j = 0; j < 4; j++){
                float z[4];
                wideStep<false>(z, hi, lo, bw[j], bu[j]);
                acc[g][j][0] = acc[g][j][0] * a0 + z[0] * d0;
                acc[g][j][1] = acc[g][j][1] * a0 + z[1] * d0;
                acc[g][j][2] = acc[g][j][2] * a1 + z[2] * d1;
                acc[g][j][3] = acc[g][j][3] * a1 + z[3] * d1;
            }
        }
        cell = nextCell;
        nextCell = cellAt(c0 + 64 + lane);
        __syncwarp();
    }

    float i0 = 1.0f / total[gid], i1 = 1.0f / total[gid + 8];
    float* ot = out + (u64)t * (QHEADS * 256) + (u64)kvh * MROWS * 256;
    #pragma unroll
    for(int g = 0; g < 2; g++){
        int d = dim0 + g * 32 + tig * 8;
        #pragma unroll
        for(int h = 0; h < 2; h++){
            int row = gid + 8 * h;
            if(row >= MROWS) continue;
            float inv = h ? i1 : i0;
            const float* gate = qt + row * 512 + 256 + d;
            float4 ga = *(const float4*)gate, gb = *(const float4*)(gate + 4);
            *(float4*)(ot + row * 256 + d) = make_float4(acc[g][0][2 * h] * inv * gateOf(ga.x), acc[g][1][2 * h] * inv * gateOf(ga.y), acc[g][2][2 * h] * inv * gateOf(ga.z), acc[g][3][2 * h] * inv * gateOf(ga.w));
            *(float4*)(ot + row * 256 + d + 4) = make_float4(acc[g][0][2 * h + 1] * inv * gateOf(gb.x), acc[g][1][2 * h + 1] * inv * gateOf(gb.y), acc[g][2][2 * h + 1] * inv * gateOf(gb.z), acc[g][3][2 * h + 1] * inv * gateOf(gb.w));
        }
    }
}

extern "C" __global__ void __launch_bounds__(128, 3) attentionIntPacked(const float* q, const i8* keys, const float* keyScales, const i8* values, const float* valueScales, const int* positions, float* out, int n) {
    __shared__ __align__(16) i8 keyRows[4][32 * 64];
    __shared__ __align__(16) i8 valueRows[4][32 * 64];
    __shared__ __align__(16) float scales[4][4][32];
    __shared__ __align__(16) float part[4][16 * 32];
    __shared__ __align__(16) float prob[16 * 32];
    __shared__ float best[16], total[16], shrink[16], rowTop[16], rowDown[16], rowPeak[4][16];
    const int PACK = MROWS >= 16 ? 1 : 16 / MROWS;
    int t0 = blockIdx.y * PACK, kvh = blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    int gid = lane >> 2, tig = lane & 3, dim0 = warp * 64;
    __shared__ int rowCount[16];
    if(tid < 16) rowCount[tid] = tid < MROWS * PACK && t0 + tid / MROWS < n ? positions[t0 + tid / MROWS] + 1 : 0;
    __syncthreads();
    int count = 0;
    #pragma unroll
    for(int r = 0; r < 16; r++) count = max(count, rowCount[r]);
    auto rowQ = [&](int row) { return q + (u64)(t0 + row / MROWS) * (QHEADS * 512) + (u64)(kvh * MROWS + row % MROWS) * 512; };

    float4 qv[2][2][2];
    float peak[2] = {0.0f, 0.0f};
    #pragma unroll
    for(int h = 0; h < 2; h++){
        int row = gid + 8 * h;
        #pragma unroll
        for(int g = 0; g < 2; g++){
            #pragma unroll
            for(int s = 0; s < 2; s++){
                float4 v = rowCount[row] > 0 ? *(const float4*)(rowQ(row) + dim0 + g * 32 + s * 16 + tig * 4) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                qv[h][g][s] = v;
                peak[h] = fmaxf(peak[h], fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
            }
        }
        peak[h] = fmaxf(peak[h], __shfl_xor_sync(0xffffffff, peak[h], 1));
        peak[h] = fmaxf(peak[h], __shfl_xor_sync(0xffffffff, peak[h], 2));
    }
    if(tig == 0){
        rowPeak[warp][gid] = peak[0];
        rowPeak[warp][gid + 8] = peak[1];
    }
    if(tid < 16){
        best[tid] = -3.0e38f;
        total[tid] = 0.0f;
    }
    __syncthreads();
    if(tid < 16) rowDown[tid] = fmaxf(fmaxf(rowPeak[0][tid], rowPeak[1][tid]), fmaxf(rowPeak[2][tid], rowPeak[3][tid])) * (0.0625f * 1.4426950408889634f / 32767.0f);
    u32 qhi[2][4], qlo[2][4];
    #pragma unroll
    for(int h = 0; h < 2; h++){
        int row = gid + 8 * h;
        float top = fmaxf(fmaxf(rowPeak[0][row], rowPeak[1][row]), fmaxf(rowPeak[2][row], rowPeak[3][row]));
        float up = top > 0.0f ? 32767.0f / top : 0.0f;
        #pragma unroll
        for(int g = 0; g < 2; g++){
            #pragma unroll
            for(int s = 0; s < 2; s++){
                float4 v = qv[h][g][s];
                packWide(qhi[g][h + 2 * s], qlo[g][h + 2 * s], __fmaf_rn(v.x, up, 12582912.0f), __fmaf_rn(v.y, up, 12582912.0f), __fmaf_rn(v.z, up, 12582912.0f), __fmaf_rn(v.w, up, 12582912.0f), 0x5140);
            }
        }
    }

    int chunks = (count + 31) / 32;
    float k0, k1, v0, v1;
    auto cellAt = [&](int c) {
        return c < count ? c : -1;
    };
    auto issue = [&](i8* to, const i8* from, const float* marks, int cell, float& s0, float& s1) {
        #pragma unroll
        for(int j = 0; j < 4; j++){
            int at = lane + 32 * j, c = at >> 2, piece = at & 3;
            int source = __shfl_sync(0xffffffff, cell, c);
            u64 offset = source >= 0 ? (u64)source * KVW + kvh * 256 + dim0 + piece * 16 : 0;
            copyAsync(&to[swizzle(c, piece * 16)], from + offset, source >= 0);
        }
        u64 scaleAt = (u64)max(cell, 0) * (KVW / 32) + kvh * 8 + warp * 2;
        s0 = cell >= 0 ? marks[scaleAt] : 0.0f;
        s1 = cell >= 0 ? marks[scaleAt + 1] : 0.0f;
    };
    int cell = cellAt(lane), nextCell = cellAt(32 + lane);
    issue(keyRows[warp], keys, keyScales, cell, k0, k1);
    copyCommit();

    float acc[2][4][4];
    #pragma unroll
    for(int g = 0; g < 2; g++){
        #pragma unroll
        for(int j = 0; j < 4; j++) acc[g][j][0] = acc[g][j][1] = acc[g][j][2] = acc[g][j][3] = 0.0f;
    }
    int flip = tig >> 1;
    u32 order = flip ? 0x1504 : 0x5140;

    for(int ci = 0; ci < chunks; ci++){
        int c0 = ci * 32;
        bool more = ci + 1 < chunks;
        issue(valueRows[warp], values, valueScales, cell, v0, v1);
        copyCommit();
        copyWaitOne();
        scales[warp][0][lane] = k0;
        scales[warp][1][lane] = k1;
        __syncwarp();
        const i8* K = keyRows[warp];
        const i8* V = valueRows[warp];
        float* mine = part[warp];
        #pragma unroll
        for(int nt = 0; nt < 4; nt++){
            int c = nt * 8 + gid;
            float z[2][4];
            #pragma unroll
            for(int g = 0; g < 2; g++){
                u32 b0 = *(const u32*)(K + swizzle(c, g * 32 + tig * 4));
                u32 b1 = *(const u32*)(K + swizzle(c, g * 32 + 16 + tig * 4));
                wideStep<true>(z[g], qhi[g], qlo[g], b0, b1);
            }
            int e = nt * 8 + 2 * tig;
            float2 sa = *(const float2*)&scales[warp][0][e], sb = *(const float2*)&scales[warp][1][e];
            *(float2*)&mine[partAt(gid, e)] = make_float2(z[0][0] * sa.x + z[1][0] * sb.x, z[0][1] * sa.y + z[1][1] * sb.y);
            *(float2*)&mine[partAt(gid + 8, e)] = make_float2(z[0][2] * sa.x + z[1][2] * sb.x, z[0][3] * sa.y + z[1][3] * sb.y);
        }
        __syncthreads();

        {
            int r = tid >> 3, sub = tid & 7;
            float dn = rowDown[r];
            float4 s0 = *(const float4*)&part[0][partAt(r, sub * 4)];
            float4 s1 = *(const float4*)&part[1][partAt(r, sub * 4)];
            float4 s2 = *(const float4*)&part[2][partAt(r, sub * 4)];
            float4 s3 = *(const float4*)&part[3][partAt(r, sub * 4)];
            float y[4] = {((s0.x + s1.x) + s2.x) + s3.x, ((s0.y + s1.y) + s2.y) + s3.y, ((s0.z + s1.z) + s2.z) + s3.z, ((s0.w + s1.w) + s2.w) + s3.w};
            float x[4], m = -3.0e38f;
            #pragma unroll
            for(int j = 0; j < 4; j++){
                x[j] = c0 + sub * 4 + j < rowCount[r]
                    ? y[j] * dn
                    : -3.0e38f;
                m = fmaxf(m, x[j]);
            }
            for(int step = 1; step < 8; step <<= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, step));
            float old = best[r], fresh = fmaxf(old, m);
            float e[4], sum = 0.0f;
            #pragma unroll
            for(int j = 0; j < 4; j++){
                e[j] = exp2f(x[j] - fresh);
                sum += e[j];
            }
            *(float4*)&prob[probAt(r, sub * 4)] = make_float4(e[0], e[1], e[2], e[3]);
            for(int step = 1; step < 8; step <<= 1) sum += __shfl_xor_sync(0xffffffff, sum, step);
            __syncwarp();
            if(sub == 0){
                float k = exp2f(old - fresh);
                shrink[r] = k;
                rowTop[r] = exp2f(m - fresh);
                total[r] = total[r] * k + sum;
                best[r] = fresh;
            }
        }
        __syncthreads();

        if(more) issue(keyRows[warp], keys, keyScales, nextCell, k0, k1);
        copyCommit();
        copyWaitOne();
        scales[warp][2][lane] = v0;
        scales[warp][3][lane] = v1;
        __syncwarp();
        float vmax = fmaxf(v0, v1);
        for(int step = 16; step > 0; step >>= 1) vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffff, vmax, step));
        float t0 = rowTop[gid] * vmax, t1 = rowTop[gid + 8] * vmax;
        float u0 = t0 > 1.0e-30f ? __fdividef(65535.0f, t0) : 0.0f, u1 = t1 > 1.0e-30f ? __fdividef(65535.0f, t1) : 0.0f;
        float d0 = t0 * (1.0f / 65535.0f), d1 = t1 * (1.0f / 65535.0f);
        float4 pa = *(const float4*)&prob[probAt(gid, tig * 4)];
        float4 pb = *(const float4*)&prob[probAt(gid + 8, tig * 4)];
        float4 pc = *(const float4*)&prob[probAt(gid, 16 + tig * 4)];
        float4 pd = *(const float4*)&prob[probAt(gid + 8, 16 + tig * 4)];
        pa = make_float4(pa.x * u0, pa.y * u0, pa.z * u0, pa.w * u0);
        pc = make_float4(pc.x * u0, pc.y * u0, pc.z * u0, pc.w * u0);
        pb = make_float4(pb.x * u1, pb.y * u1, pb.z * u1, pb.w * u1);
        pd = make_float4(pd.x * u1, pd.y * u1, pd.z * u1, pd.w * u1);
        float a0 = shrink[gid], a1 = shrink[gid + 8];
        #pragma unroll
        for(int g = 0; g < 2; g++){
            float4 wa = *(const float4*)&scales[warp][2 + g][tig * 4];
            float4 wb = *(const float4*)&scales[warp][2 + g][16 + tig * 4];
            u32 hi[4], lo[4];
            packWide(hi[0], lo[0], __fmaf_rn(pa.x, wa.x, 12582912.0f), __fmaf_rn(pa.y, wa.y, 12582912.0f), __fmaf_rn(pa.z, wa.z, 12582912.0f), __fmaf_rn(pa.w, wa.w, 12582912.0f), order);
            packWide(hi[1], lo[1], __fmaf_rn(pb.x, wa.x, 12582912.0f), __fmaf_rn(pb.y, wa.y, 12582912.0f), __fmaf_rn(pb.z, wa.z, 12582912.0f), __fmaf_rn(pb.w, wa.w, 12582912.0f), order);
            packWide(hi[2], lo[2], __fmaf_rn(pc.x, wb.x, 12582912.0f), __fmaf_rn(pc.y, wb.y, 12582912.0f), __fmaf_rn(pc.z, wb.z, 12582912.0f), __fmaf_rn(pc.w, wb.w, 12582912.0f), order);
            packWide(hi[3], lo[3], __fmaf_rn(pd.x, wb.x, 12582912.0f), __fmaf_rn(pd.y, wb.y, 12582912.0f), __fmaf_rn(pd.z, wb.z, 12582912.0f), __fmaf_rn(pd.w, wb.w, 12582912.0f), order);
            u32 w[4], u[4];
            #pragma unroll
            for(int i = 0; i < 4; i++){
                w[i] = *(const u32*)(V + swizzle(tig * 4 + (i ^ flip), g * 32 + gid * 4));
                u[i] = *(const u32*)(V + swizzle(16 + tig * 4 + (i ^ flip), g * 32 + gid * 4));
            }
            u32 bw[4], bu[4];
            turnBytes(bw, w[0], w[1], w[2], w[3]);
            turnBytes(bu, u[0], u[1], u[2], u[3]);
            #pragma unroll
            for(int j = 0; j < 4; j++){
                float z[4];
                wideStep<false>(z, hi, lo, bw[j], bu[j]);
                acc[g][j][0] = acc[g][j][0] * a0 + z[0] * d0;
                acc[g][j][1] = acc[g][j][1] * a0 + z[1] * d0;
                acc[g][j][2] = acc[g][j][2] * a1 + z[2] * d1;
                acc[g][j][3] = acc[g][j][3] * a1 + z[3] * d1;
            }
        }
        cell = nextCell;
        nextCell = cellAt(c0 + 64 + lane);
        __syncwarp();
    }

    float i0 = 1.0f / total[gid], i1 = 1.0f / total[gid + 8];
    #pragma unroll
    for(int g = 0; g < 2; g++){
        int d = dim0 + g * 32 + tig * 8;
        #pragma unroll
        for(int h = 0; h < 2; h++){
            int row = gid + 8 * h;
            if(rowCount[row] == 0) continue;
            float* ot = out + (u64)(t0 + row / MROWS) * (QHEADS * 256) + (u64)(kvh * MROWS + row % MROWS) * 256;
            float inv = h ? i1 : i0;
            const float* gate = rowQ(row) + 256 + d;
            float4 ga = *(const float4*)gate, gb = *(const float4*)(gate + 4);
            *(float4*)(ot + d) = make_float4(acc[g][0][2 * h] * inv * gateOf(ga.x), acc[g][1][2 * h] * inv * gateOf(ga.y), acc[g][2][2 * h] * inv * gateOf(ga.z), acc[g][3][2 * h] * inv * gateOf(ga.w));
            *(float4*)(ot + d + 4) = make_float4(acc[g][0][2 * h + 1] * inv * gateOf(gb.x), acc[g][1][2 * h + 1] * inv * gateOf(gb.y), acc[g][2][2 * h + 1] * inv * gateOf(gb.z), acc[g][3][2 * h + 1] * inv * gateOf(gb.w));
        }
    }
}
#endif

extern "C" __global__ void attentionSplit(const float* q, const i8* keys, const float* keyScales, const i8* values, const float* valueScales, const int* cells, const int* positions, float* part, int budget, int splits) {
    __shared__ float query[256];
    __shared__ float score[4096];
    __shared__ float merge[8][256];
    __shared__ float scratch[32];
    int t = blockIdx.y / splits, s = blockIdx.y % splits, h = blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    int kvh = h / (QHEADS / KVHEADS);
    int p = positions[t];
    const float* qh = q + (u64)t * (QHEADS * 512) + h * 512;
    query[tid] = qh[tid];
    bool all = p + 1 <= budget;
    int count = all ? p + 1 : budget;
    int span = (count + splits - 1) / splits;
    int begin = min(count, s * span);
    int end = min(count, begin + span);
    const int* list = cells + (u64)t * budget;
    __syncthreads();

    float qv[8];
    #pragma unroll
    for(int k = 0; k < 8; k++) qv[k] = query[lane * 8 + k];
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float runBest = -3.0e38f;
    float total = 0.0f;
    for(int from = begin; from < end; from += 4096){
        int width = min(4096, end - from);
        for(int i = warp; i < width; i += 8){
            int cell = all ? from + i : list[from + i];
            uint2 raw = *(const uint2*)(keys + (u64)cell * KVW + kvh * 256 + lane * 8);
            u32 words[2] = {raw.x, raw.y};
            float d = 0.0f;
            #pragma unroll
            for(int k = 0; k < 2; k++){
                #pragma unroll
                for(int b = 0; b < 4; b++) d += qv[4 * k + b] * (float)(i8)(words[k] >> (8 * b));
            }
            d *= keyScales[(u64)cell * (KVW / 32) + kvh * 8 + (lane >> 2)];
            for(int step = 16; step > 0; step >>= 1) d += __shfl_xor_sync(0xffffffff, d, step);
            if(lane == 0) score[i] = d * 0.0625f;
        }
        __syncthreads();

        float best = runBest;
        for(int i = tid; i < width; i += blockDim.x) best = fmaxf(best, score[i]);
        best = blockMax(best, scratch);
        float shrink = __expf(runBest - best);
        float sum = 0.0f;
        for(int i = tid; i < width; i += blockDim.x){
            float e = __expf(score[i] - best);
            score[i] = e;
            sum += e;
        }
        total = total * shrink + blockSum(sum, scratch);
        runBest = best;
        __syncthreads();

        #pragma unroll
        for(int k = 0; k < 8; k++) acc[k] *= shrink;
        for(int i = warp; i < width; i += 8){
            int cell = all ? from + i : list[from + i];
            uint2 raw = *(const uint2*)(values + (u64)cell * KVW + kvh * 256 + lane * 8);
            u32 words[2] = {raw.x, raw.y};
            float w = score[i] * valueScales[(u64)cell * (KVW / 32) + kvh * 8 + (lane >> 2)];
            #pragma unroll
            for(int k = 0; k < 2; k++){
                #pragma unroll
                for(int b = 0; b < 4; b++) acc[4 * k + b] += w * (float)(i8)(words[k] >> (8 * b));
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int k = 0; k < 8; k++) merge[warp][lane * 8 + k] = acc[k];
    __syncthreads();
    float o = 0.0f;
    for(int w = 0; w < 8; w++) o += merge[w][tid];
    float* at = part + (((u64)t * QHEADS + h) * splits + s) * 258;
    at[2 + tid] = o;
    if(tid == 0){
        at[0] = runBest;
        at[1] = total;
    }
}

extern "C" __global__ void attentionMerge(const float* q, const float* part, float* out, int splits) {
    int t = blockIdx.y, h = blockIdx.x, tid = threadIdx.x;
    const float* at = part + ((u64)t * QHEADS + h) * splits * 258;
    float best = -3.0e38f;
    for(int s = 0; s < splits; s++) best = fmaxf(best, at[s * 258]);
    float total = 0.0f, o = 0.0f;
    for(int s = 0; s < splits; s++){
        float f = __expf(at[s * 258] - best);
        total += at[s * 258 + 1] * f;
        o += at[s * 258 + 2 + tid] * f;
    }
    const float* qh = q + (u64)t * (QHEADS * 512) + h * 512;
    out[(u64)t * (QHEADS * 256) + h * 256 + tid] = o / total * gateOf(qh[256 + tid]);
}

#define GROUP (QHEADS / KVHEADS)
#define TILE_TOKENS (16 / GROUP > 0 ? 16 / GROUP : 1)
#define TILE_ROWS (GROUP * TILE_TOKENS)
#define SCORE_STRIDE ((TILE_ROWS + 3) & ~3)

struct TileSpace {
    float qs[TILE_ROWS][4][68];
    int cells[64][68];
    float marks[64][8];
    float score[64][SCORE_STRIDE];
    float rowBest[TILE_ROWS];
    float rowTotal[TILE_ROWS];
    float rowShrink[TILE_ROWS];
    int rowPlace[TILE_ROWS];
};

__device__ __forceinline__ void unpackBytes(int word, float* out) {
    unsigned x = (unsigned)word ^ 0x80808080u;
    out[0] = __int_as_float(__byte_perm(x, 0x4B000000, 0x7440)) - 8388736.0f;
    out[1] = __int_as_float(__byte_perm(x, 0x4B000000, 0x7441)) - 8388736.0f;
    out[2] = __int_as_float(__byte_perm(x, 0x4B000000, 0x7442)) - 8388736.0f;
    out[3] = __int_as_float(__byte_perm(x, 0x4B000000, 0x7443)) - 8388736.0f;
}

__device__ __forceinline__ void tileFetch(int4 (&words)[4], float (&marks)[2], const i8* data, const float* scaleData, int from, int width, int kvh, int tid) {
    #pragma unroll
    for(int j = 0; j < 4; j++){
        int i = tid + 256 * j, c = i >> 4;
        words[j] = c < width ? ((const int4*)(data + (u64)(from + c) * KVW + kvh * 256))[i & 15] : make_int4(0, 0, 0, 0);
    }
    #pragma unroll
    for(int j = 0; j < 2; j++){
        int i = tid + 256 * j, c = i >> 3;
        marks[j] = c < width ? scaleData[(u64)(from + c) * (KVW / 32) + kvh * 8 + (i & 7)] : 0.0f;
    }
}

__device__ __forceinline__ void tilePut(TileSpace& s, const int4 (&words)[4], const float (&marks)[2], int tid) {
    #pragma unroll
    for(int j = 0; j < 4; j++){
        int i = tid + 256 * j;
        *(int4*)&s.cells[i >> 4][(i & 15) * 4] = words[j];
    }
    #pragma unroll
    for(int j = 0; j < 2; j++){
        int i = tid + 256 * j;
        s.marks[i >> 3][i & 7] = marks[j];
    }
}

__device__ __forceinline__ int tileAttend(TileSpace& s, float (&acc)[TILE_ROWS][4], const float* q, const i8* keys, const float* keyScales, const i8* values, const float* valueScales, const int* positions, int kvh, int first, int n, int split, int splits) {
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    for(int i = tid; i < TILE_ROWS * 256; i += 256){
        int r = i >> 8, d = i & 255;
        int t = first + r / GROUP;
        s.qs[r][d >> 6][d & 63] = t < n ? q[(u64)t * (QHEADS * 512) + (kvh * GROUP + r % GROUP) * 512 + d] : 0.0f;
    }
    if(tid < TILE_ROWS){
        int t = first + tid / GROUP;
        s.rowBest[tid] = -3.0e38f;
        s.rowTotal[tid] = 0.0f;
        s.rowShrink[tid] = 1.0f;
        s.rowPlace[tid] = t < n ? positions[t] : -1;
    }
    #pragma unroll
    for(int r = 0; r < TILE_ROWS; r++){
        #pragma unroll
        for(int e = 0; e < 4; e++) acc[r][e] = 0.0f;
    }
    __syncthreads();
    int live = min(TILE_ROWS, max(0, n - first) * GROUP);
    int last = -1;
    for(int r = 0; r < TILE_ROWS; r++) last = max(last, s.rowPlace[r]);
    int count = last + 1;
    int span = ((count + splits - 1) / splits + 63) & ~63;
    int begin = min(count, split * span);
    int end = min(count, begin + span);

    int quarter = lane >> 3, c = warp * 8 + (lane & 7);
    int cell4 = tid & 63, part4 = tid >> 6;
    int4 words[4];
    float marks[2];
    if(begin < end) tileFetch(words, marks, keys, keyScales, begin, min(64, end - begin), kvh, tid);
    for(int from = begin; from < end; from += 64){
        int width = min(64, end - from);
        tilePut(s, words, marks, tid);
        __syncthreads();
        tileFetch(words, marks, values, valueScales, from, width, kvh, tid);

        float dot[TILE_ROWS];
        #pragma unroll
        for(int r = 0; r < TILE_ROWS; r++) dot[r] = 0.0f;
        if(warp * 8 < width){
            #pragma unroll
            for(int j = 0; j < 4; j++){
                int4 w4 = *(const int4*)&s.cells[c][quarter * 16 + j * 4];
                float mark = s.marks[c][quarter * 2 + (j >> 1)];
                float k[16];
                unpackBytes(w4.x, k);
                unpackBytes(w4.y, k + 4);
                unpackBytes(w4.z, k + 8);
                unpackBytes(w4.w, k + 12);
                #pragma unroll
                for(int e = 0; e < 16; e++) k[e] *= mark;
                #pragma unroll
                for(int r = 0; r < TILE_ROWS; r++){
                    if(r >= live) continue;
                    const float4* qq = (const float4*)&s.qs[r][quarter][j * 16];
                    #pragma unroll
                    for(int d = 0; d < 4; d++){
                        float4 v = qq[d];
                        dot[r] += v.x * k[d * 4] + v.y * k[d * 4 + 1] + v.z * k[d * 4 + 2] + v.w * k[d * 4 + 3];
                    }
                }
            }
        }
        #pragma unroll
        for(int r = 0; r < TILE_ROWS; r++){
            dot[r] += __shfl_xor_sync(0xffffffff, dot[r], 8);
            dot[r] += __shfl_xor_sync(0xffffffff, dot[r], 16);
        }
        if(quarter == 0){
            #pragma unroll
            for(int r = 0; r < TILE_ROWS; r++){
                s.score[c][r] = c < width && from + c <= s.rowPlace[r]
                    ? dot[r] * 0.0625f
                    : -3.0e38f;
            }
        }
        __syncthreads();

        for(int r = warp; r < live; r += 8){
            float a = s.score[lane][r];
            float b = s.score[lane + 32][r];
            float best = fmaxf(a, b);
            for(int step = 16; step > 0; step >>= 1) best = fmaxf(best, __shfl_xor_sync(0xffffffff, best, step));
            float old = s.rowBest[r];
            float now = fmaxf(old, best);
            float ea = a > -1.0e38f ? __expf(a - now) : 0.0f;
            float eb = b > -1.0e38f ? __expf(b - now) : 0.0f;
            s.score[lane][r] = ea;
            s.score[lane + 32][r] = eb;
            float sum = ea + eb;
            for(int step = 16; step > 0; step >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, step);
            if(lane == 0){
                float shrink = old > -1.0e38f ? __expf(old - now) : 1.0f;
                s.rowShrink[r] = shrink;
                s.rowTotal[r] = s.rowTotal[r] * shrink + sum;
                s.rowBest[r] = now;
            }
        }
        __syncthreads();

        tilePut(s, words, marks, tid);
        __syncthreads();
        if(from + 64 < end) tileFetch(words, marks, keys, keyScales, from + 64, min(64, end - from - 64), kvh, tid);
        #pragma unroll
        for(int r = 0; r < TILE_ROWS; r++){
            if(r >= live) continue;
            float shrink = s.rowShrink[r];
            #pragma unroll
            for(int e = 0; e < 4; e++) acc[r][e] *= shrink;
        }
        int stop = min(16, width - part4 * 16);
        for(int k = 0; k < stop; k++){
            int cell = part4 * 16 + k;
            float v[4];
            unpackBytes(s.cells[cell][cell4], v);
            float mark = s.marks[cell][cell4 >> 3];
            #pragma unroll
            for(int e = 0; e < 4; e++) v[e] *= mark;
            #pragma unroll
            for(int r4 = 0; r4 < SCORE_STRIDE / 4; r4++){
                float4 p = *(const float4*)&s.score[cell][r4 * 4];
                float pr[4] = {p.x, p.y, p.z, p.w};
                #pragma unroll
                for(int e = 0; e < 4; e++){
                    int r = r4 * 4 + e;
                    if(r >= TILE_ROWS || r >= live) continue;
                    acc[r][0] += pr[e] * v[0];
                    acc[r][1] += pr[e] * v[1];
                    acc[r][2] += pr[e] * v[2];
                    acc[r][3] += pr[e] * v[3];
                }
            }
        }
        __syncthreads();
    }

    float* gather = &s.qs[0][0][0];
    for(int step = 0; step < 4; step++){
        if(part4 == step){
            #pragma unroll
            for(int r = 0; r < TILE_ROWS; r++){
                if(r >= live) continue;
                float4* at = (float4*)&gather[r * 256 + 4 * cell4];
                if(step > 0){
                    float4 old = *at;
                    acc[r][0] += old.x;
                    acc[r][1] += old.y;
                    acc[r][2] += old.z;
                    acc[r][3] += old.w;
                }
                if(step < 3) *at = make_float4(acc[r][0], acc[r][1], acc[r][2], acc[r][3]);
            }
        }
        __syncthreads();
    }
    return live;
}

extern "C" __global__ void __launch_bounds__(256, 2) attentionTiles(const float* q, const i8* keys, const float* keyScales, const i8* values, const float* valueScales, const int* positions, float* out, int n) {
    __shared__ __align__(16) TileSpace s;
    int kvh = blockIdx.x, first = blockIdx.y * TILE_TOKENS, tid = threadIdx.x;
    float acc[TILE_ROWS][4];
    int live = tileAttend(s, acc, q, keys, keyScales, values, valueScales, positions, kvh, first, n, 0, 1);
    if(tid >> 6 != 3) return;
    int cell4 = tid & 63;
    #pragma unroll
    for(int r = 0; r < TILE_ROWS; r++){
        if(r >= live) continue;
        int t = first + r / GROUP, h = kvh * GROUP + r % GROUP;
        const float* gate = q + (u64)t * (QHEADS * 512) + h * 512 + 256 + 4 * cell4;
        float total = s.rowTotal[r];
        *(float4*)&out[(u64)t * (QHEADS * 256) + h * 256 + 4 * cell4] = make_float4(acc[r][0] / total * gateOf(gate[0]), acc[r][1] / total * gateOf(gate[1]), acc[r][2] / total * gateOf(gate[2]), acc[r][3] / total * gateOf(gate[3]));
    }
}

extern "C" __global__ void shortRows(const float* logits, int count, int keep, float* out) {
    __shared__ float values[4096];
    __shared__ int ids[4096];
    const float* row = logits + (u64)blockIdx.x * count;
    int tid = threadIdx.x;
    float best[8];
    int where[8];
    #pragma unroll
    for(int j = 0; j < 8; j++){
        best[j] = -3.0e38f;
        where[j] = 0;
    }
    for(int i = tid; i < count; i += blockDim.x){
        float v = row[i];
        if(!(v > best[7])) continue;
        int j = 7;
        while(j > 0 && best[j - 1] < v){
            best[j] = best[j - 1];
            where[j] = where[j - 1];
            j--;
        }
        best[j] = v;
        where[j] = i;
    }
    #pragma unroll
    for(int j = 0; j < 8; j++){
        values[tid * 8 + j] = best[j];
        ids[tid * 8 + j] = where[j];
    }
    __syncthreads();
    for(int size = 2; size <= 4096; size <<= 1){
        for(int stride = size >> 1; stride > 0; stride >>= 1){
            for(int i = tid; i < 4096; i += blockDim.x){
                int partner = i ^ stride;
                if(partner <= i) continue;
                float a = values[i], b = values[partner];
                bool swap = (i & size) == 0 ? a < b : a > b;
                if(!swap) continue;
                values[i] = b;
                values[partner] = a;
                int t = ids[i];
                ids[i] = ids[partner];
                ids[partner] = t;
            }
            __syncthreads();
        }
    }
    float* into = out + (u64)blockIdx.x * 2 * keep;
    for(int j = tid; j < keep; j += blockDim.x){
        into[j] = values[j];
        into[keep + j] = __int_as_float(ids[j]);
    }
}

extern "C" __global__ void __launch_bounds__(256, 2) attentionSpread(const float* q, const i8* keys, const float* keyScales, const i8* values, const float* valueScales, const int* positions, float* part, int n, int splits) {
    __shared__ __align__(16) TileSpace s;
    int kvh = blockIdx.x, split = blockIdx.y % splits, first = (blockIdx.y / splits) * TILE_TOKENS, tid = threadIdx.x;
    float acc[TILE_ROWS][4];
    int live = tileAttend(s, acc, q, keys, keyScales, values, valueScales, positions, kvh, first, n, split, splits);
    if(tid >> 6 != 3) return;
    int cell4 = tid & 63;
    #pragma unroll
    for(int r = 0; r < TILE_ROWS; r++){
        if(r >= live) continue;
        int t = first + r / GROUP, h = kvh * GROUP + r % GROUP;
        float* at = part + (((u64)t * QHEADS + h) * splits + split) * 258;
        at[2 + 4 * cell4] = acc[r][0];
        at[3 + 4 * cell4] = acc[r][1];
        at[4 + 4 * cell4] = acc[r][2];
        at[5 + 4 * cell4] = acc[r][3];
        if(cell4 == 0){
            at[0] = s.rowBest[r];
            at[1] = s.rowTotal[r];
        }
    }
}

extern "C" __global__ void pleGate(const float* key, const float* value, const float* R, const float* normKey, const float* normQuery, const float* normConv, float* gated, float* normd) {
    __shared__ float scratch[32];
    int c = blockIdx.x, t = blockIdx.y, tid = threadIdx.x;
    u64 at = (u64)t * HCW + c * HIDDEN;
    const float* kr = key + at;
    const float* qr = R + at;
    const float* vr = value + (u64)t * HIDDEN;
    const float* nk = normKey + c * HIDDEN;
    const float* nq = normQuery + c * HIDDEN;
    const float* nc = normConv + c * HIDDEN;

    float sk = 0.0f, sq = 0.0f;
    for(int d = tid; d < HIDDEN; d += blockDim.x){
        sk += kr[d] * kr[d];
        sq += qr[d] * qr[d];
    }
    float ik = rsqrtf(blockSum(sk, scratch) / (float)HIDDEN + 1e-6f);
    float iq = rsqrtf(blockSum(sq, scratch) / (float)HIDDEN + 1e-6f);
    float dot = 0.0f;
    for(int d = tid; d < HIDDEN; d += blockDim.x) dot += (kr[d] * ik * nk[d]) * (qr[d] * iq * nq[d]);
    float s = blockSum(dot, scratch) / 50.596443f;
    float sign = s > 0.0f
        ? 1.0f
        : s < 0.0f
            ? -1.0f
            : 0.0f;
    float g = gateOf(sign * sqrtf(fmaxf(fabsf(s), 1e-6f)));

    float sg = 0.0f;
    for(int d = tid; d < HIDDEN; d += blockDim.x){
        float v = vr[d] * g;
        gated[at + d] = v;
        sg += v * v;
    }
    float inv = rsqrtf(blockSum(sg, scratch) / (float)HIDDEN + 1e-6f);
    for(int d = tid; d < HIDDEN; d += blockDim.x) normd[at + d] = gated[at + d] * inv * nc[d];
}

extern "C" __global__ void pleConv(const float* gated, const float* normd, const float* history, const float* weight, float* R, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n * HCW) return;
    int t = i / HCW, ch = i % HCW;
    const float* w = weight + ch * 4;
    float conv = 0.0f;
    for(int k = 0; k < 3; k++){
        int at = t + 3 * k;
        float v = at < 9
            ? history[at * HCW + ch]
            : normd[(u64)(at - 9) * HCW + ch];
        conv += w[k] * v;
    }
    conv += w[3] * normd[i];
    R[i] += gated[i] + conv * gateOf(conv);
}

extern "C" __global__ void pleHistory(const float* normd, float* history, int n) {
    int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if(ch >= HCW) return;
    for(int k = 0; k < 9; k++){
        int at = n + k;
        history[k * HCW + ch] = at < 9
            ? history[at * HCW + ch]
            : normd[(u64)(at - 9) * HCW + ch];
    }
}
)";
