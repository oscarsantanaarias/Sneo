#pragma once

// lectura del prompt en las rtx 50 (sm_120) con sus tensor cores fp4: gate y up de los expertos nvfp4
// un bloque nvfp4 (64 valores con 4 escalas ue4m3) es justo el k de una mma m16n8k64 con escala por cada 16: los pesos van tal cual
// las activaciones se pasan a fp4 con una escala ue4m3 por cada 16 y una escala por token para que entren en su rango
// en otras tarjetas (solo para probar con SNEO_FP4_TENSOR=1) la mma se imita con shuffles en el mismo orden de registros
const char* fp4KernelSource = R"(
typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1200 && defined(__CUDA_ARCH_FEAT_SM120_ALL)
#define FP4_NATIVE 1
#else
#define FP4_NATIVE 0
#endif

// e2m1: 2 bits de exponente y 1 de mantisa, sin exponente es 0 o 0.5
__device__ __forceinline__ float fp4Real(u32 code) {
    int e = (code >> 1) & 3, m = code & 1;
    float v = e == 0
        ? 0.5f * (float)m
        : (1.0f + 0.5f * (float)m) * (float)(1 << (e - 1));
    return code & 8 ? -v : v;
}

__device__ __forceinline__ float ue4m3Real(u32 x) {
    int e = (x >> 3) & 15, m = x & 7;
    return e == 0
        ? (float)m * 0.001953125f
        : (float)(8 + m) * __uint_as_float((u32)(e + 117) << 23);
}

// la escala mas chica que deja todo el bloque dentro de 6: se redondea para arriba asi nada se recorta
__device__ __forceinline__ u32 ue4m3Code(float v) {
    if(!(v > 0.0f)) return 0;
    if(v >= 448.0f) return 0x7E;
    int e = 0;
    float f = frexpf(v, &e);
    int ex = e - 1;
    if(ex < -6){
        int m = (int)ceilf(v * 512.0f);
        return (u32)min(m, 8);
    }
    int m = (int)ceilf((2.0f * f - 1.0f) * 8.0f);
    if(m == 8){
        m = 0;
        ex++;
    }
    u32 code = (u32)(((ex + 7) << 3) | m);
    return code > 0x7Eu ? 0x7Eu : code;
}

__device__ __forceinline__ u32 e2m1Code(float v) {
    float a = fabsf(v);
    u32 m = a < 0.25f ? 0u : a < 0.75f ? 1u : a < 1.25f ? 2u : a < 1.75f ? 3u : a < 2.5f ? 4u : a < 3.5f ? 5u : a < 5.0f ? 6u : 7u;
    return v < 0.0f && m ? (m | 8u) : m;
}

// los nibbles bajos de cada byte quedan seguidos en 16 bits
__device__ __forceinline__ u32 squeeze(u32 x) {
    x &= 0x0F0F0F0Fu;
    x = (x | (x >> 4)) & 0x00FF00FFu;
    return (x | (x >> 8)) & 0x0000FFFFu;
}

// registro de a para la columna which*8: sub bloque which/2, nibbles bajos (valores 0..7) o altos (8..15) de sus 8 bytes
__device__ __forceinline__ u32 fp4Nibbles(const u8* block, int which) {
    const u32* q = (const u32*)(block + 4 + 8 * (which >> 1));
    int shift = 4 * (which & 1);
    return squeeze(q[0] >> shift) | (squeeze(q[1] >> shift) << 16);
}

// d += (a por su escala) por (b por su escala): a de 16x64 fila mayor, b de 64x8, una escala ue4m3 por cada 16 de k
// a: fila gid en a[0] y a[2], gid + 8 en a[1] y a[3], columnas tig*8 (y +32); b: columna gid, filas tig*8 (y +32)
// escalas de a: fila gid en el hilo 0 del quad y gid + 8 en el hilo 1; de b: columna gid en el hilo 0 del quad
__device__ __forceinline__ void fp4Mma(float* d, const u32* a, const u32* b, u32 sa, u32 sb) {
#if FP4_NATIVE
    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sa), "h"((u16)0), "h"((u16)0), "r"(sb), "h"((u16)0), "h"((u16)0));
#else
    int lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    u32 scaleLow = __shfl_sync(0xffffffff, sa, gid * 4);
    u32 scaleHigh = __shfl_sync(0xffffffff, sa, gid * 4 + 1);
    u32 scaleB0 = __shfl_sync(0xffffffff, sb, (2 * tig) * 4);
    u32 scaleB1 = __shfl_sync(0xffffffff, sb, (2 * tig + 1) * 4);
    for(int k = 0; k < 64; k++){
        int side = k >> 5, owner = (k & 31) >> 3, nib = 4 * (k & 7), chunk = 8 * (k >> 4);
        u32 lowRow = __shfl_sync(0xffffffff, side ? a[2] : a[0], gid * 4 + owner);
        u32 highRow = __shfl_sync(0xffffffff, side ? a[3] : a[1], gid * 4 + owner);
        u32 col0 = __shfl_sync(0xffffffff, side ? b[1] : b[0], (2 * tig) * 4 + owner);
        u32 col1 = __shfl_sync(0xffffffff, side ? b[1] : b[0], (2 * tig + 1) * 4 + owner);
        float low = fp4Real((lowRow >> nib) & 15) * ue4m3Real((scaleLow >> chunk) & 255);
        float high = fp4Real((highRow >> nib) & 15) * ue4m3Real((scaleHigh >> chunk) & 255);
        float b0 = fp4Real((col0 >> nib) & 15) * ue4m3Real((scaleB0 >> chunk) & 255);
        float b1 = fp4Real((col1 >> nib) & 15) * ue4m3Real((scaleB1 >> chunk) & 255);
        acc[0] += low * b0;
        acc[1] += low * b1;
        acc[2] += high * b0;
        acc[3] += high * b1;
    }
    #pragma unroll
    for(int i = 0; i < 4; i++) d[i] += acc[i];
#endif
}

// activaciones a fp4: escala por token, escala ue4m3 por cada 16 y codigos e2m1 seguidos (valor k en los bits 4*(k%8) de la palabra k/8)
extern "C" __global__ void fp4Pack(const float* x, u32* codes, u8* scales, float* tokenScale, int cols) {
    __shared__ float peak[8];
    int t = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const float* row = x + (u64)t * cols;
    float top = 0.0f;
    for(int i = tid; i < cols; i += blockDim.x) top = fmaxf(top, fabsf(row[i]));
    for(int step = 16; step > 0; step >>= 1) top = fmaxf(top, __shfl_xor_sync(0xffffffff, top, step));
    if(lane == 0) peak[warp] = top;
    __syncthreads();
    top = 0.0f;
    for(int w = 0; w < (int)(blockDim.x >> 5); w++) top = fmaxf(top, peak[w]);
    float global = top > 0.0f ? top / (6.0f * 448.0f) : 1.0f;
    if(tid == 0) tokenScale[t] = global;
    for(int b = tid; b < cols / 16; b += blockDim.x){
        const float* v = row + 16 * b;
        float m = 0.0f;
        #pragma unroll
        for(int j = 0; j < 16; j++) m = fmaxf(m, fabsf(v[j]));
        u32 code = ue4m3Code(m / 6.0f / global);
        float s = ue4m3Real(code) * global;
        float inverse = s > 0.0f ? 1.0f / s : 0.0f;
        u32 low = 0, high = 0;
        #pragma unroll
        for(int j = 0; j < 8; j++){
            low |= e2m1Code(v[j] * inverse) << (4 * j);
            high |= e2m1Code(v[8 + j] * inverse) << (4 * j);
        }
        codes[(u64)t * (cols / 8) + 2 * b] = low;
        codes[(u64)t * (cols / 8) + 2 * b + 1] = high;
        scales[(u64)t * (cols / 16) + b] = (u8)code;
    }
}

// gate o up de los expertos nvfp4: 4 warps de 16 filas por 64 tokens del grupo, k de a 64 (un bloque nvfp4 por fila)
extern "C" __global__ void __launch_bounds__(128) fp4Rows(const u8* weights, u64 rowBytes, u64 expertBytes, const int* groupSlot, const int* groupStart, const int* groupCount, const int* inputIndex, const u32* codes, const u32* scales, const float* tokenScale, float* output, int rows, int cols, int tiles) {
    __shared__ int source[64];
    int group = blockIdx.y / tiles, tile = blockIdx.y % tiles;
    int slot = groupSlot[group], start = groupStart[group], total = groupCount[group];
    int first = tile * 64;
    if(first >= total) return;
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2, tig = lane & 3;
    if(tid < 64){
        int j = first + tid;
        source[tid] = j >= total
            ? -1
            : inputIndex
                ? inputIndex[start + j]
                : start + j;
    }
    __syncthreads();

    int r0 = blockIdx.x * 64 + warp * 16 + gid, r1 = r0 + 8;
    const u8* base = weights + expertBytes * slot;
    const u8* w0 = r0 < rows ? base + (u64)r0 * rowBytes : nullptr;
    const u8* w1 = r1 < rows ? base + (u64)r1 * rowBytes : nullptr;
    int words = cols / 8, scaleWords = cols / 64;
    float acc[8][4];
    #pragma unroll
    for(int nt = 0; nt < 8; nt++) acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.0f;

    for(int kc = 0; kc < cols / 64; kc++){
        u32 a[4] = {0, 0, 0, 0}, sa = 0;
        if(w0){
            a[0] = fp4Nibbles(w0 + 36 * kc, tig);
            a[2] = fp4Nibbles(w0 + 36 * kc, tig + 4);
            if(tig == 0) sa = *(const u32*)(w0 + 36 * kc);
        }
        if(w1){
            a[1] = fp4Nibbles(w1 + 36 * kc, tig);
            a[3] = fp4Nibbles(w1 + 36 * kc, tig + 4);
            if(tig == 1) sa = *(const u32*)(w1 + 36 * kc);
        }
        #pragma unroll
        for(int nt = 0; nt < 8; nt++){
            int s = source[nt * 8 + gid];
            u32 b[2] = {0, 0}, sb = 0;
            if(s >= 0){
                const u32* c = codes + (u64)s * words + kc * 8;
                b[0] = c[tig];
                b[1] = c[4 + tig];
                if(tig == 0) sb = scales[(u64)s * scaleWords + kc];
            }
            fp4Mma(acc[nt], a, b, sa, sb);
        }
    }

    #pragma unroll
    for(int nt = 0; nt < 8; nt++){
        #pragma unroll
        for(int c = 0; c < 2; c++){
            int jj = nt * 8 + 2 * tig + c, j = first + jj;
            if(j >= total) continue;
            float g = tokenScale[source[jj]];
            if(r0 < rows) output[(u64)(start + j) * rows + r0] = acc[nt][c] * g;
            if(r1 < rows) output[(u64)(start + j) * rows + r1] = acc[nt][2 + c] * g;
        }
    }
}
)";
