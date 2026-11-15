#pragma once
#include "cuda.h"

struct GridTables {
    uint64_t iq2xxs;
    uint64_t iq2xs;
    uint64_t iq2s;
    uint64_t iq3xxs;
    uint64_t iq3s;
    uint64_t signs;
    uint64_t iq4nl;
    uint64_t iq1s;
};

struct RowSet {
    int count;
    int type[6];
    int rows[6];
    int start[7];
    uint64_t weights[6];
    uint64_t stride[6];
    uint64_t output[6];
};

struct StagedSet {
    int count;
    int per;
    int type[4];
    int shift[4];
    uint64_t weights[4];
    uint64_t rowBytes[4];
    uint64_t expertBytes[4];
    uint64_t slot[4];
    uint64_t used[4];
    uint64_t input[4];
    uint64_t output[4];
    uint64_t team[4];
};

const char* quantKernelSource = R"(
typedef unsigned char u8;
typedef signed char i8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;

struct GridTables {
    const u64* iq2xxs;
    const u64* iq2xs;
    const u64* iq2s;
    const u32* iq3xxs;
    const u32* iq3s;
    const u8* signs;
    const i8* iq4nl;
    const u64* iq1s;
};

struct RowSet {
    int count;
    int type[6];
    int rows[6];
    int start[7];
    u64 weights[6];
    u64 stride[6];
    u64 output[6];
};

struct StagedSet {
    int count;
    int per;
    int type[4];
    int shift[4];
    u64 weights[4];
    u64 rowBytes[4];
    u64 expertBytes[4];
    u64 slot[4];
    u64 used[4];
    u64 input[4];
    u64 output[4];
    u64 team[4];
};

__device__ __forceinline__ float half(u16 bits) {
#ifdef __HIP__
    return (float)__builtin_bit_cast(_Float16, bits);
#else
    float out;
    asm("cvt.f32.f16 %0, %1;" : "=f"(out) : "h"(bits));
    return out;
#endif
}

__device__ __forceinline__ u16 word(const u8* p) {
    return (u16)p[0] | ((u16)p[1] << 8);
}

__device__ __forceinline__ void scaleMin(int j, const u8* s, int& sc, int& m) {
    if(j < 4){
        sc = s[j] & 63;
        m = s[j + 4] & 63;
        return;
    }
    sc = (s[j + 4] & 15) | ((s[j - 4] >> 6) << 4);
    m = (s[j + 4] >> 4) | ((s[j] >> 6) << 4);
}

__device__ float groupGrid8(u64 grid, u8 signs, float scale, const float* x) {
    float sum = 0.0f;
    #pragma unroll
    for(int j = 0; j < 8; j++){
        float v = (float)((grid >> (8 * j)) & 255);
        sum += ((signs >> j) & 1 ? -v : v) * x[j];
    }
    return sum * scale;
}

__device__ float groupGrid4(u32 grid, u8 signs, int shift, float scale, const float* x) {
    float sum = 0.0f;
    #pragma unroll
    for(int j = 0; j < 4; j++){
        float v = (float)((grid >> (8 * j)) & 255);
        sum += ((signs >> (j + shift)) & 1 ? -v : v) * x[j];
    }
    return sum * scale;
}

__device__ float dotGroup(int type, const u8* block, int g, const float* x, const GridTables& t) {
    if(type == 0){
        const float* w = (const float*)block + 32 * g;
        float sum = 0.0f;
        for(int l = 0; l < 32; l++) sum += w[l] * x[l];
        return sum;
    }

    if(type == 1 || type == 30){
        const u16* w = (const u16*)block + 32 * g;
        float sum = 0.0f;
        for(int l = 0; l < 32; l++){
            float v = type == 1 ? half(w[l]) : __int_as_float((u32)w[l] << 16);
            sum += v * x[l];
        }
        return sum;
    }

    if(type == 12 || type == 13){
        float d = half(word(block));
        float dmin = half(word(block + 2));
        const u8* scales = block + 4;
        const u8* qh = block + 16;
        const u8* qs = type == 12 ? block + 16 : block + 48;
        int sc, m;
        scaleMin(g, scales, sc, m);
        float dl = d * sc;
        float ml = dmin * m;
        const u8* q = qs + 32 * (g / 2);
        int shift = (g & 1) * 4;
        float sum = 0.0f, total = 0.0f;
        for(int l = 0; l < 32; l++){
            int v = (q[l] >> shift) & 15;
            if(type == 13 && ((qh[l] >> g) & 1)) v += 16;
            sum += v * x[l];
            total += x[l];
        }
        return dl * sum - ml * total;
    }

    if(type == 14){
        const u8* ql = block + 64 * (g / 4);
        const u8* qh = block + 128 + 32 * (g / 4);
        const i8* sc = (const i8*)(block + 192) + 8 * (g / 4);
        float d = half(word(block + 208));
        int k = g % 4;
        float sum = 0.0f;
        for(int l = 0; l < 32; l++){
            int low = k == 0
                ? ql[l] & 15
                : k == 1
                    ? ql[l + 32] & 15
                    : k == 2
                        ? ql[l] >> 4
                        : ql[l + 32] >> 4;
            int high = (qh[l] >> (2 * k)) & 3;
            sum += (float)sc[2 * k + l / 16] * (float)((low | (high << 4)) - 32) * x[l];
        }
        return d * sum;
    }

    if(type == 20 || type == 23){
        float d = half(word(block));
        const u8* qs = type == 20 ? block + 2 : block + 8 + 16 * g;
        if(type == 23){
            int low = (block[4 + g / 2] >> (4 * (g % 2))) & 15;
            int high = (word(block + 2) >> (2 * g)) & 3;
            d *= (float)((low | (high << 4)) - 32);
        }
        float sum = 0.0f;
        for(int j = 0; j < 16; j++){
            sum += (float)t.iq4nl[qs[j] & 15] * x[j];
            sum += (float)t.iq4nl[qs[j] >> 4] * x[j + 16];
        }
        return d * sum;
    }

    if(type == 42){
        float d = half(word(block));
        const u8* qs = block + 2 + 8 * g;
        float sum = 0.0f;
        for(int j = 0; j < 32; j++){
            int code = (qs[j / 4] >> (2 * (j % 4))) & 3;
            sum += (float)(code - 1) * x[j];
        }
        return d * sum;
    }

    if(type == 16){
        float d = half(word(block));
        const u8* q = block + 2 + 8 * g;
        u32 aux = (u32)q[4] | ((u32)q[5] << 8) | ((u32)q[6] << 16) | ((u32)q[7] << 24);
        float dl = d * (0.5f + (float)(aux >> 28)) * 0.25f;
        float sum = 0.0f;
        for(int l = 0; l < 4; l++) sum += groupGrid8(t.iq2xxs[q[l]], t.signs[(aux >> (7 * l)) & 127], 1.0f, x + 8 * l);
        return dl * sum;
    }

    if(type == 17){
        float d = half(word(block));
        const u8* qs = block + 2 + 8 * g;
        u8 scale = block[66 + g];
        float sum = 0.0f;
        for(int l = 0; l < 4; l++){
            u16 q = word(qs + 2 * l);
            float dl = d * (0.5f + (float)((scale >> (4 * (l / 2))) & 15)) * 0.25f;
            sum += groupGrid8(t.iq2xs[q & 511], t.signs[q >> 9], dl, x + 8 * l);
        }
        return sum;
    }

    if(type == 22){
        float d = half(word(block));
        const u8* qs = block + 2 + 4 * g;
        const u8* signs = block + 34 + 4 * g;
        u8 qh = block[66 + g];
        u8 scale = block[74 + g];
        float sum = 0.0f;
        for(int l = 0; l < 4; l++){
            int index = qs[l] | ((qh << (8 - 2 * l)) & 0x300);
            float dl = d * (0.5f + (float)((scale >> (4 * (l / 2))) & 15)) * 0.25f;
            sum += groupGrid8(t.iq2s[index], signs[l], dl, x + 8 * l);
        }
        return sum;
    }

    if(type == 18){
        float d = half(word(block));
        const u8* qs = block + 2 + 8 * g;
        const u8* w = block + 66 + 4 * g;
        u32 aux = (u32)w[0] | ((u32)w[1] << 8) | ((u32)w[2] << 16) | ((u32)w[3] << 24);
        float dl = d * (0.5f + (float)(aux >> 28)) * 0.5f;
        float sum = 0.0f;
        for(int l = 0; l < 4; l++){
            u8 signs = t.signs[(aux >> (7 * l)) & 127];
            sum += groupGrid4(t.iq3xxs[qs[2 * l]], signs, 0, 1.0f, x + 8 * l);
            sum += groupGrid4(t.iq3xxs[qs[2 * l + 1]], signs, 4, 1.0f, x + 8 * l + 4);
        }
        return dl * sum;
    }

    if(type == 21){
        float d = half(word(block));
        const u8* qs = block + 2 + 8 * g;
        u8 qh = block[66 + g];
        const u8* signs = block + 74 + 4 * g;
        float dl = d * (float)(1 + 2 * ((block[106 + g / 2] >> (4 * (g % 2))) & 15));
        float sum = 0.0f;
        for(int l = 0; l < 4; l++){
            u32 first = t.iq3s[qs[2 * l] | ((qh << (8 - 2 * l)) & 256)];
            u32 second = t.iq3s[qs[2 * l + 1] | ((qh << (7 - 2 * l)) & 256)];
            sum += groupGrid4(first, signs[l], 0, 1.0f, x + 8 * l);
            sum += groupGrid4(second, signs[l], 4, 1.0f, x + 8 * l + 4);
        }
        return dl * sum;
    }

    if(type == 29){
        u16 bits = (u16)((word(block + 48) >> 12) | ((word(block + 50) >> 8) & 0x00F0) | ((word(block + 52) >> 4) & 0x0F00) | (word(block + 54) & 0xF000));
        int sc = word(block + 48 + 2 * (g / 2)) >> (6 * (g % 2));
        float sum = 0.0f;
        for(int l = 0; l < 4; l++){
            int h = block[32 + 2 * g + l / 2] >> (4 * (l % 2));
            u64 grid = t.iq1s[block[4 * g + l] | ((h & 7) << 8)];
            float shift = h & 8 ? -1.0f : 1.0f;
            float part = 0.0f;
            for(int j = 0; j < 8; j++) part += ((float)(i8)(grid >> (8 * j)) + shift) * x[8 * l + j];
            sum += part * (float)(2 * ((sc >> (3 * (l / 2))) & 7) + 1);
        }
        return half(bits) * 0.125f * sum;
    }

    if(type == 8){
        float d = half(word(block));
        float sum = 0.0f;
        for(int l = 0; l < 32; l++) sum += (float)(i8)block[2 + l] * x[l];
        return d * sum;
    }

    if(type == 7){
        float d = half(word(block));
        float m = half(word(block + 2));
        u32 qh = (u32)block[4] | ((u32)block[5] << 8) | ((u32)block[6] << 16) | ((u32)block[7] << 24);
        float sum = 0.0f, total = 0.0f;
        for(int l = 0; l < 16; l++){
            int lo = (block[8 + l] & 15) | (((qh >> l) & 1) << 4);
            int hi = (block[8 + l] >> 4) | (((qh >> (l + 16)) & 1) << 4);
            sum += (float)lo * x[l] + (float)hi * x[l + 16];
            total += x[l] + x[l + 16];
        }
        return d * sum + m * total;
    }

    return 0.0f;
}

__device__ __forceinline__ int groupsPerBlock(int type) {
    return type == 0 || type == 1 || type == 30 || type == 20 || type == 7 || type == 8 ? 1 : type == 42 ? 2 : 8;
}

__device__ __forceinline__ int blockBytes(int type) {
    switch(type){
        case 0: return 128;
        case 1: return 64;
        case 30: return 64;
        case 7: return 24;
        case 8: return 34;
        case 12: return 144;
        case 13: return 176;
        case 14: return 210;
        case 16: return 66;
        case 17: return 74;
        case 18: return 98;
        case 20: return 18;
        case 21: return 110;
        case 22: return 82;
        case 23: return 136;
        case 29: return 56;
        case 42: return 18;
    }
    return 0;
}

__device__ __forceinline__ u32 load32(const u8* p) {
    return (u32)p[0] | ((u32)p[1] << 8) | ((u32)p[2] << 16) | ((u32)p[3] << 24);
}

__device__ __forceinline__ u32 signMask(u32 bits) {
    return ((bits & 1) * 0xFFu) | (((bits >> 1) & 1) * 0xFF00u) | (((bits >> 2) & 1) * 0xFF0000u) | (((bits >> 3) & 1) * 0xFF000000u);
}

__device__ __forceinline__ int signed4(u32 magnitudes, u32 bits) {
    u32 mask = (((bits & 15u) * 0x204081u) & 0x01010101u) * 0xFFu;
    return (int)((magnitudes ^ mask) + (mask & 0x01010101u));
}

__device__ __forceinline__ int table4(const i8* table, u32 nibbles) {
    return (int)((u32)(u8)table[nibbles & 15] | ((u32)(u8)table[(nibbles >> 8) & 15] << 8) | ((u32)(u8)table[(nibbles >> 16) & 15] << 16) | ((u32)(u8)table[(nibbles >> 24) & 15] << 24));
}

__device__ float dotGroupQ8(int type, const u8* block, int g, const int* q, float dx, int qsum, const GridTables& t) {
    if(type == 12 || type == 13){
        float d = half(word(block));
        float dmin = half(word(block + 2));
        int sc, m;
        scaleMin(g, block + 4, sc, m);
        const u8* qs = (type == 12 ? block + 16 : block + 48) + 32 * (g / 2);
        const u8* qh = block + 16;
        int shift = (g & 1) * 4;
        int sum = 0;
        #pragma unroll
        for(int i = 0; i < 8; i++){
            u32 v = (load32(qs + 4 * i) >> shift) & 0x0F0F0F0Fu;
            if(type == 13) v |= ((load32(qh + 4 * i) >> g) & 0x01010101u) << 4;
            sum = __dp4a((int)v, q[i], sum);
        }
        return dx * (d * sc * sum - dmin * m * qsum);
    }

    if(type == 14){
        const u8* ql = block + 64 * (g / 4);
        const u8* qh = block + 128 + 32 * (g / 4);
        const i8* sc = (const i8*)(block + 192) + 8 * (g / 4);
        float d = half(word(block + 208));
        int k = g % 4;
        const u8* lowBytes = ql + (k & 1) * 32;
        int lowShift = (k >> 1) * 4;
        int first = 0, second = 0;
        #pragma unroll
        for(int i = 0; i < 8; i++){
            u32 low = (load32(lowBytes + 4 * i) >> lowShift) & 0x0F0F0F0Fu;
            u32 high = ((load32(qh + 4 * i) >> (2 * k)) & 0x03030303u) << 4;
            int v = (int)__vsub4(low | high, 0x20202020u);
            if(i < 4) first = __dp4a(v, q[i], first);
            else second = __dp4a(v, q[i], second);
        }
        return d * dx * ((float)sc[2 * k] * first + (float)sc[2 * k + 1] * second);
    }

    if(type == 20 || type == 23){
        float d = half(word(block));
        const u8* qs = type == 20 ? block + 2 : block + 8 + 16 * g;
        if(type == 23){
            int low = (block[4 + g / 2] >> (4 * (g % 2))) & 15;
            int high = (word(block + 2) >> (2 * g)) & 3;
            d *= (float)((low | (high << 4)) - 32);
        }
        int sum = 0;
        #pragma unroll
        for(int i = 0; i < 4; i++){
            u32 raw = load32(qs + 4 * i);
            sum = __dp4a(table4(t.iq4nl, raw & 0x0F0F0F0Fu), q[i], sum);
            sum = __dp4a(table4(t.iq4nl, (raw >> 4) & 0x0F0F0F0Fu), q[i + 4], sum);
        }
        return d * dx * sum;
    }

    if(type == 42){
        float d = half(word(block));
        const u8* qs = block + 2 + 8 * g;
        int sum = 0;
        #pragma unroll
        for(int i = 0; i < 8; i++){
            u32 b = qs[i];
            u32 v = (b & 3) | (((b >> 2) & 3) << 8) | (((b >> 4) & 3) << 16) | (((b >> 6) & 3) << 24);
            sum = __dp4a((int)__vsub4(v, 0x01010101u), q[i], sum);
        }
        return d * dx * sum;
    }

    if(type == 16){
        float d = half(word(block));
        const u8* qs = block + 2 + 8 * g;
        u32 aux = load32(qs + 4);
        float dl = d * (0.5f + (float)(aux >> 28)) * 0.25f;
        int sum = 0;
        #pragma unroll
        for(int l = 0; l < 4; l++){
            u64 grid = t.iq2xxs[qs[l]];
            u32 signs = t.signs[(aux >> (7 * l)) & 127];
            sum = __dp4a(signed4((u32)grid, signs), q[2 * l], sum);
            sum = __dp4a(signed4((u32)(grid >> 32), signs >> 4), q[2 * l + 1], sum);
        }
        return dl * dx * sum;
    }

    if(type == 17 || type == 22){
        float d = half(word(block));
        u8 scale = type == 17 ? block[66 + g] : block[74 + g];
        int halves[2] = {0, 0};
        #pragma unroll
        for(int l = 0; l < 4; l++){
            u64 grid;
            u32 signs;
            if(type == 17){
                u16 code = word(block + 2 + 8 * g + 2 * l);
                grid = t.iq2xs[code & 511];
                signs = t.signs[code >> 9];
            } else {
                int index = block[2 + 4 * g + l] | ((block[66 + g] << (8 - 2 * l)) & 0x300);
                grid = t.iq2s[index];
                signs = block[34 + 4 * g + l];
            }
            int part = __dp4a(signed4((u32)grid, signs), q[2 * l], 0);
            part = __dp4a(signed4((u32)(grid >> 32), signs >> 4), q[2 * l + 1], part);
            halves[l / 2] += part;
        }
        float low = 0.5f + (float)(scale & 15);
        float high = 0.5f + (float)(scale >> 4);
        return d * 0.25f * dx * (low * halves[0] + high * halves[1]);
    }

    if(type == 18){
        float d = half(word(block));
        const u8* qs = block + 2 + 8 * g;
        u32 aux = load32(block + 66 + 4 * g);
        float dl = d * (0.5f + (float)(aux >> 28)) * 0.5f;
        int sum = 0;
        #pragma unroll
        for(int l = 0; l < 4; l++){
            u32 signs = t.signs[(aux >> (7 * l)) & 127];
            sum = __dp4a(signed4(t.iq3xxs[qs[2 * l]], signs), q[2 * l], sum);
            sum = __dp4a(signed4(t.iq3xxs[qs[2 * l + 1]], signs >> 4), q[2 * l + 1], sum);
        }
        return dl * dx * sum;
    }

    if(type == 21){
        float d = half(word(block));
        const u8* qs = block + 2 + 8 * g;
        u8 qh = block[66 + g];
        const u8* signs = block + 74 + 4 * g;
        float dl = d * (float)(1 + 2 * ((block[106 + g / 2] >> (4 * (g % 2))) & 15));
        int sum = 0;
        #pragma unroll
        for(int l = 0; l < 4; l++){
            u32 first = t.iq3s[qs[2 * l] | ((qh << (8 - 2 * l)) & 256)];
            u32 second = t.iq3s[qs[2 * l + 1] | ((qh << (7 - 2 * l)) & 256)];
            sum = __dp4a(signed4(first, signs[l]), q[2 * l], sum);
            sum = __dp4a(signed4(second, (u32)signs[l] >> 4), q[2 * l + 1], sum);
        }
        return dl * dx * sum;
    }

    if(type == 29){
        u16 bits = (u16)((word(block + 48) >> 12) | ((word(block + 50) >> 8) & 0x00F0) | ((word(block + 52) >> 4) & 0x0F00) | (word(block + 54) & 0xF000));
        int sc = word(block + 48 + 2 * (g / 2)) >> (6 * (g % 2));
        int halves[2] = {0, 0};
        #pragma unroll
        for(int l = 0; l < 4; l++){
            int h = block[32 + 2 * g + l / 2] >> (4 * (l % 2));
            u64 grid = t.iq1s[block[4 * g + l] | ((h & 7) << 8)];
            u32 drop = h & 8 ? 0x01010101u : 0xFFFFFFFFu;
            int part = __dp4a((int)__vsub4((u32)grid, drop), q[2 * l], 0);
            halves[l / 2] += __dp4a((int)__vsub4((u32)(grid >> 32), drop), q[2 * l + 1], part);
        }
        float low = (float)(2 * (sc & 7) + 1);
        float high = (float)(2 * ((sc >> 3) & 7) + 1);
        return half(bits) * 0.125f * dx * (low * halves[0] + high * halves[1]);
    }

    if(type == 8){
        float d = half(word(block));
        int sum = 0;
        #pragma unroll
        for(int i = 0; i < 8; i++) sum = __dp4a((int)load32(block + 2 + 4 * i), q[i], sum);
        return d * dx * sum;
    }

    if(type == 7){
        float d = half(word(block));
        float m = half(word(block + 2));
        u32 qh = load32(block + 4);
        int sum = 0;
        #pragma unroll
        for(int i = 0; i < 4; i++){
            u32 raw = load32(block + 8 + 4 * i);
            u32 lowBits = (qh >> (4 * i)) & 15;
            u32 highBits = (qh >> (16 + 4 * i)) & 15;
            u32 lowTop = ((lowBits & 1) << 4) | ((lowBits & 2) << 11) | ((lowBits & 4) << 18) | ((lowBits & 8) << 25);
            u32 highTop = ((highBits & 1) << 4) | ((highBits & 2) << 11) | ((highBits & 4) << 18) | ((highBits & 8) << 25);
            sum = __dp4a((int)((raw & 0x0F0F0F0Fu) | lowTop), q[i], sum);
            sum = __dp4a((int)(((raw >> 4) & 0x0F0F0F0Fu) | highTop), q[i + 4], sum);
        }
        return dx * (d * sum + m * qsum);
    }

    return 0.0f;
}

extern "C" __global__ void quantizeInput(const float* input, int count, i8* packed, float* scales, int* sums) {
    int group = blockIdx.x * blockDim.y + threadIdx.y;
    if(group * 32 >= count) return;
    float v = input[group * 32 + threadIdx.x];
    float peak = fabsf(v);
    for(int step = 16; step > 0; step >>= 1) peak = fmaxf(peak, __shfl_xor_sync(0xffffffff, peak, step));
    float d = peak / 127.0f;
    int q = d > 0.0f ? __float2int_rn(v / d) : 0;
    packed[group * 32 + threadIdx.x] = (i8)q;
    int total = q;
    for(int step = 16; step > 0; step >>= 1) total += __shfl_xor_sync(0xffffffff, total, step);
    if(threadIdx.x == 0){
        scales[group] = d;
        sums[group] = total;
    }
}

__device__ __forceinline__ float rowDotQ8(int type, const u8* line, int cols, const float* x, const i8* packed, const float* scales, const int* sums, const GridTables& t) {
    int groups = cols / 32;
    int perBlock = groupsPerBlock(type);
    int bytes = blockBytes(type);
    bool plain = type == 0 || type == 1 || type == 30;
    float sum = 0.0f;
    for(int u = threadIdx.x; u < groups; u += 32){
        const u8* block = line + (u64)(u / perBlock) * bytes;
        if(plain){
            sum += dotGroup(type, block, u % perBlock, x + 32 * u, t);
            continue;
        }
        const int* q = (const int*)(packed + 32 * u);
        int words[8];
        #pragma unroll
        for(int i = 0; i < 8; i++) words[i] = q[i];
        sum += dotGroupQ8(type, block, u % perBlock, words, scales[u], sums[u], t);
    }
    for(int step = 16; step > 0; step >>= 1) sum += __shfl_down_sync(0xffffffff, sum, step);
    return sum;
}

extern "C" __global__ void quantRowsQ8(int type, const u8* weights, u64 rowBytes, const float* input, const i8* packed, const float* scales, const int* sums, float* output, int rows, int cols, GridTables t) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if(row >= rows) return;
    float sum = rowDotQ8(type, weights + rowBytes * row, cols, input, packed, scales, sums, t);
    if(threadIdx.x == 0) output[row] = sum;
}

extern "C" __global__ void expertRowsQ8(int type, const u8* weights, u64 rowBytes, u64 expertBytes, const int* experts, const float* input, const i8* packed, const float* scales, const int* sums, u64 inputStride, float* output, int rows, int cols, GridTables t) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if(row >= rows) return;
    int slot = blockIdx.y;
    const u8* line = weights + expertBytes * experts[slot] + rowBytes * row;
    u64 at = inputStride * slot;
    float sum = rowDotQ8(type, line, cols, input + at, packed + at, scales + at / 32, sums + at / 32, t);
    if(threadIdx.x == 0) output[(u64)rows * slot + row] = sum;
}

extern "C" __global__ void quantRows(int type, const u8* weights, u64 rowBytes, const float* input, float* output, int rows, int cols, GridTables t) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if(row >= rows) return;
    const u8* line = weights + rowBytes * row;
    int groups = cols / 32;
    int perBlock = groupsPerBlock(type);
    int bytes = blockBytes(type);
    float sum = 0.0f;
    for(int u = threadIdx.x; u < groups; u += 32){
        const u8* block = line + (u64)(u / perBlock) * bytes;
        sum += dotGroup(type, block, u % perBlock, input + 32 * u, t);
    }
    for(int step = 16; step > 0; step >>= 1) sum += __shfl_down_sync(0xffffffff, sum, step);
    if(threadIdx.x == 0) output[row] = sum;
}

extern "C" __global__ void expertRows(int type, const u8* weights, u64 rowBytes, u64 expertBytes, const int* experts, const float* input, u64 inputStride, float* output, int rows, int cols, GridTables t) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if(row >= rows) return;
    int slot = blockIdx.y;
    const u8* line = weights + expertBytes * experts[slot] + rowBytes * row;
    const float* x = input + inputStride * slot;
    int groups = cols / 32;
    int perBlock = groupsPerBlock(type);
    int bytes = blockBytes(type);
    float sum = 0.0f;
    for(int u = threadIdx.x; u < groups; u += 32){
        const u8* block = line + (u64)(u / perBlock) * bytes;
        sum += dotGroup(type, block, u % perBlock, x + 32 * u, t);
    }
    for(int step = 16; step > 0; step >>= 1) sum += __shfl_down_sync(0xffffffff, sum, step);
    if(threadIdx.x == 0) output[(u64)rows * slot + row] = sum;
}
)";

inline bool launchRows(KernelModule& module, int type, uint64_t weights, uint64_t rowBytes, uint64_t input, uint64_t output, int rows, int cols, GridTables& tables) {
    void* fn = kernelFunction(module, "quantRows");
    void* args[] = {&type, &weights, &rowBytes, &input, &output, &rows, &cols, &tables};
    return launchKernel(fn, (unsigned int)((rows + 7) / 8), 1, 32, 8, 0, args);
}

struct PackedInput {
    uint64_t values = 0;
    uint64_t scales = 0;
    uint64_t sums = 0;
    int capacity = 0;
};

inline bool packInput(KernelModule& module, uint64_t input, int count, PackedInput& packed) {
    if(packed.capacity < count){
        if(packed.values) cudaLink().release(packed.values);
        if(packed.scales) cudaLink().release(packed.scales);
        if(packed.sums) cudaLink().release(packed.sums);
        if(cudaLink().alloc(&packed.values, count) != 0 || cudaLink().alloc(&packed.scales, count / 8) != 0 || cudaLink().alloc(&packed.sums, count / 8) != 0) return false;
        packed.capacity = count;
    }
    void* fn = kernelFunction(module, "quantizeInput");
    void* args[] = {&input, &count, &packed.values, &packed.scales, &packed.sums};
    int groups = count / 32;
    return launchKernel(fn, (unsigned int)((groups + 7) / 8), 1, 32, 8, 0, args);
}

inline bool launchRowsQ8(KernelModule& module, int type, uint64_t weights, uint64_t rowBytes, uint64_t input, PackedInput& packed, uint64_t output, int rows, int cols, GridTables& tables) {
    void* fn = kernelFunction(module, "quantRowsQ8");
    void* args[] = {&type, &weights, &rowBytes, &input, &packed.values, &packed.scales, &packed.sums, &output, &rows, &cols, &tables};
    return launchKernel(fn, (unsigned int)((rows + 7) / 8), 1, 32, 8, 0, args);
}

inline bool launchExpertsQ8(KernelModule& module, int type, uint64_t weights, uint64_t rowBytes, uint64_t expertBytes, uint64_t experts, int count, uint64_t input, PackedInput& packed, uint64_t inputStride, uint64_t output, int rows, int cols, GridTables& tables) {
    void* fn = kernelFunction(module, "expertRowsQ8");
    void* args[] = {&type, &weights, &rowBytes, &expertBytes, &experts, &input, &packed.values, &packed.scales, &packed.sums, &inputStride, &output, &rows, &cols, &tables};
    return launchKernel(fn, (unsigned int)((rows + 7) / 8), (unsigned int)count, 32, 8, 0, args);
}

inline bool launchExperts(KernelModule& module, int type, uint64_t weights, uint64_t rowBytes, uint64_t expertBytes, uint64_t experts, int count, uint64_t input, uint64_t inputStride, uint64_t output, int rows, int cols, GridTables& tables) {
    void* fn = kernelFunction(module, "expertRows");
    void* args[] = {&type, &weights, &rowBytes, &expertBytes, &experts, &input, &inputStride, &output, &rows, &cols, &tables};
    return launchKernel(fn, (unsigned int)((rows + 7) / 8), (unsigned int)count, 32, 8, 0, args);
}
