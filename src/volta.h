#pragma once

// atencion del prompt con los tensor cores de la v100 (sm_70) que solo tienen mma m8n8k4
// se agrega al programa de la tarjeta solo en una v100 o con SNEO_VOLTA_ATTENTION=1 para probarla en otra nvidia
// cada warp tiene 64 dimensiones y cada grupo de 4 hilos altos y bajos (pair) calcula su propio bloque de 8x8
// q va en memoria compartida porque los 4 pares leen las mismas filas y en registros se copiaria 4 veces
const char* voltaKernelSource = R"(
#if KVHEADS > 0 && QHEADS <= 16 * KVHEADS
#define VROWS (QHEADS / KVHEADS)
#define VTILES (VROWS > 8 ? 2 : 1)
#define QSTRIDE 264
#define KSTRIDE 260

__device__ __forceinline__ void voltaMma(float* c, u32 a0, u32 a1, u32 b0, u32 b1) {
#if __CUDA_ARCH__ >= 700
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ u32 voltaPair(float a, float b) {
    u32 out = 0;
    asm("{.reg .b16 l, h; cvt.rn.f16.f32 l, %1; cvt.rn.f16.f32 h, %2; mov.b32 %0, {l, h};}" : "=r"(out) : "f"(a), "f"(b));
    return out;
}

__device__ __forceinline__ void voltaSplit(float a, float b, u32& high, u32& low) {
    high = voltaPair(a, b);
    float ha = 0.0f, hb = 0.0f;
    asm("{.reg .b16 l, h; mov.b32 {l, h}, %2; cvt.f32.f16 %0, l; cvt.f32.f16 %1, h;}" : "=f"(ha), "=f"(hb) : "r"(high));
    low = voltaPair(a - ha, b - hb);
}

// cuatro bytes con signo a dos pares de half exactos: 1152 + byte y se resta 1152
__device__ __forceinline__ void voltaCodes(u32 x, u32& first, u32& second) {
    u32 y = x ^ 0x80808080u;
    u32 a = __byte_perm(y, 0x64646464u, 0x5140);
    u32 b = __byte_perm(y, 0x64646464u, 0x5342);
    asm("sub.f16x2 %0, %1, %2;" : "=r"(first) : "r"(a), "r"(0x64806480u));
    asm("sub.f16x2 %0, %1, %2;" : "=r"(second) : "r"(b), "r"(0x64806480u));
}

extern "C" __global__ void __launch_bounds__(128) attentionVolta(const float* q, const i8* keys, const float* keyScales, const i8* values, const float* valueScales, const int* cells, const int* positions, float* out, int budget) {
    __shared__ __align__(16) u16 high[16 * QSTRIDE];
    __shared__ __align__(16) u16 low[16 * QSTRIDE];
    __shared__ __align__(16) i8 keyRows[32 * KSTRIDE];
    __shared__ __align__(16) i8 valueRows[32 * 256];
    __shared__ float keyScale[32][8], valueScale[32][8];
    __shared__ float part[4][16][33];
    __shared__ float prob[16][33];
    __shared__ float best[16], total[16], shrink[16], peak[4];
    int t = blockIdx.y, kvh = blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    int pair = (lane >> 2) & 3, mine = (lane & 3) + ((lane >> 4) << 2);
    int crow = (lane & 1) + ((lane >> 4) << 2), ccol = lane & 2;
    int dim0 = warp * 64;
    int p = positions[t];
    bool all = p + 1 <= budget;
    int count = all ? p + 1 : budget;
    const int* list = cells + (u64)t * budget;
    const float* qt = q + (u64)t * (QHEADS * 512) + (u64)kvh * VROWS * 512;

    float top = 0.0f;
    for(int i = tid; i < VROWS * 256; i += 128) top = fmaxf(top, fabsf(qt[(i >> 8) * 512 + (i & 255)]));
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
    for(int i = tid; i < 16 * 128; i += 128){
        int row = i >> 7, col = (i & 127) * 2;
        float2 x = row < VROWS ? *(const float2*)(qt + row * 512 + col) : make_float2(0.0f, 0.0f);
        u32 h = 0, l = 0;
        voltaSplit(x.x * up, x.y * up, h, l);
        *(u32*)&high[row * QSTRIDE + col] = h;
        *(u32*)&low[row * QSTRIDE + col] = l;
    }

    // el bloque siguiente se lee a registros mientras se calcula el actual: la v100 no tiene copia asincrona
    uint4 nextKeys[4], nextValues[4];
    float nextKeyScales[2], nextValueScales[2];
    auto cellAt = [&](int c) {
        return c < count
            ? all
                ? c
                : list[c]
            : -1;
    };
    auto fetch = [&](int c0) {
        #pragma unroll
        for(int r = 0; r < 4; r++){
            int at = tid + 128 * r, from = cellAt(c0 + (at >> 4));
            u64 offset = from >= 0 ? (u64)from * KVW + kvh * 256 + (at & 15) * 16 : 0;
            nextKeys[r] = from >= 0 ? __ldg((const uint4*)(keys + offset)) : make_uint4(0, 0, 0, 0);
            nextValues[r] = from >= 0 ? __ldg((const uint4*)(values + offset)) : make_uint4(0, 0, 0, 0);
        }
        #pragma unroll
        for(int r = 0; r < 2; r++){
            int at = tid + 128 * r, from = cellAt(c0 + (at >> 3));
            u64 offset = from >= 0 ? (u64)from * (KVW / 32) + kvh * 8 + (at & 7) : 0;
            nextKeyScales[r] = from >= 0 ? keyScales[offset] : 0.0f;
            nextValueScales[r] = from >= 0 ? valueScales[offset] : 0.0f;
        }
    };
    auto store = [&]() {
        #pragma unroll
        for(int r = 0; r < 4; r++){
            int at = tid + 128 * r, c = at >> 4, piece = at & 15;
            u32* key = (u32*)(keyRows + c * KSTRIDE + piece * 16);
            key[0] = nextKeys[r].x;
            key[1] = nextKeys[r].y;
            key[2] = nextKeys[r].z;
            key[3] = nextKeys[r].w;
            *(uint4*)(valueRows + c * 256 + piece * 16) = nextValues[r];
        }
        #pragma unroll
        for(int r = 0; r < 2; r++){
            int at = tid + 128 * r;
            keyScale[at >> 3][at & 7] = nextKeyScales[r];
            valueScale[at >> 3][at & 7] = nextValueScales[r];
        }
    };

    float acc[VTILES][2][8];
    #pragma unroll
    for(int mt = 0; mt < VTILES; mt++){
        #pragma unroll
        for(int i = 0; i < 8; i++) acc[mt][0][i] = acc[mt][1][i] = 0.0f;
    }

    int chunks = (count + 31) / 32;
    fetch(0);
    store();
    for(int ci = 0; ci < chunks; ci++){
        int c0 = ci * 32;
        __syncthreads();
        if(ci + 1 < chunks) fetch(c0 + 32);

        // q por k: el par toma las celdas pair * 8 .. + 7 y cada grupo de 32 dimensiones va en su cadena con su escala
        #pragma unroll
        for(int mt = 0; mt < VTILES; mt++){
            float s[4][8];
            #pragma unroll
            for(int g = 0; g < 4; g++){
                #pragma unroll
                for(int i = 0; i < 8; i++) s[g][i] = 0.0f;
            }
            const u16* qh = high + (mt * 8 + mine) * QSTRIDE + dim0;
            const u16* ql = low + (mt * 8 + mine) * QSTRIDE + dim0;
            const i8* key = keyRows + (pair * 8 + mine) * KSTRIDE + dim0;
            #pragma unroll
            for(int ks = 0; ks < 16; ks++){
                uint2 a = *(const uint2*)(qh + ks * 4);
                uint2 b = *(const uint2*)(ql + ks * 4);
                u32 k0 = 0, k1 = 0;
                voltaCodes(*(const u32*)(key + ks * 4), k0, k1);
                voltaMma(s[(ks >> 3) * 2], a.x, a.y, k0, k1);
                voltaMma(s[(ks >> 3) * 2 + 1], b.x, b.y, k0, k1);
            }
            #pragma unroll
            for(int i = 0; i < 8; i++){
                int r = mt * 8 + crow + (i & 2), c = pair * 8 + ccol + (i & 4) + (i & 1);
                part[warp][r][c] = (s[0][i] + s[1][i]) * keyScale[c][warp * 2] + (s[2][i] + s[3][i]) * keyScale[c][warp * 2 + 1];
            }
        }
        #pragma unroll
        for(int mt = VTILES; mt < 2; mt++){
            #pragma unroll
            for(int i = 0; i < 8; i++) part[warp][mt * 8 + crow + (i & 2)][pair * 8 + ccol + (i & 4) + (i & 1)] = 0.0f;
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

        // p por v: el par toma 16 dimensiones de su warp (dos bloques de 8) y p lleva la escala de v de su grupo
        float vmax = fmaxf(valueScale[lane][warp * 2], valueScale[lane][warp * 2 + 1]);
        for(int step = 16; step > 0; step >>= 1) vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffff, vmax, step));
        float vup = vmax > 0.0f ? 16384.0f / vmax : 0.0f, vdown = vmax * (1.0f / 16384.0f);
        int group = warp * 2 + (pair >> 1), base = dim0 + pair * 16 + mine;
        float sumHigh[VTILES][2][8], sumLow[VTILES][2][8];
        #pragma unroll
        for(int mt = 0; mt < VTILES; mt++){
            #pragma unroll
            for(int i = 0; i < 8; i++) sumHigh[mt][0][i] = sumHigh[mt][1][i] = sumLow[mt][0][i] = sumLow[mt][1][i] = 0.0f;
        }
        #pragma unroll
        for(int ks = 0; ks < 8; ks++){
            int c = ks * 4;
            u32 v[2][2];
            #pragma unroll
            for(int nt = 0; nt < 2; nt++){
                const i8* at = valueRows + c * 256 + base + nt * 8;
                u32 x = (u32)(u8)at[0] | ((u32)(u8)at[256] << 8) | ((u32)(u8)at[512] << 16) | ((u32)(u8)at[768] << 24);
                voltaCodes(x, v[nt][0], v[nt][1]);
            }
            float w0 = valueScale[c][group] * vup, w1 = valueScale[c + 1][group] * vup;
            float w2 = valueScale[c + 2][group] * vup, w3 = valueScale[c + 3][group] * vup;
            #pragma unroll
            for(int mt = 0; mt < VTILES; mt++){
                const float* pr = prob[mt * 8 + mine] + c;
                u32 h0 = 0, l0 = 0, h1 = 0, l1 = 0;
                voltaSplit(pr[0] * w0, pr[1] * w1, h0, l0);
                voltaSplit(pr[2] * w2, pr[3] * w3, h1, l1);
                #pragma unroll
                for(int nt = 0; nt < 2; nt++){
                    voltaMma(sumHigh[mt][nt], h0, h1, v[nt][0], v[nt][1]);
                    voltaMma(sumLow[mt][nt], l0, l1, v[nt][0], v[nt][1]);
                }
            }
        }
        #pragma unroll
        for(int mt = 0; mt < VTILES; mt++){
            #pragma unroll
            for(int i = 0; i < 8; i++){
                float k = shrink[mt * 8 + crow + (i & 2)];
                acc[mt][0][i] = acc[mt][0][i] * k + (sumHigh[mt][0][i] + sumLow[mt][0][i]) * vdown;
                acc[mt][1][i] = acc[mt][1][i] * k + (sumHigh[mt][1][i] + sumLow[mt][1][i]) * vdown;
            }
        }
        if(ci + 1 >= chunks) break;
        __syncthreads();
        store();
    }

    float* ot = out + (u64)t * (QHEADS * 256) + (u64)kvh * VROWS * 256;
    #pragma unroll
    for(int mt = 0; mt < VTILES; mt++){
        #pragma unroll
        for(int i = 0; i < 8; i++){
            int r = mt * 8 + crow + (i & 2);
            if(r >= VROWS) continue;
            float inverse = 1.0f / total[r];
            #pragma unroll
            for(int nt = 0; nt < 2; nt++){
                int d = dim0 + pair * 16 + nt * 8 + ccol + (i & 4) + (i & 1);
                ot[r * 256 + d] = acc[mt][nt][i] * inverse * gateOf(qt[r * 512 + 256 + d]);
            }
        }
    }
}
#endif
)";
