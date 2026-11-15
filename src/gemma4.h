#pragma once
#include "gguf.h"
#include "kernels.h"
#include <vector>
#include <string>

struct GemmaLayer {
    bool sliding = false;
    bool mixture = false;
    int kvHeads = 0;
    int headSize = 0;
    int slots = 0;
    GpuMatrix query, key, value, out;
    GpuMatrix gate, up, down;
    GpuMatrix expertGateUp, expertDown;
    uint64_t expertGateUpBytes = 0, expertDownBytes = 0;
    uint64_t attnNorm = 0, queryNorm = 0, keyNorm = 0, postAttnNorm = 0;
    uint64_t ffnNorm = 0, postFfnNorm = 0, postMlpNorm = 0, preMoeNorm = 0, postMoeNorm = 0;
    uint64_t router = 0, routerScale = 0, downScale = 0, outScale = 0, ropeFactors = 0;
    uint64_t keys = 0, values = 0;
    std::vector<float> downScaleHost;
};

struct GemmaShape {
    int layers = 0;
    int hidden = 0;
    int heads = 0;
    int window = 0;
    int ring = 0;
    int feedForward = 0;
    int experts = 0;
    int expertsUsed = 0;
    int expertWidth = 0;
    int vocab = 0;
    float eps = 1e-6f;
    float softcap = 0.0f;
    float ropeBase = 1e6f;
    float ropeBaseSliding = 1e4f;
    int widest = 0;
    int widestKv = 0;
    int splitLimit = 64;
};

struct GemmaBuffers {
    uint64_t x = 0, xn = 0, q = 0, k = 0, v = 0, o = 0, attn = 0, a = 0;
    uint64_t gate = 0, up = 0, hid = 0, mlp = 0;
    uint64_t moeIn = 0, routerIn = 0, ids = 0, weights = 0, both = 0, expertHid = 0, expertOut = 0, moe = 0, pair = 0;
    uint64_t positions = 0, scratch = 0, last = 0, vocab = 0, outNorm = 0;
    // en una pasada de verificacion el estado final normado y los logits de cada fila
    uint64_t normed = 0, every = 0;
    int everyCap = 0;
    PackedInput expertIn, expertMid;
    float* hostEmbed = nullptr;
    int* hostPositions = nullptr;
    float* hostLogits = nullptr;
    float* hostEvery = nullptr;
    uint64_t mappedEmbed = 0, mappedLogits = 0, mappedEvery = 0;
    int* hostRoutes = nullptr;
    int* hostPlan = nullptr;
    uint64_t mappedRoutes = 0, mappedPlan = 0, plan = 0;
    int routeCap = 0;
    int ringTop = -1;
    bool saved[4] = {};
};

// capa del ayudante que adivina tokens solo con consultas y lee claves y valores de la ultima capa del modelo con su mismo tipo de atencion
struct GemmaDraftLayer {
    bool sliding = false;
    int kvHeads = 0;
    int headSize = 0;
    int target = -1;
    GpuMatrix query, out, gate, up, down;
    uint64_t attnNorm = 0, queryNorm = 0, postAttnNorm = 0, ffnNorm = 0, postFfnNorm = 0, outScale = 0;
};

struct GemmaDraft {
    bool ready = false;
    int hidden = 0;
    int heads = 0;
    int feedForward = 0;
    float eps = 1e-6f;
    float ropeBase = 1e6f;
    float ropeBaseSliding = 1e4f;
    std::vector<GemmaDraftLayer> layers;
    GpuMatrix join, back, head;
    uint64_t outNorm = 0, ropeFactors = 0;
    // input lleva el embedding del token por la raiz del ancho y despues el estado del modelo que back reescribe para el paso siguiente
    uint64_t input = 0, x = 0, xn = 0, q = 0, o = 0, attn = 0, a = 0, gate = 0, up = 0, hid = 0, mlp = 0, fin = 0, vocab = 0;
    uint64_t position = 0, mappedPick = 0;
    int* hostPosition = nullptr;
    volatile uint32_t* hostPick = nullptr;
    uint64_t pendingFrom = 0;
    int pendingAt = -1;
    uint64_t bytes = 0;
};

const char* gemmaKernelSource = R"(
__device__ __forceinline__ u16 gemmaHalf(float v) {
#ifdef __HIP__
    return __builtin_bit_cast(unsigned short, (_Float16)v);
#else
    unsigned short r;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(r) : "f"(v));
    return r;
#endif
}

__device__ __forceinline__ float gemmaGeluOf(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

__device__ __forceinline__ float gemmaBlockSum(float v, float* shared) {
    for(int step = 16; step > 0; step >>= 1) v += __shfl_xor_sync(0xffffffff, v, step);
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if(lane == 0) shared[warp] = v;
    __syncthreads();
    int warps = (blockDim.x + 31) >> 5;
    float total = 0.0f;
    for(int w = 0; w < warps; w++) total += shared[w];
    __syncthreads();
    return total;
}

// cuantiza un grupo de 32 como el primer nivel de quantizeFine (los mismos bits) desde un warp
__device__ __forceinline__ void gemmaPackGroup(float v, int group, i8* packed, float* scales, int* sums) {
    int lane = threadIdx.x & 31;
    float peak = fabsf(v);
    for(int step = 16; step > 0; step >>= 1) peak = fmaxf(peak, __shfl_xor_sync(0xffffffff, peak, step));
    float d = peak / 127.0f;
    int q = d > 0.0f ? __float2int_rn(v / d) : 0;
    packed[group * 32 + lane] = (i8)q;
    int total = q;
    for(int step = 16; step > 0; step >>= 1) total += __shfl_xor_sync(0xffffffff, total, step);
    if(lane == 0){
        scales[group] = d;
        sums[group] = total;
    }
}

// norma de la fila y cuantizacion en un solo kernel (antes gemmaRms y quantizeFine)
extern "C" __global__ void gemmaRmsPack(const float* input, const float* weight, float* output, int cols, float eps, i8* packed, float* scales, int* sums) {
    __shared__ float partial[32];
    const float* row = input + (u64)blockIdx.x * cols;
    float* out = output + (u64)blockIdx.x * cols;
    float sum = 0.0f;
    for(int i = threadIdx.x; i < cols; i += blockDim.x) sum = fmaf(row[i], row[i], sum);
    float inv = rsqrtf(gemmaBlockSum(sum, partial) / cols + eps);
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, groups = cols / 32;
    for(int g = warp; g < groups; g += blockDim.x >> 5){
        int i = g * 32 + lane;
        float v = row[i] * inv * weight[i];
        out[i] = v;
        gemmaPackGroup(v, blockIdx.x * groups + g, packed, scales, sums);
    }
}

// suma de la atencion normada al residuo y norma con cuantizacion para la red de la capa (antes gemmaRmsAdd, gemmaRms y quantizeFine)
extern "C" __global__ void gemmaAddRmsPack(const float* input, const float* weight, const float* residual, float* output, const float* nextWeight, float* normed, int cols, float eps, i8* packed, float* scales, int* sums) {
    __shared__ float partial[32];
    const float* row = input + (u64)blockIdx.x * cols;
    const float* base = residual + (u64)blockIdx.x * cols;
    float* out = output + (u64)blockIdx.x * cols;
    float* next = normed + (u64)blockIdx.x * cols;
    float sum = 0.0f;
    for(int i = threadIdx.x; i < cols; i += blockDim.x) sum = fmaf(row[i], row[i], sum);
    float inv = rsqrtf(gemmaBlockSum(sum, partial) / cols + eps);
    float total = 0.0f;
    for(int i = threadIdx.x; i < cols; i += blockDim.x){
        float v = fmaf(row[i] * inv, weight[i], base[i]) * 1.0f;
        out[i] = v;
        total = fmaf(v, v, total);
    }
    float invNext = rsqrtf(gemmaBlockSum(total, partial) / cols + eps);
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, groups = cols / 32;
    for(int g = warp; g < groups; g += blockDim.x >> 5){
        int i = g * 32 + lane;
        float v = out[i] * invNext * nextWeight[i];
        next[i] = v;
        gemmaPackGroup(v, blockIdx.x * groups + g, packed, scales, sums);
    }
}

// entrada de los expertos y del enrutador desde la misma norma (antes dos gemmaRms y quantizeInput)
extern "C" __global__ void gemmaMoeIn(const float* input, const float* expertWeight, const float* routerScale, float shrink, int cols, float eps, float* moeIn, float* routerIn, i8* packed, float* scales, int* sums) {
    __shared__ float partial[32];
    const float* row = input + (u64)blockIdx.x * cols;
    float* expertOut = moeIn + (u64)blockIdx.x * cols;
    float* routerOut = routerIn + (u64)blockIdx.x * cols;
    float sum = 0.0f;
    for(int i = threadIdx.x; i < cols; i += blockDim.x) sum = fmaf(row[i], row[i], sum);
    float root = rsqrtf(gemmaBlockSum(sum, partial) / cols + eps);
    float inv = root * 1.0f;
    float invRouter = root * shrink;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, groups = cols / 32;
    for(int g = warp; g < groups; g += blockDim.x >> 5){
        int i = g * 32 + lane;
        float v = row[i] * inv * expertWeight[i];
        expertOut[i] = v;
        routerOut[i] = row[i] * invRouter * routerScale[i];
        gemmaPackGroup(v, blockIdx.x * groups + g, packed, scales, sums);
    }
}

// salida de los expertos mezclada con la red densa y sumada al residuo (antes gemmaMix o gemmaMixSorted, gemmaPair y gemmaRmsAdd)
extern "C" __global__ void gemmaMoeOut(const float* expertOut, const float* weights, const float* downScale, const int* ids, const int* rankOf, const float* mlp, const float* mlpNorm, const float* moeNorm, const float* finalNorm, const float* residual, float* output, const float* layerScale, int used, int cols, float eps) {
    extern __shared__ __align__(16) u8 room[];
    __shared__ float partial[32];
    float* moe = (float*)room;
    float* pair = moe + cols;
    int token = blockIdx.x;
    const float* dense = mlp + (u64)token * cols;
    for(int i = threadIdx.x; i < cols; i += blockDim.x){
        float v = 0.0f;
        for(int k = 0; k < used; k++){
            int slot = token * used + k;
            float w = weights[slot] * (downScale ? downScale[ids[slot]] : 1.0f);
            v = fmaf(w, expertOut[(u64)(rankOf ? rankOf[slot] : slot) * cols + i], v);
        }
        moe[i] = v;
    }
    float sa = 0.0f, sb = 0.0f;
    for(int i = threadIdx.x; i < cols; i += blockDim.x){
        sa = fmaf(dense[i], dense[i], sa);
        sb = fmaf(moe[i], moe[i], sb);
    }
    float ia = rsqrtf(gemmaBlockSum(sa, partial) / cols + eps);
    float ib = rsqrtf(gemmaBlockSum(sb, partial) / cols + eps);
    float sp = 0.0f;
    for(int i = threadIdx.x; i < cols; i += blockDim.x){
        float v = fmaf(dense[i] * ia, mlpNorm[i], moe[i] * ib * moeNorm[i]);
        pair[i] = v;
        sp = fmaf(v, v, sp);
    }
    float ip = rsqrtf(gemmaBlockSum(sp, partial) / cols + eps);
    float scale = layerScale ? layerScale[0] : 1.0f;
    const float* base = residual + (u64)token * cols;
    float* out = output + (u64)token * cols;
    for(int i = threadIdx.x; i < cols; i += blockDim.x) out[i] = fmaf(pair[i] * ip, finalNorm[i], base[i]) * scale;
}

// gelu de la red densa y cuantizacion (antes gemmaGelu y quantizeFine)
extern "C" __global__ void gemmaGeluPack(const float* gate, const float* up, float* output, int count, i8* packed, float* scales, int* sums) {
    int group = blockIdx.x * blockDim.y + threadIdx.y;
    if(group * 32 >= count) return;
    int i = group * 32 + threadIdx.x;
    float v = gemmaGeluOf(gate[i]) * up[i];
    output[i] = v;
    gemmaPackGroup(v, group, packed, scales, sums);
}

// gelu de los expertos con compuerta y subida juntas y cuantizacion (antes gemmaGeluFused y quantizeInput)
extern "C" __global__ void gemmaGeluBothPack(const float* both, float* output, int width, int rows, i8* packed, float* scales, int* sums) {
    int group = blockIdx.x * blockDim.y + threadIdx.y;
    if(group * 32 >= width * rows) return;
    int i = group * 32 + threadIdx.x;
    int r = i / width, c = i % width;
    const float* line = both + (u64)r * 2 * width;
    float v = gemmaGeluOf(line[c]) * line[width + c];
    output[i] = v;
    gemmaPackGroup(v, group, packed, scales, sums);
}

extern "C" __global__ void gemmaEmbed(const float* host, float* x, int count, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < count) x[i] = host[i] * scale;
}

extern "C" __global__ void gemmaRms(const float* input, const float* weight, float* output, int cols, float eps, float extra, const float* scaleVec) {
    __shared__ float partial[32];
    const float* row = input + (u64)blockIdx.x * cols;
    float* out = output + (u64)blockIdx.x * cols;
    float sum = 0.0f;
    for(int i = threadIdx.x; i < cols; i += blockDim.x) sum = fmaf(row[i], row[i], sum);
    float inv = rsqrtf(gemmaBlockSum(sum, partial) / cols + eps) * extra;
    for(int i = threadIdx.x; i < cols; i += blockDim.x){
        float v = row[i] * inv;
        if(weight) v *= weight[i];
        if(scaleVec) v *= scaleVec[i];
        out[i] = v;
    }
}

extern "C" __global__ void gemmaRmsAdd(const float* input, const float* weight, const float* residual, float* output, int cols, float eps, const float* layerScale) {
    __shared__ float partial[32];
    const float* row = input + (u64)blockIdx.x * cols;
    const float* base = residual + (u64)blockIdx.x * cols;
    float* out = output + (u64)blockIdx.x * cols;
    float sum = 0.0f;
    for(int i = threadIdx.x; i < cols; i += blockDim.x) sum = fmaf(row[i], row[i], sum);
    float inv = rsqrtf(gemmaBlockSum(sum, partial) / cols + eps);
    float scale = layerScale ? layerScale[0] : 1.0f;
    for(int i = threadIdx.x; i < cols; i += blockDim.x) out[i] = fmaf(row[i] * inv, weight[i], base[i]) * scale;
}

extern "C" __global__ void gemmaPair(const float* first, const float* firstWeight, const float* second, const float* secondWeight, float* output, int cols, float eps) {
    __shared__ float partial[32];
    const float* a = first + (u64)blockIdx.x * cols;
    const float* b = second + (u64)blockIdx.x * cols;
    float* out = output + (u64)blockIdx.x * cols;
    float sa = 0.0f, sb = 0.0f;
    for(int i = threadIdx.x; i < cols; i += blockDim.x){
        sa = fmaf(a[i], a[i], sa);
        sb = fmaf(b[i], b[i], sb);
    }
    float ia = rsqrtf(gemmaBlockSum(sa, partial) / cols + eps);
    float ib = rsqrtf(gemmaBlockSum(sb, partial) / cols + eps);
    for(int i = threadIdx.x; i < cols; i += blockDim.x) out[i] = fmaf(a[i] * ia, firstWeight[i], b[i] * ib * secondWeight[i]);
}

extern "C" __global__ void gemmaGelu(const float* gate, const float* up, float* output, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < count) output[i] = gemmaGeluOf(gate[i]) * up[i];
}

extern "C" __global__ void gemmaGeluFused(const float* both, float* output, int width, int rows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= width * rows) return;
    int r = i / width, c = i % width;
    const float* line = both + (u64)r * 2 * width;
    output[i] = gemmaGeluOf(line[c]) * line[width + c];
}

extern "C" __global__ void gemmaHeads(float* q, const float* k, const float* v, const float* queryNorm, const float* keyNorm, u16* keys, u16* values, const int* positions, int heads, int kvHeads, int size, int qStride, int kvStride, int ring, float base, const float* factors, float eps) {
    __shared__ float partial[32];
    int head = blockIdx.x;
    int token = blockIdx.y;
    int middle = size / 2;
    int i = threadIdx.x;
    int pos = positions[token];
    float angle = (float)pos * powf(base, -2.0f * i / size);
    if(factors) angle /= factors[i];
    float c, s;
    sincosf(angle, &s, &c);

    if(head < heads){
        float* row = q + (u64)token * qStride + head * size;
        float a = row[i], b = row[i + middle];
        float inv = rsqrtf(gemmaBlockSum(a * a + b * b, partial) / size + eps);
        a *= inv * queryNorm[i];
        b *= inv * queryNorm[i + middle];
        row[i] = a * c - b * s;
        row[i + middle] = b * c + a * s;
        return;
    }
    int kv = head - heads;
    int slot = ring > 0 ? pos % ring : pos;
    u64 at = ((u64)slot * kvHeads + kv) * size;
    const float* kr = k + (u64)token * kvStride + kv * size;
    const float* vr = v + (u64)token * kvStride + kv * size;
    float va = vr[i], vb = vr[i + middle];
    float vinv = rsqrtf(gemmaBlockSum(va * va + vb * vb, partial) / size + eps);
    float a = kr[i], b = kr[i + middle];
    float inv = rsqrtf(gemmaBlockSum(a * a + b * b, partial) / size + eps);
    values[at + i] = gemmaHalf(va * vinv);
    values[at + i + middle] = gemmaHalf(vb * vinv);
    a *= inv * keyNorm[i];
    b *= inv * keyNorm[i + middle];
    keys[at + i] = gemmaHalf(a * c - b * s);
    keys[at + i + middle] = gemmaHalf(b * c + a * s);
}

template<int HD, int G, int TQ, int KT>
__device__ __forceinline__ void gemmaAttend(const float* q, const u16* keys, const u16* values, const int* positions, float* output, float* scratch, int count, int heads, int kvHeads, int ring, int window, int splits) {
    constexpr int QN = G * TQ;
    constexpr int PAD = HD + 2;
    constexpr int PER = HD / 16;
    __shared__ __align__(16) u16 qs[QN][PAD];
    __shared__ __align__(16) u16 ks[KT][PAD];
    __shared__ __align__(16) u16 vs[KT][PAD];
    __shared__ float ps[QN][KT];
    __shared__ int tokenPos[TQ];

    int kv = blockIdx.x;
    int first = (blockIdx.y / splits) * TQ;
    int split = blockIdx.y % splits;
    int tid = threadIdx.x;

    if(tid < TQ) tokenPos[tid] = first + tid < count ? positions[first + tid] : -1;
    for(int i = tid; i < QN * HD; i += blockDim.x){
        int r = i / HD, d = i % HD;
        int t = first + r / G;
        int h = kv * G + r % G;
        qs[r][d] = gemmaHalf(t < count ? q[(u64)t * heads * HD + (u64)h * HD + d] : 0.0f);
    }
    __syncthreads();

    int lowest = 0x7fffffff, highest = -1;
    for(int t = 0; t < TQ; t++){
        if(tokenPos[t] < 0) continue;
        int from = window > 0 ? max(0, tokenPos[t] - window + 1) : 0;
        lowest = min(lowest, from);
        highest = max(highest, tokenPos[t]);
    }
    if(highest < 0) return;
    int total = highest - lowest + 1;
    int span = (total + splits - 1) / splits;
    int begin = lowest + split * span;
    int end = min(highest + 1, begin + span);

    int row = tid / 16;
    int lane = tid % 16;
    float acc[PER];
    #pragma unroll
    for(int j = 0; j < PER; j++) acc[j] = 0.0f;
    float m = -1e30f, l = 0.0f;
    int myPos = row < QN ? tokenPos[row / G] : -1;

    for(int at = begin; at < end; at += KT){
        for(int i = tid; i < KT * HD / 2; i += blockDim.x){
            int r = i / (HD / 2), d = 2 * (i % (HD / 2));
            int p = at + r;
            u32 kk = 0, vv = 0;
            if(p < end){
                int slot = ring > 0 ? p % ring : p;
                u64 base = ((u64)slot * kvHeads + kv) * HD + d;
                kk = *(const u32*)(keys + base);
                vv = *(const u32*)(values + base);
            }
            *(u32*)&ks[r][d] = kk;
            *(u32*)&vs[r][d] = vv;
        }
        __syncthreads();
        for(int i = tid; i < QN * KT; i += blockDim.x){
            int r = i / KT, j = i % KT;
            int p = at + j;
            int tp = tokenPos[r / G];
            bool valid = tp >= 0 && p < end && p <= tp && (window <= 0 || tp - p < window);
            float score = -1e30f;
            if(valid){
                float sum = 0.0f;
                #pragma unroll 8
                for(int d = 0; d < HD; d += 2){
                    u32 a = *(const u32*)&qs[r][d];
                    u32 b = *(const u32*)&ks[j][d];
                    sum += half((u16)(a & 0xFFFF)) * half((u16)(b & 0xFFFF)) + half((u16)(a >> 16)) * half((u16)(b >> 16));
                }
                score = sum;
            }
            ps[r][j] = score;
        }
        __syncthreads();
        if(row < QN && myPos >= 0){
            float top = m;
            #pragma unroll
            for(int j = 0; j < KT; j++) top = fmaxf(top, ps[row][j]);
            if(top > -1e29f){
                float correct = __expf(m - top);
                l *= correct;
                #pragma unroll
                for(int j = 0; j < PER; j++) acc[j] *= correct;
                #pragma unroll
                for(int j = 0; j < KT; j++){
                    float w = ps[row][j] > -1e29f ? __expf(ps[row][j] - top) : 0.0f;
                    l += w;
                    #pragma unroll
                    for(int e = 0; e < PER; e++) acc[e] += w * half(vs[j][lane + 16 * e]);
                }
                m = top;
            }
        }
        __syncthreads();
    }

    if(row >= QN || myPos < 0) return;
    int t = first + row / G;
    int h = kv * G + row % G;
    if(splits == 1){
        float inv = l > 0.0f ? 1.0f / l : 0.0f;
        float* out = output + (u64)t * heads * HD + (u64)h * HD;
        #pragma unroll
        for(int e = 0; e < PER; e++) out[lane + 16 * e] = acc[e] * inv;
        return;
    }
    float* part = scratch + (((u64)t * heads + h) * splits + split) * (HD + 2);
    if(lane == 0){
        part[HD] = m;
        part[HD + 1] = l;
    }
    #pragma unroll
    for(int e = 0; e < PER; e++) part[lane + 16 * e] = acc[e];
}

extern "C" __global__ void gemmaAttention256(const float* q, const u16* keys, const u16* values, const int* positions, float* output, float* scratch, int count, int heads, int kvHeads, int ring, int window, int splits) {
    gemmaAttend<256, 2, 8, 16>(q, keys, values, positions, output, scratch, count, heads, kvHeads, ring, window, splits);
}

extern "C" __global__ void gemmaAttention512(const float* q, const u16* keys, const u16* values, const int* positions, float* output, float* scratch, int count, int heads, int kvHeads, int ring, int window, int splits) {
    gemmaAttend<512, 8, 2, 8>(q, keys, values, positions, output, scratch, count, heads, kvHeads, ring, window, splits);
}

// pocas filas (un token por bloque): cada hilo hace el producto de una clave con su fila y despues suma los valores de sus dimensiones
// asi trabajan los 256 hilos y el contexto se reparte entre muchos bloques que despues junta gemmaMerge
template<int HD, int G>
__device__ __forceinline__ void gemmaFewAttend(const float* q, const u16* keys, const u16* values, const int* positions, float* output, float* scratch, int heads, int kvHeads, int ring, int window, int splits) {
    constexpr int KT = 256 / G;
    constexpr int PER = HD / KT;
    constexpr int WR = KT / 32;
    constexpr int LPK = HD / 64;
    constexpr int KPW = 32 / LPK;
    // cada clave la leen LPK hilos seguidos con 64 dimensiones cada uno y la consulta va traspuesta para que esos hilos lean seguido
    __shared__ float qs[G][64][LPK];
    __shared__ float ps[G][KT];
    __shared__ int slots[KT];
    __shared__ float rowTop[G], rowScale[G], rowSum[G];
    __shared__ float warpPart[8];
    int kv = blockIdx.x;
    int split = blockIdx.y % splits;
    int t = blockIdx.y / splits;
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    int r = tid / KT, j = tid % KT;
    int pos = positions[t];
    int from = window > 0 ? max(0, pos - window + 1) : 0;
    int span = (pos - from + splits) / splits;
    int begin = from + split * span;
    int end = min(pos + 1, begin + span);
    for(int i = tid; i < G * HD; i += 256){
        int row = i / HD, d = i % HD;
        qs[row][d % 64][d / 64] = q[(u64)t * heads * HD + (u64)(kv * G + row) * HD + d];
    }
    if(tid < G){
        rowTop[tid] = -1e30f;
        rowSum[tid] = 0.0f;
        rowScale[tid] = 1.0f;
    }
    float acc[PER];
    #pragma unroll
    for(int e = 0; e < PER; e++) acc[e] = 0.0f;
    __syncthreads();
    for(int at = begin; at < end; at += KT){
        int live = min(KT, end - at);
        int group = lane / LPK, lp = lane % LPK;
        for(int base = warp * KPW; base < live; base += 8 * KPW){
            int k = base + group;
            bool valid = k < live;
            float kf[64];
            if(valid){
                int p = at + k;
                int slot = ring > 0 ? p % ring : p;
                if(lp == 0) slots[k] = slot;
                const uint4* kr = (const uint4*)(keys + ((u64)slot * kvHeads + kv) * HD + lp * 64);
                #pragma unroll
                for(int c = 0; c < 8; c++){
                    uint4 w = __ldg(kr + c);
                    kf[8 * c + 0] = half((u16)(w.x & 0xFFFF));
                    kf[8 * c + 1] = half((u16)(w.x >> 16));
                    kf[8 * c + 2] = half((u16)(w.y & 0xFFFF));
                    kf[8 * c + 3] = half((u16)(w.y >> 16));
                    kf[8 * c + 4] = half((u16)(w.z & 0xFFFF));
                    kf[8 * c + 5] = half((u16)(w.z >> 16));
                    kf[8 * c + 6] = half((u16)(w.w & 0xFFFF));
                    kf[8 * c + 7] = half((u16)(w.w >> 16));
                }
            } else {
                #pragma unroll
                for(int e = 0; e < 64; e++) kf[e] = 0.0f;
            }
            #pragma unroll
            for(int row = 0; row < G; row++){
                float sum = 0.0f;
                #pragma unroll
                for(int e = 0; e < 64; e++) sum += kf[e] * qs[row][e][lp];
                #pragma unroll
                for(int step = LPK / 2; step > 0; step >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, step);
                if(valid && lp == row % LPK) ps[row][k] = sum;
            }
        }
        __syncthreads();
        float score = j < live ? ps[r][j] : -1e30f;
        float top = score;
        for(int step = 16; step > 0; step >>= 1) top = fmaxf(top, __shfl_xor_sync(0xffffffff, top, step));
        if(lane == 0) warpPart[warp] = top;
        __syncthreads();
        if(tid < G){
            float best = rowTop[tid];
            for(int w = 0; w < WR; w++) best = fmaxf(best, warpPart[tid * WR + w]);
            rowScale[tid] = __expf(rowTop[tid] - best);
            rowTop[tid] = best;
        }
        __syncthreads();
        float weight = score > -1e29f ? __expf(score - rowTop[r]) : 0.0f;
        ps[r][j] = weight;
        float total = weight;
        for(int step = 16; step > 0; step >>= 1) total += __shfl_xor_sync(0xffffffff, total, step);
        if(lane == 0) warpPart[warp] = total;
        __syncthreads();
        if(tid < G){
            float sum = 0.0f;
            for(int w = 0; w < WR; w++) sum += warpPart[tid * WR + w];
            rowSum[tid] = rowSum[tid] * rowScale[tid] + sum;
        }
        float scale = rowScale[r];
        #pragma unroll
        for(int e = 0; e < PER; e++) acc[e] *= scale;
        for(int k = 0; k < live; k++){
            float w = ps[r][k];
            const u16* line = values + ((u64)slots[k] * kvHeads + kv) * HD + j * PER;
            if(PER == 2){
                u32 v = *(const u32*)line;
                acc[0] += w * half((u16)(v & 0xFFFF));
                acc[1 % PER] += w * half((u16)(v >> 16));
            } else {
                #pragma unroll
                for(int c = 0; c < PER / 8; c++){
                    uint4 v = __ldg((const uint4*)line + c);
                    acc[(8 * c + 0) % PER] += w * half((u16)(v.x & 0xFFFF));
                    acc[(8 * c + 1) % PER] += w * half((u16)(v.x >> 16));
                    acc[(8 * c + 2) % PER] += w * half((u16)(v.y & 0xFFFF));
                    acc[(8 * c + 3) % PER] += w * half((u16)(v.y >> 16));
                    acc[(8 * c + 4) % PER] += w * half((u16)(v.z & 0xFFFF));
                    acc[(8 * c + 5) % PER] += w * half((u16)(v.z >> 16));
                    acc[(8 * c + 6) % PER] += w * half((u16)(v.w & 0xFFFF));
                    acc[(8 * c + 7) % PER] += w * half((u16)(v.w >> 16));
                }
            }
        }
        __syncthreads();
    }
    int h = kv * G + r;
    if(splits == 1){
        float inv = rowSum[r] > 0.0f ? 1.0f / rowSum[r] : 0.0f;
        float* out = output + (u64)t * heads * HD + (u64)h * HD + j * PER;
        #pragma unroll
        for(int e = 0; e < PER; e++) out[e] = acc[e] * inv;
        return;
    }
    float* part = scratch + (((u64)t * heads + h) * splits + split) * (HD + 2);
    #pragma unroll
    for(int e = 0; e < PER; e++) part[j * PER + e] = acc[e];
    if(j == 0){
        part[HD] = rowTop[r];
        part[HD + 1] = rowSum[r];
    }
}

extern "C" __global__ void __launch_bounds__(256) gemmaFew256(const float* q, const u16* keys, const u16* values, const int* positions, float* output, float* scratch, int heads, int kvHeads, int ring, int window, int splits) {
    gemmaFewAttend<256, 2>(q, keys, values, positions, output, scratch, heads, kvHeads, ring, window, splits);
}

extern "C" __global__ void __launch_bounds__(256) gemmaFew512(const float* q, const u16* keys, const u16* values, const int* positions, float* output, float* scratch, int heads, int kvHeads, int ring, int window, int splits) {
    gemmaFewAttend<512, 8>(q, keys, values, positions, output, scratch, heads, kvHeads, ring, window, splits);
}

#if __CUDA_ARCH__ >= 800
__device__ __forceinline__ void gemmaMma(float* c, const u32* a, u32 b0, u32 b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void gemmaQuad(u32* r, const void* p) {
    u32 at = (u32)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(at));
}

__device__ __forceinline__ void gemmaQuadT(u32* r, const void* p) {
    u32 at = (u32)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(at));
}

__device__ __forceinline__ u32 gemmaTwo(float low, float high) {
    u32 r;
    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(high), "f"(low));
    return r;
}

// atencion de prompt con nucleos tensoriales: 64 filas por bloque (tokens por cabezas de su clave), cada warp lleva 16 filas
// y en las capas de 512 dos warps comparten filas y cada uno suma la mitad de las dimensiones de salida
// las claves y despues los valores de un tramo pasan por la memoria compartida y el softmax se lleva en linea
template<int HD, int G, int KT>
__device__ __forceinline__ void gemmaFlashAttend(const float* q, const u16* keys, const u16* values, const int* positions, float* output, int count, int heads, int kvHeads, int ring, int window) {
    constexpr int ROWS = 64;
    constexpr int QT = ROWS / G;
    constexpr int PAD = HD + 8;
    constexpr int HALVES = HD / 256;
    constexpr int NT = KT / 8;
    extern __shared__ __align__(16) u8 room[];
    u16* qs = (u16*)room;
    u16* ks = qs + ROWS * PAD;
    int* tokenPos = (int*)(ks + KT * PAD);
    int* slotOf = tokenPos + QT;
    int kv = blockIdx.x;
    int first = blockIdx.y * QT;
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    int threads = blockDim.x;
    int group = warp / HALVES;
    int part = warp % HALVES;
    int g = lane >> 2, t4 = lane & 3;
    if(tid < QT) tokenPos[tid] = first + tid < count ? positions[first + tid] : -1;
    for(int i = tid; i < ROWS * HD / 8; i += threads){
        int row = i / (HD / 8), chunk = i % (HD / 8);
        int tok = first + row / G;
        uint4 packed = make_uint4(0, 0, 0, 0);
        if(tok < count){
            const float4* from = (const float4*)(q + ((u64)tok * heads + kv * G + row % G) * HD + chunk * 8);
            float4 a = from[0], b = from[1];
            packed = make_uint4(gemmaTwo(a.x, a.y), gemmaTwo(a.z, a.w), gemmaTwo(b.x, b.y), gemmaTwo(b.z, b.w));
        }
        *(uint4*)(qs + row * PAD + chunk * 8) = packed;
    }
    __syncthreads();
    int lowest = 0x7fffffff, highest = -1;
    for(int i = 0; i < QT; i++){
        if(tokenPos[i] < 0) continue;
        lowest = min(lowest, window > 0 ? max(0, tokenPos[i] - window + 1) : 0);
        highest = max(highest, tokenPos[i]);
    }
    // las dos filas de este hilo dentro de las 16 del warp: g y g+8
    int rowA = group * 16 + g, rowB = rowA + 8;
    int posA = tokenPos[rowA / G], posB = tokenPos[rowB / G];
    float o[32][4];
    #pragma unroll
    for(int n = 0; n < 32; n++) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
    float topA = -1e30f, topB = -1e30f, sumA = 0.0f, sumB = 0.0f;
    for(int at = lowest; at <= highest; at += KT){
        int end = min(highest + 1, at + KT);
        if(tid < KT){
            int p = at + tid;
            slotOf[tid] = p < end ? (ring > 0 ? p % ring : p) : -1;
        }
        __syncthreads();
        for(int i = tid; i < KT * HD / 8; i += threads){
            int key = i / (HD / 8), chunk = i % (HD / 8);
            int slot = slotOf[key];
            uint4 v = make_uint4(0, 0, 0, 0);
            if(slot >= 0) v = __ldg((const uint4*)(keys + ((u64)slot * kvHeads + kv) * HD + chunk * 8));
            *(uint4*)(ks + key * PAD + chunk * 8) = v;
        }
        __syncthreads();
        float s[NT][4];
        #pragma unroll
        for(int j = 0; j < NT; j++) s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.0f;
        #pragma unroll 2
        for(int kk = 0; kk < HD / 16; kk++){
            u32 a[4];
            gemmaQuad(a, qs + (group * 16 + (lane & 15)) * PAD + kk * 16 + (lane >> 4) * 8);
            #pragma unroll
            for(int j = 0; j < NT; j += 2){
                u32 b[4];
                gemmaQuad(b, ks + (j * 8 + (lane & 7) + (lane >> 4) * 8) * PAD + kk * 16 + ((lane >> 3) & 1) * 8);
                gemmaMma(s[j], a, b[0], b[1]);
                gemmaMma(s[j + 1], a, b[2], b[3]);
            }
        }
        // mascara causal y de ventana y maximo de cada fila entre los cuatro hilos que la comparten
        float newA = topA, newB = topB;
        #pragma unroll
        for(int j = 0; j < NT; j++){
            #pragma unroll
            for(int c = 0; c < 2; c++){
                int p = at + j * 8 + t4 * 2 + c;
                bool okA = posA >= 0 && p < end && p <= posA && (window <= 0 || posA - p < window);
                bool okB = posB >= 0 && p < end && p <= posB && (window <= 0 || posB - p < window);
                s[j][c] = okA ? s[j][c] : -1e30f;
                s[j][2 + c] = okB ? s[j][2 + c] : -1e30f;
                newA = fmaxf(newA, s[j][c]);
                newB = fmaxf(newB, s[j][2 + c]);
            }
        }
        newA = fmaxf(newA, __shfl_xor_sync(0xffffffff, newA, 1));
        newA = fmaxf(newA, __shfl_xor_sync(0xffffffff, newA, 2));
        newB = fmaxf(newB, __shfl_xor_sync(0xffffffff, newB, 1));
        newB = fmaxf(newB, __shfl_xor_sync(0xffffffff, newB, 2));
        float keepA = topA > -1e29f ? __expf(topA - newA) : 0.0f;
        float keepB = topB > -1e29f ? __expf(topB - newB) : 0.0f;
        topA = newA;
        topB = newB;
        sumA *= keepA;
        sumB *= keepB;
        #pragma unroll
        for(int j = 0; j < NT; j++){
            #pragma unroll
            for(int c = 0; c < 2; c++){
                float pa = s[j][c] > -1e29f ? __expf(s[j][c] - newA) : 0.0f;
                float pb = s[j][2 + c] > -1e29f ? __expf(s[j][2 + c] - newB) : 0.0f;
                s[j][c] = pa;
                s[j][2 + c] = pb;
                sumA += pa;
                sumB += pb;
            }
        }
        #pragma unroll
        for(int n = 0; n < 32; n++){
            o[n][0] *= keepA;
            o[n][1] *= keepA;
            o[n][2] *= keepB;
            o[n][3] *= keepB;
        }
        __syncthreads();
        for(int i = tid; i < KT * HD / 8; i += threads){
            int key = i / (HD / 8), chunk = i % (HD / 8);
            int slot = slotOf[key];
            uint4 v = make_uint4(0, 0, 0, 0);
            if(slot >= 0) v = __ldg((const uint4*)(values + ((u64)slot * kvHeads + kv) * HD + chunk * 8));
            *(uint4*)(ks + key * PAD + chunk * 8) = v;
        }
        __syncthreads();
        #pragma unroll
        for(int c = 0; c < KT / 16; c++){
            u32 a[4] = {gemmaTwo(s[2 * c][0], s[2 * c][1]), gemmaTwo(s[2 * c][2], s[2 * c][3]), gemmaTwo(s[2 * c + 1][0], s[2 * c + 1][1]), gemmaTwo(s[2 * c + 1][2], s[2 * c + 1][3])};
            #pragma unroll
            for(int n = 0; n < 32; n += 2){
                u32 b[4];
                gemmaQuadT(b, ks + (c * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * PAD + part * 256 + n * 8 + (lane >> 4) * 8);
                gemmaMma(o[n], a, b[0], b[1]);
                gemmaMma(o[n + 1], a, b[2], b[3]);
            }
        }
        __syncthreads();
    }
    sumA += __shfl_xor_sync(0xffffffff, sumA, 1);
    sumA += __shfl_xor_sync(0xffffffff, sumA, 2);
    sumB += __shfl_xor_sync(0xffffffff, sumB, 1);
    sumB += __shfl_xor_sync(0xffffffff, sumB, 2);
    float invA = sumA > 0.0f ? 1.0f / sumA : 0.0f;
    float invB = sumB > 0.0f ? 1.0f / sumB : 0.0f;
    int tokA = first + rowA / G, tokB = first + rowB / G;
    float* outA = output + ((u64)tokA * heads + kv * G + rowA % G) * HD + part * 256 + t4 * 2;
    float* outB = output + ((u64)tokB * heads + kv * G + rowB % G) * HD + part * 256 + t4 * 2;
    #pragma unroll
    for(int n = 0; n < 32; n++){
        if(tokA < count) *(float2*)(outA + n * 8) = make_float2(o[n][0] * invA, o[n][1] * invA);
        if(tokB < count) *(float2*)(outB + n * 8) = make_float2(o[n][2] * invB, o[n][3] * invB);
    }
}
#endif

extern "C" __global__ void __launch_bounds__(128) gemmaFlash256(const float* q, const u16* keys, const u16* values, const int* positions, float* output, int count, int heads, int kvHeads, int ring, int window) {
#if __CUDA_ARCH__ >= 800
    gemmaFlashAttend<256, 2, 64>(q, keys, values, positions, output, count, heads, kvHeads, ring, window);
#endif
}

extern "C" __global__ void __launch_bounds__(256) gemmaFlash512(const float* q, const u16* keys, const u16* values, const int* positions, float* output, int count, int heads, int kvHeads, int ring, int window) {
#if __CUDA_ARCH__ >= 800
    gemmaFlashAttend<512, 8, 32>(q, keys, values, positions, output, count, heads, kvHeads, ring, window);
#endif
}

// expertos de pocos tokens con el kernel de una fila por warp: el bloque lee el experto de su casilla en la tarjeta
template<int TYPE>
__device__ __forceinline__ void gemmaExpertDirect(const u8* weights, u64 rowBytes, u64 expertBytes, const int* ids, int perToken, const i8* packed, const float* scales, const int* sums, float* output, int rows, int cols, const GridTables& t) {
    int slot = blockIdx.y;
    u64 at = (u64)(slot / perToken) * (cols / 32);
    typedRows<TYPE, 1, true>(weights + expertBytes * ids[slot], rowBytes, packed + at * 32, scales + at, sums + at, nullptr, nullptr, nullptr, output + (u64)rows * slot, rows, cols, 0, t);
}

#define GEMMA_EXPERTS(T) extern "C" __global__ void gemmaExperts##T(const u8* weights, u64 rowBytes, u64 expertBytes, const int* ids, int perToken, const i8* packed, const float* scales, const int* sums, float* output, int rows, int cols, GridTables t) { gemmaExpertDirect<T>(weights, rowBytes, expertBytes, ids, perToken, packed, scales, sums, output, rows, cols, t); }
GEMMA_EXPERTS(12)
GEMMA_EXPERTS(13)
GEMMA_EXPERTS(14)
GEMMA_EXPERTS(20)
GEMMA_EXPERTS(21)
GEMMA_EXPERTS(23)
GEMMA_EXPERTS(42)
GEMMA_EXPERTS(7)
GEMMA_EXPERTS(8)

extern "C" __global__ void gemmaMerge(const float* scratch, float* output, int heads, int size, int splits) {
    int h = blockIdx.x, t = blockIdx.y;
    const float* parts = scratch + ((u64)t * heads + h) * splits * (size + 2);
    float top = -1e30f;
    for(int s = 0; s < splits; s++){
        const float* p = parts + (u64)s * (size + 2);
        if(p[size + 1] > 0.0f) top = fmaxf(top, p[size]);
    }
    float total = 0.0f;
    for(int s = 0; s < splits; s++){
        const float* p = parts + (u64)s * (size + 2);
        if(p[size + 1] > 0.0f) total += p[size + 1] * __expf(p[size] - top);
    }
    float inv = total > 0.0f ? 1.0f / total : 0.0f;
    float* out = output + (u64)t * heads * size + (u64)h * size;
    for(int d = threadIdx.x; d < size; d += blockDim.x){
        float v = 0.0f;
        for(int s = 0; s < splits; s++){
            const float* p = parts + (u64)s * (size + 2);
            if(p[size + 1] > 0.0f) v += p[d] * __expf(p[size] - top);
        }
        out[d] = v * inv;
    }
}

extern "C" __global__ void gemmaMixSorted(const float* expertOut, const float* weights, const float* downScale, const int* ids, const int* rankOf, float* output, int used, int cols) {
    int token = blockIdx.x;
    for(int i = threadIdx.x; i < cols; i += blockDim.x){
        float v = 0.0f;
        for(int k = 0; k < used; k++){
            int slot = token * used + k;
            float w = weights[slot] * (downScale ? downScale[ids[slot]] : 1.0f);
            v = fmaf(w, expertOut[(u64)rankOf[slot] * cols + i], v);
        }
        output[(u64)token * cols + i] = v;
    }
}

extern "C" __global__ void gemmaRouteLogits(const float* input, const float* router, float* logits, int cols, int experts) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int e = blockIdx.x * (blockDim.x >> 5) + warp;
    int token = blockIdx.y;
    if(e >= experts) return;
    const float* x = input + (u64)token * cols;
    const float* w = router + (u64)e * cols;
    float sum = 0.0f;
    for(int i = lane; i < cols; i += 32) sum += w[i] * x[i];
    for(int step = 16; step > 0; step >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, step);
    if(lane == 0) logits[(u64)token * experts + e] = sum;
}

extern "C" __global__ void gemmaRoutePick(float* logits, int* ids, float* weights, int experts, int used) {
    int token = blockIdx.x;
    int lane = threadIdx.x;
    float* row = logits + (u64)token * experts;
    float top = -1e30f;
    for(int e = lane; e < experts; e += 32) top = fmaxf(top, row[e]);
    for(int step = 16; step > 0; step >>= 1) top = fmaxf(top, __shfl_xor_sync(0xffffffff, top, step));
    for(int e = lane; e < experts; e += 32) row[e] = __expf(row[e] - top);
    __syncwarp();
    float total = 0.0f;
    if(lane == 0) for(int e = 0; e < experts; e++) total += row[e];
    total = __shfl_sync(0xffffffff, total, 0);
    float picked = 0.0f;
    int* outIds = ids + token * used;
    float* outWeights = weights + token * used;
    for(int k = 0; k < used; k++){
        float bestValue = -1.0f;
        int best = experts;
        for(int e = lane; e < experts; e += 32){
            float v = row[e];
            if(v > bestValue){
                bestValue = v;
                best = e;
            }
        }
        for(int step = 16; step > 0; step >>= 1){
            float otherValue = __shfl_xor_sync(0xffffffff, bestValue, step);
            int other = __shfl_xor_sync(0xffffffff, best, step);
            if(otherValue > bestValue || (otherValue == bestValue && other < best)){
                bestValue = otherValue;
                best = other;
            }
        }
        if(best >= experts) best = 0;
        if(lane == 0){
            outIds[k] = best;
            outWeights[k] = bestValue / total;
            picked += bestValue / total;
            row[best] = -2.0f;
        }
        __syncwarp();
    }
    if(lane == 0) for(int k = 0; k < used; k++) outWeights[k] /= fmaxf(picked, 6.103515625e-5f);
}

extern "C" __global__ void gemmaRoute(const float* input, const float* router, int* ids, float* weights, int cols, int experts, int used) {
    extern __shared__ float logits[];
    int token = blockIdx.x;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int warps = blockDim.x >> 5;
    const float* x = input + (u64)token * cols;
    for(int e = warp; e < experts; e += warps){
        const float* w = router + (u64)e * cols;
        float sum = 0.0f;
        for(int i = lane; i < cols; i += 32) sum += w[i] * x[i];
        for(int step = 16; step > 0; step >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, step);
        if(lane == 0) logits[e] = sum;
    }
    __syncthreads();
    if(threadIdx.x != 0) return;
    float top = -1e30f;
    for(int e = 0; e < experts; e++) top = fmaxf(top, logits[e]);
    float total = 0.0f;
    for(int e = 0; e < experts; e++){
        logits[e] = __expf(logits[e] - top);
        total += logits[e];
    }
    float picked = 0.0f;
    int* outIds = ids + token * used;
    float* outWeights = weights + token * used;
    for(int k = 0; k < used; k++){
        int best = 0;
        float bestValue = -1.0f;
        for(int e = 0; e < experts; e++){
            if(logits[e] > bestValue){
                bestValue = logits[e];
                best = e;
            }
        }
        outIds[k] = best;
        outWeights[k] = bestValue / total;
        picked += bestValue / total;
        logits[best] = -2.0f;
    }
    for(int k = 0; k < used; k++) outWeights[k] /= fmaxf(picked, 6.103515625e-5f);
}

extern "C" __global__ void gemmaExperts(int type, const u8* weights, u64 rowBytes, u64 expertBytes, const int* ids, int perToken, const float* input, const i8* packed, const float* scales, const int* sums, float* output, int rows, int cols, GridTables t) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if(row >= rows) return;
    int slot = blockIdx.y;
    int token = slot / perToken;
    const u8* line = weights + expertBytes * ids[slot] + rowBytes * row;
    u64 at = (u64)token * cols;
    float sum = rowDotQ8(type, line, cols, input + at, packed + at, scales + at / 32, sums + at / 32, t);
    if(threadIdx.x == 0) output[(u64)rows * slot + row] = sum;
}

extern "C" __global__ void gemmaMix(const float* expertOut, const float* weights, const float* downScale, const int* ids, float* output, int used, int cols) {
    int token = blockIdx.x;
    for(int i = threadIdx.x; i < cols; i += blockDim.x){
        float v = 0.0f;
        for(int k = 0; k < used; k++){
            int slot = token * used + k;
            float w = weights[slot] * (downScale ? downScale[ids[slot]] : 1.0f);
            v = fmaf(w, expertOut[(u64)slot * cols + i], v);
        }
        output[(u64)token * cols + i] = v;
    }
}

extern "C" __global__ void gemmaCap(const float* input, float* output, int count, float cap) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= count) return;
    float v = input[i];
    output[i] = cap > 0.0f ? cap * tanhf(v / cap) : v;
}
)";
