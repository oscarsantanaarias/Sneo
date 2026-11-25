#pragma once
#include "hy4.h"

struct GlmLinear {
    GpuMatrix q, k, v, out, beta, fa, fb, ga, gb;
    uint64_t convQ = 0, convK = 0, convV = 0, dtBias = 0, decay = 0, norm = 0;
    uint64_t state = 0, history = 0;
};

struct GlmShape {
    int linearHeads = 64;
    int linearDim = 128;
    int linearWidth = 8192;
    int convTaps = 4;
    float lowerBound = -5.0f;
    int sinkhorn = 20;
    int headKey = 256;
    int hitRows = 64;
    int attendRows = 64;
    std::vector<char> linear;
    std::vector<int> linearSlot;
};

struct GlmBuffers {
    uint64_t hcw = 0, sinks = 0;
    uint64_t lq = 0, lk = 0, lv = 0, lf = 0, lfa = 0, lbeta = 0, lgate = 0, lga = 0, lo = 0, lon = 0;
    uint64_t kvRaw = 0, idxGateRaw = 0;
    std::vector<GpuMatrix> idxGate;
    std::vector<uint64_t> idxGateRing;
    std::vector<std::vector<uint8_t>> saved;
    std::vector<size_t> savedTokens;
    uint64_t stage = 0;
    uint8_t* pinned = nullptr;
    int stageSlots = 0;
    int pairsMost = 0;
    int* hostGroups = nullptr;
    uint64_t groupsOn = 0;
    uint64_t inValues = 0, inScales = 0, inSums = 0;
    uint64_t gateOut = 0, upOut = 0, hidOut = 0, hidValues = 0, hidScales = 0, hidSums = 0;
    void* copyStream = nullptr;
    void* copied[2] = {};
    void* done[2] = {};
    uint64_t uploaded = 0;
    double uploadMs = 0, wideMs = 0;
};

const char* glmKernelSource = R"(
extern "C" __global__ void glmHcDots(const float* streams, const float* fn, float* mixes, int width, int chunk) {
    __shared__ float red[25][8];
    int t = blockIdx.x / ((width + chunk - 1) / chunk), part = blockIdx.x % ((width + chunk - 1) / chunk);
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    const float* flat = streams + (u64)t * width;
    int from = part * chunk, to = min(width, from + chunk);
    float acc[25];
    #pragma unroll
    for(int r = 0; r < 25; r++) acc[r] = 0.0f;
    for(int c = from + threadIdx.x; c < to; c += blockDim.x){
        float x = flat[c];
        acc[24] += x * x;
        #pragma unroll
        for(int r = 0; r < 24; r++) acc[r] += fn[(u64)r * width + c] * x;
    }
    #pragma unroll
    for(int r = 0; r < 25; r++){
        float v = acc[r];
        for(int s = 16; s > 0; s >>= 1) v += __shfl_xor_sync(0xffffffff, v, s);
        if(lane == 0) red[r][warp] = v;
    }
    __syncthreads();
    if(threadIdx.x < 25){
        float v = 0.0f;
        for(int w = 0; w < warps; w++) v += red[threadIdx.x][w];
        atomicAdd(mixes + (u64)t * 32 + threadIdx.x, v);
    }
}

extern "C" __global__ void glmHcCombine(const float* streams, float* mixes, const float* scale, const float* base, const float* norm, float* out, float* weights, int hidden, float eps, float hcEps, int iters, i8* packed, float* scales, int* sums) {
    __shared__ float red[32];
    __shared__ float pre[4];
    __shared__ float y[4096];
    int t = blockIdx.x;
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    int width = 4 * hidden;
    const float* flat = streams + (u64)t * width;
    if(threadIdx.x == 0){
        float mix[25];
        for(int i = 0; i < 25; i++) mix[i] = mixes[(u64)t * 32 + i];
        float inv = rsqrtf(mix[24] / width + eps);
        float* w = weights + (u64)t * 20;
        for(int i = 0; i < 4; i++){
            pre[i] = hySigmoid(mix[i] * inv * scale[0] + base[i]) + hcEps;
            w[i] = 2.0f * hySigmoid(mix[4 + i] * inv * scale[1] + base[4 + i]);
        }
        float comb[16];
        for(int i = 0; i < 4; i++){
            float top = -1e30f;
            for(int j = 0; j < 4; j++){
                comb[i * 4 + j] = mix[8 + i * 4 + j] * inv * scale[2] + base[8 + i * 4 + j];
                top = fmaxf(top, comb[i * 4 + j]);
            }
            float total = 0.0f;
            for(int j = 0; j < 4; j++){
                comb[i * 4 + j] = __expf(comb[i * 4 + j] - top);
                total += comb[i * 4 + j];
            }
            for(int j = 0; j < 4; j++) comb[i * 4 + j] = comb[i * 4 + j] / total + hcEps;
        }
        for(int j = 0; j < 4; j++){
            float total = hcEps;
            for(int i = 0; i < 4; i++) total += comb[i * 4 + j];
            for(int i = 0; i < 4; i++) comb[i * 4 + j] /= total;
        }
        for(int round = 1; round < iters; round++){
            for(int i = 0; i < 4; i++){
                float total = hcEps;
                for(int j = 0; j < 4; j++) total += comb[i * 4 + j];
                for(int j = 0; j < 4; j++) comb[i * 4 + j] /= total;
            }
            for(int j = 0; j < 4; j++){
                float total = hcEps;
                for(int i = 0; i < 4; i++) total += comb[i * 4 + j];
                for(int i = 0; i < 4; i++) comb[i * 4 + j] /= total;
            }
        }
        for(int i = 0; i < 16; i++) w[4 + i] = comb[i];
    }
    __syncthreads();
    if(threadIdx.x < 32) mixes[(u64)t * 32 + threadIdx.x] = 0.0f;
    float sq = 0.0f;
    for(int c = threadIdx.x; c < hidden; c += blockDim.x){
        float v = pre[0] * flat[c] + pre[1] * flat[hidden + c] + pre[2] * flat[2 * hidden + c] + pre[3] * flat[3 * hidden + c];
        y[c] = v;
        sq += v * v;
    }
    for(int s = 16; s > 0; s >>= 1) sq += __shfl_xor_sync(0xffffffff, sq, s);
    if(lane == 0) red[warp] = sq;
    __syncthreads();
    float total = 0.0f;
    for(int w = 0; w < warps; w++) total += red[w];
    float inv2 = rsqrtf(total / hidden + eps);
    for(int c = threadIdx.x; c < hidden; c += blockDim.x) out[(u64)t * hidden + c] = y[c] * inv2 * norm[c];
    if(!packed) return;
    for(int g = warp; g < hidden / 32; g += warps){
        int c = g * 32 + lane;
        hyQuant(y[c] * inv2 * norm[c], (u64)t * hidden + c, packed, scales, sums);
    }
}

extern "C" __global__ void glmHcSpread(float* streams, const float* x, const float* weights, int hidden, int chunk) {
    int parts = (hidden + chunk - 1) / chunk;
    int t = blockIdx.x / parts, piece = blockIdx.x % parts;
    const float* w = weights + (u64)t * 20;
    float* s = streams + (u64)t * 4 * hidden;
    const float* src = x + (u64)t * hidden;
    int from = piece * chunk, to = min(hidden, from + chunk);
    for(int c = from + threadIdx.x; c < to; c += blockDim.x){
        float old[4];
        for(int i = 0; i < 4; i++) old[i] = s[(u64)i * hidden + c];
        float v = src[c];
        for(int d = 0; d < 4; d++){
            float mixed = w[d] * v;
            for(int i = 0; i < 4; i++) mixed += w[4 + i * 4 + d] * old[i];
            s[(u64)d * hidden + c] = mixed;
        }
    }
}

extern "C" __global__ void glmHeadNorm(const float* streams, const float* norm, float* out, int hidden, float eps) {
    __shared__ float part[32];
    __shared__ float y[4096];
    int t = blockIdx.x;
    const float* s = streams + (u64)t * 4 * hidden;
    float sq = 0.0f;
    for(int c = threadIdx.x; c < hidden; c += blockDim.x){
        float v = 0.25f * (s[c] + s[hidden + c] + s[2 * hidden + c] + s[3 * hidden + c]);
        y[c] = v;
        sq += v * v;
    }
    float inv = rsqrtf(hyBlockSum(sq, part) / hidden + eps);
    for(int c = threadIdx.x; c < hidden; c += blockDim.x) out[(u64)t * hidden + c] = y[c] * inv * norm[c];
}

extern "C" __global__ void glmLatent(const float* raw, const float* norm, float* out, float eps) {
    __shared__ float part[32];
    int t = blockIdx.x;
    const float* src = raw + (u64)t * 512;
    float sq = 0.0f;
    for(int c = threadIdx.x; c < 512; c += blockDim.x) sq += src[c] * src[c];
    float inv = rsqrtf(hyBlockSum(sq, part) / 512.0f + eps);
    float* dst = out + (u64)t * 576;
    for(int c = threadIdx.x; c < 576; c += blockDim.x) dst[c] = c < 512 ? src[c] * inv * norm[c] : 0.0f;
}

extern "C" __global__ void glmAbsorbRows(const float* q, const u16* kb, float* qfull, int heads, int parts) {
    __shared__ float qn[256];
    int h = blockIdx.x, t = blockIdx.y / parts, part = blockIdx.y % parts;
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const float* src = q + ((u64)t * heads + h) * 256;
    for(int k = threadIdx.x; k < 256; k += blockDim.x) qn[k] = src[k];
    __syncthreads();
    float* dst = qfull + ((u64)t * heads + h) * 576;
    const u16* w = kb + (u64)h * 512 * 256;
    int per = (512 + parts - 1) / parts;
    int end = min(512, (part + 1) * per);
    for(int r = part * per + warp; r < end; r += 8){
        const u16* line = w + (u64)r * 256;
        float sum = 0.0f;
        for(int k = lane; k < 256; k += 32) sum += hyHalf(line[k]) * qn[k];
        for(int s = 16; s > 0; s >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, s);
        if(lane == 0) dst[r] = sum;
    }
    if(part == 0) for(int k = threadIdx.x; k < 64; k += blockDim.x) dst[512 + k] = 0.0f;
}

__device__ __forceinline__ float glmSilu(float x) {
    return x / (1.0f + __expf(-x));
}

extern "C" __global__ void glmConv(float* q, float* k, float* v, const float* wq, const float* wk, const float* wv, float* history, int n, int width) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= 3 * width) return;
    int which = index / width, c = index % width;
    float* data = which == 0 ? q : which == 1 ? k : v;
    const float* w = (which == 0 ? wq : which == 1 ? wk : wv) + (u64)c * 4;
    float* past = history + (u64)which * 3 * width + c;
    float h0 = past[0], h1 = past[width], h2 = past[2 * (u64)width];
    float w0 = w[0], w1 = w[1], w2 = w[2], w3 = w[3];
    for(int t = 0; t < n; t++){
        float x = data[(u64)t * width + c];
        data[(u64)t * width + c] = glmSilu(w0 * h0 + w1 * h1 + w2 * h2 + w3 * x);
        h0 = h1;
        h1 = h2;
        h2 = x;
    }
    past[0] = h0;
    past[width] = h1;
    past[2 * (u64)width] = h2;
}

extern "C" __global__ void glmKdaScan(const float* q, const float* k, const float* v, const float* f, const float* betaRaw, const float* dtBias, const float* decay, float* state, float* out, int n, int heads, float lowerBound) {
    __shared__ float qs[128];
    __shared__ float ks[128];
    __shared__ float gs[128];
    __shared__ float red[2][4];
    int h = blockIdx.x, j = threadIdx.x;
    int lane = j & 31, warp = j >> 5;
    int width = heads * 128;
    float* S = state + (u64)h * 128 * 128;
    float column[128];
    #pragma unroll
    for(int r = 0; r < 128; r++) column[r] = S[(u64)r * 128 + j];
    float rate = decay[h];
    float shift = dtBias[h * 128 + j];
    for(int t = 0; t < n; t++){
        u64 at = (u64)t * width + h * 128 + j;
        float qv = q[at], kv = k[at];
        float a = qv * qv, b = kv * kv;
        for(int s = 16; s > 0; s >>= 1){
            a += __shfl_xor_sync(0xffffffff, a, s);
            b += __shfl_xor_sync(0xffffffff, b, s);
        }
        __syncthreads();
        if(lane == 0){
            red[0][warp] = a;
            red[1][warp] = b;
        }
        __syncthreads();
        float qq = red[0][0] + red[0][1] + red[0][2] + red[0][3];
        float kk = red[1][0] + red[1][1] + red[1][2] + red[1][3];
        qs[j] = qv / sqrtf(qq + 1e-6f) * 0.08838834764831845f;
        ks[j] = kv / sqrtf(kk + 1e-6f);
        gs[j] = __expf(lowerBound * hySigmoid(rate * (f[at] + shift)));
        __syncthreads();
        float beta = hySigmoid(betaRaw[(u64)t * heads + h]);
        float memory = 0.0f;
        #pragma unroll
        for(int r = 0; r < 128; r++){
            column[r] *= gs[r];
            memory += column[r] * ks[r];
        }
        float delta = (v[at] - memory) * beta;
        float o = 0.0f;
        #pragma unroll
        for(int r = 0; r < 128; r++){
            column[r] += ks[r] * delta;
            o += column[r] * qs[r];
        }
        out[at] = o;
    }
    #pragma unroll
    for(int r = 0; r < 128; r++) S[(u64)r * 128 + j] = column[r];
}

extern "C" __global__ void glmExpertRows(const u8* base, u64 rowBytes, u64 slotBytes, u64 offset, const int* slots, const i8* packed, const float* scales, const int* sums, float* output, int rows, int cols, int type, GridTables t) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp, j = blockIdx.y;
    if(row >= rows) return;
    const u8* line = base + slotBytes * (u64)slots[j] + offset + rowBytes * (u64)row;
    int groups = cols / 32, perBlock = groupsPerBlock(type), bytes = blockBytes(type);
    u64 first = (u64)j * groups;
    float acc = 0.0f;
    for(int u = lane; u < groups; u += 32){
        int w[8];
        float low, high, shift;
        unpackGroup(type, line + (u / perBlock) * bytes, u % perBlock, w, low, high, shift, t);
        u64 at = first + u;
        const int4* q = (const int4*)(packed + at * 32);
        int4 a = __ldg(q);
        int4 b = __ldg(q + 1);
        int lowSum = __dp4a(w[3], a.w, __dp4a(w[2], a.z, __dp4a(w[1], a.y, __dp4a(w[0], a.x, 0))));
        int highSum = __dp4a(w[7], b.w, __dp4a(w[6], b.z, __dp4a(w[5], b.y, __dp4a(w[4], b.x, 0))));
        acc += __ldg(scales + at) * (low * (float)lowSum + high * (float)highSum - shift * (float)__ldg(sums + at));
    }
    for(int step = 16; step > 0; step >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, step);
    if(lane == 0) output[(u64)j * rows + row] = acc;
}

extern "C" __global__ void glmIdxKey(const float* raw, const float* gate, const float* weight, const float* bias, float* keys, float* gates, const int* positions, int ring, float eps) {
    __shared__ float part[32];
    int t = blockIdx.x, c = threadIdx.x;
    float v = raw[(u64)t * 128 + c];
    float mean = hyBlockSum(v, part) / 128.0f;
    __syncthreads();
    float centered = v - mean;
    float variance = hyBlockSum(centered * centered, part) / 128.0f;
    u64 slot = (u64)(positions[t] % ring) * 128 + c;
    keys[slot] = centered * rsqrtf(variance + eps) * weight[c] + bias[c];
    gates[slot] = gate[(u64)t * 128 + c];
}

extern "C" __global__ void glmIdxPool(const float* keys, const float* gates, const float* ape, const int* positions, int ring, float* pools) {
    int t = blockIdx.x, c = threadIdx.x;
    int pos = positions[t];
    if((pos & 3) != 3) return;
    int g = pos >> 2;
    float z[4], k[4];
    for(int s = 0; s < 4; s++){
        u64 slot = (u64)((4 * g + s) % ring) * 128 + c;
        z[s] = gates[slot] + ape[s * 128 + c];
        k[s] = keys[slot];
    }
    float top = fmaxf(fmaxf(z[0], z[1]), fmaxf(z[2], z[3]));
    float sum = 0.0f, pooled = 0.0f;
    for(int s = 0; s < 4; s++){
        float e = __expf(z[s] - top);
        sum += e;
        pooled += e * k[s];
    }
    pools[(u64)g * 128 + c] = pooled / sum;
}

extern "C" __global__ void glmIdxScore(const float* q, const float* w, const float* pools, const int* positions, float* scores, int stride, int keep, float scale) {
    __shared__ float4 qs[32][32];
    __shared__ float ws[32];
    int t = blockIdx.y;
    int pos = positions[t];
    int candidates = (pos >> 2) + ((pos & 3) == 3 ? 1 : 0);
    if(candidates <= keep || (int)(blockIdx.x * blockDim.x) >= candidates) return;
    const float4* src = (const float4*)(q + (u64)t * 4096);
    for(int i = threadIdx.x; i < 1024; i += blockDim.x) qs[i >> 5][i & 31] = src[i];
    if(threadIdx.x < 32) ws[threadIdx.x] = w[(u64)t * 32 + threadIdx.x];
    __syncthreads();
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if(g >= candidates) return;
    const float4* key = (const float4*)(pools + (u64)g * 128);
    float acc[32];
    #pragma unroll
    for(int h = 0; h < 32; h++) acc[h] = 0.0f;
    for(int d = 0; d < 32; d++){
        float4 kv = key[d];
        #pragma unroll
        for(int h = 0; h < 32; h++){
            float4 qv = qs[h][d];
            acc[h] += qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
        }
    }
    float score = 0.0f;
    #pragma unroll
    for(int h = 0; h < 32; h++) score += ws[h] * fmaxf(acc[h], 0.0f);
    scores[(u64)t * stride + g] = score * scale;
}

extern "C" __global__ void glmSelect(const float* scores, int stride, const int* positions, int* list, int* counts, int width, int keep) {
    __shared__ unsigned int hist[256];
    __shared__ unsigned int chosen[2];
    __shared__ int warpCount[32];
    __shared__ int warpTie[32];
    __shared__ int baseCount, baseTie;
    int t = blockIdx.x;
    int pos = positions[t];
    int candidates = (pos >> 2) + ((pos & 3) == 3 ? 1 : 0);
    int* out = list + (u64)t * width;
    if(candidates <= keep){
        for(int j = threadIdx.x; j <= pos; j += blockDim.x) out[j] = j;
        if(threadIdx.x == 0) counts[t] = pos + 1;
        return;
    }
    const float* row = scores + (u64)t * stride;
    unsigned int prefix = 0, mask = 0;
    int remaining = keep;
    for(int shift = 24; shift >= 0; shift -= 8){
        for(int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0;
        __syncthreads();
        for(int g = threadIdx.x; g < candidates; g += blockDim.x){
            unsigned int key = hyOrderKey(row[g]);
            if((key & mask) == prefix) atomicAdd(&hist[(key >> shift) & 255], 1u);
        }
        __syncthreads();
        if(threadIdx.x == 0){
            int above = 0, digit = 0;
            for(int d = 255; d >= 0; d--){
                if(above + (int)hist[d] >= remaining){
                    digit = d;
                    break;
                }
                above += hist[d];
            }
            chosen[0] = digit;
            chosen[1] = remaining - above;
        }
        __syncthreads();
        prefix |= chosen[0] << shift;
        mask |= 255u << shift;
        remaining = chosen[1];
        __syncthreads();
    }
    if(threadIdx.x == 0){
        baseCount = 0;
        baseTie = 0;
    }
    __syncthreads();
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    for(int start = 0; start < candidates; start += blockDim.x){
        int g = start + threadIdx.x;
        unsigned int key = g < candidates ? hyOrderKey(row[g]) : 0;
        bool over = g < candidates && key > prefix;
        bool tie = g < candidates && key == prefix;
        unsigned int overBits = __ballot_sync(0xffffffff, over), tieBits = __ballot_sync(0xffffffff, tie);
        if(lane == 0){
            warpCount[warp] = __popc(overBits);
            warpTie[warp] = __popc(tieBits);
        }
        __syncthreads();
        int overBefore = baseCount, tieBefore = baseTie;
        for(int w = 0; w < warp; w++){
            overBefore += warpCount[w];
            tieBefore += warpTie[w];
        }
        unsigned int below = (1u << lane) - 1u;
        overBefore += __popc(overBits & below);
        tieBefore += __popc(tieBits & below);
        int tiesTaken = min(tieBefore, remaining);
        bool take = over || (tie && tieBefore < remaining);
        if(take){
            int at = overBefore + tiesTaken;
            for(int s = 0; s < 4; s++) out[4 * at + s] = 4 * g + s;
        }
        __syncthreads();
        if(threadIdx.x == 0){
            for(int w = 0; w < warps; w++){
                baseCount += warpCount[w];
                baseTie += warpTie[w];
            }
        }
        __syncthreads();
    }
    int tail = (pos & 3) == 3 ? 0 : (pos & 3) + 1;
    for(int j = threadIdx.x; j < tail; j += blockDim.x) out[4 * keep + j] = 4 * (pos >> 2) + j;
    if(threadIdx.x == 0) counts[t] = 4 * keep + tail;
}

extern "C" __global__ void glmGatedNorm(const float* o, const float* gate, const float* weight, float* out, int heads, float eps, i8* packed, float* scales, int* sums) {
    __shared__ float red[4];
    int t = blockIdx.x / heads, h = blockIdx.x % heads, j = threadIdx.x;
    int lane = j & 31, warp = j >> 5;
    u64 at = (u64)t * heads * 128 + h * 128 + j;
    float v = o[at];
    float sq = v * v;
    for(int s = 16; s > 0; s >>= 1) sq += __shfl_xor_sync(0xffffffff, sq, s);
    if(lane == 0) red[warp] = sq;
    __syncthreads();
    float inv = rsqrtf((red[0] + red[1] + red[2] + red[3]) / 128.0f + eps);
    float result = v * inv * weight[j] * hySigmoid(gate[at]);
    out[at] = result;
    if(packed) hyQuant(result, at, packed, scales, sums);
}
)";
