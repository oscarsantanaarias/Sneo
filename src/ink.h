#pragma once
#include "glm5.h"

struct InkQ8 {
    uint64_t values = 0, scales = 0;
    int rows = 0, cols = 0;
};

struct InkLayer {
    bool full = false;
    bool loop = false;
    GpuMatrix q, k, v, gate, out, rel, sink, kProj, vProj, gateUp, down;
    InkQ8 qA, kA, vA, gateA, relA, outA, sinkA, kProjA, vProjA;
    uint64_t norm = 0, qNorm = 0, kNorm = 0, vNorm = 0;
    uint64_t qClip = 0, relClip = 0, relBank = 0, kConv = 0, vConv = 0, postConv = 0, ffnConv = 0;
    uint64_t resQuery = 0, ffnQuery = 0, ffnNorm = 0, router = 0, routerBias = 0, guScale = 0, dScale = 0;
    int queries = 0;
    uint64_t gateUpBytes = 0, downBytes = 0;
    uint64_t ctrlDown[4] = {}, ctrlForget[4] = {}, ctrlGain[4] = {};
    uint64_t msgQuery = 0;
};

struct InkRun {
    uint64_t kCache = 0, vCache = 0, kHist = 0, vHist = 0, postHist = 0, ffnHist = 0;
    int ring = 0;
};

struct InkShape {
    int layers = 24, hidden = 2304, heads = 24, kvHeads = 8, headDim = 128, relDim = 16;
    int experts = 128, used = 8, expertWidth = 1152, vocab = 0, unpadded = 0;
    int window = 1024, extent = 1024, loopStart = 6, loopEnd = 18, passes = 3, block = 6, rank = 64;
    float eps = 1e-6f, qkClip = 22.627417f, relClip = 8.0f, logFloor = 8192.0f, logAlpha = 0.1f;
    float softcap = 30.0f, gateCap = 4.0f, upCap = 25.0f, routeScale = 8.0f;
    int runs = 0;
    int fewRows = 16;
    int attendRows = 64;
};

struct InkBuffers {
    uint64_t h[2] = {}, blocks[4] = {}, prevBlock = 0, prevH = 0, msgQueries = 0, msgWeights = 0;
    uint64_t xin = 0, hq = 0, hk = 0, hv = 0, q = 0, k = 0, v = 0, rel = 0, sink = 0, gateRaw = 0, attn = 0;
    uint64_t dynK = 0, dynV = 0, proj = 0, resid = 0, mlpIn = 0, xn = 0, logits = 0, moe = 0;
    uint64_t ids = 0, weights = 0, pairs = 0, rowScale = 0, rowWeight = 0, guOut = 0, act = 0, downOut = 0;
    uint64_t scratch = 0, positions = 0, last = 0, vocab = 0, outNorm = 0, outQuery = 0, embNorm = 0;
    uint64_t scratchFloats = 0, gates = 0;
    float* hostEmbed = nullptr;
    float* hostLogits = nullptr;
    int* hostPositions = nullptr;
    int* hostIds = nullptr;
    float* hostWeights = nullptr;
    int* hostGroups = nullptr;
    uint64_t mappedEmbed = 0, mappedLogits = 0, mappedIds = 0, mappedWeights = 0, groupsOn = 0;
    uint64_t inValues = 0, inScales = 0, inSums = 0, hidValues = 0, hidScales = 0, hidSums = 0;
    InkQ8 headA;
    int pairsMost = 0;
    void* routed = nullptr;
    std::vector<std::vector<uint8_t>> saved;
    std::vector<size_t> savedTokens;
};

const char* inkKernelSource = R"(
extern "C" __global__ void inkBlockRes(const float* s0, const float* s1, const float* s2, const float* s3, const float* partial, int blocks, const float* query, int queries, float* o0, float* o1, float* o2, float* o3, int hidden, float eps) {
    __shared__ float red[25][32];
    __shared__ float weight[4][5];
    __shared__ float inv[5];
    int t = blockIdx.x;
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    int count = blocks + 1;
    const float* state[5] = {s0, s1, s2, s3, s0};
    state[blocks] = partial;
    float acc[25];
    #pragma unroll
    for(int i = 0; i < 25; i++) acc[i] = 0.0f;
    for(int c = threadIdx.x; c < hidden; c += blockDim.x){
        #pragma unroll
        for(int k = 0; k < 5; k++){
            if(k >= count) break;
            float x = state[k][(u64)t * hidden + c];
            acc[k] += x * x;
            #pragma unroll
            for(int j = 0; j < 4; j++) if(j < queries) acc[5 + j * 5 + k] += x * query[(u64)j * hidden + c];
        }
    }
    #pragma unroll
    for(int i = 0; i < 25; i++){
        float v = acc[i];
        for(int s = 16; s > 0; s >>= 1) v += __shfl_xor_sync(0xffffffff, v, s);
        if(lane == 0) red[i][warp] = v;
    }
    __syncthreads();
    if(threadIdx.x < 25){
        float v = 0.0f;
        for(int w = 0; w < warps; w++) v += red[threadIdx.x][w];
        red[threadIdx.x][0] = v;
    }
    __syncthreads();
    if(threadIdx.x == 0){
        for(int k = 0; k < count; k++) inv[k] = rsqrtf(red[k][0] / hidden + eps);
        float scale = rsqrtf((float)hidden);
        for(int j = 0; j < queries; j++){
            float top = -1e30f;
            float logit[5];
            for(int k = 0; k < count; k++){
                logit[k] = red[5 + j * 5 + k][0] * inv[k] * scale;
                top = fmaxf(top, logit[k]);
            }
            float total = 0.0f;
            for(int k = 0; k < count; k++){
                logit[k] = __expf(logit[k] - top);
                total += logit[k];
            }
            for(int k = 0; k < count; k++) weight[j][k] = logit[k] / total * inv[k];
        }
    }
    __syncthreads();
    float* out[4] = {o0, o1, o2, o3};
    for(int c = threadIdx.x; c < hidden; c += blockDim.x){
        float x[5];
        #pragma unroll
        for(int k = 0; k < 5; k++) x[k] = k < count ? state[k][(u64)t * hidden + c] : 0.0f;
        #pragma unroll
        for(int j = 0; j < 4; j++){
            if(j >= queries) break;
            float v = 0.0f;
            #pragma unroll
            for(int k = 0; k < 5; k++) if(k < count) v += weight[j][k] * x[k];
            out[j][(u64)t * hidden + c] = v;
        }
    }
}

extern "C" __global__ void inkBlockDots(const float* s0, const float* s1, const float* s2, const float* s3, const float* partial, int blocks, const float* query, int queries, float* partials, int hidden) {
    __shared__ float red[25][8];
    int t = blockIdx.x, c = blockIdx.y * 256 + threadIdx.x;
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int count = blocks + 1;
    const float* state[5] = {s0, s1, s2, s3, s0};
    state[blocks] = partial;
    float acc[25];
    #pragma unroll
    for(int i = 0; i < 25; i++) acc[i] = 0.0f;
    if(c < hidden){
        #pragma unroll
        for(int k = 0; k < 5; k++){
            if(k >= count) break;
            float x = state[k][(u64)t * hidden + c];
            acc[k] = x * x;
            #pragma unroll
            for(int j = 0; j < 4; j++) if(j < queries) acc[5 + j * 5 + k] = x * query[(u64)j * hidden + c];
        }
    }
    int need = 5 + queries * 5;
    #pragma unroll
    for(int i = 0; i < 25; i++){
        if(i >= need) break;
        float v = acc[i];
        for(int s = 16; s > 0; s >>= 1) v += __shfl_xor_sync(0xffffffff, v, s);
        if(lane == 0) red[i][warp] = v;
    }
    __syncthreads();
    if(threadIdx.x < need){
        float v = 0.0f;
        for(int w = 0; w < 8; w++) v += red[threadIdx.x][w];
        partials[((u64)t * 9 + blockIdx.y) * 25 + threadIdx.x] = v;
    }
}

extern "C" __global__ void inkBlockMix(const float* s0, const float* s1, const float* s2, const float* s3, const float* partial, int blocks, const float* partials, int queries, float* o0, float* o1, float* o2, float* o3, int hidden, float eps) {
    __shared__ float red[25];
    __shared__ float weight[4][5];
    int t = blockIdx.x, c = blockIdx.y * 256 + threadIdx.x;
    int count = blocks + 1;
    if(threadIdx.x < 5 + queries * 5){
        float v = 0.0f;
        for(int p = 0; p < 9; p++) v += partials[((u64)t * 9 + p) * 25 + threadIdx.x];
        red[threadIdx.x] = v;
    }
    __syncthreads();
    if(threadIdx.x == 0){
        float inv[5];
        for(int k = 0; k < count; k++) inv[k] = rsqrtf(red[k] / hidden + eps);
        float scale = rsqrtf((float)hidden);
        for(int j = 0; j < queries; j++){
            float top = -1e30f;
            float logit[5];
            for(int k = 0; k < count; k++){
                logit[k] = red[5 + j * 5 + k] * inv[k] * scale;
                top = fmaxf(top, logit[k]);
            }
            float total = 0.0f;
            for(int k = 0; k < count; k++){
                logit[k] = __expf(logit[k] - top);
                total += logit[k];
            }
            for(int k = 0; k < count; k++) weight[j][k] = logit[k] / total * inv[k];
        }
    }
    __syncthreads();
    if(c >= hidden) return;
    const float* state[5] = {s0, s1, s2, s3, s0};
    state[blocks] = partial;
    float x[5];
    #pragma unroll
    for(int k = 0; k < 5; k++) x[k] = k < count ? state[k][(u64)t * hidden + c] : 0.0f;
    float* out[4] = {o0, o1, o2, o3};
    #pragma unroll
    for(int j = 0; j < 4; j++){
        if(j >= queries) break;
        float v = 0.0f;
        #pragma unroll
        for(int k = 0; k < 5; k++) if(k < count) v += weight[j][k] * x[k];
        out[j][(u64)t * hidden + c] = v;
    }
}

extern "C" __global__ void inkBlockResNorm(const float* s0, const float* s1, const float* s2, const float* s3, const float* partial, int blocks, const float* query, const float* normWeight, float* out, i8* pkValues, float* pkScales, int* pkSums, int hidden, float eps) {
    __shared__ float red[10][32];
    __shared__ float weight[5];
    __shared__ float inv[5];
    __shared__ float buf[2304];
    int t = blockIdx.x;
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    int count = blocks + 1;
    const float* state[5] = {s0, s1, s2, s3, s0};
    state[blocks] = partial;
    float acc[10];
    #pragma unroll
    for(int i = 0; i < 10; i++) acc[i] = 0.0f;
    for(int c = threadIdx.x; c < hidden; c += blockDim.x){
        #pragma unroll
        for(int k = 0; k < 5; k++){
            if(k >= count) break;
            float x = state[k][(u64)t * hidden + c];
            acc[k] += x * x;
            acc[5 + k] += x * query[c];
        }
    }
    #pragma unroll
    for(int i = 0; i < 10; i++){
        float v = acc[i];
        for(int s = 16; s > 0; s >>= 1) v += __shfl_xor_sync(0xffffffff, v, s);
        if(lane == 0) red[i][warp] = v;
    }
    __syncthreads();
    if(threadIdx.x < 10){
        float v = 0.0f;
        for(int w = 0; w < warps; w++) v += red[threadIdx.x][w];
        red[threadIdx.x][0] = v;
    }
    __syncthreads();
    if(threadIdx.x == 0){
        for(int k = 0; k < count; k++) inv[k] = rsqrtf(red[k][0] / hidden + eps);
        float scale = rsqrtf((float)hidden);
        float top = -1e30f;
        float logit[5];
        for(int k = 0; k < count; k++){
            logit[k] = red[5 + k][0] * inv[k] * scale;
            top = fmaxf(top, logit[k]);
        }
        float total = 0.0f;
        for(int k = 0; k < count; k++){
            logit[k] = __expf(logit[k] - top);
            total += logit[k];
        }
        for(int k = 0; k < count; k++) weight[k] = logit[k] / total * inv[k];
    }
    __syncthreads();
    float ss = 0.0f;
    for(int c = threadIdx.x; c < hidden; c += blockDim.x){
        float v = 0.0f;
        #pragma unroll
        for(int k = 0; k < 5; k++) if(k < count) v += weight[k] * state[k][(u64)t * hidden + c];
        buf[c] = v;
        ss += v * v;
    }
    float normInv = rsqrtf(hyBlockSum(ss, red[0]) / hidden + eps);
    __syncthreads();
    for(int c = threadIdx.x; c < hidden; c += blockDim.x){
        float v = buf[c] * normInv * normWeight[c];
        u64 idx = (u64)t * hidden + c;
        out[idx] = v;
        if(pkValues) hyQuant(v, idx, pkValues, pkScales, pkSums);
    }
}

extern "C" __global__ void inkF32Wide(const float* weights, const float* in, int inStride, float* out, int outStride, int cols) {
    __shared__ float part[32];
    int row = blockIdx.x, t = blockIdx.y;
    const float4* w = (const float4*)(weights + (u64)row * cols);
    const float4* x = (const float4*)(in + (u64)t * inStride);
    int qcols = cols >> 2;
    float sum = 0.0f;
    for(int c = threadIdx.x; c < qcols; c += blockDim.x){
        float4 wv = __ldg(w + c), xv = x[c];
        sum += wv.x * xv.x + wv.y * xv.y + wv.z * xv.z + wv.w * xv.w;
    }
    sum = hyBlockSum(sum, part);
    if(threadIdx.x == 0) out[(u64)t * outStride + row] = sum;
}

extern "C" __global__ void inkWide16(const u16* weights, const float* in, int inStride, float* out, int outStride, int cols) {
    int row = blockIdx.x, t = blockIdx.y;
    const uint4* w = (const uint4*)(weights + (u64)row * cols);
    const float4* x = (const float4*)(in + (u64)t * inStride);
    int ocols = cols >> 3;
    float sum = 0.0f;
    for(int c = threadIdx.x; c < ocols; c += blockDim.x){
        uint4 wv = __ldg(w + c);
        float4 x0 = x[2 * c], x1 = x[2 * c + 1];
        sum += hyHalfAt(wv, 0) * x0.x + hyHalfAt(wv, 1) * x0.y + hyHalfAt(wv, 2) * x0.z + hyHalfAt(wv, 3) * x0.w
             + hyHalfAt(wv, 4) * x1.x + hyHalfAt(wv, 5) * x1.y + hyHalfAt(wv, 6) * x1.z + hyHalfAt(wv, 7) * x1.w;
    }
    for(int s = 16; s > 0; s >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, s);
    if(threadIdx.x == 0) out[(u64)t * outStride + row] = sum;
}

extern "C" __global__ void inkAllMsgDots(const float* s0, const float* s1, const float* const* queries, float* partials, int hidden, int numLayers, int blocksY) {
    __shared__ float sh0[256], sh1[256];
    int t = blockIdx.x, by = blockIdx.y, tid = threadIdx.x;
    int warp = tid >> 5, lane = tid & 31;
    int c = by * 256 + tid;
    float x0 = c < hidden ? s0[(u64)t * hidden + c] : 0.0f;
    float x1 = c < hidden ? s1[(u64)t * hidden + c] : 0.0f;
    sh0[tid] = x0;
    sh1[tid] = x1;
    __syncthreads();
    if(warp == 0){
        float n0 = 0.0f, n1 = 0.0f;
        #pragma unroll
        for(int i = 0; i < 8; i++){
            int idx = i * 32 + lane;
            float a = sh0[idx], b = sh1[idx];
            n0 += a * a;
            n1 += b * b;
        }
        for(int s = 16; s > 0; s >>= 1){
            n0 += __shfl_xor_sync(0xffffffff, n0, s);
            n1 += __shfl_xor_sync(0xffffffff, n1, s);
        }
        if(lane == 0){
            partials[((u64)t * blocksY + by) * 130 + 0] = n0;
            partials[((u64)t * blocksY + by) * 130 + 1] = n1;
        }
    }
    int totalQ = numLayers * 4;
    int qPerWarp = (totalQ + 7) / 8;
    int qStart = warp * qPerWarp;
    for(int qq = 0; qq < qPerWarp; qq++){
        int qIdx = qStart + qq;
        if(qIdx >= totalQ) break;
        const float* qPtr = queries[qIdx >> 2] + (u64)(qIdx & 3) * hidden + by * 256;
        float d0 = 0.0f, d1 = 0.0f;
        #pragma unroll
        for(int i = 0; i < 8; i++){
            int idx = i * 32 + lane;
            if(by * 256 + idx < hidden){
                float qv = qPtr[idx];
                d0 += sh0[idx] * qv;
                d1 += sh1[idx] * qv;
            }
        }
        for(int s = 16; s > 0; s >>= 1){
            d0 += __shfl_xor_sync(0xffffffff, d0, s);
            d1 += __shfl_xor_sync(0xffffffff, d1, s);
        }
        if(lane == 0){
            u64 base = ((u64)t * blocksY + by) * 130 + 2 + (u64)qIdx * 2;
            partials[base + 0] = d0;
            partials[base + 1] = d1;
        }
    }
}

extern "C" __global__ void inkAllMsgWeights(const float* partials, float* msgWeights, int hidden, float eps, int totalQ, int blocksY) {
    __shared__ float inv[2];
    int t = blockIdx.x, tid = threadIdx.x;
    if(tid < 2){
        float s = 0.0f;
        for(int p = 0; p < blocksY; p++) s += partials[((u64)t * blocksY + p) * 130 + tid];
        inv[tid] = rsqrtf(s / hidden + eps);
    }
    __syncthreads();
    if(tid >= totalQ) return;
    float d0 = 0.0f, d1 = 0.0f;
    for(int p = 0; p < blocksY; p++){
        u64 base = ((u64)t * blocksY + p) * 130 + 2 + (u64)tid * 2;
        d0 += partials[base + 0];
        d1 += partials[base + 1];
    }
    float scale = rsqrtf((float)hidden);
    float l0 = d0 * inv[0] * scale;
    float l1 = d1 * inv[1] * scale;
    float top = fmaxf(l0, l1);
    float e0 = __expf(l0 - top);
    float e1 = __expf(l1 - top);
    float total = e0 + e1;
    u64 out = ((u64)t * totalQ + tid) * 2;
    msgWeights[out + 0] = e0 / total * inv[0];
    msgWeights[out + 1] = e1 / total * inv[1];
}

extern "C" __global__ void inkMsgDown(const float* h, const float* prevB, const float* prevH, const float* msgWeights, int totalQ, int qIdx, const u16* down, float* gates, int hidden, int rank, float eps) {
    __shared__ float cat[4608];
    __shared__ float part[32];
    int t = blockIdx.x;
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    const float* hs = h + (u64)t * hidden;
    const float* pb = prevB + (u64)t * hidden;
    const float* ph = prevH + (u64)t * hidden;
    float w0 = msgWeights[((u64)t * totalQ + qIdx) * 2 + 0];
    float w1 = msgWeights[((u64)t * totalQ + qIdx) * 2 + 1];
    float a = 0.0f, b = 0.0f;
    for(int c = threadIdx.x; c < hidden; c += blockDim.x){
        float hv = hs[c], mv = w0 * pb[c] + w1 * ph[c];
        a += hv * hv;
        b += mv * mv;
        cat[c] = hv;
        cat[hidden + c] = mv;
    }
    float ia = rsqrtf(hyBlockSum(a, part) / hidden + eps);
    __syncthreads();
    float ib = rsqrtf(hyBlockSum(b, part) / hidden + eps);
    for(int c = threadIdx.x; c < hidden; c += blockDim.x){
        cat[c] *= ia;
        cat[hidden + c] *= ib;
    }
    __syncthreads();
    int r0 = blockIdx.y * (warps * 2) + warp, r1 = r0 + warps;
    if(r0 >= rank) return;
    const uint4* row0 = (const uint4*)(down + (u64)r0 * 2 * hidden);
    const uint4* row1 = r1 < rank ? (const uint4*)(down + (u64)r1 * 2 * hidden) : nullptr;
    const float4* cat4 = (const float4*)cat;
    int ocols = hidden >> 2;
    float v0 = 0.0f, v1 = 0.0f;
    for(int i = lane; i < ocols; i += 32){
        float4 c0 = cat4[2 * i], c1 = cat4[2 * i + 1];
        uint4 rw0 = __ldg(row0 + i);
        v0 += hyHalfAt(rw0, 0) * c0.x + hyHalfAt(rw0, 1) * c0.y + hyHalfAt(rw0, 2) * c0.z + hyHalfAt(rw0, 3) * c0.w
            + hyHalfAt(rw0, 4) * c1.x + hyHalfAt(rw0, 5) * c1.y + hyHalfAt(rw0, 6) * c1.z + hyHalfAt(rw0, 7) * c1.w;
        if(row1){
            uint4 rw1 = __ldg(row1 + i);
            v1 += hyHalfAt(rw1, 0) * c0.x + hyHalfAt(rw1, 1) * c0.y + hyHalfAt(rw1, 2) * c0.z + hyHalfAt(rw1, 3) * c0.w
                + hyHalfAt(rw1, 4) * c1.x + hyHalfAt(rw1, 5) * c1.y + hyHalfAt(rw1, 6) * c1.z + hyHalfAt(rw1, 7) * c1.w;
        }
    }
    for(int s = 16; s > 0; s >>= 1){
        v0 += __shfl_xor_sync(0xffffffff, v0, s);
        if(row1) v1 += __shfl_xor_sync(0xffffffff, v1, s);
    }
    if(lane == 0){
        gates[(u64)t * rank + r0] = v0;
        if(row1) gates[(u64)t * rank + r1] = v1;
    }
}

extern "C" __global__ void inkMsgMix(const float* h, const float* prevB, const float* prevH, const float* msgWeights, int totalQ, int qIdx, const float* gates, const u16* forget, const u16* gain, float* out, int hidden, int rank) {
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int c = blockIdx.x * (blockDim.x >> 5) + warp, t = blockIdx.y;
    if(c >= hidden) return;
    const float* gt = gates + (u64)t * rank;
    const u16* fr = forget + (u64)c * rank;
    const u16* gr = gain + (u64)c * rank;
    float f = 0.0f, k = 0.0f;
    for(int r = lane * 2; r < rank; r += 64){
        float g0 = gt[r], g1 = r + 1 < rank ? gt[r + 1] : 0.0f;
        u32 fw = *(const u32*)(fr + r);
        u32 gw = *(const u32*)(gr + r);
        f += hyHalf((u16)(fw & 0xffffu)) * g0 + hyHalf((u16)(fw >> 16)) * g1;
        k += hyHalf((u16)(gw & 0xffffu)) * g0 + hyHalf((u16)(gw >> 16)) * g1;
    }
    for(int s = 16; s > 0; s >>= 1){
        f += __shfl_xor_sync(0xffffffff, f, s);
        k += __shfl_xor_sync(0xffffffff, k, s);
    }
    u64 at = (u64)t * hidden + c;
    if(lane == 0){
        float w0 = msgWeights[((u64)t * totalQ + qIdx) * 2 + 0];
        float w1 = msgWeights[((u64)t * totalQ + qIdx) * 2 + 1];
        float mv = w0 * prevB[at] + w1 * prevH[at];
        out[at] = (1.0f - tanhf(f)) * h[at] + tanhf(k) * mv;
    }
}

extern "C" __global__ void inkMsgDown3(const float* h0, const u16* d0, const float* h1, const u16* d1, const float* h2, const u16* d2, const float* prevB, const float* prevH, const float* msgWeights, int totalQ, int layerQ, float* gates, int hidden, int rank, float eps) {
    __shared__ float cat[4608];
    __shared__ float part[32];
    int t = blockIdx.x;
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    int perGate = (rank + warps * 2 - 1) / (warps * 2);
    int k = blockIdx.y / perGate;
    int r0 = (blockIdx.y % perGate) * (warps * 2) + warp, r1 = r0 + warps;
    const float* hs = (k == 0 ? h0 : k == 1 ? h1 : h2) + (u64)t * hidden;
    const float* pb = prevB + (u64)t * hidden;
    const float* ph = prevH + (u64)t * hidden;
    const u16* down = k == 0 ? d0 : k == 1 ? d1 : d2;
    float w0 = msgWeights[((u64)t * totalQ + layerQ + k) * 2 + 0];
    float w1 = msgWeights[((u64)t * totalQ + layerQ + k) * 2 + 1];
    float a = 0.0f, b = 0.0f;
    for(int c = threadIdx.x; c < hidden; c += blockDim.x){
        float hv = hs[c], mv = w0 * pb[c] + w1 * ph[c];
        a += hv * hv;
        b += mv * mv;
        cat[c] = hv;
        cat[hidden + c] = mv;
    }
    float ia = rsqrtf(hyBlockSum(a, part) / hidden + eps);
    __syncthreads();
    float ib = rsqrtf(hyBlockSum(b, part) / hidden + eps);
    for(int c = threadIdx.x; c < hidden; c += blockDim.x){
        cat[c] *= ia;
        cat[hidden + c] *= ib;
    }
    __syncthreads();
    if(r0 >= rank) return;
    const uint4* row0 = (const uint4*)(down + (u64)r0 * 2 * hidden);
    const uint4* row1 = r1 < rank ? (const uint4*)(down + (u64)r1 * 2 * hidden) : nullptr;
    const float4* cat4 = (const float4*)cat;
    int ocols = hidden >> 2;
    float v0 = 0.0f, v1 = 0.0f;
    for(int i = lane; i < ocols; i += 32){
        float4 c0 = cat4[2 * i], c1 = cat4[2 * i + 1];
        uint4 rw0 = __ldg(row0 + i);
        v0 += hyHalfAt(rw0, 0) * c0.x + hyHalfAt(rw0, 1) * c0.y + hyHalfAt(rw0, 2) * c0.z + hyHalfAt(rw0, 3) * c0.w
            + hyHalfAt(rw0, 4) * c1.x + hyHalfAt(rw0, 5) * c1.y + hyHalfAt(rw0, 6) * c1.z + hyHalfAt(rw0, 7) * c1.w;
        if(row1){
            uint4 rw1 = __ldg(row1 + i);
            v1 += hyHalfAt(rw1, 0) * c0.x + hyHalfAt(rw1, 1) * c0.y + hyHalfAt(rw1, 2) * c0.z + hyHalfAt(rw1, 3) * c0.w
                + hyHalfAt(rw1, 4) * c1.x + hyHalfAt(rw1, 5) * c1.y + hyHalfAt(rw1, 6) * c1.z + hyHalfAt(rw1, 7) * c1.w;
        }
    }
    for(int s = 16; s > 0; s >>= 1){
        v0 += __shfl_xor_sync(0xffffffff, v0, s);
        if(row1) v1 += __shfl_xor_sync(0xffffffff, v1, s);
    }
    if(lane == 0){
        gates[((u64)t * 3 + k) * rank + r0] = v0;
        if(row1) gates[((u64)t * 3 + k) * rank + r1] = v1;
    }
}

extern "C" __global__ void inkMsgMix3(const float* h0, const u16* f0, const u16* g0, float* o0, const float* h1, const u16* f1, const u16* g1, float* o1, const float* h2, const u16* f2, const u16* g2, float* o2, const float* prevB, const float* prevH, const float* msgWeights, int totalQ, int layerQ, const float* gates, int hidden, int rank) {
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int c = blockIdx.x * (blockDim.x >> 5) + warp;
    int k = blockIdx.y % 3, t = blockIdx.y / 3;
    if(c >= hidden) return;
    const float* h = k == 0 ? h0 : k == 1 ? h1 : h2;
    const u16* forget = k == 0 ? f0 : k == 1 ? f1 : f2;
    const u16* gain = k == 0 ? g0 : k == 1 ? g1 : g2;
    float* out = k == 0 ? o0 : k == 1 ? o1 : o2;
    const float* gt = gates + ((u64)t * 3 + k) * rank;
    const u16* fr = forget + (u64)c * rank;
    const u16* gr = gain + (u64)c * rank;
    float f = 0.0f, kg = 0.0f;
    for(int r = lane * 2; r < rank; r += 64){
        float g0v = gt[r], g1v = r + 1 < rank ? gt[r + 1] : 0.0f;
        u32 fw = *(const u32*)(fr + r);
        u32 gw = *(const u32*)(gr + r);
        f += hyHalf((u16)(fw & 0xffffu)) * g0v + hyHalf((u16)(fw >> 16)) * g1v;
        kg += hyHalf((u16)(gw & 0xffffu)) * g0v + hyHalf((u16)(gw >> 16)) * g1v;
    }
    for(int s = 16; s > 0; s >>= 1){
        f += __shfl_xor_sync(0xffffffff, f, s);
        kg += __shfl_xor_sync(0xffffffff, kg, s);
    }
    u64 at = (u64)t * hidden + c;
    if(lane == 0){
        float w0 = msgWeights[((u64)t * totalQ + layerQ + k) * 2 + 0];
        float w1 = msgWeights[((u64)t * totalQ + layerQ + k) * 2 + 1];
        float mv = w0 * prevB[at] + w1 * prevH[at];
        out[at] = (1.0f - tanhf(f)) * h[at] + tanhf(kg) * mv;
    }
}

__device__ __forceinline__ float inkTau(int pos, float alpha, float floorN) {
    return 1.0f + alpha * logf(fmaxf((pos + 1.0f) / floorN, 1.0f));
}

extern "C" __global__ void inkClip(float* x, const float* scale, const int* positions, int heads, int dim, float threshold, int logScale, float alpha, float floorN) {
    __shared__ float part[32];
    int t = blockIdx.x / heads, h = blockIdx.x % heads;
    float v = 0.0f;
    float* row = x + ((u64)t * heads + h) * dim;
    int d = threadIdx.x;
    if(d < dim) v = row[d] * (scale ? scale[h * dim + d] : 1.0f);
    float total = hyBlockSum(v * v, part);
    float factor = fminf(threshold / (sqrtf(total) + 1e-6f), 1.0f);
    if(logScale) factor *= inkTau(positions[t], alpha, floorN);
    if(d < dim) row[d] = v * factor;
}

extern "C" __global__ void inkClip2(float* x0, const float* s0, int dim0, float th0, float* x1, const float* s1, int dim1, float th1, const int* positions, int heads, int logScale, float alpha, float floorN) {
    __shared__ float part[32];
    int half = gridDim.x >> 1;
    int which = blockIdx.x >= half;
    int bIdx = which ? blockIdx.x - half : blockIdx.x;
    int t = bIdx / heads, h = bIdx % heads;
    float* x = which ? x1 : x0;
    const float* scale = which ? s1 : s0;
    int dim = which ? dim1 : dim0;
    float threshold = which ? th1 : th0;
    float v = 0.0f;
    float* row = x + ((u64)t * heads + h) * dim;
    int d = threadIdx.x;
    if(d < dim) v = row[d] * (scale ? scale[h * dim + d] : 1.0f);
    float total = hyBlockSum(v * v, part);
    float factor = fminf(threshold / (sqrtf(total) + 1e-6f), 1.0f);
    if(logScale) factor *= inkTau(positions[t], alpha, floorN);
    if(d < dim) row[d] = v * factor;
}

extern "C" __global__ void inkNorm3(const float* in0, const float* w0, float* out0, const float* in1, const float* w1, float* out1, const float* in2, const float* w2, float* out2, i8* pkV0, float* pkS0, int* pkSum0, i8* pkV1, float* pkS1, int* pkSum1, i8* pkV2, float* pkS2, int* pkSum2, int cols, float eps) {
    __shared__ float part[32];
    int t = blockIdx.x / 3, k = blockIdx.x % 3;
    const float* src = (k == 0 ? in0 : k == 1 ? in1 : in2) + (u64)t * cols;
    const float* w = k == 0 ? w0 : k == 1 ? w1 : w2;
    float* dst = (k == 0 ? out0 : k == 1 ? out1 : out2) + (u64)t * cols;
    i8* pkV = k == 0 ? pkV0 : k == 1 ? pkV1 : pkV2;
    float* pkS = k == 0 ? pkS0 : k == 1 ? pkS1 : pkS2;
    int* pkSum = k == 0 ? pkSum0 : k == 1 ? pkSum1 : pkSum2;
    float ss = 0.0f;
    for(int c = threadIdx.x; c < cols; c += blockDim.x){
        float v = src[c];
        ss += v * v;
    }
    float inv = rsqrtf(hyBlockSum(ss, part) / cols + eps);
    for(int c = threadIdx.x; c < cols; c += blockDim.x){
        float v = src[c] * inv * w[c];
        dst[c] = v;
        if(pkV) hyQuant(v, (u64)t * cols + c, pkV, pkS, pkSum);
    }
}

extern "C" __global__ void inkNormPack(const float* src, const float* w, float* dst, i8* pkValues, float* pkScales, int* pkSums, int cols, float eps) {
    __shared__ float part[32];
    int t = blockIdx.x;
    const float* s = src + (u64)t * cols;
    float* d = dst + (u64)t * cols;
    float ss = 0.0f;
    for(int c = threadIdx.x; c < cols; c += blockDim.x){
        float v = s[c];
        ss += v * v;
    }
    float inv = rsqrtf(hyBlockSum(ss, part) / cols + eps);
    for(int c = threadIdx.x; c < cols; c += blockDim.x){
        float v = s[c] * inv * w[c];
        u64 idx = (u64)t * cols + c;
        d[c] = v;
        hyQuant(v, idx, pkValues, pkScales, pkSums);
    }
}

extern "C" __global__ void inkConv(float* x, const float* w, const float* dyn, float* history, int n, int width, int perHead) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if(c >= width) return;
    float* past = history + c;
    float h0 = past[0], h1 = past[width], h2 = past[2 * (u64)width];
    float w0 = w[(u64)c * 4], w1 = w[(u64)c * 4 + 1], w2 = w[(u64)c * 4 + 2], w3 = w[(u64)c * 4 + 3];
    float norm = dyn ? 0.3333333333f : 0.4472135955f;
    int head = c / perHead;
    for(int t = 0; t < n; t++){
        float now = x[(u64)t * width + c];
        float a0 = w0, a1 = w1, a2 = w2, a3 = w3;
        if(dyn){
            const float* d = dyn + (u64)t * (width / perHead) * 4 + head * 4;
            a0 += d[3];
            a1 += d[2];
            a2 += d[1];
            a3 += d[0];
        }
        x[(u64)t * width + c] = (a0 * h0 + a1 * h1 + a2 * h2 + a3 * now + now) * norm;
        h0 = h1;
        h1 = h2;
        h2 = now;
    }
    past[0] = h0;
    past[width] = h1;
    past[2 * (u64)width] = h2;
}

extern "C" __global__ void inkConv2(float* x0, const float* w0, const float* d0, float* h0p, float* x1, const float* w1, const float* d1, float* h1p, int n, int width, int perHead) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if(c >= 2 * width) return;
    int which = c >= width;
    if(which) c -= width;
    float* x = which ? x1 : x0;
    const float* w = which ? w1 : w0;
    const float* dyn = which ? d1 : d0;
    float* past = (which ? h1p : h0p) + c;
    float h0 = past[0], h1 = past[width], h2 = past[2 * (u64)width];
    float wt0 = w[(u64)c * 4], wt1 = w[(u64)c * 4 + 1], wt2 = w[(u64)c * 4 + 2], wt3 = w[(u64)c * 4 + 3];
    float norm = dyn ? 0.3333333333f : 0.4472135955f;
    int head = c / perHead;
    for(int t = 0; t < n; t++){
        float now = x[(u64)t * width + c];
        float a0 = wt0, a1 = wt1, a2 = wt2, a3 = wt3;
        if(dyn){
            const float* d = dyn + (u64)t * (width / perHead) * 4 + head * 4;
            a0 += d[3];
            a1 += d[2];
            a2 += d[1];
            a3 += d[0];
        }
        x[(u64)t * width + c] = (a0 * h0 + a1 * h1 + a2 * h2 + a3 * now + now) * norm;
        h0 = h1;
        h1 = h2;
        h2 = now;
    }
    past[0] = h0;
    past[width] = h1;
    past[2 * (u64)width] = h2;
}

extern "C" __global__ void inkConvAdd(const float* x, const float* w, float* history, const float* addend, float* out, int n, int width) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if(c >= width) return;
    float* past = history + c;
    float h0 = past[0], h1 = past[width], h2 = past[2 * (u64)width];
    float w0 = w[(u64)c * 4], w1 = w[(u64)c * 4 + 1], w2 = w[(u64)c * 4 + 2], w3 = w[(u64)c * 4 + 3];
    for(int t = 0; t < n; t++){
        u64 at = (u64)t * width + c;
        float now = x[at];
        float y = (w0 * h0 + w1 * h1 + w2 * h2 + w3 * now + now) * 0.4472135955f;
        out[at] = addend ? addend[at] + y : y;
        h0 = h1;
        h1 = h2;
        h2 = now;
    }
    past[0] = h0;
    past[width] = h1;
    past[2 * (u64)width] = h2;
}

extern "C" __global__ void inkStore(const float* k, const float* v, u16* kCache, u16* vCache, const int* positions, int ring, int kvHeads, float threshold, float eps) {
    __shared__ float part[32];
    int t = blockIdx.x / kvHeads, g = blockIdx.x % kvHeads, d = threadIdx.x;
    u64 at = ((u64)t * kvHeads + g) * 128 + d;
    float kv = k[at], vv = v[at];
    float kk = hyBlockSum(kv * kv, part);
    __syncthreads();
    float vs = hyBlockSum(vv * vv, part);
    int pos = positions[t];
    u64 slot = (u64)(ring > 0 ? pos % ring : pos);
    u64 to = (slot * kvHeads + g) * 128 + d;
    kCache[to] = toHalf(kv * fminf(threshold / (sqrtf(kk) + 1e-6f), 1.0f));
    vCache[to] = toHalf(vv * rsqrtf(vs / 128.0f + eps));
}

extern "C" __global__ void inkConvStore(const float* k, const float* kw, const float* dynK, float* kHist, const float* v, const float* vw, const float* dynV, float* vHist, u16* kCache, u16* vCache, const int* positions, int ring, int kvHeads, float threshold, float eps) {
    __shared__ float part[32];
    int g = blockIdx.x, d = threadIdx.x, c = g * 128 + d;
    auto step = [&](const float* x, const float* w, const float* dyn, float* hist){
        float* past = hist + c;
        float h0 = past[0], h1 = past[1024], h2 = past[2048], now = x[c];
        float a0 = w[(u64)c * 4], a1 = w[(u64)c * 4 + 1], a2 = w[(u64)c * 4 + 2], a3 = w[(u64)c * 4 + 3];
        if(dyn){
            const float* dp = dyn + g * 4;
            a0 += dp[3];
            a1 += dp[2];
            a2 += dp[1];
            a3 += dp[0];
        }
        float norm = dyn ? 0.3333333333f : 0.4472135955f;
        past[0] = h1;
        past[1024] = h2;
        past[2048] = now;
        return (a0 * h0 + a1 * h1 + a2 * h2 + a3 * now + now) * norm;
    };
    float kv = step(k, kw, dynK, kHist);
    float vv = step(v, vw, dynV, vHist);
    float kk = hyBlockSum(kv * kv, part);
    __syncthreads();
    float vs = hyBlockSum(vv * vv, part);
    int pos = positions[0];
    u64 slot = (u64)(ring > 0 ? pos % ring : pos);
    u64 to = (slot * kvHeads + g) * 128 + d;
    kCache[to] = toHalf(kv * fminf(threshold / (sqrtf(kk) + 1e-6f), 1.0f));
    vCache[to] = toHalf(vv * rsqrtf(vs / 128.0f + eps));
}

extern "C" __global__ void inkAttendPart(const float* q, const u16* kCache, const u16* vCache, int ring, const int* positions, const float* rel, const float* bank, float* scratch, int heads, int kvHeads, int window, int extent, float scale, int span, int splits) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int h = blockIdx.x * (blockDim.x >> 5) + warp, t = blockIdx.y / splits, part = blockIdx.y % splits;
    if(h >= heads) return;
    int g = h / (heads / kvHeads);
    float* dst = scratch + (((u64)t * heads + h) * splits + part) * 130;
    int pos = positions[t];
    int lo = window > 0 ? max(0, pos - window + 1) : 0;
    int first = lo + part * span;
    int last = min(pos, first + span - 1);
    float4 qv = ((const float4*)q)[((u64)t * heads + h) * 32 + lane];
    float rs = lane < 16 ? rel[((u64)t * heads + h) * 16 + lane] : 0.0f;
    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float best = -1e30f, total = 0.0f;
    for(int j = first; j <= last; j++){
        u64 slot = (u64)(ring > 0 ? j % ring : j);
        uint2 kw = ((const uint2*)kCache)[(slot * kvHeads + g) * 32 + lane];
        float dot = qv.x * hyHalf((u16)(kw.x & 0xffffu)) + qv.y * hyHalf((u16)(kw.x >> 16))
                  + qv.z * hyHalf((u16)(kw.y & 0xffffu)) + qv.w * hyHalf((u16)(kw.y >> 16));
        for(int s = 16; s > 0; s >>= 1) dot += __shfl_xor_sync(0xffffffff, dot, s);
        int dist = pos - j;
        float b = (lane < 16 && dist < extent) ? rs * bank[(u64)dist * 16 + lane] : 0.0f;
        for(int s = 8; s > 0; s >>= 1) b += __shfl_xor_sync(0xffffffff, b, s);
        float bias = __shfl_sync(0xffffffff, b, 0);
        float score = dot * scale + bias;
        float top = fmaxf(best, score);
        float keep = __expf(best - top);
        float p = __expf(score - top);
        total = total * keep + p;
        uint2 vw = ((const uint2*)vCache)[(slot * kvHeads + g) * 32 + lane];
        acc.x = acc.x * keep + p * hyHalf((u16)(vw.x & 0xffffu));
        acc.y = acc.y * keep + p * hyHalf((u16)(vw.x >> 16));
        acc.z = acc.z * keep + p * hyHalf((u16)(vw.y & 0xffffu));
        acc.w = acc.w * keep + p * hyHalf((u16)(vw.y >> 16));
        best = top;
    }
    if(lane == 0){
        dst[0] = best;
        dst[1] = total;
    }
    dst[2 + 4 * lane + 0] = acc.x;
    dst[2 + 4 * lane + 1] = acc.y;
    dst[2 + 4 * lane + 2] = acc.z;
    dst[2 + 4 * lane + 3] = acc.w;
}

extern "C" __global__ void inkAttendMerge(const float* scratch, const float* sinks, float* out, const float* gate, i8* pkValues, float* pkScales, int* pkSums, int heads, int splits) {
    __shared__ float shW[64];
    __shared__ float shRed[32];
    __shared__ float shTop;
    int h = blockIdx.x, t = blockIdx.y;
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const float* src = scratch + ((u64)t * heads + h) * splits * 130;
    float sink = sinks[(u64)t * heads + h];
    float localMax = -1e30f;
    for(int s = tid; s < splits; s += blockDim.x) localMax = fmaxf(localMax, src[(u64)s * 130]);
    for(int step = 16; step > 0; step >>= 1) localMax = fmaxf(localMax, __shfl_xor_sync(0xffffffff, localMax, step));
    if(lane == 0) shRed[warp] = localMax;
    __syncthreads();
    if(tid == 0){
        float best = sink;
        for(int w = 0; w < (blockDim.x >> 5); w++) best = fmaxf(best, shRed[w]);
        shTop = best;
    }
    __syncthreads();
    float top = shTop;
    float part = tid == 0 ? __expf(sink - top) : 0.0f;
    for(int s = tid; s < splits; s += blockDim.x){
        float w = __expf(src[(u64)s * 130] - top);
        if(s < 64) shW[s] = w;
        part += src[(u64)s * 130 + 1] * w;
    }
    float total = hyBlockSum(part, shRed);
    float inv = 1.0f / total;
    float v = 0.0f;
    for(int s = 0; s < splits; s++){
        float w = s < 64 ? shW[s] : __expf(src[(u64)s * 130] - top);
        v += src[(u64)s * 130 + 2 + tid] * w;
    }
    u64 idx = ((u64)t * heads + h) * 128 + tid;
    if(pkValues){
        float gv = v * inv * hySigmoid(gate[idx]);
        hyQuant(gv, idx, pkValues, pkScales, pkSums);
    } else out[idx] = v * inv;
}

extern "C" __global__ void inkPairs(const int* ids, const float* weights, const float* guScale, const float* dScale, float* rowScale, float* rowWeight, int count) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if(j >= count) return;
    int e = ids[j];
    rowScale[j] = guScale ? guScale[e] : 1.0f;
    rowWeight[j] = weights[j] * (dScale ? dScale[e] : 1.0f);
}

extern "C" __global__ void inkAct(const float* gateUp, const float* rowScale, float* out, int rows, int width, float gateCap, float upCap, i8* packed, float* scales, int* sums) {
    int group = blockIdx.x * blockDim.y + threadIdx.y;
    int perRow = width / 32;
    if(group >= rows * perRow) return;
    int r = group / perRow, c = (group % perRow) * 32 + threadIdx.x;
    float s = rowScale[r];
    float g = gateUp[(u64)r * 2 * width + c] * s;
    float u = gateUp[(u64)r * 2 * width + width + c] * s;
    float v = tanhf(g / gateCap) * gateCap * hySigmoid(g) * tanhf(u / upCap) * upCap;
    u64 i = (u64)r * width + c;
    out[i] = v;
    hyQuant(v, i, packed, scales, sums);
}

extern "C" __global__ void inkScatter(const float* rows, const float* rowWeight, float* out, int hidden, int used) {
    int t = blockIdx.x;
    for(int c = threadIdx.x; c < hidden; c += blockDim.x){
        float v = 0.0f;
        for(int k = 0; k < used; k++) v += rows[((u64)t * used + k) * hidden + c] * rowWeight[t * used + k];
        out[(u64)t * hidden + c] = v;
    }
}

template<int TYPE>
__device__ __forceinline__ void inkUnpack(const u8* block, int g, int* w, float& low, float& high, float& shift, GridTables t) {
    unpackGroup(TYPE, block, g, w, low, high, shift, t);
}

template<int TYPE>
__device__ __forceinline__ void inkGateUpActRun(const u8* base, u64 rowBytes, u64 slotBytes, const int* ids, const float* guScale, const i8* packed, const float* scales, const int* sums, i8* outValues, float* outScales, int* outSums, int hidden, int width, int used, float gateCap, float upCap, GridTables t) {
    __shared__ float sh[32];
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int lane8 = threadIdx.x & 7, chInWarp = (threadIdx.x >> 3) & 3;
    int c0 = blockIdx.x * 32, j = blockIdx.y;
    int c = c0 + warp * 4 + chInWarp;
    int e = ids[j], tok = j / used;
    const u8* expert = base + slotBytes * (u64)e;
    int groups = hidden / 32, perBlock = groupsPerBlock(TYPE), bytes = blockBytes(TYPE);
    u64 first = (u64)tok * groups;
    const u8* gRow = expert + rowBytes * (u64)c;
    const u8* uRow = expert + rowBytes * (u64)(width + c);
    float gAcc = 0.0f, uAcc = 0.0f;
    for(int u = lane8; u < groups; u += 8){
        u64 at = first + u;
        const int4* q = (const int4*)(packed + at * 32);
        int4 a = __ldg(q), b = __ldg(q + 1);
        float sc = __ldg(scales + at);
        int sm = __ldg(sums + at);
        u64 blockOff = (u64)(u / perBlock) * bytes;
        int gInBlock = u % perBlock;
        int w[8];
        float low, high, shift;
        inkUnpack<TYPE>(gRow + blockOff, gInBlock, w, low, high, shift, t);
        int lowSum = __dp4a(w[3], a.w, __dp4a(w[2], a.z, __dp4a(w[1], a.y, __dp4a(w[0], a.x, 0))));
        int highSum = __dp4a(w[7], b.w, __dp4a(w[6], b.z, __dp4a(w[5], b.y, __dp4a(w[4], b.x, 0))));
        gAcc += sc * (low * (float)lowSum + high * (float)highSum - shift * (float)sm);
        inkUnpack<TYPE>(uRow + blockOff, gInBlock, w, low, high, shift, t);
        lowSum = __dp4a(w[3], a.w, __dp4a(w[2], a.z, __dp4a(w[1], a.y, __dp4a(w[0], a.x, 0))));
        highSum = __dp4a(w[7], b.w, __dp4a(w[6], b.z, __dp4a(w[5], b.y, __dp4a(w[4], b.x, 0))));
        uAcc += sc * (low * (float)lowSum + high * (float)highSum - shift * (float)sm);
    }
    for(int s = 4; s > 0; s >>= 1){
        gAcc += __shfl_xor_sync(0xffffffff, gAcc, s);
        uAcc += __shfl_xor_sync(0xffffffff, uAcc, s);
    }
    if(lane8 == 0){
        float sc = guScale ? guScale[e] : 1.0f;
        float g = gAcc * sc;
        float u = uAcc * sc;
        sh[warp * 4 + chInWarp] = tanhf(g / gateCap) * gateCap * hySigmoid(g) * tanhf(u / upCap) * upCap;
    }
    __syncthreads();
    if(warp == 0) hyQuant(sh[lane], (u64)j * width + c0 + lane, outValues, outScales, outSums);
}

extern "C" __global__ void inkGateUpAct40(const u8* base, u64 rowBytes, u64 slotBytes, const int* ids, const float* guScale, const i8* packed, const float* scales, const int* sums, i8* outValues, float* outScales, int* outSums, int hidden, int width, int used, float gateCap, float upCap, GridTables t) {
    inkGateUpActRun<40>(base, rowBytes, slotBytes, ids, guScale, packed, scales, sums, outValues, outScales, outSums, hidden, width, used, gateCap, upCap, t);
}

extern "C" __global__ void inkGateUpAct8(const u8* base, u64 rowBytes, u64 slotBytes, const int* ids, const float* guScale, const i8* packed, const float* scales, const int* sums, i8* outValues, float* outScales, int* outSums, int hidden, int width, int used, float gateCap, float upCap, GridTables t) {
    inkGateUpActRun<8>(base, rowBytes, slotBytes, ids, guScale, packed, scales, sums, outValues, outScales, outSums, hidden, width, used, gateCap, upCap, t);
}

template<int TYPE>
__device__ __forceinline__ void inkDownSumRun(const u8* base, u64 rowBytes, u64 slotBytes, const int* ids, const float* weights, const float* dScale, const i8* packed, const float* scales, const int* sums, float* out, int hidden, int width, int used, GridTables t) {
    __shared__ int shExp[32];
    __shared__ float shWeight[32];
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int c = blockIdx.x * 8 + warp, tok = blockIdx.y;
    if(threadIdx.x < used){
        int j = tok * used + threadIdx.x;
        int e = ids[j];
        shExp[threadIdx.x] = e;
        shWeight[threadIdx.x] = weights[j] * (dScale ? dScale[e] : 1.0f);
    }
    __syncthreads();
    if(c >= hidden) return;
    int groups = width / 32, perBlock = groupsPerBlock(TYPE), bytes = blockBytes(TYPE);
    int totalGroups = used * groups;
    float total = 0.0f;
    for(int u = lane; u < totalGroups; u += 32){
        int k = u / groups, g = u % groups;
        int j = tok * used + k, e = shExp[k];
        const u8* line = base + slotBytes * (u64)e + rowBytes * (u64)c;
        int w[8];
        float low, high, shift;
        inkUnpack<TYPE>(line + (g / perBlock) * bytes, g % perBlock, w, low, high, shift, t);
        u64 at = (u64)j * groups + g;
        const int4* q = (const int4*)(packed + at * 32);
        int4 a = __ldg(q), b = __ldg(q + 1);
        int lowSum = __dp4a(w[3], a.w, __dp4a(w[2], a.z, __dp4a(w[1], a.y, __dp4a(w[0], a.x, 0))));
        int highSum = __dp4a(w[7], b.w, __dp4a(w[6], b.z, __dp4a(w[5], b.y, __dp4a(w[4], b.x, 0))));
        total += __ldg(scales + at) * (low * (float)lowSum + high * (float)highSum - shift * (float)__ldg(sums + at)) * shWeight[k];
    }
    for(int s = 16; s > 0; s >>= 1) total += __shfl_xor_sync(0xffffffff, total, s);
    if(lane == 0) out[(u64)tok * hidden + c] = total;
}

extern "C" __global__ void inkDownSum40(const u8* base, u64 rowBytes, u64 slotBytes, const int* ids, const float* weights, const float* dScale, const i8* packed, const float* scales, const int* sums, float* out, int hidden, int width, int used, GridTables t) {
    inkDownSumRun<40>(base, rowBytes, slotBytes, ids, weights, dScale, packed, scales, sums, out, hidden, width, used, t);
}

extern "C" __global__ void inkDownSum8(const u8* base, u64 rowBytes, u64 slotBytes, const int* ids, const float* weights, const float* dScale, const i8* packed, const float* scales, const int* sums, float* out, int hidden, int width, int used, GridTables t) {
    inkDownSumRun<8>(base, rowBytes, slotBytes, ids, weights, dScale, packed, scales, sums, out, hidden, width, used, t);
}

extern "C" __global__ void inkGateUpAct30(const u8* base, u64 rowBytes, u64 slotBytes, const int* ids, const float* guScale, const float* xn, float* outAct, int hidden, int width, int used, float gateCap, float upCap) {
    int warp = threadIdx.x >> 5, lane8 = threadIdx.x & 7, chInWarp = (threadIdx.x >> 3) & 3;
    int c = blockIdx.x * 32 + warp * 4 + chInWarp, j = blockIdx.y;
    int e = ids[j], tok = j / used;
    const u8* expert = base + slotBytes * (u64)e;
    const u16* gRow = (const u16*)(expert + rowBytes * (u64)c);
    const u16* uRow = (const u16*)(expert + rowBytes * (u64)(width + c));
    const float* xRow = xn + (u64)tok * hidden;
    float gAcc = 0.0f, uAcc = 0.0f;
    for(int i = lane8; i < hidden; i += 8){
        float xv = xRow[i];
        gAcc += __uint_as_float((u32)gRow[i] << 16) * xv;
        uAcc += __uint_as_float((u32)uRow[i] << 16) * xv;
    }
    for(int s = 4; s > 0; s >>= 1){
        gAcc += __shfl_xor_sync(0xffffffff, gAcc, s);
        uAcc += __shfl_xor_sync(0xffffffff, uAcc, s);
    }
    if(lane8 == 0){
        float sc = guScale ? guScale[e] : 1.0f;
        float g = gAcc * sc, u = uAcc * sc;
        outAct[(u64)j * width + c] = tanhf(g / gateCap) * gateCap * hySigmoid(g) * tanhf(u / upCap) * upCap;
    }
}

extern "C" __global__ void inkDownSum30(const u8* base, u64 rowBytes, u64 slotBytes, const int* ids, const float* weights, const float* dScale, const float* act, float* out, int hidden, int width, int used) {
    __shared__ int shExp[32];
    __shared__ float shWeight[32];
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int c = blockIdx.x * 8 + warp, tok = blockIdx.y;
    if(threadIdx.x < used){
        int j = tok * used + threadIdx.x;
        int e = ids[j];
        shExp[threadIdx.x] = e;
        shWeight[threadIdx.x] = weights[j] * (dScale ? dScale[e] : 1.0f);
    }
    __syncthreads();
    if(c >= hidden) return;
    float total = 0.0f;
    for(int k = 0; k < used; k++){
        const u16* row = (const u16*)(base + slotBytes * (u64)shExp[k] + rowBytes * (u64)c);
        const float* x = act + (u64)(tok * used + k) * width;
        float acc = 0.0f;
        for(int i = lane; i < width; i += 32) acc += __uint_as_float((u32)row[i] << 16) * x[i];
        total += acc * shWeight[k];
    }
    for(int s = 16; s > 0; s >>= 1) total += __shfl_xor_sync(0xffffffff, total, s);
    if(lane == 0) out[(u64)tok * hidden + c] = total;
}

extern "C" __global__ void inkHead(float* logits, int count, int unpadded, float cap) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= count) return;
    logits[i] = i >= unpadded ? -1e30f : tanhf(logits[i] / cap) * cap;
}

extern "C" __global__ void inkCopy(const float* a, float* b, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < count) b[i] = a[i];
}

extern "C" __global__ void inkAdd(const float* a, const float* b, float* out, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < count) out[i] = a[i] + b[i];
}

extern "C" __global__ void inkSplitQ8(const u8* src, u64 rowBytes, i8* values, u16* scales, int rows, int cols) {
    u64 g = (u64)blockIdx.x * blockDim.y + threadIdx.y;
    u64 groups = (u64)(cols / 32);
    if(g >= (u64)rows * groups) return;
    u64 r = g / groups, u = g % groups;
    const u8* blk = src + r * rowBytes + u * 34;
    if(threadIdx.x == 0) scales[g] = (u16)blk[0] | ((u16)blk[1] << 8);
    values[g * 32 + threadIdx.x] = (i8)blk[2 + threadIdx.x];
}

extern "C" __global__ void inkSplitBF16ToQ8(const u16* src, i8* values, u16* scales, int rows, int cols) {
    u64 g = (u64)blockIdx.x * blockDim.y + threadIdx.y;
    u64 groups = (u64)(cols / 32);
    if(g >= (u64)rows * groups) return;
    float v = __uint_as_float((u32)src[g * 32 + threadIdx.x] << 16);
    float peak = fabsf(v);
    for(int s = 16; s > 0; s >>= 1) peak = fmaxf(peak, __shfl_xor_sync(0xffffffff, peak, s));
    float d = peak / 127.0f;
    int q = d > 0.0f ? __float2int_rn(v / d) : 0;
    values[g * 32 + threadIdx.x] = (i8)q;
    if(threadIdx.x == 0) scales[g] = toHalf(d);
}

extern "C" __global__ void inkRowsQ8(const i8* v0, const u16* s0, int r0, float* o0, const i8* v1, const u16* s1, int r1, float* o1, const i8* v2, const u16* s2, int r2, float* o2, const i8* v3, const u16* s3, int r3, float* o3, const i8* v4, const u16* s4, int r4, float* o4, const i8* v5, const u16* s5, int r5, float* o5, const i8* packed, const float* xScales, int cols, int n) {
    int lane8 = threadIdx.x & 7;
    int row = blockIdx.x * (blockDim.x >> 3) + (threadIdx.x >> 3), tok = blockIdx.y;
    const i8* wv = v0;
    const u16* ws = s0;
    float* out = o0;
    int r = row, rows = r0;
    if(r >= rows){ r -= rows; wv = v1; ws = s1; out = o1; rows = r1; }
    if(r >= rows){ r -= rows; wv = v2; ws = s2; out = o2; rows = r2; }
    if(r >= rows){ r -= rows; wv = v3; ws = s3; out = o3; rows = r3; }
    if(r >= rows){ r -= rows; wv = v4; ws = s4; out = o4; rows = r4; }
    if(r >= rows){ r -= rows; wv = v5; ws = s5; out = o5; rows = r5; }
    if(r >= rows) return;
    int groups = cols / 32;
    u64 xFirst = (u64)tok * groups;
    u64 wFirst = (u64)r * groups;
    float acc = 0.0f;
    for(int u = lane8; u < groups; u += 8){
        const int4* wq = (const int4*)(wv + (wFirst + u) * 32);
        int4 w0 = __ldg(wq), w1 = __ldg(wq + 1);
        const int4* xq = (const int4*)(packed + (xFirst + u) * 32);
        int4 x0 = __ldg(xq), x1 = __ldg(xq + 1);
        int sum = __dp4a(w0.w, x0.w, __dp4a(w0.z, x0.z, __dp4a(w0.y, x0.y, __dp4a(w0.x, x0.x, 0))));
        sum = __dp4a(w1.w, x1.w, __dp4a(w1.z, x1.z, __dp4a(w1.y, x1.y, __dp4a(w1.x, x1.x, sum))));
        acc += hyHalf(__ldg(ws + wFirst + u)) * __ldg(xScales + xFirst + u) * (float)sum;
    }
    for(int s = 4; s > 0; s >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, s);
    if(lane8 == 0) out[(u64)tok * rows + r] = acc;
}

extern "C" __global__ void inkRouteFast(const float* logits, const float* bias, int* ids, float* weights, int experts, int used, float scale) {
    int t = blockIdx.x, lane = threadIdx.x;
    float p[4], pk[4];
    #pragma unroll
    for(int i = 0; i < 4; i++){
        int e = lane + 32 * i;
        float pr = e < experts ? hySigmoid(logits[(u64)t * experts + e]) : 0.0f;
        float v = e < experts ? pr + (bias ? bias[e] : 0.0f) : -1e30f;
        p[i] = pr;
        pk[i] = v == v ? v : -1e30f;
    }
    int chosen[8];
    float cprob[8];
    for(int k = 0; k < used; k++){
        float best = -1e30f;
        int at = 0;
        float bprob = 0.0f;
        #pragma unroll
        for(int i = 0; i < 4; i++){
            int e = lane + 32 * i;
            if(pk[i] > best || (pk[i] == best && e < at)){
                best = pk[i];
                at = e;
                bprob = p[i];
            }
        }
        for(int s = 16; s > 0; s >>= 1){
            float ob = __shfl_xor_sync(0xffffffff, best, s);
            int oa = __shfl_xor_sync(0xffffffff, at, s);
            float op = __shfl_xor_sync(0xffffffff, bprob, s);
            if(ob > best || (ob == best && oa < at)){
                best = ob;
                at = oa;
                bprob = op;
            }
        }
        if(lane == 0){
            chosen[k] = at;
            cprob[k] = bprob;
        }
        #pragma unroll
        for(int i = 0; i < 4; i++) if(lane + 32 * i == at) pk[i] = -1e30f;
    }
    if(lane == 0){
        float sum = 0.0f;
        for(int k = 0; k < used; k++) sum += cprob[k];
        float inv = sum > 0.0f ? 1.0f / sum : 1.0f;
        for(int k = 0; k < used; k++){
            ids[t * used + k] = chosen[k];
            weights[t * used + k] = cprob[k] * inv * scale;
        }
    }
}

extern "C" __global__ void inkTopPart(const float* logits, int count, float* candVal, int* candId) {
    __shared__ float values[1024];
    __shared__ int ids[1024];
    int tid = threadIdx.x;
    float best[4] = {-3.0e38f, -3.0e38f, -3.0e38f, -3.0e38f};
    int where[4] = {0, 0, 0, 0};
    for(int i = blockIdx.x * blockDim.x + tid; i < count; i += gridDim.x * blockDim.x){
        float v = logits[i];
        if(!(v > best[3])) continue;
        int j = 3;
        while(j > 0 && best[j - 1] < v){
            best[j] = best[j - 1];
            where[j] = where[j - 1];
            j--;
        }
        best[j] = v;
        where[j] = i;
    }
    #pragma unroll
    for(int j = 0; j < 4; j++){
        values[tid * 4 + j] = best[j];
        ids[tid * 4 + j] = where[j];
    }
    __syncthreads();
    for(int size = 2; size <= 1024; size <<= 1){
        for(int stride = size >> 1; stride > 0; stride >>= 1){
            for(int i = tid; i < 1024; i += blockDim.x){
                int partner = i ^ stride;
                if(partner <= i) continue;
                float a = values[i], b = values[partner];
                if((i & size) == 0 ? a < b : a > b){
                    values[i] = b;
                    values[partner] = a;
                    int t = ids[i];
                    ids[i] = ids[partner];
                    ids[partner] = t;
                }
            }
            __syncthreads();
        }
    }
    if(tid < 64){
        candVal[blockIdx.x * 64 + tid] = values[tid];
        candId[blockIdx.x * 64 + tid] = ids[tid];
    }
}

extern "C" __global__ void inkTopMerge(const float* candVal, const int* candId, int total, int keep, float* out) {
    __shared__ float values[4096];
    __shared__ int ids[4096];
    int tid = threadIdx.x;
    float best[8];
    int where[8];
    #pragma unroll
    for(int j = 0; j < 8; j++){
        best[j] = -3.0e38f;
        where[j] = 0;
    }
    for(int i = tid; i < total; i += blockDim.x){
        float v = candVal[i];
        if(!(v > best[7])) continue;
        int j = 7;
        while(j > 0 && best[j - 1] < v){
            best[j] = best[j - 1];
            where[j] = where[j - 1];
            j--;
        }
        best[j] = v;
        where[j] = candId[i];
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
                if((i & size) == 0 ? a < b : a > b){
                    values[i] = b;
                    values[partner] = a;
                    int t = ids[i];
                    ids[i] = ids[partner];
                    ids[partner] = t;
                }
            }
            __syncthreads();
        }
    }
    for(int j = tid; j < keep; j += blockDim.x){
        out[j] = values[j];
        out[keep + j] = __int_as_float(ids[j]);
    }
}
)";
