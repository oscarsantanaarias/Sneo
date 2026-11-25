#pragma once
#include "glm5.h"

struct InkLayer {
    bool full = false;
    bool loop = false;
    GpuMatrix q, k, v, gate, out, rel, sink, kProj, vProj, gateUp, down;
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
    uint64_t h[2] = {}, blocks[4] = {}, prevBlock = 0, prevH = 0, msgs[4] = {};
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
    int pairsMost = 0;
    void* routed = nullptr;
    std::vector<std::vector<uint8_t>> saved;
    std::vector<size_t> savedTokens;
};

const char* inkKernelSource = R"(
extern "C" __global__ void inkBlockRes(const float* s0, const float* s1, const float* s2, const float* s3, const float* partial, int blocks, const float* query, int queries, float* o0, float* o1, float* o2, float* o3, int hidden, float eps) {
    __shared__ float red[25][8];
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

extern "C" __global__ void inkMsgDown(const float* h, const float* m, const float* down, float* gates, int hidden, int rank, float eps) {
    __shared__ float cat[4608];
    __shared__ float part[32];
    int t = blockIdx.x;
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    const float* hs = h + (u64)t * hidden;
    const float* ms = m + (u64)t * hidden;
    float a = 0.0f, b = 0.0f;
    for(int c = threadIdx.x; c < hidden; c += blockDim.x){
        a += hs[c] * hs[c];
        b += ms[c] * ms[c];
    }
    float ia = rsqrtf(hyBlockSum(a, part) / hidden + eps);
    __syncthreads();
    float ib = rsqrtf(hyBlockSum(b, part) / hidden + eps);
    for(int c = threadIdx.x; c < hidden; c += blockDim.x){
        cat[c] = hs[c] * ia;
        cat[hidden + c] = ms[c] * ib;
    }
    __syncthreads();
    int r = blockIdx.y * warps + warp;
    if(r >= rank) return;
    const float* row = down + (u64)r * 2 * hidden;
    float v = 0.0f;
    for(int i = lane; i < 2 * hidden; i += 32) v += row[i] * cat[i];
    for(int s = 16; s > 0; s >>= 1) v += __shfl_xor_sync(0xffffffff, v, s);
    if(lane == 0) gates[(u64)t * rank + r] = v;
}

extern "C" __global__ void inkMsgMix(const float* h, const float* m, const float* gates, const float* forget, const float* gain, float* out, int hidden, int rank) {
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int c = blockIdx.x * (blockDim.x >> 5) + warp, t = blockIdx.y;
    if(c >= hidden) return;
    const float* gt = gates + (u64)t * rank;
    const float* fr = forget + (u64)c * rank;
    const float* gr = gain + (u64)c * rank;
    float f = 0.0f, k = 0.0f;
    for(int r = lane; r < rank; r += 32){
        float gv = gt[r];
        f += fr[r] * gv;
        k += gr[r] * gv;
    }
    for(int s = 16; s > 0; s >>= 1){
        f += __shfl_xor_sync(0xffffffff, f, s);
        k += __shfl_xor_sync(0xffffffff, k, s);
    }
    u64 at = (u64)t * hidden + c;
    if(lane == 0) out[at] = (1.0f - tanhf(f)) * h[at] + tanhf(k) * m[at];
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

extern "C" __global__ void inkAttendPart(const float* q, const u16* kCache, const u16* vCache, int ring, const int* positions, const float* rel, const float* bank, float* scratch, int heads, int kvHeads, int window, int extent, float scale, int span, int splits) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int h = blockIdx.x * 8 + warp, t = blockIdx.y / splits, part = blockIdx.y % splits;
    if(h >= heads) return;
    int g = h / (heads / kvHeads);
    float* dst = scratch + (((u64)t * heads + h) * splits + part) * 130;
    int pos = positions[t];
    int lo = window > 0 ? max(0, pos - window + 1) : 0;
    int first = lo + part * span;
    int last = min(pos, first + span - 1);
    const float* qr = q + ((u64)t * heads + h) * 128;
    float qv[4];
    #pragma unroll
    for(int i = 0; i < 4; i++) qv[i] = qr[lane + 32 * i];
    float rs[16];
    const float* rr = rel + ((u64)t * heads + h) * 16;
    #pragma unroll
    for(int i = 0; i < 16; i++) rs[i] = rr[i];
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float best = -1e30f, total = 0.0f;
    for(int j = first; j <= last; j++){
        u64 slot = (u64)(ring > 0 ? j % ring : j);
        const u16* kr = kCache + (slot * kvHeads + g) * 128;
        float dot = 0.0f;
        #pragma unroll
        for(int i = 0; i < 4; i++) dot += qv[i] * hyHalf(kr[lane + 32 * i]);
        for(int s = 16; s > 0; s >>= 1) dot += __shfl_xor_sync(0xffffffff, dot, s);
        int dist = pos - j;
        float bias = 0.0f;
        if(dist < extent){
            const float* br = bank + (u64)dist * 16;
            #pragma unroll
            for(int i = 0; i < 16; i++) bias += rs[i] * br[i];
        }
        float score = dot * scale + bias;
        float top = fmaxf(best, score);
        float keep = __expf(best - top);
        float p = __expf(score - top);
        total = total * keep + p;
        const u16* vr = vCache + (slot * kvHeads + g) * 128;
        #pragma unroll
        for(int i = 0; i < 4; i++) acc[i] = acc[i] * keep + p * hyHalf(vr[lane + 32 * i]);
        best = top;
    }
    if(lane == 0){
        dst[0] = best;
        dst[1] = total;
    }
    #pragma unroll
    for(int i = 0; i < 4; i++) dst[2 + lane + 32 * i] = acc[i];
}

extern "C" __global__ void inkAttendMerge(const float* scratch, const float* sinks, float* out, int heads, int splits) {
    int h = blockIdx.x, t = blockIdx.y;
    const float* src = scratch + ((u64)t * heads + h) * splits * 130;
    float sink = sinks[(u64)t * heads + h];
    float top = sink;
    for(int s = 0; s < splits; s++) top = fmaxf(top, src[(u64)s * 130]);
    float total = __expf(sink - top);
    for(int s = 0; s < splits; s++) total += src[(u64)s * 130 + 1] * __expf(src[(u64)s * 130] - top);
    float inv = 1.0f / total;
    float* dst = out + ((u64)t * heads + h) * 128;
    for(int d = threadIdx.x; d < 128; d += blockDim.x){
        float v = 0.0f;
        for(int s = 0; s < splits; s++) v += src[(u64)s * 130 + 2 + d] * __expf(src[(u64)s * 130] - top);
        dst[d] = v * inv;
    }
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
)";
