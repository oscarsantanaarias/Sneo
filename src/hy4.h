#pragma once
#include "gguf.h"
#include "kernels.h"
#include <vector>
#include <string>

// hy_v4_flash: cuatro corrientes residuales, atencion mla con sumideros, expertos en un ancho latente y tablas engram de n-gramas
struct HyLayer {
    bool sliding = false;
    bool dense = false;
    int heads = 0;
    int engram = -1;
    GpuMatrix qa, qb, kva, out, gate;
    GpuMatrix ffnGate, ffnUp, ffnDown;
    GpuMatrix routedDown, routedUp, sharedGate, sharedUp, sharedDown;
    GpuMatrix engramKv;
    uint64_t attnNorm = 0, qaNorm = 0, kvNorm = 0, sinks = 0, postAttnNorm = 0, ffnNorm = 0, postFfnNorm = 0;
    uint64_t hcAttnFn = 0, hcAttnBase = 0, hcAttnScale = 0, hcFfnFn = 0, hcFfnBase = 0, hcFfnScale = 0;
    uint64_t router = 0, routerBias = 0, kb = 0, vb = 0;
    uint64_t engramBias = 0, engramGate = 0, engramNorm = 0, engramGamma = 0, engramScale = 0;
    uint64_t cache = 0;
    int slots = 0;
    float clamp = 0.0f;
    // indexpool de las capas dispersas: elige hasta 2048 posiciones por consulta
    bool indexed = false;
    GpuMatrix idxQ, idxK;
    uint64_t idxKNorm = 0, idxProj = 0, idxApe = 0, idxNorm = 0, idxGate = 0, idxValue = 0, idxKeys = 0, idxPools = 0;
    // copia de las posiciones elegidas en la ultima pasada para que el ayudante mire lo mismo que el modelo
    uint64_t idxSnap = 0, idxSnapCount = 0;
    int idxSnapFirst = -1, idxSnapRows = 0, idxSnapCap = 0;
    // los expertos se leen del archivo mapeado y la cpu los calcula
    Matrix hostGate, hostUp, hostDown;
    uint64_t gateBytes = 0, upBytes = 0, downBytes = 0;
};

struct HyShape {
    int layers = 0;
    int hidden = 0;
    int hc = 4;
    int qLora = 0;
    int kvLora = 0;
    int rope = 64;
    int nope = 192;
    int headValue = 256;
    int window = 512;
    int ring = 0;
    int experts = 0;
    int used = 0;
    int latent = 0;
    int expertWidth = 0;
    int feedForward = 0;
    int vocab = 0;
    int widestHeads = 0;
    float eps = 1e-6f;
    float hcEps = 1e-6f;
    float hcMag = 2.0f;
    float ropeBase = 1e7f;
    float ropeBaseSliding = 1e4f;
    float expertScale = 1.0f;
    bool expertNorm = true;
    // engram
    int engramModules = 0;
    int engramHeads = 0;
    int engramDim = 0;
    int engramGram = 3;
    int engramPerGram = 8;
    std::vector<uint64_t> engramMult;
    std::vector<uint32_t> engramPrimes, engramOffsets;
    std::vector<int> engramMap;
    int engramPad = 0;
    const uint8_t* engramTable = nullptr;
    // indexpool: 511 grupos de 4 elegidos mas el grupo propio, claves de token en un anillo de 512 posiciones
    bool indexed = false;
    int idxKeep = 511, idxRing = 512, idxGroups = 0, idxWidth = 2048;
    float idxScale = 0.022097086912079608f;
};

// los expertos mas usados viven en la vram y el resto lo calcula la cpu en la misma capa
struct HyCache {
    int slots = 0;
    uint64_t slotBytes = 0;
    uint64_t base = 0;
    std::vector<int> slotOf;
    std::vector<int> owner;
    std::vector<uint32_t> heat;
    std::vector<int> free;
    uint64_t hits = 0, total = 0, uploads = 0, skipped = 0;
    double cpuMs = 0, waitMs = 0, uploadMs = 0, gpuQueueMs = 0;
    // bytes de expertos que leyo la cpu, para ver a cuantos GB/s anda contra la ram
    uint64_t cpuBytes = 0;
    uint64_t launches = 0;
    uint32_t lastStep = 0;
    int budget = 6;
    // segundo nivel: ram fijada que la gpu lee directo por pcie mientras la cpu hace el resto
    int warmSlots = 0;
    uint8_t* warmHost = nullptr;
    uint64_t warmDevice = 0;
    std::vector<int> warmOf;
    std::vector<int> warmOwner;
    std::vector<int> warmFree;
    float warmShare = 0.3f;
    std::string heatPath;
    uint64_t heatSaved = 0;
    // tercer nivel: la segunda tarjeta guarda los que siguen en calor y los calcula mientras la cpu hace el resto
    int helpSlots = 0;
    uint64_t helpBase = 0;
    std::vector<int> helpOf;
    uint64_t helpIn = 0, helpWeights = 0, helpSlotsOn = 0, helpPairs = 0, helpOut = 0;
    uint64_t helpHitIn = 0, helpGate = 0, helpUp = 0, helpHid = 0, helpDown = 0;
    uint64_t helpValues = 0, helpScales = 0, helpSums = 0;
    float* helpHostOut = nullptr;
    int* helpHostLists = nullptr;
    uint64_t helpUsed = 0;
    double helpMs = 0;
    int warmBudget = 4;
    uint64_t warmUsed = 0;
};

struct HyDraftLayer {
    int target = 0;
    GpuMatrix qa, qb, kva, gate, out, mlpGate, mlpUp, mlpDown;
    uint64_t inNorm = 0, postAttn = 0, qaNorm = 0, kvNorm = 0, sinks = 0, postProj = 0, kb = 0, vb = 0;
};

struct HyDraft {
    bool ready = false;
    std::string problem;
    std::vector<HyDraftLayer> layers;
    GpuMatrix eh, hcGate, hcUp, hcDown;
    uint64_t enorm = 0, hnorm = 0, fuseGate = 0, finalNorm = 0, hcHidden = 0, hcEmbed = 0;
    uint64_t x = 0, hidden = 0, cat = 0, fused = 0, corr = 0, mid = 0;
    float* hostRows = nullptr;
    int* hostPick = nullptr;
    uint64_t mappedRows = 0, mappedPick = 0;
    std::vector<float> maskRow;
    int mask = 120009;
    int length = 3;
    uint64_t bytes = 0;
    // de donde sale el estado del modelo para la proxima propuesta y en que posicion esta
    uint64_t from = 0;
    int at = -1;
    uint64_t proposed = 0, steps = 0, firstKept = 0, kept = 0;
    double ms = 0;
    // probabilidad minima para proponer y cuantas veces se corto por no estar seguro
    float sure = 0.5f;
    // listas de posiciones de cada fila del ayudante cuando el modelo eligio con su indexador
    uint64_t list = 0, counts = 0;
    int listWidth = 0;
    uint64_t sparseSteps = 0;
    uint64_t cut = 0;
};

struct HyBuffers {
    uint64_t streams = 0, flat = 0, mixes = 0, post = 0, cur = 0, xn = 0;
    uint64_t qa = 0, q = 0, kv = 0, qfull = 0, latentOut = 0, heads = 0, gateOut = 0, attn = 0;
    uint64_t logits = 0, bigGate = 0, bigUp = 0, bigHid = 0, ffn = 0, shared = 0, sGate = 0, sUp = 0, sHid = 0;
    uint64_t routedIn = 0, routedOut = 0, positions = 0, last = 0, vocab = 0, outNorm = 0;
    uint64_t engramIn = 0, engramNormed = 0, engramKv = 0;
    uint64_t hitIn = 0, hitGate = 0, hitUp = 0, hitHid = 0, hitDown = 0;
    uint64_t attnScratch = 0;
    // verificacion de varias filas: estado normado y logits de cada fila
    uint64_t normedRows = 0, every = 0, mappedEvery = 0;
    float* hostEvery = nullptr;
    int everyCap = 0;
    bool saved[4] = {};
    uint64_t scratchFloats = 0;
    int* hostHits = nullptr;
    uint64_t mappedHits = 0;
    uint64_t headFn = 0, headBase = 0, headScale = 0;
    // memoria del host que la gpu lee y escribe directo
    float* hostEmbed = nullptr;
    float* hostLogits = nullptr;
    float* hostLatent = nullptr;
    float* hostMoe = nullptr;
    float* hostEngram = nullptr;
    int* hostIds = nullptr;
    float* hostWeights = nullptr;
    int* hostPositions = nullptr;
    uint64_t mappedEmbed = 0, mappedLogits = 0, mappedLatent = 0, mappedMoe = 0, mappedEngram = 0, mappedIds = 0, mappedWeights = 0;
    int ringTop = -1;
    // prompt largo capa por capa: el bloque grande usa el final de la cache de expertos mientras dura
    int wideRows = 0, wideStage = 0, wideBorrow = 0, wideMin = 0;
    int* hostWideIds = nullptr;
    float* hostWideWeights = nullptr;
    float* hostWideEngram = nullptr;
    uint64_t wideIds = 0, wideWeights = 0, wideEngram = 0;
    uint64_t wideUploaded = 0;
    double wideUploadMs = 0, wideBatchMs = 0, wideEngramMs = 0, wideLayersMs = 0;
    // subida en dos mitades: la cpu llena una mitad fijada mientras la otra viaja y la gpu calcula
    uint8_t* widePinned = nullptr;
    int* hostWidePairs = nullptr;
    uint64_t widePairs = 0;
    void* wideStream = nullptr;
    void* wideCopied[2] = {};
    void* wideDone[2] = {};
    // expertos agrupados con tensor cores: entrada del bloque cuantizada y salidas por grupo de filas
    int widePairsMost = 8192;
    // filas por tramo de atencion en el prompt largo: mas filas por kernel que los 64 de siempre
    int wideSub = 0;
    // posicion de la primera fila de la pasada que corre ahora
    int firstPos = 0;
    // indexpool por fila: clave, consulta de 16 cabezas, pesos, puntajes por grupo y lista de posiciones elegidas
    uint64_t idxKey = 0, idxQ = 0, idxW = 0, idxScores = 0, idxList = 0, idxCount = 0;
    // absorcion y expansion de mla con cublas por lotes cuando hay muchas filas
    void* blas = nullptr;
    void* blasBatched = nullptr;
    uint64_t q16 = 0, lat16 = 0;
    uint64_t wideInValues = 0, wideInScales = 0, wideInSums = 0;
    uint64_t wideGate = 0, wideUp = 0, wideHid = 0, wideHidValues = 0, wideHidScales = 0, wideHidSums = 0;
};

const char* hyKernelSource = R"(
__device__ __forceinline__ float hyHalf(u16 h) {
    return half(h);
}

__device__ __forceinline__ float hyBlockSum(float v, float* shared) {
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for(int s = 16; s > 0; s >>= 1) v += __shfl_xor_sync(0xffffffff, v, s);
    __syncthreads();
    if(lane == 0) shared[warp] = v;
    __syncthreads();
    float total = 0.0f;
    int warps = (blockDim.x + 31) >> 5;
    for(int w = 0; w < warps; w++) total += shared[w];
    return total;
}

__device__ __forceinline__ float hySigmoid(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

// un warp cuantiza su grupo de 32 al formato que leen las proyecciones: int8, escala y suma por grupo
__device__ __forceinline__ void hyQuant(float v, u64 index, i8* packed, float* scales, int* sums) {
    float peak = fabsf(v);
    for(int s = 16; s > 0; s >>= 1) peak = fmaxf(peak, __shfl_xor_sync(0xffffffff, peak, s));
    float d = peak / 127.0f;
    int q = d > 0.0f ? __float2int_rn(v / d) : 0;
    packed[index] = (i8)q;
    int total = q;
    for(int s = 16; s > 0; s >>= 1) total += __shfl_xor_sync(0xffffffff, total, s);
    if((threadIdx.x & 31) == 0){
        scales[index / 32] = d;
        sums[index / 32] = total;
    }
}

// norma rms por filas: cada fila es un grupo de un token y el peso puede cambiar por grupo
extern "C" __global__ void hyNorm(const float* in, int inStride, float* out, int outStride, const float* weight, int weightStride, int groups, int cols, float eps) {
    __shared__ float part[32];
    int row = blockIdx.x;
    int t = row / groups, g = row % groups;
    const float* src = in + (u64)t * inStride + (u64)g * cols;
    float* dst = out + (u64)t * outStride + (u64)g * cols;
    const float* w = weight ? weight + (u64)g * weightStride : 0;
    float sum = 0.0f;
    for(int c = threadIdx.x; c < cols; c += blockDim.x) sum += src[c] * src[c];
    float inv = rsqrtf(hyBlockSum(sum, part) / cols + eps);
    for(int c = threadIdx.x; c < cols; c += blockDim.x) dst[c] = src[c] * inv * (w ? w[c] : 1.0f);
}

extern "C" __global__ void hyNormPack(float* x, const float* weight, int cols, float eps, i8* packed, float* scales, int* sums) {
    __shared__ float part[32];
    int t = blockIdx.x;
    float* row = x + (u64)t * cols;
    float sum = 0.0f;
    for(int c = threadIdx.x; c < cols; c += blockDim.x) sum += row[c] * row[c];
    float inv = rsqrtf(hyBlockSum(sum, part) / cols + eps);
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    for(int g = warp; g < cols / 32; g += warps){
        int c = g * 32 + lane;
        float v = row[c] * inv * weight[c];
        hyQuant(v, (u64)t * cols + c, packed, scales, sums);
    }
}

// pesos en f32 guardados por filas: un warp por fila
extern "C" __global__ void hyF32Rows(const float* weights, const float* in, int inStride, float* out, int outStride, int rows, int cols, int n) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp;
    if(row >= rows) return;
    const float* w = weights + (u64)row * cols;
    for(int t = 0; t < n; t++){
        const float* x = in + (u64)t * inStride;
        float sum = 0.0f;
        for(int c = lane; c < cols; c += 32) sum += w[c] * x[c];
        for(int s = 16; s > 0; s >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, s);
        if(lane == 0) out[(u64)t * outStride + row] = sum;
    }
}

// mezcla de entrada de las corrientes: pre las junta en una y post dice cuanto vuelve a cada una
extern "C" __global__ void hyMixPre(const float* streams, const float* mixes, int mixStride, const float* scale, const float* base, float* out, float* post, int hidden, int hc, float eps, float mag) {
    __shared__ float pre[8];
    int t = blockIdx.x;
    const float* m = mixes + (u64)t * mixStride;
    if(threadIdx.x < hc) pre[threadIdx.x] = hySigmoid(m[threadIdx.x] * scale[0] + base[threadIdx.x]) + eps;
    if(post && threadIdx.x >= hc && threadIdx.x < 2 * hc){
        int i = threadIdx.x - hc;
        post[(u64)t * hc + i] = hySigmoid(m[hc + i] * scale[1] + base[hc + i]) * mag + eps;
    }
    __syncthreads();
    const float* s = streams + (u64)t * hc * hidden;
    for(int c = threadIdx.x; c < hidden; c += blockDim.x){
        float v = 0.0f;
        for(int i = 0; i < hc; i++) v += pre[i] * s[(u64)i * hidden + c];
        out[(u64)t * hidden + c] = v;
    }
}

extern "C" __global__ void hyMixPost(float* streams, const float* x, const float* post, int hidden, int hc) {
    int t = blockIdx.x;
    float* s = streams + (u64)t * hc * hidden;
    const float* p = post + (u64)t * hc;
    for(int c = threadIdx.x; c < hidden; c += blockDim.x){
        float v = x[(u64)t * hidden + c];
        for(int i = 0; i < hc; i++) s[(u64)i * hidden + c] += v * p[i];
    }
}

extern "C" __global__ void hySpread(const float* x, float* streams, int hidden, int hc) {
    int t = blockIdx.x;
    for(int c = threadIdx.x; c < hidden; c += blockDim.x){
        float v = x[(u64)t * hidden + c];
        for(int i = 0; i < hc; i++) streams[((u64)t * hc + i) * hidden + c] = v;
    }
}

// rope normal: gira pares seguidos de la cola de cada cabeza
extern "C" __global__ void hyRope(float* data, int tokenStride, int headStride, int offset, const int* positions, float base, int dims) {
    int t = blockIdx.x, h = blockIdx.y, i = threadIdx.x;
    if(2 * i >= dims) return;
    float* p = data + (u64)t * tokenStride + (u64)h * headStride + offset + 2 * i;
    float theta = (float)positions[t] * powf(base, -2.0f * i / dims);
    float c = cosf(theta), s = sinf(theta);
    float a = p[0], b = p[1];
    p[0] = a * c - b * s;
    p[1] = a * s + b * c;
}

// la parte sin rope de la consulta pasa por k_b de su cabeza y queda en el ancho latente
extern "C" __global__ void hyAbsorb(const float* q, const u16* kb, float* qfull, int heads, int nope, int rope, int kvLora, int headWidth) {
    __shared__ float qn[256];
    int h = blockIdx.x, t = blockIdx.y;
    const float* src = q + ((u64)t * heads + h) * headWidth;
    for(int k = threadIdx.x; k < nope; k += blockDim.x) qn[k] = src[k];
    __syncthreads();
    float* dst = qfull + ((u64)t * heads + h) * (kvLora + rope);
    const u16* w = kb + (u64)h * kvLora * nope;
    for(int r = threadIdx.x; r < kvLora; r += blockDim.x){
        const u16* line = w + (u64)r * nope;
        float sum = 0.0f;
        for(int k = 0; k < nope; k++) sum += hyHalf(line[k]) * qn[k];
        dst[r] = sum;
    }
    for(int k = threadIdx.x; k < rope; k += blockDim.x) dst[kvLora + k] = src[nope + k];
}

// guarda la fila latente de cada token en la cache de la capa
// la cache latente va en f16: ocupa la mitad y deja lugar para mas expertos en la vram
extern "C" __global__ void hyStore(const float* kv, u16* cache, const int* positions, int ring, int width) {
    int t = blockIdx.x;
    int pos = positions[t];
    int slot = ring > 0 ? pos % ring : pos;
    for(int c = threadIdx.x; c < width; c += blockDim.x) cache[(u64)slot * width + c] = toHalf(kv[(u64)t * width + c]);
}

// atencion mqa sobre la cache latente: un warp por cabeza, softmax en linea y el sumidero como un logit mas sin valor
extern "C" __global__ void hyAttend(const float* qfull, const u16* cache, int ring, const int* positions, const float* sinks, float* out, int heads, int window, float scale) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int h = blockIdx.x * 8 + warp, t = blockIdx.y;
    if(h >= heads) return;
    const float* q = qfull + ((u64)t * heads + h) * 576;
    float qr[18];
    #pragma unroll
    for(int k = 0; k < 18; k++) qr[k] = q[lane + 32 * k];
    float acc[16];
    #pragma unroll
    for(int k = 0; k < 16; k++) acc[k] = 0.0f;
    int pos = positions[t];
    int lo = window > 0 ? max(0, pos - window + 1) : 0;
    float best = -1e30f, total = 0.0f;
    for(int j = lo; j <= pos; j++){
        const u16* row = cache + (u64)(ring > 0 ? j % ring : j) * 576;
        float dot = 0.0f;
        #pragma unroll
        for(int k = 0; k < 18; k++) dot += qr[k] * hyHalf(row[lane + 32 * k]);
        for(int s = 16; s > 0; s >>= 1) dot += __shfl_xor_sync(0xffffffff, dot, s);
        dot *= scale;
        float top = fmaxf(best, dot);
        float keep = __expf(best - top);
        float p = __expf(dot - top);
        total = total * keep + p;
        #pragma unroll
        for(int k = 0; k < 16; k++) acc[k] = acc[k] * keep + p * hyHalf(row[lane + 32 * k]);
        best = top;
    }
    total += __expf(sinks[h] - best);
    float inv = 1.0f / total;
    float* dst = out + ((u64)t * heads + h) * 512;
    #pragma unroll
    for(int k = 0; k < 16; k++) dst[lane + 32 * k] = acc[k] * inv;
}

// la salida latente de cada cabeza pasa por v_b de su cabeza
extern "C" __global__ void hyExpand(const float* lat, const u16* vb, float* out, int heads, int kvLora, int headValue) {
    __shared__ float l[512];
    int h = blockIdx.x, t = blockIdx.y;
    const float* src = lat + ((u64)t * heads + h) * kvLora;
    for(int k = threadIdx.x; k < kvLora; k += blockDim.x) l[k] = src[k];
    __syncthreads();
    const u16* w = vb + (u64)h * headValue * kvLora;
    float* dst = out + ((u64)t * heads + h) * headValue;
    for(int r = threadIdx.x; r < headValue; r += blockDim.x){
        const u16* line = w + (u64)r * kvLora;
        float sum = 0.0f;
        for(int k = 0; k < kvLora; k++) sum += hyHalf(line[k]) * l[k];
        dst[r] = sum;
    }
}

extern "C" __global__ void hySigmoidMul(float* a, const float* g, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < count) a[i] *= hySigmoid(g[i]);
}

extern "C" __global__ void hyAdd(float* a, const float* b, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < count) a[i] += b[i];
}

extern "C" __global__ void hyAxpy(float* out, const float* in, const float* weight, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < count) out[i] += weight[0] * in[i];
}

// los expertos que estan en la vram: cada fila toma la entrada latente de su token
extern "C" __global__ void hyGather(const float* in, const int* pairs, float* out, int latent, int used) {
    int j = blockIdx.x;
    const float* src = in + (u64)(pairs[j] / used) * latent;
    for(int c = threadIdx.x; c < latent; c += blockDim.x) out[(u64)j * latent + c] = src[c];
}

extern "C" __global__ void hyScatter(const float* rows, const int* pairs, const float* weights, float* out, int latent, int used) {
    int j = blockIdx.x;
    int pair = pairs[j];
    float w = weights[pair];
    float* dst = out + (u64)(pair / used) * latent;
    for(int c = threadIdx.x; c < latent; c += blockDim.x) atomicAdd(dst + c, w * rows[(u64)j * latent + c]);
}

// entrada de una capa: mezcla de las corrientes con su norma, junta las corrientes y aplica la norma de la capa
// un bloque por token hace todo y deja la entrada normada lista para las proyecciones
extern "C" __global__ void hyHcIn(const float* streams, const float* fn, const float* scale, const float* base, const float* norm, float* out, float* post, int hidden, int hc, int rows, float eps, float hcEps, float mag) {
    __shared__ float red[9][32];
    __shared__ float mix[9];
    __shared__ float pre[8];
    __shared__ float y[4096];
    int t = blockIdx.x;
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    int width = hc * hidden;
    const float* flat = streams + (u64)t * width;
    float acc[9];
    #pragma unroll
    for(int r = 0; r < 9; r++) acc[r] = 0.0f;
    for(int c = threadIdx.x; c < width; c += blockDim.x){
        float x = flat[c];
        acc[8] += x * x;
        #pragma unroll
        for(int r = 0; r < 8; r++) if(r < rows) acc[r] += fn[(u64)r * width + c] * x;
    }
    #pragma unroll
    for(int r = 0; r < 9; r++){
        float v = acc[r];
        for(int s = 16; s > 0; s >>= 1) v += __shfl_xor_sync(0xffffffff, v, s);
        if(lane == 0) red[r][warp] = v;
    }
    __syncthreads();
    if(threadIdx.x < 9){
        float v = 0.0f;
        for(int w = 0; w < warps; w++) v += red[threadIdx.x][w];
        mix[threadIdx.x] = v;
    }
    __syncthreads();
    float inv = rsqrtf(mix[8] / width + eps);
    if(threadIdx.x < hc) pre[threadIdx.x] = hySigmoid(mix[threadIdx.x] * inv * scale[0] + base[threadIdx.x]) + hcEps;
    if(post && threadIdx.x >= hc && threadIdx.x < 2 * hc){
        int i = threadIdx.x - hc;
        post[(u64)t * hc + i] = hySigmoid(mix[hc + i] * inv * scale[1] + base[hc + i]) * mag + hcEps;
    }
    __syncthreads();
    float sq = 0.0f;
    for(int c = threadIdx.x; c < hidden; c += blockDim.x){
        float v = 0.0f;
        for(int i = 0; i < hc; i++) v += pre[i] * flat[(u64)i * hidden + c];
        y[c] = v;
        sq += v * v;
    }
    for(int s = 16; s > 0; s >>= 1) sq += __shfl_xor_sync(0xffffffff, sq, s);
    __syncthreads();
    if(lane == 0) red[0][warp] = sq;
    __syncthreads();
    float total = 0.0f;
    for(int w = 0; w < warps; w++) total += red[0][w];
    float inv2 = rsqrtf(total / hidden + eps);
    for(int c = threadIdx.x; c < hidden; c += blockDim.x) out[(u64)t * hidden + c] = norm ? y[c] * inv2 * norm[c] : y[c];
}

// salida de una capa: norma de la salida y vuelta a cada corriente con su peso
extern "C" __global__ void hyHcOut(float* streams, const float* x, const float* norm, const float* post, int hidden, int hc, float eps) {
    __shared__ float part[32];
    int t = blockIdx.x;
    const float* src = x + (u64)t * hidden;
    float sq = 0.0f;
    for(int c = threadIdx.x; c < hidden; c += blockDim.x) sq += src[c] * src[c];
    float inv = rsqrtf(hyBlockSum(sq, part) / hidden + eps);
    float* s = streams + (u64)t * hc * hidden;
    const float* p = post + (u64)t * hc;
    for(int c = threadIdx.x; c < hidden; c += blockDim.x){
        float v = src[c] * inv * norm[c];
        for(int i = 0; i < hc; i++) s[(u64)i * hidden + c] += v * p[i];
    }
}

// f32 por filas con un bloque por fila y token: el router de 512 expertos llena la tarjeta
extern "C" __global__ void hyF32Wide(const float* weights, const float* in, int inStride, float* out, int outStride, int cols) {
    __shared__ float part[32];
    int row = blockIdx.x, t = blockIdx.y;
    const float* w = weights + (u64)row * cols;
    const float* x = in + (u64)t * inStride;
    float sum = 0.0f;
    for(int c = threadIdx.x; c < cols; c += blockDim.x) sum += w[c] * x[c];
    sum = hyBlockSum(sum, part);
    if(threadIdx.x == 0) out[(u64)t * outStride + row] = sum;
}

// k_b por cabeza con un warp por fila: las lecturas van seguidas
extern "C" __global__ void hyAbsorbRows(const float* q, const u16* kb, float* qfull, int heads, int nope, int rope, int kvLora, int headWidth, int parts) {
    __shared__ float qn[256];
    int h = blockIdx.x, t = blockIdx.y / parts, part = blockIdx.y % parts;
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const float* src = q + ((u64)t * heads + h) * headWidth;
    for(int k = threadIdx.x; k < nope; k += blockDim.x) qn[k] = src[k];
    __syncthreads();
    float* dst = qfull + ((u64)t * heads + h) * (kvLora + rope);
    const u16* w = kb + (u64)h * kvLora * nope;
    int per = (kvLora + parts - 1) / parts;
    int end = min(kvLora, (part + 1) * per);
    for(int r = part * per + warp; r < end; r += 8){
        const u16* line = w + (u64)r * nope;
        float sum = 0.0f;
        for(int k = lane; k < nope; k += 32) sum += hyHalf(line[k]) * qn[k];
        for(int s = 16; s > 0; s >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, s);
        if(lane == 0) dst[r] = sum;
    }
    if(part == 0) for(int k = threadIdx.x; k < rope; k += blockDim.x) dst[kvLora + k] = src[nope + k];
}

// q sin rope en f16 para multiplicarlo con tensor cores y la parte rope directo a qfull
extern "C" __global__ void hyQHalf(const float* q, u16* q16, float* qfull, int heads, int nope, int rope, int kvLora, int headWidth) {
    u64 row = blockIdx.x;
    const float* src = q + row * headWidth;
    for(int k = threadIdx.x; k < headWidth; k += blockDim.x){
        float v = src[k];
        if(k < nope) q16[row * headWidth + k] = toHalf(v);
        else qfull[row * (kvLora + rope) + kvLora + (k - nope)] = v;
    }
}

extern "C" __global__ void hyToHalf(const float* in, u16* out, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < count) out[i] = toHalf(in[i]);
}

extern "C" __global__ void hyExpandRows(const float* lat, const u16* vb, float* out, int heads, int kvLora, int headValue, int parts) {
    __shared__ float l[512];
    int h = blockIdx.x, t = blockIdx.y / parts, part = blockIdx.y % parts;
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const float* src = lat + ((u64)t * heads + h) * kvLora;
    for(int k = threadIdx.x; k < kvLora; k += blockDim.x) l[k] = src[k];
    __syncthreads();
    const u16* w = vb + (u64)h * headValue * kvLora;
    float* dst = out + ((u64)t * heads + h) * headValue;
    int per = (headValue + parts - 1) / parts;
    int end = min(headValue, (part + 1) * per);
    for(int r = part * per + warp; r < end; r += 8){
        const u16* line = w + (u64)r * kvLora;
        float sum = 0.0f;
        for(int k = lane; k < kvLora; k += 32) sum += hyHalf(line[k]) * l[k];
        for(int s = 16; s > 0; s >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, s);
        if(lane == 0) dst[r] = sum;
    }
}

// atencion repartida por tramos de claves: cada bloque deja su maximo, su suma y su acumulado y despues se juntan
extern "C" __global__ void hyAttendPart(const float* qfull, const u16* cache, int ring, const int* positions, float* scratch, int heads, int window, float scale, int span, int splits) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int h = blockIdx.x * 8 + warp, t = blockIdx.y / splits, part = blockIdx.y % splits;
    if(h >= heads) return;
    float* dst = scratch + (((u64)t * heads + h) * splits + part) * 514;
    int pos = positions[t];
    int lo = window > 0 ? max(0, pos - window + 1) : 0;
    int first = lo + part * span;
    int last = min(pos, first + span - 1);
    const float* q = qfull + ((u64)t * heads + h) * 576;
    float qr[18];
    #pragma unroll
    for(int k = 0; k < 18; k++) qr[k] = q[lane + 32 * k];
    float acc[16];
    #pragma unroll
    for(int k = 0; k < 16; k++) acc[k] = 0.0f;
    float best = -1e30f, total = 0.0f;
    for(int j = first; j <= last; j++){
        const u16* row = cache + (u64)(ring > 0 ? j % ring : j) * 576;
        float dot = 0.0f;
        #pragma unroll
        for(int k = 0; k < 18; k++) dot += qr[k] * hyHalf(row[lane + 32 * k]);
        for(int s = 16; s > 0; s >>= 1) dot += __shfl_xor_sync(0xffffffff, dot, s);
        dot *= scale;
        float top = fmaxf(best, dot);
        float keep = __expf(best - top);
        float p = __expf(dot - top);
        total = total * keep + p;
        #pragma unroll
        for(int k = 0; k < 16; k++) acc[k] = acc[k] * keep + p * hyHalf(row[lane + 32 * k]);
        best = top;
    }
    if(lane == 0){
        dst[0] = best;
        dst[1] = total;
    }
    #pragma unroll
    for(int k = 0; k < 16; k++) dst[2 + lane + 32 * k] = acc[k];
}

extern "C" __global__ void hyAttendMerge(const float* scratch, const float* sinks, float* out, int heads, int splits) {
    int h = blockIdx.x, t = blockIdx.y;
    const float* src = scratch + ((u64)t * heads + h) * splits * 514;
    float top = sinks[h];
    for(int s = 0; s < splits; s++) top = fmaxf(top, src[(u64)s * 514]);
    float total = __expf(sinks[h] - top);
    for(int s = 0; s < splits; s++) total += src[(u64)s * 514 + 1] * __expf(src[(u64)s * 514] - top);
    float inv = 1.0f / total;
    float* dst = out + ((u64)t * heads + h) * 512;
    for(int d = threadIdx.x; d < 512; d += blockDim.x){
        float v = 0.0f;
        for(int s = 0; s < splits; s++) v += src[(u64)s * 514 + 2 + d] * __expf(src[(u64)s * 514] - top);
        dst[d] = v * inv;
    }
}

// indexpool: clave de cada token normada que queda en un anillo de posiciones
extern "C" __global__ void hyIdxKey(const float* raw, const float* norm, float* keys, const int* positions, int ring, float eps) {
    __shared__ float part[32];
    int t = blockIdx.x, c = threadIdx.x;
    float v = raw[(u64)t * 128 + c];
    float total = hyBlockSum(v * v, part);
    keys[(u64)(positions[t] % ring) * 128 + c] = v * rsqrtf(total / 128.0f + eps) * norm[c];
}

// cada token que cierra un grupo de 4 lo junta: compuerta con softmax por canal entre los 4, norma y rope en la posicion del primero
extern "C" __global__ void hyIdxPool(const float* keys, int ring, const int* positions, const float* gate, const float* value, const float* ape, const float* norm, float* pools, float eps, float base) {
    __shared__ float k[4][128];
    __shared__ float out[128];
    __shared__ float part[32];
    int t = blockIdx.x, c = threadIdx.x;
    int pos = positions[t];
    if((pos & 3) != 3) return;
    int g = pos >> 2;
    for(int s = 0; s < 4; s++) k[s][c] = keys[(u64)((4 * g + s) % ring) * 128 + c];
    __syncthreads();
    const float* gw = gate + (u64)c * 128;
    const float* vw = value + (u64)c * 128;
    float z[4], v[4];
    for(int s = 0; s < 4; s++){
        float a = 0.0f, b = 0.0f;
        for(int i = 0; i < 128; i++){
            a += gw[i] * k[s][i];
            b += vw[i] * k[s][i];
        }
        z[s] = a + ape[s * 128 + c];
        v[s] = b;
    }
    float top = fmaxf(fmaxf(z[0], z[1]), fmaxf(z[2], z[3]));
    float sum = 0.0f, pooled = 0.0f;
    for(int s = 0; s < 4; s++){
        float e = __expf(z[s] - top);
        sum += e;
        pooled += e * v[s];
    }
    pooled /= sum;
    float total = hyBlockSum(pooled * pooled, part);
    out[c] = pooled * rsqrtf(total / 128.0f + eps) * norm[c];
    __syncthreads();
    float r = out[c];
    if(c >= 64){
        int i = (c - 64) >> 1;
        float theta = (float)(4 * g) * powf(base, -2.0f * i / 64.0f);
        float cs = cosf(theta), sn = sinf(theta);
        r = (c & 1) == 0 ? out[c] * cs - out[c + 1] * sn : out[c] * cs + out[c - 1] * sn;
    }
    pools[(u64)g * 128 + c] = r;
}

// puntaje de cada grupo anterior al propio: suma por cabeza del peso por relu de q por la clave del grupo
extern "C" __global__ void hyIdxScore(const float* q, const float* w, const float* pools, const int* positions, float* scores, int stride, int keep, float scale) {
    __shared__ float4 qs[16][32];
    __shared__ float ws[16];
    int t = blockIdx.y;
    int own = positions[t] >> 2;
    if(own <= keep) return;
    const float4* src = (const float4*)(q + (u64)t * 2048);
    for(int i = threadIdx.x; i < 512; i += blockDim.x) qs[i >> 5][i & 31] = src[i];
    if(threadIdx.x < 16) ws[threadIdx.x] = w[(u64)t * 16 + threadIdx.x];
    __syncthreads();
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if(g >= own) return;
    const float4* key = (const float4*)(pools + (u64)g * 128);
    float acc[16];
    #pragma unroll
    for(int h = 0; h < 16; h++) acc[h] = 0.0f;
    for(int d = 0; d < 32; d++){
        float4 kv = key[d];
        #pragma unroll
        for(int h = 0; h < 16; h++){
            float4 qv = qs[h][d];
            acc[h] += qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
        }
    }
    float score = 0.0f;
    #pragma unroll
    for(int h = 0; h < 16; h++) score += ws[h] * fmaxf(acc[h], 0.0f);
    scores[(u64)t * stride + g] = score * scale;
}

__device__ __forceinline__ unsigned int hyOrderKey(float v) {
    unsigned int u = __float_as_uint(v);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// los keep grupos de mas puntaje en orden de posicion, cada uno con sus 4 tokens, y al final el grupo propio hasta el token
// con pocos grupos antes van todas las posiciones, igual que la atencion completa
extern "C" __global__ void hySelect(const float* scores, int stride, const int* positions, int* list, int* counts, int width, int keep) {
    __shared__ unsigned int hist[256];
    __shared__ unsigned int chosen[2];
    __shared__ int warpCount[32];
    __shared__ int warpTie[32];
    __shared__ int baseCount, baseTie;
    int t = blockIdx.x;
    int pos = positions[t], own = pos >> 2;
    int* out = list + (u64)t * width;
    if(own <= keep){
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
        for(int g = threadIdx.x; g < own; g += blockDim.x){
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
    for(int start = 0; start < own; start += blockDim.x){
        int g = start + threadIdx.x;
        unsigned int key = g < own ? hyOrderKey(row[g]) : 0;
        bool over = g < own && key > prefix;
        bool tie = g < own && key == prefix;
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
        // los empates entran en orden hasta completar: el lugar de cada grupo es cuantos elegidos hay antes
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
    int tail = pos - 4 * own + 1;
    for(int j = threadIdx.x; j < tail; j += blockDim.x) out[4 * keep + j] = 4 * own + j;
    if(threadIdx.x == 0) counts[t] = 4 * keep + tail;
}

// cada fila del ayudante mira lo que eligio el modelo para el ultimo token aceptado y las filas suyas hasta ella
extern "C" __global__ void hyDraftList(const int* snap, const int* snapCount, int row, int anchor, int* list, int* counts, int width) {
    int k = blockIdx.x;
    int c = snapCount[row];
    const int* from = snap + (u64)row * 2048;
    int* out = list + (u64)k * width;
    for(int j = threadIdx.x; j < c; j += blockDim.x) out[j] = from[j];
    for(int i = threadIdx.x; i <= k; i += blockDim.x) out[c + i] = anchor + i;
    if(threadIdx.x == 0) counts[k] = c + k + 1;
}

// la misma atencion por tramos pero sobre la lista de posiciones elegidas de cada consulta
extern "C" __global__ void hyAttendList(const float* qfull, const u16* cache, const int* list, const int* counts, int width, float* scratch, int heads, float scale, int span, int splits) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int h = blockIdx.x * 8 + warp, t = blockIdx.y / splits, part = blockIdx.y % splits;
    if(h >= heads) return;
    float* dst = scratch + (((u64)t * heads + h) * splits + part) * 514;
    const int* picks = list + (u64)t * width;
    int first = part * span;
    int last = min(counts[t], first + span) - 1;
    const float* q = qfull + ((u64)t * heads + h) * 576;
    float qr[18];
    #pragma unroll
    for(int k = 0; k < 18; k++) qr[k] = q[lane + 32 * k];
    float acc[16];
    #pragma unroll
    for(int k = 0; k < 16; k++) acc[k] = 0.0f;
    float best = -1e30f, total = 0.0f;
    for(int j = first; j <= last; j++){
        const u16* row = cache + (u64)picks[j] * 576;
        float dot = 0.0f;
        #pragma unroll
        for(int k = 0; k < 18; k++) dot += qr[k] * hyHalf(row[lane + 32 * k]);
        for(int s = 16; s > 0; s >>= 1) dot += __shfl_xor_sync(0xffffffff, dot, s);
        dot *= scale;
        float top = fmaxf(best, dot);
        float keep = __expf(best - top);
        float p = __expf(dot - top);
        total = total * keep + p;
        #pragma unroll
        for(int k = 0; k < 16; k++) acc[k] = acc[k] * keep + p * hyHalf(row[lane + 32 * k]);
        best = top;
    }
    if(lane == 0){
        dst[0] = best;
        dst[1] = total;
    }
    #pragma unroll
    for(int k = 0; k < 16; k++) dst[2 + lane + 32 * k] = acc[k];
}

// enrutado con los 512 hilos buscando el mejor en cada vuelta
extern "C" __global__ void hyRouteFast(const float* logits, const float* bias, int* ids, float* weights, int experts, int used, float scale, int norm) {
    __shared__ float prob[1024];
    __shared__ float pick[1024];
    __shared__ float bestValue[32];
    __shared__ int bestAt[32];
    __shared__ int chosen[32];
    int t = blockIdx.x;
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    for(int e = threadIdx.x; e < experts; e += blockDim.x){
        float p = hySigmoid(logits[(u64)t * experts + e]);
        prob[e] = p;
        pick[e] = p + (bias ? bias[e] : 0.0f);
    }
    __syncthreads();
    for(int k = 0; k < used; k++){
        float v = -1e30f;
        int at = 0x7fffffff;
        for(int e = threadIdx.x; e < experts; e += blockDim.x){
            if(pick[e] > v || (pick[e] == v && e < at)){
                v = pick[e];
                at = e;
            }
        }
        for(int s = 16; s > 0; s >>= 1){
            float ov = __shfl_xor_sync(0xffffffff, v, s);
            int oa = __shfl_xor_sync(0xffffffff, at, s);
            if(ov > v || (ov == v && oa < at)){
                v = ov;
                at = oa;
            }
        }
        if(lane == 0){
            bestValue[warp] = v;
            bestAt[warp] = at;
        }
        __syncthreads();
        if(threadIdx.x == 0){
            float bv = bestValue[0];
            int ba = bestAt[0];
            for(int w = 1; w < warps; w++){
                if(bestValue[w] > bv || (bestValue[w] == bv && bestAt[w] < ba)){
                    bv = bestValue[w];
                    ba = bestAt[w];
                }
            }
            chosen[k] = ba;
            pick[ba] = -1e30f;
        }
        __syncthreads();
    }
    if(threadIdx.x != 0) return;
    float sum = 0.0f;
    for(int k = 0; k < used; k++) sum += prob[chosen[k]];
    float inv = norm && sum > 0.0f ? 1.0f / sum : 1.0f;
    for(int k = 0; k < used; k++){
        ids[t * used + k] = chosen[k];
        weights[t * used + k] = prob[chosen[k]] * inv * scale;
    }
}

// mezcla de las corrientes en dos pasos: muchos bloques suman los productos y un bloque por token arma la entrada
extern "C" __global__ void hyHcDots(const float* streams, const float* fn, float* mixes, int width, int rows, int chunk) {
    __shared__ float red[9][8];
    int t = blockIdx.x / ((width + chunk - 1) / chunk), part = blockIdx.x % ((width + chunk - 1) / chunk);
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    const float* flat = streams + (u64)t * width;
    int from = part * chunk, to = min(width, from + chunk);
    float acc[9];
    #pragma unroll
    for(int r = 0; r < 9; r++) acc[r] = 0.0f;
    for(int c = from + threadIdx.x; c < to; c += blockDim.x){
        float x = flat[c];
        acc[8] += x * x;
        #pragma unroll
        for(int r = 0; r < 8; r++) if(r < rows) acc[r] += fn[(u64)r * width + c] * x;
    }
    #pragma unroll
    for(int r = 0; r < 9; r++){
        float v = acc[r];
        for(int s = 16; s > 0; s >>= 1) v += __shfl_xor_sync(0xffffffff, v, s);
        if(lane == 0) red[r][warp] = v;
    }
    __syncthreads();
    if(threadIdx.x < 9 && (threadIdx.x < rows || threadIdx.x == 8)){
        float v = 0.0f;
        for(int w = 0; w < warps; w++) v += red[threadIdx.x][w];
        atomicAdd(mixes + (u64)t * 16 + threadIdx.x, v);
    }
}

extern "C" __global__ void hyHcCombine(const float* streams, float* mixes, const float* scale, const float* base, const float* norm, float* out, float* post, int hidden, int hc, float eps, float hcEps, float mag, i8* packed, float* scales, int* sums) {
    __shared__ float red[32];
    __shared__ float mix[16];
    __shared__ float pre[8];
    __shared__ float y[4096];
    int t = blockIdx.x;
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    int width = hc * hidden;
    const float* flat = streams + (u64)t * width;
    if(threadIdx.x < 16) mix[threadIdx.x] = mixes[(u64)t * 16 + threadIdx.x];
    __syncthreads();
    if(threadIdx.x < 16) mixes[(u64)t * 16 + threadIdx.x] = 0.0f;
    float inv = rsqrtf(mix[8] / width + eps);
    if(threadIdx.x < hc) pre[threadIdx.x] = hySigmoid(mix[threadIdx.x] * inv * scale[0] + base[threadIdx.x]) + hcEps;
    if(post && threadIdx.x >= hc && threadIdx.x < 2 * hc){
        int i = threadIdx.x - hc;
        post[(u64)t * hc + i] = hySigmoid(mix[hc + i] * inv * scale[1] + base[hc + i]) * mag + hcEps;
    }
    __syncthreads();
    float sq = 0.0f;
    for(int c = threadIdx.x; c < hidden; c += blockDim.x){
        float v = 0.0f;
        for(int i = 0; i < hc; i++) v += pre[i] * flat[(u64)i * hidden + c];
        y[c] = v;
        sq += v * v;
    }
    for(int s = 16; s > 0; s >>= 1) sq += __shfl_xor_sync(0xffffffff, sq, s);
    if(lane == 0) red[warp] = sq;
    __syncthreads();
    float total = 0.0f;
    for(int w = 0; w < warps; w++) total += red[w];
    float inv2 = rsqrtf(total / hidden + eps);
    for(int c = threadIdx.x; c < hidden; c += blockDim.x) out[(u64)t * hidden + c] = norm ? y[c] * inv2 * norm[c] : y[c];
    if(!packed) return;
    for(int g = warp; g < hidden / 32; g += warps){
        int c = g * 32 + lane;
        hyQuant(norm ? y[c] * inv2 * norm[c] : y[c], (u64)t * hidden + c, packed, scales, sums);
    }
}

// salida de la capa repartida por columnas: cada bloque saca la norma entera de la fila, que esta en la cache
extern "C" __global__ void hyHcSpread(float* streams, const float* x, const float* norm, const float* post, int hidden, int hc, float eps, int chunk) {
    __shared__ float part[32];
    int parts = (hidden + chunk - 1) / chunk;
    int t = blockIdx.x / parts, piece = blockIdx.x % parts;
    const float* src = x + (u64)t * hidden;
    float sq = 0.0f;
    for(int c = threadIdx.x; c < hidden; c += blockDim.x) sq += src[c] * src[c];
    float inv = rsqrtf(hyBlockSum(sq, part) / hidden + eps);
    float* s = streams + (u64)t * hc * hidden;
    const float* p = post + (u64)t * hc;
    int from = piece * chunk, to = min(hidden, from + chunk);
    for(int c = from + threadIdx.x; c < to; c += blockDim.x){
        float v = src[c] * inv * norm[c];
        for(int i = 0; i < hc; i++) s[(u64)i * hidden + c] += v * p[i];
    }
}

// elementales que dejan el resultado cuantizado: un warp por grupo de 32
extern "C" __global__ void hySwigluPack(const float* gate, const float* up, float* out, int count, float limit, i8* packed, float* scales, int* sums) {
    int group = blockIdx.x * blockDim.y + threadIdx.y;
    if(group * 32 >= count) return;
    int i = group * 32 + threadIdx.x;
    float g = gate[i], u = up[i];
    if(limit > 0.0f){
        g = fminf(g, limit);
        u = fminf(fmaxf(u, -limit), limit);
    }
    float v = g / (1.0f + __expf(-g)) * u;
    out[i] = v;
    hyQuant(v, i, packed, scales, sums);
}

extern "C" __global__ void hyGateMulPack(float* a, const float* g, int count, i8* packed, float* scales, int* sums) {
    int group = blockIdx.x * blockDim.y + threadIdx.y;
    if(group * 32 >= count) return;
    int i = group * 32 + threadIdx.x;
    float v = a[i] * hySigmoid(g[i]);
    a[i] = v;
    hyQuant(v, i, packed, scales, sums);
}

extern "C" __global__ void hyAddPack(float* a, const float* b, int count, i8* packed, float* scales, int* sums) {
    int group = blockIdx.x * blockDim.y + threadIdx.y;
    if(group * 32 >= count) return;
    int i = group * 32 + threadIdx.x;
    float v = a[i] + b[i];
    a[i] = v;
    hyQuant(v, i, packed, scales, sums);
}

extern "C" __global__ void hyGatherPack(const float* in, const int* pairs, float* out, int latent, int used, int count, i8* packed, float* scales, int* sums) {
    int group = blockIdx.x * blockDim.y + threadIdx.y;
    int perRow = latent / 32;
    if(group >= count * perRow) return;
    int j = group / perRow, c = (group % perRow) * 32 + threadIdx.x;
    float v = in[(u64)(pairs[j] / used) * latent + c];
    u64 i = (u64)j * latent + c;
    out[i] = v;
    hyQuant(v, i, packed, scales, sums);
}

extern "C" __global__ void hyFuseAdd(float* x, const float* fused, const float* gate, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < count) x[i] += gate[i] * fused[i];
}

extern "C" __global__ void hyArgmax(const float* v, int count, int* out) {
    __shared__ float bestValue[32];
    __shared__ int bestAt[32];
    float best = -1e30f;
    int at = 0;
    for(int i = threadIdx.x; i < count; i += blockDim.x){
        if(v[i] > best){
            best = v[i];
            at = i;
        }
    }
    for(int s = 16; s > 0; s >>= 1){
        float ov = __shfl_xor_sync(0xffffffff, best, s);
        int oa = __shfl_xor_sync(0xffffffff, at, s);
        if(ov > best || (ov == best && oa < at)){
            best = ov;
            at = oa;
        }
    }
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if(lane == 0){
        bestValue[warp] = best;
        bestAt[warp] = at;
    }
    __syncthreads();
    if(threadIdx.x != 0) return;
    for(int w = 1; w < (int)(blockDim.x >> 5); w++){
        if(bestValue[w] > best || (bestValue[w] == best && bestAt[w] < at)){
            best = bestValue[w];
            at = bestAt[w];
        }
    }
    out[0] = at;
}

// el mejor token y su probabilidad: el ayudante deja de adivinar cuando no esta seguro
extern "C" __global__ void hyArgmaxProb(const float* v, int count, int* out, float* prob) {
    __shared__ float bestValue[32];
    __shared__ int bestAt[32];
    __shared__ float sums[32];
    float best = -1e30f;
    int at = 0;
    for(int i = threadIdx.x; i < count; i += blockDim.x){
        if(v[i] > best){
            best = v[i];
            at = i;
        }
    }
    for(int s = 16; s > 0; s >>= 1){
        float ov = __shfl_xor_sync(0xffffffff, best, s);
        int oa = __shfl_xor_sync(0xffffffff, at, s);
        if(ov > best || (ov == best && oa < at)){
            best = ov;
            at = oa;
        }
    }
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    if(lane == 0){
        bestValue[warp] = best;
        bestAt[warp] = at;
    }
    __syncthreads();
    float top = bestValue[0];
    int topAt = bestAt[0];
    for(int w = 1; w < warps; w++){
        if(bestValue[w] > top || (bestValue[w] == top && bestAt[w] < topAt)){
            top = bestValue[w];
            topAt = bestAt[w];
        }
    }
    float sum = 0.0f;
    for(int i = threadIdx.x; i < count; i += blockDim.x) sum += __expf(v[i] - top);
    for(int s = 16; s > 0; s >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, s);
    if(lane == 0) sums[warp] = sum;
    __syncthreads();
    if(threadIdx.x != 0) return;
    float total = 0.0f;
    for(int w = 0; w < warps; w++) total += sums[w];
    out[0] = topAt;
    prob[0] = 1.0f / total;
}

extern "C" __global__ void hyFill(float* out, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < count) out[i] = 0.0f;
}

extern "C" __global__ void hyCopy(const float* a, float* b, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < count) b[i] = a[i];
}

// swiglu con tope: la compuerta solo arriba y la subida a los dos lados, sin tope si limit es cero
extern "C" __global__ void hySwiglu(const float* gate, const float* up, float* out, int count, float limit) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= count) return;
    float g = gate[i], u = up[i];
    if(limit > 0.0f){
        g = fminf(g, limit);
        u = fminf(fmaxf(u, -limit), limit);
    }
    out[i] = g / (1.0f + __expf(-g)) * u;
}

// enrutado con sigmoide: el sesgo solo elige, los pesos salen de la probabilidad sin sesgo y se normalizan
extern "C" __global__ void hyRoute(const float* logits, const float* bias, int* ids, float* weights, int experts, int used, float scale, int norm) {
    __shared__ float prob[1024];
    __shared__ float pick[1024];
    int t = blockIdx.x;
    for(int e = threadIdx.x; e < experts; e += blockDim.x){
        float p = hySigmoid(logits[(u64)t * experts + e]);
        prob[e] = p;
        pick[e] = p + (bias ? bias[e] : 0.0f);
    }
    __syncthreads();
    if(threadIdx.x != 0) return;
    float sum = 0.0f;
    int chosen[32];
    for(int k = 0; k < used; k++){
        int bestAt = 0;
        float bestValue = -1e30f;
        for(int e = 0; e < experts; e++){
            if(pick[e] > bestValue){
                bestValue = pick[e];
                bestAt = e;
            }
        }
        pick[bestAt] = -1e30f;
        chosen[k] = bestAt;
        sum += prob[bestAt];
    }
    float inv = norm && sum > 0.0f ? 1.0f / sum : 1.0f;
    for(int k = 0; k < used; k++){
        ids[t * used + k] = chosen[k];
        weights[t * used + k] = prob[chosen[k]] * inv * scale;
    }
}

// engram: compuerta por corriente con la clave de la tabla y suma del valor normado a cada corriente
extern "C" __global__ void hyEngramMix(float* streams, const float* kv, const float* bias, const float* gate, const float* gamma, const float* hcScale, int hidden, int hc, float eps) {
    __shared__ float part[32];
    __shared__ float gates[8];
    int t = blockIdx.x;
    float* s = streams + (u64)t * hc * hidden;
    const float* row = kv + (u64)t * (hc + 1) * hidden;
    float invValue;
    {
        float sum = 0.0f;
        for(int c = threadIdx.x; c < hidden; c += blockDim.x){
            float v = row[(u64)hc * hidden + c] + bias[(u64)hc * hidden + c];
            sum += v * v;
        }
        invValue = rsqrtf(hyBlockSum(sum, part) / hidden + eps);
    }
    for(int i = 0; i < hc; i++){
        float xx = 0.0f, kk = 0.0f, xk = 0.0f;
        for(int c = threadIdx.x; c < hidden; c += blockDim.x){
            float x = s[(u64)i * hidden + c];
            float k = row[(u64)i * hidden + c] + bias[(u64)i * hidden + c];
            xx += x * x;
            kk += k * k;
            xk += x * k * gate[(u64)i * hidden + c];
        }
        float sx = hyBlockSum(xx, part);
        float sk = hyBlockSum(kk, part);
        float sxk = hyBlockSum(xk, part);
        if(threadIdx.x == 0){
            float dot = sxk * rsqrtf(sx / hidden + eps) * rsqrtf(sk / hidden + eps) / sqrtf((float)hidden);
            float mag = sqrtf(fmaxf(fabsf(dot), 1e-6f));
            gates[i] = hySigmoid(dot < 0.0f ? -mag : mag);
        }
    }
    __syncthreads();
    for(int c = threadIdx.x; c < hidden; c += blockDim.x){
        float signal = (row[(u64)hc * hidden + c] + bias[(u64)hc * hidden + c]) * invValue * gamma[c];
        for(int i = 0; i < hc; i++) s[(u64)i * hidden + c] += signal * (gates[i] + hcScale[(u64)i * hidden + c]);
    }
}
)";
