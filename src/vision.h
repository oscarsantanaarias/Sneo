#pragma once
#include "gguf.h"
#include "cuda.h"
#include "picture.h"
#include <cmath>
#include <cstring>

inline const char* visionKernelSource = R"(
typedef unsigned long long u64;
typedef unsigned short u16;

__device__ __forceinline__ float wideBf16(u16 v) { return __uint_as_float((unsigned int)v << 16); }

__device__ __forceinline__ float blockTotal(float v, float* scratch) {
    for(int step = 16; step > 0; step >>= 1) v += __shfl_xor_sync(0xffffffff, v, step);
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    __syncthreads();
    if(lane == 0) scratch[warp] = v;
    __syncthreads();
    int warps = (blockDim.x + 31) >> 5;
    v = lane < warps ? scratch[lane] : 0.0f;
    for(int step = 16; step > 0; step >>= 1) v += __shfl_xor_sync(0xffffffff, v, step);
    return v;
}

__device__ __forceinline__ float blockPeak(float v, float* scratch) {
    for(int step = 16; step > 0; step >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, step));
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    __syncthreads();
    if(lane == 0) scratch[warp] = v;
    __syncthreads();
    int warps = (blockDim.x + 31) >> 5;
    v = lane < warps ? scratch[lane] : -3.0e38f;
    for(int step = 16; step > 0; step >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, step));
    return v;
}

extern "C" __global__ void __launch_bounds__(256) seeMultiply(const float* a, const void* b, float* c, const float* bias, int m, int n, int k, int lda, int ldb, int ldc, u64 aStep, u64 bStep, u64 cStep, int half, int across, float scale, int add) {
    __shared__ float as[16][68];
    __shared__ float bs[16][68];
    int z = blockIdx.z;
    a += aStep * z;
    c += cStep * z;
    int row0 = blockIdx.y * 64, col0 = blockIdx.x * 64;
    int tid = threadIdx.x, tr = tid >> 4, tc = tid & 15;
    float acc[4][4];
    for(int i = 0; i < 4; i++) for(int j = 0; j < 4; j++) acc[i][j] = 0.0f;
    for(int k0 = 0; k0 < k; k0 += 16){
        for(int i = tid; i < 1024; i += 256){
            int r = i >> 4, kk = i & 15;
            int gr = row0 + r, gk = k0 + kk;
            as[kk][r] = gr < m && gk < k ? a[(u64)gr * lda + gk] : 0.0f;
        }
        for(int i = tid; i < 1024; i += 256){
            int cc = across ? i & 63 : i >> 4;
            int kk = across ? i >> 6 : i & 15;
            int gc = col0 + cc, gk = k0 + kk;
            float v = 0.0f;
            if(gc < n && gk < k){
                u64 at = bStep * z + (across ? (u64)gk * ldb + gc : (u64)gc * ldb + gk);
                v = half ? wideBf16(((const u16*)b)[at]) : ((const float*)b)[at];
            }
            bs[kk][cc] = v;
        }
        __syncthreads();
        for(int kk = 0; kk < 16; kk++){
            float av[4], bv[4];
            for(int i = 0; i < 4; i++){
                av[i] = as[kk][tr * 4 + i];
                bv[i] = bs[kk][tc * 4 + i];
            }
            for(int i = 0; i < 4; i++) for(int j = 0; j < 4; j++) acc[i][j] += av[i] * bv[j];
        }
        __syncthreads();
    }
    for(int i = 0; i < 4; i++){
        int gr = row0 + tr * 4 + i;
        if(gr >= m) continue;
        for(int j = 0; j < 4; j++){
            int gc = col0 + tc * 4 + j;
            if(gc >= n) continue;
            float v = acc[i][j] * scale + (bias ? bias[gc] : 0.0f);
            u64 at = (u64)gr * ldc + gc;
            c[at] = add ? c[at] + v : v;
        }
    }
}

extern "C" __global__ void seeNorm(const float* x, float* y, const float* weight, const float* bias, int width, float eps) {
    __shared__ float scratch[32];
    const float* row = x + (u64)blockIdx.x * width;
    float* out = y + (u64)blockIdx.x * width;
    float sum = 0.0f;
    for(int i = threadIdx.x; i < width; i += blockDim.x) sum += row[i];
    float mean = blockTotal(sum, scratch) / width;
    float spread = 0.0f;
    for(int i = threadIdx.x; i < width; i += blockDim.x){
        float d = row[i] - mean;
        spread += d * d;
    }
    float inv = rsqrtf(blockTotal(spread, scratch) / width + eps);
    for(int i = threadIdx.x; i < width; i += blockDim.x) out[i] = (row[i] - mean) * inv * weight[i] + bias[i];
}

extern "C" __global__ void seeSoftmax(float* s, int width) {
    __shared__ float scratch[32];
    float* row = s + (u64)blockIdx.x * width;
    float peak = -3.0e38f;
    for(int i = threadIdx.x; i < width; i += blockDim.x) peak = fmaxf(peak, row[i]);
    peak = blockPeak(peak, scratch);
    float sum = 0.0f;
    for(int i = threadIdx.x; i < width; i += blockDim.x){
        float e = __expf(row[i] - peak);
        row[i] = e;
        sum += e;
    }
    float inv = 1.0f / blockTotal(sum, scratch);
    for(int i = threadIdx.x; i < width; i += blockDim.x) row[i] *= inv;
}

extern "C" __global__ void seeGelu(float* x, u64 count, int exact) {
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= count) return;
    float v = x[i];
    x[i] = exact
        ? 0.5f * v * (1.0f + erff(v * 0.70710678118654752f))
        : 0.5f * v * (1.0f + tanhf(0.79788456080286536f * (v + 0.044715f * v * v * v)));
}
)";

inline const char* visionExtraSource = R"(
extern "C" __global__ void seeRope(float* qkv, const float* ropeCos, const float* ropeSin, int width, int heads, int headWidth) {
    int t = blockIdx.x, part = blockIdx.y, half = headWidth / 2;
    float* row = qkv + (u64)t * 3 * width + (u64)part * width;
    for(int j = threadIdx.x; j < heads * half; j += blockDim.x){
        int h = j / half, d = j % half;
        float c = ropeCos[(u64)t * half + d], s = ropeSin[(u64)t * half + d];
        float a = row[h * headWidth + d], b = row[h * headWidth + d + half];
        row[h * headWidth + d] = a * c - b * s;
        row[h * headWidth + d + half] = b * c + a * s;
    }
}
)";

struct VisionEncoder {
    void* context = nullptr;
    int device = -1;
    bool own = false;
    KernelModule module;
    std::vector<uint64_t> allocations;
    int width = 1152, heads = 16, depth = 27, inner = 4304, patch = 16, merge = 2, outWidth = 2560, side = 48, most = 2304;
    float eps = 1e-6f, mean[3] = {0.5f, 0.5f, 0.5f}, spread[3] = {0.5f, 0.5f, 0.5f};
    int imageSize = 768;
    int leastTokens = 8, mostTokens = 1024;
    struct Block {
        uint64_t norm1w = 0, norm1b = 0, norm2w = 0, norm2b = 0, qkvw = 0, qkvb = 0, outw = 0, outb = 0, upw = 0, upb = 0, downw = 0, downb = 0;
        int qkvHalf = 1, outHalf = 1, upHalf = 1, downHalf = 1;
    };
    std::vector<Block> blocks;
    uint64_t patchw = 0, patchb = 0, postw = 0, postb = 0, mix0w = 0, mix0b = 0, mix2w = 0, mix2b = 0;
    int mix0Half = 1, mix2Half = 1;
    std::vector<float> position;
    uint64_t pixels = 0, x = 0, h = 0, qkv = 0, mid = 0, scores = 0, joined = 0, outRows = 0, ropeCos = 0, ropeSin = 0;
    int group = 4;

    void close() {
        CudaLink& cu = cudaLink();
        if(context && cu.ready && cu.setCurrent(context) == 0){
            cu.sync();
            for(uint64_t p : allocations) cu.release(p);
            if(own && cu.drop) cu.drop(device);
        }
        if(cu.ready && cu.context) cu.setCurrent(cu.context);
        allocations.clear();
        blocks.clear();
        position.clear();
        module = KernelModule();
        context = nullptr;
        device = -1;
        own = false;
    }

    uint64_t take(uint64_t bytes) {
        uint64_t ptr = 0;
        if(cudaLink().alloc(&ptr, (size_t)std::max<uint64_t>(bytes, 256)) != 0) return 0;
        allocations.push_back(ptr);
        return ptr;
    }

    bool pull(const ModelFile& file, const std::string& name, std::vector<uint8_t>& bytes, const TensorInfo** info = nullptr) {
        const TensorInfo* t = findTensor(file, name);
        if(!t) return false;
        if(info) *info = t;
        FILE* f = nullptr;
        fopen_s(&f, file.paths[t->file].c_str(), "rb");
        if(!f) return false;
        bytes.resize((size_t)t->bytes);
        bool ok = _fseeki64(f, (long long)t->offset, SEEK_SET) == 0 && fread(bytes.data(), 1, bytes.size(), f) == bytes.size();
        fclose(f);
        return ok;
    }

    uint64_t upload(const ModelFile& file, const std::string& name, int* half = nullptr) {
        std::vector<uint8_t> bytes;
        const TensorInfo* t = nullptr;
        if(!pull(file, name, bytes, &t)) return 0;
        if(t->type == 1){
            std::vector<float> wide((size_t)t->elements);
            const uint16_t* from = (const uint16_t*)bytes.data();
            for(size_t i = 0; i < wide.size(); i++){
                uint16_t v = from[i];
                uint32_t sign = (uint32_t)(v & 0x8000) << 16, power = v >> 10 & 31, fraction = v & 1023;
                uint32_t bits = power == 0
                    ? (fraction == 0 ? sign : 0)
                    : power == 31
                        ? sign | 0x7F800000 | fraction << 13
                        : sign | (power + 112) << 23 | fraction << 13;
                float value = 0.0f;
                memcpy(&value, &bits, 4);
                if(power == 0 && fraction != 0) value = (sign ? -1.0f : 1.0f) * ldexpf((float)fraction, -24);
                wide[i] = value;
            }
            bytes.assign((const uint8_t*)wide.data(), (const uint8_t*)wide.data() + wide.size() * 4);
        } else if(t->type != 0 && t->type != 30) return 0;
        if(half) *half = t->type == 30 ? 1 : 0;
        uint64_t ptr = take(bytes.size());
        if(!ptr || cudaLink().copyUp(ptr, bytes.data(), bytes.size()) != 0) return 0;
        return ptr;
    }

    bool open(const std::string& path, int dev, std::string& error) {
        close();
        CudaLink& cu = cudaLink();
        if(!useCuda()){
            error = "Vision needs a cuda card";
            return false;
        }
        int major = cu.major, minor = cu.minor;
        device = dev;
        context = cu.context;
        if(dev != cu.device){
            if(!cu.retain || cu.retain(&context, dev) != 0){
                context = nullptr;
                error = "the second card could not be opened for Vision";
                return false;
            }
            own = true;
            cu.attribute(&major, 75, dev);
            cu.attribute(&minor, 76, dev);
        }
        if(!buildModule(std::string(visionKernelSource) + visionExtraSource, module, context, major, minor)){
            error = "the picture reader kernels did not build: " + module.log;
            close();
            return false;
        }
        ModelFile file;
        if(!openModel(path, file, error)){
            close();
            return false;
        }
        auto whole = [&](const char* key, int fallback) { return (int)metaWhole(file, key, (uint32_t)fallback); };
        auto kind = file.meta.find("clip.projector_type");
        if(kind == file.meta.end() || kind->second.text != "qwen3vl_merger"){
            error = "the picture reader file is not one this engine can run";
            close();
            return false;
        }
        width = whole("clip.vision.embedding_length", 1152);
        heads = whole("clip.vision.attention.head_count", 16);
        depth = whole("clip.vision.block_count", 27);
        inner = whole("clip.vision.feed_forward_length", 4304);
        patch = whole("clip.vision.patch_size", 16);
        merge = whole("clip.vision.spatial_merge_size", 2);
        outWidth = whole("clip.vision.projection_dim", 2560);
        imageSize = whole("clip.vision.image_size", 768);
        auto epsilon = file.meta.find("clip.vision.attention.layer_norm_epsilon");
        if(epsilon != file.meta.end()) eps = (float)epsilon->second.real;
        auto means = file.meta.find("clip.vision.image_mean");
        auto spreads = file.meta.find("clip.vision.image_std");
        for(int c = 0; c < 3; c++){
            if(means != file.meta.end() && means->second.reals.size() == 3) mean[c] = (float)means->second.reals[c];
            if(spreads != file.meta.end() && spreads->second.reals.size() == 3) spread[c] = (float)spreads->second.reals[c];
        }

        std::vector<uint8_t> first, second;
        const TensorInfo* shape = nullptr;
        if(!pull(file, "v.patch_embd.weight", first, &shape) || shape->type != 0 || !pull(file, "v.patch_embd.weight.1", second) || second.size() != first.size()){
            error = "the picture reader file has no patch weights";
            close();
            return false;
        }
        std::vector<float> joinedPatch(first.size() / 4);
        for(size_t i = 0; i < joinedPatch.size(); i++) joinedPatch[i] = ((const float*)first.data())[i] + ((const float*)second.data())[i];
        patchw = take(joinedPatch.size() * 4);
        bool ok = patchw && cu.copyUp(patchw, joinedPatch.data(), joinedPatch.size() * 4) == 0;
        patchb = upload(file, "v.patch_embd.bias");
        std::vector<uint8_t> table;
        const TensorInfo* placed = nullptr;
        ok = ok && patchb && pull(file, "v.position_embd.weight", table, &placed) && placed->type == 0;
        if(ok){
            position.assign((const float*)table.data(), (const float*)table.data() + table.size() / 4);
            side = (int)std::lround(std::sqrt((double)(position.size() / width)));
            ok = side * side * width == (int)position.size();
        }
        for(int b = 0; b < depth && ok; b++){
            std::string at = "v.blk." + std::to_string(b) + ".";
            Block k;
            k.norm1w = upload(file, at + "ln1.weight");
            k.norm1b = upload(file, at + "ln1.bias");
            k.norm2w = upload(file, at + "ln2.weight");
            k.norm2b = upload(file, at + "ln2.bias");
            k.qkvw = upload(file, at + "attn_qkv.weight", &k.qkvHalf);
            k.qkvb = upload(file, at + "attn_qkv.bias");
            k.outw = upload(file, at + "attn_out.weight", &k.outHalf);
            k.outb = upload(file, at + "attn_out.bias");
            k.upw = upload(file, at + "ffn_up.weight", &k.upHalf);
            k.upb = upload(file, at + "ffn_up.bias");
            k.downw = upload(file, at + "ffn_down.weight", &k.downHalf);
            k.downb = upload(file, at + "ffn_down.bias");
            ok = k.norm1w && k.norm1b && k.norm2w && k.norm2b && k.qkvw && k.qkvb && k.outw && k.outb && k.upw && k.upb && k.downw && k.downb;
            blocks.push_back(k);
        }
        postw = ok ? upload(file, "v.post_ln.weight") : 0;
        postb = ok ? upload(file, "v.post_ln.bias") : 0;
        mix0w = ok ? upload(file, "mm.0.weight", &mix0Half) : 0;
        mix0b = ok ? upload(file, "mm.0.bias") : 0;
        mix2w = ok ? upload(file, "mm.2.weight", &mix2Half) : 0;
        mix2b = ok ? upload(file, "mm.2.bias") : 0;
        ok = ok && postw && postb && mix0w && mix0b && mix2w && mix2b;
        if(!ok){
            error = "the picture reader weights could not be loaded on the card";
            close();
            return false;
        }

        most = mostTokens * merge * merge;
        int half = width / heads / 2;
        pixels = take((uint64_t)most * 3 * patch * patch * 4);
        x = take((uint64_t)most * width * 4);
        h = take((uint64_t)most * width * 4);
        qkv = take((uint64_t)most * 3 * width * 4);
        mid = take((uint64_t)most * inner * 4);
        joined = take((uint64_t)most * width * 4);
        outRows = take((uint64_t)most / (merge * merge) * outWidth * 4);
        ropeCos = take((uint64_t)most * half * 4);
        ropeSin = take((uint64_t)most * half * 4);
        group = heads;
        while(group > 1 && (uint64_t)group * most * most * 4 > (128ULL << 20)) group /= 2;
        scores = take((uint64_t)group * most * most * 4);
        if(!pixels || !x || !h || !qkv || !mid || !joined || !outRows || !ropeCos || !ropeSin || !scores){
            error = "no room on the card for the picture reader";
            close();
            return false;
        }
        cu.setCurrent(cu.context);
        return true;
    }

    template<typename... T>
    bool run(const char* name, unsigned int gx, unsigned int gy, unsigned int gz, unsigned int threads, T... values) {
        void* args[] = {(void*)&values...};
        void* fn = kernelFunction(module, name);
        return fn && cudaLink().launch(fn, gx, gy, gz, threads, 1, 1, 0, nullptr, args, nullptr) == 0;
    }

    bool multiply(uint64_t a, uint64_t b, uint64_t c, uint64_t bias, int m, int n, int k, int lda, int ldb, int ldc, int half, int add, float scale = 1.0f, int across = 0, int count = 1, uint64_t aStep = 0, uint64_t bStep = 0, uint64_t cStep = 0) {
        return run("seeMultiply", (unsigned int)((n + 63) / 64), (unsigned int)((m + 63) / 64), (unsigned int)count, 256, a, b, c, bias, m, n, k, lda, ldb, ldc, aStep, bStep, cStep, half, across, scale, add);
    }

    void fitSize(int w, int h, int& tw, int& th) {
        int factor = patch * merge;
        double top = (double)mostTokens * factor * factor, least = (double)leastTokens * factor * factor;
        th = std::max(factor, (int)std::nearbyint((double)h / factor) * factor);
        tw = std::max(factor, (int)std::nearbyint((double)w / factor) * factor);
        if((double)th * tw > top){
            double beta = std::sqrt((double)h * w / top);
            th = std::max(factor, (int)std::floor(h / beta / factor) * factor);
            tw = std::max(factor, (int)std::floor(w / beta / factor) * factor);
            return;
        }
        if((double)th * tw >= least) return;
        double beta = std::sqrt(least / ((double)h * w));
        th = (int)std::ceil(h * beta / factor) * factor;
        tw = (int)std::ceil(w * beta / factor) * factor;
    }

    static double cubic(double x) {
        double a = -0.5;
        x = std::fabs(x);
        return x < 1.0
            ? ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
            : x < 2.0
                ? (((x - 5.0) * x + 8.0) * x - 4.0) * a
                : 0.0;
    }

    static void resample(const std::vector<uint8_t>& in, int inW, int inH, std::vector<uint8_t>& out, int outW, int outH, bool across) {
        int from = across ? inW : inH, to = across ? outW : outH;
        double scale = (double)from / to, stretch = std::max(scale, 1.0), support = 2.0 * stretch;
        int reach = (int)std::ceil(support) * 2 + 1;
        std::vector<int> first(to), count(to), fixed((size_t)to * reach);
        std::vector<double> weight(reach);
        for(int o = 0; o < to; o++){
            double center = (o + 0.5) * scale, total = 0.0;
            int lo = std::max(0, (int)(center - support + 0.5)), hi = std::min(from, (int)(center + support + 0.5));
            for(int k = 0; k < hi - lo; k++){
                weight[k] = cubic((k + lo - center + 0.5) / stretch);
                total += weight[k];
            }
            for(int k = 0; k < hi - lo; k++){
                double v = total != 0.0 ? weight[k] / total : weight[k];
                fixed[(size_t)o * reach + k] = (int)(v * (1 << 22) + (v < 0 ? -0.5 : 0.5));
            }
            first[o] = lo;
            count[o] = hi - lo;
        }
        out.assign((size_t)outW * outH * 3, 0);
        for(int y = 0; y < outH; y++){
            for(int x = 0; x < outW; x++){
                int o = across ? x : y;
                for(int c = 0; c < 3; c++){
                    int sum = 1 << 21;
                    for(int k = 0; k < count[o]; k++){
                        int sx = across ? first[o] + k : x, sy = across ? y : first[o] + k;
                        sum += in[((size_t)sy * inW + sx) * 3 + c] * fixed[(size_t)o * reach + k];
                    }
                    int v = sum >> 22;
                    out[((size_t)y * outW + x) * 3 + c] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
                }
            }
        }
    }

    bool see(const Picture& picture, std::vector<float>& rows, int& high, int& wide, std::string& error) {
        CudaLink& cu = cudaLink();
        if(!context){
            error = "the picture reader is not loaded";
            return false;
        }
        int tw = 0, th = 0;
        fitSize(picture.width, picture.height, tw, th);
        int gh = th / patch, gw = tw / patch, n = gh * gw, half = width / heads / 2, quarter = half / 2;
        if(n > most || gh % merge || gw % merge){
            error = "the picture is too large for the picture reader";
            return false;
        }
        std::vector<uint8_t> wider, sized;
        const std::vector<uint8_t>* level = &picture.rgb;
        if(tw != picture.width){
            resample(*level, picture.width, picture.height, wider, tw, picture.height, true);
            level = &wider;
        }
        if(th != picture.height){
            resample(*level, tw, picture.height, sized, tw, th, false);
            level = &sized;
        }
        std::vector<float> image((size_t)3 * th * tw);
        for(int y = 0; y < th; y++){
            for(int xx = 0; xx < tw; xx++){
                for(int c = 0; c < 3; c++) image[((size_t)c * th + y) * tw + xx] = ((*level)[((size_t)y * tw + xx) * 3 + c] / 255.0f - mean[c]) / spread[c];
            }
        }
        int span = 3 * patch * patch;
        std::vector<float> patches((size_t)n * span), start((size_t)n * width), cosines((size_t)n * half), sines((size_t)n * half);
        int at = 0;
        for(int by = 0; by < gh / merge; by++){
            for(int bx = 0; bx < gw / merge; bx++){
                for(int dy = 0; dy < merge; dy++){
                    for(int dx = 0; dx < merge; dx++, at++){
                        int py = by * merge + dy, px = bx * merge + dx;
                        float* p = patches.data() + (size_t)at * span;
                        for(int c = 0; c < 3; c++) for(int ky = 0; ky < patch; ky++) for(int kx = 0; kx < patch; kx++) p[(c * patch + ky) * patch + kx] = image[((size_t)c * th + py * patch + ky) * tw + px * patch + kx];
                        double fy = gh > 1 ? (double)py * (side - 1) / (gh - 1) : 0.0, fx = gw > 1 ? (double)px * (side - 1) / (gw - 1) : 0.0;
                        int y0 = (int)fy, x0 = (int)fx, y1 = std::min(y0 + 1, side - 1), x1 = std::min(x0 + 1, side - 1);
                        double wy = fy - y0, wx = fx - x0;
                        float* s = start.data() + (size_t)at * width;
                        for(int d = 0; d < width; d++){
                            auto cell = [&](int yy, int xx) { return (double)position[((size_t)yy * side + xx) * width + d]; };
                            s[d] = (float)((cell(y0, x0) * (1 - wx) + cell(y0, x1) * wx) * (1 - wy) + (cell(y1, x0) * (1 - wx) + cell(y1, x1) * wx) * wy);
                        }
                        for(int d = 0; d < half; d++){
                            double angle = (d < quarter ? py : px) * std::pow(10000.0, -2.0 * (d % quarter) / half);
                            cosines[(size_t)at * half + d] = (float)std::cos(angle);
                            sines[(size_t)at * half + d] = (float)std::sin(angle);
                        }
                    }
                }
            }
        }

        if(cu.setCurrent(context) != 0){
            error = "the picture reader card is not available";
            return false;
        }
        int headWidth = width / heads, kept = n / (merge * merge), joinedWidth = width * merge * merge;
        bool ok = cu.copyUp(pixels, patches.data(), patches.size() * 4) == 0
            && cu.copyUp(x, start.data(), start.size() * 4) == 0
            && cu.copyUp(ropeCos, cosines.data(), cosines.size() * 4) == 0
            && cu.copyUp(ropeSin, sines.data(), sines.size() * 4) == 0
            && multiply(pixels, patchw, x, patchb, n, width, span, span, span, width, 0, 1);
        uint64_t scoreStep = (uint64_t)n * n;
        for(const Block& k : blocks){
            ok = ok && run("seeNorm", (unsigned int)n, 1, 1, 256, x, h, k.norm1w, k.norm1b, width, eps)
                && multiply(h, k.qkvw, qkv, k.qkvb, n, 3 * width, width, width, width, 3 * width, k.qkvHalf, 0)
                && run("seeRope", (unsigned int)n, 2, 1, 256, qkv, ropeCos, ropeSin, width, heads, headWidth);
            for(int g = 0; g < heads && ok; g += group){
                int count = std::min(group, heads - g);
                uint64_t q = qkv + 4ULL * g * headWidth, key = qkv + 4ULL * (width + g * headWidth), value = qkv + 4ULL * (2 * width + g * headWidth);
                ok = multiply(q, key, scores, 0, n, n, headWidth, 3 * width, 3 * width, n, 0, 0, 1.0f / std::sqrt((float)headWidth), 0, count, headWidth, headWidth, scoreStep)
                    && run("seeSoftmax", (unsigned int)(count * n), 1, 1, 256, scores, n)
                    && multiply(scores, value, h + 4ULL * g * headWidth, 0, n, headWidth, n, n, 3 * width, width, 0, 0, 1.0f, 1, count, scoreStep, headWidth, headWidth);
            }
            ok = ok && multiply(h, k.outw, x, k.outb, n, width, width, width, width, width, k.outHalf, 1)
                && run("seeNorm", (unsigned int)n, 1, 1, 256, x, h, k.norm2w, k.norm2b, width, eps)
                && multiply(h, k.upw, mid, k.upb, n, inner, width, width, width, inner, k.upHalf, 0)
                && run("seeGelu", (unsigned int)(((uint64_t)n * inner + 255) / 256), 1, 1, 256, mid, (uint64_t)n * inner, 0)
                && multiply(mid, k.downw, x, k.downb, n, width, inner, inner, inner, width, k.downHalf, 1);
        }
        ok = ok && run("seeNorm", (unsigned int)n, 1, 1, 256, x, joined, postw, postb, width, eps)
            && multiply(joined, mix0w, x, mix0b, kept, joinedWidth, joinedWidth, joinedWidth, joinedWidth, joinedWidth, mix0Half, 0)
            && run("seeGelu", (unsigned int)(((uint64_t)kept * joinedWidth + 255) / 256), 1, 1, 256, x, (uint64_t)kept * joinedWidth, 1)
            && multiply(x, mix2w, outRows, mix2b, kept, outWidth, joinedWidth, joinedWidth, joinedWidth, outWidth, mix2Half, 0);
        rows.assign((size_t)kept * outWidth, 0.0f);
        ok = ok && cu.sync() == 0 && cu.copyDown(rows.data(), outRows, rows.size() * 4) == 0;
        cu.setCurrent(cu.context);
        if(!ok){
            error = "the picture reader failed on the card";
            return false;
        }
        high = gh / merge;
        wide = gw / merge;
        return true;
    }
};
