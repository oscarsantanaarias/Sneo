#pragma once
#include "gguf.h"
#include "quant_tables.h"
#include <immintrin.h>
#include <cmath>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <chrono>

inline uint32_t blockElements(uint32_t type) {
    return type == 42 ? 64 : typeInfo(type).blockElements;
}

inline uint64_t rowBytes(uint32_t type, uint64_t elements) {
    if(type == 42) return elements / 64 * 18;
    TypeInfo info = typeInfo(type);
    if(!info.blockElements) return 0;
    return elements / info.blockElements * info.blockBytes;
}

inline bool knownType(uint32_t type) {
    switch(type){
        case 0: case 1: case 12: case 13: case 14: case 16: case 17: case 18:
        case 7: case 8: case 20: case 21: case 22: case 23: case 29: case 30: case 39: case 40: case 42:
            return true;
    }
    return false;
}

inline float half(const uint8_t* p) {
    uint16_t h;
    memcpy(&h, p, 2);
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h)));
}

// los valores fp4 (e2m1) van por 2 para que sean enteros: 0 0.5 1 1.5 2 3 4 6 pasan a 0 1 2 3 4 6 8 12
alignas(16) inline const int8_t fp4Values[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

// mxfp4: la escala es un exponente de 8 bits y ya va por la mitad para compensar los valores por 2
inline float mxScale(uint8_t e) {
    uint32_t bits = e < 2
        ? 0x00200000u << e
        : (uint32_t)(e - 1) << 23;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

// nvfp4: la escala de cada 16 valores es fp8 sin signo (4 de exponente y 3 de mantisa) y tambien va por la mitad
inline float nvScale(uint8_t x) {
    if(x == 0 || x == 0x7F) return 0.0f;
    int e = (x >> 3) & 15, m = x & 7;
    return e == 0
        ? (float)m * 0.0009765625f
        : (float)(8 + m) * ldexpf(1.0f, e - 11);
}

// n bytes de codigos fp4: los nibbles bajos son los primeros n valores y los altos los n siguientes
inline void fp4Codes(const uint8_t* qs, int n, float scale, float* y) {
    for(int j = 0; j < n; j++){
        y[j] = (float)fp4Values[qs[j] & 15] * scale;
        y[j + n] = (float)fp4Values[qs[j] >> 4] * scale;
    }
}

// iq1_m guarda la escala del bloque en los 4 bits altos de sus cuatro escalas y va entre 8 porque grid1s va por 8
inline float grid1Scale(const uint8_t* block) {
    uint16_t sc[4];
    memcpy(sc, block + 48, 8);
    uint16_t bits = (uint16_t)((sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000));
    return half((const uint8_t*)&bits) * 0.125f;
}

inline __m256 signedGrid(uint64_t magnitudes, uint32_t signs, float scale) {
    const __m256i bits = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
    __m256 v = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_cvtsi64_si128((long long)magnitudes)));
    __m256i flip = _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_set1_epi32((int)signs), bits), bits);
    v = _mm256_xor_ps(v, _mm256_castsi256_ps(_mm256_slli_epi32(flip, 31)));
    return _mm256_mul_ps(v, _mm256_set1_ps(scale));
}

inline void nibbleBlock(const uint8_t* qs, float scale, float* y) {
    const __m128i table = _mm_loadu_si128((const __m128i*)nonLinear4);
    const __m128i mask = _mm_set1_epi8(15);
    __m128i bytes = _mm_loadu_si128((const __m128i*)qs);
    __m128i low = _mm_shuffle_epi8(table, _mm_and_si128(bytes, mask));
    __m128i high = _mm_shuffle_epi8(table, _mm_and_si128(_mm_srli_epi16(bytes, 4), mask));
    __m256 s = _mm256_set1_ps(scale);
    _mm256_storeu_ps(y, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(low)), s));
    _mm256_storeu_ps(y + 8, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(low, 8))), s));
    _mm256_storeu_ps(y + 16, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(high)), s));
    _mm256_storeu_ps(y + 24, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(high, 8))), s));
}

inline void dequantRow(uint32_t type, const uint8_t* src, float* out, int64_t count) {
    if(type == 0){
        memcpy(out, src, (size_t)count * 4);
        return;
    }

    if(type == 1 || type == 30){
        int64_t i = 0;
        for(; i + 8 <= count; i += 8){
            __m128i raw = _mm_loadu_si128((const __m128i*)(src + 2 * i));
            __m256 v = type == 1
                ? _mm256_cvtph_ps(raw)
                : _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(raw), 16));
            _mm256_storeu_ps(out + i, v);
        }
        for(; i < count; i++){
            uint16_t h;
            memcpy(&h, src + 2 * i, 2);
            uint32_t wide = (uint32_t)h << 16;
            float f;
            memcpy(&f, &wide, 4);
            out[i] = type == 1 ? _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h))) : f;
        }
        return;
    }

    const __m256i fifteen = _mm256_set1_epi32(15);
    const __m256i three = _mm256_set1_epi32(3);
    const __m256i one = _mm256_set1_epi32(1);
    const __m256i thirtyTwo = _mm256_set1_epi32(32);
    const __m256i pairs = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);

    uint32_t per = blockElements(type);
    uint64_t size = rowBytes(type, per);
    int64_t blocks = count / per;

    for(int64_t b = 0; b < blocks; b++){
        const uint8_t* q = src + b * size;
        float* y = out + b * per;

        switch(type){
            case 12:
            case 13: {
                float d = half(q), dmin = half(q + 2);
                const uint8_t* s = q + 4;
                const uint8_t* qh = q + 16;
                const uint8_t* qs = type == 12 ? q + 16 : q + 48;
                for(int c = 0; c < 4; c++){
                    float scale[2], low[2];
                    for(int k = 0; k < 2; k++){
                        int j = 2 * c + k;
                        int sc = j < 4
                            ? s[j] & 63
                            : (s[j + 4] & 15) | ((s[j - 4] >> 6) << 4);
                        int m = j < 4
                            ? s[j + 4] & 63
                            : (s[j + 4] >> 4) | ((s[j] >> 6) << 4);
                        scale[k] = d * (float)sc;
                        low[k] = dmin * (float)m;
                    }
                    __m256 d1 = _mm256_set1_ps(scale[0]), m1 = _mm256_set1_ps(low[0]);
                    __m256 d2 = _mm256_set1_ps(scale[1]), m2 = _mm256_set1_ps(low[1]);
                    __m128i shiftLow = _mm_cvtsi32_si128(2 * c);
                    __m128i shiftHigh = _mm_cvtsi32_si128(2 * c + 1);
                    for(int l = 0; l < 32; l += 8){
                        __m256i v = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(qs + 32 * c + l)));
                        __m256i lo = _mm256_and_si256(v, fifteen);
                        __m256i hi = _mm256_srli_epi32(v, 4);
                        if(type == 13){
                            __m256i h = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(qh + l)));
                            lo = _mm256_or_si256(lo, _mm256_slli_epi32(_mm256_and_si256(_mm256_srl_epi32(h, shiftLow), one), 4));
                            hi = _mm256_or_si256(hi, _mm256_slli_epi32(_mm256_and_si256(_mm256_srl_epi32(h, shiftHigh), one), 4));
                        }
                        _mm256_storeu_ps(y + 64 * c + l, _mm256_sub_ps(_mm256_mul_ps(d1, _mm256_cvtepi32_ps(lo)), m1));
                        _mm256_storeu_ps(y + 64 * c + 32 + l, _mm256_sub_ps(_mm256_mul_ps(d2, _mm256_cvtepi32_ps(hi)), m2));
                    }
                }
                break;
            }

            case 14: {
                float d = half(q + 208);
                for(int n = 0; n < 2; n++){
                    const uint8_t* ql = q + 64 * n;
                    const uint8_t* qh = q + 128 + 32 * n;
                    const int8_t* sc = (const int8_t*)(q + 192 + 8 * n);
                    float* o = y + 128 * n;
                    for(int l = 0; l < 32; l += 8){
                        int is = l / 16;
                        __m256i a = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(ql + l)));
                        __m256i c = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(ql + l + 32)));
                        __m256i h = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(qh + l)));
                        __m256i q1 = _mm256_or_si256(_mm256_and_si256(a, fifteen), _mm256_slli_epi32(_mm256_and_si256(h, three), 4));
                        __m256i q2 = _mm256_or_si256(_mm256_and_si256(c, fifteen), _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(h, 2), three), 4));
                        __m256i q3 = _mm256_or_si256(_mm256_srli_epi32(a, 4), _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(h, 4), three), 4));
                        __m256i q4 = _mm256_or_si256(_mm256_srli_epi32(c, 4), _mm256_slli_epi32(_mm256_srli_epi32(h, 6), 4));
                        _mm256_storeu_ps(o + l, _mm256_mul_ps(_mm256_set1_ps(d * (float)sc[is]), _mm256_cvtepi32_ps(_mm256_sub_epi32(q1, thirtyTwo))));
                        _mm256_storeu_ps(o + l + 32, _mm256_mul_ps(_mm256_set1_ps(d * (float)sc[is + 2]), _mm256_cvtepi32_ps(_mm256_sub_epi32(q2, thirtyTwo))));
                        _mm256_storeu_ps(o + l + 64, _mm256_mul_ps(_mm256_set1_ps(d * (float)sc[is + 4]), _mm256_cvtepi32_ps(_mm256_sub_epi32(q3, thirtyTwo))));
                        _mm256_storeu_ps(o + l + 96, _mm256_mul_ps(_mm256_set1_ps(d * (float)sc[is + 6]), _mm256_cvtepi32_ps(_mm256_sub_epi32(q4, thirtyTwo))));
                    }
                }
                break;
            }

            case 16: {
                float d = half(q);
                const uint8_t* qs = q + 2;
                for(int ib = 0; ib < 8; ib++){
                    uint32_t index, aux;
                    memcpy(&index, qs + 8 * ib, 4);
                    memcpy(&aux, qs + 8 * ib + 4, 4);
                    float scale = d * (0.5f + (float)(aux >> 28)) * 0.25f;
                    for(int l = 0; l < 4; l++){
                        _mm256_storeu_ps(y + 32 * ib + 8 * l, signedGrid(grid2xxs[(index >> (8 * l)) & 255], signTable[(aux >> (7 * l)) & 127], scale));
                    }
                }
                break;
            }

            case 17: {
                float d = half(q);
                const uint8_t* scales = q + 66;
                for(int ib = 0; ib < 8; ib++){
                    for(int l = 0; l < 4; l++){
                        uint16_t v;
                        memcpy(&v, q + 2 + 2 * (4 * ib + l), 2);
                        float scale = d * (0.5f + (float)((scales[ib] >> (4 * (l / 2))) & 15)) * 0.25f;
                        _mm256_storeu_ps(y + 32 * ib + 8 * l, signedGrid(grid2xs[v & 511], signTable[v >> 9], scale));
                    }
                }
                break;
            }

            case 18: {
                float d = half(q);
                const uint8_t* qs = q + 2;
                for(int ib = 0; ib < 8; ib++){
                    uint32_t aux;
                    memcpy(&aux, q + 66 + 4 * ib, 4);
                    float scale = d * (0.5f + (float)(aux >> 28)) * 0.5f;
                    for(int l = 0; l < 4; l++){
                        const uint8_t* g = qs + 8 * ib + 2 * l;
                        uint64_t magnitudes = grid3xxs[g[0]] | ((uint64_t)grid3xxs[g[1]] << 32);
                        _mm256_storeu_ps(y + 32 * ib + 8 * l, signedGrid(magnitudes, signTable[(aux >> (7 * l)) & 127], scale));
                    }
                }
                break;
            }

            case 20:
                nibbleBlock(q + 2, half(q), y);
                break;

            case 39:
                fp4Codes(q + 1, 16, mxScale(q[0]), y);
                break;

            case 40:
                for(int s = 0; s < 4; s++) fp4Codes(q + 4 + 8 * s, 8, nvScale(q[s]), y + 16 * s);
                break;

            case 21: {
                float d = half(q);
                const uint8_t* qs = q + 2;
                const uint8_t* qh = q + 66;
                const uint8_t* signs = q + 74;
                const uint8_t* scales = q + 106;
                for(int ib = 0; ib < 8; ib++){
                    float scale = d * (float)(1 + 2 * ((scales[ib / 2] >> (4 * (ib % 2))) & 15));
                    for(int l = 0; l < 4; l++){
                        int a = qs[8 * ib + 2 * l] | ((qh[ib] << (8 - 2 * l)) & 256);
                        int c = qs[8 * ib + 2 * l + 1] | ((qh[ib] << (7 - 2 * l)) & 256);
                        uint64_t magnitudes = grid3s[a] | ((uint64_t)grid3s[c] << 32);
                        _mm256_storeu_ps(y + 32 * ib + 8 * l, signedGrid(magnitudes, signs[4 * ib + l], scale));
                    }
                }
                break;
            }

            case 22: {
                float d = half(q);
                const uint8_t* qs = q + 2;
                const uint8_t* qh = q + 66;
                const uint8_t* scales = q + 74;
                for(int ib = 0; ib < 8; ib++){
                    for(int l = 0; l < 4; l++){
                        int index = qs[4 * ib + l] | ((qh[ib] << (8 - 2 * l)) & 0x300);
                        float scale = d * (0.5f + (float)((scales[ib] >> (4 * (l / 2))) & 15)) * 0.25f;
                        _mm256_storeu_ps(y + 32 * ib + 8 * l, signedGrid(grid2s[index], qs[32 + 4 * ib + l], scale));
                    }
                }
                break;
            }

            case 29: {
                float d = grid1Scale(q);
                for(int ib = 0; ib < 8; ib++){
                    uint16_t sc;
                    memcpy(&sc, q + 48 + 2 * (ib / 2), 2);
                    int s = sc >> (6 * (ib % 2));
                    for(int l = 0; l < 4; l++){
                        int h = q[32 + 2 * ib + l / 2] >> (4 * (l % 2));
                        __m256 v = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_cvtsi64_si128((long long)grid1s[q[4 * ib + l] | ((h & 7) << 8)])));
                        float scale = d * (float)(2 * ((s >> (3 * (l / 2))) & 7) + 1);
                        _mm256_storeu_ps(y + 32 * ib + 8 * l, _mm256_mul_ps(_mm256_add_ps(v, _mm256_set1_ps(h & 8 ? -1.0f : 1.0f)), _mm256_set1_ps(scale)));
                    }
                }
                break;
            }

            case 23: {
                float d = half(q);
                uint16_t high;
                memcpy(&high, q + 2, 2);
                const uint8_t* lows = q + 4;
                for(int ib = 0; ib < 8; ib++){
                    int ls = ((lows[ib / 2] >> (4 * (ib % 2))) & 15) | (((high >> (2 * ib)) & 3) << 4);
                    nibbleBlock(q + 8 + 16 * ib, d * (float)(ls - 32), y + 32 * ib);
                }
                break;
            }

            case 42: {
                __m256 d = _mm256_set1_ps(half(q));
                for(int g = 0; g < 8; g++){
                    uint16_t two;
                    memcpy(&two, q + 2 + 2 * g, 2);
                    __m256i codes = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32(two), pairs), three);
                    _mm256_storeu_ps(y + 8 * g, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(codes, one)), d));
                }
                break;
            }

            case 8: {
                float d = half(q);
                for(int i = 0; i < 32; i++) y[i] = d * (float)(int8_t)q[2 + i];
                break;
            }

            case 7: {
                float d = half(q), m = half(q + 2);
                uint32_t qh;
                memcpy(&qh, q + 4, 4);
                for(int i = 0; i < 16; i++){
                    int lo = (q[8 + i] & 15) | (((qh >> i) & 1) << 4);
                    int hi = (q[8 + i] >> 4) | (((qh >> (i + 16)) & 1) << 4);
                    y[i] = d * (float)lo + m;
                    y[i + 16] = d * (float)hi + m;
                }
                break;
            }

            default:
                memset(y, 0, per * sizeof(float));
                break;
        }
    }
}

inline float dot(const float* a, const float* b, int n) {
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps(), s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
    int i = 0;
    for(; i + 32 <= n; i += 32){
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
        s2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), s2);
        s3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), s3);
    }
    for(; i + 8 <= n; i += 8) s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
    __m256 s = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
    __m128 h = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
    h = _mm_add_ps(h, _mm_movehl_ps(h, h));
    h = _mm_add_ss(h, _mm_movehdup_ps(h));
    float total = _mm_cvtss_f32(h);
    for(; i < n; i++) total += a[i] * b[i];
    return total;
}

struct ThreadPool {
    std::vector<std::thread> threads;
    std::mutex lock;
    std::condition_variable wake;
    alignas(64) const std::function<void(int)>* jobs[2] = {nullptr, nullptr};
    int totals[2] = {0, 0};
    std::atomic<bool> stop{false};
    alignas(64) std::atomic<uint64_t> round{0};
    alignas(64) std::atomic<uint64_t> claim{0};
    alignas(64) std::atomic<int> finished{0};

    void take(uint64_t seen) {
        int side = (int)(seen & 1);
        uint64_t tag = (seen & 0xFFFFFFFFull) << 32;
        uint64_t c = claim.load(std::memory_order_acquire);
        while((c & ~0xFFFFFFFFull) == tag && (int)(c & 0xFFFFFFFFull) < totals[side]){
            if(!claim.compare_exchange_weak(c, c + 1, std::memory_order_acq_rel, std::memory_order_acquire)) continue;
            (*jobs[side])((int)(c & 0xFFFFFFFFull));
            finished.fetch_add(1, std::memory_order_acq_rel);
            c = claim.load(std::memory_order_acquire);
        }
    }

    void start(int count, std::vector<GROUP_AFFINITY> places = {}) {
        for(int w = 0; w < count; w++){
            GROUP_AFFINITY place = places.empty() ? GROUP_AFFINITY{} : places[w % places.size()];
            threads.emplace_back([this, place]{
                if(place.Mask) SetThreadGroupAffinity(GetCurrentThread(), &place, nullptr);
                uint64_t seen = 0;
                while(true){
                    auto idle = std::chrono::steady_clock::now();
                    while(round.load(std::memory_order_acquire) == seen && std::chrono::steady_clock::now() - idle < std::chrono::milliseconds(4)){
                        for(int spin = 0; spin < 64 && round.load(std::memory_order_acquire) == seen; spin++) _mm_pause();
                    }
                    if(round.load(std::memory_order_acquire) == seen){
                        std::unique_lock<std::mutex> guard(lock);
                        wake.wait(guard, [&]{ return round.load() != seen; });
                    }
                    if(stop) return;
                    seen = round.load(std::memory_order_acquire);
                    take(seen);
                }
            });
        }
    }

    void run(int count, const std::function<void(int)>& work) {
        if(threads.empty() || count <= 1){
            for(int i = 0; i < count; i++) work(i);
            return;
        }
        uint64_t next = round.load(std::memory_order_relaxed) + 1;
        int side = (int)(next & 1);
        jobs[side] = &work;
        totals[side] = count;
        finished.store(0, std::memory_order_relaxed);
        claim.store((next & 0xFFFFFFFFull) << 32, std::memory_order_release);
        {
            std::lock_guard<std::mutex> guard(lock);
            round.store(next, std::memory_order_release);
        }
        wake.notify_all();
        take(next);
        while(finished.load(std::memory_order_acquire) < count) _mm_pause();
    }

    ~ThreadPool() {
        stop = true;
        {
            std::lock_guard<std::mutex> guard(lock);
            round++;
        }
        wake.notify_all();
        for(std::thread& t : threads) t.join();
    }
};

struct Matrix {
    const uint8_t* data = nullptr;
    uint32_t type = 0;
    int cols = 0;
    int rows = 0;
    uint64_t stride = 0;
};

struct Product {
    Matrix weight;
    const float* input = nullptr;
    float* output = nullptr;
    int count = 1;
    float scale = 1.0f;
};

inline void multiply(ThreadPool& pool, const std::vector<Product>& list) {
    std::vector<int> owner, first, span;
    for(size_t p = 0; p < list.size(); p++){
        const Matrix& m = list[p].weight;
        if(!m.data || m.rows == 0 || m.cols > 12288) continue;
        int step = std::max(1, std::min(m.rows, 49152 / std::max(1, m.cols)));
        for(int r = 0; r < m.rows; r += step){
            owner.push_back((int)p);
            first.push_back(r);
            span.push_back(std::min(step, m.rows - r));
        }
    }

    pool.run((int)owner.size(), [&](int chunk){
        const Product& job = list[owner[chunk]];
        const Matrix& m = job.weight;
        alignas(32) float row[12288];
        int end = first[chunk] + span[chunk];
        for(int r = first[chunk]; r < end; r++){
            dequantRow(m.type, m.data + (uint64_t)r * m.stride, row, m.cols);
            for(int t = 0; t < job.count; t++){
                job.output[(size_t)t * m.rows + r] = dot(row, job.input + (size_t)t * m.cols, m.cols) * job.scale;
            }
        }
    });
}
