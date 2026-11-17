#pragma once
#include "quant.h"
#include <intrin.h>

struct HostPacked {
    std::vector<int8_t> levels;
    std::vector<float> units;
    std::vector<int> sums;

    void fill(const float* x, int count) {
        levels.resize(2 * (size_t)count);
        units.resize(count / 16);
        sums.resize(count / 16);
        const __m256 sign = _mm256_set1_ps(-0.0f);
        const __m256i order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
        auto widest = [](__m256 v){
            __m128 m = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
            m = _mm_max_ps(m, _mm_movehl_ps(m, m));
            m = _mm_max_ss(m, _mm_movehdup_ps(m));
            return _mm_cvtss_f32(m);
        };
        auto narrow = [&](const __m256i* c, int8_t* out){
            __m256i bytes = _mm256_packs_epi16(_mm256_packs_epi32(c[0], c[1]), _mm256_packs_epi32(c[2], c[3]));
            _mm256_storeu_si256((__m256i*)out, _mm256_permutevar8x32_epi32(bytes, order));
        };
        for(int g = 0; g < count / 32; g++){
            const float* v = x + 32 * g;
            int8_t* both = levels.data() + 64 * (size_t)g;
            __m256 a[4], rest[4];
            __m256i c[4], f[4];
            __m256 top = _mm256_setzero_ps();
            for(int k = 0; k < 4; k++){
                a[k] = _mm256_loadu_ps(v + 8 * k);
                top = _mm256_max_ps(top, _mm256_andnot_ps(sign, a[k]));
            }
            float scale = widest(top) / 127.0f;
            float inv = scale > 0 ? 1.0f / scale : 0.0f;
            __m256 scales = _mm256_set1_ps(scale);
            __m256 invs = _mm256_set1_ps(inv);
            __m256 restTop = _mm256_setzero_ps();
            for(int k = 0; k < 4; k++){
                c[k] = _mm256_cvtps_epi32(_mm256_mul_ps(a[k], invs));
                rest[k] = _mm256_sub_ps(a[k], _mm256_mul_ps(scales, _mm256_cvtepi32_ps(c[k])));
                restTop = _mm256_max_ps(restTop, _mm256_andnot_ps(sign, rest[k]));
            }
            float fine = widest(restTop) / 127.0f;
            float fineInv = fine > 0 ? 1.0f / fine : 0.0f;
            __m256 fines = _mm256_set1_ps(fineInv);
            for(int k = 0; k < 4; k++) f[k] = _mm256_cvtps_epi32(_mm256_mul_ps(rest[k], fines));
            narrow(c, both);
            narrow(f, both + 32);
            units[2 * (size_t)g] = scale;
            units[2 * (size_t)g + 1] = fine;
            int sumQ = 0, sumR = 0;
            for(int k = 0; k < 32; k++){
                sumQ += both[k];
                sumR += both[32 + k];
            }
            sums[2 * (size_t)g] = sumQ;
            sums[2 * (size_t)g + 1] = sumR;
        }
    }
};

inline void hostSwiglu(float* gate, const float* up, size_t count) {
    const __m256 low = _mm256_set1_ps(-87.0f);
    const __m256 high = _mm256_set1_ps(88.0f);
    const __m256 one = _mm256_set1_ps(1.0f);
    size_t i = 0;
    for(; i + 8 <= count; i += 8){
        __m256 x = _mm256_loadu_ps(gate + i);
        __m256 t = _mm256_min_ps(_mm256_max_ps(_mm256_sub_ps(_mm256_setzero_ps(), x), low), high);
        __m256 n = _mm256_round_ps(_mm256_mul_ps(t, _mm256_set1_ps(1.44269504f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256 rr = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.693359375f), t);
        rr = _mm256_fnmadd_ps(n, _mm256_set1_ps(-2.12194440e-4f), rr);
        __m256 p = _mm256_set1_ps(1.9875691500e-4f);
        p = _mm256_fmadd_ps(p, rr, _mm256_set1_ps(1.3981999507e-3f));
        p = _mm256_fmadd_ps(p, rr, _mm256_set1_ps(8.3334519073e-3f));
        p = _mm256_fmadd_ps(p, rr, _mm256_set1_ps(4.1665795894e-2f));
        p = _mm256_fmadd_ps(p, rr, _mm256_set1_ps(1.6666665459e-1f));
        p = _mm256_fmadd_ps(p, rr, _mm256_set1_ps(5.0000001201e-1f));
        p = _mm256_fmadd_ps(p, _mm256_mul_ps(rr, rr), _mm256_add_ps(rr, one));
        __m256 power = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23));
        __m256 denominator = _mm256_add_ps(one, _mm256_mul_ps(p, power));
        _mm256_storeu_ps(gate + i, _mm256_div_ps(_mm256_mul_ps(x, _mm256_loadu_ps(up + i)), denominator));
    }
    for(; i < count; i++) gate[i] = gate[i] / (1.0f + expf(-gate[i])) * up[i];
}

inline bool hostDotType(uint32_t type) {
    return type == 16 || type == 17 || type == 18 || type == 20 || type == 21 || type == 22 || type == 42 || type == 23 || type == 8 || type == 7 || type == 12 || type == 13 || type == 14 || type == 29;
}

inline __m256i bitBytes(uint32_t bits) {
    const __m256i spread = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    const __m256i pick = _mm256_set1_epi64x((long long)0x8040201008040201ULL);
    __m256i v = _mm256_shuffle_epi8(_mm256_set1_epi32((int)bits), spread);
    return _mm256_cmpeq_epi8(_mm256_and_si256(v, pick), pick);
}

inline __m256i signBytes(uint32_t bits) {
    return _mm256_or_si256(bitBytes(bits), _mm256_set1_epi8(1));
}

inline void kScaleMin(int j, const uint8_t* s, int& sc, int& m) {
    if(j < 4){
        sc = s[j] & 63;
        m = s[j + 4] & 63;
        return;
    }
    sc = (s[j + 4] & 15) | ((s[j - 4] >> 6) << 4);
    m = (s[j + 4] >> 4) | ((s[j] >> 6) << 4);
}

template<int TYPE>
inline void groupCodes(const uint8_t* block, int g, float base, __m256i& mag, __m256i& sign, __m256& scale, float& drop) {
    const __m256i nibble = _mm256_set1_epi8(15);
    if constexpr(TYPE == 23){
        const __m128i table = _mm_loadu_si128((const __m128i*)nonLinear4);
        const __m128i mask = _mm_set1_epi8(15);
        __m128i raw = _mm_loadu_si128((const __m128i*)(block + 8 + 16 * g));
        __m128i lo = _mm_shuffle_epi8(table, _mm_and_si128(raw, mask));
        __m128i hi = _mm_shuffle_epi8(table, _mm_and_si128(_mm_srli_epi16(raw, 4), mask));
        __m256i w = _mm256_set_m128i(hi, lo);
        mag = _mm256_sign_epi8(w, w);
        sign = w;
        int low = (block[4 + g / 2] >> (4 * (g % 2))) & 15;
        int high = (((int)block[2] | ((int)block[3] << 8)) >> (2 * g)) & 3;
        scale = _mm256_set1_ps(base * (float)((low | (high << 4)) - 32));
    } else if constexpr(TYPE == 8){
        __m256i w = _mm256_loadu_si256((const __m256i*)(block + 2));
        mag = _mm256_sign_epi8(w, w);
        sign = w;
        scale = _mm256_set1_ps(base);
    } else if constexpr(TYPE == 7){
        uint32_t qh;
        memcpy(&qh, block + 4, 4);
        const __m128i mask = _mm_set1_epi8(15);
        __m128i raw = _mm_loadu_si128((const __m128i*)(block + 8));
        __m256i w = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(raw, 4), mask), _mm_and_si128(raw, mask));
        mag = _mm256_or_si256(w, _mm256_and_si256(bitBytes(qh), _mm256_set1_epi8(16)));
        sign = _mm256_set1_epi8(1);
        scale = _mm256_set1_ps(base);
        drop = -half(block + 2);
    } else if constexpr(TYPE == 12 || TYPE == 13){
        int sc, mn;
        kScaleMin(g, block + 4, sc, mn);
        __m256i raw = _mm256_loadu_si256((const __m256i*)((TYPE == 12 ? block + 16 : block + 48) + 32 * (g / 2)));
        __m256i w = (g & 1) ? _mm256_and_si256(_mm256_srli_epi16(raw, 4), nibble) : _mm256_and_si256(raw, nibble);
        if constexpr(TYPE == 13){
            __m256i top = _mm256_and_si256(_mm256_srli_epi16(_mm256_loadu_si256((const __m256i*)(block + 16)), g), _mm256_set1_epi8(1));
            w = _mm256_or_si256(w, _mm256_slli_epi16(top, 4));
        }
        mag = w;
        sign = _mm256_set1_epi8(1);
        scale = _mm256_set1_ps(base * (float)sc);
        drop = half(block + 2) * (float)mn;
    } else if constexpr(TYPE == 14){
        int k = g % 4;
        const int8_t* sc = (const int8_t*)(block + 192) + 8 * (g / 4);
        __m256i low = _mm256_loadu_si256((const __m256i*)(block + 64 * (g / 4) + (k & 1) * 32));
        low = (k >> 1) ? _mm256_and_si256(_mm256_srli_epi16(low, 4), nibble) : _mm256_and_si256(low, nibble);
        __m256i high = _mm256_and_si256(_mm256_srli_epi16(_mm256_loadu_si256((const __m256i*)(block + 128 + 32 * (g / 4))), 2 * k), _mm256_set1_epi8(3));
        __m256i v = _mm256_sub_epi8(_mm256_or_si256(low, _mm256_slli_epi16(high, 4)), _mm256_set1_epi8(32));
        mag = _mm256_sign_epi8(v, v);
        sign = v;
        scale = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_set1_ps(base * (float)sc[2 * k])), _mm_set1_ps(base * (float)sc[2 * k + 1]), 1);
    } else if constexpr(TYPE == 16){
        uint32_t index, aux;
        memcpy(&index, block + 2 + 8 * g, 4);
        memcpy(&aux, block + 6 + 8 * g, 4);
        mag = _mm256_set_epi64x((long long)grid2xxs[index >> 24], (long long)grid2xxs[(index >> 16) & 255], (long long)grid2xxs[(index >> 8) & 255], (long long)grid2xxs[index & 255]);
        sign = signBytes((uint32_t)signTable[aux & 127] | ((uint32_t)signTable[(aux >> 7) & 127] << 8) | ((uint32_t)signTable[(aux >> 14) & 127] << 16) | ((uint32_t)signTable[(aux >> 21) & 127] << 24));
        scale = _mm256_set1_ps(base * (0.5f + (float)(aux >> 28)) * 0.25f);
    } else if constexpr(TYPE == 17){
        uint64_t c;
        memcpy(&c, block + 2 + 8 * g, 8);
        uint32_t c0 = (uint32_t)c & 0xFFFF, c1 = (uint32_t)(c >> 16) & 0xFFFF, c2 = (uint32_t)(c >> 32) & 0xFFFF, c3 = (uint32_t)(c >> 48);
        mag = _mm256_set_epi64x((long long)grid2xs[c3 & 511], (long long)grid2xs[c2 & 511], (long long)grid2xs[c1 & 511], (long long)grid2xs[c0 & 511]);
        sign = signBytes((uint32_t)signTable[c0 >> 9] | ((uint32_t)signTable[c1 >> 9] << 8) | ((uint32_t)signTable[c2 >> 9] << 16) | ((uint32_t)signTable[c3 >> 9] << 24));
        uint8_t s = block[66 + g];
        float dd = base * 0.25f;
        scale = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_set1_ps(dd * (0.5f + (float)(s & 15)))), _mm_set1_ps(dd * (0.5f + (float)(s >> 4))), 1);
    } else if constexpr(TYPE == 22){
        const uint8_t* qs = block + 2 + 4 * g;
        uint32_t qh = block[66 + g];
        mag = _mm256_set_epi64x((long long)grid2s[qs[3] | ((qh << 2) & 0x300)], (long long)grid2s[qs[2] | ((qh << 4) & 0x300)], (long long)grid2s[qs[1] | ((qh << 6) & 0x300)], (long long)grid2s[qs[0] | ((qh << 8) & 0x300)]);
        uint32_t bits;
        memcpy(&bits, block + 34 + 4 * g, 4);
        sign = signBytes(bits);
        uint8_t s = block[74 + g];
        float dd = base * 0.25f;
        scale = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_set1_ps(dd * (0.5f + (float)(s & 15)))), _mm_set1_ps(dd * (0.5f + (float)(s >> 4))), 1);
    } else if constexpr(TYPE == 18){
        const uint8_t* qs = block + 2 + 8 * g;
        uint32_t aux;
        memcpy(&aux, block + 66 + 4 * g, 4);
        uint64_t m0 = grid3xxs[qs[0]] | ((uint64_t)grid3xxs[qs[1]] << 32);
        uint64_t m1 = grid3xxs[qs[2]] | ((uint64_t)grid3xxs[qs[3]] << 32);
        uint64_t m2 = grid3xxs[qs[4]] | ((uint64_t)grid3xxs[qs[5]] << 32);
        uint64_t m3 = grid3xxs[qs[6]] | ((uint64_t)grid3xxs[qs[7]] << 32);
        mag = _mm256_set_epi64x((long long)m3, (long long)m2, (long long)m1, (long long)m0);
        sign = signBytes((uint32_t)signTable[aux & 127] | ((uint32_t)signTable[(aux >> 7) & 127] << 8) | ((uint32_t)signTable[(aux >> 14) & 127] << 16) | ((uint32_t)signTable[(aux >> 21) & 127] << 24));
        scale = _mm256_set1_ps(base * (0.5f + (float)(aux >> 28)) * 0.5f);
    } else if constexpr(TYPE == 21){
        const uint8_t* qs = block + 2 + 8 * g;
        uint32_t qh = block[66 + g];
        uint64_t m0 = grid3s[qs[0] | ((qh << 8) & 256)] | ((uint64_t)grid3s[qs[1] | ((qh << 7) & 256)] << 32);
        uint64_t m1 = grid3s[qs[2] | ((qh << 6) & 256)] | ((uint64_t)grid3s[qs[3] | ((qh << 5) & 256)] << 32);
        uint64_t m2 = grid3s[qs[4] | ((qh << 4) & 256)] | ((uint64_t)grid3s[qs[5] | ((qh << 3) & 256)] << 32);
        uint64_t m3 = grid3s[qs[6] | ((qh << 2) & 256)] | ((uint64_t)grid3s[qs[7] | ((qh << 1) & 256)] << 32);
        mag = _mm256_set_epi64x((long long)m3, (long long)m2, (long long)m1, (long long)m0);
        uint32_t bits;
        memcpy(&bits, block + 74 + 4 * g, 4);
        sign = signBytes(bits);
        scale = _mm256_set1_ps(base * (float)(1 + 2 * ((block[106 + g / 2] >> (4 * (g % 2))) & 15)));
    } else if constexpr(TYPE == 20){
        const __m128i table = _mm_loadu_si128((const __m128i*)nonLinear4);
        const __m128i mask = _mm_set1_epi8(15);
        __m128i raw = _mm_loadu_si128((const __m128i*)(block + 2));
        __m128i lo = _mm_shuffle_epi8(table, _mm_and_si128(raw, mask));
        __m128i hi = _mm_shuffle_epi8(table, _mm_and_si128(_mm_srli_epi16(raw, 4), mask));
        __m256i w = _mm256_set_m128i(hi, lo);
        mag = _mm256_sign_epi8(w, w);
        sign = w;
        scale = _mm256_set1_ps(base);
    } else if constexpr(TYPE == 29){
        const uint8_t* qs = block + 4 * g;
        const uint8_t* qh = block + 32 + 2 * g;
        long long parts[4], shifts[4];
        for(int l = 0; l < 4; l++){
            int h = qh[l / 2] >> (4 * (l % 2));
            parts[l] = (long long)grid1s[qs[l] | ((h & 7) << 8)];
            shifts[l] = h & 8 ? (long long)0xFFFFFFFFFFFFFFFFULL : 0x0101010101010101LL;
        }
        __m256i w = _mm256_add_epi8(_mm256_set_epi64x(parts[3], parts[2], parts[1], parts[0]), _mm256_set_epi64x(shifts[3], shifts[2], shifts[1], shifts[0]));
        mag = _mm256_sign_epi8(w, w);
        sign = w;
        uint16_t sc;
        memcpy(&sc, block + 48 + 2 * (g / 2), 2);
        int s = sc >> (6 * (g % 2));
        scale = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_set1_ps(base * (float)(2 * (s & 7) + 1))), _mm_set1_ps(base * (float)(2 * ((s >> 3) & 7) + 1)), 1);
    } else {
        uint64_t packedCodes;
        memcpy(&packedCodes, block + 2 + 8 * g, 8);
        const __m256i spread = _mm256_setr_epi8(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7);
        const __m256i first = _mm256_set1_epi32(0x00000003);
        const __m256i second = _mm256_set1_epi32(0x00000300);
        const __m256i third = _mm256_set1_epi32(0x00030000);
        const __m256i fourth = _mm256_set1_epi32(0x03000000);
        __m256i v = _mm256_shuffle_epi8(_mm256_set1_epi64x((long long)packedCodes), spread);
        __m256i code = _mm256_or_si256(_mm256_and_si256(v, first), _mm256_and_si256(_mm256_srli_epi16(v, 2), second));
        code = _mm256_or_si256(code, _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(v, 4), third), _mm256_and_si256(_mm256_srli_epi16(v, 6), fourth)));
        __m256i w = _mm256_sub_epi8(code, _mm256_set1_epi8(1));
        mag = _mm256_sign_epi8(w, w);
        sign = w;
        scale = _mm256_set1_ps(base);
    }
}

// avx-vnni de las intel desde la 12a generacion multiplica bytes y suma de a cuatro en una instruccion
// sin vnni van maddubs y madd que dan lo mismo porque los productos de estos pesos no llenan los 16 bits intermedios
inline bool hostVnni() {
    static const bool found = []{
        int info[4] = {};
        __cpuidex(info, 7, 1);
        return (info[0] & (1 << 4)) != 0;
    }();
    return found;
}

template<bool VNNI>
inline __m256i byteDots(__m256i mag, __m256i x) {
    if constexpr(VNNI) return _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), mag, x);
    else return _mm256_madd_epi16(_mm256_maddubs_epi16(mag, x), _mm256_set1_epi16(1));
}

template<int TYPE, bool RESIDUAL, int COUNT, bool VNNI>
inline void rowDots(const uint8_t* row, int cols, const HostPacked& in, size_t at, float* out, size_t outStep) {
    constexpr int span = TYPE == 20 || TYPE == 8 || TYPE == 7
        ? 32
        : TYPE == 42
            ? 64
            : 256;
    constexpr int perBlock = span / 32;
    constexpr uint64_t bytes = TYPE == 16
        ? 66
        : TYPE == 17
            ? 74
            : TYPE == 22
                ? 82
                : TYPE == 18
                    ? 98
                    : TYPE == 21
                        ? 110
                        : TYPE == 23
                            ? 136
                            : TYPE == 12
                                ? 144
                                : TYPE == 13
                                    ? 176
                                    : TYPE == 14
                                        ? 210
                                        : TYPE == 8
                                            ? 34
                                            : TYPE == 7
                                                ? 24
                                                : TYPE == 29
                                                    ? 56
                                                    : 18;
    constexpr bool lowered = TYPE == 12 || TYPE == 13 || TYPE == 7;
    __m256 even[COUNT], odd[COUNT];
    float minus[COUNT];
    const int8_t* x[COUNT];
    const float* s[COUNT];
    const int* sum[COUNT];
    for(int t = 0; t < COUNT; t++){
        size_t place = (at + (size_t)t * cols) / 32;
        even[t] = _mm256_setzero_ps();
        odd[t] = _mm256_setzero_ps();
        minus[t] = 0.0f;
        x[t] = in.levels.data() + 64 * place;
        s[t] = in.units.data() + 2 * place;
        sum[t] = in.sums.data() + 2 * place;
    }
    auto group = [&](const uint8_t* block, int g, float base, size_t u, __m256* acc){
        __m256i mag, sign;
        __m256 scale;
        float drop = 0.0f;
        groupCodes<TYPE>(block, g, base, mag, sign, scale, drop);
        for(int t = 0; t < COUNT; t++){
            const int8_t* v = x[t] + 64 * u;
            const float* k = s[t] + 2 * u;
            __m256i qv = _mm256_loadu_si256((const __m256i*)v);
            __m256 part = _mm256_mul_ps(_mm256_cvtepi32_ps(byteDots<VNNI>(mag, lowered ? qv : _mm256_sign_epi8(qv, sign))), _mm256_set1_ps(k[0]));
            if constexpr(RESIDUAL){
                __m256i rv = _mm256_loadu_si256((const __m256i*)(v + 32));
                part = _mm256_fmadd_ps(_mm256_cvtepi32_ps(byteDots<VNNI>(mag, lowered ? rv : _mm256_sign_epi8(rv, sign))), _mm256_set1_ps(k[1]), part);
            }
            acc[t] = _mm256_fmadd_ps(part, scale, acc[t]);
            if constexpr(TYPE == 12 || TYPE == 13 || TYPE == 7) minus[t] += drop * (k[0] * (float)sum[t][2 * u] + (RESIDUAL ? k[1] * (float)sum[t][2 * u + 1] : 0.0f));
        }
    };
    size_t blocks = (size_t)(cols / span);
    if constexpr(perBlock == 1){
        size_t b = 0;
        for(; b + 1 < blocks; b += 2){
            const uint8_t* block = row + b * bytes;
            group(block, 0, half(block), b, even);
            group(block + bytes, 0, half(block + bytes), b + 1, odd);
        }
        if(b < blocks) group(row + b * bytes, 0, half(row + b * bytes), b, even);
    } else {
        for(size_t b = 0; b < blocks; b++){
            const uint8_t* block = row + b * bytes;
            float base = TYPE == 29
                ? grid1Scale(block)
                : half(TYPE == 14 ? block + 208 : block);
            for(int g = 0; g < perBlock; g += 2){
                group(block, g, base, b * perBlock + g, even);
                group(block, g + 1, base, b * perBlock + g + 1, odd);
            }
        }
    }
    for(int t = 0; t < COUNT; t++){
        __m256 acc = _mm256_add_ps(even[t], odd[t]);
        __m128 h = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
        h = _mm_add_ps(h, _mm_movehl_ps(h, h));
        h = _mm_add_ss(h, _mm_movehdup_ps(h));
        out[(size_t)t * outStep] = _mm_cvtss_f32(h) - minus[t];
    }
}

template<int TYPE, bool RESIDUAL, int COUNT, bool VNNI>
inline void blockDots(const uint8_t* row, int cols, const HostPacked& in, size_t at, float* out, size_t outStep) {
    constexpr uint64_t bytes = TYPE == 18
        ? 98
        : TYPE == 22
            ? 82
            : 110;
    const __m256i spread = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    const __m256i pick = _mm256_set1_epi64x((long long)0x8040201008040201ULL);
    const __m256i unit = _mm256_set1_epi8(1);
    __m256 even[COUNT], odd[COUNT];
    const int8_t* x[COUNT];
    const float* s[COUNT];
    for(int t = 0; t < COUNT; t++){
        size_t place = (at + (size_t)t * cols) / 32;
        even[t] = _mm256_setzero_ps();
        odd[t] = _mm256_setzero_ps();
        x[t] = in.levels.data() + 64 * place;
        s[t] = in.units.data() + 2 * place;
    }
    alignas(32) uint32_t signs[8];
    alignas(32) float low[8], high[8];
    alignas(32) uint32_t index[64];
    auto group = [&](const uint8_t* block, size_t g, size_t u, __m256* acc){
        __m128i front, back;
        uint32_t bits = 0;
        if constexpr(TYPE == 18){
            const uint8_t* qs = block + 2 + 8 * g;
            front = _mm_insert_epi32(_mm_insert_epi32(_mm_insert_epi32(_mm_cvtsi32_si128((int)grid3xxs[qs[0]]), (int)grid3xxs[qs[1]], 1), (int)grid3xxs[qs[2]], 2), (int)grid3xxs[qs[3]], 3);
            back = _mm_insert_epi32(_mm_insert_epi32(_mm_insert_epi32(_mm_cvtsi32_si128((int)grid3xxs[qs[4]]), (int)grid3xxs[qs[5]], 1), (int)grid3xxs[qs[6]], 2), (int)grid3xxs[qs[7]], 3);
            bits = signs[g];
        } else if constexpr(TYPE == 22){
            const uint32_t* at4 = index + 4 * g;
            front = _mm_castpd_si128(_mm_loadh_pd(_mm_load_sd((const double*)(grid2s + at4[0])), (const double*)(grid2s + at4[1])));
            back = _mm_castpd_si128(_mm_loadh_pd(_mm_load_sd((const double*)(grid2s + at4[2])), (const double*)(grid2s + at4[3])));
            memcpy(&bits, block + 34 + 4 * g, 4);
        } else {
            const uint32_t* at8 = index + 8 * g;
            front = _mm_insert_epi32(_mm_insert_epi32(_mm_insert_epi32(_mm_cvtsi32_si128((int)grid3s[at8[0]]), (int)grid3s[at8[1]], 1), (int)grid3s[at8[2]], 2), (int)grid3s[at8[3]], 3);
            back = _mm_insert_epi32(_mm_insert_epi32(_mm_insert_epi32(_mm_cvtsi32_si128((int)grid3s[at8[4]]), (int)grid3s[at8[5]], 1), (int)grid3s[at8[6]], 2), (int)grid3s[at8[7]], 3);
            memcpy(&bits, block + 74 + 4 * g, 4);
        }
        __m256i mag = _mm256_inserti128_si256(_mm256_castsi128_si256(front), back, 1);
        __m256i sign = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32((int)bits), spread), pick), pick), unit);
        __m256 scale = TYPE == 22
            ? _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_broadcast_ss(low + g)), _mm_broadcast_ss(high + g), 1)
            : _mm256_broadcast_ss(low + g);
        for(int t = 0; t < COUNT; t++){
            const int8_t* v = x[t] + 64 * u;
            const float* k = s[t] + 2 * u;
            __m256i qv = _mm256_loadu_si256((const __m256i*)v);
            __m256 part = _mm256_mul_ps(_mm256_cvtepi32_ps(byteDots<VNNI>(mag, _mm256_sign_epi8(qv, sign))), _mm256_set1_ps(k[0]));
            if constexpr(RESIDUAL){
                __m256i rv = _mm256_loadu_si256((const __m256i*)(v + 32));
                part = _mm256_fmadd_ps(_mm256_cvtepi32_ps(byteDots<VNNI>(mag, _mm256_sign_epi8(rv, sign))), _mm256_set1_ps(k[1]), part);
            }
            acc[t] = _mm256_fmadd_ps(part, scale, acc[t]);
        }
    };
    size_t blocks = (size_t)(cols / 256);
    for(size_t b = 0; b < blocks; b++){
        const uint8_t* block = row + b * bytes;
        float base = half(block);
        if constexpr(TYPE == 18){
            __m256i aux = _mm256_loadu_si256((const __m256i*)(block + 66));
            __m256i s = _mm256_or_si256(_mm256_or_si256(_mm256_and_si256(aux, _mm256_set1_epi32(0x7F)), _mm256_and_si256(_mm256_slli_epi32(aux, 1), _mm256_set1_epi32(0x7F00))), _mm256_or_si256(_mm256_and_si256(_mm256_slli_epi32(aux, 2), _mm256_set1_epi32(0x7F0000)), _mm256_and_si256(_mm256_slli_epi32(aux, 3), _mm256_set1_epi32(0x7F000000))));
            __m256i p = _mm256_xor_si256(s, _mm256_srli_epi32(s, 4));
            p = _mm256_xor_si256(p, _mm256_srli_epi32(p, 2));
            p = _mm256_xor_si256(p, _mm256_srli_epi32(p, 1));
            _mm256_store_si256((__m256i*)signs, _mm256_or_si256(s, _mm256_slli_epi32(_mm256_and_si256(p, _mm256_set1_epi32(0x01010101)), 7)));
            __m256 level = _mm256_add_ps(_mm256_set1_ps(0.5f), _mm256_cvtepi32_ps(_mm256_srli_epi32(aux, 28)));
            _mm256_store_ps(low, _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(base), level), _mm256_set1_ps(0.5f)));
        } else if constexpr(TYPE == 22){
            __m256i top = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(block + 66)));
            for(int j = 0; j < 4; j++){
                __m256i codes = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(block + 2 + 8 * j)));
                __m256i lift = _mm256_permutevar8x32_epi32(top, _mm256_add_epi32(_mm256_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1), _mm256_set1_epi32(2 * j)));
                _mm256_store_si256((__m256i*)(index + 8 * j), _mm256_or_si256(codes, _mm256_and_si256(_mm256_sllv_epi32(lift, _mm256_setr_epi32(8, 6, 4, 2, 8, 6, 4, 2)), _mm256_set1_epi32(0x300))));
            }
            __m256i packed = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(block + 74)));
            __m256 dd = _mm256_set1_ps(base * 0.25f);
            _mm256_store_ps(low, _mm256_mul_ps(dd, _mm256_add_ps(_mm256_set1_ps(0.5f), _mm256_cvtepi32_ps(_mm256_and_si256(packed, _mm256_set1_epi32(15))))));
            _mm256_store_ps(high, _mm256_mul_ps(dd, _mm256_add_ps(_mm256_set1_ps(0.5f), _mm256_cvtepi32_ps(_mm256_srli_epi32(packed, 4)))));
        } else {
            __m256i top = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(block + 66)));
            for(int g = 0; g < 8; g++){
                __m256i codes = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(block + 2 + 8 * g)));
                __m256i lift = _mm256_permutevar8x32_epi32(top, _mm256_set1_epi32(g));
                _mm256_store_si256((__m256i*)(index + 8 * g), _mm256_or_si256(codes, _mm256_and_si256(_mm256_sllv_epi32(lift, _mm256_setr_epi32(8, 7, 6, 5, 4, 3, 2, 1)), _mm256_set1_epi32(256))));
            }
            uint32_t nibbles;
            memcpy(&nibbles, block + 106, 4);
            __m256i k = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32((int)nibbles), _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28)), _mm256_set1_epi32(15));
            _mm256_store_ps(low, _mm256_mul_ps(_mm256_set1_ps(base), _mm256_cvtepi32_ps(_mm256_or_si256(_mm256_slli_epi32(k, 1), _mm256_set1_epi32(1)))));
        }
        for(size_t g = 0; g < 8; g += 2){
            group(block, g, 8 * b + g, even);
            group(block, g + 1, 8 * b + g + 1, odd);
        }
    }
    for(int t = 0; t < COUNT; t++){
        __m256 acc = _mm256_add_ps(even[t], odd[t]);
        __m128 h = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
        h = _mm_add_ps(h, _mm_movehl_ps(h, h));
        h = _mm_add_ss(h, _mm_movehdup_ps(h));
        out[(size_t)t * outStep] = _mm_cvtss_f32(h);
    }
}

template<int TYPE, bool RESIDUAL, int COUNT, bool VNNI>
inline void dotsOf(const uint8_t* row, int cols, const HostPacked& in, size_t at, float* out, size_t outStep) {
    if constexpr(TYPE == 18 || TYPE == 21 || TYPE == 22) blockDots<TYPE, RESIDUAL, COUNT, VNNI>(row, cols, in, at, out, outStep);
    else rowDots<TYPE, RESIDUAL, COUNT, VNNI>(row, cols, in, at, out, outStep);
}

struct HostProduct {
    Matrix weight;
    const HostPacked* input = nullptr;
    int first = 0;
    float* output = nullptr;
    int count = 1;
};

template<int TYPE, bool RESIDUAL, bool VNNI>
inline void productRows(const HostProduct& job, int start, int end) {
    const Matrix& m = job.weight;
    for(int row = start; row < end; row++){
        const uint8_t* line = m.data + (uint64_t)row * m.stride;
        for(int t = 0; t < job.count; t += 8){
            int left = std::min(8, job.count - t);
            size_t at = (size_t)(job.first + t) * m.cols;
            float* out = job.output + (size_t)t * m.rows + row;
            switch(left){
                case 8: dotsOf<TYPE, RESIDUAL, 8, VNNI>(line, m.cols, *job.input, at, out, (size_t)m.rows); break;
                case 7: dotsOf<TYPE, RESIDUAL, 7, VNNI>(line, m.cols, *job.input, at, out, (size_t)m.rows); break;
                case 6: dotsOf<TYPE, RESIDUAL, 6, VNNI>(line, m.cols, *job.input, at, out, (size_t)m.rows); break;
                case 5: dotsOf<TYPE, RESIDUAL, 5, VNNI>(line, m.cols, *job.input, at, out, (size_t)m.rows); break;
                case 4: dotsOf<TYPE, RESIDUAL, 4, VNNI>(line, m.cols, *job.input, at, out, (size_t)m.rows); break;
                case 3: dotsOf<TYPE, RESIDUAL, 3, VNNI>(line, m.cols, *job.input, at, out, (size_t)m.rows); break;
                case 2: dotsOf<TYPE, RESIDUAL, 2, VNNI>(line, m.cols, *job.input, at, out, (size_t)m.rows); break;
                default: dotsOf<TYPE, RESIDUAL, 1, VNNI>(line, m.cols, *job.input, at, out, (size_t)m.rows); break;
            }
        }
    }
}

template<bool RESIDUAL, bool VNNI>
inline void productRowsOf(const HostProduct& job, int first, int end) {
    switch(job.weight.type){
        case 16: productRows<16, RESIDUAL, VNNI>(job, first, end); break;
        case 17: productRows<17, RESIDUAL, VNNI>(job, first, end); break;
        case 18: productRows<18, RESIDUAL, VNNI>(job, first, end); break;
        case 20: productRows<20, RESIDUAL, VNNI>(job, first, end); break;
        case 21: productRows<21, RESIDUAL, VNNI>(job, first, end); break;
        case 22: productRows<22, RESIDUAL, VNNI>(job, first, end); break;
        case 23: productRows<23, RESIDUAL, VNNI>(job, first, end); break;
        case 8: productRows<8, RESIDUAL, VNNI>(job, first, end); break;
        case 7: productRows<7, RESIDUAL, VNNI>(job, first, end); break;
        case 12: productRows<12, RESIDUAL, VNNI>(job, first, end); break;
        case 13: productRows<13, RESIDUAL, VNNI>(job, first, end); break;
        case 14: productRows<14, RESIDUAL, VNNI>(job, first, end); break;
        case 29: productRows<29, RESIDUAL, VNNI>(job, first, end); break;
        default: productRows<42, RESIDUAL, VNNI>(job, first, end); break;
    }
}

inline void hostMultiply(ThreadPool& pool, const std::vector<HostProduct>& list, bool residual = true) {
    std::vector<int> owner, start, span;
    double left = 0.0;
    for(const HostProduct& job : list) left += (double)job.weight.rows * job.weight.cols * (2 + job.count);
    double share = 2.0 * (double)(pool.threads.size() + 1);
    for(size_t p = 0; p < list.size(); p++){
        const Matrix& m = list[p].weight;
        int cols = std::max(1, m.cols);
        // los expertos van en pedazos chicos y las matrices grandes de un denso en pedazos de un millon de pesos
        // con el tope chico eran cientos de miles de tareas de una a seis filas por paso y la cpu leia la ram a menos de la mitad
        int most = std::max(1, std::min(m.rows, ((uint64_t)m.rows * cols >= (1ULL << 22) ? (1 << 20) : 32768) / cols));
        int least = std::max(1, std::min(most, 4096 / cols));
        double cost = (double)cols * (2 + list[p].count);
        for(int r = 0; r < m.rows;){
            int step = std::min(m.rows - r, std::max(least, std::min(most, (int)(left / share / cost))));
            owner.push_back((int)p);
            start.push_back(r);
            span.push_back(step);
            r += step;
            left -= step * cost;
        }
    }

    pool.run((int)owner.size(), [&](int chunk){
        const HostProduct& job = list[owner[chunk]];
        int first = start[chunk];
        int end = first + span[chunk];
        bool vnni = hostVnni();
        if(residual && vnni) productRowsOf<true, true>(job, first, end);
        else if(residual) productRowsOf<true, false>(job, first, end);
        else if(vnni) productRowsOf<false, true>(job, first, end);
        else productRowsOf<false, false>(job, first, end);
    });
}
