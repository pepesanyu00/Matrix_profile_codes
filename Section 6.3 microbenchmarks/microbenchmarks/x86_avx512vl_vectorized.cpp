// Proposed vectorized horizontal-max reduction for AVX512VL (256-bit vectors, native mask registers).
// Mirrors the existing #elif defined(AVX512VL) branch (VPACK=4) in scamp-v.cpp/scamp-v2.cpp.
// Godbolt flags: -O2 -mavx512vl -mfma -mbmi   (compiler: x86-64 gcc/clang, trunk; also test -O3)
// -mbmi is required for _tzcnt_u32; it is implied by -mavx2 and up on real hardware but not by the compiler flags alone.
#include <immintrin.h>

#define VPACK 4

// AVX512VL still lacks a native reduce-max instruction; only the mask/index step benefits from __mmask8.
static inline double hmax_avx512vl(__m256d v)
{
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d hi = _mm256_extractf128_pd(v, 1);
    __m128d m1 = _mm_max_pd(lo, hi);
    __m128d m2 = _mm_permute_pd(m1, 0x1);
    __m128d m3 = _mm_max_pd(m1, m2);
    return _mm_cvtsd_f64(m3);
}

extern "C" __attribute__((noinline))
void find_max_avx512vl_vectorized(const double* mem, double current_max, long current_index, long base_index,
                                   double* out_max, long* out_index)
{
    __m256d correlation_v = _mm256_loadu_pd(mem);

    double corr_hmax = hmax_avx512vl(correlation_v);
    double new_max = corr_hmax > current_max ? corr_hmax : current_max;

    __mmask8 mask = _mm256_cmp_pd_mask(correlation_v, _mm256_set1_pd(new_max), _CMP_EQ_OQ);

    if (mask != 0)
    {
        int idx = _tzcnt_u32((unsigned)mask); // no movemask needed: mask is already a native integer type
        *out_max = new_max;
        *out_index = base_index + idx;
    }
    else
    {
        *out_max = current_max;
        *out_index = current_index;
    }
}
