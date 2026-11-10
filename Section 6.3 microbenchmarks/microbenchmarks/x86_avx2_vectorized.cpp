// Proposed vectorized horizontal-max reduction for AVX2 (VPACK=4, 256-bit).
// Mirrors the ARM SVE REDMAX_PD/CMP_PD_EQ/GETFIRST_MASK pattern using AVX2 intrinsics.
// Godbolt flags: -O2 -mavx2 -mfma   (compiler: x86-64 gcc/clang, trunk; also test -O3)
#include <immintrin.h>

#define VPACK 4

// Horizontal max of a 256-bit vector of 4 doubles.
static inline double hmax_avx2(__m256d v)
{
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d hi = _mm256_extractf128_pd(v, 1);
    __m128d m1 = _mm_max_pd(lo, hi);      // max(v0,v2), max(v1,v3)
    __m128d m2 = _mm_permute_pd(m1, 0x1); // swap lanes
    __m128d m3 = _mm_max_pd(m1, m2);      // full max, broadcast in both lanes
    return _mm_cvtsd_f64(m3);
}

extern "C" __attribute__((noinline))
void find_max_avx2_vectorized(const double* mem, double current_max, long current_index, long base_index,
                               double* out_max, long* out_index)
{
    __m256d correlation_v = _mm256_loadu_pd(mem);

    double corr_hmax = hmax_avx2(correlation_v);
    double new_max = corr_hmax > current_max ? corr_hmax : current_max;

    __m256d eq_mask = _mm256_cmp_pd(correlation_v, _mm256_set1_pd(new_max), _CMP_EQ_OQ);
    int bitmask = _mm256_movemask_pd(eq_mask);

    if (bitmask != 0)
    {
        int idx = __builtin_ctz((unsigned)bitmask); // index of the winning lane
        *out_max = new_max;
        *out_index = base_index + idx;
    }
    else
    {
        // max came from current_max, not from this diagonal's vector
        *out_max = current_max;
        *out_index = current_index;
    }
}
