// Proposed vectorized horizontal-max reduction for AVX512F (512-bit vectors, VPACK=8).
// Includes two reduction variants (compiler-builtin vs manual shuffle chain) since codegen
// for _mm512_reduce_max_pd varies across compilers/versions - compare both in Godbolt.
// Godbolt flags: -O2 -mavx512f -mfma -mbmi   (compiler: x86-64 gcc/clang, trunk; also test -O3)
// -mbmi is required for _tzcnt_u32; it is implied by -mavx2 and up on real hardware but not by the compiler flags alone.
#include <immintrin.h>

#define VPACK 8

extern "C" __attribute__((noinline))
void find_max_avx512f_vectorized(const double* mem, double current_max, long current_index, long base_index,
                                  double* out_max, long* out_index)
{
    __m512d correlation_v = _mm512_loadu_pd(mem);

    double corr_hmax = _mm512_reduce_max_pd(correlation_v);
    double new_max = corr_hmax > current_max ? corr_hmax : current_max;

    __mmask8 mask = _mm512_cmp_pd_mask(correlation_v, _mm512_set1_pd(new_max), _CMP_EQ_OQ);

    if (mask != 0)
    {
        int idx = _tzcnt_u32((unsigned)mask);
        *out_max = new_max;
        *out_index = base_index + idx;
    }
    else
    {
        *out_max = current_max;
        *out_index = current_index;
    }
}

// Manual reduction chain (8 -> 4 -> 2 -> 1), alternative to the compiler-builtin reduce above.
static inline double hmax_avx512f_manual(__m512d v)
{
    __m256d lo = _mm512_castpd512_pd256(v);
    __m256d hi = _mm512_extractf64x4_pd(v, 1);
    __m256d m1 = _mm256_max_pd(lo, hi); // reduce 8 -> 4

    __m128d lo2 = _mm256_castpd256_pd128(m1);
    __m128d hi2 = _mm256_extractf128_pd(m1, 1);
    __m128d m2 = _mm_max_pd(lo2, hi2);  // reduce 4 -> 2

    __m128d m3 = _mm_permute_pd(m2, 0x1);
    __m128d m4 = _mm_max_pd(m2, m3);    // reduce 2 -> 1
    return _mm_cvtsd_f64(m4);
}

extern "C" __attribute__((noinline))
void find_max_avx512f_vectorized_manual(const double* mem, double current_max, long current_index, long base_index,
                                         double* out_max, long* out_index)
{
    __m512d correlation_v = _mm512_loadu_pd(mem);

    double corr_hmax = hmax_avx512f_manual(correlation_v);
    double new_max = corr_hmax > current_max ? corr_hmax : current_max;

    __mmask8 mask = _mm512_cmp_pd_mask(correlation_v, _mm512_set1_pd(new_max), _CMP_EQ_OQ);

    if (mask != 0)
    {
        int idx = _tzcnt_u32((unsigned)mask);
        *out_max = new_max;
        *out_index = base_index + idx;
    }
    else
    {
        *out_max = current_max;
        *out_index = current_index;
    }
}
