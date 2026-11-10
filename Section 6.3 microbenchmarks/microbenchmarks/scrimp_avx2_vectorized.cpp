// ==============================================================================
// Microbenchmark: SCRIMP - x86 AVX2 con Reducción Horizontal Vectorizada (Ablación Fig. 9)
// Fichero: scrimp_avx2_vectorized.cpp
//
// Protocolo Godbolt (Compiler Explorer):
// - Compilador Principal: x86-64 gcc 14.2 (o trunk)
//   Flags: -O2 -mavx2 -mfma
// - Compilador Secundario: x86-64 clang 18.1
//   Flags: -O2 -mavx2 -mfma
// ==============================================================================

#include <stdint.h>
#include <immintrin.h>

#define VPACK 4
#define DTYPE double
#define ITYPE int64_t

#define VDTYPE __m256d
#define VITYPE __m256i
#define VMTYPE VITYPE

#define SETZERO_PD() _mm256_setzero_pd()
#define SET1_PD(a) _mm256_set1_pd(a)
#define SET1_EPI(a) _mm256_set1_epi64x(a)
#define LOADU_PD(a) _mm256_loadu_pd(a)
#define STOREU_PD(a, b) _mm256_storeu_pd(a, b)

#define FMADD_PD(a, b, c) _mm256_fmadd_pd(a, b, c)
#define FMSUB_PD(a, b, c) _mm256_fmsub_pd(a, b, c)
#define SUB_PD(a, b) _mm256_sub_pd(a, b)
#define ADD_PD(a, b) _mm256_add_pd(a, b)
#define MUL_PD(a, b) _mm256_mul_pd(a, b)
#define DIV_PD(a, b) _mm256_div_pd(a, b)

#define CMP_PD(a, b, c) (VMTYPE) _mm256_cmp_pd(a, b, c)
#define MASKSTOREU_PD(a, b, c) _mm256_maskstore_pd(a, b, c)
#define MASKSTOREU_EPI(a, b, c) _mm256_maskstore_epi64((long long int*)(a), b, c)

// Reducción horizontal de mínimo para 4 doubles (con saneado de colas NaN -> +Inf)
static inline double hmin_avx2(__m256d v)
{
    __m256d valid = _mm256_cmp_pd(v, v, _CMP_ORD_Q);
    v = _mm256_blendv_pd(_mm256_set1_pd(__builtin_inf()), v, valid);
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d hi = _mm256_extractf128_pd(v, 1);
    __m128d m1 = _mm_min_pd(lo, hi);
    __m128d m2 = _mm_permute_pd(m1, 0x1);
    __m128d m3 = _mm_min_pd(m1, m2);
    return _mm_cvtsd_f64(m3);
}

extern "C" {

// MB2: Paso incremental con reducción horizontal VECTORIZADA de mínimo
__attribute__((noinline))
VDTYPE scrimp_diag_step_avx2_vectorized(VDTYPE dotProd_v,
                                        const DTYPE* __restrict tSeries,
                                        const DTYPE* __restrict means,
                                        const DTYPE* __restrict devs,
                                        DTYPE* __restrict profile_tmp,
                                        ITYPE* __restrict profileIndex_tmp,
                                        ITYPE i,
                                        ITYPE j,
                                        ITYPE windowSize)
{
    VDTYPE tSeriesj0_v = LOADU_PD(&tSeries[j + windowSize - 1]);
    VDTYPE tSeriesj1_v = LOADU_PD(&tSeries[j - 1]);
    VDTYPE tSeriesi0_v = SET1_PD(tSeries[i + windowSize - 1]);
    VDTYPE tSeriesi1_v = SET1_PD(tSeries[i - 1]);

    dotProd_v = ADD_PD(FMSUB_PD(tSeriesj0_v, tSeriesi0_v, MUL_PD(tSeriesj1_v, tSeriesi1_v)), dotProd_v);

    VDTYPE meansj_v = LOADU_PD(&means[j]);
    VDTYPE devsj_v = LOADU_PD(&devs[j]);
    VDTYPE meansi_v = SET1_PD(means[i]);
    VDTYPE devsi_v = SET1_PD(devs[i]);
    VDTYPE windowSize_v = SET1_PD((double)windowSize);

    VDTYPE triple_product_v = MUL_PD(windowSize_v, MUL_PD(meansi_v, meansj_v));
    VDTYPE prod_devs_v = MUL_PD(devsi_v, devsj_v);
    VDTYPE division_v = DIV_PD(SUB_PD(dotProd_v, triple_product_v), prod_devs_v);
    VDTYPE distance_v = MUL_PD(SET1_PD(2.0), SUB_PD(windowSize_v, division_v));

    // Reducción horizontal vectorizada de mínimo
    double dist_hmin = hmin_avx2(distance_v);
    double cur_min = profile_tmp[i];
    double new_min = dist_hmin < cur_min ? dist_hmin : cur_min;

    __m256d eq_mask = _mm256_cmp_pd(distance_v, _mm256_set1_pd(new_min), _CMP_EQ_OQ);
    int bitmask = _mm256_movemask_pd(eq_mask);
    if (bitmask != 0)
    {
        int idx = __builtin_ctz((unsigned)bitmask);
        profile_tmp[i] = new_min;
        profileIndex_tmp[i] = j + idx;
    }

    // Actualización de columna enmascarada
    VDTYPE profilej_v = LOADU_PD(&profile_tmp[j]);
    VMTYPE mask = CMP_PD(distance_v, profilej_v, _CMP_LT_OQ);
    MASKSTOREU_PD(&profile_tmp[j], mask, distance_v);
    MASKSTOREU_EPI(&profileIndex_tmp[j], mask, SET1_EPI(i));

    return dotProd_v;
}

}

