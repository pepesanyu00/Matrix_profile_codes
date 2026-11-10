// ==============================================================================
// Microbenchmark: SCAMP - x86 AVX2 con Reducción Horizontal Vectorizada (Ablación Fig. 9)
// Fichero (PH3 generado por tools/gen_vectorized_diag_loop.py) — origen: scamp_avx2_vectorized.cpp
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
#define SUB_PD(a, b) _mm256_sub_pd(a, b)
#define ADD_PD(a, b) _mm256_add_pd(a, b)
#define MUL_PD(a, b) _mm256_mul_pd(a, b)
#define CMP_PD(a, b, c) (VMTYPE) _mm256_cmp_pd(a, b, c)
#define MASKSTOREU_PD(a, b, c) _mm256_maskstore_pd(a, b, c)
#define MASKSTOREU_EPI(a, b, c) _mm256_maskstore_epi64((long long int*)(a), b, c)

// Reducción horizontal del vector de 4 doubles a 1 escalar (con saneado de colas NaN -> -Inf)
static inline double hmax_avx2(__m256d v)
{
    __m256d valid = _mm256_cmp_pd(v, v, _CMP_ORD_Q);
    v = _mm256_blendv_pd(_mm256_set1_pd(-__builtin_inf()), v, valid);
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d hi = _mm256_extractf128_pd(v, 1);
    __m128d m1 = _mm_max_pd(lo, hi);
    __m128d m2 = _mm_permute_pd(m1, 0x1);
    __m128d m3 = _mm_max_pd(m1, m2);
    return _mm_cvtsd_f64(m3);
}

extern "C" {

// MB3: bucle completo de la diagonal con reducción horizontal VECTORIZADA
__attribute__((noinline))
void scamp_diag_loop_avx2_vectorized(VDTYPE covariance_v,
                          const DTYPE* __restrict df,
                          const DTYPE* __restrict dg,
                          const DTYPE* __restrict norms,
                          DTYPE* __restrict profile_tmp,
                          ITYPE* __restrict profileIndex_tmp,
                          ITYPE diag,
                          ITYPE profileLength)
{
    ITYPE i = 1;
    for (ITYPE j = diag + 1; j < profileLength; j++)
    {

        VDTYPE dfj_v = LOADU_PD(&df[j - 1]);
        VDTYPE dgj_v = LOADU_PD(&dg[j - 1]);
        VDTYPE dfi_v = SET1_PD(df[i - 1]);
        VDTYPE dgi_v = SET1_PD(dg[i - 1]);
        covariance_v = ADD_PD(covariance_v, FMADD_PD(dfi_v, dgj_v, MUL_PD(dfj_v, dgi_v)));

        VDTYPE normsi_v = SET1_PD(norms[i]);
        VDTYPE normsj_v = LOADU_PD(&norms[j]);
        VDTYPE correlation_v = MUL_PD(MUL_PD(covariance_v, normsi_v), normsj_v);

        // Reducción horizontal vectorizada (árbol de shuffles)
        double corr_hmax = hmax_avx2(correlation_v);
        double cur_max = profile_tmp[i];
        double new_max = corr_hmax > cur_max ? corr_hmax : cur_max;

        __m256d eq_mask = _mm256_cmp_pd(correlation_v, _mm256_set1_pd(new_max), _CMP_EQ_OQ);
        int bitmask = _mm256_movemask_pd(eq_mask);
        if (bitmask != 0)
        {
            int idx = __builtin_ctz((unsigned)bitmask);
            profile_tmp[i] = new_max;
            profileIndex_tmp[i] = j + idx;
        }

        // Actualización de columna enmascarada
        VDTYPE profilej_v = LOADU_PD(&profile_tmp[j]);
        VMTYPE mask = CMP_PD(correlation_v, profilej_v, _CMP_GT_OQ);
        MASKSTOREU_PD(&profile_tmp[j], mask, correlation_v);
        MASKSTOREU_EPI(&profileIndex_tmp[j], mask, SET1_EPI(i));
        i++;
    }
}

}
