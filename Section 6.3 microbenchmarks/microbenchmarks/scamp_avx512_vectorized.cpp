// ==============================================================================
// Microbenchmark: SCAMP - x86 AVX-512 con Reducción Horizontal Vectorizada
//                 (Ablación Fig. 9 extendida a 512 bits)
// Fichero: scamp_avx512_vectorized.cpp
//
// Procedencia: scamp_avx512.cpp (V0, algoritmos/avx/scamp-v.cpp -DAVX512).
// Ablación de UNA sola variable: idéntico a la V0 salvo la búsqueda horizontal
// de la fila i, que sustituye el store a pila + bucle escalar de VPACK=8
// iteraciones por el mismo esquema que AVX2-VecRed / SVE / RVV:
//   reducción horizontal (max) sembrada con profile_tmp[i]
//   -> comparación de igualdad a __mmask8 -> máscara a índice (ctz)
//   -> store condicional.
// La actualización de columna (vcmppd k + stores enmascarados EVEX) no cambia.
//
// Contiene dos funciones MB2:
//   - scamp_diag_step_avx512_vectorized         : árbol de shuffles manual 8->4->2->1
//   - scamp_diag_step_avx512_vectorized_builtin : control con _mm512_reduce_max_pd
//     (sirve para comprobar que el intrínseco de Intel no oculta hardware de
//      reducción horizontal y genera la misma cadena de shuffles)
//
// Protocolo Godbolt (Compiler Explorer):
// - Compilador Principal: x86-64 gcc 14.2
//   Flags: -O2 -mavx512f -mfma
// - Compilador Secundario: x86-64 clang 18.1
//   Flags: -O2 -mavx512f -mfma
// ==============================================================================

#include <stdint.h>
#include <immintrin.h>

#define VPACK 8
#define DTYPE double
#define ITYPE int64_t

#define VDTYPE __m512d
#define VITYPE __m512i
#define VMTYPE __mmask8

#define SET1_PD(a) _mm512_set1_pd(a)
#define SET1_EPI(a) _mm512_set1_epi64(a)
#define LOADU_PD(a) _mm512_loadu_pd(a)
#define FMADD_PD(a, b, c) _mm512_fmadd_pd(a, b, c)
#define ADD_PD(a, b) _mm512_add_pd(a, b)
#define MUL_PD(a, b) _mm512_mul_pd(a, b)
#define CMP_PD(a, b, c) _mm512_cmp_pd_mask(a, b, c)
#define MASKSTOREU_PD(a, b, c) _mm512_mask_storeu_pd(a, b, c)
#define MASKSTOREU_EPI(a, b, c) _mm512_mask_storeu_epi64(a, b, c)

// Reducción horizontal de máximo para 8 doubles: árbol 8 -> 4 -> 2 -> 1 (con saneado de colas NaN -> -Inf)
static inline double hmax_avx512(__m512d v)
{
    __mmask8 valid = _mm512_cmp_pd_mask(v, v, _CMP_ORD_Q);
    v = _mm512_mask_blend_pd(valid, _mm512_set1_pd(-__builtin_inf()), v);
    __m256d lo4 = _mm512_castpd512_pd256(v);
    __m256d hi4 = _mm512_extractf64x4_pd(v, 1);
    __m256d m4  = _mm256_max_pd(lo4, hi4);
    __m128d lo2 = _mm256_castpd256_pd128(m4);
    __m128d hi2 = _mm256_extractf128_pd(m4, 1);
    __m128d m2  = _mm_max_pd(lo2, hi2);
    __m128d m1  = _mm_max_pd(m2, _mm_permute_pd(m2, 0x1));
    return _mm_cvtsd_f64(m1);
}

extern "C" {

// MB2: Paso incremental con reducción horizontal VECTORIZADA (árbol manual)
__attribute__((noinline))
VDTYPE scamp_diag_step_avx512_vectorized(VDTYPE covariance_v,
                                         const DTYPE* __restrict df,
                                         const DTYPE* __restrict dg,
                                         const DTYPE* __restrict norms,
                                         DTYPE* __restrict profile_tmp,
                                         ITYPE* __restrict profileIndex_tmp,
                                         ITYPE i,
                                         ITYPE j)
{
    VDTYPE dfj_v = LOADU_PD(&df[j - 1]);
    VDTYPE dgj_v = LOADU_PD(&dg[j - 1]);
    VDTYPE dfi_v = SET1_PD(df[i - 1]);
    VDTYPE dgi_v = SET1_PD(dg[i - 1]);
    covariance_v = ADD_PD(covariance_v, FMADD_PD(dfi_v, dgj_v, MUL_PD(dfj_v, dgi_v)));

    VDTYPE normsi_v = SET1_PD(norms[i]);
    VDTYPE normsj_v = LOADU_PD(&norms[j]);
    VDTYPE correlation_v = MUL_PD(MUL_PD(covariance_v, normsi_v), normsj_v);

    // Reducción horizontal vectorizada (árbol de shuffles 8->4->2->1)
    double corr_hmax = hmax_avx512(correlation_v);
    double cur_max = profile_tmp[i];
    double new_max = corr_hmax > cur_max ? corr_hmax : cur_max;

    VMTYPE eq_mask = CMP_PD(correlation_v, SET1_PD(new_max), _CMP_EQ_OQ);
    unsigned bitmask = (unsigned)eq_mask;
    if (bitmask != 0)
    {
        int idx = __builtin_ctz(bitmask);
        profile_tmp[i] = new_max;
        profileIndex_tmp[i] = j + idx;
    }

    // Actualización de columna enmascarada con __mmask8 (idéntica a V0)
    VDTYPE profilej_v = LOADU_PD(&profile_tmp[j]);
    VMTYPE mask = CMP_PD(correlation_v, profilej_v, _CMP_GT_OQ);
    MASKSTOREU_PD(&profile_tmp[j], mask, correlation_v);
    MASKSTOREU_EPI(&profileIndex_tmp[j], mask, SET1_EPI(i));

    return covariance_v;
}

// MB2 (control): igual que la anterior, pero con el intrínseco _mm512_reduce_max_pd
__attribute__((noinline))
VDTYPE scamp_diag_step_avx512_vectorized_builtin(VDTYPE covariance_v,
                                                 const DTYPE* __restrict df,
                                                 const DTYPE* __restrict dg,
                                                 const DTYPE* __restrict norms,
                                                 DTYPE* __restrict profile_tmp,
                                                 ITYPE* __restrict profileIndex_tmp,
                                                 ITYPE i,
                                                 ITYPE j)
{
    VDTYPE dfj_v = LOADU_PD(&df[j - 1]);
    VDTYPE dgj_v = LOADU_PD(&dg[j - 1]);
    VDTYPE dfi_v = SET1_PD(df[i - 1]);
    VDTYPE dgi_v = SET1_PD(dg[i - 1]);
    covariance_v = ADD_PD(covariance_v, FMADD_PD(dfi_v, dgj_v, MUL_PD(dfj_v, dgi_v)));

    VDTYPE normsi_v = SET1_PD(norms[i]);
    VDTYPE normsj_v = LOADU_PD(&norms[j]);
    VDTYPE correlation_v = MUL_PD(MUL_PD(covariance_v, normsi_v), normsj_v);

    // Reducción horizontal con el intrínseco de conveniencia de Intel (con saneado de colas NaN -> -Inf)
    __mmask8 valid = _mm512_cmp_pd_mask(correlation_v, correlation_v, _CMP_ORD_Q);
    VDTYPE corr_clean = _mm512_mask_blend_pd(valid, SET1_PD(-__builtin_inf()), correlation_v);
    double corr_hmax = _mm512_reduce_max_pd(corr_clean);
    double cur_max = profile_tmp[i];
    double new_max = corr_hmax > cur_max ? corr_hmax : cur_max;

    VMTYPE eq_mask = CMP_PD(correlation_v, SET1_PD(new_max), _CMP_EQ_OQ);
    unsigned bitmask = (unsigned)eq_mask;
    if (bitmask != 0)
    {
        int idx = __builtin_ctz(bitmask);
        profile_tmp[i] = new_max;
        profileIndex_tmp[i] = j + idx;
    }

    VDTYPE profilej_v = LOADU_PD(&profile_tmp[j]);
    VMTYPE mask = CMP_PD(correlation_v, profilej_v, _CMP_GT_OQ);
    MASKSTOREU_PD(&profile_tmp[j], mask, correlation_v);
    MASKSTOREU_EPI(&profileIndex_tmp[j], mask, SET1_EPI(i));

    return covariance_v;
}

}

