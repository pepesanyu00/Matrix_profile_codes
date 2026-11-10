// ==============================================================================
// Microbenchmark: SCRIMP - x86 AVX-512 con Reducción Horizontal Vectorizada
//                 (Ablación Fig. 9 extendida a 512 bits)
// Fichero: scrimp_avx512_vectorized.cpp
//
// Procedencia: scrimp_avx512.cpp (V0, algoritmos/avx/scrimp-v.cpp -DAVX512).
// Ablación de UNA sola variable: idéntico a la V0 salvo la búsqueda horizontal
// de la fila i, que sustituye el store a pila + bucle escalar de VPACK=8
// iteraciones por el mismo esquema que AVX2-VecRed / SVE / RVV:
//   reducción horizontal (min) sembrada con profile_tmp[i]
//   -> comparación de igualdad a __mmask8 -> máscara a índice (ctz)
//   -> store condicional.
// La actualización de columna (vcmppd k + stores enmascarados EVEX) no cambia.
//
// Contiene dos funciones MB2:
//   - scrimp_diag_step_avx512_vectorized         : árbol de shuffles manual 8->4->2->1
//   - scrimp_diag_step_avx512_vectorized_builtin : control con _mm512_reduce_min_pd
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
#define FMSUB_PD(a, b, c) _mm512_fmsub_pd(a, b, c)
#define SUB_PD(a, b) _mm512_sub_pd(a, b)
#define ADD_PD(a, b) _mm512_add_pd(a, b)
#define MUL_PD(a, b) _mm512_mul_pd(a, b)
#define DIV_PD(a, b) _mm512_div_pd(a, b)
#define CMP_PD(a, b, c) _mm512_cmp_pd_mask(a, b, c)
#define MASKSTOREU_PD(a, b, c) _mm512_mask_storeu_pd(a, b, c)
#define MASKSTOREU_EPI(a, b, c) _mm512_mask_storeu_epi64(a, b, c)

// Reducción horizontal de mínimo para 8 doubles: árbol 8 -> 4 -> 2 -> 1 (con saneado de colas NaN -> +Inf)
static inline double hmin_avx512(__m512d v)
{
    __mmask8 valid = _mm512_cmp_pd_mask(v, v, _CMP_ORD_Q);
    v = _mm512_mask_blend_pd(valid, _mm512_set1_pd(__builtin_inf()), v);
    __m256d lo4 = _mm512_castpd512_pd256(v);
    __m256d hi4 = _mm512_extractf64x4_pd(v, 1);
    __m256d m4  = _mm256_min_pd(lo4, hi4);
    __m128d lo2 = _mm256_castpd256_pd128(m4);
    __m128d hi2 = _mm256_extractf128_pd(m4, 1);
    __m128d m2  = _mm_min_pd(lo2, hi2);
    __m128d m1  = _mm_min_pd(m2, _mm_permute_pd(m2, 0x1));
    return _mm_cvtsd_f64(m1);
}

extern "C" {

// MB2: Paso incremental con reducción horizontal VECTORIZADA de mínimo (árbol manual)
__attribute__((noinline))
VDTYPE scrimp_diag_step_avx512_vectorized(VDTYPE dotProd_v,
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

    // Reducción horizontal vectorizada de mínimo (árbol de shuffles 8->4->2->1)
    double dist_hmin = hmin_avx512(distance_v);
    double cur_min = profile_tmp[i];
    double new_min = dist_hmin < cur_min ? dist_hmin : cur_min;

    VMTYPE eq_mask = CMP_PD(distance_v, SET1_PD(new_min), _CMP_EQ_OQ);
    unsigned bitmask = (unsigned)eq_mask;
    if (bitmask != 0)
    {
        int idx = __builtin_ctz(bitmask);
        profile_tmp[i] = new_min;
        profileIndex_tmp[i] = j + idx;
    }

    // Actualización de columna enmascarada con __mmask8 (idéntica a V0)
    VDTYPE profilej_v = LOADU_PD(&profile_tmp[j]);
    VMTYPE mask = CMP_PD(distance_v, profilej_v, _CMP_LT_OQ);
    MASKSTOREU_PD(&profile_tmp[j], mask, distance_v);
    MASKSTOREU_EPI(&profileIndex_tmp[j], mask, SET1_EPI(i));

    return dotProd_v;
}

// MB2 (control): igual que la anterior, pero con el intrínseco _mm512_reduce_min_pd
__attribute__((noinline))
VDTYPE scrimp_diag_step_avx512_vectorized_builtin(VDTYPE dotProd_v,
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

    // Reducción horizontal con el intrínseco de conveniencia de Intel (con saneado de colas NaN -> +Inf)
    __mmask8 valid = _mm512_cmp_pd_mask(distance_v, distance_v, _CMP_ORD_Q);
    VDTYPE dist_clean = _mm512_mask_blend_pd(valid, SET1_PD(__builtin_inf()), distance_v);
    double dist_hmin = _mm512_reduce_min_pd(dist_clean);
    double cur_min = profile_tmp[i];
    double new_min = dist_hmin < cur_min ? dist_hmin : cur_min;

    VMTYPE eq_mask = CMP_PD(distance_v, SET1_PD(new_min), _CMP_EQ_OQ);
    unsigned bitmask = (unsigned)eq_mask;
    if (bitmask != 0)
    {
        int idx = __builtin_ctz(bitmask);
        profile_tmp[i] = new_min;
        profileIndex_tmp[i] = j + idx;
    }

    VDTYPE profilej_v = LOADU_PD(&profile_tmp[j]);
    VMTYPE mask = CMP_PD(distance_v, profilej_v, _CMP_LT_OQ);
    MASKSTOREU_PD(&profile_tmp[j], mask, distance_v);
    MASKSTOREU_EPI(&profileIndex_tmp[j], mask, SET1_EPI(i));

    return dotProd_v;
}

}

