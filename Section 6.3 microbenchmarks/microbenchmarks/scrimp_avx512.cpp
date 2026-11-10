// ==============================================================================
// Microbenchmark: SCRIMP - x86 AVX-512 (Baseline Packed-SIMD V0 Original)
// Fichero: scrimp_avx512.cpp
//
// Procedencia: algoritmos/avx/scrimp-v.cpp (-DAVX512)
// Incluye búsqueda horizontal secuencial con store a stack y bucle escalar (VPACK=8).
//
// Protocolo Godbolt (Compiler Explorer):
// - Compilador Principal: x86-64 gcc 14.2 (o trunk)
//   Flags: -O2 -mavx512f -mfma
// - Compilador Secundario: x86-64 clang 18.1
//   Flags: -O2 -mavx512f -mfma
// ==============================================================================

#include <stdint.h>
#include <immintrin.h>

#define ALIGN 64
#define VPACK 8
#define DTYPE double
#define ITYPE int64_t

#define VDTYPE __m512d
#define VITYPE __m512i
#define VMTYPE __mmask8

#define SETZERO_PD() _mm512_setzero_pd()
#define SET1_PD(a) _mm512_set1_pd(a)
#define SET1_EPI(a) _mm512_set1_epi64(a)
#define LOADU_PD(a) _mm512_loadu_pd(a)
#define LOADU_SI(a) _mm512_loadu_si512((const void*)(a))
#define STORE_PD(a, b) _mm512_store_pd(a, b)
#define STOREU_PD(a, b) _mm512_storeu_pd(a, b)
#define STOREU_SI(a, b) _mm512_storeu_si512((void*)(a), b)

#define FMADD_PD(a, b, c) _mm512_fmadd_pd(a, b, c)
#define FMSUB_PD(a, b, c) _mm512_fmsub_pd(a, b, c)
#define SUB_PD(a, b) _mm512_sub_pd(a, b)
#define ADD_PD(a, b) _mm512_add_pd(a, b)
#define MUL_PD(a, b) _mm512_mul_pd(a, b)
#define DIV_PD(a, b) _mm512_div_pd(a, b)

#define CMP_PD(a, b, c) _mm512_cmp_pd_mask(a, b, c)
#define BLEND_EPI(a, b, c) _mm512_mask_blend_epi64(c, a, b)
#define BLEND_PD(a, b, c) _mm512_mask_blend_pd(c, a, b)
#define MASKSTOREU_PD(a, b, c) _mm512_mask_storeu_pd(a, b, c)
#define MASKSTOREU_EPI(a, b, c) _mm512_mask_storeu_epi64(a, b, c)

extern "C" {

// MB1: Bucle de ventana inicial para dot product O(w)
__attribute__((noinline))
VDTYPE scrimp_init_dot_avx512(const DTYPE* __restrict tSeries,
                              ITYPE diag,
                              ITYPE windowSize)
{
    VDTYPE dotProd_v = SETZERO_PD();
    for (ITYPE j = diag; j < windowSize + diag; j++)
    {
        VDTYPE tSeriesj_v = LOADU_PD(&tSeries[j]);
        VDTYPE tSeriesMinusDiag_v = SET1_PD(tSeries[j - diag]);
        dotProd_v = FMADD_PD(tSeriesj_v, tSeriesMinusDiag_v, dotProd_v);
    }
    return dotProd_v;
}

// MB2: Un paso incremental de la diagonal caliente en SCRIMP AVX-512
__attribute__((noinline))
VDTYPE scrimp_diag_step_avx512(VDTYPE dotProd_v,
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

    // Búsqueda horizontal secuencial original para fila (i)
    alignas(ALIGN) DTYPE distance[VPACK];
    STORE_PD(distance, distance_v);
    for (int ii = 0; ii < VPACK; ii++)
    {
        if (distance[ii] < profile_tmp[i])
        {
            profile_tmp[i] = distance[ii];
            profileIndex_tmp[i] = j + ii;
        }
    }

    // Actualización de columna enmascarada con __mmask8
    VDTYPE profilej_v = LOADU_PD(&profile_tmp[j]);
    VMTYPE mask = CMP_PD(distance_v, profilej_v, _CMP_LT_OQ);
    MASKSTOREU_PD(&profile_tmp[j], mask, distance_v);
    MASKSTOREU_EPI(&profileIndex_tmp[j], mask, SET1_EPI(i));

    return dotProd_v;
}

// MB3: Bucle completo de la diagonal caliente en SCRIMP AVX-512
__attribute__((noinline))
void scrimp_diag_loop_avx512(VDTYPE dotProd_v,
                             const DTYPE* __restrict tSeries,
                             const DTYPE* __restrict means,
                             const DTYPE* __restrict devs,
                             DTYPE* __restrict profile_tmp,
                             ITYPE* __restrict profileIndex_tmp,
                             ITYPE diag,
                             ITYPE profileLength,
                             ITYPE windowSize)
{
    VDTYPE windowSize_v = SET1_PD((double)windowSize);
    ITYPE i = 1;

    for (ITYPE j = diag + 1; j < profileLength; j++)
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

        VDTYPE triple_product_v = MUL_PD(windowSize_v, MUL_PD(meansi_v, meansj_v));
        VDTYPE prod_devs_v = MUL_PD(devsi_v, devsj_v);
        VDTYPE division_v = DIV_PD(SUB_PD(dotProd_v, triple_product_v), prod_devs_v);
        VDTYPE distance_v = MUL_PD(SET1_PD(2.0), SUB_PD(windowSize_v, division_v));

        alignas(ALIGN) DTYPE distance[VPACK];
        STORE_PD(distance, distance_v);
        for (int ii = 0; ii < VPACK; ii++)
        {
            if (distance[ii] < profile_tmp[i])
            {
                profile_tmp[i] = distance[ii];
                profileIndex_tmp[i] = j + ii;
            }
        }

        VDTYPE profilej_v = LOADU_PD(&profile_tmp[j]);
        VMTYPE mask = CMP_PD(distance_v, profilej_v, _CMP_LT_OQ);
        MASKSTOREU_PD(&profile_tmp[j], mask, distance_v);
        MASKSTOREU_EPI(&profileIndex_tmp[j], mask, SET1_EPI(i));

        i++;
    }
}

// MB4: Reducción inter-hilos (mínimo) vectorizada para SCRIMP AVX-512
__attribute__((noinline))
void scrimp_thread_reduce_avx512(const DTYPE* __restrict profile_tmp,
                                 const ITYPE* __restrict profileIndex_tmp,
                                 DTYPE* __restrict profile,
                                 ITYPE* __restrict profileIndex,
                                 ITYPE colum,
                                 ITYPE numThreads,
                                 ITYPE profileLength)
{
    VDTYPE min_dist_v = SET1_PD(1.0 / 0.0); // +infinity
    VITYPE min_indices_v = SET1_EPI(1);

    for (ITYPE th = 0; th < numThreads; th++)
    {
        VDTYPE p_v = LOADU_PD(&profile_tmp[colum + (th * profileLength)]);
        VITYPE idx_v = LOADU_SI(&profileIndex_tmp[colum + (th * profileLength)]);
        VMTYPE mask = CMP_PD(p_v, min_dist_v, _CMP_LT_OQ);
        min_indices_v = BLEND_EPI(min_indices_v, idx_v, mask);
        min_dist_v = BLEND_PD(min_dist_v, p_v, mask);
    }

    STOREU_PD(&profile[colum], min_dist_v);
    STOREU_SI(&profileIndex[colum], min_indices_v);
}

}

