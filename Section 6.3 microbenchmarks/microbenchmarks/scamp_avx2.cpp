// ==============================================================================
// Microbenchmark: SCAMP - x86 AVX2 (Baseline Packed-SIMD V0 Original)
// Fichero: scamp_avx2.cpp
//
// Procedencia: algoritmos/avx/scamp-v.cpp (-DAVX2)
// Incluye búsqueda horizontal secuencial con store a stack y bucle escalar (VPACK=4).
//
// Protocolo Godbolt (Compiler Explorer):
// - Compilador Principal: x86-64 gcc 14.2 (o trunk)
//   Flags: -O2 -mavx2 -mfma
// - Compilador Secundario: x86-64 clang 18.1
//   Flags: -O2 -mavx2 -mfma
// ==============================================================================

#include <stdint.h>
#include <immintrin.h>

#define ALIGN 32
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
#define LOADU_SI(a) _mm256_loadu_si256((const __m256i*)(a))
#define STORE_PD(a, b) _mm256_store_pd(a, b)
#define STOREU_PD(a, b) _mm256_storeu_pd(a, b)
#define STOREU_SI(a, b) _mm256_storeu_si256((__m256i*)(a), b)
#define FMADD_PD(a, b, c) _mm256_fmadd_pd(a, b, c)
#define SUB_PD(a, b) _mm256_sub_pd(a, b)
#define ADD_PD(a, b) _mm256_add_pd(a, b)
#define MUL_PD(a, b) _mm256_mul_pd(a, b)
#define CMP_PD(a, b, c) (VMTYPE) _mm256_cmp_pd(a, b, c)
#define BLEND_EPI(a, b, c) _mm256_blendv_epi8(a, b, c)
#define BLEND_PD(a, b, c) _mm256_blendv_pd(a, b, (VDTYPE)c)
#define MASKSTOREU_PD(a, b, c) _mm256_maskstore_pd(a, b, c)
#define MASKSTOREU_EPI(a, b, c) _mm256_maskstore_epi64((long long int*)(a), b, c)

extern "C" {

// MB1: Bucle de ventana inicial para covarianza (vectorizado en bloques de VPACK)
__attribute__((noinline))
VDTYPE scamp_init_covariance_avx2(const DTYPE* __restrict tSeries,
                                  const DTYPE* __restrict means,
                                  ITYPE diag,
                                  ITYPE windowSize)
{
    VDTYPE covariance_v = SETZERO_PD();
    VDTYPE meansWinDiag_v = LOADU_PD(&means[diag]);
    VDTYPE meansWin0_v = SET1_PD(means[0]);

    for (ITYPE i = 0; i < windowSize; i++)
    {
        VDTYPE tSeriesWinDiag_v = LOADU_PD(&tSeries[diag + i]);
        VDTYPE tSeriesWin0_v = SET1_PD(tSeries[i]);
        covariance_v = FMADD_PD(SUB_PD(tSeriesWinDiag_v, meansWinDiag_v),
                                SUB_PD(tSeriesWin0_v, meansWin0_v),
                                covariance_v);
    }
    return covariance_v;
}

// MB2: Un paso incremental de la diagonal (actualización incremental + búsqueda horiz + masked store)
__attribute__((noinline))
VDTYPE scamp_diag_step_avx2(VDTYPE covariance_v,
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

    // Búsqueda horizontal secuencial original (almacén en stack + bucle escalar)
    alignas(ALIGN) DTYPE correlation[VPACK];
    STORE_PD(correlation, correlation_v);
    for (int ii = 0; ii < VPACK; ii++)
    {
        if (correlation[ii] > profile_tmp[i])
        {
            profile_tmp[i] = correlation[ii];
            profileIndex_tmp[i] = j + ii;
        }
    }

    // Actualización de columna enmascarada
    VDTYPE profilej_v = LOADU_PD(&profile_tmp[j]);
    VMTYPE mask = CMP_PD(correlation_v, profilej_v, _CMP_GT_OQ);
    MASKSTOREU_PD(&profile_tmp[j], mask, correlation_v);
    MASKSTOREU_EPI(&profileIndex_tmp[j], mask, SET1_EPI(i));

    return covariance_v;
}

// MB3: Bucle de la diagonal caliente completo
__attribute__((noinline))
void scamp_diag_loop_avx2(VDTYPE covariance_v,
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

        alignas(ALIGN) DTYPE correlation[VPACK];
        STORE_PD(correlation, correlation_v);
        for (int ii = 0; ii < VPACK; ii++)
        {
            if (correlation[ii] > profile_tmp[i])
            {
                profile_tmp[i] = correlation[ii];
                profileIndex_tmp[i] = j + ii;
            }
        }

        VDTYPE profilej_v = LOADU_PD(&profile_tmp[j]);
        VMTYPE mask = CMP_PD(correlation_v, profilej_v, _CMP_GT_OQ);
        MASKSTOREU_PD(&profile_tmp[j], mask, correlation_v);
        MASKSTOREU_EPI(&profileIndex_tmp[j], mask, SET1_EPI(i));

        i++;
    }
}

// MB4: Reducción inter-hilos vectorizada para un bloque de columnas
__attribute__((noinline))
void scamp_thread_reduce_avx2(const DTYPE* __restrict profile_tmp,
                              const ITYPE* __restrict profileIndex_tmp,
                              DTYPE* __restrict profile,
                              ITYPE* __restrict profileIndex,
                              ITYPE colum,
                              ITYPE numThreads,
                              ITYPE profileLength)
{
    VDTYPE max_corr_v = SET1_PD(-1.0 / 0.0); // -infinity
    VITYPE max_indices_v = SET1_EPI(-1);

    for (ITYPE th = 0; th < numThreads; th++)
    {
        VDTYPE p_v = LOADU_PD(&profile_tmp[colum + (th * profileLength)]);
        VITYPE idx_v = LOADU_SI(&profileIndex_tmp[colum + (th * profileLength)]);
        VMTYPE mask = CMP_PD(p_v, max_corr_v, _CMP_GT_OQ);
        max_indices_v = BLEND_EPI(max_indices_v, idx_v, mask);
        max_corr_v = BLEND_PD(max_corr_v, p_v, mask);
    }

    STOREU_PD(&profile[colum], max_corr_v);
    STOREU_SI(&profileIndex[colum], max_indices_v);
}

}

