// ==============================================================================
// Microbenchmark: SCAMP - Baseline Escalar
// Fichero: scamp_scalar.cpp
//
// Protocolo Godbolt (Compiler Explorer):
// - Compilador Principal: x86-64 gcc 14.2 (o aarch64 gcc / riscv64 gcc)
//   Flags: -O2 -fno-tree-vectorize
// - Compilador Secundario: x86-64 clang 18.1 (o clang aarch64 / riscv64)
//   Flags: -O2 -fno-vectorize -fno-slp-vectorize
// ==============================================================================

#include <stdint.h>
#include <math.h>

#define DTYPE double
#define ITYPE int64_t

extern "C" {

// MB1: Bucle de ventana inicial para covarianza O(w)
__attribute__((noinline))
DTYPE scamp_init_covariance_scalar(const DTYPE* __restrict tSeries,
                                   const DTYPE* __restrict means,
                                   ITYPE diag,
                                   ITYPE windowSize)
{
    DTYPE covariance = 0.0;
    DTYPE mean_diag = means[diag];
    DTYPE mean_0 = means[0];

    for (ITYPE i = 0; i < windowSize; i++)
    {
        covariance += (tSeries[diag + i] - mean_diag) * (tSeries[i] - mean_0);
    }
    return covariance;
}

// MB2: Una iteración (paso elemental) de la diagonal caliente
__attribute__((noinline))
void scamp_diag_step_scalar(DTYPE* __restrict covariance,
                            const DTYPE* __restrict df,
                            const DTYPE* __restrict dg,
                            const DTYPE* __restrict norms,
                            DTYPE* __restrict profile_tmp,
                            ITYPE* __restrict profileIndex_tmp,
                            ITYPE i,
                            ITYPE j)
{
    *covariance += (df[i - 1] * dg[j - 1] + df[j - 1] * dg[i - 1]);
    DTYPE correlation = (*covariance) * norms[i] * norms[j];

    // Actualización horizontal fila (i)
    if (correlation > profile_tmp[i])
    {
        profile_tmp[i] = correlation;
        profileIndex_tmp[i] = j;
    }

    // Actualización columna (j)
    if (correlation > profile_tmp[j])
    {
        profile_tmp[j] = correlation;
        profileIndex_tmp[j] = i;
    }
}

// MB3: Bucle completo de la diagonal caliente
__attribute__((noinline))
void scamp_diag_loop_scalar(DTYPE covariance,
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
        covariance += (df[i - 1] * dg[j - 1] + df[j - 1] * dg[i - 1]);
        DTYPE correlation = covariance * norms[i] * norms[j];

        if (correlation > profile_tmp[i])
        {
            profile_tmp[i] = correlation;
            profileIndex_tmp[i] = j;
        }

        if (correlation > profile_tmp[j])
        {
            profile_tmp[j] = correlation;
            profileIndex_tmp[j] = i;
        }
        i++;
    }
}

// MB4: Reducción final entre hilos para una columna/rango
__attribute__((noinline))
void scamp_thread_reduce_scalar(const DTYPE* __restrict profile_tmp,
                                const ITYPE* __restrict profileIndex_tmp,
                                DTYPE* __restrict profile,
                                ITYPE* __restrict profileIndex,
                                ITYPE colum,
                                ITYPE numThreads,
                                ITYPE profileLength)
{
    DTYPE max_corr = -INFINITY;
    ITYPE max_index = 0;

    for (ITYPE th = 0; th < numThreads; th++)
    {
        ITYPE idx = colum + th * profileLength;
        if (profile_tmp[idx] > max_corr)
        {
            max_corr = profile_tmp[idx];
            max_index = profileIndex_tmp[idx];
        }
    }
    profile[colum] = max_corr;
    profileIndex[colum] = max_index;
}

}

