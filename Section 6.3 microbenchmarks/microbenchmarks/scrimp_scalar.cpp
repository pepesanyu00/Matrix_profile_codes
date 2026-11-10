// ==============================================================================
// Microbenchmark: SCRIMP - Baseline Escalar
// Fichero: scrimp_scalar.cpp
//
// Protocolo Godbolt (Compiler Explorer):
// - Compilador Principal: x86-64 gcc 14.2 (o aarch64 gcc / riscv64 gcc)
//   Flags: -O2 -fno-tree-vectorize
// - Compilador Secundario: x86-64 clang 18.1
//   Flags: -O2 -fno-vectorize -fno-slp-vectorize
// ==============================================================================

#include <stdint.h>
#include <math.h>

#define DTYPE double
#define ITYPE int64_t

extern "C" {

// MB1: Bucle de ventana inicial para dot product O(w)
__attribute__((noinline))
DTYPE scrimp_init_dot_scalar(const DTYPE* __restrict tSeries,
                             ITYPE diag,
                             ITYPE windowSize)
{
    DTYPE dotProd = 0.0;
    for (ITYPE j = diag; j < windowSize + diag; j++)
    {
        dotProd += tSeries[j] * tSeries[j - diag];
    }
    return dotProd;
}

// MB2: Un paso incremental de la diagonal caliente (SCRIMP)
__attribute__((noinline))
void scrimp_diag_step_scalar(DTYPE* __restrict dotProd,
                             const DTYPE* __restrict tSeries,
                             const DTYPE* __restrict means,
                             const DTYPE* __restrict devs,
                             DTYPE* __restrict profile_tmp,
                             ITYPE* __restrict profileIndex_tmp,
                             ITYPE i,
                             ITYPE j,
                             ITYPE windowSize)
{
    *dotProd += (tSeries[j + windowSize - 1] * tSeries[i + windowSize - 1]) - (tSeries[j - 1] * tSeries[i - 1]);
    DTYPE distance = 2.0 * (windowSize - (*dotProd - means[j] * means[i] * windowSize) / (devs[j] * devs[i]));

    // Actualización columna (j)
    if (distance < profile_tmp[j])
    {
        profile_tmp[j] = distance;
        profileIndex_tmp[j] = i;
    }

    // Actualización fila (i)
    if (distance < profile_tmp[i])
    {
        profile_tmp[i] = distance;
        profileIndex_tmp[i] = j;
    }
}

// MB3: Bucle completo de la diagonal caliente en SCRIMP
__attribute__((noinline))
void scrimp_diag_loop_scalar(DTYPE dotProd,
                             const DTYPE* __restrict tSeries,
                             const DTYPE* __restrict means,
                             const DTYPE* __restrict devs,
                             DTYPE* __restrict profile_tmp,
                             ITYPE* __restrict profileIndex_tmp,
                             ITYPE diag,
                             ITYPE profileLength,
                             ITYPE windowSize)
{
    ITYPE i = 1;
    for (ITYPE j = diag + 1; j < profileLength; j++)
    {
        dotProd += (tSeries[j + windowSize - 1] * tSeries[i + windowSize - 1]) - (tSeries[j - 1] * tSeries[i - 1]);
        DTYPE distance = 2.0 * (windowSize - (dotProd - means[j] * means[i] * windowSize) / (devs[j] * devs[i]));

        if (distance < profile_tmp[j])
        {
            profile_tmp[j] = distance;
            profileIndex_tmp[j] = i;
        }

        if (distance < profile_tmp[i])
        {
            profile_tmp[i] = distance;
            profileIndex_tmp[i] = j;
        }
        i++;
    }
}

// MB4: Reducción final entre hilos (mínimo) para SCRIMP
__attribute__((noinline))
void scrimp_thread_reduce_scalar(const DTYPE* __restrict profile_tmp,
                                 const ITYPE* __restrict profileIndex_tmp,
                                 DTYPE* __restrict profile,
                                 ITYPE* __restrict profileIndex,
                                 ITYPE colum,
                                 ITYPE numThreads,
                                 ITYPE profileLength)
{
    DTYPE min_distance = INFINITY;
    ITYPE min_index = 0;

    for (ITYPE th = 0; th < numThreads; th++)
    {
        ITYPE idx = colum + th * profileLength;
        if (profile_tmp[idx] < min_distance)
        {
            min_distance = profile_tmp[idx];
            min_index = profileIndex_tmp[idx];
        }
    }
    profile[colum] = min_distance;
    profileIndex[colum] = min_index;
}

}

