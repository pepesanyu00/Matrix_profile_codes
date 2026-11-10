// ==============================================================================
// Microbenchmark: SCRIMP - ARM SVE (V0 Original As-is, con corrección de Bug A9)
// Fichero: scrimp_arm_sve.cpp
//
// Procedencia: algoritmos/arm/scrimp-v.cpp
//
// Protocolo Godbolt (Compiler Explorer):
// - Compilador Principal: ARM64 gcc 14.2 (aarch64)
//   Flags: -O2 -march=armv8.2-a+sve
// - Compilador Secundario: ARM64 clang 18.1 (aarch64)
//   Flags: -O2 -march=armv8.2-a+sve
// ==============================================================================

#include <stdint.h>
#include <arm_sve.h>

#define DTYPE double
#define ITYPE uint64_t

#define VDTYPE svfloat64_t
#define VITYPE svuint64_t
#define VMTYPE svbool_t

#define SETZERO_PD() svdup_f64(0.0)

static inline svfloat64_t set1_pd(double a, svbool_t vl) {
    VDTYPE inactive = svdup_f64(0.0);
    return svdup_f64_m(inactive, vl, a);
}
#define SET1_PD(a, vl) set1_pd(a, vl)

static inline svuint64_t set1_epi(double a, svbool_t vl) {
    VITYPE inactive = svdup_u64(0.0);
    return svdup_u64_m(inactive, vl, (uint64_t)a);
}
#define SET1_EPI(a, vl) set1_epi(a, vl)

#define GETFIRST_PD(vec) svlastb_f64(svptrue_pat_b64(SV_VL1), vec)
#define REDMIN_PD(a, b, vl) svminv_f64(vl, svmin_f64_m(vl, a, svdup_f64(b)))

static inline uint64_t get_first_mask(svbool_t mask, svbool_t vl) {
    svbool_t first_true = svbrkb_z(vl, mask);
    ITYPE index = svcntp_b64(vl, first_true);
    if (index == svcntp_b64(vl, vl))
        return (uint64_t)-1;
    return index;
}
#define GETFIRST_MASK(mask, vl) get_first_mask(mask, vl)

#define LOADU_PD(a, vl) svld1_f64(vl, a)
#define LOADU_SI(a, vl) svld1_u64(vl, (const uint64_t*)(a))
#define STORE_PD(a, b, vl) svst1_f64(vl, a, b)
#define STORE_SI(a, b, vl) svst1_u64(vl, (uint64_t*)(a), b)

#define FMADD_PD(a, b, c, vl) svmla_f64_x(vl, c, a, b)
#define FMSUB_PD(a, b, c, vl) svsub_f64_x(vl, svmul_f64_x(vl, a, b), c)
#define SUB_PD(a, b, vl) svsub_f64_x(vl, a, b)
#define ADD_PD(a, b, vl) svadd_f64_x(vl, a, b)
#define MUL_PD(a, b, vl) svmul_f64_x(vl, a, b)
#define DIV_PD(a, b, vl) svdiv_f64_x(vl, a, b)

#define CMP_PD_LT(a, b, vl) svcmplt_f64(vl, a, b)
#define CMP_PD_EQ(a, b, vl) svcmpeq_f64(vl, a, svdup_f64(b))

#define BLEND_EPI(a, b, mask) svsel(mask, b, a)
#define BLEND_PD(a, b, mask) svsel(mask, b, a)

#define MASKSTOREU_PD(mask, a, b) svst1_f64(mask, a, b)
#define MASKSTOREU_EPI(mask, a, b) svst1_u64(mask, (uint64_t*)(a), b)

extern "C" {

// MB1: Bucle de ventana inicial para dot product en SVE O(w)
__attribute__((noinline))
VDTYPE scrimp_init_dot_sve(const DTYPE* __restrict tSeries,
                           ITYPE diag,
                           ITYPE windowSize,
                           svbool_t vlOuter)
{
    VDTYPE dotProd_v = SETZERO_PD();
    for (ITYPE j = diag; j < windowSize + diag; j++)
    {
        VDTYPE tSeriesj_v = LOADU_PD(&tSeries[j], vlOuter);
        VDTYPE tSeriesMinusDiag_v = SET1_PD(tSeries[j - diag], vlOuter);
        dotProd_v = FMADD_PD(tSeriesj_v, tSeriesMinusDiag_v, dotProd_v, vlOuter);
    }
    return dotProd_v;
}

// MB2: Un paso incremental de la diagonal caliente en SCRIMP SVE
__attribute__((noinline))
VDTYPE scrimp_diag_step_sve(VDTYPE dotProd_v,
                            const DTYPE* __restrict tSeries,
                            const DTYPE* __restrict means,
                            const DTYPE* __restrict devs,
                            DTYPE* __restrict profile_tmp,
                            ITYPE* __restrict profileIndex_tmp,
                            ITYPE i,
                            ITYPE j,
                            ITYPE windowSize,
                            svbool_t vlInner)
{
    VDTYPE tSeriesj0_v = LOADU_PD(&tSeries[j + windowSize - 1], vlInner);
    VDTYPE tSeriesj1_v = LOADU_PD(&tSeries[j - 1], vlInner);
    VDTYPE tSeriesi0_v = SET1_PD(tSeries[i + windowSize - 1], vlInner);
    VDTYPE tSeriesi1_v = SET1_PD(tSeries[i - 1], vlInner);

    dotProd_v = ADD_PD(FMSUB_PD(tSeriesj0_v, tSeriesi0_v, MUL_PD(tSeriesj1_v, tSeriesi1_v, vlInner), vlInner),
                       dotProd_v,
                       vlInner);

    VDTYPE meansj_v = LOADU_PD(&means[j], vlInner);
    VDTYPE devsj_v = LOADU_PD(&devs[j], vlInner);
    VDTYPE meansi_v = SET1_PD(means[i], vlInner);
    VDTYPE devsi_v = SET1_PD(devs[i], vlInner);
    VDTYPE windowSize_v = SET1_PD((double)windowSize, vlInner);

    VDTYPE triple_product_v = MUL_PD(windowSize_v, MUL_PD(meansi_v, meansj_v, vlInner), vlInner);
    VDTYPE prod_devs_v = MUL_PD(devsi_v, devsj_v, vlInner);
    VDTYPE division_v = DIV_PD(SUB_PD(dotProd_v, triple_product_v, vlInner), prod_devs_v, vlInner);
    VDTYPE distance_v = MUL_PD(SET1_PD(2.0, vlInner), SUB_PD(windowSize_v, division_v, vlInner), vlInner);

    // Búsqueda horizontal para mínimo
    DTYPE corr_max = REDMIN_PD(distance_v, profile_tmp[i], vlInner);
    VMTYPE mask = CMP_PD_EQ(distance_v, corr_max, vlInner);
    uint64_t index_max = GETFIRST_MASK(mask, vlInner);

    if (index_max != (uint64_t)-1)
    {
        profile_tmp[i] = corr_max;
        profileIndex_tmp[i] = j + index_max;
    }

    // Actualización de columna enmascarada
    VDTYPE profilej_v = LOADU_PD(&profile_tmp[j], vlInner);
    mask = CMP_PD_LT(distance_v, profilej_v, vlInner);
    MASKSTOREU_PD(mask, &profile_tmp[j], distance_v);
    MASKSTOREU_EPI(mask, &profileIndex_tmp[j], SET1_EPI(i, vlInner));

    return dotProd_v;
}

// MB3: Bucle completo de la diagonal caliente en SCRIMP SVE
__attribute__((noinline))
void scrimp_diag_loop_sve(VDTYPE dotProd_v,
                          const DTYPE* __restrict tSeries,
                          const DTYPE* __restrict means,
                          const DTYPE* __restrict devs,
                          DTYPE* __restrict profile_tmp,
                          ITYPE* __restrict profileIndex_tmp,
                          ITYPE diag,
                          ITYPE profileLength,
                          ITYPE windowSize)
{
    uint64_t VLMAX = svcntd();
    ITYPE i = 1;

    for (ITYPE j = diag + 1; j < profileLength; j++)
    {
        svbool_t vlInner = svwhilelt_b64((ITYPE)0, (profileLength - j < (ITYPE)VLMAX ? profileLength - j : (ITYPE)VLMAX));

        VDTYPE tSeriesj0_v = LOADU_PD(&tSeries[j + windowSize - 1], vlInner);
        VDTYPE tSeriesj1_v = LOADU_PD(&tSeries[j - 1], vlInner);
        VDTYPE tSeriesi0_v = SET1_PD(tSeries[i + windowSize - 1], vlInner);
        VDTYPE tSeriesi1_v = SET1_PD(tSeries[i - 1], vlInner);

        dotProd_v = ADD_PD(FMSUB_PD(tSeriesj0_v, tSeriesi0_v, MUL_PD(tSeriesj1_v, tSeriesi1_v, vlInner), vlInner),
                           dotProd_v,
                           vlInner);

        VDTYPE meansj_v = LOADU_PD(&means[j], vlInner);
        VDTYPE devsj_v = LOADU_PD(&devs[j], vlInner);
        VDTYPE meansi_v = SET1_PD(means[i], vlInner);
        VDTYPE devsi_v = SET1_PD(devs[i], vlInner);
        VDTYPE windowSize_v = SET1_PD((double)windowSize, vlInner);

        VDTYPE triple_product_v = MUL_PD(windowSize_v, MUL_PD(meansi_v, meansj_v, vlInner), vlInner);
        VDTYPE prod_devs_v = MUL_PD(devsi_v, devsj_v, vlInner);
        VDTYPE division_v = DIV_PD(SUB_PD(dotProd_v, triple_product_v, vlInner), prod_devs_v, vlInner);
        VDTYPE distance_v = MUL_PD(SET1_PD(2.0, vlInner), SUB_PD(windowSize_v, division_v, vlInner), vlInner);

        DTYPE corr_max = REDMIN_PD(distance_v, profile_tmp[i], vlInner);
        VMTYPE mask = CMP_PD_EQ(distance_v, corr_max, vlInner);
        uint64_t index_max = GETFIRST_MASK(mask, vlInner);

        if (index_max != (uint64_t)-1)
        {
            profile_tmp[i] = corr_max;
            profileIndex_tmp[i] = j + index_max;
        }

        VDTYPE profilej_v = LOADU_PD(&profile_tmp[j], vlInner);
        mask = CMP_PD_LT(distance_v, profilej_v, vlInner);
        MASKSTOREU_PD(mask, &profile_tmp[j], distance_v);
        MASKSTOREU_EPI(mask, &profileIndex_tmp[j], SET1_EPI(i, vlInner));

        i++;
    }
}

// MB4: Reducción inter-hilos (mínimo) SVE para un bloque de columnas
__attribute__((noinline))
void scrimp_thread_reduce_sve(const DTYPE* __restrict profile_tmp,
                              const ITYPE* __restrict profileIndex_tmp,
                              DTYPE* __restrict profile,
                              ITYPE* __restrict profileIndex,
                              ITYPE colum,
                              ITYPE numThreads,
                              ITYPE profileLength,
                              svbool_t vlRed)
{
    VDTYPE min_dist_v = SET1_PD(1.0 / 0.0, vlRed); // +infinity
    VITYPE min_indices_v = SET1_EPI(1.0, vlRed);

    for (ITYPE th = 0; th < numThreads; th++)
    {
        VDTYPE p_v = LOADU_PD(&profile_tmp[colum + (th * profileLength)], vlRed);
        VITYPE idx_v = LOADU_SI(&profileIndex_tmp[colum + (th * profileLength)], vlRed);
        VMTYPE mask = CMP_PD_LT(p_v, min_dist_v, vlRed);
        min_indices_v = BLEND_EPI(min_indices_v, idx_v, mask);
        min_dist_v = BLEND_PD(min_dist_v, p_v, mask);
    }

    STORE_PD(&profile[colum], min_dist_v, vlRed);
    STORE_SI(&profileIndex[colum], min_indices_v, vlRed);
}

}

