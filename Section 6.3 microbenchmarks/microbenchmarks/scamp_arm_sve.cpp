// ==============================================================================
// Microbenchmark: SCAMP - ARM SVE (V0 Original As-is)
// Fichero: scamp_arm_sve.cpp
//
// Procedencia: algoritmos/arm/scamp-v.cpp
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
#define REDMAX_PD(a, b, vl) svmaxv_f64(vl, svmax_f64_m(vl, a, svdup_f64(b)))

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
#define ADD_PD(a, b, vl) svadd_f64_x(vl, a, b)
#define SUB_PD(a, b, vl) svsub_f64_x(vl, a, b)
#define MUL_PD(a, b, vl) svmul_f64_x(vl, a, b)

#define CMP_PD_GT(a, b, vl) svcmpgt_f64(vl, a, b)
#define CMP_PD_EQ(a, b, vl) svcmpeq_f64(vl, a, svdup_f64(b))

#define BLEND_EPI(a, b, mask) svsel(mask, b, a)
#define BLEND_PD(a, b, mask) svsel(mask, b, a)

#define MASKSTOREU_PD(mask, a, b) svst1_f64(mask, a, b)
#define MASKSTOREU_EPI(mask, a, b) svst1_u64(mask, (uint64_t*)(a), b)

extern "C" {

// MB1: Bucle de ventana inicial para covarianza con predicados SVE
__attribute__((noinline))
VDTYPE scamp_init_covariance_sve(const DTYPE* __restrict tSeries,
                                 const DTYPE* __restrict means,
                                 ITYPE diag,
                                 ITYPE windowSize,
                                 svbool_t vlOuter)
{
    VDTYPE covariance_v = SETZERO_PD();
    VDTYPE meansWinDiag_v = LOADU_PD(&means[diag], vlOuter);
    VDTYPE meansWin0_v = SET1_PD(means[0], vlOuter);

    for (ITYPE i = 0; i < windowSize; i++)
    {
        VDTYPE tSeriesWinDiag_v = LOADU_PD(&tSeries[diag + i], vlOuter);
        VDTYPE tSeriesWin0_v = SET1_PD(tSeries[i], vlOuter);
        covariance_v = FMADD_PD(SUB_PD(tSeriesWinDiag_v, meansWinDiag_v, vlOuter),
                                SUB_PD(tSeriesWin0_v, meansWin0_v, vlOuter),
                                covariance_v, vlOuter);
    }
    return covariance_v;
}

// MB2: Un paso incremental de la diagonal SVE (reducción horizontal nativa svmaxv + predicados)
__attribute__((noinline))
VDTYPE scamp_diag_step_sve(VDTYPE covariance_v,
                           const DTYPE* __restrict df,
                           const DTYPE* __restrict dg,
                           const DTYPE* __restrict norms,
                           DTYPE* __restrict profile_tmp,
                           ITYPE* __restrict profileIndex_tmp,
                           ITYPE i,
                           ITYPE j,
                           svbool_t vlInner)
{
    VDTYPE dfj_v = LOADU_PD(&df[j - 1], vlInner);
    VDTYPE dgj_v = LOADU_PD(&dg[j - 1], vlInner);
    VDTYPE dfi_v = SET1_PD(df[i - 1], vlInner);
    VDTYPE dgi_v = SET1_PD(dg[i - 1], vlInner);

    covariance_v = ADD_PD(covariance_v,
                          FMADD_PD(dfi_v, dgj_v, MUL_PD(dfj_v, dgi_v, vlInner), vlInner),
                          vlInner);

    VDTYPE normsi_v = SET1_PD(norms[i], vlInner);
    VDTYPE normsj_v = LOADU_PD(&norms[j], vlInner);
    VDTYPE correlation_v = MUL_PD(MUL_PD(covariance_v, normsi_v, vlInner), normsj_v, vlInner);

    // Búsqueda horizontal con reducción vectorial SVE
    DTYPE corr_max = REDMAX_PD(correlation_v, profile_tmp[i], vlInner);
    VMTYPE mask = CMP_PD_EQ(correlation_v, corr_max, vlInner);
    uint64_t index_max = GETFIRST_MASK(mask, vlInner);

    if (index_max != (uint64_t)-1)
    {
        profile_tmp[i] = corr_max;
        profileIndex_tmp[i] = j + index_max;
    }

    // Actualización de columna enmascarada
    VDTYPE profilej_v = LOADU_PD(&profile_tmp[j], vlInner);
    mask = CMP_PD_GT(correlation_v, profilej_v, vlInner);
    MASKSTOREU_PD(mask, &profile_tmp[j], correlation_v);
    MASKSTOREU_EPI(mask, &profileIndex_tmp[j], SET1_EPI(i, vlInner));

    return covariance_v;
}

// MB3: Bucle de la diagonal caliente completo en SVE (con cálculo de predicados VLA por iteración)
__attribute__((noinline))
void scamp_diag_loop_sve(VDTYPE covariance_v,
                         const DTYPE* __restrict df,
                         const DTYPE* __restrict dg,
                         const DTYPE* __restrict norms,
                         DTYPE* __restrict profile_tmp,
                         ITYPE* __restrict profileIndex_tmp,
                         ITYPE diag,
                         ITYPE profileLength)
{
    uint64_t VLMAX = svcntd();
    ITYPE i = 1;

    for (ITYPE j = diag + 1; j < profileLength; j++)
    {
        svbool_t vlInner = svwhilelt_b64((ITYPE)0, (profileLength - j < (ITYPE)VLMAX ? profileLength - j : (ITYPE)VLMAX));

        VDTYPE dfj_v = LOADU_PD(&df[j - 1], vlInner);
        VDTYPE dgj_v = LOADU_PD(&dg[j - 1], vlInner);
        VDTYPE dfi_v = SET1_PD(df[i - 1], vlInner);
        VDTYPE dgi_v = SET1_PD(dg[i - 1], vlInner);

        covariance_v = ADD_PD(covariance_v,
                              FMADD_PD(dfi_v, dgj_v, MUL_PD(dfj_v, dgi_v, vlInner), vlInner),
                              vlInner);

        VDTYPE normsi_v = SET1_PD(norms[i], vlInner);
        VDTYPE normsj_v = LOADU_PD(&norms[j], vlInner);
        VDTYPE correlation_v = MUL_PD(MUL_PD(covariance_v, normsi_v, vlInner), normsj_v, vlInner);

        DTYPE corr_max = REDMAX_PD(correlation_v, profile_tmp[i], vlInner);
        VMTYPE mask = CMP_PD_EQ(correlation_v, corr_max, vlInner);
        uint64_t index_max = GETFIRST_MASK(mask, vlInner);

        if (index_max != (uint64_t)-1)
        {
            profile_tmp[i] = corr_max;
            profileIndex_tmp[i] = j + index_max;
        }

        VDTYPE profilej_v = LOADU_PD(&profile_tmp[j], vlInner);
        mask = CMP_PD_GT(correlation_v, profilej_v, vlInner);
        MASKSTOREU_PD(mask, &profile_tmp[j], correlation_v);
        MASKSTOREU_EPI(mask, &profileIndex_tmp[j], SET1_EPI(i, vlInner));

        i++;
    }
}

// MB4: Reducción inter-hilos SVE para un bloque de columnas
__attribute__((noinline))
void scamp_thread_reduce_sve(const DTYPE* __restrict profile_tmp,
                             const ITYPE* __restrict profileIndex_tmp,
                             DTYPE* __restrict profile,
                             ITYPE* __restrict profileIndex,
                             ITYPE colum,
                             ITYPE numThreads,
                             ITYPE profileLength,
                             svbool_t vlRed)
{
    VDTYPE max_corr_v = SET1_PD(-1.0 / 0.0, vlRed);
    VITYPE max_indices_v = SET1_EPI(-1.0, vlRed);

    for (ITYPE th = 0; th < numThreads; th++)
    {
        VDTYPE p_v = LOADU_PD(&profile_tmp[colum + (th * profileLength)], vlRed);
        VITYPE idx_v = LOADU_SI(&profileIndex_tmp[colum + (th * profileLength)], vlRed);
        VMTYPE mask = CMP_PD_GT(p_v, max_corr_v, vlRed);
        max_indices_v = BLEND_EPI(max_indices_v, idx_v, mask);
        max_corr_v = BLEND_PD(max_corr_v, p_v, mask);
    }

    STORE_PD(&profile[colum], max_corr_v, vlRed);
    STORE_SI(&profileIndex[colum], max_indices_v, vlRed);
}

}

