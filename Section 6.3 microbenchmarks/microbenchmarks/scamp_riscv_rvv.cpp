// ==============================================================================
// Microbenchmark: SCAMP - RISC-V RVV (V0 Original As-is)
// Fichero: scamp_riscv_rvv.cpp
//
// Procedencia: algoritmos/riscv/scamp-v.cpp
//
// Protocolo Godbolt (Compiler Explorer):
// - Compilador Principal: RISC-V 64 gcc 14.2 (o rv64gc gcc trunk)
//   Flags: -O2 -march=rv64gcv -mabi=lp64d
// - Compilador Secundario: RISC-V 64 clang 18.1
//   Flags: -O2 -march=rv64gcv -mabi=lp64d
// ==============================================================================

#include <stdint.h>
#include <riscv_vector.h>

#define DTYPE double
#define ITYPE unsigned long

#define VDTYPE vfloat64m1_t
#define VITYPE vuint64m1_t
#define VMTYPE vbool64_t

#define SETZERO_PD(vl) __riscv_vfmv_v_f_f64m1(0.0, vl)
#define SET1_PD(a, vl) __riscv_vfmv_v_f_f64m1(a, vl)
#define SET1_EPI(a, vl) __riscv_vmv_v_x_u64m1(a, vl)

#define GETFIRST_PD(a) __riscv_vfmv_f_s_f64m1_f64(a)
#define SETFIRST_PD(a, vl) __riscv_vfmv_s_f_f64m1(a, vl)
#define REDMAX_PD(a, b, vl) __riscv_vfredmax_vs_f64m1_f64m1(a, b, vl)
#define GETFIRST_MASK(mask, vl) __riscv_vfirst_m_b64(mask, vl)

#define LOADU_PD(a, vl) __riscv_vle64_v_f64m1(a, vl)
#define LOADU_SI(a, vl) __riscv_vle64_v_u64m1((const unsigned long*)(a), vl)

#define STORE_PD(a, b, vl) __riscv_vse64_v_f64m1(a, b, vl)
#define STORE_SI(a, b, vl) __riscv_vse64_v_u64m1((unsigned long*)(a), b, vl)

#define FMADD_PD(a, b, c, vl) __riscv_vfmadd_vv_f64m1(a, b, c, vl)
#define SUB_PD(a, b, vl) __riscv_vfsub_vv_f64m1(a, b, vl)
#define ADD_PD(a, b, vl) __riscv_vfadd_vv_f64m1(a, b, vl)
#define MUL_PD(a, b, vl) __riscv_vfmul_vv_f64m1(a, b, vl)

#define CMP_PD_GT(a, b, vl) __riscv_vmfgt_vv_f64m1_b64(a, b, vl)
#define CMP_PD_EQ(a, b, vl) __riscv_vmfeq_vf_f64m1_b64(a, b, vl)

#define BLEND_EPI(a, b, mask, vl) __riscv_vmerge_vvm_u64m1(a, b, mask, vl)
#define BLEND_PD(a, b, mask, vl) __riscv_vmerge_vvm_f64m1(a, b, mask, vl)

#define MASKSTOREU_PD(mask, a, b, vl) __riscv_vse64_v_f64m1_m(mask, a, b, vl)
#define MASKSTOREU_EPI(mask, a, b, vl) __riscv_vse64_v_u64m1_m(mask, (unsigned long*)(a), b, vl)

extern "C" {

// MB1: Bucle de ventana inicial para covarianza RVV
__attribute__((noinline))
VDTYPE scamp_init_covariance_rvv(const DTYPE* __restrict tSeries,
                                 const DTYPE* __restrict means,
                                 ITYPE diag,
                                 ITYPE windowSize,
                                 size_t vlOuter)
{
    VDTYPE covariance_v = SETZERO_PD(vlOuter);
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

// MB2: Un paso incremental de la diagonal RVV (reducción horizontal vfredmax + vfirst.m)
__attribute__((noinline))
VDTYPE scamp_diag_step_rvv(VDTYPE covariance_v,
                           const DTYPE* __restrict df,
                           const DTYPE* __restrict dg,
                           const DTYPE* __restrict norms,
                           DTYPE* __restrict profile_tmp,
                           ITYPE* __restrict profileIndex_tmp,
                           ITYPE i,
                           ITYPE j,
                           size_t vlInner)
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

    DTYPE corr_max = GETFIRST_PD(REDMAX_PD(correlation_v, SETFIRST_PD(profile_tmp[i], vlInner), vlInner));
    VMTYPE mask = CMP_PD_EQ(correlation_v, corr_max, vlInner);
    long index_max = GETFIRST_MASK(mask, vlInner);

    if (index_max != -1)
    {
        profile_tmp[i] = corr_max;
        profileIndex_tmp[i] = j + index_max;
    }

    VDTYPE profilej_v = LOADU_PD(&profile_tmp[j], vlInner);
    mask = CMP_PD_GT(correlation_v, profilej_v, vlInner);
    MASKSTOREU_PD(mask, &profile_tmp[j], correlation_v, vlInner);
    MASKSTOREU_EPI(mask, &profileIndex_tmp[j], SET1_EPI(i, vlInner), vlInner);

    return covariance_v;
}

// MB3: Bucle de la diagonal caliente completo en RVV
__attribute__((noinline))
void scamp_diag_loop_rvv(VDTYPE covariance_v,
                         const DTYPE* __restrict df,
                         const DTYPE* __restrict dg,
                         const DTYPE* __restrict norms,
                         DTYPE* __restrict profile_tmp,
                         ITYPE* __restrict profileIndex_tmp,
                         ITYPE diag,
                         ITYPE profileLength)
{
    uint64_t VLMAX = __riscv_vsetvlmax_e64m1();
    ITYPE i = 1;

    for (ITYPE j = diag + 1; j < profileLength; j++)
    {
        size_t vlInner = (profileLength - j < (ITYPE)VLMAX ? profileLength - j : (ITYPE)VLMAX);

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

        DTYPE corr_max = GETFIRST_PD(REDMAX_PD(correlation_v, SETFIRST_PD(profile_tmp[i], vlInner), vlInner));
        VMTYPE mask = CMP_PD_EQ(correlation_v, corr_max, vlInner);
        long index_max = GETFIRST_MASK(mask, vlInner);

        if (index_max != -1)
        {
            profile_tmp[i] = corr_max;
            profileIndex_tmp[i] = j + index_max;
        }

        VDTYPE profilej_v = LOADU_PD(&profile_tmp[j], vlInner);
        mask = CMP_PD_GT(correlation_v, profilej_v, vlInner);
        MASKSTOREU_PD(mask, &profile_tmp[j], correlation_v, vlInner);
        MASKSTOREU_EPI(mask, &profileIndex_tmp[j], SET1_EPI(i, vlInner), vlInner);

        i++;
    }
}

// MB4: Reducción inter-hilos RVV para un bloque de columnas
__attribute__((noinline))
void scamp_thread_reduce_rvv(const DTYPE* __restrict profile_tmp,
                             const ITYPE* __restrict profileIndex_tmp,
                             DTYPE* __restrict profile,
                             ITYPE* __restrict profileIndex,
                             ITYPE colum,
                             ITYPE numThreads,
                             ITYPE profileLength,
                             size_t vlRed)
{
    VDTYPE max_corr_v = SET1_PD(-1.0 / 0.0, vlRed);
    VITYPE max_indices_v = SET1_EPI(-1, vlRed);

    for (ITYPE th = 0; th < numThreads; th++)
    {
        VDTYPE p_v = LOADU_PD(&profile_tmp[colum + (th * profileLength)], vlRed);
        VITYPE idx_v = LOADU_SI(&profileIndex_tmp[colum + (th * profileLength)], vlRed);
        VMTYPE mask = CMP_PD_GT(p_v, max_corr_v, vlRed);
        max_indices_v = BLEND_EPI(max_indices_v, idx_v, mask, vlRed);
        max_corr_v = BLEND_PD(max_corr_v, p_v, mask, vlRed);
    }

    STORE_PD(&profile[colum], max_corr_v, vlRed);
    STORE_SI(&profileIndex[colum], max_indices_v, vlRed);
}

}
