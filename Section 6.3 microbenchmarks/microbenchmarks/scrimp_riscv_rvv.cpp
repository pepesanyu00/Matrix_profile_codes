// ==============================================================================
// Microbenchmark: SCRIMP - RISC-V RVV (V0 Original As-is)
// Fichero: scrimp_riscv_rvv.cpp
//
// Procedencia: algoritmos/riscv/scrimp-v.cpp
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
#define REDMIN_PD(a, b, vl) __riscv_vfredmin_vs_f64m1_f64m1(a, b, vl)
#define GETFIRST_MASK(mask, vl) __riscv_vfirst_m_b64(mask, vl)

#define LOADU_PD(a, vl) __riscv_vle64_v_f64m1(a, vl)
#define LOADU_SI(a, vl) __riscv_vle64_v_u64m1((const unsigned long*)(a), vl)

#define STORE_PD(a, b, vl) __riscv_vse64_v_f64m1(a, b, vl)
#define STORE_SI(a, b, vl) __riscv_vse64_v_u64m1((unsigned long*)(a), b, vl)

#define FMADD_PD(a, b, c, vl) __riscv_vfmadd_vv_f64m1(a, b, c, vl)
#define FMSUB_PD(a, b, c, vl) __riscv_vfmsub_vv_f64m1(a, b, c, vl)
#define SUB_PD(a, b, vl) __riscv_vfsub_vv_f64m1(a, b, vl)
#define ADD_PD(a, b, vl) __riscv_vfadd_vv_f64m1(a, b, vl)
#define MUL_PD(a, b, vl) __riscv_vfmul_vv_f64m1(a, b, vl)
#define DIV_PD(a, b, vl) __riscv_vfdiv_vv_f64m1(a, b, vl)

#define CMP_PD_LT(a, b, vl) __riscv_vmflt_vv_f64m1_b64(a, b, vl)
#define CMP_PD_EQ(a, b, vl) __riscv_vmfeq_vf_f64m1_b64(a, b, vl)

#define BLEND_EPI(a, b, mask, vl) __riscv_vmerge_vvm_u64m1(a, b, mask, vl)
#define BLEND_PD(a, b, mask, vl) __riscv_vmerge_vvm_f64m1(a, b, mask, vl)

#define MASKSTOREU_PD(mask, a, b, vl) __riscv_vse64_v_f64m1_m(mask, a, b, vl)
#define MASKSTOREU_EPI(mask, a, b, vl) __riscv_vse64_v_u64m1_m(mask, (unsigned long*)(a), b, vl)

extern "C" {

// MB1: Bucle de ventana inicial para dot product en RVV O(w)
__attribute__((noinline))
VDTYPE scrimp_init_dot_rvv(const DTYPE* __restrict tSeries,
                           ITYPE diag,
                           ITYPE windowSize,
                           size_t vlOuter)
{
    VDTYPE dotProd_v = SETZERO_PD(vlOuter);
    for (ITYPE j = diag; j < windowSize + diag; j++)
    {
        VDTYPE tSeriesj_v = LOADU_PD(&tSeries[j], vlOuter);
        VDTYPE tSeriesMinusDiag_v = SET1_PD(tSeries[j - diag], vlOuter);
        dotProd_v = FMADD_PD(tSeriesj_v, tSeriesMinusDiag_v, dotProd_v, vlOuter);
    }
    return dotProd_v;
}

// MB2: Un paso incremental de la diagonal caliente en SCRIMP RVV
__attribute__((noinline))
VDTYPE scrimp_diag_step_rvv(VDTYPE dotProd_v,
                            const DTYPE* __restrict tSeries,
                            const DTYPE* __restrict means,
                            const DTYPE* __restrict devs,
                            DTYPE* __restrict profile_tmp,
                            ITYPE* __restrict profileIndex_tmp,
                            ITYPE i,
                            ITYPE j,
                            ITYPE windowSize,
                            size_t vlInner)
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

    // Búsqueda horizontal para mínimo en RVV
    DTYPE corr_max = GETFIRST_PD(REDMIN_PD(distance_v, SETFIRST_PD(profile_tmp[i], vlInner), vlInner));
    VMTYPE mask = CMP_PD_EQ(distance_v, corr_max, vlInner);
    long index_max = GETFIRST_MASK(mask, vlInner);

    if (index_max != -1)
    {
        profile_tmp[i] = corr_max;
        profileIndex_tmp[i] = j + index_max;
    }

    // Actualización de columna enmascarada
    VDTYPE profilej_v = LOADU_PD(&profile_tmp[j], vlInner);
    mask = CMP_PD_LT(distance_v, profilej_v, vlInner);
    MASKSTOREU_PD(mask, &profile_tmp[j], distance_v, vlInner);
    MASKSTOREU_EPI(mask, &profileIndex_tmp[j], SET1_EPI(i, vlInner), vlInner);

    return dotProd_v;
}

// MB3: Bucle completo de la diagonal caliente en SCRIMP RVV
__attribute__((noinline))
void scrimp_diag_loop_rvv(VDTYPE dotProd_v,
                          const DTYPE* __restrict tSeries,
                          const DTYPE* __restrict means,
                          const DTYPE* __restrict devs,
                          DTYPE* __restrict profile_tmp,
                          ITYPE* __restrict profileIndex_tmp,
                          ITYPE diag,
                          ITYPE profileLength,
                          ITYPE windowSize)
{
    uint64_t VLMAX = __riscv_vsetvlmax_e64m1();
    ITYPE i = 1;

    for (ITYPE j = diag + 1; j < profileLength; j++)
    {
        size_t vlInner = (profileLength - j < (ITYPE)VLMAX ? profileLength - j : (ITYPE)VLMAX);

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

        DTYPE corr_max = GETFIRST_PD(REDMIN_PD(distance_v, SETFIRST_PD(profile_tmp[i], vlInner), vlInner));
        VMTYPE mask = CMP_PD_EQ(distance_v, corr_max, vlInner);
        long index_max = GETFIRST_MASK(mask, vlInner);

        if (index_max != -1)
        {
            profile_tmp[i] = corr_max;
            profileIndex_tmp[i] = j + index_max;
        }

        VDTYPE profilej_v = LOADU_PD(&profile_tmp[j], vlInner);
        mask = CMP_PD_LT(distance_v, profilej_v, vlInner);
        MASKSTOREU_PD(mask, &profile_tmp[j], distance_v, vlInner);
        MASKSTOREU_EPI(mask, &profileIndex_tmp[j], SET1_EPI(i, vlInner), vlInner);

        i++;
    }
}

// MB4: Reducción inter-hilos (mínimo) RVV para un bloque de columnas
__attribute__((noinline))
void scrimp_thread_reduce_rvv(const DTYPE* __restrict profile_tmp,
                              const ITYPE* __restrict profileIndex_tmp,
                              DTYPE* __restrict profile,
                              ITYPE* __restrict profileIndex,
                              ITYPE colum,
                              ITYPE numThreads,
                              ITYPE profileLength,
                              size_t vlRed)
{
    VDTYPE min_dist_v = SET1_PD(1.0 / 0.0, vlRed); // +infinity
    VITYPE min_indices_v = SET1_EPI(1, vlRed);

    for (ITYPE th = 0; th < numThreads; th++)
    {
        VDTYPE p_v = LOADU_PD(&profile_tmp[colum + (th * profileLength)], vlRed);
        VITYPE idx_v = LOADU_SI(&profileIndex_tmp[colum + (th * profileLength)], vlRed);
        VMTYPE mask = CMP_PD_LT(p_v, min_dist_v, vlRed);
        min_indices_v = BLEND_EPI(min_indices_v, idx_v, mask, vlRed);
        min_dist_v = BLEND_PD(min_dist_v, p_v, mask, vlRed);
    }

    STORE_PD(&profile[colum], min_dist_v, vlRed);
    STORE_SI(&profileIndex[colum], min_indices_v, vlRed);
}

}
