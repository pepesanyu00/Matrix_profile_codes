// Verbatim extraction of REDMAX_PD/GETFIRST_PD/CMP_PD_EQ/GETFIRST_MASK from
// codigo_mio/Matrix_Profile_Vectorization-main/algorithms/riscv/scamp-v.cpp, isolated for instruction counting.
// Godbolt flags: -O2 -march=rv64gcv -mabi=lp64d   (compiler: RISC-V rv64 gcc/clang, trunk;
// verify Godbolt has a vector-intrinsics-capable build, otherwise use a local riscv64-unknown-linux-gnu-g++ toolchain)
#include <riscv_vector.h>

#define VDTYPE vfloat64m1_t
#define VMTYPE vbool64_t

#define SETFIRST_PD(a, vl) __riscv_vfmv_s_f_f64m1(a, vl)
#define REDMAX_PD(a, b, vl) __riscv_vfredmax_vs_f64m1_f64m1(a, b, vl)
#define GETFIRST_PD(a) __riscv_vfmv_f_s_f64m1_f64(a)
#define CMP_PD_EQ(a, b, vl) __riscv_vmfeq_vf_f64m1_b64(a, b, vl)
#define GETFIRST_MASK(mask, vl) __riscv_vfirst_m_b64(mask, vl)

extern "C" __attribute__((noinline))
void find_max_rvv(const double* mem, double current_max, long current_index, long base_index,
                   double* out_max, long* out_index)
{
    size_t vl = __riscv_vsetvlmax_e64m1();
    VDTYPE correlation_v = __riscv_vle64_v_f64m1(mem, vl);

    double corr_max = GETFIRST_PD(REDMAX_PD(correlation_v, SETFIRST_PD(current_max, vl), vl));
    VMTYPE mask = CMP_PD_EQ(correlation_v, corr_max, vl);
    long index_max = GETFIRST_MASK(mask, vl);

    if (index_max != -1)
    {
        *out_max = corr_max;
        *out_index = base_index + index_max;
    }
    else
    {
        *out_max = current_max;
        *out_index = current_index;
    }
}
