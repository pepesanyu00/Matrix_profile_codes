// Verbatim extraction of REDMAX_PD/CMP_PD_EQ/GETFIRST_MASK from
// codigo_mio/Matrix_Profile_Vectorization-main/algorithms/arm/scamp-v.cpp, isolated for instruction counting.
// Godbolt flags: -O2 -march=armv8.2-a+sve   (compiler: ARM64 gcc/clang, trunk)
#include <arm_sve.h>
#include <cstdint>

#define VDTYPE svfloat64_t
#define VMTYPE svbool_t

#define REDMAX_PD(a, b, vl) svmaxv_f64(vl, svmax_f64_m(vl, a, svdup_f64(b)))
#define CMP_PD_EQ(a, b, vl) svcmpeq_f64(vl, a, svdup_f64(b))

static inline uint64_t get_first_mask(svbool_t mask, svbool_t vl)
{
    svbool_t first_true = svbrkb_z(vl, mask); // break to first true element
    uint64_t index = svcntp_b64(vl, first_true);
    if (index == svcntp_b64(vl, vl)) // no true found
        return (uint64_t)-1;
    return index;
}
#define GETFIRST_MASK(mask, vl) get_first_mask(mask, vl)

extern "C" __attribute__((noinline))
void find_max_sve(const double* mem, double current_max, long current_index, long base_index,
                   double* out_max, long* out_index)
{
    svbool_t vl = svptrue_b64();
    VDTYPE correlation_v = svld1_f64(vl, mem);

    double corr_max = REDMAX_PD(correlation_v, current_max, vl);
    VMTYPE mask = CMP_PD_EQ(correlation_v, corr_max, vl);
    long index_max = (long)GETFIRST_MASK(mask, vl);

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
