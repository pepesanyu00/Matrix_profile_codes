// Baseline control: literal extraction of the production AVX2 horizontal-max logic
// (see codigo_ric/MPvect_RISC-V-main/scamp-v.cpp, lines ~208-224).
// Godbolt flags: -O2 -mavx2 -mfma   (compiler: x86-64 gcc/clang, trunk; also test -O3)
#include <immintrin.h>

#define ALIGN 32
#define VPACK 4

extern "C" __attribute__((noinline))
void find_max_avx2_baseline(const double* mem, double current_max, long current_index, long base_index,
                             double* out_max, long* out_index)
{
    __m256d correlation_v = _mm256_loadu_pd(mem);

    alignas(ALIGN) double correlation[VPACK];
    _mm256_store_pd(correlation, correlation_v);

    double max_val = current_max;
    long max_idx = current_index;
    for (int ii = 0; ii < VPACK; ii++)
    {
        if (correlation[ii] > max_val)
        {
            max_val = correlation[ii];
            max_idx = base_index + ii;
        }
    }

    *out_max = max_val;
    *out_index = max_idx;
}
