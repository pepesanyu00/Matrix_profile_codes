# Section 6.3 Microbenchmarks: ISA Complexity & Vectorization Suite

This directory contains the standalone microbenchmarks, compiler assembly traces, compilation guides, and analytical studies used to evaluate instruction-level ISA complexity for **SCAMP** and **SCRIMP** across **x86-64 (AVX2 / AVX-512)**, **ARM (SVE)**, and **RISC-V (RVV)**.

---

## Directory Contents & Purpose

```
Section 6.3 microbenchmarks/
├── microbenchmarks/              # Standalone C++ kernels isolating algorithmic phases (PH1–PH4)
│   ├── scamp_*.cpp               # SCAMP kernels (scalar, avx2, avx512, sve, rvv, and VecRed variants)
│   └── scrimp_*.cpp              # SCRIMP kernels (scalar, avx2, avx512, sve, rvv, and VecRed variants)
├── outputs/                      # Assembly outputs from Compiler Explorer (Godbolt)
│   ├── gcc/                  # 18 complete assembly traces compiled with GCC 14.2 -O2
│   └── clang/                # 18 complete assembly traces compiled with Clang 18.1 -O2
```

---

## Role in the Paper & Reviewer Responses

### 1. Section 6.3: Vectorized Reduction Ablation (Figure 9)
Addresses reviewer inquiries regarding horizontal search overhead:
- **Baseline Bottleneck (`V0`)**: The original reference code dumped SIMD vectors to the stack (`alignas(64) correlation[VPACK]`) and iterated sequentially ($4\times$ in AVX2, $8\times$ in AVX-512).
- **In-Register Reduction (`VecRed`)**: Replaces the stack spill with shuffle-based reduction trees. This achieves:
  - **AVX2 (256-bit)**: **-45.6%** dynamic instructions in SCAMP (60–68 $\to$ 33–37) and **-31.0%** in SCRIMP (73–81 $\to$ 51–55).
  - **AVX-512 (512-bit)**: **-55.0%** dynamic instructions in SCAMP (77–93 $\to$ 36–40) and **-47.4%** in SCRIMP (95–111 $\to$ 52–56).
- **Intel Built-in Validation**: Confirms that Intel intrinsics (`_mm512_reduce_max_pd` / `_mm512_reduce_min_pd`) emit the exact same 7-instruction shuffle tree as the manual implementation.

### 2. Section 6.3: Microarchitecture-Independent Instruction Counts (Tables 5, 6 & 7)
- **Static vs. Dynamic Counting**: Uncovers that in GCC 14.2 `-O2`, V0 code preserves an internal backward loop, causing static counts (37–44 instructions) to hide the real dynamic execution cost (59–93 instructions).
- **Taxonomic Breakdown (Table 7)**: Isolates the horizontal search block into 5 stages (*Fold*, *Seed/Move*, *Cmp*, *Mask-to-Index*, *Commit*). Demonstrates that:
  - x86 AVX requires an $O(\log_2 \text{VPACK})$ shuffle tree (5 instructions for AVX2, 7 for AVX-512).
  - ARM SVE collapses the reduction in 2 instructions (`fmax` + `fmaxv`).
  - RISC-V RVV completes the reduction in 1 atomic instruction (`vfredmax.vs` / `vfredmin.vs`) and extracts the match index in 1 instruction (`vfirst.m`).

### 3. ARM SVE Predication Optimization (`_z` vs. `_x`)
- Demonstrates that switching from zeroing (`_z`) to agnostic (`_x`) predication in SVE eliminates **100% of arithmetic `movprfx` instructions**.
- Total `movprfx` count drops from 10 to 5 in SCAMP (**-50%**) and from 19 to 8 in SCRIMP (**-57.9%**) under GCC.
- Under Clang 18.1, `movprfx` drops to **0 in SCAMP** and **1 in SCRIMP**, proving that prior SVE overhead was an intrinsic artifact rather than an inherent ISA defect.

### 4. VLA Software Scalability (Table Comparison)
- Shows that a single universal VLA binary (ARM SVE / RISC-V RVV) scales dynamically from 256-bit to 2048-bit vectors ($8\times$ throughput gain), dropping dynamic instructions per element down to 1.34–1.78, whereas x86 requires binary recompilation, code branching, and stops at 512 bits.

---

## Experimental Setup & Reproduction Summary

All 36 traces were compiled via Compiler Explorer ([Godbolt](https://godbolt.org/)) with the following flags:

| Architecture | Primary Compiler (GCC 14.2) | Secondary Compiler (Clang 18.1) |
|---|---|---|
| **Scalar Baseline** | `-O2 -fno-tree-vectorize` | `-O2 -fno-vectorize -fno-slp-vectorize` |
| **x86 AVX2** | `-O2 -mavx2 -mfma` | `-O2 -mavx2 -mfma` |
| **x86 AVX-512** | `-O2 -mavx512f -mfma` | `-O2 -mavx512f -mfma` |
| **ARM AArch64 SVE** | `-O2 -march=armv8.2-a+sve` | `-O2 -march=armv8.2-a+sve` |
| **RISC-V 64 RVV** | `-O2 -march=rv64gcv -mabi=lp64d` | `-O2 -march=rv64gcv -mabi=lp64d` |
