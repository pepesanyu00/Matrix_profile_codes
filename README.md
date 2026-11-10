# Cross-ISA Matrix Profile: SCAMP & SCRIMP Evaluation Suite

This repository contains the source code, microbenchmarks, compiler assembly traces, and analytical toolset used in the scientific evaluation of the **SCAMP** and **SCRIMP** Matrix Profile algorithms across fixed-width packed SIMD (**x86-64 AVX2 / AVX-512**) and Vector-Length Agnostic (VLA) architectures (**ARM SVE** and **RISC-V RVV**).

---

## Repository Structure & Folder Breakdown

```
matrix_profile_codes/
├── source codes/                  # Full algorithm implementations per architecture
│   ├── arm/                       # ARM AArch64 scalar and SVE implementations (+ gem5 traces)
│   └── riscv/                     # RISC-V RV64 scalar and RVV implementations (OpenMP, QEMU, gem5)
|   └── gem5_config_scripts        # Gem5 config scripts for RISC-V and ARM
├── Section 6.3 microbenchmarks/   # Phase-isolated microbenchmarks & ISA complexity study
│   ├── microbenchmarks/           # Standalone C++ kernels isolating phases PH1 to PH4
│   ├── outputs/                   # 36 Godbolt compiler assembly traces (GCC 14.2 & Clang 18.1)
│   └── README.md                  # Dedicated documentation for this section
└── tools/                         # Automated analytical and validation toolchain
    ├── README.md                  # Exhaustive catalog and workflow description of all tools
    ├── (Block 1: ISA Complexity)  # Scripts for CFG analysis, instruction counting, taxonomy & LaTeX
    └── (Block 2: gem5 Validation) # Scripts for diagonal sampling validation, uarch audits & figures
```

---

## Detailed Contents by Directory

### 1. `source codes/`
Contains the complete, end-to-end implementations of SCAMP and SCRIMP tailored to ARM and RISC-V architectures:
- **`arm/`**:
  - `scamp.cpp`, `scrimp.cpp`: Scalar baseline implementations on ARM AArch64.
  - `scamp-v.cpp`, `scrimp-v.cpp`: Vectorized implementations using ARM SVE intrinsics (includes agnostic predication `_x`).
  - `scamp-v-nored.cpp`, `scrimp-v-nored.cpp`: Ablation variants without horizontal reduction (used to isolate reduction overhead).
  - `scamp-v_initial`, `scamp-v_preds`: Intermediate versions evaluated during SVE predication tuning.
  - `m5out/`: gem5 simulation output directory (`stats.txt`, `config.ini`).
- **`riscv/`**:
  - `scamp.cpp`, `scrimp.cpp`: Scalar baseline implementations on RISC-V RV64.
  - `scamp-v.cpp`, `scrimp-v.cpp`: Vectorized implementations using native RISC-V Vector (RVV 1.0) intrinsics.
  - `scamp-vomp.cpp`, `scrimp-vomp.cpp`: Multi-threaded RVV implementations parallelized with OpenMP (`schedule(dynamic)`).
  - `scamp-v-nored.cpp`, `scrimp-v-nored.cpp`: RVV ablation variants without horizontal reduction.
  - `*-qemu`: Precompiled cross-architecture binaries for functional validation under QEMU.
- **`gem5_config_scripts/`**:
  - custom_cache: cache configuration for both ISAS, needed to execute the scripts.
  - gem5_config_riscv.py: config script for RISC-V implementation.
  - gem5_config_arm.py: config script for ARM implementation. 


### 2. `Section 6.3 microbenchmarks/`
Contains the phase-isolated kernels, assembly traces, and compiler evaluation artifacts used to analyze ISA complexity free of simulator or hardware clock biases:
- **`microbenchmarks/`**: Standalone, self-contained C++ source files isolating the 4 algorithmic phases:
  - `scamp_scalar.cpp`, `scrimp_scalar.cpp`: Sequential reference baselines.
  - `scamp_avx2.cpp`, `scamp_avx512.cpp`: Legacy packed SIMD variants (V0) with sequential stack-based reduction.
  - `*_vectorized.cpp` & `*_vectorized_loop.cpp`: Optimized AVX2 and AVX-512 kernels implementing in-register shuffle-based reduction trees (`VecRed`).
  - `scamp_arm_sve.cpp`, `scrimp_arm_sve.cpp`: ARM SVE kernels evaluated under zeroing (`_z`) and agnostic (`_x`) predication.
  - `scamp_riscv_rvv.cpp`, `scrimp_riscv_rvv.cpp`: RISC-V RVV kernels exploiting atomic vector reductions (`vfredmax`/`vfredmin`) and mask search (`vfirst.m`).
- **`outputs/`**:  36 complete assembly traces (18 from GCC 14.2 and 18 from Clang 18.1 with `-O2`) covering all architectures and variants.

### 3. `tools/`
Automated Python and C++ toolchain divided into two major functional blocks (see `tools/README.md` for full documentation):

#### Block 1: Instruction-Level Analytical Evaluation & ISA Complexity
- **`cfg_dynamic_counts.py` & `cfg_check_asm.py`**: Builds Control Flow Graphs (CFG) from assembly traces; extracts exact static instruction counts, dynamic min–max bounds, and separates loop setup overhead from per-iteration cost.
- **`gen_ph_tables.py` & `gen_latex_table.py`**: Aggregates metrics across phases and emits LaTeX code for **Table 5**.
- **`horizontal_reduction_breakdown.py` & `verify_horizontal_blocks.py`**: Performs instruction-by-instruction taxonomy of the horizontal search (Fold, Broadcast, Mask Gen, Index Extraction, Commit) for **Table 6 (`tab:horizontal_breakdown`)**, with formal CFG verification.
- **`isa_efficiency_metrics.py` & `calc_phase_metrics_full.py`**: Analyzes dependency chains, critical path depth ($D_{\text{cp}}$), Instruction-Level Parallelism ($\text{ILP} = I_d / D_{\text{cp}}$), and CISC micro-operation ($\mu\text{Ops}$) decomposition for **Table 8 (`tab:ILP_metrics`)**.
- **`gen_sec4.py` & `gen_tau.py`**: Computes exact numeric deltas quoted in the narrative text of Section 4 and vector code expansion factors ($\tau_{\text{exp}}$).
- **`gen_vectorized_diag_loop.py` & `godbolt_compile_ph3_vec.py`**: Synthesizes full diagonal loops for AVX vectorized reductions and automates remote compilation via the Godbolt REST API.
- **`taxonomic_breakdown.py`, `analyze_all_traces.py`, `generate_tables.py`**: General mnemonic classifiers, branch density auditors, and terminal inspection utilities.

#### Block 2: Methodological Validation of Diagonal Sampling
- **`sample_sel.cpp` & `sample_sel`**: High-performance C++ reference emulator of the pseudo-random diagonal shuffling algorithm; computes exact matrix cell coverage.
- **`common.py`**: Shared engine providing OpenMP dynamic makespan scheduling models, LRU caches, and statistical correlations (Spearman, log-Pearson).
- **`extract.py`**: Multiprocessing parser extracting Region of Interest (ROI) metrics from gem5 `stats.txt` runs into `runs.csv`.
- **`speedups.py`**: Generates raw and cell-coverage-corrected speedup metrics across sampling fractions (10%, 25%, 100%).
- **`model.py`**: Fits and validates the affine linear extrapolation model ($T = a + cX$) against full 100% ground-truth simulations.
- **`trends.py`**: Proves that 10% sampling preserves optimal thread points and relative cross-ISA performance rankings.
- **`uarch.py`**: Audits microarchitectural stability, demonstrating $<2.7\%$ variation in IPC, cache miss rates (L1/L2/L3), and DRAM traffic per cell between 10% and 100% runs.
- **`mpq.py` & `check_profile_correctness.py`**: Validates Matrix Profile fidelity (motif/discord preservation) and numerical floating-point correctness across gem5 runs.
- **`make_figs.py`**: Generates figures in PDF and PNG formats, including **Figure 5** of the manuscript.

---

---

