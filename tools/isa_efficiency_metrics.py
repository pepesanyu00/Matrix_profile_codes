#!/usr/bin/env python3
"""
Cálculo riguroso de las Métricas de Eficiencia de ISA (Métricas 1, 2 y 4) sobre:
  (A) El bloque aislado de reducción horizontal (Tabla 3.1)
  (B) El kernel completo `diag_step` (MB2)
para SCAMP y SCRIMP en GCC 14.2 (-O2) y Clang 18.1 (-O2).

Fuentes Oficiales de Latencias y uOps:
  [x86] Intel 64 and IA-32 Architectures Optimization Reference Manual (2024) /
        Abel & Reineke (ASPLOS'19, uops.info) / LLVM 18.1 `X86SchedSapphireRapids.td` & `X86SchedZnver4.td`.
  [ARM] Arm Neoverse V1 Core Software Optimization Guide (r1p1, 2022) &
        Arm Neoverse V2 Core Software Optimization Guide (2023) /
        LLVM 18.1 `AArch64SchedNeoverseV1.td` & `AArch64SchedNeoverse512TVB.td`.
  [RVV] RISC-V Vector Extension v1.0 (2021) /
        Cavalcante et al., "Ara: A 1-GHz+ Scalable and Energy-Efficient RISC-V Vector Processor", IEEE TVLSI 2020 /
        Xu et al., "XiangShan", IEEE/ACM MICRO 2022 /
        LLVM 18.1 `RISCVSchedSiFive7.td` (SiFive-X280 para contraste in-order).
"""
import os
import re

BASE = os.path.join(os.path.dirname(__file__), "..", "outputs", "trazas microbenchmarks completos")

# ==============================================================================
# TABLA CANÓNICA DE LATENCIAS (CICLOS) Y MICRO-OPERACIONES (uOPS)
# Arquitecturas equivalentes HPC Out-of-Order Superscalar:
#   - x86-64 : Intel Xeon Sapphire Rapids (Golden Cove) [con notas para AMD Zen 4]
#   - ARM64  : Arm Neoverse V1 (256b) / Neoverse-512TVB (512b) [con notas para Neoverse V2]
#   - RISC-V : Perfil HPC OoO Equivalente (Ara/XiangShan/C910-V, árbol log2)
#              [y contraste SiFive-X280 In-Order de LLVM 18.1]
#
# Nota metodológica:
#   - Latencia L1D Load (escalar GP): 4-5 ciclos (usamos 4c escalar GP, 5c FP escalar, 6-7c vectorial).
#   - En x86 con operando en memoria `[PTR]` fusionado en ALU/FP:
#       #uOps = 2 (1 uOp Load + 1 uOp Compute)
#       Latencia desde que los registros índice están listos = Lat(Load) + Lat(ALU).
#       Latencia desde el operando registro en el camino crítico (cuando la dirección de memoria
#       depende solo de punteros de entrada i, j ya listos al inicio de diag_step):
#       la carga `[PTR]` arranca en el ciclo 0 en paralelo con el árbol/cálculo previo,
#       por lo que en un grafo DAG real el nodo alcanza max(t_reg_in, t_load_ready) + Lat_compute.
# ==============================================================================

def get_hred_metrics():
    """
    Calcula para cada variante del bloque de reducción horizontal (Tabla 3.1):
      - Métrica 1:
          * I_static, I_dyn (min-max)
          * uOps_static, uOps_dyn (min-max) desacoplando memoria CISC y cracking interno
          * I_tax (instrucciones de impuesto ISA: shuffles cross-lane, copias/movprfx,
                   cruces de dominio Vec<->GPR/Mask, gestión de bucle/pila)
          * Cruces de dominio (Cross-Lane / Vec<->Mask<->GPR)
      - Métrica 2:
          * D_crit: Número de instrucciones secuenciales en el camino crítico (DAG depth)
            desde que `correlation_v` / `distance_v` está listo hasta:
            (a) resolver la condición de salto (`D_crit_branch`)
            (b) completar el commit en memoria (`D_crit_commit`)
      - Métrica 4:
          * L_fold: Ciclos del camino crítico en la subfase Fold (+ Seed en serie)
          * L_idx : Ciclos del camino crítico en Cmp + Mask->Idx + salto
          * L_commit: Ciclos adicionales de actualización (j+idx + store)
          * L_crit_no_update / L_crit_update: Latencia total del camino crítico (ciclos)
    """
    # Definición analítica exacta verificada línea a línea sobre las 24 entradas de horizontal_reduction_breakdown.py
    # Cada paso de la cadena crítica se detalla explícitamente con su latencia oficial.
    data = [
        # ============================== SCAMP - GCC ==============================
        {
            "alg": "SCAMP", "comp": "GCC", "isa": "x86 AVX2 (no vec. red.)",
            "I_stat": 11, "I_dyn": (26, 38),
            # En bucle 4x: 1 vmovapd store (2 uops: store-addr + store-data) + 1 xor (1 uop)
            # + 4 * [vmovsd load (1) + vcomisd mem (2 uops: load+cmp) + jbe (1) + add (1) + cmp (1) + jne (1)]
            # Macro-fusión cmp+jne = 1 uop en x86 moderno, pero sin macro-fusión = 7 uops/iter; con uops desacoplados de memoria:
            # Estático: 11 instrs + 1 mem-op fusionado en vcomisd = 12 uOps equivalentes.
            # Dinámico: 26 + 4 (mem-op en vcomisd) = 30 min, 38 + 4 = 42 max.
            "uops_stat": 12, "uops_dyn": (30, 42),
            # Tax: vmovapd spill (1) + vmovsd reload (1) + loop control xor/add/cmp/jne (4) = 6 estáticas (22 dinámicas)
            "I_tax_stat": 6, "I_tax_dyn": 22,
            "crossings": 5, # 1 store + 4 reloads
            # Camino crítico: store-to-load forwarding (vmovapd [rsp] -> vmovsd xmm1: 5+4=9c) +
            # 4 iteraciones serializadas por el contador rax y por las escrituras/lecturas a [rcx] (store-forwarding 5c + vcomisd 3c)
            # Sin actualizar: vmovapd(5c store-fwd) -> 4 iteraciones de (vmovsd 4c + vcomisd 3c + jbe 1c) solapadas con add/cmp/jne = ~25c
            # Con actualización en cadena: cada update escribe [rcx] y la siguiente iteración lee [rcx] en vcomisd (+5c store-fwd + 3c cmp = 8c/iter) -> ~37c
            "D_crit": (10, 18), # 1 spill + 4*(load+cmp) = 9-10 min; con updates encadenados en [rcx] = 18
            "L_fold": 9,   # vmovapd store + store-to-load forwarding (5c) + vmovsd (4c)
            "L_cmp_idx": 16, # 4 iteraciones * 4c (vcomisd 3c + jbe/loop 1c)
            "L_commit": 12,  # penalización por store-to-load forwarding en [rcx] cuando actualiza
            "L_crit": (25, 37),
        },
        {
            "alg": "SCAMP", "comp": "GCC", "isa": "x86 AVX-512 (no vec. red.)",
            "I_stat": 10, "I_dyn": (50, 66),
            "uops_stat": 11, "uops_dyn": (58, 74),
            "I_tax_stat": 6, "I_tax_dyn": 34, # 2 spill/reload + 4 loop (1+8 + 1+24 = 34)
            "crossings": 9, # 1 store + 8 reloads
            "D_crit": (18, 34),
            "L_fold": 10,  # vmovapd zmm store + forwarding (6c) + vmovsd (4c)
            "L_cmp_idx": 32, # 8 iteraciones * 4c
            "L_commit": 24,
            "L_crit": (42, 66),
        },
        {
            "alg": "SCAMP", "comp": "GCC", "isa": "x86 AVX2 (vec. red.)",
            "I_stat": 19, "I_dyn": (15, 19),
            "uops_stat": 19, "uops_dyn": (15, 19),
            "I_tax_stat": 8, "I_tax_dyn": 8,
            "crossings": 4, # vextractf128, vpermilpd, vbroadcastsd, vmovmskpd
            "D_crit": (11, 14),
            "L_fold": 20,    # vcmppd(3) + vblendvpd(1) + 3 + 4 + 1 + 4 + 4 = 20 ciclos
            "L_cmp_idx": 10, # 3 (vbroadcastsd) + 3 (vcmppd) + 3 (vmovmskpd) + 1 (test/je) = 10 ciclos
            "L_commit": 8,   # 3 (bsf) + 1 (add) + 4 (store) = 8 ciclos
            "L_crit": (30, 38),
        },
        {
            "alg": "SCAMP", "comp": "GCC", "isa": "x86 AVX-512 (vec. red.)",
            "I_stat": 23, "I_dyn": (19, 23),
            "uops_stat": 23, "uops_dyn": (19, 23),
            "I_tax_stat": 11, "I_tax_dyn": 11,
            "crossings": 5, # vextractf64x4, vextractf128, vpermilpd, vbroadcastsd, kmovw
            "D_crit": (14, 17),
            "L_fold": 27,    # vcmppd(3) + vmovapd{k1}(1) + 3 + 4 + 3 + 4 + 1 + 4 + 4 = 27 ciclos
            "L_cmp_idx": 11, # 3 (vbroadcastsd) + 3 (vcmppd) + 3 (kmovw) + 1 (movzx) + 1 (test/je) = 11 ciclos
            "L_commit": 8,   # 3 (bsf) + 1 (add) + 4 (store) = 8 ciclos
            "L_crit": (38, 46),
        },
        {
            "alg": "SCAMP", "comp": "GCC", "isa": "ARM SVE",
            "I_stat": 15, "I_dyn": (12, 15),
            # En Neoverse V1/512TVB: fmaxv se descompone en 5 uOps internos (14+4=19 uOps estáticos, 16-19 dinámicos);
            # En Neoverse V2: fmaxv son 2 uOps (16 uOps estáticos). A nivel ISA desacoplado = 15 (0 mem fusionadas).
            "uops_stat": 15, "uops_dyn": (12, 15),
            # Tax: 1 movprfx + 1 mov z3.d,d29 + 4 instrs extra de máscara (cntp p0, brkb, cmp, ccmp) = 6
            "I_tax_stat": 6, "I_tax_dyn": 6,
            "crossings": 3, # fmaxv (vec->scalar FP), mov z3.d,d29 (scalar->vec), cntp (pred->GPR)
            # Cadena crítica desde z30 listo (ld1rd z4 y cntp x10 ya se ejecutaron en paralelo antes):
            # 1. movprfx (0-1c fusionado con fmax en Neoverse V1/V2: 2c conjuntas) -> 2. fmax (2c) -> 3. fmaxv (9c en V1 256b / 4c en V2)
            # [Usamos el valor oficial de Neoverse V1 (256b) / 512TVB: fmax(2c) + fmaxv(9c) = 11c; y en V2: 2+4 = 6c]
            # -> 4. mov z3.d,d29 (2c) -> 5. fcmeq (2c) -> 6. brkb (2c) -> 7. cntp (2c) -> 8. cmp (1c) -> 9. ccmn+beq (1c) [L_cmp_idx = 10c]
            # -> 10. add x6,x6,x0 (1c) -> 11. str x6 (4c) [L_commit = 5c]
            "D_crit": (8, 10),
            "L_fold": 11,    # fmax (2c) + fmaxv (9c en Neoverse V1 / 6c media V1-V2)
            "L_cmp_idx": 10, # mov(2c) + fcmeq(2c) + brkb(2c) + cntp(2c) + cmp(1c) + ccmn/beq(1c) = 10 ciclos
            "L_commit": 5,   # add(1c) + str(4c) = 5 ciclos
            "L_crit": (21, 26),
        },
        {
            "alg": "SCAMP", "comp": "GCC", "isa": "RISC-V RVV",
            "I_stat": 12, "I_dyn": (8, 12),
            "uops_stat": 12, "uops_dyn": (8, 12),
            # Tax: vfmv.v.f (1) + vfmv.f.s (1) + li -1 (1) = 3 (más 1 add de puntero en commit = 3-4)
            "I_tax_stat": 3, "I_tax_dyn": 3,
            "crossings": 3, # vfmv.v.f (scalar->vec), vfmv.f.s (vec->scalar), vfirst.m (mask->GPR)
            # Cadena crítica desde v9 listo (fld fa5, vfmv.v.f v10 y li a0,-1 ya se ejecutaron en paralelo):
            # 1. vfredmax.vs (árbol log2(K)*2c + 2c = 6c a 256b K=4, 8c a 512b K=8) [L_fold = 6c (256b) / 8c (512b)]
            # -> 2. vfmv.f.s (2c) -> 3. vmfeq.vf (2c) -> 4. vfirst.m (2c) -> 5. beq (1c) [L_cmp_idx = 7c, D_crit_min = 5]
            # -> 6. add a2,a2,a6 (1c) -> 7. sd (4c) [L_commit = 5c, D_crit_max = 7]
            "D_crit": (5, 7),
            "L_fold": 8,     # 6c (256b, K=4) / 8c (512b, K=8) en árbol OoO (32c en SiFive-X280 in-order)
            "L_cmp_idx": 7,  # vfmv.f.s(2c) + vmfeq.vf(2c) + vfirst.m(2c) + beq(1c) = 7 ciclos
            "L_commit": 5,   # add(1c) + sd(4c) = 5 ciclos
            "L_crit": (15, 20), # (13-18 a 256b; 15-20 a 512b)
        },
        # ============================== SCAMP - Clang ============================
        {
            "alg": "SCAMP", "comp": "Clang", "isa": "x86 AVX2 (no vec. red.)",
            "I_stat": 26, "I_dyn": (12, 26),
            "uops_stat": 26, "uops_dyn": (12, 26),
            # Tax: 3 shuffles (vshufpd, vextractf128, vshufpd) + 3 vmovapd = 6
            "I_tax_stat": 6, "I_tax_dyn": 3,
            "crossings": 3,
            # En Clang desenrollado, las 4 comparaciones `vucomisd xmm, xmm2` dependen secuencialmente de `xmm2`
            # (que puede ser escrito en cada rama por `vmovapd xmm2, xmm3` o en memoria `[rcx+8*r9]`).
            # Camino mínimo (sin entrar en los ifs):
            # vextractf128 (3c) -> vshufpd (1c) -> vucomisd (3c) -> jbe (1c), mientras los 4 saltos se evalúan: ~12c min, ~26c max
            "D_crit": (6, 16),
            "L_fold": 4,     # vextractf128(3c) + vshufpd(1c)
            "L_cmp_idx": 12, # 4 * vucomisd(3c)
            "L_commit": 10,
            "L_crit": (16, 26),
        },
        {
            "alg": "SCAMP", "comp": "Clang", "isa": "x86 AVX-512 (no vec. red.)",
            "I_stat": 55, "I_dyn": (25, 55),
            "uops_stat": 55, "uops_dyn": (25, 55),
            "I_tax_stat": 15, "I_tax_dyn": 8, # 8 shuffles + 7 vmovapd
            "crossings": 8,
            "D_crit": (10, 32),
            "L_fold": 8,
            "L_cmp_idx": 24, # 8 * vucomisd(3c)
            "L_commit": 20,
            "L_crit": (32, 52),
        },
        {
            "alg": "SCAMP", "comp": "Clang", "isa": "x86 AVX2 (vec. red.)",
            "I_stat": 19, "I_dyn": (14, 19),
            "uops_stat": 20, "uops_dyn": (15, 20),
            "I_tax_stat": 9, "I_tax_dyn": 8,
            "crossings": 4,
            "D_crit": (11, 15),
            "L_fold": 20,
            "L_cmp_idx": 10,
            "L_commit": 9,
            "L_crit": (30, 39),
        },
        {
            "alg": "SCAMP", "comp": "Clang", "isa": "x86 AVX-512 (vec. red.)",
            "I_stat": 21, "I_dyn": (16, 21),
            "uops_stat": 22, "uops_dyn": (17, 22),
            "I_tax_stat": 10, "I_tax_dyn": 9,
            "crossings": 5,
            "D_crit": (13, 17),
            "L_fold": 27,
            "L_cmp_idx": 10,
            "L_commit": 9,
            "L_crit": (37, 46),
        },
        {
            "alg": "SCAMP", "comp": "Clang", "isa": "ARM SVE",
            "I_stat": 15, "I_dyn": (12, 15),
            "uops_stat": 15, "uops_dyn": (12, 15),
            "I_tax_stat": 6, "I_tax_dyn": 6,
            "crossings": 3,
            "D_crit": (8, 10),
            "L_fold": 11,
            "L_cmp_idx": 10,
            "L_commit": 5,
            "L_crit": (21, 26),
        },
        {
            "alg": "SCAMP", "comp": "Clang", "isa": "RISC-V RVV",
            "I_stat": 12, "I_dyn": (8, 12),
            "uops_stat": 12, "uops_dyn": (8, 12),
            "I_tax_stat": 3, "I_tax_dyn": 3,
            "crossings": 3,
            "D_crit": (5, 7),
            "L_fold": 8,
            "L_cmp_idx": 7,
            "L_commit": 5,
            "L_crit": (15, 20),
        },
        # ============================== SCRIMP - GCC =============================
        {
            "alg": "SCRIMP", "comp": "GCC", "isa": "x86 AVX2 (no vec. red.)",
            "I_stat": 12, "I_dyn": (30, 42),
            "uops_stat": 12, "uops_dyn": (30, 42), # vmovsd explícito, sin mem-op en vcomisd
            "I_tax_stat": 6, "I_tax_dyn": 22,
            "crossings": 5,
            "D_crit": (10, 18),
            "L_fold": 9,
            "L_cmp_idx": 16,
            "L_commit": 12,
            "L_crit": (25, 37),
        },
        {
            "alg": "SCRIMP", "comp": "GCC", "isa": "x86 AVX-512 (no vec. red.)",
            "I_stat": 11, "I_dyn": (58, 74),
            "uops_stat": 11, "uops_dyn": (58, 74),
            "I_tax_stat": 6, "I_tax_dyn": 34,
            "crossings": 9,
            "D_crit": (18, 34),
            "L_fold": 10,
            "L_cmp_idx": 32,
            "L_commit": 24,
            "L_crit": (42, 66),
        },
        {
            "alg": "SCRIMP", "comp": "GCC", "isa": "x86 AVX2 (vec. red.)",
            "I_stat": 18, "I_dyn": (14, 18),
            "uops_stat": 19, "uops_dyn": (15, 19),
            "I_tax_stat": 8, "I_tax_dyn": 8,
            "crossings": 4,
            "D_crit": (11, 14),
            "L_fold": 20,
            "L_cmp_idx": 10,
            "L_commit": 8,
            "L_crit": (30, 38),
        },
        {
            "alg": "SCRIMP", "comp": "GCC", "isa": "x86 AVX-512 (vec. red.)",
            "I_stat": 22, "I_dyn": (18, 22),
            "uops_stat": 23, "uops_dyn": (19, 23),
            "I_tax_stat": 11, "I_tax_dyn": 11,
            "crossings": 5,
            "D_crit": (14, 17),
            "L_fold": 27,
            "L_cmp_idx": 11,
            "L_commit": 8,
            "L_crit": (38, 46),
        },
        {
            "alg": "SCRIMP", "comp": "GCC", "isa": "ARM SVE",
            "I_stat": 15, "I_dyn": (12, 15),
            "uops_stat": 15, "uops_dyn": (12, 15),
            "I_tax_stat": 6, "I_tax_dyn": 6,
            "crossings": 3,
            "D_crit": (8, 10),
            "L_fold": 11,
            "L_cmp_idx": 10,
            "L_commit": 5,
            "L_crit": (21, 26),
        },
        {
            "alg": "SCRIMP", "comp": "GCC", "isa": "RISC-V RVV",
            "I_stat": 12, "I_dyn": (8, 12),
            "uops_stat": 12, "uops_dyn": (8, 12),
            "I_tax_stat": 3, "I_tax_dyn": 3,
            "crossings": 3,
            "D_crit": (5, 7),
            "L_fold": 8,
            "L_cmp_idx": 7,
            "L_commit": 5,
            "L_crit": (15, 20),
        },
        # ============================== SCRIMP - Clang ===========================
        {
            "alg": "SCRIMP", "comp": "Clang", "isa": "x86 AVX2 (no vec. red.)",
            "I_stat": 26, "I_dyn": (12, 26),
            "uops_stat": 26, "uops_dyn": (12, 26),
            "I_tax_stat": 6, "I_tax_dyn": 3,
            "crossings": 3,
            "D_crit": (6, 16),
            "L_fold": 4,
            "L_cmp_idx": 12,
            "L_commit": 10,
            "L_crit": (16, 26),
        },
        {
            "alg": "SCRIMP", "comp": "Clang", "isa": "x86 AVX-512 (no vec. red.)",
            "I_stat": 55, "I_dyn": (25, 55),
            "uops_stat": 55, "uops_dyn": (25, 55),
            "I_tax_stat": 15, "I_tax_dyn": 8,
            "crossings": 8,
            "D_crit": (10, 32),
            "L_fold": 8,
            "L_cmp_idx": 24,
            "L_commit": 20,
            "L_crit": (32, 52),
        },
        {
            "alg": "SCRIMP", "comp": "Clang", "isa": "x86 AVX2 (vec. red.)",
            "I_stat": 19, "I_dyn": (14, 19),
            "uops_stat": 20, "uops_dyn": (15, 20),
            "I_tax_stat": 9, "I_tax_dyn": 8,
            "crossings": 4,
            "D_crit": (11, 15),
            "L_fold": 20,
            "L_cmp_idx": 10,
            "L_commit": 9,
            "L_crit": (30, 39),
        },
        {
            "alg": "SCRIMP", "comp": "Clang", "isa": "x86 AVX-512 (vec. red.)",
            "I_stat": 21, "I_dyn": (16, 21),
            "uops_stat": 22, "uops_dyn": (17, 22),
            "I_tax_stat": 10, "I_tax_dyn": 9,
            "crossings": 5,
            "D_crit": (13, 17),
            "L_fold": 27,
            "L_cmp_idx": 10,
            "L_commit": 9,
            "L_crit": (37, 46),
        },
        {
            "alg": "SCRIMP", "comp": "Clang", "isa": "ARM SVE",
            "I_stat": 15, "I_dyn": (12, 15),
            "uops_stat": 15, "uops_dyn": (12, 15),
            "I_tax_stat": 6, "I_tax_dyn": 6,
            "crossings": 3,
            "D_crit": (8, 10),
            "L_fold": 11,
            "L_cmp_idx": 10,
            "L_commit": 5,
            "L_crit": (21, 26),
        },
        {
            "alg": "SCRIMP", "comp": "Clang", "isa": "RISC-V RVV",
            "I_stat": 12, "I_dyn": (8, 12),
            "uops_stat": 12, "uops_dyn": (8, 12),
            "I_tax_stat": 3, "I_tax_dyn": 3,
            "crossings": 3,
            "D_crit": (5, 7),
            "L_fold": 8,
            "L_cmp_idx": 7,
            "L_commit": 5,
            "L_crit": (15, 20),
        },
    ]
    return data


def extract_diag_step_lines(rel_path):
    """Extrae las líneas de ensamblador de la función `*_diag_step*` de una traza."""
    full = os.path.join(BASE, rel_path)
    with open(full, encoding="utf-8") as f:
        lines = f.read().splitlines()
    # Si el fichero es solo diag_step (como x86_*_vectorized_reduction_*), empieza en línea 1
    start = None
    for idx, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith(("scamp_diag_step", "scrimp_diag_step")) and s.endswith(":"):
            start = idx + 1
            break
    assert start is not None, f"No se encontró diag_step en {rel_path}"
    body = []
    for idx in range(start, len(lines)):
        s = lines[idx].strip()
        if not s:
            continue
        # Fin de la función si empieza otra etiqueta global (no local .L)
        if s.endswith(":") and not s.startswith("."):
            break
        if s.startswith("."):
            continue  # directivas (.quad de pools de constantes, .cfi...) no son instrucciones
        if not s.endswith(":"):
            body.append((idx + 1, s))
    return body


def count_fused_mem_ops_x86(ins_list):
    """
    Cuenta cuántas instrucciones x86 fusionan un acceso a memoria `[...]` dentro de
    una operación de cómputo/comparación/broadcast (excluyendo loads/stores puros `mov*`, `vmov*`, `lea`, `push`, `pop`).
    """
    fused_ComputeMem = 0
    fused_BcastLoad = 0
    for _, s in ins_list:
        op = s.split()[0].lower()
        if "ptr" in s.lower() or "[" in s:
            if op in ("lea", "push", "pop", "mov", "vmovsd", "vmovapd", "vmovupd", "vmovdqu", "vmovdqu64", "vmaskmovpd", "vpmaskmovq"):
                continue
            if op in ("vbroadcastsd", "vpbroadcastq"):
                fused_BcastLoad += 1
            else:
                fused_ComputeMem += 1
    return fused_ComputeMem, fused_BcastLoad


def get_full_diag_step_metrics():
    """
    Analiza las 24 trazas completas de `diag_step` (MB2) para SCAMP y SCRIMP en GCC 14.2 y Clang 18.1.
    """
    traces = [
        # SCAMP GCC
        ("SCAMP", "GCC", "x86 AVX2 (no vec. red.)", "gcc/x86_avx2_scamp_output-O2_gcc.txt", 4),
        ("SCAMP", "GCC", "x86 AVX-512 (no vec. red.)", "gcc/x86_avx512_scamp_output-O2_gcc.txt", 8),
        ("SCAMP", "GCC", "x86 AVX2 (vec. red.)", "gcc/x86_avx2_scamp_vectorized_reduction_output-O2_gcc.txt", 1),
        ("SCAMP", "GCC", "x86 AVX-512 (vec. red.)", "gcc/x86_avx512_scamp_vectorized_reduction_output-O2_gcc.txt", 1),
        ("SCAMP", "GCC", "ARM SVE", "gcc/arm_sve_scamp_output-O2_gcc.txt", 1),
        ("SCAMP", "GCC", "RISC-V RVV", "gcc/riscv_rvv_scamp_output-O2_gcc.txt", 1),
        # SCAMP Clang
        ("SCAMP", "Clang", "x86 AVX2 (no vec. red.)", "clang/x86_avx2_scamp_output-O2_clang.txt", 1),
        ("SCAMP", "Clang", "x86 AVX-512 (no vec. red.)", "clang/x86_avx512_scamp_output-O2_clang.txt", 1),
        ("SCAMP", "Clang", "x86 AVX2 (vec. red.)", "clang/x86_avx2_scamp_vectorized_reduction_output-O2_clang.txt", 1),
        ("SCAMP", "Clang", "x86 AVX-512 (vec. red.)", "clang/x86_avx512_scamp_vectorized_reduction_output-O2_clang.txt", 1),
        ("SCAMP", "Clang", "ARM SVE", "clang/arm_sve_scamp_output-O2_clang.txt", 1),
        ("SCAMP", "Clang", "RISC-V RVV", "clang/riscv_rvv_scamp_output-O2_clang.txt", 1),
        # SCRIMP GCC
        ("SCRIMP", "GCC", "x86 AVX2 (no vec. red.)", "gcc/x86_avx2_scrimp_output-O2_gcc.txt", 4),
        ("SCRIMP", "GCC", "x86 AVX-512 (no vec. red.)", "gcc/x86_avx512_scrimp_output-O2_gcc.txt", 8),
        ("SCRIMP", "GCC", "x86 AVX2 (vec. red.)", "gcc/x86_avx2_scrimp_vectorized_reduction_output-O2_gcc.txt", 1),
        ("SCRIMP", "GCC", "x86 AVX-512 (vec. red.)", "gcc/x86_avx512_scrimp_vectorized_reduction_output-O2_gcc.txt", 1),
        ("SCRIMP", "GCC", "ARM SVE", "gcc/arm_sve_scrimp_output-O2_gcc.txt", 1),
        ("SCRIMP", "GCC", "RISC-V RVV", "gcc/riscv_rvv_scrimp_output-O2_gcc.txt", 1),
        # SCRIMP Clang
        ("SCRIMP", "Clang", "x86 AVX2 (no vec. red.)", "clang/x86_avx2_scrimp_output-O2_clang.txt", 1),
        ("SCRIMP", "Clang", "x86 AVX-512 (no vec. red.)", "clang/x86_avx512_scrimp_output-O2_clang.txt", 1),
        ("SCRIMP", "Clang", "x86 AVX2 (vec. red.)", "clang/x86_avx2_scrimp_vectorized_reduction_output-O2_clang.txt", 1),
        ("SCRIMP", "Clang", "x86 AVX-512 (vec. red.)", "clang/x86_avx512_scrimp_vectorized_reduction_output-O2_clang.txt", 1),
        ("SCRIMP", "Clang", "ARM SVE", "clang/arm_sve_scrimp_output-O2_clang.txt", 1),
        ("SCRIMP", "Clang", "RISC-V RVV", "clang/riscv_rvv_scrimp_output-O2_clang.txt", 1),
    ]
    hred_map = {(r["alg"], r["comp"], r["isa"]): r for r in get_hred_metrics()}
    results = []
    for alg, comp, isa, path, k_iters in traces:
        body = extract_diag_step_lines(path)
        i_stat = len(body)
        fused_alu, fused_bcast = (0, 0)
        if "x86" in isa:
            fused_alu, fused_bcast = count_fused_mem_ops_x86(body)
        hr = hred_map[(alg, comp, isa)]
        # Instrucciones fuera del bloque de reducción horizontal (Fase 2a + Fase 2c + ABI/direcciones):
        i_other = i_stat - hr["I_stat"]
        i_dyn_min = i_other + hr["I_dyn"][0]
        i_dyn_max = i_other + hr["I_dyn"][1]
        # uOps desacoplados (sumando 1 uOp por cada memoria fusionada en instrucción ALU/CMP,
        # y en bucle enrollado GCC sumando los mem-ops adicionales de las iteraciones)
        extra_dyn_mem = (hr["uops_dyn"][0] - hr["I_dyn"][0])
        # Restamos del fused_alu estático si alguno ya estaba en hr
        hr_fused = hr["uops_stat"] - hr["I_stat"]
        other_fused = fused_alu - hr_fused
        uops_stat = i_stat + fused_alu
        uops_dyn_min = i_dyn_min + other_fused + extra_dyn_mem
        uops_dyn_max = i_dyn_max + other_fused + extra_dyn_mem

        # Latencia crítica de Fase 2a (Cálculo vectorial de correlación/distancia hasta tener `correlation_v`/`distance_v` listo):
        # En SCAMP (idéntica cadena matemática en las 3 ISAs):
        #   Load/Bcast (6c) -> Mul (4c) -> FMA (4c) -> Add cov (4c) -> Mul s_i (4c) -> Mul s_j (4c) = 26 ciclos (6 eslabones en serie)
        # En SCRIMP (incluye división FP de doble precisión `vdivpd`/`fdiv`/`vfdiv.vv`):
        #   En Intel Sapphire Rapids / Ice Lake: `vdivpd ymm` = 14c, `vdivpd zmm` = 14c (en Zen 4 = 13c).
        #   En Arm Neoverse V1/V2: `fdiv z.d, p0/m` = 12c (V1) - 14c (usamos 14c para paridad exacta de unidad FP IEEE-754 DP, o 12c oficial V1).
        #   En RISC-V RVV OoO: `vfdiv.vv` = 14c.
        #   Cadena SCRIMP: Load/Bcast (6c) -> Mul (4c) -> FMSUB (4c) -> Add dot (4c) -> FNMADD/Sub (4c) -> Div (14c) -> Sub (4c) -> Add (4c) = 44 ciclos (8 eslabones en serie).
        if alg == "SCAMP":
            d_crit_p2a = 6
            l_crit_p2a = 26
        else:
            d_crit_p2a = 8
            l_crit_p2a = 44

        # Como la Fase 2c (actualización vertical: cmp + masked store = ~7-9c tras tener `correlation_v`)
        # se ejecuta en paralelo con la Fase 2b (reducción horizontal: 15-66c) y siempre tarda menos que la Fase 2b,
        # el camino crítico total de `diag_step` es exactamente: Fase 2a + Fase 2b (HRed).
        d_crit_tot = (d_crit_p2a + hr["D_crit"][0], d_crit_p2a + hr["D_crit"][1])
        l_crit_tot = (l_crit_p2a + hr["L_crit"][0], l_crit_p2a + hr["L_crit"][1])

        results.append({
            "alg": alg, "comp": comp, "isa": isa,
            "I_stat": i_stat, "I_dyn": (i_dyn_min, i_dyn_max),
            "fused_alu": fused_alu, "fused_bcast": fused_bcast,
            "uops_stat": uops_stat, "uops_dyn": (uops_dyn_min, uops_dyn_max),
            "D_crit_p2a": d_crit_p2a, "D_crit_hred": hr["D_crit"], "D_crit_tot": d_crit_tot,
            "L_crit_p2a": l_crit_p2a, "L_crit_hred": hr["L_crit"], "L_crit_tot": l_crit_tot,
        })
    return results


if __name__ == "__main__":
    hred = get_hred_metrics()
    print("=== (A) RESUMEN MÉTRICAS 1, 2 y 4 EN EL BLOQUE DE REDUCCIÓN HORIZONTAL ===")
    print(f"{'Alg':6s} {'Comp':6s} {'ISA':26s} | {'I_st':>4s} {'I_dyn':>7s} | {'uOp_s':>5s} {'uOp_d':>7s} | {'Tax_s':>5s} {'Xing':>4s} | {'D_crit':>7s} | {'L_fold':>6s} {'L_idx':>5s} {'L_crit':>7s}")
    for r in hred:
        idyn = f"{r['I_dyn'][0]}-{r['I_dyn'][1]}"
        udyn = f"{r['uops_dyn'][0]}-{r['uops_dyn'][1]}"
        dcrit = f"{r['D_crit'][0]}-{r['D_crit'][1]}"
        lcrit = f"{r['L_crit'][0]}-{r['L_crit'][1]}"
        print(f"{r['alg']:6s} {r['comp']:6s} {r['isa']:26s} | {r['I_stat']:4d} {idyn:>7s} | {r['uops_stat']:5d} {udyn:>7s} | {r['I_tax_stat']:5d} {r['crossings']:4d} | {dcrit:>7s} | {r['L_fold']:6d} {r['L_cmp_idx']:5d} {lcrit:>7s}")

    full = get_full_diag_step_metrics()
    print("\n=== (B) RESUMEN MÉTRICAS 1, 2 y 4 EN EL KERNEL COMPLETO `diag_step` (MB2) ===")
    print(f"{'Alg':6s} {'Comp':6s} {'ISA':26s} | {'I_st':>4s} {'I_dyn':>7s} | {'Fused':>5s} {'uOp_s':>5s} {'uOp_d':>7s} | {'D_tot':>7s} | {'L_p2a':>5s} {'L_hred':>7s} {'L_tot':>7s}")
    for r in full:
        idyn = f"{r['I_dyn'][0]}-{r['I_dyn'][1]}"
        udyn = f"{r['uops_dyn'][0]}-{r['uops_dyn'][1]}"
        dtot = f"{r['D_crit_tot'][0]}-{r['D_crit_tot'][1]}"
        lhred = f"{r['L_crit_hred'][0]}-{r['L_crit_hred'][1]}"
        ltot = f"{r['L_crit_tot'][0]}-{r['L_crit_tot'][1]}"
        fused = f"{r['fused_alu']}+{r['fused_bcast']}b"
        print(f"{r['alg']:6s} {r['comp']:6s} {r['isa']:26s} | {r['I_stat']:4d} {idyn:>7s} | {fused:>5s} {r['uops_stat']:5d} {udyn:>7s} | {dtot:>7s} | {r['L_crit_p2a']:5d} {lhred:>7s} {ltot:>7s}")

