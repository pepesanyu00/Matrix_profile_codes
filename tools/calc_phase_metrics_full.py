#!/usr/bin/env python3
"""
tools/calc_phase_metrics_full.py
Genera el desglose completo de métricas microarquitectónicas para las 4 fases:
  - PH1: init_window (MB1)
  - PH2: diag_step (MB2)
  - PH3: diag_loop iteración (MB3-Iter)
  - PH4: thread_reduce (MB4)
para SCAMP y SCRIMP en GCC 14.2 y Clang 18.1.
"""
import sys

def get_phase_data():
    rows = []
    
    # =========================================================================
    # SCAMP - GCC 14.2
    # =========================================================================
    rows.append({
        "alg": "SCAMP", "comp": "GCC", "isa": "x86 AVX2 (vec. red.)", "vl": 256, "k": 4,
        "PH1": {"I_stat": 25, "I_dyn": (22, 22), "fused": 3, "uops_dyn": (25, 25), "D_crit": (4, 4), "tax": 2},
        "PH2": {"I_stat": 43, "I_dyn": (39, 43), "fused": 7, "uops_dyn": (46, 50), "D_crit": (17, 20), "tax": 8},
        "PH3_iter": {"I_stat": 55, "I_dyn": (31, 36), "fused": 8, "uops_dyn": (39, 44), "D_crit": (18, 21), "tax": 8},
        "PH4": {"I_stat": 31, "I_dyn": (22, 22), "fused": 2, "uops_dyn": (24, 24), "D_crit": (3, 3), "tax": 1},
    })
    rows.append({
        "alg": "SCAMP", "comp": "GCC", "isa": "x86 AVX-512 (vec. red.)", "vl": 512, "k": 8,
        "PH1": {"I_stat": 25, "I_dyn": (22, 22), "fused": 3, "uops_dyn": (25, 25), "D_crit": (4, 4), "tax": 2},
        "PH2": {"I_stat": 44, "I_dyn": (40, 44), "fused": 7, "uops_dyn": (47, 51), "D_crit": (20, 23), "tax": 11},
        "PH3_iter": {"I_stat": 54, "I_dyn": (33, 38), "fused": 8, "uops_dyn": (41, 46), "D_crit": (21, 24), "tax": 11},
        "PH4": {"I_stat": 27, "I_dyn": (22, 22), "fused": 2, "uops_dyn": (24, 24), "D_crit": (3, 3), "tax": 1},
    })
    rows.append({
        "alg": "SCAMP", "comp": "GCC", "isa": "ARM SVE", "vl": 512, "k": 8, # VLA
        "PH1": {"I_stat": 22, "I_dyn": (20, 20), "fused": 0, "uops_dyn": (20, 20), "D_crit": (5, 5), "tax": 3},
        "PH2": {"I_stat": 57, "I_dyn": (54, 57), "fused": 0, "uops_dyn": (54, 57), "D_crit": (14, 16), "tax": 6},
        "PH3_iter": {"I_stat": 62, "I_dyn": (51, 53), "fused": 0, "uops_dyn": (51, 53), "D_crit": (15, 17), "tax": 6},
        "PH4": {"I_stat": 34, "I_dyn": (28, 28), "fused": 0, "uops_dyn": (28, 28), "D_crit": (3, 3), "tax": 2},
    })
    rows.append({
        "alg": "SCAMP", "comp": "GCC", "isa": "RISC-V RVV", "vl": 512, "k": 8, # VLA
        "PH1": {"I_stat": 22, "I_dyn": (22, 22), "fused": 0, "uops_dyn": (22, 22), "D_crit": (3, 3), "tax": 1},
        "PH2": {"I_stat": 46, "I_dyn": (42, 46), "fused": 0, "uops_dyn": (42, 46), "D_crit": (11, 13), "tax": 3},
        "PH3_iter": {"I_stat": 66, "I_dyn": (43, 46), "fused": 0, "uops_dyn": (43, 46), "D_crit": (12, 14), "tax": 3},
        "PH4": {"I_stat": 27, "I_dyn": (27, 27), "fused": 0, "uops_dyn": (27, 27), "D_crit": (3, 3), "tax": 1},
    })

    # =========================================================================
    # SCRIMP - GCC 14.2
    # =========================================================================
    rows.append({
        "alg": "SCRIMP", "comp": "GCC", "isa": "x86 AVX2 (vec. red.)", "vl": 256, "k": 4,
        "PH1": {"I_stat": 15, "I_dyn": (15, 15), "fused": 2, "uops_dyn": (17, 17), "D_crit": (4, 4), "tax": 1},
        "PH2": {"I_stat": 59, "I_dyn": (55, 59), "fused": 10, "uops_dyn": (65, 69), "D_crit": (19, 22), "tax": 8},
        "PH3_iter": {"I_stat": 69, "I_dyn": (37, 41), "fused": 10, "uops_dyn": (47, 51), "D_crit": (20, 23), "tax": 8},
        "PH4": {"I_stat": 32, "I_dyn": (22, 22), "fused": 2, "uops_dyn": (24, 24), "D_crit": (3, 3), "tax": 1},
    })
    rows.append({
        "alg": "SCRIMP", "comp": "GCC", "isa": "x86 AVX-512 (vec. red.)", "vl": 512, "k": 8,
        "PH1": {"I_stat": 15, "I_dyn": (15, 15), "fused": 2, "uops_dyn": (17, 17), "D_crit": (4, 4), "tax": 1},
        "PH2": {"I_stat": 62, "I_dyn": (58, 62), "fused": 10, "uops_dyn": (68, 72), "D_crit": (22, 25), "tax": 11},
        "PH3_iter": {"I_stat": 72, "I_dyn": (40, 44), "fused": 10, "uops_dyn": (50, 54), "D_crit": (23, 26), "tax": 11},
        "PH4": {"I_stat": 32, "I_dyn": (22, 22), "fused": 2, "uops_dyn": (24, 24), "D_crit": (3, 3), "tax": 1},
    })
    rows.append({
        "alg": "SCRIMP", "comp": "GCC", "isa": "ARM SVE", "vl": 512, "k": 8,
        "PH1": {"I_stat": 18, "I_dyn": (16, 16), "fused": 0, "uops_dyn": (16, 16), "D_crit": (5, 5), "tax": 2},
        "PH2": {"I_stat": 75, "I_dyn": (72, 75), "fused": 0, "uops_dyn": (72, 75), "D_crit": (16, 18), "tax": 6},
        "PH3_iter": {"I_stat": 79, "I_dyn": (66, 68), "fused": 0, "uops_dyn": (66, 68), "D_crit": (17, 19), "tax": 6},
        "PH4": {"I_stat": 34, "I_dyn": (28, 28), "fused": 0, "uops_dyn": (28, 28), "D_crit": (3, 3), "tax": 2},
    })
    rows.append({
        "alg": "SCRIMP", "comp": "GCC", "isa": "RISC-V RVV", "vl": 512, "k": 8,
        "PH1": {"I_stat": 19, "I_dyn": (19, 19), "fused": 0, "uops_dyn": (19, 19), "D_crit": (3, 3), "tax": 1},
        "PH2": {"I_stat": 67, "I_dyn": (63, 67), "fused": 0, "uops_dyn": (63, 67), "D_crit": (13, 15), "tax": 3},
        "PH3_iter": {"I_stat": 94, "I_dyn": (55, 58), "fused": 0, "uops_dyn": (55, 58), "D_crit": (14, 16), "tax": 3},
        "PH4": {"I_stat": 27, "I_dyn": (27, 27), "fused": 0, "uops_dyn": (27, 27), "D_crit": (3, 3), "tax": 1},
    })

    return rows

def print_summary_table():
    rows = get_phase_data()
    print("=======================================================================================================================")
    print("TABLA DETALLADA DE EFICIENCIA POR FASES: PH2 (diag_step) y PH3 (Iteración Diag Loop)")
    print("=======================================================================================================================")
    print(f"{'Alg':6s} {'ISA':24s} | {'PH2 Idyn':>9s} {'PH2 uOps':>9s} {'PH2 D_cr':>8s} {'PH2 ILP':>7s} | {'PH3 Idyn':>9s} {'PH3 uOps':>9s} {'PH3 D_cr':>8s} {'PH3 ILP':>7s} | {'PH3 Idyn/K':>10s}")
    print("-----------------------------------------------------------------------------------------------------------------------")
    for r in rows:
        p2 = r["PH2"]
        p3 = r["PH3_iter"]
        p2_idyn = f"{p2['I_dyn'][0]}–{p2['I_dyn'][1]}"
        p2_uops = f"{p2['uops_dyn'][0]}–{p2['uops_dyn'][1]}"
        p2_dcrit = f"{p2['D_crit'][0]}–{p2['D_crit'][1]}"
        p2_ilp = f"{((p2['I_dyn'][0]+p2['I_dyn'][1])/(p2['D_crit'][0]+p2['D_crit'][1])):.2f}"
        
        p3_idyn = f"{p3['I_dyn'][0]}–{p3['I_dyn'][1]}"
        p3_uops = f"{p3['uops_dyn'][0]}–{p3['uops_dyn'][1]}"
        p3_dcrit = f"{p3['D_crit'][0]}–{p3['D_crit'][1]}"
        p3_ilp = f"{((p3['I_dyn'][0]+p3['I_dyn'][1])/(p3['D_crit'][0]+p3['D_crit'][1])):.2f}"

        k = r["k"]
        p3_per_elem = f"{p3['I_dyn'][0]/k:.2f}–{p3['I_dyn'][1]/k:.2f}"
        
        print(f"{r['alg']:6s} {r['isa']:24s} | {p2_idyn:>9s} {p2_uops:>9s} {p2_dcrit:>8s} {p2_ilp:>7s} | {p3_idyn:>9s} {p3_uops:>9s} {p3_dcrit:>8s} {p3_ilp:>7s} | {p3_per_elem:>10s}")

    print("\n=======================================================================================================================")
    print("ESCALABILIDAD VLA: INSTRUCCIONES DINÁMICAS POR ELEMENTO EN PH3 (Iteración Diag Loop)")
    print("=======================================================================================================================")
    print(f"{'Alg':6s} {'ISA':24s} | {'256b (K=4)':>12s} {'512b (K=8)':>12s} {'1024b (K=16)':>14s} {'2048b (K=32)':>14s} | {'VLA Nativo?':>11s}")
    print("-----------------------------------------------------------------------------------------------------------------------")
    for r in rows:
        p3 = r["PH3_iter"]
        mid_idyn = (p3['I_dyn'][0] + p3['I_dyn'][1]) / 2.0
        if "AVX2" in r["isa"]:
            c256 = f"{mid_idyn/4:.2f}"
            c512 = "N/A"
            c1024 = "N/A"
            c2048 = "N/A"
            vla = "No (Fijo 4)"
        elif "AVX-512" in r["isa"]:
            c256 = "N/A"
            c512 = f"{mid_idyn/8:.2f}"
            c1024 = "N/A"
            c2048 = "N/A"
            vla = "No (Fijo 8)"
        else: # SVE or RVV
            c256 = f"{mid_idyn/4:.2f}"
            c512 = f"{mid_idyn/8:.2f}"
            c1024 = f"{mid_idyn/16:.2f}"
            c2048 = f"{mid_idyn/32:.2f}"
            vla = "Sí (1 binario)"
        print(f"{r['alg']:6s} {r['isa']:24s} | {c256:>12s} {c512:>12s} {c1024:>14s} {c2048:>14s} | {vla:>11s}")

if __name__ == "__main__":
    print_summary_table()

