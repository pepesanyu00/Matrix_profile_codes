#!/usr/bin/env python3
"""
Script unificado y riguroso para calcular los dos tipos de medias de speedup
a partir de las estadísticas de gem5 en stats_finales/arena/stats1 y stats2:

  - MÉTRICA A: Speedup respecto a 1 hilo secuencial (Global Speedup vs 1-thread scalar)
               Speedup(ts, vlen, th) = T_sec(ts, 1th) / T_vect(ts, vlen, th)
               (Tal y como se grafica en el eje Y en generate_graphics.py)

  - MÉTRICA B: Mejora pura de vectorización a igualdad de hilos (Improvement vs N-thread scalar)
               Improvement(ts, vlen, th) = vect_speedup / no_vect_speedup = T_sec(ts, th) / T_vect(ts, vlen, th)
               (Tal y como se calcula en OVERALL AVERAGES al final de generate_graphics.py)

Ambas métricas se evalúan sobre las 6 series temporales completas.
"""

import os
import sys
import argparse

DEFAULT_STATS_DIRS = [
    "stats_finales/arena/stats1",
    "stats_finales/arena/stats2",
]

ALGORITHMS = ["scamp", "scrimp"]

TIMESERIES = [
    "power-MPIII-SVF_n180000",
    "seismology-MPIII-SVE_n180000",
    "e0103_n180000",
    "penguin_sample_TutorialMPweb",
    "audio-MPIII-SVD",
    "human_activity-MPIII-SVC",
]

TS_DISPLAY_NAMES = {
    "power-MPIII-SVF_n180000": "power",
    "seismology-MPIII-SVE_n180000": "seismology",
    "e0103_n180000": "ecg",
    "penguin_sample_TutorialMPweb": "penguin",
    "audio-MPIII-SVD": "audio",
    "human_activity-MPIII-SVC": "human_activity",
}

THREADS = [1, 2, 4, 8, 16, 32, 64]


def extract_sim_seconds(stats_file):
    """Extrae el valor de simSeconds de un archivo stats.txt de gem5."""
    if not os.path.exists(stats_file):
        return None
    with open(stats_file, "r") as f:
        for line in f:
            if line.startswith("simSeconds"):
                parts = line.split()
                if len(parts) >= 2:
                    return float(parts[1])
    return None


def collect_raw_data(target_dirs, algorithm):
    """
    Recorre las carpetas de estadísticas indicadas y devuelve los tiempos medios (simSeconds)
    por arquitectura y modo, promediando entre stats1 y stats2 de manera idéntica a generate_graphics.py.
    """
    arm_no_vect_raw = {}
    arm_vect_raw = {}
    riscv_no_vect_raw = {}
    riscv_vect_raw = {}

    for stats_dir in target_dirs:
        if not os.path.exists(stats_dir):
            print(f"[ADVERTENCIA] No existe el directorio: {stats_dir}", file=sys.stderr)
            continue

        for arq_mode in sorted(os.listdir(stats_dir)):
            if "-" not in arq_mode:
                continue
            arq = arq_mode.split("-")[0]
            is_no_hor = ("no_hor" in arq_mode or "no_horizontal" in arq_mode)
            is_sec = ("-sec" in arq_mode)
            is_vect = ("-vect" in arq_mode and not is_no_hor)

            alg_dir = os.path.join(stats_dir, arq_mode, algorithm)
            if not os.path.exists(alg_dir):
                continue

            for ts in sorted(os.listdir(alg_dir)):
                ts_dir = os.path.join(alg_dir, ts)
                if not os.path.isdir(ts_dir):
                    continue

                if is_sec:
                    for thread in sorted(os.listdir(ts_dir)):
                        thread_number = thread.split("_")[0]
                        stats_file = os.path.join(ts_dir, thread, "stats.txt")
                        sim_seconds = extract_sim_seconds(stats_file)
                        if sim_seconds is not None:
                            if arq == "arm":
                                arm_no_vect_raw.setdefault(ts, {}).setdefault(thread_number, []).append(sim_seconds)
                            elif arq == "riscv":
                                riscv_no_vect_raw.setdefault(ts, {}).setdefault(thread_number, []).append(sim_seconds)

                elif is_vect:
                    for vlen in sorted(os.listdir(ts_dir)):
                        if "-" not in vlen:
                            continue
                        vlen_number = vlen.split("-")[1]
                        vlen_dir = os.path.join(ts_dir, vlen)
                        if not os.path.isdir(vlen_dir):
                            continue
                        for thread in sorted(os.listdir(vlen_dir)):
                            thread_number = thread.split("_")[0]
                            stats_file = os.path.join(vlen_dir, thread, "stats.txt")
                            sim_seconds = extract_sim_seconds(stats_file)
                            if sim_seconds is not None:
                                if arq == "arm":
                                    arm_vect_raw.setdefault(ts, {}).setdefault(vlen_number, {}).setdefault(thread_number, []).append(sim_seconds)
                                elif arq == "riscv":
                                    riscv_vect_raw.setdefault(ts, {}).setdefault(vlen_number, {}).setdefault(thread_number, []).append(sim_seconds)

    # Calculamos la media de tiempos de ejecución entre carpetas
    arm_no_vect_mean = {}
    riscv_no_vect_mean = {}
    for raw_dict, mean_dict in [(riscv_no_vect_raw, riscv_no_vect_mean), (arm_no_vect_raw, arm_no_vect_mean)]:
        for ts in raw_dict:
            mean_dict[ts] = {}
            for t_str, vals in raw_dict[ts].items():
                if vals:
                    mean_dict[ts][t_str] = [sum(vals) / len(vals)]

    arm_vect_mean = {}
    riscv_vect_mean = {}
    for raw_dict, mean_dict in [(riscv_vect_raw, riscv_vect_mean), (arm_vect_raw, arm_vect_mean)]:
        for ts in raw_dict:
            mean_dict[ts] = {}
            for vlen_str in raw_dict[ts]:
                mean_dict[ts][vlen_str] = {}
                for t_str, vals in raw_dict[ts][vlen_str].items():
                    if vals:
                        mean_dict[ts][vlen_str][t_str] = [sum(vals) / len(vals)]

    return arm_no_vect_mean, arm_vect_mean, riscv_no_vect_mean, riscv_vect_mean


def compute_all_metrics(arm_no_vect, arm_vect, riscv_no_vect, riscv_vect, timeseries_list):
    """
    Calcula simultáneamente:
      - Métricas A (Speedup vs 1 hilo secuencial)
      - Métricas B (Improvement vs N hilos no-vectorizados)
    """
    speedup_riscv_2k = {}
    speedup_riscv_16k = {}
    speedup_arm_2k = {}

    speedup_riscv_no_vect = {}
    speedup_arm_no_vect = {}

    vlen_fixed = "2048"
    vlen_16k = "16384"

    # Paso 1: Speedup vs 1-th secuencial
    for ts in timeseries_list:
        riscv_t1 = riscv_no_vect[ts]["1"][0] if ts in riscv_no_vect and "1" in riscv_no_vect[ts] else None
        arm_t1 = arm_no_vect[ts]["1"][0] if ts in arm_no_vect and "1" in arm_no_vect[ts] else None

        if riscv_t1 is not None:
            if ts in riscv_vect and vlen_fixed in riscv_vect[ts]:
                for thread in THREADS:
                    t_str = str(thread)
                    if t_str in riscv_vect[ts][vlen_fixed] and riscv_vect[ts][vlen_fixed][t_str]:
                        speedup_riscv_2k.setdefault(ts, {})[thread] = riscv_t1 / riscv_vect[ts][vlen_fixed][t_str][0]

            if ts in riscv_vect and vlen_16k in riscv_vect[ts]:
                for thread in THREADS:
                    t_str = str(thread)
                    if t_str in riscv_vect[ts][vlen_16k] and riscv_vect[ts][vlen_16k][t_str]:
                        speedup_riscv_16k.setdefault(ts, {})[thread] = riscv_t1 / riscv_vect[ts][vlen_16k][t_str][0]

            if ts in riscv_no_vect:
                for thread in THREADS:
                    t_str = str(thread)
                    if t_str in riscv_no_vect[ts] and riscv_no_vect[ts][thread_str := t_str]:
                        speedup_riscv_no_vect.setdefault(ts, {})[thread] = riscv_t1 / riscv_no_vect[ts][thread_str][0]

        if arm_t1 is not None:
            if ts in arm_vect and vlen_fixed in arm_vect[ts]:
                for thread in THREADS:
                    t_str = str(thread)
                    if t_str in arm_vect[ts][vlen_fixed] and arm_vect[ts][vlen_fixed][t_str]:
                        speedup_arm_2k.setdefault(ts, {})[thread] = arm_t1 / arm_vect[ts][vlen_fixed][t_str][0]

            if ts in arm_no_vect:
                for thread in THREADS:
                    t_str = str(thread)
                    if t_str in arm_no_vect[ts] and arm_no_vect[ts][thread_str := t_str]:
                        speedup_arm_no_vect.setdefault(ts, {})[thread] = arm_t1 / arm_no_vect[ts][thread_str][0]

    # Paso 2: Improvement (Métrica B = vect_speedup / no_vect_speedup)
    imp_riscv_2k = {}
    imp_riscv_16k = {}
    imp_arm_2k = {}

    for ts in timeseries_list:
        if ts in speedup_riscv_2k and ts in speedup_riscv_no_vect:
            for thread in THREADS:
                if thread in speedup_riscv_2k[ts] and thread in speedup_riscv_no_vect[ts]:
                    imp_riscv_2k.setdefault(ts, {})[thread] = (
                        speedup_riscv_2k[ts][thread] / speedup_riscv_no_vect[ts][thread]
                    )

        if ts in speedup_riscv_16k and ts in speedup_riscv_no_vect:
            for thread in THREADS:
                if thread in speedup_riscv_16k[ts] and thread in speedup_riscv_no_vect[ts]:
                    imp_riscv_16k.setdefault(ts, {})[thread] = (
                        speedup_riscv_16k[ts][thread] / speedup_riscv_no_vect[ts][thread]
                    )

        if ts in speedup_arm_2k and ts in speedup_arm_no_vect:
            for thread in THREADS:
                if thread in speedup_arm_2k[ts] and thread in speedup_arm_no_vect[ts]:
                    imp_arm_2k.setdefault(ts, {})[thread] = (
                        speedup_arm_2k[ts][thread] / speedup_arm_no_vect[ts][thread]
                    )

    return (
        (speedup_riscv_2k, speedup_riscv_16k, speedup_arm_2k),
        (imp_riscv_2k, imp_riscv_16k, imp_arm_2k),
    )


def print_table(matrix_dict, label, timeseries_list):
    """Imprime una tabla de métricas por serie temporal y número de hilos."""
    print(f"\n  {label}:")
    header = f"    {'Serie Temporal':<18} | " + " | ".join(f"{t:>6}th" for t in THREADS) + " |   Media TS"
    print(header)
    print("    " + "-" * (len(header) - 4))
    all_vals = []
    for ts in timeseries_list:
        short_name = TS_DISPLAY_NAMES.get(ts, ts)
        row_vals = []
        row_strs = []
        for t in THREADS:
            val = matrix_dict.get(ts, {}).get(t)
            if val is not None:
                row_vals.append(val)
                all_vals.append(val)
                row_strs.append(f"{val:7.2f}x")
            else:
                row_strs.append(f"{'Falta*':>8}")
        ts_mean = sum(row_vals) / len(row_vals) if row_vals else 0.0
        print(f"    {short_name:<18} | " + " | ".join(row_strs) + f" | {ts_mean:8.2f}x (n={len(row_vals)})")

    col_strs = []
    for t in THREADS:
        col_vals = [matrix_dict[ts][t] for ts in timeseries_list if ts in matrix_dict and t in matrix_dict[ts]]
        if col_vals:
            col_mean = sum(col_vals) / len(col_vals)
            col_strs.append(f"{col_mean:7.2f}x")
        else:
            col_strs.append(f"{'N/A':>8}")
    overall_mean = sum(all_vals) / len(all_vals) if all_vals else 0.0
    print("    " + "-" * (len(header) - 4))
    print(f"    {'Media por hilos':<18} | " + " | ".join(col_strs) + f" | {overall_mean:8.2f}x (n={len(all_vals)})")


def flatten_values(matrix_dict, timeseries_list):
    vals = []
    for ts in timeseries_list:
        if ts in matrix_dict:
            for t in THREADS:
                if t in matrix_dict[ts]:
                    vals.append(matrix_dict[ts][t])
    return vals


def main():
    parser = argparse.ArgumentParser(
        description="Calcula los dos tipos de medias de speedup sobre las 6 series completas: Métrica A y Métrica B."
    )
    parser.add_argument(
        "--stats_dirs",
        nargs="+",
        default=DEFAULT_STATS_DIRS,
        help="Directorios de estadísticas a promediar (por defecto: stats_finales/arena/stats1 y stats2)",
    )
    args = parser.parse_args()

    print("=" * 110)
    print(" COMPARACIÓN INTEGRAL DE MÉTRICAS DE SPEEDUP (LAS 6 SERIES TEMPORALES COMPLETAS)")
    print("=" * 110)
    print(f"Directorios procesados: {', '.join(args.stats_dirs)}")
    print(f"Series temporales (6) : {', '.join(TS_DISPLAY_NAMES[ts] for ts in TIMESERIES)}")
    print(f"Configuraciones hilos : {', '.join(str(t) for t in THREADS)}")

    data_by_alg = {}
    for alg in ALGORITHMS:
        a_nv, a_v, r_nv, r_v = collect_raw_data(args.stats_dirs, alg)
        speedup_tuple, imp_tuple = compute_all_metrics(a_nv, a_v, r_nv, r_v, TIMESERIES)
        data_by_alg[alg] = {
            "sp_r2k": speedup_tuple[0],
            "sp_r16k": speedup_tuple[1],
            "sp_a2k": speedup_tuple[2],
            "imp_r2k": imp_tuple[0],
            "imp_r16k": imp_tuple[1],
            "imp_a2k": imp_tuple[2],
        }

    # =========================================================================
    # SECCIÓN 1: MÉTRICA A (Speedup respecto a 1 hilo secuencial)
    # =========================================================================
    print("\n" + "#" * 110)
    print(" MÉTRICA A: SPEEDUP RESPECTO A 1 HILO SECUENCIAL (T_sec(1) / T_vect(N))")
    print(" [Incluye el escalado multihilo + la aceleración vectorial global]")
    print("#" * 110)

    sp_summary = {}
    global_sp_r2k = []
    global_sp_r16k = []
    global_sp_a2k = []

    for alg in ALGORITHMS:
        print(f"\n--- ALGORITMO: {alg.upper()} (MÉTRICA A) ---")
        d = data_by_alg[alg]
        print_table(d["sp_r2k"], f"[{alg.upper()}] RISC-V RVV (VLEN = 2048 / 2k)", TIMESERIES)
        print_table(d["sp_r16k"], f"[{alg.upper()}] RISC-V RVV (VLEN = 16384 / 16k)", TIMESERIES)
        print_table(d["sp_a2k"], f"[{alg.upper()}] ARM SVE (VLEN = 2048 / 2k)", TIMESERIES)

        v_r2k = flatten_values(d["sp_r2k"], TIMESERIES)
        v_r16k = flatten_values(d["sp_r16k"], TIMESERIES)
        v_a2k = flatten_values(d["sp_a2k"], TIMESERIES)

        global_sp_r2k.extend(v_r2k)
        global_sp_r16k.extend(v_r16k)
        global_sp_a2k.extend(v_a2k)

        sp_summary[alg] = {
            "r2k": (sum(v_r2k) / len(v_r2k), len(v_r2k)),
            "r16k": (sum(v_r16k) / len(v_r16k), len(v_r16k)),
            "a2k": (sum(v_a2k) / len(v_a2k), len(v_a2k)),
        }

    # =========================================================================
    # SECCIÓN 2: MÉTRICA B (Improvement a igualdad de hilos)
    # =========================================================================
    print("\n" + "#" * 110)
    print(" MÉTRICA B: MEJORA DE VECTORIZACIÓN A IGUALDAD DE HILOS (T_sec(N) / T_vect(N))")
    print(" [Aísla estrictamente la ganancia que aporta la extensión vectorial frente a no vectorizar]")
    print("#" * 110)

    imp_summary = {}
    global_imp_r2k = []
    global_imp_r16k = []
    global_imp_a2k = []

    for alg in ALGORITHMS:
        print(f"\n--- ALGORITMO: {alg.upper()} (MÉTRICA B) ---")
        d = data_by_alg[alg]
        print_table(d["imp_r2k"], f"[{alg.upper()}] RISC-V RVV (VLEN = 2048 vs Non-Vectorized)", TIMESERIES)
        print_table(d["imp_r16k"], f"[{alg.upper()}] RISC-V RVV (VLEN = 16384 vs Non-Vectorized)", TIMESERIES)
        print_table(d["imp_a2k"], f"[{alg.upper()}] ARM SVE (VLEN = 2048 vs Non-Vectorized)", TIMESERIES)

        v_r2k = flatten_values(d["imp_r2k"], TIMESERIES)
        v_r16k = flatten_values(d["imp_r16k"], TIMESERIES)
        v_a2k = flatten_values(d["imp_a2k"], TIMESERIES)

        global_imp_r2k.extend(v_r2k)
        global_imp_r16k.extend(v_r16k)
        global_imp_a2k.extend(v_a2k)

        imp_summary[alg] = {
            "r2k": (sum(v_r2k) / len(v_r2k), len(v_r2k)),
            "r16k": (sum(v_r16k) / len(v_r16k), len(v_r16k)),
            "a2k": (sum(v_a2k) / len(v_a2k), len(v_a2k)),
        }

    # =========================================================================
    # TABLA COMPARATIVA FINAL
    # =========================================================================
    print("\n" + "=" * 110)
    print(" RESUMEN COMPARATIVO FINAL DE AMBAS MÉTRICAS (6 SERIES TEMPORALES)")
    print("=" * 110)

    header = f"{'Implementación':<28} | {'SCAMP (Métrica A)':<18} | {'SCAMP (Métrica B)':<18} | {'SCRIMP (Métrica A)':<19} | {'SCRIMP (Métrica B)':<19}"
    print(header)
    print("-" * len(header))

    def row(impl_name, key):
        sa = f"{sp_summary['scamp'][key][0]:.2f}x"
        sb = f"{imp_summary['scamp'][key][0]:.2f}x"
        ra = f"{sp_summary['scrimp'][key][0]:.2f}x"
        rb = f"{imp_summary['scrimp'][key][0]:.2f}x"
        return f"{impl_name:<28} | {sa:<18} | {sb:<18} | {ra:<19} | {rb:<19}"

    print(row("RISC-V RVV (2k / 2048)", "r2k"))
    print(row("RISC-V RVV (16k / 16384)", "r16k"))
    print(row("ARM SVE    (2k / 2048)", "a2k"))
    print("-" * len(header))

    print("\n--- MEDIAS GLOBALES CONJUNTAS (SCAMP + SCRIMP) ---")
    print(f"{'Implementación':<28} | {'Métrica A (Global Speedup)':<28} | {'Métrica B (Pura Vectorización)':<28}")
    print("-" * 90)
    print(f"{'RISC-V RVV (2k / 2048)':<28} | {sum(global_sp_r2k)/len(global_sp_r2k):8.2f}x                   | {sum(global_imp_r2k)/len(global_imp_r2k):8.2f}x")
    print(f"{'RISC-V RVV (16k / 16384)':<28} | {sum(global_sp_r16k)/len(global_sp_r16k):8.2f}x                   | {sum(global_imp_r16k)/len(global_imp_r16k):8.2f}x")
    print(f"{'ARM SVE    (2k / 2048)':<28} | {sum(global_sp_a2k)/len(global_sp_a2k):8.2f}x                   | {sum(global_imp_a2k)/len(global_imp_a2k):8.2f}x")
    print("=" * 110)


if __name__ == "__main__":
    main()
