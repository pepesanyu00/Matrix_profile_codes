#!/usr/bin/env python3
"""
Script de validación de corrección numérica entre implementaciones del Matrix Profile
(Escalar vs. Vectorial, SVE vs. RVV, SCAMP vs. SCRIMP, y Con/Sin Reducción Horizontal).

Permite especificar el número de decimales de precisión (--decimals) para comparar:
1. Los valores del Matrix Profile (profile[i]) fila a fila.
2. Los índices del vecino más cercano escogido (profileIndex[i]) fila a fila.
3. El Motif global (el profile mínimo final escogido de toda la serie y su índice).
"""

import argparse
import glob
import math
import os
import sys
from typing import Dict, List, Optional, Tuple


def load_mp_csv(path: str) -> Tuple[List[float], List[int]]:
    """Carga un archivo CSV de salida de SCAMP/SCRIMP y devuelve (profile, profileIndex)."""
    profile: List[float] = []
    indices: List[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) == 4:
                profile.append(float(parts[2]))
                indices.append(int(parts[3]))
    return profile, indices


def compare_profiles(
    prof_a: List[float],
    idx_a: List[int],
    prof_b: List[float],
    idx_b: List[int],
    decimals: int = 6,
    skip_row0: bool = False,
) -> Dict[str, object]:
    """
    Compara dos salidas de Matrix Profile con una precisión de `decimals` decimales.
    """
    if len(prof_a) != len(prof_b):
        raise ValueError(f"Longitudes distintas: {len(prof_a)} vs {len(prof_b)}")

    start = 1 if skip_row0 else 0
    n = len(prof_a) - start
    tol = 10.0 ** (-decimals)

    val_round_matches = 0
    val_tol_matches = 0
    idx_exact_matches = 0
    idx_or_tie_matches = 0
    max_abs_err = 0.0
    sum_abs_err = 0.0
    nan_a = 0
    nan_b = 0

    min_a_val = math.inf
    min_a_row = -1
    min_a_nn = -1

    min_b_val = math.inf
    min_b_row = -1
    min_b_nn = -1

    for i in range(start, len(prof_a)):
        pa, ia = prof_a[i], idx_a[i]
        pb, ib = prof_b[i], idx_b[i]

        if math.isnan(pa):
            nan_a += 1
        if math.isnan(pb):
            nan_b += 1

        if not math.isnan(pa) and pa < min_a_val:
            min_a_val = pa
            min_a_row = i
            min_a_nn = ia

        if not math.isnan(pb) and pb < min_b_val:
            min_b_val = pb
            min_b_row = i
            min_b_nn = ib

        err = abs(pa - pb)
        if err > max_abs_err:
            max_abs_err = err
        sum_abs_err += err

        same_round = round(pa, decimals) == round(pb, decimals)
        same_tol = err <= tol
        same_idx = ia == ib

        if same_round:
            val_round_matches += 1
        if same_tol:
            val_tol_matches += 1
        if same_idx:
            idx_exact_matches += 1
        if same_idx or same_tol:
            idx_or_tie_matches += 1

    mae = sum_abs_err / n if n > 0 else 0.0

    # Comprobación del profile final global escogido (Motif global: min(profile))
    global_val_match_round = round(min_a_val, decimals) == round(min_b_val, decimals)
    global_val_match_tol = abs(min_a_val - min_b_val) <= tol
    global_row_match = min_a_row == min_b_row
    global_nn_match = min_a_nn == min_b_nn
    # Nota: un motif es un par simétrico (row, nn), por lo que (r, nn) == (nn, r) es el mismo par
    global_pair_match = {min_a_row, min_a_nn} == {min_b_row, min_b_nn}

    return {
        "n": n,
        "decimals": decimals,
        "tol": tol,
        "val_round_matches": val_round_matches,
        "val_round_pct": 100.0 * val_round_matches / n if n else 0.0,
        "val_tol_matches": val_tol_matches,
        "val_tol_pct": 100.0 * val_tol_matches / n if n else 0.0,
        "idx_exact_matches": idx_exact_matches,
        "idx_exact_pct": 100.0 * idx_exact_matches / n if n else 0.0,
        "idx_or_tie_matches": idx_or_tie_matches,
        "idx_or_tie_pct": 100.0 * idx_or_tie_matches / n if n else 0.0,
        "max_abs_err": max_abs_err,
        "mae": mae,
        "nan_a": nan_a,
        "nan_b": nan_b,
        "min_a": (min_a_val, min_a_row, min_a_nn),
        "min_b": (min_b_val, min_b_row, min_b_nn),
        "global_val_match_round": global_val_match_round,
        "global_val_match_tol": global_val_match_tol,
        "global_row_match": global_row_match,
        "global_nn_match": global_nn_match,
        "global_pair_match": global_pair_match,
    }


def find_csv(base_dir: str, impl: str, alg: str, dataset: str, vlen: Optional[str], threads: str) -> Optional[str]:
    if vlen:
        pattern = os.path.join(base_dir, impl, alg, dataset, vlen, threads, "*.csv")
    else:
        pattern = os.path.join(base_dir, impl, alg, dataset, threads, "*.csv")
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def print_table(title: str, rows: List[Tuple[str, str, Dict[str, object]]], decimals: int) -> None:
    print(f"\n{'=' * 138}")
    print(f" {title} (Precisión: {decimals} decimales | Tolerancia <= 1e-{decimals})")
    print(f"{'=' * 138}")
    header = (
        f"{'Caso / Dataset':<32} | {'Comp.':<24} | {'N':>7} | "
        f"{'Profile (Tol)':>22} | {'Profile Index Exacto':>22} | {'Max |Err|':>10} | "
        f"{'Motif Global':<12}"
    )
    print(header)
    print("-" * 138)
    for label, comp_name, res in rows:
        n = res["n"]
        val_str = f"{res['val_tol_pct']:6.2f}% ({res['val_tol_matches']}/{n})"
        idx_str = f"{res['idx_exact_pct']:6.2f}% ({res['idx_exact_matches']}/{n})"
        val_ok = "SI" if res["global_val_match_tol"] else "NO"
        pair_ok = "SI" if res["global_pair_match"] else "NO"
        motif_str = f"{val_ok} / {pair_ok}"
        print(
            f"{label:<32} | {comp_name:<24} | {n:>7d} | "
            f"{val_str:>22} | {idx_str:>22} | "
            f"{res['max_abs_err']:>10.3e} | {motif_str:<12}"
        )


def run_stats1_suite(
    stats_dir: str, decimals: int, vlen: str = "vlen-512", threads: str = "1_threads", skip_row0: bool = False
) -> None:
    datasets = sorted(os.listdir(os.path.join(stats_dir, "arm-sec", "scamp")))
    algorithms = ["scamp", "scrimp"]

    # 1. Vectorizado con Reducción Horizontal vs Vectorizado con Reducción Escalar (mismo subconjunto de diagonales)
    rows_vec_vs_nohor = []
    for arch_prefix, arch_label in [("arm", "ARM SVE"), ("riscv", "RISC-V RVV")]:
        for alg in algorithms:
            for ds in datasets:
                f_vec = find_csv(stats_dir, f"{arch_prefix}-vect", alg, ds, vlen, threads)
                f_nohor = find_csv(stats_dir, f"{arch_prefix}-vect-no_hor", alg, ds, vlen, threads)
                if f_vec and f_nohor:
                    pa, ia = load_mp_csv(f_nohor)
                    pb, ib = load_mp_csv(f_vec)
                    res = compare_profiles(pa, ia, pb, ib, decimals=decimals, skip_row0=skip_row0)
                    rows_vec_vs_nohor.append((f"{alg.upper()} - {ds[:22]}", f"{arch_label} no_hor vs vect", res))
    print_table(
        f"1. REDUCCIÓN ESCALAR (vect-no_hor) vs. REDUCCIÓN VECTORIZADA (vect) [{vlen}, {threads}]",
        rows_vec_vs_nohor,
        decimals,
    )

    # 2. Comparativa Inter-Arquitectura: ARM vs RISC-V (Escalar, Vectorizado, y No-Hor)
    rows_arch = []
    for mode, mode_label, vl in [
        ("sec", "Escalar (ARM vs RV)", None),
        ("vect", f"Vectorial ({vlen})", vlen),
        ("vect-no_hor", f"Vec No-Hor ({vlen})", vlen),
    ]:
        for alg in algorithms:
            for ds in datasets:
                f_arm = find_csv(stats_dir, f"arm-{mode}", alg, ds, vl, threads)
                f_rv = find_csv(stats_dir, f"riscv-{mode}", alg, ds, vl, threads)
                if f_arm and f_rv:
                    pa, ia = load_mp_csv(f_arm)
                    pb, ib = load_mp_csv(f_rv)
                    res = compare_profiles(pa, ia, pb, ib, decimals=decimals, skip_row0=skip_row0)
                    rows_arch.append((f"{alg.upper()} - {ds[:22]}", mode_label, res))
    print_table(
        f"2. COMPARATIVA INTER-ARQUITECTURA: ARM SVE vs. RISC-V RVV [{threads}]",
        rows_arch,
        decimals,
    )

    # 3. Comparativa Inter-Algoritmo: SCAMP vs SCRIMP en cada implementación
    rows_alg = []
    for impl, impl_label, vl in [
        ("arm-sec", "ARM Escalar", None),
        ("riscv-sec", "RISC-V Escalar", None),
        ("arm-vect", f"ARM SVE ({vlen})", vlen),
        ("riscv-vect", f"RISC-V RVV ({vlen})", vlen),
        ("arm-vect-no_hor", f"ARM No-Hor ({vlen})", vlen),
        ("riscv-vect-no_hor", f"RV No-Hor ({vlen})", vlen),
    ]:
        for ds in datasets:
            f_scamp = find_csv(stats_dir, impl, "scamp", ds, vl, threads)
            f_scrimp = find_csv(stats_dir, impl, "scrimp", ds, vl, threads)
            if f_scamp and f_scrimp:
                pa, ia = load_mp_csv(f_scamp)
                pb, ib = load_mp_csv(f_scrimp)
                res = compare_profiles(pa, ia, pb, ib, decimals=decimals, skip_row0=skip_row0)
                rows_alg.append((f"{ds[:30]}", impl_label, res))
    print_table(
        f"3. COMPARATIVA ENTRE ALGORITMOS: SCAMP vs. SCRIMP [{threads}]",
        rows_alg,
        decimals,
    )

    # 4. Comparativa directa en stats1: Escalar (sec, pdiags=10% diag. sueltas) vs Vectorial (vect, pdiags=10% bloques VLMAX)
    rows_sec_vs_vec = []
    for arch_prefix, arch_label in [("arm", "ARM"), ("riscv", "RISC-V")]:
        for alg in algorithms:
            for ds in datasets:
                f_sec = find_csv(stats_dir, f"{arch_prefix}-sec", alg, ds, None, threads)
                f_vec = find_csv(stats_dir, f"{arch_prefix}-vect", alg, ds, vlen, threads)
                if f_sec and f_vec:
                    pa, ia = load_mp_csv(f_sec)
                    pb, ib = load_mp_csv(f_vec)
                    res = compare_profiles(pa, ia, pb, ib, decimals=decimals, skip_row0=skip_row0)
                    rows_sec_vs_vec.append((f"{alg.upper()} - {ds[:22]}", f"{arch_label} sec vs {vlen}", res))
    print_table(
        f"4. ESCALAR (sec) vs. VECTORIAL (vect) DIRECTO EN stats1 (NOTA: pdiags=10% con SHUFFLE_DIAGS) [{vlen}, {threads}]",
        rows_sec_vs_vec,
        decimals,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comprueba la corrección entre versiones escalares y vectoriales del Matrix Profile."
    )
    parser.add_argument(
        "--file-a",
        type=str,
        default=None,
        help="Ruta al primer archivo CSV (por ejemplo, versión escalar).",
    )
    parser.add_argument(
        "--file-b",
        type=str,
        default=None,
        help="Ruta al segundo archivo CSV (por ejemplo, versión vectorial).",
    )
    parser.add_argument(
        "--stats-dir",
        type=str,
        default="/home/jsanchez/Escritorio/revision_vectorizacion/godbolt_kernels/stats1",
        help="Directorio raíz stats1 para ejecución masiva.",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=6,
        help="Número de decimales a tener en cuenta en la comparación del profile (default: 6).",
    )
    parser.add_argument(
        "--vlen",
        type=str,
        default="vlen-512",
        help="Longitud vectorial representativa a usar en modo stats1 (default: vlen-512).",
    )
    parser.add_argument(
        "--threads",
        type=str,
        default="1_threads",
        help="Número de hilos representativo a usar en modo stats1 (default: 1_threads).",
    )
    parser.add_argument(
        "--skip-row0",
        action="store_true",
        help="Omitir el índice i=0 al comparar (útil para aislar el caso i=0 en arm-vect/scrimp).",
    )

    args = parser.parse_args()

    if args.file_a and args.file_b:
        pa, ia = load_mp_csv(args.file_a)
        pb, ib = load_mp_csv(args.file_b)
        res = compare_profiles(pa, ia, pb, ib, decimals=args.decimals, skip_row0=args.skip_row0)
        print(f"Comparando:\n  A: {args.file_a}\n  B: {args.file_b}")
        print(f"Elementos comparados: {res['n']} (Precisión: {args.decimals} decimales, tol={res['tol']:.1e})")
        print(f"  Coincidencia Profile (redondeo a {args.decimals} dec.): {res['val_round_matches']}/{res['n']} ({res['val_round_pct']:.4f}%)")
        print(f"  Coincidencia Profile (|A - B| <= 1e-{args.decimals}):  {res['val_tol_matches']}/{res['n']} ({res['val_tol_pct']:.4f}%)")
        print(f"  Coincidencia exacta de Profile Index:         {res['idx_exact_matches']}/{res['n']} ({res['idx_exact_pct']:.4f}%)")
        print(f"  Coincidencia Index o Empate a {args.decimals} dec.:       {res['idx_or_tie_matches']}/{res['n']} ({res['idx_or_tie_pct']:.4f}%)")
        print(f"  Error absoluto máximo: {res['max_abs_err']:.6e} | MAE: {res['mae']:.6e}")
        ma = res["min_a"]
        mb = res["min_b"]
        print(f"  Motif Global A: valor={ma[0]:.{args.decimals}f} en fila={ma[1]}, vecino={ma[2]}")
        print(f"  Motif Global B: valor={mb[0]:.{args.decimals}f} en fila={mb[1]}, vecino={mb[2]}")
        print(f"  ¿Mismo Motif Global escogido?: Valor={res['global_val_match_tol']}, Par de índices={res['global_pair_match']}")
    else:
        run_stats1_suite(
            args.stats_dir,
            decimals=args.decimals,
            vlen=args.vlen,
            threads=args.threads,
            skip_row0=args.skip_row0,
        )


if __name__ == "__main__":
    main()
