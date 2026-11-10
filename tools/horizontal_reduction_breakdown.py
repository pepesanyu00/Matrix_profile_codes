"""
Taxonomia de instrucciones SOLO para el bloque de busqueda horizontal (idiom
`corr_max = REDMAX(vec, profile[i]); mask = (vec == corr_max); idx = FIRST(mask);
if (idx != -1) {profile[i] = corr_max; profileIndex[i] = j + idx;}`).

Cada traza se corta a mano (lineas exactas del fichero) y cada linea se asigna a una
categoria. El script imprime la instruccion asociada a cada linea para poder auditar.

Categorias:
  FOLD    : plegado de carriles (extract/shuffle/max/min vectorial, fmaxv/fminv, vfredmax/min)
  SEED    : semilla profile[i], copias de registro y transferencias vector<->escalar
  CMP     : (broadcast de corr_max si existe) + comparacion de igualdad a mascara
  IDX     : mascara -> indice (movmsk/kmov/test/bsf, brkb/cntp/cmp/ccmp, vfirst + centinela -1)
  COMMIT  : salto condicional + calculo de j+idx + 2 stores (if index != -1)
Se excluye el calculo de direcciones (lea/slli/add de punteros base), que no pertenece al idiom.
"""
import os

BASE = os.path.join(os.path.dirname(__file__), "..", "outputs", "trazas microbenchmarks completos")

def R(a, b):
    return list(range(a, b + 1))

# (algoritmo, compilador, isa, fichero, {categoria: [lineas]}, loop_iters_if_rolled)
ENTRIES = [
    # ------------------------------ SCAMP / GCC ------------------------------
    ("SCAMP", "GCC", "x86 AVX2 (no vec. red.)", "gcc/x86_avx2_scamp_output-O2_gcc.txt",
        {"FOLD": [53, 55], "SEED": [], "CMP": [56], "IDX": [34] + R(62, 64), "COMMIT": R(57, 60)}, 4),
    ("SCAMP", "GCC", "x86 AVX-512 (no vec. red.)", "gcc/x86_avx512_scamp_output-O2_gcc.txt",
        {"FOLD": [50, 52], "SEED": [], "CMP": [53], "IDX": [43] + R(58, 60), "COMMIT": R(54, 56)}, 8),
    ("SCAMP", "GCC", "x86 AVX2 (vec. red.)", "gcc/x86_avx2_scamp_vectorized_reduction_output-O2_gcc.txt",
        {"SEED": [6, 14, 21], "FOLD": [19, 20] + R(22, 26), "CMP": R(27, 28), "IDX": R(29, 30) + [32], "COMMIT": [31] + R(33, 35)}, 1),
    ("SCAMP", "GCC", "x86 AVX-512 (vec. red.)", "gcc/x86_avx512_scamp_vectorized_reduction_output-O2_gcc.txt",
        {"SEED": [13, 15, 20, 23], "FOLD": [18, 19, 21, 22] + R(24, 28), "CMP": R(29, 30), "IDX": R(31, 33) + [35], "COMMIT": [34] + R(36, 38)}, 1),
    ("SCAMP", "GCC", "ARM SVE", "gcc/arm_sve_scamp_output-O2_gcc.txt",
        {"SEED": [51, 59], "FOLD": R(60, 61), "CMP": R(62, 63), "IDX": [32] + R(64, 67), "COMMIT": R(68, 71)}, 1),
    ("SCAMP", "GCC", "RISC-V RVV", "gcc/riscv_rvv_scamp_output-O2_gcc.txt",
        {"SEED": [50, 53, 57], "FOLD": [56], "CMP": [58], "IDX": [51, 59], "COMMIT": R(60, 64)}, 1),
    # ------------------------------ SCAMP / Clang ----------------------------
    ("SCAMP", "Clang", "x86 AVX2 (no vec. red.)", "clang/x86_avx2_scamp_output-O2_clang.txt",
        {"FOLD": [69, 77, 85], "SEED": [62, 67, 75, 83], "CMP": [63, 70, 78, 86], "IDX": [],
         "COMMIT": R(64, 66) + R(71, 74) + R(79, 82) + R(87, 90)}, 1),
    ("SCAMP", "Clang", "x86 AVX-512 (no vec. red.)", "clang/x86_avx512_scamp_output-O2_clang.txt",
        {"FOLD": [68, 76, 84, 85, 93, 101, 109, 117],
         "SEED": [61, 66, 74, 82, 91, 99, 107, 115],
         "CMP": [62, 69, 77, 86, 94, 102, 110, 118],
         "IDX": [],
         "COMMIT": R(63, 65) + R(70, 73) + R(78, 81) + R(87, 90) + R(95, 98) + R(103, 106) + R(111, 114) + R(119, 122)}, 1),
    ("SCAMP", "Clang", "x86 AVX2 (vec. red.)", "clang/x86_avx2_scamp_vectorized_reduction_output-O2_clang.txt",
        {"SEED": [13, 15], "FOLD": [14, 16] + R(17, 21), "CMP": R(22, 23), "IDX": R(24, 25) + R(27, 28), "COMMIT": [26] + R(29, 31)}, 1),
    ("SCAMP", "Clang", "x86 AVX-512 (vec. red.)", "clang/x86_avx512_scamp_vectorized_reduction_output-O2_clang.txt",
        {"SEED": [12, 14], "FOLD": [13, 15] + R(16, 22), "CMP": R(23, 24), "IDX": R(25, 26) + R(28, 29), "COMMIT": [27] + R(30, 32)}, 1),
    ("SCAMP", "Clang", "ARM SVE", "clang/arm_sve_scamp_output-O2_clang.txt",
        {"SEED": [49, 50], "FOLD": R(51, 52), "CMP": R(53, 54), "IDX": [38] + R(55, 58), "COMMIT": R(59, 62)}, 1),
    ("SCAMP", "Clang", "RISC-V RVV", "clang/riscv_rvv_scamp_output-O2_clang.txt",
        {"SEED": [43, 47, 49], "FOLD": [48], "CMP": [50], "IDX": R(51, 52), "COMMIT": R(53, 57)}, 1),
    # ------------------------------ SCRIMP / GCC -----------------------------
    ("SCRIMP", "GCC", "x86 AVX2 (no vec. red.)", "gcc/x86_avx2_scrimp_output-O2_gcc.txt",
        {"FOLD": [53, 55], "SEED": [56], "CMP": [57], "IDX": [47] + R(63, 65), "COMMIT": R(58, 61)}, 4),
    ("SCRIMP", "GCC", "x86 AVX-512 (no vec. red.)", "gcc/x86_avx512_scrimp_output-O2_gcc.txt",
        {"FOLD": [54, 56], "SEED": [57], "CMP": [58], "IDX": [49] + R(63, 65), "COMMIT": R(59, 61)}, 8),
    ("SCRIMP", "GCC", "x86 AVX2 (vec. red.)", "gcc/x86_avx2_scrimp_vectorized_reduction_output-O2_gcc.txt",
        {"SEED": [30, 35], "FOLD": [33, 34] + R(36, 40), "CMP": R(41, 42), "IDX": R(43, 44) + [46], "COMMIT": [45] + R(47, 49)}, 1),
    ("SCRIMP", "GCC", "x86 AVX-512 (vec. red.)", "gcc/x86_avx512_scrimp_vectorized_reduction_output-O2_gcc.txt",
        {"SEED": [31, 35, 38], "FOLD": [33, 34, 36, 37] + R(39, 43), "CMP": R(44, 45), "IDX": R(46, 48) + [50], "COMMIT": [49] + R(51, 53)}, 1),
    ("SCRIMP", "GCC", "ARM SVE", "gcc/arm_sve_scrimp_output-O2_gcc.txt",
        {"SEED": [46, 73], "FOLD": R(74, 75), "CMP": R(76, 77), "IDX": [39] + R(78, 81), "COMMIT": R(82, 85)}, 1),
    ("SCRIMP", "GCC", "RISC-V RVV", "gcc/riscv_rvv_scrimp_output-O2_gcc.txt",
        {"SEED": [68, 71, 75], "FOLD": [74], "CMP": [76], "IDX": [69, 77], "COMMIT": R(78, 82)}, 1),
    # ------------------------------ SCRIMP / Clang ---------------------------
    ("SCRIMP", "Clang", "x86 AVX2 (no vec. red.)", "clang/x86_avx2_scrimp_output-O2_clang.txt",
        {"FOLD": [48, 56, 64], "SEED": [41, 46, 54, 62], "CMP": [42, 49, 57, 65], "IDX": [],
         "COMMIT": R(43, 45) + R(50, 53) + R(58, 61) + R(66, 69)}, 1),
    ("SCRIMP", "Clang", "x86 AVX-512 (no vec. red.)", "clang/x86_avx512_scrimp_output-O2_clang.txt",
        {"FOLD": [47, 55, 63, 64, 72, 80, 88, 96],
         "SEED": [40, 45, 53, 61, 70, 78, 86, 94],
         "CMP": [41, 48, 56, 65, 73, 81, 89, 97],
         "IDX": [],
         "COMMIT": R(42, 44) + R(49, 52) + R(57, 60) + R(66, 69) + R(74, 77) + R(82, 85) + R(90, 93) + R(98, 101)}, 1),
    ("SCRIMP", "Clang", "x86 AVX2 (vec. red.)", "clang/x86_avx2_scrimp_vectorized_reduction_output-O2_clang.txt",
        {"SEED": [24, 26], "FOLD": [25, 27] + R(28, 32), "CMP": R(33, 34), "IDX": R(35, 36) + R(38, 39), "COMMIT": [37] + R(40, 42)}, 1),
    ("SCRIMP", "Clang", "x86 AVX-512 (vec. red.)", "clang/x86_avx512_scrimp_vectorized_reduction_output-O2_clang.txt",
        {"SEED": [24, 26], "FOLD": [25, 27] + R(28, 34), "CMP": R(35, 36), "IDX": R(37, 38) + R(40, 41), "COMMIT": [39] + R(42, 44)}, 1),
    ("SCRIMP", "Clang", "ARM SVE", "clang/arm_sve_scrimp_output-O2_clang.txt",
        {"SEED": [62, 64], "FOLD": R(65, 66), "CMP": R(67, 68), "IDX": [37] + R(69, 72), "COMMIT": R(73, 76)}, 1),
    ("SCRIMP", "Clang", "RISC-V RVV", "clang/riscv_rvv_scrimp_output-O2_clang.txt",
        {"SEED": [59, 63, 65], "FOLD": [64], "CMP": [66], "IDX": R(67, 68), "COMMIT": R(69, 73)}, 1),
]

CATS = ["FOLD", "SEED", "CMP", "IDX", "COMMIT"]


def load(path):
    with open(os.path.join(BASE, path), encoding="utf-8") as f:
        return f.read().splitlines()


def main(verbose=True):
    rows = []
    for alg, comp, isa, path, cats, k_iters in ENTRIES:
        lines = load(path)
        seen = set()
        counts = {}
        if verbose:
            print(f"\n=== {alg} | {comp} | {isa} ({path}) ===")
        for cat in CATS:
            ls = sorted(cats.get(cat, []))
            counts[cat] = len(ls)
            for ln in ls:
                assert ln not in seen, f"linea duplicada {ln} en {path}"
                seen.add(ln)
                if verbose:
                    print(f"  {cat:6s} L{ln:<3d} {lines[ln - 1].strip()}")
        counts["TOTAL"] = sum(counts[c] for c in CATS)
        if k_iters > 1:
            # Bucle escalar enrollado en GCC (Prior V0):
            # 1 store vectorial pre-bucle (FOLD[0]) + 1 init de contador pre-bucle (IDX[0])
            # El resto se ejecuta k_iters veces; lo que va tras `jbe` en COMMIT es condicional.
            commit_sorted = sorted(cats["COMMIT"])
            assert lines[commit_sorted[0] - 1].split()[0] == "jbe"
            cond_commit = len(commit_sorted) - 1
            loop_uncond = (counts["FOLD"] - 1) + counts["SEED"] + counts["CMP"] + (counts["IDX"] - 1) + 1
            counts["DYN_MIN"] = 2 + k_iters * loop_uncond
            counts["DYN_MAX"] = counts["DYN_MIN"] + k_iters * cond_commit
        elif "no vec. red." in isa:
            # Cascada desenrollada en Clang (Prior V0):
            # En cada carril se ejecutan siempre FOLD, SEED[0] (carga inicial), CMP y los saltos `jbe`.
            jbes = [ln for ln in cats["COMMIT"] if lines[ln - 1].split()[0] == "jbe"]
            counts["DYN_MIN"] = counts["FOLD"] + 1 + counts["CMP"] + len(jbes)
            counts["DYN_MAX"] = counts["TOTAL"]
        else:
            # Camino dinamico minimo (sin actualizar el perfil): se salta todo lo que hay
            # DESPUES del salto condicional (je / beq / b.eq) dentro del bloque IDX+COMMIT.
            block = sorted(cats.get("IDX", []) + cats.get("COMMIT", []))
            br = [ln for ln in block if lines[ln - 1].split()[0] in ("je", "beq", "b.eq")]
            assert len(br) == 1, f"salto no unico en {path}"
            skipped = len([ln for ln in block if ln > br[0]])
            counts["DYN_MIN"] = counts["TOTAL"] - skipped
            counts["DYN_MAX"] = counts["TOTAL"]
        rows.append((alg, comp, isa, counts))
    print("\n\n=========== RESUMEN ===========")
    print(f"{'Alg':6s} {'Comp':6s} {'ISA':28s} TOTAL FOLD SEED CMP IDX COMMIT | PURE(FOLD+SEED) | DYN(min-max)")
    for alg, comp, isa, c in rows:
        print(f"{alg:6s} {comp:6s} {isa:28s} {c['TOTAL']:5d} {c['FOLD']:4d} {c['SEED']:4d} "
              f"{c['CMP']:3d} {c['IDX']:3d} {c['COMMIT']:6d} | {c['FOLD'] + c['SEED']:15d} | "
              f"{c['DYN_MIN']}-{c['DYN_MAX']}")


if __name__ == "__main__":
    main()
