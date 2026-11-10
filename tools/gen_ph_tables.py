#!/usr/bin/env python3
"""Genera las tablas PH1-PH4 (estático (din. mín–máx) [Setup | Iter]) a partir
de tools/cfg_dynamic_counts.py (CFG real de las trazas)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cfg_dynamic_counts as C

rows = C.run()
idx = {}
for r in rows:
    idx[(r["comp"], r["file"].replace("_output-O2_%s.txt" % r["comp"], ""), r["fn"])] = r["res"]

def get(comp, stem, ph, fn_filter=None):
    for (c, s, fn), res in idx.items():
        if c == comp and s == stem and C.phase_of(fn) == ph and (fn_filter is None or fn_filter(fn)) and "builtin" not in fn:
            return res
    return None

def cell(res, loopinfo=False):
    if not res: return "—"
    s = f"**{res['static']} ({res['dyn_min']}–{res['dyn_max']})**"
    if loopinfo and "iter" in res:
        s += f" *[Setup {res['setup'][0]}{'' if res['setup'][0]==res['setup'][1] else '–'+str(res['setup'][1])} \\| Iter {res['iter'][0]}–{res['iter'][1]}]*"
    return s

def stat(res): return "—" if not res else str(res["static"])

def variants(alg):
    a = alg
    return [
      ("Escalar x86-64 Baseline", f"x86_baseline_{a}_scalar", None),
      ("Escalar ARM64 Baseline", f"arm_baseline_{a}_scalar", None),
      ("Escalar RISC-V Baseline", f"riscv_baseline_{a}_scalar", None),
      ("x86 AVX2 (Prior Packed-SIMD V0)", f"x86_avx2_{a}", None),
      ("x86 AVX2 (Vectorized Reduction)", f"x86_avx2_{a}_vectorized_reduction", f"x86_avx2_{a}_vectorized_loop"),
      ("x86 AVX-512 (Prior Packed-SIMD V0)", f"x86_avx512_{a}", None),
      ("x86 AVX-512 (Vectorized Reduction)", f"x86_avx512_{a}_vectorized_reduction", f"x86_avx512_{a}_vectorized_loop"),
      ("ARM AArch64 SVE (`_x`)", f"arm_sve_{a}", None),
      ("RISC-V RV64 RVV", f"riscv_rvv_{a}", None),
    ]

def table(comp, alg):
    out = []
    for name, stem, loopstem in variants(alg):
        ph1 = get(comp, stem, "PH1"); ph2 = get(comp, stem, "PH2")
        ph3 = get(comp, loopstem or stem, "PH3"); ph4 = get(comp, stem, "PH4")
        out.append(f"| **{name}** | {stat(ph1)} | {cell(ph2)} | {cell(ph3, True)} | {stat(ph4)} |")
    return "\n".join(out)

def combined(alg):
    out = []
    for name, stem, loopstem in variants(alg):
        cells = []
        for comp in ("gcc", "clang"):
            ph1 = get(comp, stem, "PH1"); ph2 = get(comp, stem, "PH2")
            ph3 = get(comp, loopstem or stem, "PH3"); ph4 = get(comp, stem, "PH4")
            c3 = "—" if not ph3 else f"**{ph3['static']} ({ph3['dyn_min']}–{ph3['dyn_max']})**" + (f" *[{ph3['iter'][0]}–{ph3['iter'][1]}]*" if "iter" in ph3 else "")
            cells += [stat(ph1), cell(ph2), c3, stat(ph4)]
        out.append(f"| **{alg.upper()}** | **{name}** | " + " | ".join(cells) + " |")
    return "\n".join(out)

if __name__ == "__main__" and "--tables" not in sys.argv and False:
    pass

if __name__ == "__main__":
    for comp in ("gcc", "clang"):
        for alg in ("scamp", "scrimp"):
            print(f"### {comp} {alg}")
            print(table(comp, alg)); print()
    # reducciones dinamicas (punto medio)
    def mid(r): return (r["dyn_min"] + r["dyn_max"]) / 2
    for comp in ("gcc", "clang"):
        for alg in ("scamp", "scrimp"):
            for isa in ("avx2", "avx512"):
                v0 = get(comp, f"x86_{isa}_{alg}", "PH2"); vr = get(comp, f"x86_{isa}_{alg}_vectorized_reduction", "PH2")
                p3v0 = get(comp, f"x86_{isa}_{alg}", "PH3"); p3vr = get(comp, f"x86_{isa}_{alg}_vectorized_loop", "PH3")
                print(comp, alg, isa, "PH2 red: %.1f%%" % (100*(1-mid(vr)/mid(v0))), "| PH3 red: %.1f%%" % (100*(1-mid(p3vr)/mid(p3v0))),
                      "| iter red: %.1f%%" % (100*(1-(sum(p3vr['iter'])/2)/(sum(p3v0['iter'])/2))))
