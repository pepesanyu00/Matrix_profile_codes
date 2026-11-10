#!/usr/bin/env python3
"""Verifica con el CFG real los rangos dinámicos del bloque de búsqueda
horizontal definidos en horizontal_reduction_breakdown.py (líneas auditadas)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cfg_dynamic_counts as C
import horizontal_reduction_breakdown as H

H_ROWS = []
for alg, comp, isa, path, cats, k in H.ENTRIES:
    full = os.path.join(H.BASE, path)
    isa_k = "x86" if "x86" in path else ("arm" if "arm" in path else "riscv")
    allln = set(l for v in cats.values() for l in v)
    funcs = C.parse_functions(full)
    for fn, lines in funcs.items():
        if "diag_step" not in fn or "builtin" in fn: continue
        nums = [n for n, _ in lines]
        if not (min(nums) <= min(allln) and max(allln) <= max(nums)): continue
        ins, succ, term = C.build_cfg(lines, isa_k)
        ln = C.build_cfg.linenos
        wset = {i for i, l in enumerate(ln) if l in allln}
        r = C.analyse(ins, succ, term, k if k > 1 else None, wset)
        H_ROWS.append((alg, comp, isa, len(allln), r["dyn_min"], r["dyn_max"]))
    # valores de la herramienta antigua (para comparar)
import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    H.main(verbose=False)
old = {}
for line in buf.getvalue().splitlines():
    if "|" in line and line.split()[0] in ("SCAMP", "SCRIMP"):
        parts = line.split()
        old[(parts[0], parts[1], " ".join(parts[2:-9]) if False else None)] = None
print(f"{'Alg':6} {'Comp':6} {'ISA':28} est  CFG(min-max)")
for r in H_ROWS:
    print(f"{r[0]:6} {r[1]:6} {r[2]:28} {r[3]:3d}  {r[4]}-{r[5]}")
