#!/usr/bin/env python3
"""Aplica cfg_dynamic_counts a ficheros .s locales: cfg_check_asm.py f.s [trip_V0]"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cfg_dynamic_counts as C
path = sys.argv[1]; trip = int(sys.argv[2]) if len(sys.argv) > 2 else None
for fn, lines in C.parse_functions(path).items():
    ph = C.phase_of(fn)
    if not ph: continue
    ins, succ, term = C.build_cfg(lines, "x86")
    print(f"{os.path.basename(path)[:-2]:28} {fn:42} {ph} {C.fmt(C.analyse(ins, succ, term, trip if ph in ('PH2','PH3') else None))}")
