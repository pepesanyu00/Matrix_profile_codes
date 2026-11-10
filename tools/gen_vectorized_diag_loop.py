#!/usr/bin/env python3
"""Genera microbenchmarks/<alg>_<isa>_vectorized_loop.cpp (PH3 con reducción
horizontal vectorizada) a partir de los ficheros *_vectorized.cpp existentes:
  * reutiliza literalmente el preámbulo (macros + árbol de shuffles hmax/hmin)
  * reutiliza literalmente el cuerpo de diag_step_*_vectorized (sin _builtin)
    envuelto en el mismo bucle que usa V0 en diag_loop (i = 1; j = diag+1..).
"""
import os, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MB = os.path.join(ROOT, "microbenchmarks")
for alg in ("scamp", "scrimp"):
    for isa in ("avx2", "avx512"):
        src = open(os.path.join(MB, f"{alg}_{isa}_vectorized.cpp")).read()
        pre = src[:src.index('extern "C" {')]
        fn = f"{alg}_diag_step_{isa}_vectorized("
        k = src.index(fn)
        body = src[src.index("{", src.index(")", k)):]
        body = body[1:body.index("    return ")]
        scrimp = alg == "scrimp"
        if scrimp:
            body = re.sub(r"\n\s*VDTYPE windowSize_v = [^\n]*\n", "\n", body)
        # indentar un nivel más (dentro del for)
        body = "\n".join(("    " + l if l.strip() else l) for l in body.split("\n"))
        acc = "dotProd_v" if scrimp else "covariance_v"
        args = open(os.path.join(MB, f"{alg}_{isa}.cpp")).read()
        sig_start = args.index(f"void {alg}_diag_loop_{isa}(")
        sig = args[sig_start:args.index(")", sig_start) + 1]
        sig = sig.replace(f"_diag_loop_{isa}(", f"_diag_loop_{isa}_vectorized(")
        pre_line = ("    VDTYPE windowSize_v = SET1_PD((double)windowSize);\n"
                    if scrimp else "")
        out = (pre.replace("Fichero:", "Fichero (PH3 generado por tools/gen_vectorized_diag_loop.py) — origen:")
               + 'extern "C" {\n\n// MB3: bucle completo de la diagonal con reducción horizontal VECTORIZADA\n'
               + "__attribute__((noinline))\n" + sig + "\n{\n" + pre_line
               + "    ITYPE i = 1;\n    for (ITYPE j = diag + 1; j < profileLength; j++)\n    {\n"
               + body.rstrip() + "\n        i++;\n    }\n}\n\n}\n")
        open(os.path.join(MB, f"{alg}_{isa}_vectorized_loop.cpp"), "w").write(out)
        print("generado", f"{alg}_{isa}_vectorized_loop.cpp")
