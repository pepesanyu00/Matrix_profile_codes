#!/usr/bin/env python3
"""
cfg_dynamic_counts.py
=====================
Cálculo REPRODUCIBLE y exacto (sobre el CFG real del ensamblador) del número
de instrucciones estáticas y dinámicas de cada función de las trazas.

Para cada función se construye el grafo de flujo de control a nivel de
instrucción (x86-64 Intel, AArch64, RISC-V) y se calcula:

  static   : nº de instrucciones del listado (sin etiquetas ni directivas).
  dyn_min / dyn_max : mínimo / máximo de instrucciones ejecutadas en UNA
                      invocación "de un paso", definida así:
        * Bucle principal (el más externo): se ejecuta EXACTAMENTE una
          iteración (el salto hacia atrás nunca se toma; el camino sale del
          bucle en esa misma iteración, ya sea por el latch o por una
          salida intermedia). Se excluye el camino de "cero iteraciones"
          (guarda de entrada): el camino debe pasar por la cabecera.
        * Bucle de búsqueda secuencial de V0 en GCC x86 (el más interno):
          se ejecuta con su recuento real de iteraciones (4 en AVX2, 8 en
          AVX-512), con todas las combinaciones de ramas internas.
        * En funciones SIN bucle (p. ej. diag_step) el rango es simplemente
          el camino más corto / más largo desde la entrada hasta `ret`.
  setup / iter : (solo funciones con bucle principal) descomposición
          dyn = setup (prólogo+epílogo, 1 vez por llamada)
              + iter  (cuerpo del bucle principal, 1 vez por iteración).

Uso:  python3 tools/cfg_dynamic_counts.py [--md] [--json salida.json]
"""
import glob
import json
import os
import re
import sys
from functools import lru_cache

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACE_DIR = os.path.join(ROOT, "outputs", "trazas microbenchmarks completos")

PHASES = [("init_", "PH1"), ("diag_step", "PH2"), ("diag_loop", "PH3"),
          ("thread_reduce", "PH4")]

X86_COND = re.compile(r"^j(?!mp$)[a-z]+$")
RV_COND = {"beq", "bne", "blt", "bge", "bltu", "bgeu", "ble", "bgt", "bleu",
           "bgtu", "beqz", "bnez", "blez", "bgez", "bltz", "bgtz"}
A64_COND_REG = {"cbz", "cbnz", "tbz", "tbnz"}


def isa_of(fname):
    f = os.path.basename(fname)
    if f.startswith("x86"):
        return "x86"
    if f.startswith("arm"):
        return "arm"
    return "riscv"


def parse_functions(path):
    """Devuelve {nombre_funcion: [lineas_crudas]} (solo funciones globales)."""
    funcs, cur = {}, None
    for lineno, raw in enumerate(open(path, encoding="utf-8", errors="replace"), 1):
        line = raw.rstrip("\r\n")
        s = line.strip()
        if not s:
            continue
        m = re.match(r"^([A-Za-z_][\w$]*):\s*(#.*)?$", s)
        if m and not line.startswith((" ", "\t")):
            cur = m.group(1)
            funcs[cur] = []
            continue
        if cur is not None:
            funcs[cur].append((lineno, s))
    return funcs


def build_cfg(lines, isa):
    """Instrucciones, etiquetas->idx, sucesores (idx_dest, tipo)."""
    instrs, labels, linenos = [], {}, []
    for lineno, s in lines:
        s = re.sub(r"\s(#|//|;).*$", "", s).strip()
        if not s or s.startswith("#"):
            continue
        m = re.match(r"^(\.?[\w$.]+):$", s)
        if m:
            labels[m.group(1)] = len(instrs)
            continue
        if s.startswith("."):          # directivas (.cfi, .size, .quad ...)
            continue
        instrs.append(s)
        linenos.append(lineno)
    n = len(instrs)
    succ = [[] for _ in range(n)]
    term = [False] * n
    for i, s in enumerate(instrs):
        parts = s.replace(",", " ").split()
        mn = parts[0].lower()
        ops = parts[1:]
        target = ops[-1] if ops else None
        tgt = labels.get(target) if target else None
        kind = None
        if isa == "x86":
            if mn == "ret":
                kind = "ret"
            elif mn == "jmp":
                kind = "jmp"
            elif X86_COND.match(mn):
                kind = "cond"
        elif isa == "arm":
            if mn == "ret":
                kind = "ret"
            elif mn == "b":
                kind = "jmp"
            elif (mn.startswith("b.") or mn in A64_COND_REG
                  or re.match(r"^b(eq|ne|cs|cc|hs|lo|mi|pl|vs|vc|hi|ls|ge|lt|gt|le)$", mn)):
                kind = "cond"
            elif mn == "br":
                kind = "ret"
        else:
            if mn in ("ret", "jr"):
                kind = "ret"
            elif mn == "j":
                kind = "jmp"
            elif mn in RV_COND:
                kind = "cond"
            elif mn in ("tail",):
                kind = "ret"
        if kind == "ret":
            term[i] = True
        elif kind == "jmp":
            if tgt is None:
                term[i] = True          # salto a función externa = salida
            else:
                succ[i].append((tgt, "taken"))
        elif kind == "cond":
            if tgt is not None:
                succ[i].append((tgt, "taken"))
            if i + 1 < n:
                succ[i].append((i + 1, "fall"))
        else:
            if i + 1 < n:
                succ[i].append((i + 1, "fall"))
            else:
                term[i] = True
    build_cfg.linenos = linenos
    return instrs, succ, term


def find_back_edges(succ, n):
    color = [0] * n
    back = set()
    stack = [(0, iter(succ[0]))]
    color[0] = 1
    while stack:
        u, it = stack[-1]
        adv = False
        for v, _ in it:
            if color[v] == 1:
                back.add((u, v))
            elif color[v] == 0:
                color[v] = 1
                stack.append((v, iter(succ[v])))
                adv = True
                break
        if not adv:
            color[u] = 2
            stack.pop()
    return back


def natural_loops(succ, n, back):
    """Agrupa por cabecera. Devuelve {header: (cuerpo:set, latches:set)}."""
    pred = [[] for _ in range(n)]
    for u in range(n):
        for v, _ in succ[u]:
            pred[v].append(u)
    loops = {}
    for (u, h) in back:
        body = {h}
        work = [u]
        while work:
            x = work.pop()
            if x in body:
                continue
            body.add(x)
            work.extend(pred[x])
        b, l = loops.get(h, (set(), set()))
        loops[h] = (b | body, l | {u})
    return loops


def analyse(instrs, succ, term, search_trip=None, wset=None):
    n = len(instrs)
    res = {"static": n}
    if n == 0:
        return None
    back = find_back_edges(succ, n)
    loops = natural_loops(succ, n, back)
    # profundidad de cada bucle = nº de bucles que lo contienen estrictamente
    depth = {}
    for h, (b, _) in loops.items():
        depth[h] = sum(1 for h2, (b2, _) in loops.items()
                       if h2 != h and b < b2)
    maxd = max(depth.values()) if depth else 0
    outer = [h for h in loops if depth[h] == 0]
    # bucles con recuento fijo
    fixed = {}
    if search_trip:
        for h in loops:
            if depth[h] == maxd and (maxd > 0 or not outer or True):
                fixed[h] = search_trip
    main_loops = [h for h in outer if h not in fixed]
    main_body = set()
    main_latches = set()
    main_headers = set(main_loops)
    for h in main_loops:
        main_body |= loops[h][0]
        main_latches |= loops[h][1]
    latch_of = {}                       # latch -> header (para contar)
    for h, (b, l) in loops.items():
        for u in l:
            latch_of[u] = h
    fixed_hdrs = sorted(fixed)

    def weight(i, mode):
        if wset is not None:
            return 1 if i in wset else 0
        if mode == "setup" and i in main_body:
            return 0
        return 1

    def solve(mode):
        """mode: 'full' | 'setup'. Camino entrada->ret con restricciones."""
        sys.setrecursionlimit(100000)
        NEED = bool(main_loops)

        @lru_cache(maxsize=None)
        def f(i, cnts, seen):
            # cnts: tupla de contadores de back-edges de bucles fijos
            w = weight(i, mode)
            if i in main_headers:
                seen = True
            best = []
            if term[i]:
                if (not NEED) or seen:
                    ok = all(cnts[k] in (0, fixed[fixed_hdrs[k]] - 1)
                             for k in range(len(fixed_hdrs)))
                    if ok:
                        best.append((w, w))
                return (min(b[0] for b in best), max(b[1] for b in best)) \
                    if best else None
            outs = []
            for v, kind in succ[i]:
                is_back = (i, v) in back
                c2 = cnts
                if is_back:
                    h = v
                    if h in fixed:
                        k = fixed_hdrs.index(h)
                        if cnts[k] >= fixed[h] - 1:
                            continue
                        c2 = cnts[:k] + (cnts[k] + 1,) + cnts[k + 1:]
                    else:
                        continue          # bucle principal: no se retoma
                else:
                    # salir de un bucle fijo por el latch => exige T-1 vueltas
                    if i in latch_of and latch_of[i] in fixed:
                        k = fixed_hdrs.index(latch_of[i])
                        if cnts[k] != fixed[latch_of[i]] - 1:
                            continue
                r = f(v, c2, seen)
                if r:
                    outs.append(r)
            if not outs:
                return None
            return (w + min(o[0] for o in outs), w + max(o[1] for o in outs))

        return f(0, tuple(0 for _ in fixed_hdrs), False)

    full = solve("full")
    if full is None:
        return None
    res["dyn_min"], res["dyn_max"] = full
    res["loops"] = len(loops)
    res["fixed_trip"] = search_trip if fixed else None
    if main_loops:
        s = solve("setup")
        res["setup"] = s
        # iter: de la cabecera al latch dentro del cuerpo, incluyendo latch

        @lru_cache(maxsize=None)
        def g(i, cnts):
            w = 1
            outs = []
            if i in main_latches:
                outs.append((0, 0))     # tomar el back-edge: fin de iteración
            for v, kind in succ[i]:
                if (i, v) in back:
                    if v in fixed:
                        k = fixed_hdrs.index(v)
                        if cnts[k] >= fixed[v] - 1:
                            continue
                        c2 = cnts[:k] + (cnts[k] + 1,) + cnts[k + 1:]
                    else:
                        continue
                else:
                    c2 = cnts
                    if i in latch_of and latch_of[i] in fixed:
                        k = fixed_hdrs.index(latch_of[i])
                        if cnts[k] != fixed[latch_of[i]] - 1:
                            continue
                if v not in main_body:
                    continue
                r = g(v, c2)
                if r:
                    outs.append(r)
            if not outs:
                return None
            return (w + min(o[0] for o in outs), w + max(o[1] for o in outs))
        its = [g(h, tuple(0 for _ in fixed_hdrs)) for h in main_loops]
        its = [x for x in its if x]
        if its:
            res["iter"] = (min(x[0] for x in its), max(x[1] for x in its))
    return res


def phase_of(fn):
    for key, ph in PHASES:
        if key in fn:
            return ph
    return None


def run():
    rows = []
    for comp in ("gcc", "clang"):
        for path in sorted(glob.glob(os.path.join(TRACE_DIR, comp, "*.txt"))):
            base = os.path.basename(path)
            isa = isa_of(base)
            algo = "SCAMP" if "scamp" in base else "SCRIMP"
            funcs = parse_functions(path)
            for fn, lines in funcs.items():
                ph = phase_of(fn)
                if not ph:
                    continue
                instrs, succ, term = build_cfg(lines, isa)
                trip = None
                v0 = ("avx" in base and "vectorized" not in base
                      and comp == "gcc" and ph in ("PH2", "PH3"))
                if v0:
                    trip = 8 if "avx512" in base else 4
                r = analyse(instrs, succ, term, trip)
                rows.append({"comp": comp, "file": base, "isa": isa,
                             "algo": algo, "fn": fn, "ph": ph, "res": r})
    return rows


def fmt(r):
    if not r:
        return "n/d"
    s = f"{r['static']} ({r['dyn_min']}–{r['dyn_max']})"
    if "iter" in r:
        su = r["setup"]
        s += f" [Setup {su[0]}–{su[1]} | Iter {r['iter'][0]}–{r['iter'][1]}]"
    return s


if __name__ == "__main__":
    rows = run()
    if "--json" in sys.argv:
        out = sys.argv[sys.argv.index("--json") + 1]
        json.dump(rows, open(out, "w"), indent=1, default=list)
    for x in rows:
        print(f"{x['comp']:5} {x['file'][:-4]:55} {x['fn']:42} "
              f"{x['ph']} {fmt(x['res'])}")
