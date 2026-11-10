#!/usr/bin/env python3
"""Compila con la API pública de Godbolt (Compiler Explorer) las 4 funciones
PH3 con reducción vectorizada, con GCC 14.2 y Clang 18.1 (mismos compiladores
y flags que el resto de trazas) y guarda la salida en
outputs/trazas microbenchmarks completos/{gcc,clang}/ con el mismo esquema de
nombres que las demás trazas:  x86_<isa>_<alg>_vectorized_loop_output-O2_<cc>.txt
"""
import json, os, sys, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "trazas microbenchmarks completos")
API = "https://godbolt.org/api"


def get(url):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=60))


def find(names):
    comps = get(f"{API}/compilers/c++")
    for want in names:
        for c in comps:
            if c["name"].lower() == want and c["instructionSet"] == "amd64":
                return c["id"], c["name"]
    raise SystemExit("compilador no encontrado: %s" % names)


gcc_id, gcc_name = find(["x86-64 gcc 14.2"])
clang_id, clang_name = find(["x86-64 clang 18.1.0", "x86-64 clang 18.1"])
print("GCC:", gcc_id, gcc_name, "| Clang:", clang_id, clang_name)

for alg in ("scamp", "scrimp"):
    for isa, flag in (("avx2", "-mavx2 -mfma"), ("avx512", "-mavx512f -mfma")):
        for src_suffix, out_tag in (
            ("vectorized.cpp", "vectorized_reduction"),
            ("vectorized_loop.cpp", "vectorized_loop"),
        ):
            src = open(os.path.join(ROOT, "microbenchmarks",
                                    f"{alg}_{isa}_{src_suffix}")).read()
            for cc, cid, folder in (("gcc", gcc_id, "gcc"), ("clang", clang_id, "clang")):
                body = {"source": src, "options": {
                    "userArguments": f"-O2 {flag}",
                    "filters": {"intel": True, "labels": True, "directives": True,
                                "commentOnly": True, "binary": False,
                                "execute": False, "demangle": False,
                                "libraryCode": False, "trim": False},
                    "compilerOptions": {"executorRequest": False}}}
                req = urllib.request.Request(
                    f"{API}/compiler/{cid}/compile", json.dumps(body).encode(),
                    {"Content-Type": "application/json", "Accept": "application/json"})
                res = json.load(urllib.request.urlopen(req, timeout=120))
                if res.get("code") != 0:
                    print("ERROR", alg, isa, src_suffix, cc, res.get("stderr"))
                    continue
                asm = "\n".join(l["text"] for l in res["asm"]) + "\n"
                fn = os.path.join(OUT, folder,
                                  f"x86_{isa}_{alg}_{out_tag}_output-O2_{cc}.txt")
                open(fn, "w").write(asm)
                print("OK", fn)

