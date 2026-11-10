import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_ph_tables as G
mid=lambda a:(a[0]+a[1])/2
for comp in ("gcc","clang"):
  for alg in ("scamp","scrimp"):
    def it(stem, ph="PH3"): r=G.get(comp,stem,ph); return mid(r["iter"])
    def dy(stem, ph): r=G.get(comp,stem,ph); return mid((r["dyn_min"],r["dyn_max"]))
    def st(stem, ph): return G.get(comp,stem,ph)["static"]
    sc={"x86":f"x86_baseline_{alg}_scalar","arm":f"arm_baseline_{alg}_scalar","riscv":f"riscv_baseline_{alg}_scalar"}
    cfg=[("AVX2 vec","x86",f"x86_avx2_{alg}_vectorized_loop",f"x86_avx2_{alg}_vectorized_reduction"),
         ("AVX512 vec","x86",f"x86_avx512_{alg}_vectorized_loop",f"x86_avx512_{alg}_vectorized_reduction"),
         ("SVE","arm",f"arm_sve_{alg}",f"arm_sve_{alg}"),("RVV","riscv",f"riscv_rvv_{alg}",f"riscv_rvv_{alg}")]
    print(comp,alg)
    for n,k,loop,step in cfg:
        s_it=it(sc[k]); v_it=it(loop)
        s2=dy(sc[k],"PH2"); v2=dy(step,"PH2")
        print(f"  {n:11} PH3-iter: scalar {s_it:.1f} -> {v_it:.1f} (tau {100*(v_it/s_it-1):+.1f}%) | PH2-dyn: {s2:.1f} -> {v2:.1f} (tau {100*(v2/s2-1):+.1f}%)")
