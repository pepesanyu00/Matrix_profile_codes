import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_ph_tables as G
def d(stem, alg_ph="PH2", comp="gcc"):
    r = G.get(comp, stem, alg_ph); return (r["dyn_min"], r["dyn_max"])
def rng(a, L, f="%.2f"): return (f % (a[0]/L)) + " – " + (f % (a[1]/L))
for alg in ("scamp","scrimp"):
    print("###", alg)
    sc = {k: d(f"{k}_baseline_{alg}_scalar") for k in ("x86","arm","riscv")}
    v0_2 = d(f"x86_avx2_{alg}"); vr_2 = d(f"x86_avx2_{alg}_vectorized_reduction")
    v0_5 = d(f"x86_avx512_{alg}"); vr_5 = d(f"x86_avx512_{alg}_vectorized_reduction")
    sve = d(f"arm_sve_{alg}"); rvv = d(f"riscv_rvv_{alg}")
    for k,v in sc.items(): print(f"| scalar {k} | 64 | 1 | {v[0]} – {v[1]} | {rng(v,1)} |")
    for name,v,L in [("AVX2 V0",v0_2,4),("AVX2 vec",vr_2,4),("SVE 256",sve,4),("RVV 256",rvv,4),
                     ("AVX512 V0",v0_5,8),("AVX512 vec",vr_5,8),("SVE 512",sve,8),("RVV 512",rvv,8),
                     ("SVE 1024",sve,16),("RVV 1024",rvv,16),("SVE 2048",sve,32),("RVV 2048",rvv,32)]:
        print(f"| {name} | {L*64} | {L} | {v[0]} – {v[1]} | {rng(v,L)} |")
    # factores 4.3: escalar propio / (din/lanes), pareando min-min / max-max
    def fac(s, v, L):
        a = s[0]/(v[0]/L); b = s[1]/(v[1]/L); lo, hi = sorted((a,b)); return f"{lo:.2f}× – {hi:.2f}×"
    print("4.3:")
    for name,s,v,L in [("AVX2 V0",sc['x86'],v0_2,4),("AVX2 vec",sc['x86'],vr_2,4),("AVX512 V0",sc['x86'],v0_5,8),("AVX512 vec",sc['x86'],vr_5,8),
        ("SVE256",sc['arm'],sve,4),("SVE512",sc['arm'],sve,8),("SVE2048",sc['arm'],sve,32),("RVV256",sc['riscv'],rvv,4),("RVV512",sc['riscv'],rvv,8),("RVV2048",sc['riscv'],rvv,32)]:
        print(name, fac(s,v,L))
    mid=lambda r:(r[0]+r[1])/2
    print("RVV512 vs AVX512 V0 red: %.1f%%"%(100*(1-mid(rvv)/mid(v0_5))), " SVE: %.1f%%"%(100*(1-mid(sve)/mid(v0_5))))
    print("RVV512 vs AVX512 vec: %+.1f%%"%(100*(mid(rvv)/mid(vr_5)-1)), " SVE: %+.1f%%"%(100*(mid(sve)/mid(vr_5)-1)))
