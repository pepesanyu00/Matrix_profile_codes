"""Construye la tabla de speedups (Metrica A del paper) por fraccion, con y sin correccion de cobertura."""
import sys
from common import *


def build(d, variant="shuf", rep=1, focus=("audio", "human_activity")):
    x = d[(d.variant == variant) & (d.rep == rep) & d.ds.isin(focus)]
    sec = x[x["mode"] == "sec"]
    vec = x[x["mode"] == "vect"]
    base = sec[sec.threads == 1].set_index(["fraction", "isa", "alg", "ds"]).sim_seconds
    rows = []
    for r in vec.itertuples():
        key = (r.fraction, r.isa, r.alg, r.ds)
        if key not in base.index: continue
        ds = r.dataset
        shuf = variant == "shuf"
        sv = sampling_row(ds, r.vlen // 64, r.fraction, shuf, r.threads)
        ss = sampling_row(ds, 1, r.fraction, shuf, 1)
        S = base[key] / r.sim_seconds
        rows.append(dict(fraction=r.fraction, isa=r.isa, alg=r.alg, ds=r.ds, vlen=r.vlen, threads=r.threads,
                         t_vect=r.sim_seconds, t_sec1=base[key], S=S,
                         S_corr=S * sv["cov"] / ss["cov"],
                         cov_vect=sv["cov"], cov_sec=ss["cov"], bundles=sv["bundles_sel"],
                         units_per_thread=sv["bundles_sel"] / r.threads, bal_eff=sv["bal_eff"], max_par=sv["max_par"]))
    return pd.DataFrame(rows)


def label(df):
    def f(r):
        if r.isa == "riscv" and r.vlen == 2048: return "RVV_2k"
        if r.isa == "riscv" and r.vlen == 16384: return "RVV_16k"
        if r.isa == "arm" and r.vlen == 2048: return "SVE_2k"
        return None
    df["series"] = df.apply(f, axis=1)
    return df


if __name__ == "__main__":
    d = load_runs()
    t = build(d)
    t.to_csv(f"{OUT}/speedups_shuf.csv", index=False)
    print(t.shape)
