"""Calidad del Matrix Profile muestreado frente a la referencia (100% de diagonales)."""
import os
from common import *

EXT = ["audio-MPIII-SVD", "human_activity-MPIII-SVC", "penguin_sample_TutorialMPweb"]


def read_mp(path):
    a = np.loadtxt(path, delimiter=",", skiprows=5)
    return a[:, 2], a[:, 3].astype(np.int64)          # profile (distancia), index


def topk_overlap(p, q, k, excl):
    """Solapamiento de las k posiciones mas bajas (motifs) con tolerancia +-excl."""
    def pick(x, largest):
        v = np.where(np.isfinite(x), x, -np.inf if largest else np.inf)
        o = np.argsort(-v if largest else v)
        sel = []
        for i in o:
            if all(abs(i - s) > excl for s in sel):
                sel.append(i)
                if len(sel) == k: break
        return sel
    return pick


def metrics(P, I, Pr, Ir, w):
    fin = np.isfinite(P)
    ex = w // 4
    out = dict(coverage=fin.mean())
    m = fin & np.isfinite(Pr)
    e = np.abs(P[m] - Pr[m]); rel = e / np.maximum(Pr[m], 1e-12)
    out.update(mae=e.mean(), rmse=float(np.sqrt((e ** 2).mean())), maxabs=e.max(),
               mre=rel.mean(), mre_med=np.median(rel), mre_p95=np.quantile(rel, .95),
               pearson=float(np.corrcoef(P[m], Pr[m])[0, 1]), spearman=spearman(P[m], Pr[m]),
               frac_exact=float((e <= 1e-9).mean()),                      # posicion con distancia exacta
               frac_within1=float((rel <= 0.01).mean()),                  # error relativo <=1%
               nn_same=float((I[m] == Ir[m]).mean()))
    # motif / discord
    Pm = np.where(fin, P, np.inf); Prm = np.where(np.isfinite(Pr), Pr, np.inf)
    mo, mr = int(np.argmin(Pm)), int(np.argmin(Prm))
    out["motif_dist_err"] = abs(Pm[mo] - Prm[mr]) / Prm[mr]
    out["motif_same"] = int(abs(mo - mr) <= ex or abs(mo - Ir[mr]) <= ex)
    Pd = np.where(fin, P, -np.inf); Prd = np.where(np.isfinite(Pr), Pr, -np.inf)
    do, dr = int(np.argmax(Pd)), int(np.argmax(Prd))
    out["discord_same"] = int(abs(do - dr) <= ex)
    out["discord_dist_err"] = abs(Pd[do] - Prd[dr]) / Prd[dr]
    # top-10 discords y motifs (con zona de exclusion), recall frente a referencia
    for name, largest in (("motif", False), ("discord", True)):
        def pick(x):
            v = np.where(np.isfinite(x), x, -np.inf if largest else np.inf)
            o = np.argsort(-v if largest else v); sel = []
            for i in o:
                if all(abs(i - s) > w for s in sel):
                    sel.append(int(i))
                    if len(sel) == 10: break
            return sel
        a, b = pick(P), pick(Pr)
        out[f"top10_{name}_recall"] = float(np.mean([any(abs(x - y) <= ex for x in a) for y in b]))
    return out


def run(variants=("shuf", "noshuf"), vlens=(256, 2048, 16384)):
    d = pd.read_csv(f"{OUT}/runs.csv")
    d["vlen"] = d.vlen.astype(int); d["threads"] = d.threads.astype(int)
    d = d[(d.csv.notna()) & d.dataset.isin(EXT) & (d.rep == 1)]
    d = d[d.csv.map(lambda p: isinstance(p, str) and os.path.exists(p))]
    rows = []
    for ds in EXT:
        w = DS_INFO[ds][1]
        for alg in ("scamp", "scrimp"):
            # referencia: 100% barajado, escalar 1 hilo si existe; si no vect riscv vlen 256, 1 hilo
          for isa in ("riscv", "arm"):
            c = d[(d.dataset == ds) & (d.alg == alg) & (d.variant == "shuf") & (d.fraction == 100) & (d.threads == 1) & (d.isa == isa)]
            ref = c[c["mode"] == "sec"]
            if len(ref) == 0: ref = c[(c["mode"] == "vect") & (c.vlen == 256)]
            if len(ref) == 0: continue
            refp = ref.iloc[0].csv; Pr, Ir = read_mp(refp); refmode = ref.iloc[0]["mode"]
            sel = d[(d.dataset == ds) & (d.alg == alg) & (d.threads == 1) & (d.variant.isin(variants)) & (d.isa == isa) &
                    ((d["mode"] == "sec") | d.vlen.isin(vlens if isa == "riscv" else (2048,)))]
            for r in sel.itertuples():
                P, I = read_mp(r.csv)
                m = metrics(P, I, Pr, Ir, w)
                m.update(ds=DS_SHORT[ds], alg=alg, variant=r.variant, fraction=r.fraction, isa=r.isa, mode=r.mode,
                         vlen=r.vlen, ref=refmode)
                m["bundles"] = sampling_row(ds, (r.vlen // 64) if r.mode == "vect" else 1, r.fraction, r.variant == "shuf", 1)["cov"]
                rows.append(m)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = run()
    df.to_csv(f"{OUT}/mp_quality.csv", index=False)
    print(df.shape)
