"""Genera figuras y tablas finales de la validacion del muestreo de diagonales."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *
from speedups import label

FIG = f"{OUT}/figs"; os.makedirs(FIG, exist_ok=True)
COL = {10: "#d95f02", 25: "#1b9e77", 100: "#222222"}
LAB = {10: "10 %", 25: "25 %", 100: "100 %"}
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .3})


def save(fig, name):
    fig.tight_layout(); fig.savefig(f"{FIG}/{name}.png", dpi=200); fig.savefig(f"{FIG}/{name}.pdf"); plt.close(fig)


sh = label(pd.read_csv(f"{OUT}/speedups_shuf.csv"))
DSN = {"audio": "audio", "human_activity": "human activity"}

# ---- Fig 1: barrido de VLEN (1 hilo), 10/25/100 -------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(8, 5.5), sharex=True)
for i, ds in enumerate(DSN):
    for j, alg in enumerate(("scamp", "scrimp")):
        a = ax[i, j]
        for f in (10, 25, 100):
            for isa, ls, mk in (("riscv", "-", "o"), ("arm", "--", "s")):
                g = sh[(sh.ds == ds) & (sh.alg == alg) & (sh.threads == 1) & (sh.fraction == f) & (sh.isa == isa)].sort_values("vlen")
                a.plot(g.vlen, g.S, ls, marker=mk, ms=3.5, color=COL[f], label=f"{LAB[f]} {'RVV' if isa == 'riscv' else 'SVE'}")
        a.set_xscale("log", base=2); a.set_yscale("log"); a.set_title(f"{DSN[ds]} - {alg.upper()}")
        a.set_xticks([256, 512, 1024, 2048, 4096, 8192, 16384]); a.set_xticklabels(["256", "512", "1k", "2k", "4k", "8k", "16k"])
        if i == 1: a.set_xlabel("VLEN (bits)")
        if j == 0: a.set_ylabel("Speedup vs. scalar (1 thread)")
ax[0, 0].legend(ncol=2, fontsize=6.5)
save(fig, "fig1_vlen_sweep")

# ---- Fig 2: barrido de hilos --------------------------------------------------------------------
for alg in ("scamp", "scrimp"):
    fig, ax = plt.subplots(2, 3, figsize=(9.5, 5.2), sharex=True)
    for i, ds in enumerate(DSN):
        for j, s in enumerate(("RVV_2k", "SVE_2k", "RVV_16k")):
            a = ax[i, j]
            for f in (10, 25, 100):
                g = sh[(sh.ds == ds) & (sh.alg == alg) & (sh.series == s) & (sh.fraction == f)].sort_values("threads")
                a.plot(g.threads, g.S, "-o", ms=3.5, color=COL[f], label=LAB[f])
            a.set_xscale("log", base=2); a.set_title(f"{DSN[ds]} - {s}")
            a.set_xticks([1, 2, 4, 8, 16, 32, 64]); a.set_xticklabels(["1", "2", "4", "8", "16", "32", "64"])
            if i == 1: a.set_xlabel("Threads")
            if j == 0: a.set_ylabel("Speedup vs. scalar (1 thread)")
    ax[0, 0].legend(title="diagonals", fontsize=7)
    fig.suptitle(alg.upper(), y=1.0)
    save(fig, f"fig2_thread_sweep_{alg}")

# ---- Fig 3: error vs. bundles por hilo -----------------------------------------------------------
err = pd.read_csv(f"{OUT}/err_shuf.csv")
bins = [0, 1, 2, 4, 8, 16, 32, 64, 1e9]
labs = ["<1", "1-2", "2-4", "4-8", "8-16", "16-32", "32-64", ">64"]
err["ubin"] = pd.cut(err.units_per_thread, bins, labels=labs)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for a, col, ttl in ((ax[0], "e", "Measured (as in the paper)"), (ax[1], "e_corr", "Coverage-normalised")):
    for f in (10, 25):
        g = err[err.fraction == f]
        a.scatter(g.units_per_thread, g[col].abs() * 100, s=9, alpha=.35, color=COL[f])
        med = g.groupby("ubin", observed=True).apply(lambda x: pd.Series(dict(u=x.units_per_thread.median(), m=x[col].abs().median() * 100)), include_groups=False)
        a.plot(med.u, med.m, "-D", ms=4, color=COL[f], label=f"{LAB[f]} (median per bin)")
    a.set_xscale("log"); a.set_yscale("log"); a.axhline(10, color="gray", ls=":"); a.set_title(ttl)
    a.set_xlabel("Sampled bundles per thread  (N$_{bundles}$ / threads)")
ax[0].set_ylabel("|speedup error| vs. 100 % run (%)"); ax[0].legend(fontsize=7)
save(fig, "fig3_error_vs_bundles_per_thread")

# ---- Fig 4: calidad del MP ----------------------------------------------------------------------
mp = pd.read_csv(f"{OUT}/mp_quality.csv")
mp = mp[(mp["mode"] == "sec") & (mp.isa == "riscv") & (mp.alg == "scamp")]
fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.3))
for v, ls in (("shuf", "-"), ("noshuf", "--")):
    for ds, c in zip(("audio", "human_activity", "penguin"), ("#1f77b4", "#d62728", "#2ca02c")):
        g = mp[(mp.ds == ds) & (mp.variant == v)].sort_values("fraction")
        g = pd.concat([g, pd.DataFrame(dict(fraction=[100], pearson=[1.0], mre_med=[0.0], mre=[0.0], variant=[v]))])
        ax[0].plot(g.fraction, g.pearson, ls, marker="o", color=c, label=f"{ds} ({'shuffled' if v == 'shuf' else 'natural'})")
        ax[1].plot(g.fraction, g.mre * 100, ls, marker="o", color=c)
ax[0].set_ylabel("Pearson r (MP$_f$ vs MP$_{100}$)"); ax[1].set_ylabel("Mean relative error (%)")
for a in ax: a.set_xlabel("% of diagonals"); a.set_xticks([10, 25, 100])
ax[0].legend(fontsize=6)
save(fig, "fig4_mp_quality")

# ---- Fig 5: efecto del orden a 100% --------------------------------------------------------------
ns = label(pd.read_csv(f"{OUT}/speedups_noshuf.csv"))
key = ["isa", "alg", "ds", "vlen", "threads"]
a = ns[ns.fraction == 100].set_index(key).join(sh[sh.fraction == 100].set_index(key), lsuffix="_ns", rsuffix="_sh", how="inner").reset_index()
a["rt"] = a.t_vect_ns / a.t_vect_sh
fig, ax = plt.subplots(figsize=(4.6, 3.2))
g = a.groupby("threads").rt
ax.fill_between(g.min().index, g.min(), g.max(), color="#9ecae1", alpha=.6, label="min-max (all VLEN, ISA, alg)")
ax.plot(g.mean().index, g.mean(), "-o", color="#08519c", label="mean"); ax.axhline(1, color="k", lw=.7)
ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8, 16, 32, 64]); ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64"])
ax.set_xlabel("Threads"); ax.set_ylabel("T$_{natural}$ / T$_{shuffled}$ (100 % diagonals)"); ax.legend(fontsize=7)
save(fig, "fig5_order_effect_100pct")
print("figs ok")
