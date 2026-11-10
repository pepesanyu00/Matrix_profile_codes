"""Utilidades comunes: carga de runs.csv, replica del muestreo y metricas."""
import os, subprocess, heapq, functools
import numpy as np
import pandas as pd

BASE = "/home/jsanchez/Escritorio/revision_vectorizacion/analisis_revision"
HELPER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_sel")
if not os.path.exists(HELPER):
    HELPER = os.path.join(BASE, "sample_sel")
OUT = os.path.join(BASE, "out")

DS_INFO = {  # profile length P, window w
    "audio-MPIII-SVD": (20035, 200),
    "human_activity-MPIII-SVC": (7878, 120),
    "penguin_sample_TutorialMPweb": (109043, 800),
}
DS_SHORT = {"audio-MPIII-SVD": "audio", "human_activity-MPIII-SVC": "human_activity",
            "penguin_sample_TutorialMPweb": "penguin"}
FOCUS = list(DS_INFO)

# series del paper: (etiqueta, isa, vlen)
SERIES = [("RVV_2k", "riscv", 2048), ("RVV_16k", "riscv", 16384), ("SVE_2k", "arm", 2048)]


@functools.lru_cache(maxsize=None)
def selection(ds, step, pct, shuffle):
    P, w = DS_INFO[ds]
    out = subprocess.run([HELPER, str(P), str(w), str(step), str(pct), str(int(shuffle))],
                         capture_output=True, text=True, check=True).stdout.splitlines()
    total = int(out[0].split()[1])
    cells = np.array([int(l.split()[1]) for l in out[1:]], dtype=np.float64)
    return total, cells


@functools.lru_cache(maxsize=None)
def total_cells(ds):
    _, c = selection(ds, 1, 100, False)
    return c.sum()


def dyn_makespan(costs, threads):
    """Reparto 'schedule(dynamic)' en orden: el siguiente bundle va al primer hilo libre."""
    if threads <= 1:
        return costs.sum()
    h = [0.0] * min(threads, max(len(costs), 1))
    heapq.heapify(h)
    for c in costs:
        heapq.heappush(h, heapq.heappop(h) + c)
    return max(h)


def sampling_row(ds, step, pct, shuffle, threads):
    total, cells = selection(ds, step, pct, shuffle)
    cov = cells.sum() / total_cells(ds)
    ms = dyn_makespan(cells, threads)
    return dict(bundles_total=total, bundles_sel=len(cells), cov=cov,
                max_par=cells.sum() / ms if ms else np.nan,       # speedup paralelo maximo por balanceo
                bal_eff=cells.sum() / (threads * ms) if ms else np.nan)


def load_runs():
    d = pd.read_csv(os.path.join(OUT, "runs.csv"))
    d = d[d.sim_seconds > 0].copy()
    d["vlen"] = d["vlen"].astype(int)
    d["threads"] = d["threads"].astype(int)
    d["ds"] = d.dataset.map(DS_SHORT).fillna(d.dataset)
    return d


def gmean(x):
    x = np.asarray(x, float)
    return float(np.exp(np.mean(np.log(x))))


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3: return np.nan
    ra, rb = pd.Series(a).rank().values, pd.Series(b).rank().values
    if ra.std() == 0 or rb.std() == 0: return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def pearson_log(a, b):
    a, b = np.log(np.asarray(a, float)), np.log(np.asarray(b, float))
    if len(a) < 3 or a.std() == 0 or b.std() == 0: return np.nan
    return float(np.corrcoef(a, b)[0, 1])
