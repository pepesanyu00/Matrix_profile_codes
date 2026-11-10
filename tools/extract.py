#!/usr/bin/env python3
"""Extrae metricas del bloque ROI (primer dump) de cada stats.txt de gem5 a un CSV unico.

Una fila por (variant, rep, fraction, isa, mode, alg, dataset, vlen, threads).
"""
import os, re, sys, csv
from multiprocessing import Pool

ROOT = "/home/jsanchez/Escritorio/revision_vectorizacion/stats_finales"
OUT = "/home/jsanchez/Escritorio/revision_vectorizacion/analisis_revision/out/runs.csv"

# (variant, rep, fraction) -> directory
SOURCES = {
    ("shuf", 1, 10): "arena/stats1",
    ("shuf", 1, 25): "stats1_25",
    ("shuf", 1, 100): "stats1_100",
    ("shuf", 2, 10): "arena/stats2",
    ("shuf", 2, 25): "stats2_25",
    ("shuf", 2, 100): "stats2_100",
    ("noshuf", 1, 10): "stats_no_shuffle_10",
    ("noshuf", 1, 25): "stats_no_shuffle_25",
    ("noshuf", 1, 100): "stats_no_shuffle_100",
}

RE_CORE = re.compile(r"^(?:board\.processor\.cores\d*\.core|system\.cpu\d*)\.(\S+)\s+(\S+)")
RE_L1 = re.compile(r"^(?:board\.cache_hierarchy\.l1d-cache(?:-\d+)?|system\.cpu\d*\.dcache)\.overall(Misses|Accesses)::total\s+(\S+)")
RE_L2 = re.compile(r"^(?:board\.cache_hierarchy\.l2-cache(?:-\d+)?|system\.cpu\d*\.l2cache)\.overall(Misses|Accesses)::total\s+(\S+)")
RE_L3 = re.compile(r"^(?:board\.cache_hierarchy\.l3-cache|system\.l3cache)\.overall(Misses|Accesses)::total\s+(\S+)")
CORE_KEYS = {
    "iew.iqFullEvents": "iq_full", "iew.lsqFullEvents": "lsq_full",
    "commit.branchMispredicts": "br_misp", "commitStats0.numVecInsts": "vec_insts",
    "rename.blockCycles": "rename_block",
    "rename.IQFullEvents": "rename_iq_full", "rename.LQFullEvents": "rename_lq_full",
    "rename.SQFullEvents": "rename_sq_full", "iew.blockCycles": "iew_block",
}
SCALARS = {
    "simSeconds": "sim_seconds", "simInsts": "sim_insts", "simOps": "sim_ops",
    "board.memory.mem_ctrl.bytesReadSys": "dram_rd", "board.memory.mem_ctrl.bytesWrittenSys": "dram_wr",
    "system.mem_ctrl.bytesReadSys": "dram_rd", "system.mem_ctrl.bytesWrittenSys": "dram_wr",
}
FIELDS = ["variant", "rep", "fraction", "isa", "mode", "alg", "dataset", "vlen", "threads",
          "sim_seconds", "sim_insts", "sim_ops", "dram_rd", "dram_wr", "iq_full", "lsq_full", "br_misp",
          "vec_insts", "rename_block", "rename_iq_full", "rename_lq_full", "rename_sq_full", "iew_block",
          "fu_busy", "fu_busy_simd", "l1_miss", "l1_acc", "l2_miss", "l2_acc", "l3_miss", "l3_acc", "csv"]


def parse_stats(path):
    d = {k: 0.0 for k in FIELDS[9:-1]}
    with open(path) as f:
        for line in f:
            if line.startswith("---------- End Simulation"):
                break
            if line.startswith("simSeconds") or line.startswith("simInsts") or line.startswith("simOps"):
                p = line.split()
                d[SCALARS[p[0]]] = float(p[1]); continue
            if (line.startswith("board.memory.mem_ctrl.bytes") or line.startswith("system.mem_ctrl.bytes")) and "Sys" in line:
                p = line.split()
                if p[0] in SCALARS: d[SCALARS[p[0]]] = float(p[1])
                continue
            m = RE_CORE.match(line)
            if m:
                key, val = m.group(1), m.group(2)
                if key in CORE_KEYS:
                    d[CORE_KEYS[key]] += float(val); continue
                if key.startswith("statFuBusy::"):
                    v = float(val); d["fu_busy"] += v
                    if "Simd" in key or "Float" in key: d["fu_busy_simd"] += v
                    continue
            for rx, pre in ((RE_L1, "l1"), (RE_L2, "l2"), (RE_L3, "l3")):
                m = rx.match(line)
                if m:
                    d[f"{pre}_{'miss' if m.group(1) == 'Misses' else 'acc'}"] += float(m.group(2)); break
    return d


def jobs():
    for (variant, rep, frac), sub in SOURCES.items():
        base = os.path.join(ROOT, sub)
        if not os.path.isdir(base): continue
        for am in sorted(os.listdir(base)):
            if am.count("-") != 1: continue            # excluye *-no_hor
            isa, mode = am.split("-")
            for alg in ("scamp", "scrimp"):
                ad = os.path.join(base, am, alg)
                if not os.path.isdir(ad): continue
                for ds in sorted(os.listdir(ad)):
                    dd = os.path.join(ad, ds)
                    if mode == "sec":
                        for th in os.listdir(dd):
                            yield (variant, rep, frac, isa, mode, alg, ds, 0, th.split("_")[0], os.path.join(dd, th))
                    else:
                        for vl in os.listdir(dd):
                            if not vl.startswith("vlen-"): continue
                            for th in os.listdir(os.path.join(dd, vl)):
                                yield (variant, rep, frac, isa, mode, alg, ds, vl.split("-")[1], th.split("_")[0], os.path.join(dd, vl, th))


def work(j):
    *key, rdir = j
    sp = os.path.join(rdir, "stats.txt")
    if not os.path.exists(sp): return None
    try:
        d = parse_stats(sp)
    except Exception as e:
        print("ERR", sp, e, file=sys.stderr); return None
    csvs = [x for x in os.listdir(rdir) if x.endswith(".csv")]
    row = dict(zip(FIELDS[:9], key)); row.update(d)
    row["csv"] = os.path.join(rdir, csvs[0]) if csvs else ""
    return row


if __name__ == "__main__":
    js = list(jobs()); print("jobs", len(js), file=sys.stderr)
    with Pool(8) as p, open(OUT, "w", newline="") as fo:
        w = csv.DictWriter(fo, FIELDS); w.writeheader(); n = 0
        for r in p.imap_unordered(work, js, chunksize=4):
            if r: w.writerow(r); n += 1
    print("rows", n, file=sys.stderr)
