#!/usr/bin/env python3
"""
perf_report.py  --  summarise 0:\\perf.csv written by main.cpp RUN_PERF_LOG.

Latency (mode 1 or 2): per-stage mean / p50 / p95 in ms, plus FPS of the
hold loop. Only boots matching --boot (default: the last one) are used.

Scores (mode 2): tell it which seg is which person / condition. A seg whose
person is enrolled is a genuine set; a seg tagged "-" is an impostor
(stranger) set. Acceptance mirrors Recognizer.cpp: person score (mean of
top-3 cosine) > thr AND it leads the runner-up person by >= margin.
FRR = genuine frames not accepted as that person; FAR = impostor frames
accepted as anybody.

    python perf_report.py /Volumes/SD/perf.csv
    python perf_report.py perf.csv --seg 0=dylan 1=dylan 2=- 3=-
    python perf_report.py perf.csv --seg 0=dylan 1=- --sweep
"""
import argparse
import csv
import statistics as st
import sys

STAGES = ["cap", "gest", "det", "det_npu", "rec", "emb", "emb_npu", "match",
          "blit", "loop"]


def load(path: str, boot: int) -> list[dict]:
    boots: list[list[dict]] = []
    header = None
    with open(path, newline="") as f:
        for line in f:
            if line.startswith("# boot"):
                boots.append([])
                continue
            if line.startswith("ms,"):
                header = line.strip().split(",")
                continue
            if header is None or not boots:
                continue
            row = next(csv.reader([line]))
            if len(row) == len(header):
                boots[-1].append(dict(zip(header, row)))
    if not boots:
        sys.exit("no '# boot' section found")
    print(f"{len(boots)} boot(s) in file, using #{boot % len(boots)} "
          f"({len(boots[boot])} rows)")
    return boots[boot]


def pct(v: list[float], p: float) -> float:
    v = sorted(v)
    return v[min(len(v) - 1, int(round(p / 100 * (len(v) - 1))))]


def latency(rows: list[dict]) -> None:
    print("\nstage        n     mean    p50    p95   (ms)")
    for s in STAGES:
        if s in ("det", "det_npu"):
            sel = [r for r in rows if int(r["nface"]) >= 0]
        elif s in ("rec", "emb", "emb_npu", "match"):
            sel = [r for r in rows if float(r["score"]) > -1]
        else:
            sel = rows
        v = [int(r[s]) / 1000 for r in sel]
        if v:
            print(f"{s:<10}{len(v):>5}{st.mean(v):>9.2f}{pct(v, 50):>7.2f}"
                  f"{pct(v, 95):>7.2f}")
    full = [int(r["loop"]) / 1000 for r in rows if float(r["score"]) > -1]
    allv = [int(r["loop"]) / 1000 for r in rows]
    if full:
        print(f"\nfull pipeline (face found + recognised): "
              f"{st.mean(full):.1f} ms/frame = {1000 / st.mean(full):.1f} fps")
    print(f"all hold-loop frames: {st.mean(allv):.1f} ms/frame = "
          f"{1000 / st.mean(allv):.1f} fps")


def accepted(r: dict, thr: float, margin: float) -> bool:
    s, s2 = float(r["score"]), float(r["second"])
    return s > thr and (s2 < 0 or s - s2 >= margin)


def scores(rows: list[dict], segmap: dict[int, str], thr: float,
           margin: float, sweep: bool) -> None:
    gen, imp, wrong = [], [], 0
    print("\nseg  who      frames  face  mean   min    max    hit%")
    for seg in sorted({int(r["seg"]) for r in rows}):
        sr = [r for r in rows if int(r["seg"]) == seg]
        rec = [r for r in sr if float(r["score"]) > -1]
        who = segmap.get(seg, "?")
        if rec:
            v = [float(r["score"]) for r in rec]
            hit = sum(int(r["hit"]) for r in rec) / len(rec) * 100
            print(f"{seg:<5}{who:<9}{len(sr):>6}{len(rec):>6}{st.mean(v):>7.3f}"
                  f"{min(v):>7.3f}{max(v):>7.3f}{hit:>7.1f}")
        else:
            print(f"{seg:<5}{who:<9}{len(sr):>6}{0:>6}   (no face)")
        if who == "-":
            imp += rec
        elif who != "?":
            gen += rec
            wrong += sum(r["best"] != who for r in rec)

    def frr(t: float) -> float:
        ok = [r for r in gen if r["best"] == segmap[int(r["seg"])]
              and accepted(r, t, margin)]
        return (1 - len(ok) / len(gen)) * 100

    def far(t: float) -> float:
        return sum(accepted(r, t, margin) for r in imp) / len(imp) * 100

    print(f"\nthr {thr:.2f}  margin {margin:.2f}")
    if gen:
        print(f"genuine frames {len(gen)}: FRR = {frr(thr):.1f}% "
              f"(best person was somebody else: {wrong})")
    if imp:
        print(f"impostor frames {len(imp)}: FAR = {far(thr):.1f}%  "
              f"(max impostor score {max(float(r['score']) for r in imp):.3f})")
    if sweep and gen and imp:
        print("\nthr    FAR%   FRR%")
        for t in [x / 100 for x in range(30, 81, 5)]:
            print(f"{t:.2f}{far(t):>7.1f}{frr(t):>7.1f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--boot", type=int, default=-1, help="boot index, -1 = last")
    ap.add_argument("--seg", nargs="*", default=[],
                    help="seg=label (enrolled person) or seg=- (stranger)")
    ap.add_argument("--thr", type=float, default=0.50)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args()

    rows = load(a.csv, a.boot)
    latency(rows)
    segmap = {int(k): v for k, v in (s.split("=", 1) for s in a.seg)}
    scores(rows, segmap, a.thr, a.margin, a.sweep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
