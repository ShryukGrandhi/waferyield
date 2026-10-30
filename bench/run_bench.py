"""Benchmark: accuracy, throughput and detection performance of every module.

    python bench/run_bench.py            # full run, writes bench/results.json
    python bench/run_bench.py --quick    # smaller sizes, used in CI

All data is synthetic and generated from fixed seeds, so results are
reproducible on the same library versions.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import sklearn

from waferyield.classify import PatternClassifier
from waferyield.commonality import analyze, simulate_route
from waferyield.geometry import WaferGeometry
from waferyield.spc import Baseline, cusum_low, ewma, run_length, western_electric
from waferyield.synth import RandomDefectModel, make_dataset, make_lot, simulate_wafer_yields
from waferyield.yieldmodels import compare_models, fit_negbin_window

GEOM = WaferGeometry(300.0, 10.0, 10.0, 3.0)


def bench_classification(n_train: int, n_test: int) -> dict:
    Xtr, ytr = make_dataset(n_train, GEOM, seed=100)
    Xte, yte = make_dataset(n_test, GEOM, seed=200)
    clf = PatternClassifier(GEOM, n_estimators=300, seed=0)

    t0 = time.perf_counter()
    Ftr = clf.extractor.transform(Xtr)
    t_feat = time.perf_counter() - t0

    t0 = time.perf_counter()
    clf.model.fit(Ftr, ytr)
    t_fit = time.perf_counter() - t0

    t0 = time.perf_counter()
    rep = clf.evaluate(Xte, yte)
    t_pred = time.perf_counter() - t0

    top = clf.feature_importance()[:5]
    return {
        "train_wafers": int(len(ytr)),
        "test_wafers": int(len(yte)),
        "accuracy": rep.accuracy,
        "macro_f1": rep.macro_f1,
        "recall": rep.recall,
        "confusion_classes": rep.classes,
        "confusion": rep.confusion.tolist(),
        "confusion_text": rep.confusion_text(),
        "feature_wafers_per_s": len(ytr) / t_feat,
        "fit_seconds": t_fit,
        "end_to_end_predict_wafers_per_s": len(yte) / t_pred,
        "top_features": [[n, float(v)] for n, v in top],
    }


def bench_yield(n_wafers: int, seeds: int) -> dict:
    cases = [(0.1, 1.0), (0.3, 0.5), (0.3, 2.0), (0.3, 8.0), (0.6, 2.0)]
    rows = []
    for d0, alpha in cases:
        d0_err, a_err, t_fit = [], [], []
        rmse = {"poisson": [], "murphy": [], "seeds": [], "negbin": []}
        for s in range(seeds):
            maps = make_lot(n_wafers, GEOM, RandomDefectModel(d0, alpha), seed=1000 + s)
            t0 = time.perf_counter()
            fit = fit_negbin_window(maps, GEOM)
            t_fit.append(time.perf_counter() - t0)
            d0_err.append(abs(fit.d0_per_cm2 - d0) / d0)
            a_err.append(abs(fit.alpha - alpha) / alpha)
            for k, v in compare_models(maps, GEOM).items():
                rmse[k].append(v)
        rows.append({
            "d0": d0, "alpha": alpha,
            "d0_rel_err_mean": float(np.mean(d0_err)),
            "alpha_rel_err_mean": float(np.mean(a_err)),
            "fit_ms_mean": 1000 * float(np.mean(t_fit)),
            "rmse": {k: float(np.mean(v)) for k, v in rmse.items()},
        })
    return {"wafers_per_lot": n_wafers, "seeds": seeds, "die_area_cm2": GEOM.die_area_cm2,
            "gross_die": GEOM.gross_die, "cases": rows}


def _arl(chart, runs: int, phase1: int, phase2: int, model0, model1, rng) -> tuple[float, float]:
    """Mean run length (wafers to first signal in phase II) and fraction of runs
    that never signalled within `phase2` wafers (counted as phase2)."""
    rls = []
    misses = 0
    for _ in range(runs):
        base = Baseline.from_data(simulate_wafer_yields(phase1, GEOM, model0, rng))
        x = simulate_wafer_yields(phase2, GEOM, model1, rng)
        rl = run_length(chart(x, base))
        if rl is None:
            misses += 1
            rl = phase2
        rls.append(rl)
    return float(np.mean(rls)), misses / runs


def bench_spc(runs: int, phase2: int) -> dict:
    alpha = 8.0
    d0 = 0.2
    m0 = RandomDefectModel(d0, alpha)
    charts = {
        "WE rule 1 only": lambda x, b: western_electric(x, b, rules=(1,)),
        "WE rules 1-4": lambda x, b: western_electric(x, b),
        "EWMA (0.2, 2.86)": lambda x, b: ewma(x, b),
        "CUSUM low (0.5, 4.77)": lambda x, b: cusum_low(x, b),
    }
    shifts = [0.0, 0.25, 0.5, 1.0]
    out = {"d0": d0, "alpha": alpha, "phase1_wafers": 100, "phase2_cap": phase2, "runs": runs,
           "shifts": shifts, "charts": {}}
    for name, chart in charts.items():
        rng = np.random.default_rng(7)
        res = []
        for sh in shifts:
            m1 = RandomDefectModel(d0 * (1 + sh), alpha)
            arl, miss = _arl(chart, runs, 100, phase2, m0, m1, rng)
            res.append({"shift": sh, "arl": arl, "censored_frac": miss})
        out["charts"][name] = res
    return out


def bench_commonality(trials: int) -> dict:
    steps = {"litho": 4, "etch": 6, "depo": 4, "cmp": 3, "implant": 2, "clean": 3}
    names = list(steps)
    rng = np.random.default_rng(11)
    out = {"wafers": 300, "noise_sd": 0.03, "steps": steps, "trials": trials, "hits": []}
    for hit in (0.0, 0.01, 0.02, 0.03):
        step_ok = tool_ok = flagged = 0
        t0 = time.perf_counter()
        for _ in range(trials):
            bad = names[rng.integers(len(names))]
            tool = int(rng.integers(steps[bad]))
            df = simulate_route(300, steps, rng, noise=0.03,
                                bad_step=bad if hit > 0 else None, bad_tool=tool, yield_hit=hit)
            res = analyze(df, "yield", names)
            top = res.iloc[0]
            sig = top["p_bonferroni"] < 0.05
            flagged += sig
            if hit > 0 and sig and top["step"] == bad:
                step_ok += 1
                tool_ok += top["worst_tool"] == f"{bad}_T{tool}"
        dt = time.perf_counter() - t0
        out["hits"].append({
            "yield_hit": hit,
            "flag_rate": flagged / trials,
            "top1_step_correct": step_ok / trials if hit > 0 else None,
            "top1_step_and_tool_correct": tool_ok / trials if hit > 0 else None,
            "ms_per_analysis": 1000 * dt / trials,
        })
    return out


def _md(r: dict) -> str:
    c = r["classification"]
    lines = ["## Defect-pattern classification",
             f"train {c['train_wafers']} wafers, test {c['test_wafers']} wafers (8 balanced classes)", "",
             f"- accuracy: {c['accuracy']:.4f}",
             f"- macro F1: {c['macro_f1']:.4f}",
             f"- feature extraction: {c['feature_wafers_per_s']:.0f} wafers/s (single thread)",
             f"- end-to-end predict: {c['end_to_end_predict_wafers_per_s']:.0f} wafers/s",
             f"- random forest fit: {c['fit_seconds']:.2f} s", "",
             "| class | recall |", "|---|---|"]
    lines += [f"| {k} | {v:.3f} |" for k, v in c["recall"].items()]
    lines += ["", "```", c["confusion_text"], "```", ""]

    y = r["yield"]
    lines += ["## Yield model fit (window method)",
              f"{y['wafers_per_lot']} wafers per lot, {y['seeds']} lots per case, "
              f"{y['gross_die']} dies/wafer, die area {y['die_area_cm2']} cm2", "",
              "| true D0 | true alpha | D0 rel err | alpha rel err | fit ms | RMSE Poisson | RMSE Murphy | RMSE Seeds | RMSE NB |",
              "|---|---|---|---|---|---|---|---|---|"]
    for row in y["cases"]:
        e = row["rmse"]
        lines.append(f"| {row['d0']} | {row['alpha']} | {row['d0_rel_err_mean']:.3f} | "
                     f"{row['alpha_rel_err_mean']:.3f} | {row['fit_ms_mean']:.1f} | {e['poisson']:.4f} | "
                     f"{e['murphy']:.4f} | {e['seeds']:.4f} | {e['negbin']:.4f} |")
    lines.append("")

    s = r["spc"]
    lines += ["## SPC average run length on wafer yield",
              f"D0={s['d0']}/cm2, alpha={s['alpha']}, Phase I = {s['phase1_wafers']} wafers, "
              f"{s['runs']} runs per cell, capped at {s['phase2_cap']} wafers. "
              "Shift = relative increase in D0.", "",
              "| chart | " + " | ".join(f"shift +{int(100 * x)}%" for x in s["shifts"]) + " |",
              "|---|" + "---|" * len(s["shifts"])]
    for name, res in s["charts"].items():
        cells = []
        for q in res:
            cell = f"{q['arl']:.1f}"
            if q["censored_frac"] > 0:
                cell += f" ({100 * q['censored_frac']:.1f}% hit cap)"
            cells.append(cell)
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")

    m = r["commonality"]
    lines += ["## Equipment commonality",
              f"{m['wafers']} wafers, {len(m['steps'])} steps, yield noise sd {m['noise_sd']}, "
              f"{m['trials']} trials per row; flagged = Bonferroni p < 0.05", "",
              "| injected yield hit | flagged | top-1 step correct | top-1 step+tool correct | ms/analysis |",
              "|---|---|---|---|---|"]
    for h in m["hits"]:
        f = lambda v: "-" if v is None else f"{v:.3f}"
        lines.append(f"| {h['yield_hit']} | {h['flag_rate']:.3f} | {f(h['top1_step_correct'])} | "
                     f"{f(h['top1_step_and_tool_correct'])} | {h['ms_per_analysis']:.2f} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.quick:
        cfg = dict(n_train=60, n_test=30, n_wafers=200, seeds=2, runs=100, phase2=1000, trials=50)
    else:
        cfg = dict(n_train=400, n_test=200, n_wafers=500, seeds=5, runs=1000, phase2=5000, trials=300)

    t0 = time.perf_counter()
    r = {
        "config": cfg,
        "env": {"python": platform.python_version(), "machine": platform.machine(),
                "system": platform.system(), "numpy": np.__version__, "sklearn": sklearn.__version__},
        "classification": bench_classification(cfg["n_train"], cfg["n_test"]),
        "yield": bench_yield(cfg["n_wafers"], cfg["seeds"]),
        "spc": bench_spc(cfg["runs"], cfg["phase2"]),
        "commonality": bench_commonality(cfg["trials"]),
    }
    r["total_seconds"] = time.perf_counter() - t0
    print(_md(r))
    print(f"\ntotal benchmark time: {r['total_seconds']:.1f} s")
    out = args.out or ("bench/results_quick.json" if args.quick else "bench/results.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
