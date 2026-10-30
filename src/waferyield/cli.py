"""Command-line entry point.

    waferyield show scratch --seed 3
    waferyield fit-yield --d0 0.3 --alpha 2 --wafers 500
    waferyield commonality wafers.csv --yield-col yield --steps litho etch cmp
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from .geometry import WaferGeometry
from .render import ascii_map
from .synth import PATTERNS, RandomDefectModel, make_lot, make_wafer
from .yieldmodels import compare_models, fit_negbin_window, wafer_yield


def _geom(args) -> WaferGeometry:
    return WaferGeometry(args.diameter, args.die, args.die, args.edge)


def cmd_show(args) -> int:
    g = _geom(args)
    m = make_wafer(args.pattern, g, np.random.default_rng(args.seed))
    print(ascii_map(m))
    print(f"\npattern={args.pattern} gross_die={g.gross_die} yield={wafer_yield(m):.3f}")
    return 0


def cmd_fit_yield(args) -> int:
    g = _geom(args)
    maps = make_lot(args.wafers, g, RandomDefectModel(args.d0, args.alpha), seed=args.seed)
    fit = fit_negbin_window(maps, g)
    print(f"true    D0={args.d0:.4f}/cm2 alpha={args.alpha:.3f}")
    print(f"fitted  D0={fit.d0_per_cm2:.4f}/cm2 alpha={fit.alpha:.3f} (window-yield rmse {fit.rmse:.4f})")
    print("model rmse vs window yields:")
    for name, err in sorted(compare_models(maps, g).items(), key=lambda p: p[1]):
        print(f"  {name:8s} {err:.4f}")
    return 0


def cmd_commonality(args) -> int:
    from .commonality import analyze

    df = pd.read_csv(args.csv)
    res = analyze(df, args.yield_col, args.steps)
    with pd.option_context("display.width", 140, "display.max_columns", 20):
        print(res.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="waferyield")
    p.add_argument("--diameter", type=float, default=300.0, help="wafer diameter, mm")
    p.add_argument("--die", type=float, default=10.0, help="square die edge, mm")
    p.add_argument("--edge", type=float, default=3.0, help="edge exclusion, mm")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("show", help="render one synthetic wafer map")
    s.add_argument("pattern", choices=PATTERNS)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_show)

    s = sub.add_parser("fit-yield", help="simulate a lot and fit yield models")
    s.add_argument("--d0", type=float, default=0.3, help="defects per cm2")
    s.add_argument("--alpha", type=float, default=2.0, help="cluster parameter")
    s.add_argument("--wafers", type=int, default=500)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_fit_yield)

    s = sub.add_parser("commonality", help="rank process steps by tool-yield association")
    s.add_argument("csv")
    s.add_argument("--yield-col", default="yield")
    s.add_argument("--steps", nargs="+", required=True)
    s.set_defaults(func=cmd_commonality)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
