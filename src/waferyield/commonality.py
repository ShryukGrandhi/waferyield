"""Equipment commonality analysis.

Given one row per wafer with its yield and the tool/chamber it ran on at each
process step, find which step's tool assignment best explains the yield
variation. For every step a Kruskal-Wallis test compares the yield
distributions of the tools used at that step; the steps are ranked by p-value,
and the worst tool at each step is reported with its median yield gap.

This is the standard first pass a yield engineer runs when a lot excursion
appears: "what do the bad wafers have in common?"
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def analyze(df: pd.DataFrame, yield_col: str, step_cols: list[str]) -> pd.DataFrame:
    """Return one row per step, most suspicious first.

    Columns: step, n_tools, H, p_value, eta2, worst_tool, worst_median,
    others_median, median_gap.
    """
    if yield_col not in df:
        raise KeyError(yield_col)
    rows = []
    n = len(df)
    for step in step_cols:
        groups = {tool: g[yield_col].to_numpy() for tool, g in df.groupby(step, observed=True)}
        groups = {t: v for t, v in groups.items() if len(v) > 0}
        k = len(groups)
        if k < 2:
            rows.append(dict(step=step, n_tools=k, H=0.0, p_value=1.0, eta2=0.0,
                             worst_tool=None, worst_median=np.nan,
                             others_median=np.nan, median_gap=0.0))
            continue
        vals = list(groups.values())
        if np.ptp(np.concatenate(vals)) == 0:
            H, p = 0.0, 1.0
        else:
            H, p = stats.kruskal(*vals)
        eta2 = max(0.0, (H - k + 1) / (n - k)) if n > k else 0.0
        medians = {t: float(np.median(v)) for t, v in groups.items()}
        worst = min(medians, key=medians.get)
        others = df.loc[df[step] != worst, yield_col]
        others_med = float(np.median(others))
        rows.append(dict(step=step, n_tools=k, H=float(H), p_value=float(p), eta2=float(eta2),
                         worst_tool=worst, worst_median=medians[worst],
                         others_median=others_med, median_gap=others_med - medians[worst]))
    out = pd.DataFrame(rows).sort_values(["p_value", "H"], ascending=[True, False])
    out["p_bonferroni"] = np.minimum(1.0, out["p_value"] * len(step_cols))
    return out.reset_index(drop=True)


def simulate_route(
    n_wafers: int,
    steps: dict[str, int],
    rng: np.random.Generator,
    base_yield: float = 0.9,
    noise: float = 0.03,
    bad_step: str | None = None,
    bad_tool: int = 0,
    yield_hit: float = 0.03,
) -> pd.DataFrame:
    """Random tool assignment per step, with an optional degraded tool.

    `steps` maps step name -> number of parallel tools/chambers.
    """
    data = {"wafer": np.arange(n_wafers)}
    for step, n_tools in steps.items():
        data[step] = np.array([f"{step}_T{t}" for t in rng.integers(0, n_tools, n_wafers)])
    y = base_yield + rng.normal(0, noise, n_wafers)
    if bad_step is not None:
        y = y - yield_hit * (data[bad_step] == f"{bad_step}_T{bad_tool}")
    data["yield"] = np.clip(y, 0, 1)
    return pd.DataFrame(data)
