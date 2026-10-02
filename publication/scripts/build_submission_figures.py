"""Build submission follow-up figures from staged, hashed evidence only."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


BLUE = "#23699D"
ORANGE = "#C76B25"
GREY = "#727D89"
DARK = "#243648"
PALE = "#D9E0E6"
METHOD_COLORS = {"P": ORANGE, "B4": BLUE, "B5": GREY, "P-no-rank": "#9AA3AB"}
STAGE_ORDER = [
    "request_preparation", "transfer_grid", "encoder", "branch", "trunk",
    "field_assembly", "scoring", "return", "projection", "verification",
]
STAGE_LABELS = {
    "request_preparation": "Request prep.", "transfer_grid": "Transfer / grid",
    "encoder": "Encoder", "branch": "Branch", "trunk": "Trunk",
    "field_assembly": "Field assembly / other", "scoring": "Scoring",
    "return": "Return", "projection": "Projection", "verification": "Verification",
}
# Fixed categorical map: related muted roots, with no library-default cycle.
STAGE_COLORS = {
    "request_preparation": "#596B7A", "transfer_grid": "#9FC5DF",
    "encoder": BLUE, "branch": ORANGE, "trunk": "#E7B968",
    "field_assembly": "#84956B", "scoring": "#C986A6",
    "return": "#7B8793", "projection": "#D6A276", "verification": "#C8CED3",
}
METRIC_MARKERS = {"E_obs": "o", "E_prop": "s", "E_total": "D"}
METRIC_LABELS = {"E_obs": r"$E_{obs}$", "E_prop": r"$E_{prop}$", "E_total": r"$E_{total}$"}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _style() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.size": 7.5,
        "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.edgecolor": DARK,
        "axes.linewidth": .65, "text.color": DARK,
        "axes.labelcolor": DARK, "xtick.color": DARK,
        "ytick.color": DARK, "svg.fonttype": "none",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def build(O):
    """Create available D02/D06/D07 figures under O and return provenance records.

    `O` is the submission Overleaf root. X1 is optional until its analysis CSV is
    staged; D02 and D07 require their respective staged evidence.
    """
    root = Path(O)
    data = root / "data" / "followup"
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    _style()
    records = []

    def save(fig, name: str, description: str, sources: list[str]) -> None:
        hashes = []
        for source in sources:
            path = root / source
            if not path.is_file():
                raise FileNotFoundError(f"Figure {name} source missing: {path}")
            hashes.append({"path": str(path.relative_to(root)), "sha256": _sha256(path)})
        outputs = {}
        for suffix in ("pdf", "svg", "png"):
            path = figures / f"{name}.{suffix}"
            options = {"dpi": 300} if suffix == "png" else {}
            fig.savefig(path, bbox_inches="tight", pad_inches=.05, **options)
            outputs[suffix] = _sha256(path)
        records.append({
            "id": name, "description": description, "sources": sources,
            "input_sha256": hashes, "output_sha256": outputs,
            "pdf_sha256": outputs["pdf"], "vector": True,
        })
        plt.close(fig)

    x2_path = data / "X2" / "X2a_stage_shares.csv"
    x3_path = data / "X3" / "X3_OPERATOR_SUMMARIES.json"
    if x2_path.is_file():
        stage_rows = _read_csv(x2_path)
        _build_d02(stage_rows, save)
    if x3_path.is_file():
        x3 = json.loads(x3_path.read_text(encoding="utf-8"))
        if not isinstance(x3.get("rows"), list):
            raise ValueError("X3_OPERATOR_SUMMARIES.json must contain a rows list")
        _build_d07(x3["rows"], save)

    x1_path = _find_x1_csv(data)
    if x1_path is not None:
        _build_d06(_read_csv(x1_path), str(x1_path.relative_to(root)).replace("\\", "/"), save)
    return records


def _build_d02(rows: list[dict[str, str]], save) -> None:
    required = {"pde", "method", "K", "cache", "stage", "share_of_summed_instrumented_stages", "requests"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"X2a stage CSV lacks required columns: {sorted(required - set(rows[0] if rows else []))}")
    cases: dict[tuple, dict[str, float]] = defaultdict(dict)
    for row in rows:
        key = (row["pde"], row["method"], int(row["K"]), row["cache"].lower() in {"true", "1"})
        stage = row["stage"]
        if stage not in STAGE_ORDER or stage in cases[key]:
            raise ValueError(f"unexpected or duplicate stage in X2a CSV: {key}/{stage}")
        if int(row["requests"]) != 500:
            raise ValueError(f"X2a profile should contain 500 records per case: {key}")
        share = float(row["share_of_summed_instrumented_stages"])
        if not math.isfinite(share) or share < 0:
            raise ValueError(f"invalid X2a share: {key}/{stage}")
        mean = float(row["mean_ms"])
        if not math.isfinite(mean) or mean < 0 or not np.isclose(mean / float(row["instrumented_stage_sum_ms"]), share, atol=2e-8, rtol=0):
            raise ValueError(f"X2a mean/share inconsistency: {key}/{stage}")
        cases[key][stage] = mean
    expected = {(pde, method, k, cache) for pde in ("burgers", "heat")
                for method in ("P", "B4") for k in (10, 200) for cache in (False, True)}
    if set(cases) != expected or any(set(values) != set(STAGE_ORDER) for values in cases.values()):
        raise ValueError("X2a must contain all 16 PDE/method/K/cache profiles and ten stages each")
    upper = math.ceil(max(sum(values.values()) for values in cases.values()))

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 4.4), sharex=True)
    for ax, pde in zip(axes, ("burgers", "heat")):
        case_keys = [(pde, method, k, cache) for method in ("P", "B4")
                     for k in (10, 200) for cache in (False, True)]
        labels = []
        for key in case_keys:
            _, method, k, cache = key
            labels.append(f"{method}  K={k}  {'on' if cache else 'off'}")
        y = np.arange(len(case_keys))
        left = np.zeros(len(case_keys))
        for stage in STAGE_ORDER:
            widths = np.asarray([cases[key][stage] for key in case_keys])
            ax.barh(y, widths, left=left, height=.67, color=STAGE_COLORS[stage],
                    edgecolor="white", linewidth=.3)
            left += widths
        ax.set_title("Burgers" if pde == "burgers" else "SH heat", loc="left", fontsize=8)
        ax.set_yticks(y, labels)
        for tick, key in zip(ax.get_yticklabels(), case_keys):
            tick.set_color(METHOD_COLORS[key[1]])
            tick.set_fontweight("bold")
        ax.invert_yaxis()
        ax.set_xlim(0, upper)
        ax.grid(axis="x", color="#E5E9ED", lw=.55, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("Summed mean instrumented time (ms)")
    axes[0].set_ylabel("Profile case")
    fig.text(.5, .965, "Stage means from 500 instrumented requests per case  ·  label color: P orange, B4 blue",
             ha="center", va="top", fontsize=7.5)
    fig.legend(handles=[Patch(facecolor=STAGE_COLORS[s], edgecolor="none", label=STAGE_LABELS[s])
                        for s in STAGE_ORDER], loc="lower center", ncol=5,
               frameon=False, bbox_to_anchor=(.5, .035), fontsize=7)
    fig.subplots_adjust(left=.20, right=.99, top=.90, bottom=.25, wspace=.34)
    save(fig, "figD02_stage_means",
         "Stage mean durations for all 16 profiles (two PDEs × P/B4 × K=10/200 × cache off/on); ten separately synchronized stages and 500 records per case. The sum is not a separately recorded total profiled-request duration and gives no end-to-end or p99 deletion bound. Stage shares are reported in the accompanying table.",
         ["data/followup/X2/X2a_stage_shares.csv"])


def _family_label(row: dict) -> str:
    stage, pde, method = row["stage"], row["pde"], row["method"]
    method_label = "P-nr" if method == "P-no-rank" else method
    if stage == "primary":
        pde_label = {"burgers": "Burgers", "heat": "SH heat"}.get(pde, pde.capitalize())
        return f"Primary {pde_label}  ·  {method_label}"
    if stage == "expanded_burgers":
        factor = row.get("target_factor")
        factor_label = "?" if factor in (None, "") else str(int(factor))
        return f"Burgers ×{factor_label}  ·  {method_label}"
    if stage == "nonnegative_heat":
        return f"NH heat  ·  {method_label}"
    raise ValueError(f"unrecognized X3 family/stage: {stage}")


def _build_d07(rows: list[dict], save) -> None:
    chosen = [r for r in rows if r.get("stage") in {"primary", "expanded_burgers", "nonnegative_heat"}]
    if not chosen:
        raise ValueError("X3 operator summary contains no supported family rows")
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in chosen:
        for field in ("E_obs", "E_prop", "E_total"):
            value = float(row[field])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid X3 {field} value")
        groups[_family_label(row)].append(row)
    for label, group in groups.items():
        if len(group) != 3 or len({int(r["seed"]) for r in group}) != 3:
            raise ValueError(f"expected exactly three distinct-seed rows for {label}")
    ordered_groups = sorted(groups.items(), key=lambda item: _x3_sort_key(item[1][0]))
    labels = [label for label, _ in ordered_groups]
    n = len(labels)
    y = np.arange(n, dtype=float)
    metric_offsets = {"E_obs": .19, "E_prop": 0.0, "E_total": -.19}
    fig, ax = plt.subplots(figsize=(7.16, max(4.6, .20 * n + 1.0)))
    for label, group in ordered_groups:
        idx = labels.index(label)
        method = group[0]["method"]
        color = METHOD_COLORS.get(method, GREY)
        for metric, marker in METRIC_MARKERS.items():
            vals = np.asarray([float(row[metric]) for row in group], dtype=float)
            mean, low, high = float(vals.mean()), float(vals.min()), float(vals.max())
            yy = y[idx] + metric_offsets[metric]
            ax.hlines(yy, low, high, color=color, lw=.85, zorder=2)
            ax.vlines([low, high], yy-.045, yy+.045, color=color, lw=.85, zorder=2)
            ax.plot(mean, yy, marker=marker, linestyle="none", color=color,
                    markersize=4.1, markeredgewidth=.55,
                    markerfacecolor=("white" if method in {"P-no-rank", "B5"} else color),
                    markeredgecolor=color, zorder=3)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Normalized RMSE")
    ax.set_ylabel("Family / PDE / method")
    ax.axvline(.05, color=GREY, lw=.8, linestyle=(0, (3, 2)), zorder=1)
    upper = max(float(r[k]) for r in chosen for k in ("E_obs", "E_prop", "E_total"))
    ax.set_xlim(0, math.ceil(upper * 1.08 / .05) * .05)
    ax.grid(axis="x", color="#E5E9ED", lw=.55)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend = [Line2D([], [], marker=METRIC_MARKERS[k], color=DARK, linestyle="none",
                     markersize=4.2, label=METRIC_LABELS[k]) for k in METRIC_MARKERS]
    legend += [Line2D([], [], marker="o", color=METHOD_COLORS["P"], linestyle="none",
                      markersize=4, label="P"),
               Line2D([], [], marker="o", color=METHOD_COLORS["B4"], linestyle="none",
                      markersize=4, label="B4"),
               Line2D([], [], color=GREY, linestyle=(0, (3, 2)), lw=.8,
                      label=r"0.05 gate for $E_{total}$")]
    fig.legend(handles=legend, loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(.72, .99), fontsize=7)
    fig.text(.30, .012, "Points: mean over 3 seeds; whiskers: seed min–max (not confidence intervals). Metrics are separate, not additive.",
             ha="left", va="bottom", fontsize=7)
    fig.subplots_adjust(left=.30, right=.99, top=.91, bottom=.16)
    save(fig, "figD07_error_decomposition",
         "Non-additive nRMSE dot plot by evaluation family, PDE, target factor, and method. Marker shape identifies E_obs, E_prop, or E_total; horizontal whiskers show the minimum and maximum of three model seeds, not a confidence interval. Ratios and errors do not imply additive variance decomposition.",
         ["data/followup/X3/X3_OPERATOR_SUMMARIES.json"])


def _x3_sort_key(row: dict) -> tuple:
    stage_order = {"primary": 0, "expanded_burgers": 1, "nonnegative_heat": 2}
    pde_order = {"burgers": 0, "heat": 1}
    method_order = {"P": 0, "P-no-rank": 1, "B4": 2, "B5": 3}
    factor = row.get("target_factor")
    return (stage_order[row["stage"]], pde_order.get(row["pde"], 9),
            int(factor) if factor not in (None, "") else 0,
            method_order.get(row["method"], 9))


def _find_x1_csv(data: Path) -> Path | None:
    candidates = [data / "X1_ANALYSIS.csv", data / "X1" / "X1_ANALYSIS.csv"]
    return next((path for path in candidates if path.is_file()), None)


def _build_d06(rows: list[dict[str, str]], source: str, save) -> None:
    required = {"pde", "population", "cell", "K", "H", "feedback_rollout", "n",
                "relative_excess_fixed", "fixed_ci_low", "fixed_ci_high", "hold_selection_rate", "applied_no_change_rate"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"X1 analysis CSV lacks required columns: {sorted(required - set(rows[0] if rows else []))}")
    matched = [r for r in rows if int(r["n"]) == 128 and "anchor_full384" not in r["population"]]
    anchors = [r for r in rows if "anchor_full384" in r["population"]]
    if not matched:
        raise ValueError("X1 analysis CSV has no matched n=128 design rows")
    # Do not chart an incomparable full-population anchor as another matched cell.
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.75), sharex="col", layout="constrained")
    colors = {10: BLUE, 50: ORANGE}
    for pi, pde in enumerate(("burgers", "heat")):
        sub = [r for r in matched if r["pde"] == pde]
        for method_k in (10, 50):
            curve = sorted([r for r in sub if int(r["K"]) == method_k and
                            str(r["feedback_rollout"]).lower() not in {"true", "1"}],
                           key=lambda r: int(r["H"]))
            if len(curve) != 4:
                raise ValueError(f"expected four matched H values for {pde}/K={method_k}")
            x = np.asarray([int(r["H"]) for r in curve], dtype=float)
            mean = np.asarray([100*float(r["relative_excess_fixed"]) for r in curve])
            low = np.asarray([100*float(r["fixed_ci_low"]) for r in curve])
            high = np.asarray([100*float(r["fixed_ci_high"]) for r in curve])
            axes[0, pi].errorbar(x, mean, yerr=np.vstack((mean-low, high-mean)),
                                 color=colors[method_k], marker="o", ms=4, lw=.9,
                                 capsize=2, label=f"K={method_k} (n=128 matched)")
            hold = np.asarray([100*float(r["applied_no_change_rate"]) for r in curve])
            axes[1, pi].plot(x, hold, color=colors[method_k], marker="o", ms=4,
                             lw=.9, label=f"K={method_k}")
        fb = [r for r in sub if str(r["feedback_rollout"]).lower() in {"true", "1"}]
        for row in fb:
            x = int(row["H"])
            y_cost = 100*float(row["relative_excess_fixed"])
            y_hold = 100*float(row["applied_no_change_rate"])
            axes[0, pi].errorbar([x], [y_cost], yerr=[[100*(float(row["relative_excess_fixed"])-float(row["fixed_ci_low"]))],
                                                      [100*(float(row["fixed_ci_high"])-float(row["relative_excess_fixed"]))]],
                                 color=DARK, marker="D", ms=4, capsize=2, lw=.8,
                                 linestyle="none", label="Feedback rollout (n=128)")
            axes[1, pi].plot([x], [y_hold], marker="D", color=DARK, ms=4,
                             linestyle="none", label="Feedback rollout")
        # Highlight the matched n=128 K=10/H=8 baseline independently of the
        # full-population Burgers anchor; do not connect the latter into a trend.
        baseline = [r for r in sub if int(r["K"]) == 10 and int(r["H"]) == 8 and
                    str(r["feedback_rollout"]).lower() not in {"true", "1"}]
        for row in baseline:
            axes[0, pi].plot([8], [100*float(row["relative_excess_fixed"])], marker="o",
                             color=DARK, markerfacecolor="white", markeredgewidth=1,
                             markersize=7, linestyle="none", label="Matched baseline (n=128)")
        anchor = [r for r in anchors if r["pde"] == "burgers" and int(r["K"]) == 10 and int(r["H"]) == 8]
        if pde == "burgers" and len(anchor) == 1:
            row = anchor[0]
            axes[0, pi].plot([8], [100*float(row["relative_excess_fixed"])], marker="*",
                             color=DARK, markerfacecolor="white", markeredgewidth=.8,
                             markersize=8, linestyle="none", label="Full Burgers anchor (n=384)")
            axes[1, pi].plot([8], [100*float(row["applied_no_change_rate"])], marker="*",
                             color=DARK, markerfacecolor="white", markeredgewidth=.8,
                             markersize=8, linestyle="none", label="Full anchor (n=384)")
        axes[0, pi].set_title("Burgers" if pde == "burgers" else "NH heat", loc="left", fontsize=8)
        for ax in (axes[0, pi], axes[1, pi]):
            ax.set_xticks([1, 4, 8, 16])
            ax.grid(axis="y", color="#E5E9ED", lw=.55)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        axes[0, pi].axhline(0, color=DARK, lw=.65)
        axes[0, pi].axhline(5, color=GREY, lw=.8, linestyle=(0, (3, 2)))
        axes[0, pi].set_ylabel("Fixed-denominator excess cost (%)")
        axes[1, pi].set_ylabel("Applied action unchanged (%)")
        axes[1, pi].set_xlabel("Candidate horizon H (control steps)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.53, 1.025),
               ncol=3, frameon=False, fontsize=6.5)
    save(fig, "figD06_candidate_design",
         "X1 candidate-design ablation on matched 128-scenario subsets: fixed-denominator relative excess cost with pointwise paired-scenario 95% intervals and exact applied-action no-change rate. Designated hold-slot rates are reported separately because K=50 contains duplicate zero-offset candidates. Horizontal references mark 0% and the 5% margin. The matched K=10/H=8 baseline is open-circle-marked. Feedback rollout is a separate candidate-set point. The full Burgers n=384 anchor is open-star-marked, outside the matched design trend.",
         [source])

