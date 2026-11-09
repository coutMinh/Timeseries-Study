"""What PATRA's latent decomposition keeps of a volatile series, for five failure cases of the official re-run.

PATRA normalizes each series, cuts it into 16-point patches, encodes each patch into one token
(ts_encoder), and splits the tokens into a trend (moving average over 25 tokens) and a seasonal part
(token minus trend) before alignment (modeling_qwen2.py:130-160, :172-192, :456-457).
This script recomputes those tokens on CPU with the published weights, reusing alignment_mechanics.py,
and maps every 3,584-dimensional token to one number with PCA, as Section 5.10 of the paper does.
The PCA is fitted on all single-series evaluation samples, one fit for trend tokens and one for
seasonal tokens. The sign of each axis is chosen so that larger values go with larger patch means.

Writes results/analysis/failure_decomposition.json, the figure
docs/case_study/failure_decomposition.pdf and one PNG per case in
docs/case_study/decomposition/.

    python scripts/failure_decomposition.py --cache-dir <folder with align_tensors.pt, outside the repo>
Needs torch, numpy, matplotlib. No GPU.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from alignment_mechanics import PATCH, Encoder, load_weights, moving_average, sp_encoding  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "results" / "runs" / "patra__official" / "predictions.jsonl"
OUT_JSON = ROOT / "results" / "analysis" / "failure_decomposition.json"
# Not report/figures: the report has a hand-edited figure with the same name.
OUT_PDF = ROOT / "docs" / "case_study" / "failure_decomposition.pdf"
OUT_PNG = ROOT / "docs" / "case_study" / "decomposition"
CASES = ("reasoning:5", "reasoning:77", "reasoning:380", "reasoning:615", "comprehension:89")

# Reference palette (dataviz skill, light mode), slots 2, 1, 3: series, trend, seasonal.
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
C_SERIES, C_TREND, C_SEASONAL = "#eb6834", "#2a78d6", "#1baf7a"
NUM = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")


def decompose(encoder, series):
    with torch.no_grad():
        tokens = encoder([series])
        trend = moving_average(tokens)
    return tokens, trend, tokens - trend


def patch_means(series):
    x = sp_encoding(series).numpy()
    count = (len(x) + PATCH - 1) // PATCH
    x = np.concatenate([x, np.zeros(count * PATCH - len(x))]).reshape(count, PATCH)
    return x.mean(axis=1)


def pca_axis(rows, orient):
    """First principal axis of the stacked tokens, signed to correlate positively with `orient`."""
    x = np.concatenate(rows)
    mean = x.mean(axis=0)
    _, s, vt = np.linalg.svd(x - mean, full_matrices=False)
    axis = vt[0]
    if np.corrcoef((x - mean) @ axis, np.concatenate(orient))[0, 1] < 0:
        axis = -axis
    return mean, axis, float(s[0] ** 2 / (s ** 2).sum())


def think_text(response):
    """The reasoning sentence of a labeled answer, or the whole open-ended answer."""
    match = re.search(r"<think>(.*?)</think>", response, re.S)
    return (match.group(1) if match else response).strip()


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=7, length=2)


def draw_steps(ax, values, length, color):
    """One horizontal segment per token over the points it encodes."""
    for i, v in enumerate(values):
        start, stop = i * PATCH, min((i + 1) * PATCH, length) - 1
        ax.plot([start, stop], [v, v], color=color, linewidth=2.5, solid_capstyle="round")
        ax.plot([(start + stop) / 2], [v], "o", color=color, markersize=4.5, markeredgecolor=SURFACE)


def draw_case(axes, case, ylims, first_column):
    series = np.asarray(case["series"], dtype=float)
    t = np.arange(len(series))
    for ax in axes:
        style(ax)
        ax.set_xlim(-0.5, max(len(series) - 0.5, 1))
        for b in range(PATCH, len(series), PATCH):
            ax.axvline(b - 0.5, color=MUTED, linewidth=0.8, linestyle=(0, (3, 2)))
    axes[0].plot(t, series, color=C_SERIES, linewidth=2.2)
    label = f"gold {case['gold_short']} | PATRA {case['pred_short']}" if case["pred"] else "open-ended"
    axes[0].set_title(f"{case['sample_id']}\n{label}", fontsize=7.5, color=INK, loc="left")
    draw_steps(axes[1], case["trend_pc1"], len(series), C_TREND)
    draw_steps(axes[2], case["seasonal_pc1"], len(series), C_SEASONAL)
    axes[1].set_ylim(*ylims["trend"])
    axes[2].set_ylim(*ylims["seasonal"])
    axes[2].set_xlabel("time step", fontsize=7, color=INK_2)
    if first_column:
        axes[0].set_ylabel("raw value", fontsize=7, color=INK_2)
        axes[1].set_ylabel("latent trend\n(PC1)", fontsize=7, color=INK_2)
        axes[2].set_ylabel("latent seasonal\n(PC1)", fontsize=7, color=INK_2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    encoder = Encoder(load_weights(args.cache_dir))

    rows = [json.loads(line) for line in PRED.open(encoding="utf-8")]
    single = [r for r in rows if len(r["timeseries"]) == 1]
    trends, seasonals, orient_t, orient_s = [], [], [], []
    for r in single:
        _, trend, seasonal = decompose(encoder, r["timeseries"][0])
        means = patch_means(r["timeseries"][0])
        trends.append(trend.numpy())
        seasonals.append(seasonal.numpy())
        orient_t.append(means)
        orient_s.append(means - means.mean())
    mean_t, axis_t, evr_t = pca_axis(trends, orient_t)
    mean_s, axis_s, evr_s = pca_axis(seasonals, orient_s)

    by_id = {r["sample_id"]: r for r in rows}
    cases = []
    for sid in CASES:
        r = by_id[sid]
        series = r["timeseries"][0]
        tokens, trend, seasonal = decompose(encoder, series)
        answer = think_text(r["response"])
        gold = str(r["answer"]).strip()
        cases.append({
            "sample_id": sid, "question": r["input"], "question_format": r["question_format"],
            "series": series, "n_points": len(series), "n_tokens": int(tokens.shape[0]),
            "gold": gold, "gold_short": gold if len(gold) <= 3 else "open-ended",
            "pred": r["pred_official"], "pred_short": r["pred_official"] or "open-ended",
            "score": float(r["score"]), "response": r["response"], "reasoning": answer,
            "trend_pc1": [round(float(v), 4) for v in (trend.numpy() - mean_t) @ axis_t],
            "seasonal_pc1": [round(float(v), 4) for v in (seasonal.numpy() - mean_s) @ axis_s],
            "seasonal_norm_over_token_norm": [round(float(a), 4) for a in seasonal.norm(dim=1) / tokens.norm(dim=1)],
            "numbers_in_answer": NUM.findall(answer),
            "numbers_in_raw_series": [n for n in NUM.findall(answer) if any(abs(float(n) - v) < 0.006 for v in series)],
            "numbers_in_normalized_series": [n for n in NUM.findall(answer)
                                             if any(abs(float(n) - float(v)) < 0.006 for v in sp_encoding(series))],
        })

    def limits(key):
        values = [v for c in cases for v in c[key]]
        pad = 0.15 * (max(values) - min(values) or 1.0)
        return min(values) - pad, max(values) + pad

    ylims = {"trend": limits("trend_pc1"), "seasonal": limits("seasonal_pc1")}
    report = {
        "source": str(PRED.relative_to(ROOT)).replace("\\", "/"), "patch": PATCH,
        "pca": {"n_single_series_samples": len(single), "trend_pc1_explained_variance": round(evr_t, 4),
                "seasonal_pc1_explained_variance": round(evr_s, 4)},
        "cases": cases,
    }
    OUT_JSON.write_text(json.dumps(report, indent=1), encoding="utf-8")

    # Report figure: one column per case; rows are the series and its two latent components.
    fig, axes = plt.subplots(3, len(cases), figsize=(7.0, 3.6), facecolor=SURFACE, sharex="col",
                             gridspec_kw={"height_ratios": [1.3, 1, 1]})
    for j, case in enumerate(cases):
        draw_case(axes[:, j], case, ylims, j == 0)
    fig.tight_layout(pad=0.4, h_pad=0.3, w_pad=0.4)
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, facecolor=SURFACE)
    fig.savefig(OUT_PNG.parent / "failure_decomposition.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)

    OUT_PNG.mkdir(parents=True, exist_ok=True)
    for case in cases:
        fig, axes = plt.subplots(3, 1, figsize=(4.2, 4.2), facecolor=SURFACE, sharex=True,
                                 gridspec_kw={"height_ratios": [1.3, 1, 1]})
        draw_case(axes, case, ylims, True)
        fig.tight_layout(pad=0.4)
        fig.savefig(OUT_PNG / (case["sample_id"].replace(":", "_") + ".png"), dpi=170, facecolor=SURFACE)
        plt.close(fig)

    print(json.dumps({"pca": report["pca"], "cases": [{k: c[k] for k in (
        "sample_id", "n_tokens", "trend_pc1", "seasonal_pc1", "seasonal_norm_over_token_norm",
        "numbers_in_answer", "numbers_in_raw_series", "numbers_in_normalized_series")} for c in cases]}, indent=1))


if __name__ == "__main__":
    main()
