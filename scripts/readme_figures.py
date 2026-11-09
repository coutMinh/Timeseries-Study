"""Draw the two README figures from results/.

Run:  python scripts/readme_figures.py
Writes docs/figures/repro_vs_paper.png and docs/figures/reads_the_series.png.
Backgrounds are opaque white so the figures read the same in GitHub light and dark mode.
"""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "docs" / "figures"

INK = "#1E2229"
MUTED = "#5A5F69"
REFERENCE = "#8A919C"  # paper values and baselines that ignore the series
ORANGE = "#D9661F"  # PATRA re-run
ORANGE_LIGHT = "#ECA87C"  # PATRA with the series path disabled
BLUE = "#1F77B4"  # baseline that reads series statistics
TASKS = ["comprehension", "recognition", "reasoning", "prescience"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
    "axes.edgecolor": "#C9CDD3", "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
})


def load(*parts):
    return json.loads(RESULTS.joinpath(*parts).read_text(encoding="utf8"))


def rows(*parts):
    with RESULTS.joinpath(*parts).open(encoding="utf8", newline="") as f:
        return list(csv.DictReader(f))


def bar_ends(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(left=False)


def repro_vs_paper():
    table = {r["task"]: r for r in rows("tables", "table1_repro.csv") if r["system"] == "patra" and r["condition"] == "official"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    width = 0.38
    for ax, (key, paper_key, title, top) in zip(axes, [("acc_pct", "paper_acc", "Accuracy (%)", 75), ("rouge_pct", "paper_rouge", "ROUGE-L", 36)]):
        x = range(len(TASKS))
        paper = [float(table[t][paper_key]) for t in TASKS]
        rerun = [float(table[t][key]) for t in TASKS]
        for offset, values, color, label in ((-width / 2, paper, REFERENCE, "Paper"), (width / 2, rerun, ORANGE, "Re-run (released checkpoint)")):
            bars = ax.bar([i + offset for i in x], values, width - 0.03, color=color, label=label)
            for b, v in zip(bars, values):
                ax.text(b.get_x() + b.get_width() / 2, v + top * 0.015, f"{v:.1f}", ha="center", va="bottom", fontsize=9.5, color=INK)
        ax.set_xticks(list(x), [t.capitalize() for t in TASKS])
        ax.set_ylim(0, top)
        ax.set_yticks([])
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
        bar_ends(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2, fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT / "repro_vs_paper.png", dpi=200)
    plt.close(fig)


def reads_the_series():
    paired = {r["b"]: r for r in rows("analysis", "paired.csv") if r["a"] == "patra__official" and r["grouping"] == "overall"}
    labeled = load("analysis", "benchmark_probes.json")["labeled"]
    lift = labeled["series_lift"]
    total = sum(int(r["acc_n"]) for r in rows("tables", "table1_repro.csv") if r["system"] == "patra" and r["condition"] == "official")
    bars = [
        ("Always answer False / B", labeled["constant_from_train_labels"]["acc_pct"], REFERENCE),
        ("PATRA re-run", labeled["patra_official"]["acc_pct"], ORANGE),
        ("PATRA, series replaced by zeros", float(paired["patra__ts_zero"]["acc_b"]), ORANGE_LIGHT),
        ("PATRA, random encoder + alignment", float(paired["patra__reinit_modules"]["acc_b"]), ORANGE_LIGHT),
        ("Question only (logistic regression)", labeled["question_only_tfidf_logreg"]["acc_pct"], REFERENCE),
        ("Question + 16 series statistics (trees)", lift["question_plus_series_stats_trees"]["acc_pct"], BLUE),
        ("Same trees, series shuffled at test", lift["same_model_with_series_shuffled_at_test"]["acc_pct"], REFERENCE),
        ("PATRA in the paper (converted)", labeled["paper_total_correct_converted"] / total * 100, "white"),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    y = list(range(len(bars)))[::-1]
    for yi, (name, value, color) in zip(y, bars):
        edge = MUTED if color == "white" else color
        ax.barh(yi, value, 0.68, color=color, edgecolor=edge, linestyle="--" if color == "white" else "-", linewidth=1.2)
        ax.text(value + 0.6, yi, f"{value:.2f}", va="center", fontsize=10, color=INK, fontweight="bold" if name == "PATRA re-run" else "normal")
    ax.set_yticks(y, [b[0] for b in bars])
    ax.set_xlim(0, 66)
    ax.set_xticks([])
    ax.set_title(f"Accuracy (%) on the {total:,} labeled questions", loc="left", fontsize=12, fontweight="bold")
    bar_ends(ax)
    ax.spines["bottom"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "reads_the_series.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    repro_vs_paper()
    reads_the_series()
    print("wrote", OUT)
