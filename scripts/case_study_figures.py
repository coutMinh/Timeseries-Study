"""Find highly volatile series in the official PATRA run, draw them, and write docs/case_study/CASE_STUDY.md.

Section 5.8 of the paper names "highly volatile, non-stationary sequences" as a limitation, using
reasoning:5 of PATRA-EVAL (its Fig. 5). This script finds series like it in the official re-run and
shows each one with the question, the gold answer and PATRA's answer.

Volatility of a series y: roughness = mean(|y[t+1] - y[t]|) / std(y). A smooth trend scores near 0;
white noise scores about 1.13. "Volatile" = roughness at or above the 90th percentile of all
single-series eval samples (reasoning:5 is at the 96th). Non-stationarity is reported per series as the
change between its two halves: std ratio and mean shift in units of std.

Input: results/runs/patra__official/predictions.jsonl
Output: results/analysis/volatile_cases.json, docs/case_study/*.png, docs/case_study/CASE_STUDY.md
Usage (from the repository root): python scripts/case_study_figures.py
"""
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PRED = ROOT / "results" / "runs" / "patra__official" / "predictions.jsonl"
STATS = ROOT / "results" / "analysis" / "volatile_cases.json"
OUT = ROOT / "docs" / "case_study"
PERCENTILE = 90
MIN_LENGTH = 12
TREND = re.compile(r"trend|direction|general behavior|overall movement", re.I)
OPEN_EXAMPLES = ("comprehension:89", "reasoning:305", "comprehension:291")  # open-ended trend questions, also non-stationary

# Reference palette (dataviz skill, light mode). One series per panel, so one hue.
SURFACE, INK, INK_2, MUTED, GRID, SERIES = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#2a78d6"


def features(y):
    y = np.asarray(y, dtype=float)
    if len(y) < MIN_LENGTH or y.std() == 0:
        return None
    half = len(y) // 2
    s1, s2 = y[:half].std(), y[half:].std()
    return {
        "roughness": float(np.abs(np.diff(y)).mean() / y.std()),
        "lag1_autocorr": float(np.corrcoef(y[:-1], y[1:])[0, 1]),
        "std_ratio_halves": float(max(s1, s2) / max(min(s1, s2), 1e-9)),
        "mean_shift_halves": float(abs(y[half:].mean() - y[:half].mean()) / y.std()),
        "ols_slope": float(np.polyfit(np.arange(len(y)), y, 1)[0]),
    }


def esc(text):
    """Keep tags such as <think> and <ts><ts/> visible in Markdown previews."""
    return str(text).replace("\n", " ").strip().replace("|", "/").replace("<", "&lt;").replace(">", "&gt;")


def verdict(row):
    return "correct" if float(row["score"]) == 1 else "WRONG"


def panel(ax, row):
    y = np.asarray(row["timeseries"][0], dtype=float)
    t = np.arange(len(y))
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.plot(t, y, color=SERIES, linewidth=2, zorder=3)
    ax.plot(t, y, "o", color=SERIES, markersize=4, markeredgecolor=SURFACE, markeredgewidth=1, zorder=4)
    ax.set_title(row["sample_id"], loc="left", fontsize=10, color=INK, fontweight="bold", pad=18)
    if row["metric"] == "acc":
        sub = f"gold {row['answer']}  |  PATRA {row['pred_official']}  {verdict(row)}"
    else:
        sub = f"open-ended  |  ROUGE-L {float(row['score']) * 100:.1f}"
    ax.text(0, 1.03, sub, transform=ax.transAxes, fontsize=8, color=INK_2, va="bottom")


def figure(row, path):
    """One small plot per question, so the answers sit right under their series."""
    fig, ax = plt.subplots(figsize=(6.4, 2.4), facecolor=SURFACE)
    panel(ax, row)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def section(row, plot_name, heading_note=""):
    lines = [
        f"## `{row['sample_id']}`{heading_note}",
        "",
        f"![{row['sample_id']}](plots/{plot_name})",
        "",
        f"**Câu hỏi:** {esc(row['input'])}",
        "",
        f"**Đáp án đúng:** {esc(row['answer'])}",
        "",
    ]
    if row["metric"] == "acc":
        lines += [f"**PATRA:** {esc(row['pred_official'])} ({verdict(row).lower()})", "",
                  f"**Câu trả lời đầy đủ của PATRA:** {esc(row['response'])}", ""]
    else:
        lines += [f"**PATRA:** {esc(row['response'])}", ""]
    lines += [
        f"Độ biến động {row['roughness']:.2f}, độ dốc {row['ols_slope']:+.2f}, tỷ lệ độ lệch chuẩn hai nửa "
        f"{row['std_ratio_halves']:.2f}. Chuỗi: " + ", ".join(f"{v:g}" for v in row["timeseries"][0]),
        "",
    ]
    return lines


def accuracy(rows):
    labeled = [r for r in rows if r["metric"] == "acc"]
    correct = sum(float(r["score"]) == 1 for r in labeled)
    return {"n": len(labeled), "correct": correct, "acc_pct": round(100 * correct / len(labeled), 2) if labeled else None}


def main():
    rows = [json.loads(line) for line in PRED.open(encoding="utf-8")]
    single = []
    for r in rows:
        if len(r["timeseries"]) == 1:
            f = features(r["timeseries"][0])
            if f:
                single.append({**r, **f})
    threshold = float(np.percentile([r["roughness"] for r in single], PERCENTILE))
    volatile = [r for r in single if r["roughness"] >= threshold]
    rest = [r for r in single if r["roughness"] < threshold]

    def is_trend(r):
        return bool(TREND.search(r["input"]))

    trend_vol = sorted([r for r in volatile if r["metric"] == "acc" and is_trend(r)],
                       key=lambda r: (r["task"], r["row"]))
    trend_rest = [r for r in rest if r["metric"] == "acc" and is_trend(r)]
    ref = next(r for r in single if r["sample_id"] == "reasoning:5")
    open_rows = [next(r for r in volatile if r["sample_id"] == s) for s in OPEN_EXAMPLES]

    stats = {
        "source": str(PRED.relative_to(ROOT)).replace("\\", "/"),
        "definition": "roughness = mean|diff| / std; volatile = roughness >= percentile over single-series samples",
        "percentile": PERCENTILE, "min_length": MIN_LENGTH, "threshold": round(threshold, 4),
        "n_single_series": len(single), "n_volatile": len(volatile),
        "reference_reasoning_5": {k: round(ref[k], 4) for k in ("roughness", "lag1_autocorr", "std_ratio_halves",
                                                                  "mean_shift_halves", "ols_slope")},
        "reference_roughness_percentile": round(100 * float(np.mean([r["roughness"] < ref["roughness"] for r in single])), 1),
        "reference_percentile_std_ratio": round(100 * float(np.mean([r["std_ratio_halves"] < ref["std_ratio_halves"]
                                                                     for r in single])), 1),
        "reference_percentile_mean_shift": round(100 * float(np.mean([r["mean_shift_halves"] < ref["mean_shift_halves"]
                                                                      for r in single])), 1),
        "labeled_all_single": accuracy(single), "labeled_volatile": accuracy(volatile), "labeled_rest": accuracy(rest),
        "trend_labeled_volatile": accuracy(trend_vol), "trend_labeled_rest": accuracy(trend_rest),
        "trend_volatile_samples": [r["sample_id"] for r in trend_vol],
    }
    STATS.write_text(json.dumps(stats, indent=1), encoding="utf-8")

    OUT.mkdir(parents=True, exist_ok=True)
    for old in ("reasoning_5.png", "reasoning_432.png", "volatile_trend_labeled.png", "volatile_trend_open.png"):
        (OUT / old).unlink(missing_ok=True)
    plots = OUT / "plots"
    plots.mkdir(exist_ok=True)
    for old in plots.glob("*.png"):
        old.unlink()

    def plot_name(row):
        return row["sample_id"].replace(":", "_") + ".png"

    for row in trend_vol + open_rows:
        figure(row, plots / plot_name(row))

    tv, tr = stats["trend_labeled_volatile"], stats["trend_labeled_rest"]
    lv, lr = stats["labeled_volatile"], stats["labeled_rest"]
    lines = [
        "# PATRA trên chuỗi biến động mạnh (highly volatile)",
        "",
        "Section 5.8 của paper nêu hạn chế khi xử lý \"highly volatile, non-stationary sequences\", minh họa bằng "
        "Fig. 5, chính là mẫu `reasoning:5` của PATRA-EVAL. File này tìm các chuỗi giống vậy trong lần chạy lại "
        "`patra__official` và đặt câu trả lời của PATRA cạnh đáp án đúng.",
        "",
        f"Sinh bởi [`case_study_figures.py`](../../scripts/case_study_figures.py) từ "
        f"[`predictions.jsonl`](../../results/runs/patra__official/predictions.jsonl); số liệu trong "
        f"[`volatile_cases.json`](../../results/analysis/volatile_cases.json).",
        "",
        "## Cách chọn",
        "",
        "- **Độ biến động** = trung bình |bước nhảy giữa hai điểm liên tiếp| chia cho độ lệch chuẩn của chuỗi. "
        "Chuỗi trơn có xu hướng rõ cho giá trị gần 0; nhiễu trắng khoảng 1.13.",
        f"- **Chuỗi biến động mạnh** = độ biến động từ phân vị {PERCENTILE} trở lên (≥ {threshold:.2f}) trong "
        f"{len(single):,} mẫu một chuỗi, dài từ {MIN_LENGTH} điểm. Có **{len(volatile)}** chuỗi như vậy.",
        f"- `reasoning:5` có độ biến động {ref['roughness']:.2f}, ở phân vị {stats['reference_roughness_percentile']}.",
        f"- **Về \"non-stationary\":** nếu đo bằng thay đổi giữa hai nửa chuỗi thì `reasoning:5` không nổi bật: tỷ lệ "
        f"độ lệch chuẩn hai nửa {ref['std_ratio_halves']:.2f} (phân vị {stats['reference_percentile_std_ratio']}), "
        f"chênh trung bình {ref['mean_shift_halves']:.2f} lần độ lệch chuẩn (phân vị "
        f"{stats['reference_percentile_mean_shift']}). Vì vậy file này lọc theo độ biến động, còn tỷ lệ độ lệch chuẩn "
        "ghi trong bảng để tham khảo.",
        "",
        "## Tóm tắt",
        "",
        "| Nhóm câu có nhãn | Số câu | PATRA đúng | Accuracy |",
        "|---|---|---|---|",
        f"| Câu về xu hướng, chuỗi biến động mạnh | {tv['n']} | {tv['correct']} | {tv['acc_pct']}% |",
        f"| Câu về xu hướng, các chuỗi còn lại | {tr['n']} | {tr['correct']} | {tr['acc_pct']}% |",
        f"| Mọi câu, chuỗi biến động mạnh | {lv['n']} | {lv['correct']} | {lv['acc_pct']}% |",
        f"| Mọi câu, các chuỗi còn lại | {lr['n']} | {lr['correct']} | {lr['acc_pct']}% |",
        "",
        "Câu về xu hướng: câu hỏi chứa \"trend\", \"direction\", \"general behavior\" hoặc \"overall movement\". "
        "Chưa kiểm định thống kê; các nhóm nhỏ.",
        "",
        "## Ghi chú",
        "",
        "- Mọi câu trả lời lấy từ lần chạy lại `patra__official` (weight PATRA-7B công bố, prompt eval chính thức, "
        "greedy), không chép từ paper.",
        "- Câu trả lời của `reasoning:5` trùng phần `<think>` trong Fig. 5 của paper; phần đáp án khác định dạng "
        "(`\\boxed{True}` thay vì `<answer>True</answer>`).",
        "- Accuracy chạy lại thấp hơn paper ở mọi task, nên các tỷ lệ trên mô tả bản công bố, không phải số trong paper.",
        f"- Phía dưới: {tv['n']} câu có nhãn về xu hướng trên chuỗi biến động mạnh (tất cả, không chọn lọc, xếp theo "
        f"task và số dòng), rồi {len(open_rows)} câu mở chọn tay (chuỗi vừa biến động mạnh vừa đổi biên độ giữa hai "
        "nửa, tỷ lệ độ lệch chuẩn hai nửa từ phân vị 75 trở lên; ROUGE-L chỉ để tham khảo).",
        "",
    ]
    for r in trend_vol:
        note = " (Fig. 5 của paper)" if r["sample_id"] == "reasoning:5" else ""
        lines += section(r, plot_name(r), f" · có nhãn{note}")
    for r in open_rows:
        lines += section(r, plot_name(r), " · câu mở")
    (OUT / "CASE_STUDY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: stats[k] for k in ("threshold", "n_volatile", "trend_labeled_volatile", "trend_labeled_rest",
                                            "labeled_volatile", "labeled_rest", "reference_roughness_percentile")}))


if __name__ == "__main__":
    main()
