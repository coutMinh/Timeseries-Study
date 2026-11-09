"""Evidence for the five PATRA code-review items. CPU only, no model weights.

Reads the rerun in results/runs plus the pinned upstream code and training data, then writes
results/analysis/code_review_checks.json. Upstream code is only read as text, never executed;
the small functions copied below are marked with their upstream location.

    python scripts/code_review_checks.py --cache-dir <folder outside the repo>
"""
import argparse
import hashlib
import json
import math
import random
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_REV = "5cf8191e"
TRAIN_REV = "511dcecc2b1acda439e81d4758f998c246257036"
UPSTREAM = {
    "modeling_qwen2.py": f"https://huggingface.co/DecisionIntelligence/PATRA-7B/resolve/{MODEL_REV}/modeling_qwen2.py",
    "processing_qwen2_ts.py": f"https://huggingface.co/DecisionIntelligence/PATRA-7B/resolve/{MODEL_REV}/processing_qwen2_ts.py",
    "config.json": f"https://huggingface.co/DecisionIntelligence/PATRA-7B/resolve/{MODEL_REV}/config.json",
    "grpo.jsonl": f"https://huggingface.co/datasets/DecisionIntelligence/PATRA-TRAIN/resolve/{TRAIN_REV}/grpo.jsonl",
    "sft.jsonl": f"https://huggingface.co/datasets/DecisionIntelligence/PATRA-TRAIN/resolve/{TRAIN_REV}/sft.jsonl",
}
EXPECTED_LINES = {
    "modeling_qwen2.py": {
        137: "patch_cnt = (valid_lengths + self.patch_size - 1) // self.patch_size",
        145: "xi = x[i, :vl, :1]",
        200: "return res, moving_mean",
        344: "series_decomp(kernel_size=config.get('decomp_kernel_size', 25))",
        456: "ts_features = ts_features.unsqueeze(0)",
        457: "ts_seasonal, ts_trend = self.ts_decomposer(ts_features)",
        470: "trend_qkv = torch.cat([text_trend[i:i+1], seasonal_kv], dim=1)",
        471: "seasonal_qkv = torch.cat([text_seasonal[i:i+1], trend_kv], dim=1)",
    },
    "processing_qwen2_ts.py": {
        43: '# prompt = f"[Value Offset: {-mean:.4f}|Value Scaling: {scale_factor:.4f}]<ts>"',
        44: "prompt = '<ts>'",
    },
}
# Table 1 of the paper, read from the PDF. Task sizes come from the eval data.
TASK_N = {"comprehension": 116, "recognition": 725, "reasoning": 148, "prescience": 108}
PAPER_PATRA = {"comprehension": 56.03, "recognition": 64.69, "reasoning": 44.59, "prescience": 52.78}
PAPER_BEST_OTHER = {
    "comprehension": ("ChatTS-7B", 44.83),
    "recognition": ("Qwen2.5-7B", 45.51),
    "reasoning": ("TimeOmni-1-7B", 42.56),
    "prescience": ("ITFormer-7B", 44.44),
}
PLACEHOLDER = "<ts><ts/>"
NUMBER_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")
SMALL_COUNTS = {"0", "1", "2", "3", "4", "5", "10"}
COUNT_NOUNS = r"(?:points?|values?|steps?|hours?|days?|weeks?|months?|years?|minutes?|seconds?|observations?|readings?|samples?|series)"


def fetch(cache, name):
    path = cache / name
    if not path.exists():
        with urllib.request.urlopen(UPSTREAM[name], timeout=300) as response:
            path.write_bytes(response.read())
    return path


def load_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def pct(k, n):
    return round(100 * k / n, 2) if n else None


def percentile(values, q):
    return round(float(np.percentile(values, q)), 4)


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def two_prop_p(k1, k2, n):
    pooled = (k1 + k2) / (2 * n)
    se = math.sqrt(2 * pooled * (1 - pooled) / n)
    return math.erfc(abs(k1 - k2) / n / se / math.sqrt(2)) if se else 1.0


def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    centre, half = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (centre - half), 2), round(100 * (centre + half), 2)]


# ---- Item 2: copied from modeling_qwen2.py:172-179 and :190-192, in numpy.
def moving_avg(x, kernel=25):
    pad = (kernel - 1) // 2
    padded = np.concatenate([np.repeat(x[:1], pad, axis=0), x, np.repeat(x[-1:], pad, axis=0)], axis=0)
    return np.stack([padded[i:i + kernel].mean(axis=0) for i in range(len(padded) - kernel + 1)])


def decompose(x, kernel=25):
    trend = moving_avg(x, kernel)
    return x - trend, trend


def n_patches(length, patch_size=16):
    return (length + patch_size - 1) // patch_size  # modeling_qwen2.py:137


def patch_table(rows):
    per_series, per_sample = Counter(), Counter()
    lengths = []
    for row in rows:
        counts = [n_patches(len(s)) for s in row["timeseries"]]
        lengths += [len(s) for s in row["timeseries"]]
        per_series.update(counts)
        per_sample[sum(counts)] += 1
    total = sum(per_series.values())
    return {
        "n_series": total,
        "pct_series_by_patches": {str(k): pct(v, total) for k, v in sorted(per_series.items())},
        "series_over_2_patches": sum(v for k, v in per_series.items() if k > 2),
        "samples_with_over_2_patches_total": sum(v for k, v in per_sample.items() if k > 2),
        "longest_series": max(lengths),
        "most_common_length": Counter(lengths).most_common(1)[0],
    }


def item2(eval_rows, grpo):
    rng = np.random.default_rng(0)
    one = rng.normal(size=(1, 3584))
    res_1, trend_1 = decompose(one)
    two = rng.normal(size=(2, 3584))
    res_2, trend_2 = decompose(two)
    difference = two[0] - two[1]
    # Patches of a batch are concatenated before the moving average (:456-457).
    a, b, other = rng.normal(size=(2, 8)), rng.normal(size=(2, 8)), rng.normal(size=(2, 8))
    alone = moving_avg(a)
    with_b = moving_avg(np.concatenate([a, b]))[:2]
    with_other = moving_avg(np.concatenate([a, other]))[:2]
    return {
        "kernel_size": 25,
        "kernel_in_raw_points": 25 * 16,
        "one_patch": {
            "residual_max_abs": float(np.abs(res_1).max()),
            "trend_equals_input": bool(np.allclose(trend_1, one)),
        },
        "two_patches": {
            "residual_vs_difference_ratio": round(float(np.linalg.norm(res_2[0]) / np.linalg.norm(difference)), 4),
            "residuals_are_opposite": bool(np.allclose(res_2[0], -res_2[1])),
            "trend_gap_vs_difference_ratio": round(float(np.linalg.norm(trend_2[0] - trend_2[1]) / np.linalg.norm(difference)), 4),
        },
        "neighbour_sample_changes_trend": {
            "max_abs_change_when_batched_with_b": round(float(np.abs(alone - with_b).max()), 4),
            "max_abs_change_b_vs_other_neighbour": round(float(np.abs(with_b - with_other).max()), 4),
        },
        "eval": patch_table(eval_rows),
        "train_grpo": patch_table(grpo),
        "train_batch_size_sft_script": 2,
        "eval_batch_size_this_rerun": 1,
    }


# ---- Item 3
def count_xml(text):
    """Copied from train/src/llamafactory/train/grpo/workflow.py:164-180."""
    count = 0.0
    if text.count("<task_type>") == 1:
        count += 0.125
    if text.count("</task_type>") == 1:
        count += 0.125
    if text.count("<think>") == 1:
        count += 0.125
    if text.count("</think>\n") == 1:
        count += 0.125
    if text.count("<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("</answer>\n")[-1]) * 0.001
    if text.count("</answer>\n") == 1:
        count += 0.125
        count -= (len(text.split("</answer>\n")[-1]) - 1) * 0.001
    return count


def advantages(rewards):
    """grpo_trainer.py:1313-1322 and Eq. 14: standardise within one group, torch.std is unbiased."""
    r = np.asarray(rewards, dtype=float)
    return (r - r.mean()) / (r.std(ddof=1) + 1e-4)


def pair_case(format_a, task_a, format_b, task_b):
    out = {}
    for label, factor in (("task_reward_0_to_2", 1.0), ("task_reward_0_to_1", 0.5)):
        rewards = [format_a + factor * task_a, format_b + factor * task_b]
        out[label] = {"rewards": rewards, "advantage_of_a": round(float(advantages(rewards)[0]), 4)}
    return out


def item3(grpo):
    rng = np.random.default_rng(0)
    group = rng.uniform(0, 2, 8)
    well_formed = ("<think>x</think>\n<answer>\nA\n</answer>\n")
    with_task_type = well_formed + "<task_type>multiple_choice</task_type>"
    no_closing = "<think>" + "x" * 800 + "</think>\n<answer>\nA\n</answer>"
    return {
        "published_config": {"num_generations": 2, "scale_rewards_default": True, "reward_weights_default": None},
        "scale_invariance_G8": {
            "max_abs_change_after_multiplying_rewards_by_0.5": round(float(np.abs(advantages(group) - advantages(group * 0.5)).max()), 6),
        },
        "advantage_G2_by_reward_gap": {str(gap): round(float(advantages([gap, 0.0])[0]), 4) for gap in (0.001, 0.01, 0.1, 0.5, 1.0, 2.0)},
        "pair_cases": {
            "correct_bad_format_vs_wrong_full_format": pair_case(0.0, 2.0, 0.75, 0.5),
            "rouge_gap_0.05_vs_one_tag_0.125": pair_case(0.0, 2 * 0.30, 0.125, 2 * 0.25),
            "rouge_gap_0.10_vs_one_tag_0.125": pair_case(0.0, 2 * 0.35, 0.125, 2 * 0.25),
            "both_same_answer_same_format": pair_case(0.75, 2.0, 0.75, 2.0),
        },
        "format_reward_range_checks": {
            "without_task_type_tags": round(count_xml(well_formed), 4),
            "with_task_type_required_by_prompt": round(count_xml(with_task_type), 4),
            "response_ending_right_after_answer_tag_800_chars": round(count_xml(no_closing), 4),
        },
        "train_grpo_question_formats": dict(Counter(row["question_format"] for row in grpo)),
    }


# ---- Item 4
def item4(paper_counts_runs):
    table = {}
    for task, n in TASK_N.items():
        name, other_pct = PAPER_BEST_OTHER[task]
        k1, k2 = round(PAPER_PATRA[task] * n / 100), round(other_pct * n / 100)
        table[task] = {
            "n": n, "patra_correct": k1, "best_other": name, "best_other_correct": k2, "gap_questions": k1 - k2,
            "patra_ci95": wilson(k1, n), "best_other_ci95": wilson(k2, n),
            "unpaired_p": round(two_prop_p(k1, k2, n), 4),
            "lowest_possible_mcnemar_p": mcnemar_exact(k1 - k2, 0),
        }
    return {"paper_table1": table, "this_rerun": paper_counts_runs}


# ---- Item 5
def sp_encoding(series):
    """Copied from processing_qwen2_ts.py:36-41, without the mask channel."""
    x = np.asarray(series, dtype=float)
    mean = x.mean()
    scaled = x - mean
    scale = 1.0
    if np.any(np.abs(scaled) >= 3.0):
        scale = np.max(np.abs(scaled)) / 3.0
        scaled = scaled / scale
    return scaled, float(mean), float(scale)


def donor_map(rows):
    pools = defaultdict(list)
    for row in rows:
        for index, series in enumerate(row["timeseries"]):
            pools[len(series)].append((id(row), index, series))
    mapping = {}
    for key, pool in sorted(pools.items()):
        order = list(pool)
        random.Random(f"code-review:{key}").shuffle(order)
        for position, (row_id, index, _) in enumerate(order):
            mapping[(row_id, index)] = order[(position + 1) % len(order)][2]
    return mapping


def grounding(rows, field):
    donors = donor_map(rows)
    quoted = raw = norm = other = with_numbers = 0
    for row in rows:
        in_question = set(NUMBER_RE.findall(row["input"].replace(PLACEHOLDER, " ")))
        numbers = {n for n in NUMBER_RE.findall(row[field]) if n not in SMALL_COUNTS and n not in in_question}
        if not numbers:
            continue
        with_numbers += 1
        raw_values = {v for s in row["timeseries"] for v in s}
        other_values = {v for i in range(len(row["timeseries"])) for v in donors[(id(row), i)]}
        normalized = [v for s in row["timeseries"] for v in sp_encoding(s)[0]]
        for number in numbers:
            value = float(number)
            quoted += 1
            raw += value in raw_values
            other += value in other_values
            norm += any(abs(value - v) <= 0.005 for v in normalized)
    return {"texts": len(rows), "texts_with_numbers": with_numbers, "numbers": quoted, "pct_equal_raw": pct(raw, quoted),
            "pct_equal_raw_of_other_series": pct(other, quoted), "pct_near_normalized": pct(norm, quoted)}


def value_dependent(question):
    low = question.replace(PLACEHOLDER, " ").lower()
    no_counts = re.sub(rf"\b\d+\s+(?:\w+\s+){{0,2}}?{COUNT_NOUNS}\b", " ", low)
    return bool(re.search(r"(?<![A-Za-z])-?\d+(?:\.\d+)?", no_counts))


def context_bucket(row):
    """near_raw: mean about 0 and no rescaling, so dropping offset and scale loses almost nothing."""
    near_raw = all(abs(sp_encoding(s)[1]) < 0.1 and sp_encoding(s)[2] == 1.0 for s in row["timeseries"])
    return "near_raw" if near_raw else "context_lost"


def paired_table(patra, qwen):
    qwen_by_id = {r["sample_id"]: r for r in qwen}
    groups = defaultdict(list)
    for row in patra:
        if row["metric"] != "acc":
            continue
        other = qwen_by_id[row["sample_id"]]
        groups[f"{context_bucket(row)}|value_dependent={value_dependent(row['input'])}"].append((row["score"], other["score"]))
        groups[f"{context_bucket(row)}|all"].append((row["score"], other["score"]))
        groups[f"all|value_dependent={value_dependent(row['input'])}"].append((row["score"], other["score"]))
    out = {}
    for key, pairs in sorted(groups.items()):
        b = sum(1 for a, q in pairs if a and not q)
        c = sum(1 for a, q in pairs if q and not a)
        out[key] = {"n": len(pairs), "patra_pct": pct(sum(a for a, _ in pairs), len(pairs)),
                    "qwen_pct": pct(sum(q for _, q in pairs), len(pairs)), "patra_only": b, "qwen_only": c,
                    "mcnemar_p": round(mcnemar_exact(b, c), 5)}
    return out


def rouge_table(patra, qwen):
    qwen_by_id = {r["sample_id"]: r for r in qwen}
    groups = defaultdict(list)
    for row in patra:
        if row["metric"] != "acc":
            groups[context_bucket(row)].append((row["score"], qwen_by_id[row["sample_id"]]["score"]))
    return {k: {"n": len(v), "patra_mean": round(float(np.mean([a for a, _ in v])), 4),
                "qwen_mean": round(float(np.mean([q for _, q in v])), 4)} for k, v in sorted(groups.items())}


def offset_table(rows):
    means, scales = [], []
    for row in rows:
        for series in row["timeseries"]:
            if series:
                _, mean, scale = sp_encoding(series)
                means.append(abs(mean))
                scales.append(scale)
    means, scales = np.array(means), np.array(scales)
    return {"n_series": len(means), "abs_offset_median": percentile(means, 50), "abs_offset_p90": percentile(means, 90),
            "pct_abs_offset_ge_1": pct(int((means >= 1).sum()), len(means)),
            "pct_scale_above_1": pct(int((scales > 1).sum()), len(means)), "scale_max": round(float(scales.max()), 2)}


def item5(eval_patra, eval_qwen, grpo, sft):
    s = np.array([12.0, 15.0, 11.0, 14.0, 13.0, 15.0])
    wide = 100 * s + 5000
    series_match = sum(
        1 for a, b in zip(sft, grpo)
        if len(a["timeseries"]) == len(b["timeseries"]) and all(
            len(x) == 2 * len(y) and np.allclose([v[0] for v in x[::2]], sp_encoding(y)[0], atol=0.01)
            for x, y in zip(a["timeseries"], b["timeseries"]))
    )
    open_patra = [r for r in eval_patra if r["metric"] != "acc"]
    open_qwen = [r for r in eval_qwen if r["metric"] != "acc"]
    labeled_patra = [r for r in eval_patra if r["metric"] == "acc"]
    frequent = re.compile(r"(?<![\d.])-0\.49(?!\d)")
    return {
        "invariance": {
            "shift_only_changes_input": float(np.abs(sp_encoding(s)[0] - sp_encoding(s + 1000)[0]).max()),
            "scale_above_threshold_changes_input": float(np.abs(sp_encoding(wide)[0] - sp_encoding(wide * 7)[0]).max()),
            "encoded_mean_max_abs": max(abs(float(sp_encoding(ser)[0].mean())) for r in labeled_patra for ser in r["timeseries"]),
        },
        "offset_scale": {"eval": offset_table(eval_patra), "train_grpo": offset_table(grpo)},
        "prompt_contains_value_offset": {
            "eval": sum("Value Offset" in r["input"] for r in eval_patra),
            "train_grpo": sum("Value Offset" in r["input"] for r in grpo),
        },
        "sft_series_equal_encoded_grpo_series": {"matching": series_match, "n": len(sft)},
        "number_grounding": {
            "train_grpo_gold_open_ended": grounding([r for r in grpo if r["question_format"] == "open_ended_question"], "answer"),
            "train_sft_gold_all": grounding([{"input": a["input"], "output": a["output"], "timeseries": b["timeseries"]} for a, b in zip(sft, grpo)], "output"),
            "eval_gold_open_ended": grounding(open_patra, "answer"),
            "eval_patra_open_ended": grounding(open_patra, "response"),
            "eval_qwen_open_ended": grounding(open_qwen, "response"),
        },
        "patra_open_answers_quoting_minus_0.49": {"count": sum(bool(frequent.search(r["response"])) for r in open_patra), "n": len(open_patra)},
        "labeled_patra_vs_qwen": paired_table(eval_patra, eval_qwen),
        "open_rouge_by_bucket": rouge_table(eval_patra, eval_qwen),
        "not_run": "value_context exists in patra_reproduce.py but has no run in results/runs.",
    }


def check_lines(cache):
    report = {}
    for name, expected in EXPECTED_LINES.items():
        path = fetch(cache, name)
        lines = path.read_text(encoding="utf-8").splitlines()
        for number, text in expected.items():
            assert text in lines[number - 1], f"{name}:{number} is not the expected line"
        report[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "checked_lines": sorted(expected)}
    config = json.loads(fetch(cache, "config.json").read_text(encoding="utf-8"))
    assert "decomp_kernel_size" not in config["ts"], "config now sets the kernel, so the default 25 does not apply"
    report["config_ts"] = config["ts"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True, help="where upstream files are stored, outside the repo")
    args = parser.parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    lines = check_lines(args.cache_dir)
    patra = load_jsonl(ROOT / "results/runs/patra__official/predictions.jsonl")
    qwen = load_jsonl(ROOT / "results/runs/qwen__official/predictions.jsonl")
    grpo = load_jsonl(fetch(args.cache_dir, "grpo.jsonl"))
    sft = load_jsonl(fetch(args.cache_dir, "sft.jsonl"))

    # Checks against numbers already reported, so a mismatch here stops the script.
    labeled_qwen = {r["sample_id"]: r["score"] for r in qwen if r["metric"] == "acc"}
    labeled_patra = [r for r in patra if r["metric"] == "acc"]
    assert len(labeled_patra) == 1097 and sum(r["score"] for r in labeled_patra) == 521
    b = sum(1 for r in labeled_patra if r["score"] and not labeled_qwen[r["sample_id"]])
    c = sum(1 for r in labeled_patra if labeled_qwen[r["sample_id"]] and not r["score"])
    assert (b, c) == (227, 194) and abs(mcnemar_exact(b, c) - 0.11875) < 1e-3
    assert sum(value_dependent(r["input"]) for r in labeled_patra) == 361

    reasoning = {}
    for system, rows in (("patra", patra), ("qwen", qwen)):
        members = [r for r in rows if r["metric"] == "acc" and r["task"] == "reasoning"]
        reasoning[system] = {"n": len(members), "correct": sum(r["score"] for r in members)}

    report = {
        "revisions": {"model": MODEL_REV, "train_data": TRAIN_REV},
        "upstream_files": lines,
        "item2_decomposition": item2(patra, grpo),
        "item3_reward": item3(grpo),
        "item4_reasoning": item4({"reasoning": reasoning}),
        "item5_offset_scale": item5(patra, qwen, grpo, sft),
    }
    out = ROOT / "results/analysis/code_review_checks.json"
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
