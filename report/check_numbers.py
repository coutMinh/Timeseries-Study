"""Check every number in main.tex against ../results and the paper.

Numbers from results/ are read from the files, so a changed result fails the check.
Numbers from the paper (arXiv 2602.23161v4) are listed by hand in PAPER.
Any other number in the text fails, apart from small integers in CONTEXT.

Usage: python check_numbers.py
"""
import csv
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE.parent / "results"
TEX = HERE / "main.tex"

PAPER = {
    "56.03", "64.69", "44.59", "52.78",  # Table 1 accuracy
    "25.67", "25.46", "27.36", "27.06",  # Table 1 ROUGE-L
    "0.986", "0.938",  # Section 5.10
    "200,000",  # Section 5.1
    "13.51", "16.66",  # Table 3, W/O Reasoning-Enhanced Stage (Reasoning, Prescience)
    "28.37", "30.55",  # Table 3, W/O Pattern-Aware Alignment (Reasoning, Prescience)
}
# Table 5 accuracy: (balanced reward = PATRA, original reward) per task. The report quotes the range of the gaps.
TABLE5 = {"comprehension": (56.03, 39.65), "recognition": (64.69, 48.83), "reasoning": (44.59, 37.84), "prescience": (52.78, 35.18)}
_gaps = sorted(round(a - b, 2) for a, b in TABLE5.values())
PAPER |= {f"{_gaps[0]:.2f}", f"{_gaps[-1]:.2f}"}
# Not measurements: section and equation labels, list markers, reward range, patch size, library version.
CONTEXT = {"1", "2", "3", "4", "5", "6", "14", "16", "0,2", "4.46", "5.8", "5.10"}


def load(*parts):
    return json.loads(RESULTS.joinpath(*parts).read_text(encoding="utf8"))


def rows(*parts):
    with RESULTS.joinpath(*parts).open(encoding="utf8", newline="") as f:
        return list(csv.DictReader(f))


def results_numbers():
    table = [r for r in rows("tables", "table1_repro.csv") if r["system"] == "patra" and r["condition"] == "official"]
    paired = {r["b"]: r for r in rows("analysis", "paired.csv") if r["a"] == "patra__official" and r["grouping"] == "overall"}
    probes = load("analysis", "benchmark_probes.json")
    mech = load("analysis", "alignment_mechanics.json")
    review = load("analysis", "code_review_checks.json")
    ts = load("analysis", "ts_probe_summary.json")
    labeled, open_ended = probes["labeled"], probes["open_ended"]
    lift = labeled["series_lift"]
    grounding = review["item5_offset_scale"]["number_grounding"]
    decomp = review["item2_decomposition"]
    advantage = review["item3_reward"]["advantage_G2_by_reward_gap"]
    reasoning = review["item4_reasoning"]["paper_table1"]["reasoning"]
    align = mech["alignment_attention_by_branch"]["complete"]
    token = mech["first_patch_token_single_series"]
    variants = ["grpo_format", "prompt_chatml", "prompt_paper", "no_repetition_penalty"]
    deltas = sorted(float(r["delta_acc"]) for r in table)
    rouge_deltas = sorted(float(r["delta_rouge"]) for r in table)
    total = sum(int(r["acc_n"]) for r in table)
    open_n = sum(int(r["rouge_n"]) for r in table)

    volatile = load("analysis", "volatile_cases.json")
    failures = load("analysis", "failure_decomposition.json")["cases"]
    failure_numbers = {n.lstrip("-") for c in failures
                       for text in (c["question"], c["gold"], c["reasoning"]) for n in re.findall(r"-?\d+(?:\.\d+)?", text)}

    def p(name, digits=2):
        return f"{float(paired['patra__' + name]['mcnemar_p']):.{digits}f}"

    values = [
        # Limitation (ii) and its table: volatile series, failure cases quoted verbatim
        str(volatile["n_volatile"]), str(100 - volatile["percentile"]),
        str(volatile["trend_labeled_volatile"]["correct"]), str(volatile["trend_labeled_volatile"]["n"]),
        *failure_numbers,
        # Ablation: share of changed answers
        *[f"{float(paired['patra__' + n]['answer_changed_pct']):.2f}"
          for n in ("ts_zero", "reinit_modules", "no_align", "prompt_chatml", "prompt_paper")],
        # Table 1 of this report
        *[r["acc_pct"] for r in table], *[r["rouge_pct"] for r in table], *[r["acc_n"] for r in table],
        f"{total:,}", f"{open_n:,}", f"{total + open_n:,}",
        f"{labeled['patra_official']['acc_pct']:.2f}", f"{labeled['paper_total_correct_converted'] / total * 100:.2f}",
        f"{open_ended['patra']['overall_pct']:.2f}",
        f"{abs(deltas[-1]):.2f}", f"{abs(deltas[0]):.2f}", f"{rouge_deltas[0]:.2f}", f"{rouge_deltas[-1]:.2f}",
        # Setup checks
        str(ts["weights"]["in_checkpoint"]), f"{max(float(paired['patra__' + v]['acc_b']) for v in variants):.2f}",
        # Table 2 of this report
        *[f"{float(paired['patra__' + n]['acc_b']):.2f}" for n in ("ts_zero", "reinit_modules", "no_align", "no_causal_mask")],
        *[p(n) for n in ("ts_zero", "reinit_modules", "no_align", "no_causal_mask")],
        f"{labeled['constant_from_train_labels']['acc_pct']:.2f}", f"{labeled['constant_from_train_labels']['mcnemar_p_vs_patra']:.2f}",
        f"{lift['question_only_trees']['acc_pct']:.2f}", f"{lift['question_only_trees']['mcnemar_p_vs_patra']:.3f}",
        f"{lift['question_plus_series_stats_trees']['acc_pct']:.2f}", "0.00001",
        f"{lift['same_model_with_series_shuffled_at_test']['acc_pct']:.2f}",
        f"{lift['same_model_with_series_shuffled_at_test']['mcnemar_p_vs_patra']:.3f}",
        # Ablation text
        str(lift["question_plus_series_stats_trees"]["correct"] - lift["question_only_trees"]["correct"]),
        str(lift["question_plus_series_stats_trees"]["correct"] - lift["same_model_with_series_shuffled_at_test"]["correct"]),
        str(labeled["constant_from_train_labels"]["correct"]),
        f"{labeled['question_only_tfidf_logreg']['acc_pct']:.2f}",
        # Insight 1
        f"{align['max_gap_from_uniform']:.4f}",
        f"{mech['output_change_if_attention_is_replaced_by_plain_average']['mean_relative'] * 100:.3f}",
        f"{token['aligned']['constant_energy_share'] * 100:.2f}", f"{token['encoded']['constant_energy_share'] * 100:.2f}",
        f"{ts['overall']['encoded_rel_zero']['mean'] * 100:.2f}", f"{ts['overall']['aligned_rel_zero']['mean'] * 100:.2f}",
        # Insight 2
        str(decomp["eval"]["longest_series"]), str(decomp["eval"]["pct_series_by_patches"]["2"]),
        str(decomp["eval"]["pct_series_by_patches"]["1"]), str(decomp["kernel_size"]), str(decomp["kernel_in_raw_points"]),
        str(decomp["eval"]["most_common_length"][0]),
        # Insight 3
        str(grounding["train_sft_gold_all"]["pct_equal_raw"]), str(grounding["train_sft_gold_all"]["pct_equal_raw_of_other_series"]),
        str(grounding["eval_patra_open_ended"]["pct_equal_raw"]), str(grounding["eval_patra_open_ended"]["pct_equal_raw_of_other_series"]),
        str(grounding["eval_qwen_open_ended"]["pct_equal_raw"]),
        # Insight 4
        str(advantage["0.01"]), str(advantage["2.0"]), "0.01",
        # Insight 5
        str(open_ended["fixed_sentence_no_question_no_series"]["overall_pct"]), str(open_ended["gold_with_opposite_direction"]["rouge_pct"]),
        # Strengths and limitations
        f"{open_ended['qwen']['overall_pct']:.2f}", str(open_ended["nearest_train_question_answer_no_series"]["overall_pct"]),
        f"{open_ended['words_median']['patra']:.0f}", f"{open_ended['words_median']['eval_gold']:.0f}",
        str(reasoning["n"]), f"{reasoning['lowest_possible_mcnemar_p']:.2f}",
        f"{review['item5_offset_scale']['sft_series_equal_encoded_grpo_series']['n']:,}",
        # Reflection
        f"{open_ended['patra']['overall_pct'] - open_ended['nearest_train_question_answer_no_series']['overall_pct']:.2f}",
    ]
    return set(values)


def checks_that_must_hold():
    """Claims made in words."""
    probes = load("analysis", "benchmark_probes.json")
    fixed = probes["open_ended"]["fixed_sentence_no_question_no_series"]["by_task"]
    # Best ROUGE-L of any baseline in paper Table 1, per task.
    best_baseline = {"comprehension": 18.77, "recognition": 14.61, "reasoning": 18.72, "prescience": 15.53}
    assert all(fixed[t] > best_baseline[t] for t in best_baseline), "fixed sentence no longer beats every baseline"

    paired = [r for r in rows("analysis", "paired.csv") if r["a"] == "patra__official" and r["grouping"] == "overall"]
    for r in paired:
        if r["b"] in ("patra__ts_zero", "patra__reinit_modules", "patra__no_align", "patra__no_causal_mask"):
            assert float(r["acc_b"]) >= float(r["acc_a"]), f"{r['b']} lowered accuracy"
            assert float(r["mcnemar_p"]) > 0.05, f"{r['b']} is significant"
    readme = next(r for r in paired if r["b"] == "patra__env_readme")
    assert float(readme["answer_changed_pct"]) == 0, "answers changed under transformers 4.46.3"
    assert "transformers==4.46.3" in RESULTS.joinpath("env", "pip_freeze_readme.txt").read_text(encoding="utf8")

    ts = load("analysis", "ts_probe_summary.json")
    assert ts["weights"]["differs_from_loaded"] == 0, "loaded weights differ from the checkpoint"
    parity = load("env", "smoke.json")["parity"].values()
    assert all(t["match"] for t in parity), "runner no longer matches the official script"


def tex_numbers():
    body = TEX.read_text(encoding="utf8").split(r"\begin{document}")[1]
    body = re.sub(r"(?m)(?<!\\)%.*$", "", body)
    # Citation keys, labels, paths and code names are not numbers.
    body = re.sub(r"\\(?:cite[a-z]*|label|ref|includegraphics(?:\[[^\]]*\])?|bibliography|bibliographystyle|texttt)\{[^}]*\}", "", body)
    body = re.sub(r"\\vskip\s*-?[\d.]+in", "", body)
    # Column specs such as p{4.3cm} are layout, not numbers.
    body = re.sub(r"\\begin\{tabular\}\{[^\n]*\}", "", body)
    body = re.sub(r"Qwen2\.5-7B(?:-Instruct)?|PATRA-7B|TimeOmni-1|Time-R1|DeepSeek-R1", "", body)
    # A leading + or - is a sign; 10.19--25.86 is a range.
    return re.findall(r"\d[\d,]*\.?\d*(?<![.,])", body)


def main():
    checks_that_must_hold()
    allowed = results_numbers() | PAPER | CONTEXT
    found = tex_numbers()
    unknown = sorted({n for n in found if n not in allowed})
    if unknown:
        print("Not traced to results/ or the paper:", unknown)
        sys.exit(1)
    print(f"OK: {len(found)} numbers in main.tex traced to results/, the paper or CONTEXT.")


if __name__ == "__main__":
    main()
