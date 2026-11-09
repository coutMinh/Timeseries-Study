"""What do the PATRA eval metrics reward? Baselines that never read the series. CPU only.

Writes results/analysis/benchmark_probes.json. Every model and hyperparameter is chosen on the
public training data; the eval set is only used for the final score.

    python scripts/benchmark_probes.py --cache-dir <folder holding grpo.jsonl>
Needs scikit-learn and rouge-score.
"""
import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from rouge_score import rouge_scorer
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("comprehension", "recognition", "reasoning", "prescience")
PAPER_ACC_TOTAL_CORRECT = 657  # README: Table 1 percentages converted to questions
PAPER_ROUGE = {"comprehension": 25.67, "recognition": 25.46, "reasoning": 27.36, "prescience": 27.06}
SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)  # same call as PATRA/eval/patra_eval.py:73


def load_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def rouge(reference, text):
    return SCORER.score(reference, text)["rougeL"].fmeasure  # same call as patra_eval.py:68


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def clean_question(text):
    return re.sub(r"\s+", " ", text.replace("<ts><ts/>", " TS ")).strip().lower()


def gold_key(row):
    return str(row["answer"]).strip().upper()[:1]


def summarize_labeled(eval_rows, hits, patra_scores):
    """hits and patra_scores are aligned with eval_rows."""
    by_task = {}
    for task in TASKS:
        index = [i for i, r in enumerate(eval_rows) if r["task"] == task]
        by_task[task] = {"n": len(index), "correct": int(sum(hits[i] for i in index)),
                         "acc_pct": round(100 * sum(hits[i] for i in index) / len(index), 2)}
    b = sum(1 for h, p in zip(hits, patra_scores) if h and not p)
    c = sum(1 for h, p in zip(hits, patra_scores) if p and not h)
    return {"correct": int(sum(hits)), "acc_pct": round(100 * sum(hits) / len(hits), 2), "by_task": by_task,
            "only_this_correct": b, "only_patra_correct": c, "mcnemar_p_vs_patra": round(mcnemar_exact(b, c), 5)}


def fit_question_model(train_rows):
    """TF-IDF of the question text only. C is chosen by 5-fold CV on training rows."""
    questions = [clean_question(r["input"]) for r in train_rows]
    labels = [gold_key(r) for r in train_rows]
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=200000)
    features = hstack([word.fit_transform(questions), char.fit_transform(questions)]).tocsr()
    search = GridSearchCV(LogisticRegression(max_iter=2000), {"C": [0.3, 1, 3, 10]}, cv=5, n_jobs=-1)
    search.fit(features, labels)
    return search.best_estimator_, word, char, {"best_C": search.best_params_["C"], "cv_accuracy_pct": round(100 * search.best_score_, 2), "n_train": len(train_rows)}


def series_features(row):
    """Plain statistics of the first series, in raw units, plus how many series there are."""
    x = np.asarray(row["timeseries"][0], dtype=float)
    n = len(x)
    t = np.arange(n)
    slope = float(np.polyfit(t, x, 1)[0]) if n > 1 else 0.0
    std = float(x.std())
    steps = np.diff(x) if n > 1 else np.zeros(1)
    lag1 = float(np.corrcoef(x[:-1], x[1:])[0, 1]) if n > 2 and x[:-1].std() > 0 and x[1:].std() > 0 else 0.0
    return [len(row["timeseries"]), n, x.mean(), std, x.min(), x.max(), x[0], x[-1], slope, slope * n / (std + 1e-9),
            (x[-1] - x[0]) / (std + 1e-9), float((steps > 0).mean()), lag1, x.argmax() / n, x.argmin() / n,
            float(np.abs(x - x.mean()).max() / (std + 1e-9))]


def series_lift(train_labeled, eval_rows, patra_scores):
    """Same tree model with and without series statistics. Settings are fixed before looking at eval."""
    report, question_hits, real_hits, shuffled_hits = {}, [0] * len(eval_rows), [0] * len(eval_rows), [0] * len(eval_rows)
    rng = np.random.default_rng(0)
    for fmt in ("true/false", "multiple_choice"):
        train = [r for r in train_labeled if r["question_format"] == fmt and r["timeseries"] and r["timeseries"][0]]
        index = [i for i, r in enumerate(eval_rows) if r["question_format"] == fmt and r["timeseries"] and r["timeseries"][0]]
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        svd = TruncatedSVD(n_components=150, random_state=0)
        q_train = svd.fit_transform(vectorizer.fit_transform(clean_question(r["input"]) for r in train))
        q_eval = svd.transform(vectorizer.transform(clean_question(eval_rows[i]["input"]) for i in index))
        s_train = np.array([series_features(r) for r in train])
        s_eval = np.array([series_features(eval_rows[i]) for i in index])
        labels = [gold_key(r) for r in train]
        params = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=0)
        only_question = HistGradientBoostingClassifier(**params).fit(q_train, labels).predict(q_eval)
        with_series_model = HistGradientBoostingClassifier(**params).fit(np.hstack([q_train, s_train]), labels)
        real = with_series_model.predict(np.hstack([q_eval, s_eval]))
        shuffled = with_series_model.predict(np.hstack([q_eval, s_eval[rng.permutation(len(s_eval))]]))
        for position, i in enumerate(index):
            gold = gold_key(eval_rows[i])
            question_hits[i], real_hits[i], shuffled_hits[i] = int(only_question[position] == gold), int(real[position] == gold), int(shuffled[position] == gold)
    b = sum(1 for r, q in zip(real_hits, question_hits) if r and not q)
    c = sum(1 for r, q in zip(real_hits, question_hits) if q and not r)
    report["question_only_trees"] = summarize_labeled(eval_rows, question_hits, patra_scores)
    report["question_plus_series_stats_trees"] = summarize_labeled(eval_rows, real_hits, patra_scores)
    report["same_model_with_series_shuffled_at_test"] = summarize_labeled(eval_rows, shuffled_hits, patra_scores)
    report["series_stats_vs_question_only"] = {"only_with_series_correct": b, "only_question_correct": c, "mcnemar_p": round(mcnemar_exact(b, c), 5)}
    return report


def labeled_probes(grpo, patra_rows, qwen_rows):
    eval_rows = [r for r in patra_rows if r["metric"] == "acc"]
    patra_scores = [int(r["score"]) for r in eval_rows]
    qwen_by_id = {r["sample_id"]: int(r["score"]) for r in qwen_rows if r["metric"] == "acc"}
    train_labeled = [r for r in grpo if r["question_format"] != "open_ended_question"]
    out = {"patra_official": summarize_labeled(eval_rows, patra_scores, patra_scores),
           "qwen_official": summarize_labeled(eval_rows, [qwen_by_id[r["sample_id"]] for r in eval_rows], patra_scores),
           "paper_total_correct_converted": PAPER_ACC_TOTAL_CORRECT}

    constants = {}
    for fmt in ("true/false", "multiple_choice"):
        constants[fmt] = Counter(gold_key(r) for r in train_labeled if r["question_format"] == fmt).most_common(1)[0][0]
    out["constant_from_train_labels"] = {"labels": constants, **summarize_labeled(
        eval_rows, [int(gold_key(r) == constants[r["question_format"]]) for r in eval_rows], patra_scores)}

    hits, meta, cues = [0] * len(eval_rows), {}, {}
    for fmt in ("true/false", "multiple_choice"):
        model, word, char, info = fit_question_model([r for r in train_labeled if r["question_format"] == fmt])
        index = [i for i, r in enumerate(eval_rows) if r["question_format"] == fmt]
        questions = [clean_question(eval_rows[i]["input"]) for i in index]
        predicted = model.predict(hstack([word.transform(questions), char.transform(questions)]).tocsr())
        for i, label in zip(index, predicted):
            hits[i] = int(gold_key(eval_rows[i]) == label)
        info["eval_correct"], info["eval_n"] = sum(hits[i] for i in index), len(index)
        info["patra_eval_correct"] = sum(patra_scores[i] for i in index)
        meta[fmt] = info
        if fmt == "true/false":
            words = np.array(word.get_feature_names_out())
            positive = list(model.classes_).index("T")
            weights = model.coef_[0][: len(words)] * (1 if positive == 1 else -1)
            order = np.argsort(weights)
            cues = {"towards_true": words[order[::-1][:15]].tolist(), "towards_false": words[order[:15]].tolist()}
    seen = {re.sub(r"\s+", " ", r["input"].strip().lower()) for r in grpo}
    unseen = [i for i, r in enumerate(eval_rows) if re.sub(r"\s+", " ", r["input"].strip().lower()) not in seen]
    out["question_only_tfidf_logreg"] = {
        "model_selection": meta, "true_false_word_cues": cues, **summarize_labeled(eval_rows, hits, patra_scores),
        "questions_never_seen_in_train": {"n": len(unseen), "this_model_correct": sum(hits[i] for i in unseen),
                                          "patra_correct": sum(patra_scores[i] for i in unseen)}}
    out["series_lift"] = series_lift(train_labeled, eval_rows, patra_scores)
    return out


def nearest_train_answers(train_open, eval_open):
    """The training answer of the most similar training question. The series is never read."""
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    train_matrix = vectorizer.fit_transform(clean_question(r["input"]) for r in train_open)
    eval_matrix = vectorizer.transform(clean_question(r["input"]) for r in eval_open)
    nearest = np.asarray((eval_matrix @ train_matrix.T).argmax(axis=1)).ravel()
    return [train_open[i]["answer"] for i in nearest]


def medoid_answer(train_open, n_refs=300, n_candidates=300, seed=0):
    """One fixed sentence: the training answer that scores best against other training answers."""
    rng = np.random.default_rng(seed)
    refs = rng.choice(len(train_open), n_refs, replace=False)
    candidates = rng.choice(len(train_open), n_candidates, replace=False)
    scored = [(np.mean([rouge(train_open[r]["answer"], train_open[c]["answer"]) for r in refs]), c) for c in candidates]
    return train_open[max(scored)[1]]["answer"]


def first_words(text, n):
    return " ".join(text.split()[:n])


def mean_pct(values):
    return round(100 * float(np.mean(values)), 2)


def by_task_rouge(rows, texts):
    groups = defaultdict(list)
    for row, text in zip(rows, texts):
        groups[row["task"]].append(rouge(row["answer"], text))
    return {t: mean_pct(groups[t]) for t in TASKS if t in groups}


def open_probes(grpo, patra_rows, qwen_rows):
    train_open = [r for r in grpo if r["question_format"] == "open_ended_question"]
    patra = [r for r in patra_rows if r["metric"] != "acc"]
    qwen_by_id = {r["sample_id"]: r for r in qwen_rows}
    qwen = [qwen_by_id[r["sample_id"]] for r in patra]

    recomputed = np.mean([rouge(r["answer"], r["response"]) for r in patra])
    stored = np.mean([float(r["score"]) for r in patra])
    assert abs(recomputed - stored) < 1e-6, "rouge-score does not reproduce the stored scores"

    # Length is fixed from training answers, not from eval answers.
    train_median_words = int(np.median([len(r["answer"].split()) for r in train_open]))
    medoid = medoid_answer(train_open)
    nearest = nearest_train_answers(train_open, patra)
    variants = {
        "patra": [r["response"] for r in patra],
        "qwen": [r["response"] for r in qwen],
        "qwen_first_train_median_words": [first_words(r["response"], train_median_words) for r in qwen],
        "patra_first_train_median_words": [first_words(r["response"], train_median_words) for r in patra],
        "fixed_sentence_no_question_no_series": [medoid] * len(patra),
        "nearest_train_question_answer_no_series": nearest,
    }
    report = {"paper_patra_by_task": PAPER_ROUGE, "train_median_answer_words": train_median_words, "fixed_sentence": medoid,
              "words_median": {"eval_gold": float(np.median([len(r["answer"].split()) for r in patra])),
                               "patra": float(np.median([len(r["response"].split()) for r in patra])),
                               "qwen": float(np.median([len(r["response"].split()) for r in qwen]))}}
    for name, texts in variants.items():
        report[name] = {"overall_pct": mean_pct([rouge(r["answer"], t) for r, t in zip(patra, texts)]), "by_task": by_task_rouge(patra, texts)}

    # Does ROUGE-L notice a wrong trend direction? Flip only the direction words of the gold answer.
    swap = {"upward": "downward", "downward": "upward", "increasing": "decreasing", "decreasing": "increasing",
            "increase": "decrease", "decrease": "increase", "rising": "falling", "falling": "rising"}
    pattern = re.compile(r"\b(" + "|".join(swap) + r")\b", re.IGNORECASE)
    flips = [(r["answer"], pattern.sub(lambda m: swap[m.group(1).lower()], r["answer"])) for r in patra if pattern.search(r["answer"])]
    report["gold_with_opposite_direction"] = {"n": len(flips), "rouge_pct": mean_pct([rouge(a, b) for a, b in flips])}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    grpo = load_jsonl(args.cache_dir / "grpo.jsonl")
    patra = load_jsonl(ROOT / "results/runs/patra__official/predictions.jsonl")
    qwen = load_jsonl(ROOT / "results/runs/qwen__official/predictions.jsonl")
    report = {"labeled": labeled_probes(grpo, patra, qwen), "open_ended": open_probes(grpo, patra, qwen)}
    out = ROOT / "results/analysis/benchmark_probes.json"
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
