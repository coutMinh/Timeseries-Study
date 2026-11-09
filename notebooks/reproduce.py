# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "huggingface-hub==0.36.2",
#     "rouge-score==0.1.2",
#     "transformers==4.52.4",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def imports():
    import csv
    import functools
    import importlib.util
    import io
    import json
    import math
    import os
    import re
    import subprocess
    import time
    import types
    import zipfile
    from pathlib import Path

    os.environ.setdefault("USE_TF", "0")  # transformers would otherwise import TensorFlow from the base image

    import marimo as mo
    import torch
    from huggingface_hub import snapshot_download
    from rouge_score import rouge_scorer
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    return (
        AutoModelForCausalLM,
        AutoProcessor,
        AutoTokenizer,
        Path,
        csv,
        functools,
        importlib,
        io,
        json,
        math,
        mo,
        re,
        rouge_scorer,
        snapshot_download,
        subprocess,
        time,
        torch,
        types,
        zipfile,
    )


@app.cell(hide_code=True)
def intro(mo):
    mo.md(r"""
    # PATRA: reproduce Table 1 and the series interventions

    Inference only, with the released checkpoint, data and evaluation script of
    [PATRA](https://arxiv.org/abs/2602.23161), all pinned to fixed revisions. Nothing is retrained.

    1. The first cells download the upstream code and the evaluation data.
    2. Pick a run and press **Run**. The model (about 16 GB) downloads on first use.
       Finished questions are skipped, so a stopped run resumes where it left off.
    3. The table at the bottom compares every finished run with the paper.
       Download the zip and unpack it into `results/` of the repository.

    On one RTX PRO 6000, an official run (2,965 questions) takes about 3 hours and an
    intervention run (1,097 labeled questions) about 1 hour.
    """)
    return


@app.cell
def pins(Path):
    WORK = Path.home() / "patra_work"  # code, data, weights and results
    RESULTS = WORK / "results"
    UPSTREAM = ("https://github.com/decisionintelligence/PATRA.git", "ed19f182f9ad2ec8241d5cfbbd2e9539264b6daf")
    EVAL_DATA = ("DecisionIntelligence/PATRA-EVAL", "c79132c5fa44925f90c701edde51f57d58e14f76")
    MODELS = {
        "patra": ("DecisionIntelligence/PATRA-7B", "5cf8191e8a71eab7f936a3dd3e2f62bc78debed1"),
        "qwen": ("Qwen/Qwen2.5-7B", "d149729398750b98c0af14eb82c78cfe92750796"),
    }
    TASKS = ("comprehension", "recognition", "reasoning", "prescience")
    LABELED = ("multiple_choice", "true/false")
    MAX_NEW_TOKENS = 512
    SEED = 42
    # Table 1 of arXiv:2602.23161v4 as (accuracy, ROUGE-L), for comparison only.
    PAPER = {
        "patra": {"comprehension": (56.03, 25.67), "recognition": (64.69, 25.46),
                  "reasoning": (44.59, 27.36), "prescience": (52.78, 27.06)},
        "qwen": {"comprehension": (42.24, 18.77), "recognition": (45.51, 10.32),
                 "reasoning": (36.48, 18.72), "prescience": (26.85, 10.67)},
    }
    return (
        EVAL_DATA,
        LABELED,
        MAX_NEW_TOKENS,
        MODELS,
        PAPER,
        RESULTS,
        SEED,
        TASKS,
        UPSTREAM,
        WORK,
    )


@app.cell
def data(
    EVAL_DATA,
    LABELED,
    Path,
    TASKS,
    UPSTREAM,
    WORK,
    importlib,
    json,
    mo,
    snapshot_download,
    subprocess,
):
    _repo = WORK / "PATRA"
    if not (_repo / ".git").exists():
        subprocess.run(["git", "clone", "-q", UPSTREAM[0], str(_repo)], check=True)
    subprocess.run(["git", "-C", str(_repo), "checkout", "-q", "--detach", UPSTREAM[1]], check=True)

    # The unmodified upstream evaluation script supplies the prompt (build_prompt) and the answer rule (BOXED_RE).
    _spec = importlib.util.spec_from_file_location("patra_eval", _repo / "eval" / "patra_eval.py")
    upstream = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(upstream)

    _data = Path(snapshot_download(EVAL_DATA[0], revision=EVAL_DATA[1], repo_type="dataset",
                                   local_dir=str(WORK / "data"), allow_patterns=["*.jsonl"]))
    samples = []
    for _task in TASKS:
        _lines = (_data / f"{_task}.jsonl").read_text(encoding="utf-8").splitlines()
        _rows = [json.loads(_line) for _line in _lines if _line.strip()]
        samples += [{**_row, "task": _task, "row": _i, "sample_id": f"{_task}:{_i}"} for _i, _row in enumerate(_rows)]
    _n_labeled = sum(_s["question_format"] in LABELED for _s in samples)
    mo.md(f"Loaded {len(samples):,} questions: {_n_labeled:,} labeled and {len(samples) - _n_labeled:,} open-ended.")
    return samples, upstream


@app.cell(hide_code=True)
def runs_intro(mo):
    mo.md(r"""
    ## Runs

    The two official runs answer every question. The other runs change one thing in PATRA and answer
    the 1,097 labeled questions only, so their accuracy can be compared question by question with
    `patra__official`.
    """)
    return


@app.cell
def runs(LABELED, json, re, upstream):
    RUNS = {
        "patra__official": "PATRA-7B with the official evaluation script",
        "qwen__official": "Qwen2.5-7B, each series written into the prompt as a list of values",
        "patra__ts_zero": "every series replaced by zeros",
        "patra__reinit_modules": "series encoder and alignment module replaced by random weights",
        "patra__no_align": "alignment module bypassed",
        "patra__no_causal_mask": "attention mask removed inside the alignment module",
        "patra__prompt_chatml": "official prompt with the newlines of the checkpoint's chat template",
        "patra__prompt_paper": "system prompt from Appendix B of the paper",
        "patra__grpo_format": "prompt and answer tags used in GRPO training",
        "patra__no_repetition_penalty": "repetition penalty turned off",
    }
    PLACEHOLDER = "<ts><ts/>"
    TEXT_EOS = [151645, 151643]  # Qwen2.5 end-of-turn and end-of-text
    # Appendix B of arXiv:2602.23161v4.
    PAPER_SYSTEM = (
        "You are an assistant for evaluating time series questions.\n\n"
        "Task: Answer the question concisely in the required format.\n"
        "Tips:\n"
        "1. Your thinking steps should be in <think> tags.\n"
        "2. Always output exactly one answer, using the following formats:\n"
        "- For True/False questions: \\boxed{True} or \\boxed{False}\n"
        "- For Multiple Choice questions (options A, B, C, D): \\boxed{A}, \\boxed{B}, \\boxed{C}, or \\boxed{D}\n"
    )
    # Copied from train/src/llamafactory/train/grpo/workflow.py at the pinned commit.
    _head = ("\nYou must first reason step-by-step about the current situation. This reasoning process MUST be "
             "enclosed within <think>  \nand </think> tags. Once you've finished your reasoning, you must ")
    _tail = (" and then present it within \n<answer> and </answer> tags. Finally, name the task and present it "
             "within <task_type> and </task_type> tags.\n")
    GRPO_SUFFIX = {
        "multiple_choice": _head + "choose ONE from A, B, C and D" + _tail,
        "true/false": _head + "choose ONE from True and False" + _tail,
        "open_ended_question": _head + "give the final answer then" + _tail.replace(" and then", "", 1),
    }


    def chatml(system, user):
        """The format of chat_template.jinja in the checkpoint, including its newlines."""
        return (f"<|im_start|>system\n{system}<|im_end|>\n"
                f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")


    def prompt_for(sample, condition):
        labeled = sample["question_format"] in LABELED
        if condition == "prompt_chatml":
            return chatml(upstream.SYS if labeled else "You are a helpful assistant.", sample["input"])
        if condition == "prompt_paper":
            return chatml(PAPER_SYSTEM if labeled else "You are a helpful assistant.", sample["input"])
        if condition == "grpo_format":
            return chatml("You are a helpful assistant.", sample["input"] + GRPO_SUFFIX[sample["question_format"]])
        return upstream.build_prompt(sample)


    def series_as_text(sample):
        """Qwen has no series encoder, so each series replaces its placeholder as a list of values."""
        parts = sample["input"].split(PLACEHOLDER)
        text = parts[0]
        for series, tail in zip(sample["timeseries"], parts[1:]):
            text += json.dumps(series) + tail
        return {**sample, "input": text}


    def score(sample, response, condition, rouge):
        """Accuracy as in upstream evaluate_sample (first \\boxed{...}); ROUGE-L F1 for open-ended questions."""
        fmt = sample["question_format"]
        if fmt not in LABELED:
            return "rougeL", rouge.score(sample["answer"], response)["rougeL"].fmeasure, None
        expected = sample["answer"] if fmt == "multiple_choice" else ("True" if sample["answer"] == "T" else "False")
        if condition == "grpo_format":  # reward rule of GRPO training: the last <answer> tag decides
            found = re.findall(r"<answer>(.*?)</answer>", response, re.DOTALL)
            answer = found[-1].strip() if found else None
        else:
            match = upstream.BOXED_RE.search(response)
            answer = match.group(1) if match else None
        return "acc", int(answer is not None and answer.lower() == expected.lower()), answer

    return RUNS, TEXT_EOS, prompt_for, score, series_as_text


@app.cell
def model(
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    MAX_NEW_TOKENS,
    MODELS,
    SEED,
    TEXT_EOS,
    WORK,
    functools,
    prompt_for,
    series_as_text,
    snapshot_download,
    torch,
    types,
    upstream,
):
    def bypass_align(self, input_ids, text_embeddings, ts_features, patch_counts, text_attention_mask=None):
        return ts_features, patch_counts


    def load_model(system, condition):
        """Load one model at its pinned revision and apply the condition of the run."""
        path = snapshot_download(MODELS[system][0], revision=MODELS[system][1], local_dir=str(WORK / "models" / system))
        remote = system == "patra"
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=remote, torch_dtype=torch.float16)
        model = model.to("cuda").eval()
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=remote)
        tokenizer.pad_token = tokenizer.eos_token
        processor = AutoProcessor.from_pretrained(path, trust_remote_code=True, tokenizer=tokenizer) if remote else None
        if condition == "no_align":
            model.align.forward = types.MethodType(bypass_align, model.align)
        if condition == "reinit_modules":
            # Fresh N(0, 0.02) weights and zero biases, as _init_weights in the checkpoint's modeling file.
            _source = torch.Generator().manual_seed(SEED)
            for _module in (model.ts_encoder, model.align):
                for _parameter in _module.parameters():
                    _fresh = torch.empty(_parameter.shape, dtype=torch.float32)
                    _fresh = _fresh.normal_(0.0, 0.02, generator=_source) if _parameter.ndim > 1 else _fresh.zero_()
                    _parameter.data = _fresh.to(_parameter.dtype).to(_parameter.device)
        if condition == "no_causal_mask":
            # is_causal=True is the only upstream branch that skips the mask (modeling_qwen2.py, lines 307-313).
            for _branch in ("complete", "trend", "seasonal"):
                for _kind in ("text_attn", "cross_attn"):
                    _attn = getattr(model.align, f"{_branch}_{_kind}")
                    _attn.forward = functools.partial(_attn.forward, is_causal=True)
        return model, tokenizer, processor


    @torch.no_grad()
    def answer(model, tokenizer, processor, sample, condition):
        """Greedy decoding with the settings of the official evaluation script."""
        if processor is not None:
            used = {**sample, "timeseries": [[0.0] * len(s) for s in sample["timeseries"]]} if condition == "ts_zero" else sample
            inputs = processor(text=[prompt_for(used, condition)], timeseries=used["timeseries"], padding=True, return_tensors="pt")
            extra = {"repetition_penalty": 1.0} if condition == "no_repetition_penalty" else {}
            eos = model.generation_config.eos_token_id
            eos_ids = {eos} if isinstance(eos, int) else set(eos)
        else:
            used = sample
            inputs = tokenizer(upstream.build_prompt(series_as_text(sample)), return_tensors="pt")
            extra = {"eos_token_id": TEXT_EOS, "repetition_penalty": 1.05}
            eos_ids = set(TEXT_EOS)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id, **extra)
        generated = output[0][int(inputs["attention_mask"][0].sum()):]
        truncated = len(generated) >= MAX_NEW_TOKENS and int(generated[-1]) not in eos_ids
        return used, tokenizer.decode(generated, skip_special_tokens=True), len(generated), truncated

    return answer, load_model


@app.cell
def controls(RUNS, mo):
    run_picker = mo.ui.dropdown({f"{k}: {v}": k for k, v in RUNS.items()},
                                value=f"patra__official: {RUNS['patra__official']}", label="Run")
    limit = mo.ui.number(start=0, stop=3000, value=0, label="Questions to answer now (0 = all remaining)")
    start = mo.ui.run_button(label="Run")
    mo.hstack([run_picker, limit, start], justify="start", align="end")
    return limit, run_picker, start


@app.cell
def run(
    LABELED,
    RESULTS,
    answer,
    json,
    limit,
    load_model,
    mo,
    rouge_scorer,
    run_picker,
    samples,
    score,
    start,
    time,
    torch,
):
    mo.stop(not start.value, mo.md("Pick a run and press **Run**."))

    _run = run_picker.value
    _system, _condition = _run.split("__")
    _chosen = samples if _condition == "official" else [s for s in samples if s["question_format"] in LABELED]
    _dir = RESULTS / "runs" / _run
    _dir.mkdir(parents=True, exist_ok=True)
    _file = _dir / "predictions.jsonl"
    _done = {json.loads(l)["sample_id"] for l in _file.read_text(encoding="utf-8").splitlines() if l.strip()} if _file.exists() else set()
    _pending = [s for s in _chosen if s["sample_id"] not in _done]
    if limit.value:
        _pending = _pending[: limit.value]

    if _pending:
        _model, _tokenizer, _processor = load_model(_system, _condition)
        _rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        with open(_file, "a", encoding="utf-8") as _out:
            for _sample in mo.status.progress_bar(_pending, title=_run, show_eta=True, show_rate=True):
                _began = time.time()
                _used, _response, _n_gen, _truncated = answer(_model, _tokenizer, _processor, _sample, _condition)
                _metric, _score, _parsed = score(_sample, _response, _condition, _rouge)
                _record = {
                    "sample_id": _sample["sample_id"], "task": _sample["task"], "row": _sample["row"],
                    "question_format": _sample["question_format"], "input": _sample["input"], "used_input": _used["input"],
                    "answer": _sample["answer"], "timeseries": _used["timeseries"], "response": _response,
                    "metric": _metric, "score": _score, "pred_official": _parsed, "n_gen": _n_gen,
                    "truncated": _truncated, "seconds": round(time.time() - _began, 3),
                    "system": _system, "condition": _condition,
                }
                _out.write(json.dumps(_record, ensure_ascii=False) + "\n")
                _out.flush()
        del _model
        torch.cuda.empty_cache()

    # Keep the file in dataset order, so runs can be compared line by line.
    _order = {s["sample_id"]: i for i, s in enumerate(samples)}
    _rows = [json.loads(l) for l in _file.read_text(encoding="utf-8").splitlines() if l.strip()]
    _rows.sort(key=lambda r: _order[r["sample_id"]])
    _file.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in _rows), encoding="utf-8")
    finished_run = (_run, len(_rows), len(_chosen))
    mo.md(f"`{_run}`: {len(_rows):,} of {len(_chosen):,} questions answered.")
    return (finished_run,)


@app.cell
def table(
    LABELED,
    PAPER,
    RESULTS,
    TASKS,
    csv,
    finished_run,
    json,
    math,
    mo,
    samples,
):
    finished_run  # refresh after every run


    def _read(run_dir):
        path = run_dir / "predictions.jsonl"
        return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()] if path.exists() else []


    def _mcnemar(b, c):
        """Exact two-sided McNemar test on the discordant pairs."""
        n = b + c
        return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


    def _write_csv(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


    _expected = {t: {"acc": sum(s["task"] == t and s["question_format"] in LABELED for s in samples),
                     "rougeL": sum(s["task"] == t and s["question_format"] not in LABELED for s in samples)} for t in TASKS}
    _runs = {d.name: _read(d) for d in sorted((RESULTS / "runs").glob("*__*"))}
    _runs = {k: v for k, v in _runs.items() if v}
    _table = []
    for _name, _rows in _runs.items():
        _system, _condition = _name.split("__")
        for _task in TASKS:
            _acc = [r["score"] for r in _rows if r["task"] == _task and r["metric"] == "acc"]
            _rl = [r["score"] for r in _rows if r["task"] == _task and r["metric"] == "rougeL"]
            _paper = PAPER[_system][_task]
            _acc_pct = round(100 * sum(_acc) / len(_acc), 2) if _acc else None
            _rl_pct = round(100 * sum(_rl) / len(_rl), 2) if _rl else None
            _table.append({
                "system": _system, "condition": _condition, "task": _task,
                "acc_correct": sum(_acc), "acc_n": len(_acc), "acc_expected_n": _expected[_task]["acc"], "acc_pct": _acc_pct,
                "rouge_n": len(_rl), "rouge_expected_n": _expected[_task]["rougeL"], "rouge_pct": _rl_pct,
                "paper_acc": _paper[0], "paper_rouge": _paper[1],
                "delta_acc": round(_acc_pct - _paper[0], 2) if _condition == "official" and _acc_pct is not None else "",
                "delta_rouge": round(_rl_pct - _paper[1], 2) if _condition == "official" and _rl_pct is not None else "",
            })

    # Each run against patra__official on the labeled questions both runs answered.
    _paired = []
    _base = {r["sample_id"]: r for r in _runs.get("patra__official", []) if r["metric"] == "acc"}
    for _name, _rows in _runs.items():
        if _name == "patra__official" or not _base:
            continue
        _pairs = [(_base[r["sample_id"]], r) for r in _rows if r["sample_id"] in _base]
        for _grouping, _group, _members in [("overall", "all", _pairs)] + [
                ("task", t, [p for p in _pairs if p[0]["task"] == t]) for t in sorted(TASKS)]:
            if not _members:
                continue
            _a_only = sum(x["score"] and not y["score"] for x, y in _members)
            _b_only = sum(y["score"] and not x["score"] for x, y in _members)
            _both = sum(x["score"] and y["score"] for x, y in _members)
            _paired.append({
                "a": "patra__official", "b": _name, "grouping": _grouping, "group": _group, "n_labeled": len(_members),
                "both_correct": _both, "a_only": _a_only, "b_only": _b_only,
                "both_wrong": len(_members) - _both - _a_only - _b_only,
                "acc_a": round(100 * sum(x["score"] for x, _ in _members) / len(_members), 2),
                "acc_b": round(100 * sum(y["score"] for _, y in _members) / len(_members), 2),
                "mcnemar_p": round(_mcnemar(_a_only, _b_only), 5),
                "answer_changed_pct": round(100 * sum(x["pred_official"] != y["pred_official"] for x, y in _members) / len(_members), 2),
            })

    if _table:
        _write_csv(RESULTS / "tables" / "table1_repro.csv", _table)
    if _paired:
        _write_csv(RESULTS / "analysis" / "paired.csv", _paired)
    mo.vstack([
        mo.md("## Results (paper in parentheses)"),
        mo.ui.table([{
            "run": f"{r['system']}__{r['condition']}", "task": r["task"],
            "accuracy": "" if r["acc_pct"] is None else f"{r['acc_pct']:.2f} ({r['paper_acc']:.2f})",
            "ROUGE-L": "" if r["rouge_pct"] is None else f"{r['rouge_pct']:.2f} ({r['paper_rouge']:.2f})",
            "answered": f"{r['acc_n'] + r['rouge_n']:,}",
        } for r in _table], selection=None, pagination=False) if _table else mo.md("No finished run yet."),
        mo.md("Against `patra__official` on the same labeled questions:"),
        mo.ui.table([{k: r[k] for k in ("b", "n_labeled", "acc_a", "acc_b", "mcnemar_p", "answer_changed_pct")}
                     for r in _paired if r["grouping"] == "overall"], selection=None, pagination=False) if _paired else mo.md(""),
    ])
    return


@app.cell
def download(Path, RESULTS, finished_run, io, mo, zipfile):
    finished_run


    def _zip_results():
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(RESULTS.rglob("*")):
                if path.is_file():
                    archive.write(path, Path("results") / path.relative_to(RESULTS))
        return buffer.getvalue()


    mo.download(data=_zip_results, filename="patra_results.zip", label="Download results.zip")
    return


if __name__ == "__main__":
    app.run()
