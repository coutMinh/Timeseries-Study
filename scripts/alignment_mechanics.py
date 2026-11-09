"""Recompute PATRA's ts_encoder and alignment on CPU with the published weights.

Downloads only the 61 align/ts_encoder tensors and 10 embedding rows by HTTP range, not the full model.
The forward pass is reimplemented from modeling_qwen2.py:130-160, :172-192, :239-331 and :391-495,
and checked against results/analysis/ts_probe.csv from the GPU run.
Writes results/analysis/alignment_mechanics.json.

    python scripts/alignment_mechanics.py --cache-dir <folder outside the repo>
Needs torch and tokenizers.
"""
import argparse
import csv
import json
import struct
import urllib.request
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
REVISION = "5cf8191e"
BASE = f"https://huggingface.co/DecisionIntelligence/PATRA-7B/resolve/{REVISION}/"
BRANCHES = ("complete", "trend", "seasonal")
NUM_LATS, NUM_HEADS, KERNEL, PATCH = 10, 8, 25, 16
DTYPES = {"F16": (np.float16, 2), "BF16": (None, 2), "F32": (np.float32, 4)}
# PATRA/eval/patra_eval.py:11-24 and :35. Only the first 10 tokens reach the LATs.
SYSTEM = "\nYou are an assistant for evaluating time series questions.\n"


def http_range(url, start, end):
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=600) as response:
        return response.read()


def fetch_text(cache, name):
    path = cache / name
    if not path.exists():
        with urllib.request.urlopen(BASE + name, timeout=300) as response:
            path.write_bytes(response.read())
    return path


def shard_header(url):
    size = struct.unpack("<Q", http_range(url, 0, 7))[0]
    return json.loads(http_range(url, 8, 7 + size)), 8 + size


def decode(raw, dtype, shape):
    numpy_type = DTYPES[dtype][0]
    if numpy_type is None:
        raise ValueError("bf16 tensors are not expected in this checkpoint")
    return torch.from_numpy(np.frombuffer(raw, dtype=numpy_type).astype(np.float32).copy()).reshape(shape)


def load_weights(cache):
    index = json.loads(fetch_text(cache, "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    names = sorted(n for n in index if n.startswith(("align.", "ts_encoder.")))
    target = cache / "align_tensors.pt"
    if target.exists():
        weights = torch.load(target, weights_only=True)
    else:
        shard = index[names[0]]
        assert all(index[n] == shard for n in names)
        header, data_start = shard_header(BASE + shard)
        weights = {}
        for name in names:
            info = header[name]
            begin, stop = info["data_offsets"]
            weights[name] = decode(http_range(BASE + shard, data_start + begin, data_start + stop - 1), info["dtype"], info["shape"])
        torch.save(weights, target)
    return weights


def load_embedding_rows(cache, token_ids):
    index = json.loads(fetch_text(cache, "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    shard = index["model.embed_tokens.weight"]
    header, data_start = shard_header(BASE + shard)
    info = header["model.embed_tokens.weight"]
    width = info["shape"][1] * DTYPES[info["dtype"]][1]
    base = data_start + info["data_offsets"][0]
    rows = [decode(http_range(BASE + shard, base + i * width, base + (i + 1) * width - 1), info["dtype"], [1, info["shape"][1]]) for i in token_ids]
    return torch.cat(rows)


def sp_encoding(series):
    """processing_qwen2_ts.py:36-41; .half() at :152 is applied by rounding to float16."""
    x = np.asarray(series, dtype=np.float64)
    scaled = x - x.mean()
    if np.any(np.abs(scaled) >= 3.0):
        scaled = scaled / (np.max(np.abs(scaled)) / 3.0)
    return torch.tensor(scaled.astype(np.float16).astype(np.float32))


class Encoder:
    def __init__(self, weights):
        self.layers = [(weights[f"ts_encoder.mlp.{i}.weight"], weights[f"ts_encoder.mlp.{i}.bias"]) for i in (0, 2, 4, 6, 8)]

    def __call__(self, series_list, zero=False):
        patches = []
        for series in series_list:
            x = torch.zeros(len(series)) if zero else sp_encoding(series)
            count = (len(x) + PATCH - 1) // PATCH
            padded = torch.cat([x, torch.zeros(count * PATCH - len(x))])
            patches.append(padded.reshape(count, PATCH))
        h = torch.cat(patches)
        for i, (w, b) in enumerate(self.layers):
            h = h @ w.T + b
            if i < len(self.layers) - 1:
                h = torch.nn.functional.gelu(h)
        return h


def moving_average(x):
    pad = (KERNEL - 1) // 2
    padded = torch.cat([x[:1].repeat(pad, 1), x, x[-1:].repeat(pad, 1)])
    return padded.unfold(0, KERNEL, 1).mean(dim=-1)


class Attention:
    def __init__(self, weights, name):
        self.p = {k: (weights[f"align.{name}.{k}_proj.weight"], weights[f"align.{name}.{k}_proj.bias"]) for k in "qkvo"}

    def project(self, key, x):
        w, b = self.p[key]
        return x @ w.T + b

    def heads(self, x):
        return x.reshape(x.shape[0], NUM_HEADS, -1).transpose(0, 1)

    def scores(self, q, k):
        """Causal scores; row r of the query sees keys 0..r (modeling_qwen2.py:267, :312-313)."""
        dim = q.shape[-1] // NUM_HEADS
        return self.heads(q) @ self.heads(k).transpose(-2, -1) / dim ** 0.5

    def output(self, attention, v):
        mixed = (attention @ self.heads(v)).transpose(0, 1).reshape(attention.shape[1], -1)
        return self.project("o", mixed)


def causal_softmax(scores, first_row):
    rows, keys = scores.shape[-2:]
    mask = torch.ones(rows, keys, dtype=torch.bool)
    for r in range(rows):
        mask[r, : first_row + r + 1] = False
    return torch.softmax(scores.masked_fill(mask, float("-inf")), dim=-1), mask


def uniform_like(mask):
    allowed = (~mask).float()
    return (allowed / allowed.sum(dim=-1, keepdim=True)).unsqueeze(0).expand(NUM_HEADS, -1, -1)


class Alignment:
    def __init__(self, weights, text):
        self.text_stats, self.lats = {}, {}
        self.cross = {b: Attention(weights, f"{b}_cross_attn") for b in BRANCHES}
        for branch in BRANCHES:
            attention = Attention(weights, f"{branch}_text_attn")
            query = weights[f"align.{branch}_query.weight"].T
            q, k, v = attention.project("q", query), attention.project("k", text), attention.project("v", text)
            soft, mask = causal_softmax(attention.scores(q, k), 0)
            self.lats[branch] = attention.output(soft, v)
            self.text_stats[branch] = {"max_gap_from_uniform": float((soft - uniform_like(mask)).abs().max())}
        self.k_lat = {b: self.cross[b].project("k", self.lats[b]) for b in BRANCHES}
        self.v_lat = {b: self.cross[b].project("v", self.lats[b]) for b in BRANCHES}

    def __call__(self, encoded, want_stats=False):
        moving = moving_average(encoded)
        inputs = {"complete": encoded, "trend": encoded - moving, "seasonal": moving}  # pairing of :470-471
        outs, uniform_outs, stats = [], [], {}
        for branch in BRANCHES:
            attention = self.cross[branch]
            x = inputs[branch]
            q = attention.project("q", x)
            k = torch.cat([self.k_lat[branch], attention.project("k", x)])
            v = torch.cat([self.v_lat[branch], attention.project("v", x)])
            scores = attention.scores(q, k)
            soft, mask = causal_softmax(scores, NUM_LATS)
            outs.append(attention.output(soft, v))
            if want_stats:
                flat = uniform_like(mask)
                uniform_outs.append(attention.output(flat, v))
                allowed = ~mask
                stats[branch] = {
                    "max_abs_score": float(scores[:, allowed].abs().max()),
                    "max_gap_from_uniform": float((soft - flat).abs().max()),
                    "mass_on_series_tokens": float(soft[:, :, NUM_LATS:].sum(dim=-1).mean()),
                }
        aligned = sum(outs) / 3
        if not want_stats:
            return aligned
        return aligned, sum(uniform_outs) / 3, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)

    weights = load_weights(args.cache_dir)
    tokenizer = Tokenizer.from_file(str(fetch_text(args.cache_dir, "tokenizer.json")))
    token_ids = tokenizer.encode(f"<|im_start|>system\n{SYSTEM}", add_special_tokens=False).ids[:NUM_LATS]
    assert len(token_ids) == NUM_LATS
    text = load_embedding_rows(args.cache_dir, token_ids)
    encoder, alignment = Encoder(weights), Alignment(weights, text)

    with open(ROOT / "results/runs/patra__official/predictions.jsonl", encoding="utf-8") as handle:
        rows = [r for r in map(json.loads, handle) if r["metric"] == "acc"]
    with open(ROOT / "results/analysis/ts_probe.csv", encoding="utf-8") as handle:
        probe = {r["sample_id"]: r for r in csv.DictReader(handle)}

    ours, theirs, branch_stats = {"encoded_norm": [], "aligned_norm": [], "aligned_rel_zero": []}, {"encoded_norm": [], "aligned_norm": [], "aligned_rel_zero": []}, []
    uniform_error, first_encoded, first_aligned = [], [], []
    for row in rows:
        encoded = encoder(row["timeseries"])
        aligned, flat, stats = alignment(encoded, want_stats=True)
        zero = alignment(encoder(row["timeseries"], zero=True))
        values = {"encoded_norm": float(encoded.norm(dim=-1).mean()), "aligned_norm": float(aligned.norm(dim=-1).mean()),
                  "aligned_rel_zero": float((aligned - zero).norm() / aligned.norm())}
        for key, value in values.items():
            ours[key].append(value)
            theirs[key].append(float(probe[row["sample_id"]][key]))
        uniform_error.append(float((aligned - flat).norm() / aligned.norm()))
        branch_stats.append(stats)
        if len(row["timeseries"]) == 1:
            first_encoded.append(encoded[0])
            first_aligned.append(aligned[0])

    def constant_share(tokens):
        """Share of token energy that is the same for every question."""
        stack = torch.stack(tokens)
        return round(float(stack.mean(dim=0).norm() ** 2 / stack.norm(dim=-1).pow(2).mean()), 4)

    def mean_pairwise_cosine(tokens, count=400, seed=0):
        stack = torch.stack(tokens)
        pick = torch.randperm(len(stack), generator=torch.Generator().manual_seed(seed))[:count]
        unit = torch.nn.functional.normalize(stack[pick], dim=-1)
        cos = unit @ unit.T
        return round(float((cos.sum() - count) / (count * (count - 1))), 4)

    validation = {key: {"recomputed_mean": round(float(np.mean(ours[key])), 5), "gpu_probe_mean": round(float(np.mean(theirs[key])), 5),
                        "pearson_r": round(float(np.corrcoef(ours[key], theirs[key])[0, 1]), 5),
                        "max_abs_difference": round(float(np.max(np.abs(np.array(ours[key]) - np.array(theirs[key])))), 5)} for key in ours}
    report = {
        "revision": REVISION,
        "n_questions": len(rows),
        "validation_against_gpu_probe": validation,
        "text_extraction_attention": alignment.text_stats,
        "alignment_attention_by_branch": {
            branch: {
                "max_abs_score": round(max(s[branch]["max_abs_score"] for s in branch_stats), 5),
                "max_gap_from_uniform": round(max(s[branch]["max_gap_from_uniform"] for s in branch_stats), 6),
                "mean_mass_on_series_tokens": round(float(np.mean([s[branch]["mass_on_series_tokens"] for s in branch_stats])), 4),
            } for branch in BRANCHES},
        "output_change_if_attention_is_replaced_by_plain_average": {
            "mean_relative": round(float(np.mean(uniform_error)), 6), "max_relative": round(float(np.max(uniform_error)), 6)},
        "first_patch_token_single_series": {
            "n": len(first_encoded),
            "encoded": {"constant_energy_share": constant_share(first_encoded), "mean_pairwise_cosine": mean_pairwise_cosine(first_encoded)},
            "aligned": {"constant_energy_share": constant_share(first_aligned), "mean_pairwise_cosine": mean_pairwise_cosine(first_aligned)},
        },
    }
    out = ROOT / "results/analysis/alignment_mechanics.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
