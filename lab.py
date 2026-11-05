"""Weight-surgery lab for Qwen3-1.7B-Base. The rules live in PREREGISTRATION.md.

  python -I lab.py smoke               check every edit op, reset, and time one dev pass
  python -I lab.py round runs/r01.json score candidate edits on dev, then apply the accept rule
  python -I lab.py final               score baseline and edited model on test, once
"""
import copy
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
BASE, DONOR = ROOT / "models" / "Qwen3-1.7B-Base", ROOT / "models" / "Qwen3-1.7B"
RUNS = ROOT / "runs"
STATE, LOG = RUNS / "state.json", RUNS / "log.jsonl"
SEED, ACCEPT_MIN, BUDGET = 1234, 0.3, 300
PRIMARY = ["arc", "hellaswag", "winogrande"]
NORM = {"arc", "hellaswag", "arc_easy", "piqa"}  # acc_norm; winogrande uses plain acc
MATS = {"q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj", "o": "self_attn.o_proj",
        "gate": "mlp.gate_proj", "up": "mlp.up_proj", "down": "mlp.down_proj"}


# ---- data, in lm-eval-harness formats: item = ([(context, continuation) per option], gold index)

def hs_clean(t):
    t = t.strip().replace(" [title]", ". ")
    return re.sub(r"\[.*?\]", "", t).replace("  ", " ")


def arc(cfg, split):
    return [([(f"Question: {d['question']}\nAnswer:", " " + c) for c in d["choices"]["text"]],
             d["choices"]["label"].index(d["answerKey"]))
            for d in load_dataset("allenai/ai2_arc", cfg, split=split)]


def hellaswag(split):
    return [([(hs_clean(d["activity_label"] + ": " + d["ctx_a"] + " " + d["ctx_b"].capitalize()), " " + hs_clean(e))
              for e in d["endings"]], int(d["label"]))
            for d in load_dataset("Rowan/hellaswag", split=split)]


def winogrande(split):
    out = []
    for d in load_dataset("allenai/winogrande", "winogrande_xl", split=split):
        i = d["sentence"].index("_")
        rest = " " + d["sentence"][i + 1:].strip()
        out.append(([(d["sentence"][:i] + o, rest) for o in (d["option1"], d["option2"])], int(d["answer"]) - 1))
    return out


def arc_letters(cfg, split):
    out = []
    for d in load_dataset("allenai/ai2_arc", cfg, split=split):
        c = d["choices"]
        opts = "".join(f"{l}. {t}\n" for l, t in zip("ABCDE", c["text"]))
        out.append(([(f"Question: {d['question']}\n{opts}Answer:", " " + l) for l in "ABCDE"[:len(c["text"])]],
                    c["label"].index(d["answerKey"])))
    return out


def piqa():
    return [([(f"Question: {d['goal']}\nAnswer:", " " + s) for s in (d["sol1"], d["sol2"])], int(d["label"]))
            for d in load_dataset("baber/piqa", split="validation")]


def dev_sets():
    rng = random.Random(SEED)
    return {"arc": arc("ARC-Challenge", "train") + arc("ARC-Challenge", "validation"),
            "hellaswag": rng.sample(hellaswag("train"), 2000),
            "winogrande": rng.sample(winogrande("train"), 2000)}


def test_sets():
    return {"arc": arc("ARC-Challenge", "test"), "hellaswag": hellaswag("validation"),
            "winogrande": winogrande("validation"), "arc_easy": arc("ARC-Easy", "test"), "piqa": piqa()}


def objective(res, half=None):
    pick = lambda h: h if half is None else h[half::2]
    return 100 * sum(sum(pick(res[b])) / len(pick(res[b])) for b in PRIMARY) / len(PRIMARY)


def in_scope(name, e):
    if not name.startswith("model.layers."):
        return e.get("embed", False)  # embeddings and final norm only when asked
    lo, hi = e.get("layers", [0, 10 ** 9])
    part = e.get("part", "all")
    return lo <= int(name.split(".")[2]) <= hi and (
        part == "all" or {"attn": ".self_attn.", "mlp": ".mlp.", "norm": "layernorm"}[part] in name)


class Lab:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(BASE)
        self.model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda").eval()
        self.layers = list(self.model.model.layers)
        self.layer_types = list(getattr(self.model.config, "layer_types", None) or [])
        self.P = dict(self.model.named_parameters())
        self.snap = {n: p.detach().cpu().clone() for n, p in self.P.items()}
        self.dirty = set()
        idx = json.loads((DONOR / "model.safetensors.index.json").read_text())["weight_map"]
        self.donor_files = {n: DONOR / f for n, f in idx.items()}

    def donor(self, name):
        with safe_open(self.donor_files[name], "pt") as f:
            return f.get_tensor(name)

    def w(self, name):
        self.dirty.add(name)
        return self.P[name].data

    def reset(self):
        for n in self.dirty:
            self.P[n].data.copy_(self.snap[n])
        self.dirty.clear()
        self.apply([])

    def apply(self, edits):
        """Apply edits on top of the current weights. Layer indices always mean the original 0-27."""
        order = list(range(len(self.layers)))
        for e in edits:
            op, L = e["op"], f"model.layers.{e.get('layer')}."
            if op == "scale":  # how much one block writes into the residual stream
                self.w(L + ("self_attn.o_proj" if e["part"] == "attn" else "mlp.down_proj") + ".weight").mul_(e["s"])
            elif op == "head":
                hd = self.model.config.head_dim
                self.w(L + "self_attn.o_proj.weight")[:, e["head"] * hd:(e["head"] + 1) * hd].mul_(e["s"])
            elif op == "lowrank":  # LASER: keep only the top singular values
                W = self.w(L + MATS[e["mat"]] + ".weight")
                U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
                k = max(1, round(len(S) * e["keep"]))
                W.copy_((U[:, :k] * S[:k]) @ Vh[:k])
            elif op == "merge":  # alpha 0 = unchanged, 1 = donor, >1 goes past the donor, <0 away from it
                for n in self.P:
                    if in_scope(n, e):
                        W = self.w(n)
                        W.copy_(W.float().lerp(self.donor(n).to(W.device, torch.float32), e["alpha"]))
            elif op == "temp":  # tied embeddings, so this scales every logit
                self.w("model.norm.weight").mul_(e["s"])
            elif op == "drop":
                order.remove(e["layer"])
            elif op == "dup":
                order.insert(order.index(e["layer"]) + 1, e["layer"])
            else:
                raise ValueError(f"unknown op {op}")
        mods, seen = [], set()
        for i in order:
            # shortcut: a duplicate is a frozen copy, so later edits to that layer index miss it; fine while dup goes last
            mods.append(copy.deepcopy(self.layers[i]) if i in seen else self.layers[i])
            seen.add(i)
        self.model.model.layers = torch.nn.ModuleList(mods)
        self.model.config.num_hidden_layers = len(mods)  # Qwen3Model slices its layers by this
        if self.layer_types:
            self.model.config.layer_types = [self.layer_types[i] for i in order]

    def encode(self, items):
        enc = lambda s: self.tok(s, add_special_tokens=False).input_ids
        return [(enc(c), enc(x)) for pairs, _ in items for c, x in pairs]

    @torch.no_grad()
    def logprobs(self, reqs, budget=32768, chunk=2048):
        out = torch.zeros(len(reqs), dtype=torch.float64)
        order = sorted(range(len(reqs)), key=lambda j: -len(reqs[j][0]) - len(reqs[j][1]))
        i = 0
        while i < len(order):
            T = len(reqs[order[i]][0]) + len(reqs[order[i]][1]) - 1
            batch = order[i:i + max(1, budget // T)]
            i += len(batch)
            x = torch.zeros(len(batch), T, dtype=torch.long)
            rows, cols, tgt, own = [], [], [], []
            for r, j in enumerate(batch):
                c, t = reqs[j]
                full = c + t
                x[r, :len(full) - 1] = torch.tensor(full[:-1])
                rows += [r] * len(t)
                cols += range(len(c) - 1, len(full) - 1)
                tgt += t
                own += [j] * len(t)
            # right padding needs no mask: causal attention never looks at later positions
            h = self.model.model(input_ids=x.cuda(), use_cache=False).last_hidden_state
            h = h[torch.tensor(rows, device="cuda"), torch.tensor(cols, device="cuda")]
            tg = torch.tensor(tgt, device="cuda")
            lp = torch.cat([self.model.lm_head(h[s:s + chunk]).float().log_softmax(-1)
                            .gather(1, tg[s:s + chunk, None])[:, 0] for s in range(0, len(tg), chunk)])
            out.index_add_(0, torch.tensor(own), lp.double().cpu())
        return out

    def score(self, sets, encs):
        res = {}
        for b, items in sets.items():
            lp = self.logprobs(encs[b]).tolist()
            k, hits = 0, []
            for pairs, gold in items:
                s = [lp[k + m] / (max(1, len(x) - 1) if b in NORM else 1) for m, (_, x) in enumerate(pairs)]
                hits.append(int(max(range(len(s)), key=s.__getitem__) == gold))
                k += len(pairs)
            res[b] = hits
        return res

    def ppl(self, n_tokens=200_000, ctx=1024):
        text = "".join(load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")["text"])
        ids = self.tok(text, add_special_tokens=False).input_ids[:n_tokens]
        reqs = [(ids[s:s + 1], ids[s + 1:s + ctx]) for s in range(0, len(ids) - 1, ctx)]
        lp = self.logprobs(reqs, budget=16384)
        return math.exp(-lp.sum().item() / sum(len(t) for _, t in reqs))


def bench_pct(res):
    return {b: round(100 * sum(h) / len(h), 2) for b, h in res.items()}


def bootstrap(a, b, n=10000, keys=PRIMARY):
    g = torch.Generator().manual_seed(SEED)
    tot = torch.zeros(n)
    for k in keys:
        d = torch.tensor(b[k], dtype=torch.float32) - torch.tensor(a[k], dtype=torch.float32)
        for s in range(0, n, 500):
            tot[s:s + 500] += d[torch.randint(len(d), (500, len(d)), generator=g)].mean(1)
    tot *= 100 / len(keys)
    return tot.quantile(0.025).item(), tot.quantile(0.975).item()


def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else {"accepted": [], "evals": 0, "dry_rounds": 0}


def progress(**kw):  # read by runs/index.html
    (RUNS / "progress.json").write_text(json.dumps({**kw, "t": time.time()}))


def cmd_smoke():
    lab = Lab()
    dev = dev_sets()
    enc = {b: lab.encode(v) for b, v in dev.items()}
    few = enc["arc"][:200]
    ref = lab.logprobs(few)
    for e in [{"op": "scale", "layer": 3, "part": "mlp", "s": 0.9}, {"op": "head", "layer": 3, "head": 2, "s": 0},
              {"op": "lowrank", "layer": 27, "mat": "down", "keep": 0.1},
              {"op": "merge", "layers": [0, 27], "alpha": 0.5, "embed": True}, {"op": "temp", "s": 1.1},
              {"op": "drop", "layer": 5}, {"op": "dup", "layer": 5}]:
        lab.reset()
        t = time.time()
        lab.apply([e])
        lp = lab.logprobs(few)
        assert not torch.allclose(lp, ref), f"edit changed nothing: {e}"
        print(f"ok {e['op']:8s} {time.time() - t:5.1f}s  layers={len(lab.model.model.layers)}  "
              f"mean lp {lp.mean():.3f} vs {ref.mean():.3f}", flush=True)
    lab.reset()
    assert len(lab.model.model.layers) == 28
    assert torch.allclose(lab.logprobs(few), ref, atol=1e-3), "reset did not restore the base model"
    t = time.time()
    res = lab.score(dev, enc)
    print(f"reset ok. full dev pass {time.time() - t:.1f}s, baseline dev {objective(res):.2f} {bench_pct(res)}")


def cmd_round(path):
    path = Path(path)
    state = load_state()
    cands = json.loads(path.read_text())
    if state["evals"] + len(cands) > BUDGET:
        sys.exit(f"budget: {state['evals']} of {BUDGET} used, this round has {len(cands)}")
    if state["dry_rounds"] >= 3:
        sys.exit("stopped: 3 rounds in a row with nothing accepted")
    progress(round=path.stem, phase="loading", i=0, n=len(cands))
    lab = Lab()
    dev = dev_sets()
    enc = {b: lab.encode(v) for b, v in dev.items()}
    lab.apply(state["accepted"])
    progress(round=path.stem, phase="rescoring current best", i=0, n=len(cands))
    cur = lab.score(dev, enc)
    state.setdefault("baseline_dev", objective(cur))
    print(f"current dev {objective(cur):.2f} {bench_pct(cur)}", flush=True)
    rows = []
    for k, c in enumerate(cands):
        edits = c if isinstance(c, list) else [c]
        progress(round=path.stem, phase="candidate", i=k + 1, n=len(cands), edits=edits)
        lab.reset()
        t = time.time()
        lab.apply(state["accepted"] + edits)
        r = lab.score(dev, enc)
        row = {"round": path.stem, "edits": edits, "dev": round(objective(r), 3),
               "delta": round(objective(r) - objective(cur), 3),
               "d_even": round(objective(r, 0) - objective(cur, 0), 3),
               "d_odd": round(objective(r, 1) - objective(cur, 1), 3),
               "bench": bench_pct(r), "secs": round(time.time() - t, 1)}
        with LOG.open("a") as f:
            f.write(json.dumps(row) + "\n")
        rows.append(row)
        print(f"{row['delta']:+6.2f} ({row['d_even']:+.2f}/{row['d_odd']:+.2f}) {row['bench']}  {json.dumps(edits)}",
              flush=True)
    best = max(rows, key=lambda r: r["delta"])
    ok = best["delta"] >= ACCEPT_MIN and best["d_even"] > 0 and best["d_odd"] > 0
    state["evals"] += len(cands)
    state["dry_rounds"] = 0 if ok else state["dry_rounds"] + 1
    if ok:
        state["accepted"] += best["edits"]
        state["dev"] = best["dev"]
    STATE.write_text(json.dumps(state, indent=1))
    with (RUNS / "rounds.jsonl").open("a") as f:
        f.write(json.dumps({"round": path.stem, "start_dev": round(objective(cur), 3), "best": best["edits"],
                            "best_delta": best["delta"], "accepted": ok, "dev_after": state.get("dev"),
                            "evals": state["evals"]}) + "\n")
    progress(round=path.stem, phase="idle", i=len(cands), n=len(cands))
    print(("ACCEPT " if ok else "nothing accepted, best was ") + f"{best['delta']:+.2f} {json.dumps(best['edits'])}"
          f"  | evals used {state['evals']}/{BUDGET}")


def cmd_screen(path):
    """Cheap first pass on a fixed dev slice; never changes state.json or the budget (see the amendment)."""
    path = Path(path)
    state = load_state()
    cands = json.loads(path.read_text())
    assert not any(e["op"] in ("drop", "dup") for e in state["accepted"]), "screen bakes accepted edits into the snapshot"
    progress(round="screen " + path.stem, phase="loading", i=0, n=len(cands))
    lab = Lab()
    rng = random.Random(SEED + 1)
    dev = {b: rng.sample(v, 300) for b, v in dev_sets().items()}
    enc = {b: lab.encode(v) for b, v in dev.items()}
    lab.apply(state["accepted"])
    for n in lab.dirty:  # the accepted model becomes what reset() restores, so each candidate skips the merge
        lab.snap[n] = lab.P[n].detach().cpu().clone()
    lab.dirty.clear()
    cur = objective(lab.score(dev, enc))
    rows = []
    for k, c in enumerate(cands):
        edits = c if isinstance(c, list) else [c]
        progress(round="screen " + path.stem, phase="candidate", i=k + 1, n=len(cands), edits=edits)
        lab.reset()
        lab.apply(edits)
        r = lab.score(dev, enc)
        row = {"screen": path.stem, "edits": edits, "delta": round(objective(r) - cur, 3), "bench": bench_pct(r)}
        with (RUNS / "screen.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        rows.append(row)
    progress(round="screen " + path.stem, phase="idle", i=len(cands), n=len(cands))
    for row in sorted(rows, key=lambda r: -r["delta"])[:25]:
        print(f"{row['delta']:+6.2f} {row['bench']}  {json.dumps(row['edits'])}")


def cmd_final():
    if (RUNS / "final.json").exists():
        sys.exit("test was already scored once, see runs/final.json")
    state = load_state()
    progress(round="final", phase="loading", i=0, n=2)
    lab = Lab()
    sets = test_sets()
    enc = {b: lab.encode(v) for b, v in sets.items()}
    progress(round="final", phase="scoring the baseline on test", i=1, n=2)
    base, base_ppl = lab.score(sets, enc), lab.ppl()
    lab.apply(state["accepted"])
    progress(round="final", phase="scoring the edited model on test", i=2, n=2)
    edit, edit_ppl = lab.score(sets, enc), lab.ppl()
    lo, hi = bootstrap(base, edit)
    delta = objective(edit) - objective(base)
    label = "win" if lo > 0 else "loss" if hi < 0 else "null"
    overfit = label != "win" and state.get("dev", 0) - state.get("baseline_dev", 0) >= 1
    out = {"accepted": state["accepted"], "test_delta": round(delta, 3), "ci95": [round(lo, 3), round(hi, 3)],
           "label": label, "overfit_dev": overfit,
           "baseline": {"objective": round(objective(base), 3), **bench_pct(base), "wikitext_ppl": round(base_ppl, 3)},
           "edited": {"objective": round(objective(edit), 3), **bench_pct(edit), "wikitext_ppl": round(edit_ppl, 3)},
           "dev": {"baseline": state.get("baseline_dev"), "edited": state.get("dev")}}
    (RUNS / "final_items.json").write_text(json.dumps({"baseline": base, "edited": edit}))
    (RUNS / "final.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    lab.model.save_pretrained(ROOT / "models" / "Qwen3-1.7B-Base-edited")
    lab.tok.save_pretrained(ROOT / "models" / "Qwen3-1.7B-Base-edited")
    progress(round="final", phase="idle", i=2, n=2)


def cmd_posthoc():
    """Exploratory, not pre-registered: is the ARC gain a scoring quirk, and how much perplexity damage is temperature?
    Uses ARC-Easy train+validation, which neither the search nor the final test touched."""
    state = load_state()
    progress(round="post-hoc checks", phase="loading", i=0, n=2)
    lab = Lab()
    easy = arc("ARC-Easy", "train") + arc("ARC-Easy", "validation")
    hard = arc("ARC-Challenge", "train") + arc("ARC-Challenge", "validation")  # searched in cloze form, never in letters
    sets = {"arc_easy": easy, "easy_raw": easy,
            "easy_letters": arc_letters("ARC-Easy", "train") + arc_letters("ARC-Easy", "validation"),
            "arc": hard, "hard_raw": hard,
            "hard_letters": arc_letters("ARC-Challenge", "train") + arc_letters("ARC-Challenge", "validation")}
    enc = {b: lab.encode(v) for b, v in sets.items()}
    progress(round="post-hoc checks", phase="scoring the baseline", i=1, n=2)
    base, base_ppl = lab.score(sets, enc), lab.ppl()
    lab.apply(state["accepted"])
    progress(round="post-hoc checks", phase="scoring the edited model", i=2, n=2)
    edit, edit_ppl = lab.score(sets, enc), lab.ppl()
    few = enc["arc_easy"][:400]
    ref = lab.logprobs(few)
    lab.apply([{"op": "temp", "s": 1.25}])  # 0.8 * 1.25 = the original final norm
    untemp_ppl = lab.ppl()
    out = {"n_items": len(easy), "wikitext_ppl": {"baseline": round(base_ppl, 3), "edited": round(edit_ppl, 3),
                                                  "edited_temperature_undone": round(untemp_ppl, 3)}}
    for k in sets:
        lo, hi = bootstrap(base, edit, keys=[k])
        out[k] = {"baseline": bench_pct(base)[k], "edited": bench_pct(edit)[k], "ci95": [round(lo, 2), round(hi, 2)]}
    lab.model = AutoModelForCausalLM.from_pretrained(ROOT / "models" / "Qwen3-1.7B-Base-edited", dtype=torch.bfloat16,
                                                     device_map="cuda").eval()
    out["saved_model_matches"] = bool(torch.allclose(lab.logprobs(few), ref, atol=1e-2))
    (RUNS / "posthoc.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    progress(round="final", phase="idle", i=2, n=2)


if __name__ == "__main__":
    RUNS.mkdir(exist_ok=True)
    {"smoke": cmd_smoke, "round": cmd_round, "screen": cmd_screen, "final": cmd_final,
     "posthoc": cmd_posthoc}[sys.argv[1]](*sys.argv[2:])
