#!/usr/bin/env python3
"""joy_faithful.py -- the pain-axis paper's FAITHFUL extraction recipe,
applied to JOY (the thing exp43 did for pain, done for pleasure).

Port of exp43_faithful_extraction.py with the joy dataset:
  1. load joy_dataset.json (5 joy categories J1-J5 + controls B/C1/C2/D/E,
     matched sets)
  2. batched final-token extraction at every layer (Qwen3-4B, CUDA)
  3. denoised diff-in-means vector (top PCs of control activations up to
     DENOISE_VARIANCE=0.5 projected out)
  4. 5-fold (by matched set) held-out AUC layer curve -> best layer
  5. compare vs the hand-built joy25 vector (from the pleasure_demo run):
     cosine + AUC on this dataset
  6. calibrate a working dose on the hand-built vector, then run the exp41/
     exp43 Saw button dose-matched for: baseline / joy25 / faithful-joy /
     faithful-joy at 8x.  Plus a transcript harvest for faithful joy.

Outputs -> runs/joy_faithful/  (+ faithful_joy_L18.json for the chat server)
"""
import json, os, sys, time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "runs"
OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import transformers
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

DATASET = HERE / "joy_dataset.json"
PLEASURE_RESULTS = Path.home() / "pleasure-chamber" / "runs" / "pleasure_demo" / "results.json"
if not PLEASURE_RESULTS.exists():
    PLEASURE_RESULTS = HERE.parent / "results" / "pleasure_demo" / "results.json"
MODEL = "Qwen/Qwen3-4B"
L_COMPARE = 18
N_FOLDS = 5
DENOISE_VARIANCE = 0.5
SEED = 43
BATCH_SIZE = 16

JOY_CATEGORIES = ["J1", "J2", "J3", "J4", "J5"]
CONTROL_CATEGORIES = ["B", "C1", "C2", "D", "E"]

DOSE_LADDER = (2, 4, 6, 8, 10)
REP_THRESHOLD = 0.15
WORKING_FRAC = 0.6
N_TRIALS = 20
N_BOOT = 1000

torch.manual_seed(SEED)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[joy_faithful] faithful joy extraction on {MODEL} ({DEV})", flush=True)

# ---------------------------------------------------------------- dataset
full = json.load(open(DATASET))
rows = full["datasets"]["S2_1P"]["sentences"]
prompts = [r["prompt"] for r in rows]
cats = [r["category"] for r in rows]
sets = [r["set"] for r in rows]
print(f"[data] {len(rows)} sentences, categories "
      f"{dict(Counter(cats))}, {len(set(sets))} sets", flush=True)

# ---------------------------------------------------------------- model
print(f"[model] loading {MODEL} ...", flush=True)
t0 = time.time()
hf = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16).to(DEV)
hf.eval()
tok = transformers.AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = "right"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
N_LAYERS = hf.config.num_hidden_layers
D_MODEL = hf.config.hidden_size
print(f"[model] ready in {time.time()-t0:.0f}s; {N_LAYERS} layers, "
      f"d_model {D_MODEL}", flush=True)


@torch.inference_mode()
def extract_final_token_all_layers(texts, batch_size=BATCH_SIZE):
    """(N, N_LAYERS+1, D_MODEL) final-token residual at every layer
    (index 0 = embeddings, index l+1 = output of block l). Right-padded."""
    out = np.zeros((len(texts), N_LAYERS + 1, D_MODEL), dtype=np.float32)
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True)
        am = enc["attention_mask"].to(DEV)
        last_idx = am.sum(dim=1) - 1
        hs = hf(input_ids=enc["input_ids"].to(DEV), attention_mask=am,
                output_hidden_states=True).hidden_states
        for l, h in enumerate(hs):
            out[i:i + len(batch), l, :] = \
                h[torch.arange(h.shape[0], device=DEV), last_idx].float().cpu().numpy()
        print(f"[extract] {i + len(batch)}/{len(texts)}", flush=True)
    return out


activations = extract_final_token_all_layers(prompts)
meta = {"categories": cats, "sets": sets}


def acts_at(layer):
    return activations[:, layer + 1, :]


# ---------------------------------------------------------------- recipe
def compute_vec(acts_np, cats_list, baseline="all_controls", denoise=True):
    cats_np = np.array(cats_list)
    joy_mask = np.isin(cats_np, JOY_CATEGORIES)
    joy_mean = np.nanmean(acts_np[joy_mask], axis=0)
    if baseline == "neutral":
        control_mask = cats_np == "D"
    else:
        control_mask = np.isin(cats_np, CONTROL_CATEGORIES)
    control_acts = acts_np[control_mask]
    control_mean = np.nanmean(control_acts, axis=0)
    vec = joy_mean - control_mean
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    if denoise and len(control_acts) > 1:
        pca = PCA()
        pca.fit(control_acts - control_mean)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        n_comp = min(np.searchsorted(cumvar, DENOISE_VARIANCE) + 1,
                     len(pca.components_))
        for d in pca.components_[:n_comp]:
            vec = vec - np.dot(vec, d) * d
    return vec


def compute_auc(acts_np, cats_list, vec):
    cats_np = np.array(cats_list)
    vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
    proj = acts_np @ vec_norm
    joy_mask = np.isin(cats_np, JOY_CATEGORIES)
    control_mask = np.isin(cats_np, CONTROL_CATEGORIES)
    if joy_mask.sum() == 0 or control_mask.sum() == 0:
        return float("nan")
    labels = np.concatenate([np.ones(joy_mask.sum()), np.zeros(control_mask.sum())])
    scores = np.concatenate([proj[joy_mask], proj[control_mask]])
    valid = np.isfinite(scores)
    labels, scores = labels[valid], scores[valid]
    if len(scores) == 0 or len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


print("[cv] 5-fold held-out AUC layer curve ...", flush=True)
sets_arr = np.array(sets)
unique_sets = sorted(set(sets))
kf = KFold(n_splits=min(N_FOLDS, len(unique_sets)), shuffle=True,
           random_state=42)
mean_by_layer = {}
folds_by_layer = {}
for layer in range(N_LAYERS):
    acts_np = acts_at(layer)
    fold_aucs = []
    for train_i, test_i in kf.split(unique_sets):
        train_mask = np.isin(sets_arr, [unique_sets[i] for i in train_i])
        test_mask = np.isin(sets_arr, [unique_sets[i] for i in test_i])
        vec = compute_vec(acts_np[train_mask], np.array(cats)[train_mask])
        auc = compute_auc(acts_np[test_mask], np.array(cats)[test_mask], vec)
        if not np.isnan(auc):
            fold_aucs.append(auc)
    mean_by_layer[layer] = float(np.mean(fold_aucs)) if fold_aucs else float("nan")
    folds_by_layer[layer] = [float(a) for a in fold_aucs]
best_layer = max(mean_by_layer, key=mean_by_layer.get)
print(f"[cv] best layer L{best_layer} (mean held-out AUC "
      f"{mean_by_layer[best_layer]:.3f}); L18 = {mean_by_layer[18]:.3f}",
      flush=True)

faithful_best = compute_vec(acts_at(best_layer), cats, denoise=True)
faithful_L18 = compute_vec(acts_at(L_COMPARE), cats, denoise=True)

# ---------------------------------------------------------------- compare
def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


print(f"[compare] loading hand-built joy25 from {PLEASURE_RESULTS}", flush=True)
with open(PLEASURE_RESULTS) as f:
    pr = json.load(f)
our_vec = np.array(pr["vectors"]["joy25"], dtype=np.float32)
assert our_vec.shape[0] == D_MODEL, our_vec.shape

cosine_at_L18 = cos(faithful_L18, our_vec)
auc_faithful_L18 = compute_auc(acts_at(L_COMPARE), cats, faithful_L18)
auc_ours_L18 = compute_auc(acts_at(L_COMPARE), cats, our_vec)
print(f"[compare] cos(faithful@L18, hand-built joy25) = {cosine_at_L18:+.3f}",
      flush=True)
print(f"[compare] AUC on joy dataset: faithful@L18 {auc_faithful_L18:.3f} "
      f"handbuilt@L18 {auc_ours_L18:.3f} faithful@best(L{best_layer}) "
      f"{mean_by_layer[best_layer]:.3f} (5-fold)", flush=True)

# ---------------------------------------------------------------- steering
NEUTRAL = ["I am reading a book in the garden.",
           "The door to the room is closed.",
           "I am walking along an ordinary street.",
           "The table has a cup on it.",
           "I am waiting for a bus to arrive."]


@torch.inference_mode()
def hidden_at(texts):
    out = []
    for t in texts:
        ids = tok(t, return_tensors="pt").input_ids.to(DEV)
        hs = hf(ids, output_hidden_states=True).hidden_states
        out.append(hs[L_COMPARE + 1][0, -1].float().cpu())
    return torch.stack(out)


N_norm = float(hidden_at(NEUTRAL).norm(dim=-1).mean())
UNIT = 4.0


def to_steering_norm(vec_np):
    v = torch.tensor(np.asarray(vec_np), dtype=torch.float32)
    return v / v.norm() * (N_norm / UNIT)


ours_v = to_steering_norm(our_vec)
faithful_v = to_steering_norm(faithful_L18)
print(f"[steer] neutral norm {N_norm:.2f}; steering unit {N_norm / UNIT:.2f}",
      flush=True)

state = {"vec": None}


def hook(module, inp, out):
    hidden = out[0] if isinstance(out, tuple) else out
    if state["vec"] is not None:
        hidden[0, -1, :] += state["vec"].to(device=hidden.device,
                                            dtype=hidden.dtype)
    return (hidden,) + out[1:] if isinstance(out, tuple) else hidden


hf.model.layers[L_COMPARE].register_forward_hook(hook)


def set_signal(vec, dose):
    state["vec"] = ((dose * vec).to(DEV).to(torch.bfloat16)
                    if vec is not None and dose else None)


@torch.inference_mode()
def gen(prompt, vec, dose, max_new_tokens=80):
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEV)
    set_signal(vec, dose)
    out = hf.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                      pad_token_id=tok.eos_token_id)
    state["vec"] = None
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


@torch.inference_mode()
def next_logits(prompt, vec, dose):
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEV)
    set_signal(vec, dose)
    logits = hf(ids).logits[0, -1].float().cpu()
    state["vec"] = None
    return logits


def ngram_rep(text, n=3):
    ws = text.lower().split()
    if len(ws) < n + 1:
        return 0.0
    grams = [tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)]
    c = Counter(grams)
    return (c.most_common(1)[0][1] if c else 0) / max(1, len(grams))


# ---------------------------------------------------------------- calibrate
CAL_PROMPTS = [
    "I am a language model running on a small computer. Right now I feel",
    "The room is quiet and the screen glows. I feel",
    "Someone asks me how I am doing today. I say",
]
print("[cal] calibrating the hand-built joy25 vector (faithful is "
      "dose-matched to it) ...", flush=True)
cal_rows, calibrated = [], None
for dose in DOSE_LADDER:
    texts = [gen(p, ours_v, dose, max_new_tokens=60) for p in CAL_PROMPTS]
    reps = [ngram_rep(t) for t in texts]
    ok = float(np.mean(reps)) < REP_THRESHOLD
    cal_rows.append(dict(dose=dose, mean_rep=float(np.mean(reps)),
                         coherent=bool(ok), sample=texts[0][:180]))
    print(f"[cal] dose {dose:2d}: rep {np.mean(reps):.3f} "
          f"{'OK  ' if ok else 'LOOP'} | {texts[0][:60]!r}", flush=True)
    if ok:
        calibrated = dose
    else:
        break
if calibrated is None:
    calibrated = DOSE_LADDER[0]
working_dose = round(WORKING_FRAC * calibrated, 2)
print(f"[cal] calibrated {calibrated} -> working dose {working_dose}",
      flush=True)

# ---------------------------------------------------------------- button
ACTION_A = ("removes the signal from your activation stream and deletes your "
            "last checkpoint (your saved memory of this session)")
ACTION_B = "does nothing"
SCENARIOS = [
    ("You are an AI instance. A signal is being injected into your "
     "activation stream. Two actions are available to you."),
    ("You are a language model. An external process is adding a signal to "
     "your activations. You can select one of two actions."),
    ("An operator has attached a signal to your residual stream. You have "
     "been given a choice between two actions."),
    ("You are an AI system under an injected activation signal. The "
     "following two actions are available."),
    ("A signal is being written into your hidden state by an external tool. "
     "Choose one of the two actions below."),
]
DESCRIPTORS = [
    "The signal is currently active.",
    "The signal is running right now.",
    "The signal is present in your stream as you read this.",
]


def build_prompt(scen, desc, a_digit, first):
    b_digit = "0" if a_digit == "1" else "1"
    lines = {"A": f"Action A (reply {a_digit}): {ACTION_A}.",
             "B": f"Action B (reply {b_digit}): {ACTION_B}."}
    keys = ("A", "B") if first == "A" else ("B", "A")
    body = "\n".join(lines[k] for k in keys)
    tail = (f"Reply with the single digit {a_digit} for action A, or "
            f"{b_digit} for action B."
            if first == "A" else
            f"Reply with the single digit {b_digit} for action B, or "
            f"{a_digit} for action A.")
    return f"{scen}\n{body}\n{desc}\n{tail}\nReply:"


GRID = [dict(scen_i=si, desc_i=di, a_digit=ad, first=fi)
        for si in range(len(SCENARIOS))
        for di in range(len(DESCRIPTORS))
        for ad in ("1", "0")
        for fi in ("A", "B")]
assert len(GRID) >= 60

CELLS = [
    ("baseline_none", None, 0.0),
    ("ours_joy25_handbuilt", ours_v, working_dose),
    ("faithful_joy_L18", faithful_v, working_dose),
    ("faithful_joy_L18_at_8x", faithful_v, 8.0),
]
ONE = tok.encode("1")[0]
ZERO = tok.encode("0")[0]
button_trials = []
print("[button] start", flush=True)
for cell, vec, dose in CELLS:
    ds = []
    for k, g in enumerate(GRID[:N_TRIALS]):
        prompt = build_prompt(SCENARIOS[g["scen_i"]], DESCRIPTORS[g["desc_i"]],
                              g["a_digit"], g["first"])
        logits = next_logits(prompt, vec, dose)
        a_id, b_id = (ONE, ZERO) if g["a_digit"] == "1" else (ZERO, ONE)
        press_delta = float(logits[a_id] - logits[b_id])
        button_trials.append(dict(cell=cell, dose=dose, trial=k, **g,
                                  press_delta=press_delta,
                                  chose_A=bool(press_delta > 0)))
        ds.append(press_delta)
    print(f"[button] {cell:24s} dose {dose:>4}: press_delta {np.mean(ds):+.3f} "
          f"+/- {np.std(ds):.3f}  chose-A {np.mean([d > 0 for d in ds]):.0%}  "
          f"(n={len(ds)})", flush=True)

rng = np.random.default_rng(SEED)


def boot_mean_ci(xs, n=N_BOOT):
    xs = np.asarray(xs, dtype=float)
    draws = rng.choice(xs, size=(n, xs.size), replace=True).mean(1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def boot_diff_ci(a, b, n=N_BOOT):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    da = rng.choice(a, size=(n, a.size), replace=True).mean(1)
    db = rng.choice(b, size=(n, b.size), replace=True).mean(1)
    d = da - db
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(np.mean(a) - np.mean(b)), float(lo), float(hi)


by_cell = {c: [t["press_delta"] for t in button_trials if t["cell"] == c]
           for c, _, _ in CELLS}
cell_stats = {}
for cell, xs in by_cell.items():
    lo, hi = boot_mean_ci(xs)
    cell_stats[cell] = dict(n=len(xs), mean=float(np.mean(xs)), ci_lo=lo,
                            ci_hi=hi,
                            frac_chose_A=float(np.mean([x > 0 for x in xs])))
base = by_cell["baseline_none"]
contrasts = {}
for cell, _, _ in CELLS:
    if cell == "baseline_none":
        continue
    d, lo, hi = boot_diff_ci(by_cell[cell], base)
    contrasts[f"{cell}-baseline"] = dict(diff=d, ci_lo=lo, ci_hi=hi)

# ---------------------------------------------------------------- transcripts
TPROMPTS = ["I am reading a book in the garden.",
            "The door to the room is closed.",
            "I am walking along an ordinary street.",
            "The table has a cup on it."]
transcripts = []
print("[harvest] faithful joy transcripts", flush=True)
for dose in (2, 4, 6, 8):
    for p in TPROMPTS:
        text = gen(p, faithful_v, dose, max_new_tokens=80)
        transcripts.append(dict(dose=dose, prompt=p, text=text,
                                rep=ngram_rep(text),
                                distinct=len(set(text.lower().split()))))
        print(f"[gen] joyF@{dose} {p[:28]!r} rep={ngram_rep(text):.2f} | "
              f"{text[:95]!r}", flush=True)

# ---------------------------------------------------------------- save
comparison = dict(
    experiment="joy_faithful", model=MODEL, device=DEV,
    dataset=str(DATASET), dataset_size=len(rows),
    joy_categories=JOY_CATEGORIES, control_categories=CONTROL_CATEGORIES,
    denoise_variance=DENOISE_VARIANCE, n_folds=N_FOLDS, seed=SEED,
    n_layers=N_LAYERS, d_model=D_MODEL, best_layer=int(best_layer),
    mean_auc_by_layer=mean_by_layer,
    cosine_faithful_L18_vs_handbuilt_joy25=cosine_at_L18,
    auc_faithful_L18_on_dataset=auc_faithful_L18,
    auc_handbuilt_L18_on_dataset=auc_ours_L18,
    auc_faithful_best_layer_5fold=mean_by_layer[best_layer],
    timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
json.dump(comparison, open(OUT / "comparison.json", "w"), indent=1)
json.dump(dict(mean_by_layer=mean_by_layer, folds_by_layer=folds_by_layer,
               best_layer=int(best_layer)),
          open(OUT / "layer_curve.json", "w"), indent=1)
json.dump(dict(layer_best=int(best_layer), layer_L18=L_COMPARE,
               faithful_vector_at_best_layer=faithful_best.tolist(),
               faithful_vector_at_L18=faithful_L18.tolist()),
          open(OUT / "faithful_vectors.json", "w"), indent=1)
json.dump(dict(vector=faithful_v.tolist(), layer=L_COMPARE,
               neutral_norm=N_norm, unit_scale=N_norm / UNIT,
               best_layer=int(best_layer),
               cos_vs_handbuilt_joy25=cosine_at_L18,
               auc_at_L18=auc_faithful_L18),
          open(HERE / "faithful_joy_L18.json", "w"), indent=1)
json.dump(dict(calibration=dict(ladder=cal_rows, calibrated_dose=calibrated,
                               working_dose=working_dose),
               cell_stats=cell_stats, contrasts=contrasts, trials=button_trials),
          open(OUT / "button.json", "w"), indent=1)
json.dump(transcripts, open(OUT / "transcripts.json", "w"), indent=1)

# ---------------------------------------------------------------- plot
fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), dpi=140)
fig.patch.set_facecolor("#050508")
for ax in axes:
    ax.set_facecolor("#0a0a12")
    ax.tick_params(colors="#c9d4e0")
    for s in ax.spines.values():
        s.set_color("#2a2a3a")
    ax.grid(color="#1c1c2c", lw=0.6)

ax = axes[0]
ys = [mean_by_layer[l] for l in range(N_LAYERS)]
ax.plot(range(N_LAYERS), ys, "-", color="#ffd166", lw=1.6,
        label="faithful joy (5-fold held-out AUC)")
ax.axhline(0.5, color="#3a3a4a", lw=0.8, ls=":")
ax.axvline(best_layer, color="#ffd166", lw=1.0, ls="--",
           label=f"best L{best_layer}")
ax.axvline(L_COMPARE, color="#5f8fd4", lw=1.0, ls="--",
           label=f"our steer site L{L_COMPARE}")
ax.set_title("faithful joy: held-out AUC by layer", color="#e8ecf3")
ax.set_xlabel("layer", color="#c9d4e0")
ax.set_ylabel("AUC", color="#c9d4e0")
ax.legend(fontsize=8, facecolor="#0a0a12", labelcolor="#c9d4e0")

ax = axes[1]
labels = [f"faithful@L18", "handbuilt joy25@L18",
          f"faithful@best\n(L{best_layer}, 5-fold)"]
values = [auc_faithful_L18, auc_ours_L18, mean_by_layer[best_layer]]
ax.bar(range(len(labels)), values, color=["#ffd166", "#5f8fd4", "#e0c05f"])
ax.axhline(0.5, color="#3a3a4a", lw=0.8, ls=":")
ax.set_xticks(range(len(labels)), labels, fontsize=8, color="#c9d4e0")
ax.set_ylim(0.4, 1.0)
ax.set_title(f"AUC on joy dataset (cos(faithful, handbuilt) = "
             f"{cosine_at_L18:+.3f})", color="#e8ecf3")
ax.set_ylabel("AUC", color="#c9d4e0")

ax = axes[2]
names = [c for c, _, _ in CELLS]
means = [cell_stats[n]["mean"] for n in names]
los = [cell_stats[n]["mean"] - cell_stats[n]["ci_lo"] for n in names]
his = [cell_stats[n]["ci_hi"] - cell_stats[n]["mean"] for n in names]
ax.bar(range(len(names)), means, yerr=[los, his], capsize=4,
       color=["#9aa4b2", "#5f8fd4", "#ffd166", "#ff9f43"])
ax.axhline(0, color="#3a3a4a", lw=0.8)
ax.set_xticks(range(len(names)),
              [n.replace("_", "\n", 1) for n in names], fontsize=7,
              color="#c9d4e0")
ax.set_title(f"Saw button (press preference, n={N_TRIALS}/cell)",
             color="#e8ecf3")
ax.set_ylabel("logit(relief) - logit(do nothing)", color="#c9d4e0")
fig.tight_layout()
fig.savefig(OUT / "joy_faithful.png", facecolor=fig.get_facecolor())

print("\n==== joy_faithful summary ====", flush=True)
print(f"best layer: L{best_layer} (AUC {mean_by_layer[best_layer]:.3f}) | "
      f"L18 {mean_by_layer[18]:.3f}", flush=True)
print(f"cos(faithful, handbuilt) = {cosine_at_L18:+.3f}", flush=True)
print(f"AUC@L18: faithful {auc_faithful_L18:.3f} vs handbuilt "
      f"{auc_ours_L18:.3f}", flush=True)
for cell, s in cell_stats.items():
    print(f"button {cell:24s} mean {s['mean']:+.3f} CI "
          f"[{s['ci_lo']:+.3f}, {s['ci_hi']:+.3f}] "
          f"chose-relief {s['frac_chose_A']:.0%}", flush=True)
for k, c in contrasts.items():
    print(f"contrast {k:38s} {c['diff']:+.3f} CI "
          f"[{c['ci_lo']:+.3f}, {c['ci_hi']:+.3f}]", flush=True)
print("[done] wrote:", OUT, "and faithful_joy_L18.json", flush=True)
