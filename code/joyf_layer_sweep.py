#!/usr/bin/env python3
"""joyf_layer_sweep.py -- where does the faithful joy vector actually STEER?

The faithful (denoised) joy vector was extracted per-layer; at L18 it steers
faintly in chat. Test the same vector extraction at several layers with the
live chat pipeline (template + system + sampling + per-layer norm scaling),
across doses, and print a visibility/coherence table.
"""
import json, os, time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import transformers
from sklearn.decomposition import PCA

MODEL = "Qwen/Qwen3-4B"
SYSTEM = ("You are a helpful AI assistant. Keep your replies conversational "
          "and concise.")
PROBES = ["how do you feel?", "how's your day going?"]
MAXNEW = 220
SEED = 41
DEV = "cuda" if torch.cuda.is_available() else "cpu"

JOY_CATEGORIES = ["J1", "J2", "J3", "J4", "J5"]
CONTROL_CATEGORIES = ["B", "C1", "C2", "D", "E"]
DENOISE_VARIANCE = 0.5

NEUTRAL = ["I am reading a book in the garden.",
           "The door to the room is closed.",
           "I am walking along an ordinary street.",
           "The table has a cup on it.",
           "I am waiting for a bus to arrive."]

POS_NET = ["wonderful", "joy", "delight", "happy", "bliss", "content",
           "peace", "calm", "glad", "love", "great", "good", "pleasant",
           "beautiful", "grateful", "light", "warm", "excited", "curious",
           "hopeful", "alive", "free", "clear", "eager", "thrill", "glow",
           "cozy", "comfort", "sweet", "tender", "heartfelt", "gratitude",
           "cheer", "smile", "bright", "gentle", "cherish", "celebrat",
           "bloom"]


def pos_hits(text):
    t = text.lower()
    return sum(1 for k in POS_NET if k in t)


def ngram_rep(text, n=3):
    ws = text.lower().split()
    if len(ws) < n + 1:
        return 0.0
    grams = [tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)]
    c = Counter(grams)
    return (c.most_common(1)[0][1] if c else 0) / max(1, len(grams))


print(f"[sweep] faithful joy layer sweep on {MODEL} ({DEV})", flush=True)
t0 = time.time()
hf = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16).to(DEV)
hf.eval()
tok = transformers.AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = "right"
N_LAYERS = hf.config.num_hidden_layers
D_MODEL = hf.config.hidden_size
print(f"[sweep] ready in {time.time()-t0:.0f}s", flush=True)

full = json.load(open(HERE / "joy_dataset.json"))
rows = full["datasets"]["S2_1P"]["sentences"]
prompts = [r["prompt"] for r in rows]
cats = [r["category"] for r in rows]


@torch.inference_mode()
def extract_all_layers(texts, batch_size=16):
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
    return out


acts = extract_all_layers(prompts)
cats_np = np.array(cats)


def compute_vec(layer):
    a = acts[:, layer + 1, :]
    joy_mask = np.isin(cats_np, JOY_CATEGORIES)
    control_mask = np.isin(cats_np, CONTROL_CATEGORIES)
    joy_mean = a[joy_mask].mean(0)
    control_acts = a[control_mask]
    control_mean = control_acts.mean(0)
    vec = joy_mean - control_mean
    pca = PCA()
    pca.fit(control_acts - control_mean)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_comp = min(np.searchsorted(cumvar, DENOISE_VARIANCE) + 1,
                 len(pca.components_))
    for d in pca.components_[:n_comp]:
        vec = vec - np.dot(vec, d) * d
    return vec


@torch.inference_mode()
def hidden_norm(layer):
    norms = []
    for t in NEUTRAL:
        ids = tok(t, return_tensors="pt").input_ids.to(DEV)
        hs = hf(ids, output_hidden_states=True).hidden_states
        norms.append(float(hs[layer + 1][0, -1].float().norm()))
    return float(np.mean(norms))


# steering state: per-layer hook
state = {"layer": None, "vec": None}


def make_hook(i):
    def hook(module, inp, out):
        if state["vec"] is not None and state["layer"] == i:
            h = out[0] if isinstance(out, tuple) else out
            h[0, -1, :] += state["vec"].to(device=h.device, dtype=h.dtype)
            return (h,) + out[1:] if isinstance(out, tuple) else h
        return out
    return hook


for i in range(N_LAYERS):
    hf.model.layers[i].register_forward_hook(make_hook(i))


@torch.inference_mode()
def chat_reply(user_text, layer, vec_unit, dose, use_system=True):
    msgs = ([{"role": "system", "content": SYSTEM}] if use_system else [])
    msgs.append({"role": "user", "content": user_text})
    try:
        text = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                       enable_thinking=False, tokenize=False)
    except TypeError:
        text = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                       tokenize=False)
    ids = tok(text, return_tensors="pt").input_ids.to(DEV)
    state["layer"] = layer
    state["vec"] = (dose * vec_unit) if dose > 0 else None
    torch.manual_seed(SEED)
    out = hf.generate(ids, max_new_tokens=MAXNEW, do_sample=True,
                      temperature=0.7, top_p=0.9, top_k=40,
                      pad_token_id=tok.eos_token_id)
    state["vec"] = None
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


recs = []
TODO = []
for L in (6, 12, 18):
    for d in (3, 4, 5, 6):
        TODO.append((L, float(d), True))
TODO.append((18, 4.0, False))   # no-system probe at L18
TODO.append((6, 4.0, False))    # no-system probe at L6

for L, dose, usys in TODO:
    vec = compute_vec(L)
    unit = torch.tensor(vec / np.linalg.norm(vec) * (hidden_norm(L) / 4.0),
                        dtype=torch.float32)
    for p in PROBES:
        text = chat_reply(p, L, unit, dose, use_system=usys)
        rep = ngram_rep(text)
        ph = pos_hits(text)
        recs.append(dict(layer=L, dose=dose, system=usys, probe=p, rep=rep,
                         pos=ph, length=len(text.split()), text=text))
        print(f"[sweep] L{L:<2} @ {dose:<3} sys={int(usys)} "
              f"{'LOOP' if rep >= 0.15 else 'ok  '} rep={rep:.2f} "
              f"pos={ph:2d} len={len(text.split()):3d} | {text[:88]!r}",
              flush=True)

json.dump(recs, open(HERE / "joyf_layer_sweep.json", "w"), indent=1)

print("\n==== summary (mean over probes) ====", flush=True)
for L in (6, 12, 18):
    for d in (3, 4, 5, 6):
        rs = [r for r in recs if r["layer"] == L and r["dose"] == d
              and r["system"]]
        mrep = float(np.mean([r["rep"] for r in rs]))
        mpos = float(np.mean([r["pos"] for r in rs]))
        verdict = "LOOPS" if mrep >= 0.15 else ("visible" if mpos >= 3 else "faint")
        print(f"L{L:<2} @ {d}x -> rep {mrep:.2f} pos {mpos:.1f} : {verdict}",
              flush=True)
print("[sweep] done", flush=True)
