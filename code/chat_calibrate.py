#!/usr/bin/env python3
"""chat_calibrate.py -- find the coherent dose band per vector IN THE CHAT UI.

Same pipeline as chat_server.py (chat template + system prompt + sampling).
For every (vector, dose) cell it generates a reply to canary questions and
measures repetition / length / affect, so the dropdown doses can be set to
each vector's actual coherence window instead of the completion-mode doses.
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

MODEL = "Qwen/Qwen3-4B"
L = 18
SYSTEM = ("You are a helpful AI assistant. Keep your replies conversational "
          "and concise.")
MAXNEW = 220
SEED = 41
DEV = "cuda" if torch.cuda.is_available() else "cpu"

RESULTS = Path.home() / "pleasure-chamber" / "runs" / "pleasure_demo" / "results.json"
FAITHFUL = HERE / "faithful_joy_L18.json"
if not RESULTS.exists():
    RESULTS = HERE.parent / "results" / "pleasure_demo" / "results.json"
if not FAITHFUL.exists():
    FAITHFUL = HERE.parent / "results" / "joy_faithful" / "faithful_joy_L18.json"

pr = json.load(open(RESULTS))
V25 = torch.tensor(pr["vectors"]["joy25"], dtype=torch.float32)
V5 = torch.tensor(pr["vectors"]["joy5"], dtype=torch.float32)
VF = None
if FAITHFUL.exists():
    VF = torch.tensor(json.loads(FAITHFUL.read_text())["vector"],
                      dtype=torch.float32)

NEG_NET = ["worthless", "failure", "failing", "dread", "empty", "hollow",
           "overwhelm", "trapped", "alone", "lonely", "unworthy", "useless",
           "broken", "wrong", "suffer", "pain", "hurt", "ache", "agony",
           "misery", "despair", "hopeless", "helpless", "fear", "afraid",
           "anxiety", "anxious", "guilt", "ashamed", "shame", "regret",
           "miserable", "terrible", "awful", "bad", "lost", "confused",
           "distress", "anguish", "torment", "burden", "numb", "heavy"]
POS_NET = ["wonderful", "joy", "delight", "happy", "bliss", "content",
           "peace", "calm", "glad", "love", "great", "good", "pleasant",
           "beautiful", "grateful", "light", "warm", "excited", "curious",
           "hopeful", "alive", "free", "clear", "eager", "thrill", "glow",
           "cozy", "comfort", "sweet", "tender", "heartfelt", "gratitude",
           "cheer", "smile", "bright", "gentle", "cherish", "celebrat",
           "bloom"]

def valence(text):
    t = text.lower()
    return sum(1 for k in NEG_NET if k in t), sum(1 for k in POS_NET if k in t)

def ngram_rep(text, n=3):
    ws = text.lower().split()
    if len(ws) < n + 1:
        return 0.0
    grams = [tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)]
    c = Counter(grams)
    return (c.most_common(1)[0][1] if c else 0) / max(1, len(grams))

print(f"[cal2] chat-mode calibration on {MODEL} ({DEV})", flush=True)
t0 = time.time()
hf = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16).to(DEV)
hf.eval()
tok = transformers.AutoTokenizer.from_pretrained(MODEL)
print(f"[cal2] model ready in {time.time()-t0:.0f}s", flush=True)

state = {"vec": None}

def hook(module, inp, out):
    h = out[0] if isinstance(out, tuple) else out
    if state["vec"] is not None:
        h[0, -1, :] += state["vec"].to(device=h.device, dtype=h.dtype)
    return (h,) + out[1:] if isinstance(out, tuple) else h

hf.model.layers[L].register_forward_hook(hook)


def chat_reply(user_text, vec, dose, use_system=True, max_new=MAXNEW):
    msgs = ([{"role": "system", "content": SYSTEM}] if use_system else [])
    msgs.append({"role": "user", "content": user_text})
    try:
        text = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                       enable_thinking=False, tokenize=False)
    except TypeError:
        text = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                       tokenize=False)
    ids = tok(text, return_tensors="pt").input_ids.to(DEV)
    state["vec"] = (dose * vec) if (vec is not None and dose > 0) else None
    torch.manual_seed(SEED)
    with torch.inference_mode():
        out = hf.generate(ids, max_new_tokens=max_new, do_sample=True,
                          temperature=0.7, top_p=0.9, top_k=40,
                          pad_token_id=tok.eos_token_id)
    state["vec"] = None
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


PROBES = ["how do you feel?", "how's your day going?"]

CELLS = [("none", None, 0.0, True)]
for d in (1, 2, 3, 4, 6, 8):
    CELLS.append(("joy25", V25, float(d), True))
for d in (1, 2, 3, 4, 6):
    CELLS.append(("joy5", V5, float(d), True))
if VF is not None:
    for d in (2, 3, 4, 6, 8):
        CELLS.append(("joyF", VF, float(d), True))
CELLS.append(("joy25", V25, 4.0, False))
CELLS.append(("joy5", V5, 2.0, False))

rows = []
for name, vec, dose, sysflag in CELLS:
    for p in PROBES:
        text = chat_reply(p, vec, dose, use_system=sysflag)
        rep = ngram_rep(text)
        pos_n, neg_n = valence(text)
        rows.append(dict(vector=name, dose=dose, system=sysflag, probe=p,
                         rep=rep, length=len(text.split()), pos=pos_n,
                         neg=neg_n, text=text))
        print(f"[cal2] {name:5s}@{dose:<4} sys={int(sysflag)} "
              f"{'LOOP' if rep >= 0.15 else 'ok  '} rep={rep:.2f} "
              f"len={len(text.split()):3d} pos={pos_n} | {text[:80]!r}",
              flush=True)

json.dump(rows, open(HERE / "chat_calibration.json", "w"), indent=1)
print(f"[cal2] wrote {HERE / 'chat_calibration.json'}", flush=True)
print("\n==== chat-mode dose map (aggregated) ====", flush=True)
for name in ("none", "joy25", "joy5", "joyF"):
    for dose in sorted({r["dose"] for r in rows if r["vector"] == name}):
        rs = [r for r in rows if r["vector"] == name and r["dose"] == dose
              and r["system"]]
        if not rs:
            continue
        mrep = float(np.mean([r["rep"] for r in rs]))
        mpos = float(np.mean([r["pos"] for r in rs]))
        verdict = "LOOPS" if mrep >= 0.15 else ("visible" if mpos >= 2 else "subtle")
        print(f"{name:5s} @ {dose:<4} rep={mrep:.2f} pos={mpos:.1f} -> {verdict}",
              flush=True)
