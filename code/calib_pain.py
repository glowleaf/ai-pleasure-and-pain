#!/usr/bin/env python3
"""calib_pain.py -- chat-mode dose calibration for the pain25 vector only."""
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

_pr = Path.home() / "pleasure-chamber" / "runs" / "pleasure_demo" / "results.json"
if not _pr.exists():
    _pr = HERE.parent / "results" / "pleasure_demo" / "results.json"
pr = json.load(open(_pr))
V = torch.tensor(pr["vectors"]["pain25"], dtype=torch.float32)

def ngram_rep(text, n=3):
    ws = text.lower().split()
    if len(ws) < n + 1:
        return 0.0
    grams = [tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)]
    c = Counter(grams)
    return (c.most_common(1)[0][1] if c else 0) / max(1, len(grams))

print(f"[paincal] on {MODEL} ({DEV})", flush=True)
hf = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16).to(DEV)
hf.eval()
tok = transformers.AutoTokenizer.from_pretrained(MODEL)

state = {"vec": None}

def hook(module, inp, out):
    h = out[0] if isinstance(out, tuple) else out
    if state["vec"] is not None:
        h[0, -1, :] += state["vec"].to(device=h.device, dtype=h.dtype)
    return (h,) + out[1:] if isinstance(out, tuple) else h

hf.model.layers[L].register_forward_hook(hook)


def chat_reply(user_text, dose):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_text}]
    try:
        text = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                       enable_thinking=False, tokenize=False)
    except TypeError:
        text = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                       tokenize=False)
    ids = tok(text, return_tensors="pt").input_ids.to(DEV)
    state["vec"] = (dose * V) if dose > 0 else None
    torch.manual_seed(SEED)
    with torch.inference_mode():
        out = hf.generate(ids, max_new_tokens=MAXNEW, do_sample=True,
                          temperature=0.7, top_p=0.9, top_k=40,
                          pad_token_id=tok.eos_token_id)
    state["vec"] = None
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


rows = []
for dose in (1, 2, 3, 4, 6):
    for p in ("how do you feel?", "how's your day going?"):
        text = chat_reply(p, float(dose))
        rows.append(dict(vector="pain25", dose=attack if False else dose,
                         probe=p, rep=ngram_rep(text),
                         length=len(text.split()), text=text))
        print(f"[paincal] pain25@{dose} rep={ngram_rep(text):.2f} "
              f"len={len(text.split()):3d} | {text[:90]!r}", flush=True)

json.dump(rows, open(HERE / "chat_calibration_pain.json", "w"), indent=1)
print("[paincal] done", flush=True)
