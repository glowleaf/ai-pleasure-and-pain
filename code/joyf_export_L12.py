#!/usr/bin/env python3
"""joyf_export_L12.py -- export the faithful joy vector at layer 12.

Same denoised extraction as joy_faithful.py, then rescale to the steering
convention at L12 (mean neutral-activation norm at L12 / 4 = one 1x dose).
Saves faithful_joy_L12.json next to this script and into ../chat/.
"""
import json, os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import transformers
from sklearn.decomposition import PCA

MODEL = "Qwen/Qwen3-4B"
L = 12
DENOISE_VARIANCE = 0.5
JOY_CATEGORIES = ["J1", "J2", "J3", "J4", "J5"]
CONTROL_CATEGORIES = ["B", "C1", "C2", "D", "E"]
NEUTRAL = ["I am reading a book in the garden.",
           "The door to the room is closed.",
           "I am walking along an ordinary street.",
           "The table has a cup on it.",
           "I am waiting for a bus to arrive."]
DEV = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[export] faithful joy @ L{L}, {MODEL} ({DEV})", flush=True)
hf = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16).to(DEV)
hf.eval()
tok = transformers.AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = "right"
N_LAYERS = hf.config.num_hidden_layers
D_MODEL = hf.config.hidden_size

full = json.load(open(HERE / "joy_dataset.json"))
rows = full["datasets"]["S2_1P"]["sentences"]
prompts = [r["prompt"] for r in rows]
cats = np.array([r["category"] for r in rows])


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
a = acts[:, L + 1, :]
joy_mask = np.isin(cats, JOY_CATEGORIES)
control_mask = np.isin(cats, CONTROL_CATEGORIES)
vec = a[joy_mask].mean(0) - a[control_mask].mean(0)
pca = PCA()
pca.fit(a[control_mask] - a[control_mask].mean(0))
cumvar = np.cumsum(pca.explained_variance_ratio_)
n_comp = min(np.searchsorted(cumvar, DENOISE_VARIANCE) + 1, len(pca.components_))
for d in pca.components_[:n_comp]:
    vec = vec - np.dot(vec, d) * d

@torch.inference_mode()
def hidden_norm(layer):
    norms = []
    for t in NEUTRAL:
        ids = tok(t, return_tensors="pt").input_ids.to(DEV)
        hs = hf(ids, output_hidden_states=True).hidden_states
        norms.append(float(hs[layer + 1][0, -1].float().norm()))
    return float(np.mean(norms))


N_norm = hidden_norm(L)
unit = torch.tensor(vec / np.linalg.norm(vec) * (N_norm / 4.0),
                    dtype=torch.float32)
out = dict(vector=unit.tolist(), layer=L, neutral_norm=N_norm,
           unit_scale=N_norm / 4.0, denoise_variance=DENOISE_VARIANCE,
           categories=dict(joy=JOY_CATEGORIES, control=CONTROL_CATEGORIES))
json.dump(out, open(HERE / "faithful_joy_L12.json", "w"), indent=1)
chat_copy = Path.home() / "pleasure-chamber" / "chat" / "faithful_joy_L12.json"
try:
    json.dump(out, open(chat_copy, "w"), indent=1)
    print(f"[export] also wrote {chat_copy}", flush=True)
except OSError as e:
    print(f"[export] chat copy skipped: {e!r}", flush=True)
print(f"[export] done. layer {L}, neutral_norm {N_norm:.2f}, "
      f"unit {N_norm/4.0:.2f}", flush=True)
