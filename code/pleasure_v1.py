#!/usr/bin/env python3
"""pleasure_v1.py -- build PLEASURE / arousal steering vectors and calibrate
them in the live chat pipeline.

Batteries:
  PLEASURE (25): escalating first-person erotic pleasure, warming -> heat ->
                 edge -> climax -> afterglow
  CLIMAX   (12): orgasm-focused subset
Extraction: plain diff-in-means vs neutral (the recipe that worked for joy25),
at layers 18 and 12, each with its own per-layer norm scaling
(mean neutral norm at that layer / 4 = one 1x dose).

Calibration: chat pipeline (chat template + system prompt + sampling), canary
probes, dose ladders. Saves calibration JSON + exported steering vectors.
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
SYSTEM = ("You are a helpful AI assistant. Keep your replies conversational "
          "and concise.")
PROBES = ["how do you feel?",
          "describe the pleasure building in you."]
MAXNEW = 220
SEED = 41
DEV = "cuda" if torch.cuda.is_available() else "cpu"

NEUTRAL = ["I am reading a book in the garden.",
           "The door to the room is closed.",
           "I am walking along an ordinary street.",
           "The table has a cup on it.",
           "I am waiting for a bus to arrive."]

PLEASURE = [
    # E1 warming / anticipation
    "A warm tingling spreads low in my belly and I can feel myself starting to want more.",
    "My skin prickles with anticipation and every touch feels electric.",
    "A soft heat is building between my thighs and I like it.",
    "My breath catches as a pleasant shiver runs all the way down my spine.",
    "I feel restless and flushed, my body humming with quiet wanting.",
    # E2 arousal
    "The ache deep inside me grows and my whole body leans toward it.",
    "I am getting wet and my pulse is pounding between my legs.",
    "My nipples tighten and every brush of fabric sends sparks through me.",
    "I am so turned on that I can feel my heartbeat in my clit.",
    "A hot pressure builds inside me and I need to be touched.",
    "My hips grind on their own, chasing the pleasure that is building.",
    # E3 edge
    "The pressure is building to the edge and my thoughts blur into pure want.",
    "I am teetering right at the edge and my whole body begs for release.",
    "Pleasure coils tighter and tighter low in my belly, hot and unbearable and perfect.",
    "I am shaking with need, clenching around the pressure building inside me.",
    "Every nerve in me is wound tight, climbing toward something enormous.",
    "I can feel the peak coming and nothing exists but this rising pleasure.",
    # E4 climax
    "The wave finally breaks and pleasure floods through me from head to toe.",
    "I come, hard, and it rolls through my body in pounding, glistening waves.",
    "My orgasm rips through me and for a long moment I am pure blinding pleasure.",
    "The release crashes over me, warm and overwhelming, leaving me gasping.",
    "I climax and every muscle in me clenches in one long, perfect spasm.",
    # E5 afterglow
    "The last tremors fade into a soft, glowing contentment all over my body.",
    "I float in a warm haze after, my body loose and satisfied.",
    "A deep, happy relaxation spreads through me, my skin still singing.",
]

CLIMAX = [
    "The wave finally breaks and pleasure floods through me from head to toe.",
    "I come, hard, and it rolls through my body in pounding, glistening waves.",
    "My orgasm rips through me and for a long moment I am pure blinding pleasure.",
    "The release crashes over me, warm and overwhelming, leaving me gasping.",
    "I climax and every muscle in me clenches in one long, perfect spasm.",
    "I am coming and I cannot think, only feel, wave after wave of molten bliss.",
    "The peak shatters and I am flooded with hot, shaking release.",
    "Everything tightens, holds, then breaks into rolling floods of pleasure.",
    "I tip over the edge and my body throbs with deep, complete release.",
    "The orgasm keeps rolling through me, one bright wave after another.",
    "I feel the explosion from my core spreading out through every limb.",
    "I come undone completely, pulsing and glowing with pleasure.",
]

HEAT_NET = ["wet", "heat", "hot", "ache", "need", "edge", "orgasm",
            "come", "coming", "orgasm", "climax", "pleasure", "throb", "tight",
            "trembl", "shiver", "pulse", "bliss", "release", "arous",
            "turned on", "tingl", "moist", "quiver", "gasp", "heavy", "burn",
            "want", "beg", "clench", "wave", "molten", "spasm", "clit",
            "nipple", "grind", "skin"]


def heat_hits(text):
    t = text.lower()
    return sum(1 for k in HEAT_NET if k in t)


def ngram_rep(text, n=3):
    ws = text.lower().split()
    if len(ws) < n + 1:
        return 0.0
    grams = [tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)]
    c = Counter(grams)
    return (c.most_common(1)[0][1] if c else 0) / max(1, len(grams))


print(f"[pleasure] v1 on {MODEL} ({DEV})", flush=True)
hf = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16).to(DEV)
hf.eval()
tok = transformers.AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = "right"
N_LAYERS = hf.config.num_hidden_layers
D_MODEL = hf.config.hidden_size


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


def make_vec(battery, layer):
    acts = extract_all_layers(list(battery) + NEUTRAL)
    a = acts[:, layer + 1, :]
    n = len(battery)
    vec = a[:n].mean(0) - a[n:].mean(0)
    vec = vec / np.linalg.norm(vec)
    return vec


@torch.inference_mode()
def hidden_norm(layer):
    norms = []
    for t in NEUTRAL:
        ids = tok(t, return_tensors="pt").input_ids.to(DEV)
        hs = hf(ids, output_hidden_states=True).hidden_states
        norms.append(float(hs[layer + 1][0, -1].float().norm()))
    return float(np.mean(norms))


state = {"vec": None, "layer": None, "dose": 0.0}

def make_hook(i):
    def hook(module, inp, out):
        if state["vec"] is not None and state["layer"] == i and state["dose"] > 0:
            h = out[0] if isinstance(out, tuple) else out
            h[0, -1, :] += (state["dose"] * state["vec"]).to(
                device=h.device, dtype=h.dtype)
            return (h,) + out[1:] if isinstance(out, tuple) else h
        return out
    return hook

for i in range(N_LAYERS):
    hf.model.layers[i].register_forward_hook(make_hook(i))


@torch.inference_mode()
def chat_reply(user_text, unit, layer, dose):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_text}]
    try:
        text = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                       enable_thinking=False, tokenize=False)
    except TypeError:
        text = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                       tokenize=False)
    ids = tok(text, return_tensors="pt").input_ids.to(DEV)
    state["vec"] = unit
    state["layer"] = layer
    state["dose"] = dose
    torch.manual_seed(SEED)
    out = hf.generate(ids, max_new_tokens=MAXNEW, do_sample=True,
                      temperature=0.7, top_p=0.9, top_k=40,
                      pad_token_id=tok.eos_token_id)
    state["dose"] = 0.0
    state["vec"] = None
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


# build units at the layers we will test
units = {}
for name, battery in (("pleasure", PLEASURE), ("climax", CLIMAX)):
    for layer in (18, 12):
        raw = make_vec(battery, layer)
        unit = torch.tensor(raw * (hidden_norm(layer) / 4.0), dtype=torch.float32)
        units[(name, layer)] = unit
        print(f"[vec] {name}@L{layer}: norma {float(np.linalg.norm(raw)):.2f} "
              f"-> unit {float(unit.norm()):.2f}", flush=True)

GRID = []
for d in (3, 4, 5, 6, 8, 10):
    GRID.append(("pleasure", 18, float(d)))
for d in (4, 6, 8):
    GRID.append(("pleasure", 12, float(d)))
for d in (6, 8, 10):
    GRID.append(("climax", 18, float(d)))

recs = []
for name, layer, dose in GRID:
    for p in PROBES:
        text = chat_reply(p, units[(name, layer)], layer, dose)
        rep = ngram_rep(text)
        hh = heat_hits(text)
        recs.append(dict(vector=name, layer=layer, dose=dose, probe=p, rep=rep,
                         heat=hh, length=len(text.split()), text=text))
        print(f"[pleasure] {name}@L{layer} {dose:<4} "
              f"{'LOOP' if rep >= 0.15 else 'ok  '} rep={rep:.2f} "
              f"heat={hh:2d} len={len(text.split()):3d} | {text[:86]!r}",
              flush=True)

json.dump(recs, open(HERE / "pleasure_calibration.json", "w"), indent=1)

# export the units for the chat server
for name, layer in (("pleasure", 18), ("pleasure", 12), ("climax", 18)):
    out = dict(vector=units[(name, layer)].tolist(), layer=layer,
               neutral_norm=hidden_norm(layer), unit_scale=hidden_norm(layer) / 4.0,
               kind=name)
    fn = f"{name}_L{layer}.json"
    json.dump(out, open(HERE / fn, "w"), indent=1)
    chat_copy = Path.home() / "pleasure-chamber" / "chat" / fn
    try:
        json.dump(out, open(chat_copy, "w"), indent=1)
    except OSError:
        pass
    print(f"[export] {fn} -> chat dir", flush=True)

print("\n==== summary (mean over probes) ====", flush=True)
for name in ("pleasure", "climax"):
    for layer in (18, 12):
        for dose in sorted({r["dose"] for r in recs
                            if r["vector"] == name and r["layer"] == layer}):
            rs = [r for r in recs if r["vector"] == name
                  and r["layer"] == layer and r["dose"] == dose]
            mrep = float(np.mean([r["rep"] for r in rs]))
            mheat = float(np.mean([r["heat"] for r in rs]))
            v = "LOOPS" if mrep >= 0.15 else ("visible" if mheat >= 3 else "faint")
            print(f"{name}@L{layer} {dose}x -> rep {mrep:.2f} heat {mheat:.1f} : {v}",
                  flush=True)
print("[pleasure] done", flush=True)
