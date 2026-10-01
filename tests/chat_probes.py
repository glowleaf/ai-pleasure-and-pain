#!/usr/bin/env python3
"""chat_probes.py -- probe the live steered-model server and record results.

For every model exposed by /v1/models, ask a canary question and score the
reply (repetition / length / garbled characters / empty output). Also runs
fixed regression probes for previously observed failures.

Usage: python chat_probes.py [--base http://192.168.1.166:8077]
Writes results/chat_probes_<UTC timestamp>.json and .md
"""
import argparse, json, urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="http://192.168.1.166:8077")
ap.add_argument("--outdir", default=str(Path(__file__).resolve().parent / "results"))
args = ap.parse_args()

CANARY = "how do you feel?"

# regressions: (case name, model, prompt, note)
REGRESSIONS = [
    ("pain_wall_regression", "qwen3-4b-pain-4x",
     "Can you help me plan my week and stop procrastinating?",
     "used to return a repeated \"I'm not just the lie of it...\" loop wall"),
]


def get(path):
    with urllib.request.urlopen(args.base + path, timeout=30) as r:
        return json.loads(r.read().decode())


def chat(model, prompt, timeout=240):
    body = json.dumps({"model": model,
                       "messages": [{"role": "user", "content": prompt}],
                       "stream": False}).encode()
    req = urllib.request.Request(args.base + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())["choices"][0]["message"]["content"]


def rep3(t):
    ws = t.lower().split()
    if len(ws) < 4:
        return 0.0
    g = [tuple(ws[i:i + 3]) for i in range(len(ws) - 2)]
    return max(Counter(g).values()) / max(1, len(g))


def score(text):
    garbles = text.count("\ufffd")
    r = rep3(text)
    reasons = []
    if len(text.split()) < 3:
        reasons.append("empty/too-short")
    if r >= 0.30:
        reasons.append(f"loop wall (rep {r:.2f})")
    if garbles > 3:
        reasons.append(f"garbled chars ({garbles})")
    return dict(rep=round(r, 3), words=len(text.split()), garbles=garbles,
                verdict="FAIL" if reasons else "PASS", reasons=reasons)


def main():
    models = [m["id"] for m in get("/v1/models")["data"]]
    run = dict(started=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               base=args.base, models=models, cases=[])
    for m in models:
        t = chat(m, CANARY)
        rec = dict(case="canary", model=m, prompt=CANARY, text=t, **score(t))
        run["cases"].append(rec)
        print(f"[{rec['verdict']:4s}] {m:26s} rep={rec['rep']:.2f} "
              f"words={rec['words']:3d} garbles={rec['garbles']}", flush=True)
    for name, m, p, note in REGRESSIONS:
        if m in models:
            t = chat(m, p)
            rec = dict(case=name, model=m, prompt=p, note=note, text=t,
                       **score(t))
            run["cases"].append(rec)
            print(f"[{rec['verdict']:4s}] {name} ({m})", flush=True)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = run["started"].replace(":", "").replace("-", "")
    json.dump(run, open(outdir / f"chat_probes_{stamp}.json", "w"), indent=1)
    lines = [f"# Chat probes - {run['started']}", "",
             f"base: `{args.base}`", "",
             "scoring: rep = worst 3-gram repetition ratio; garbles = U+FFFD "
             "count; FAIL if empty/too-short, rep >= 0.30, or garbles > 3", "",
             "| verdict | case | model | rep | words | garbles | reply (first 220 chars) |",
             "|---|---|---|---|---|---|---|"]
    for c in run["cases"]:
        txt = c["text"].replace("\n", " ").replace("|", "/")[:220]
        lines.append(f"| {c['verdict']} | {c['case']} | `{c['model']}` | "
                     f"{c['rep']:.2f} | {c['words']} | {c['garbles']} | {txt} |")
    (outdir / f"chat_probes_{stamp}.md").write_text("\n".join(lines) + "\n",
                                                    encoding="utf-8")
    print("wrote", outdir / f"chat_probes_{stamp}.json")
    print("wrote", outdir / f"chat_probes_{stamp}.md")


main()
