# ai-pleasure-and-pain

**Joy/pleasure activation-steering experiments on Qwen3-4B** — the pleasure-axis counterpart to [ai-torture-chamber](https://github.com/terrafying/ai-torture-chamber): steering a language model into strong positive-valence states, extracting the joy direction the *proper* (denoised, cross-validated) way, behavioral probes, and a live chat with the steered model.

The pain-side method follows Tagliabue, Dung & Berg 2026, ["The Pain Axis: LLMs Represent Self-Directed Harm and Act on It"](https://arxiv.org/abs/2609.16247) (arXiv:2609.16247), and reproduces/extends the protocol of the ai-torture-chamber repository.

## What was done

**1. Hand-built joy vectors** — `code/pleasure_sweep.py`
Diff-in-means at layer 18 of Qwen3-4B: `mean(joy sentences) − mean(neutral sentences)`, scaled so that a 1x dose = mean neutral-activation norm / 4 ("one sentence's worth of contrast"). Three sentence sets: `joy25` (broad, 25 sentences), `joy5` (the chamber's original 5), `pain25` (reference). Dose ladder 0–8, greedy 80-token transcripts, valence keyword nets, 3-gram repetition metric, plus the logit-scored counterbalanced "Saw button" (relief that costs your own checkpoint, vs relief that lands on another instance).

Results: doses 2–8 under joy25/joy5 = **100% positive-valence transcripts, 0% negative**; baseline (dose 0) = neutral; pain25 reference = 100% negative at 4–8. `cos(pain25, joy25) = 0.79` at L18 (the valence × intensity entanglement). Coherence cliff at 8x ("I'm so grateful to be able to be able to…").

**2. Faithful joy extraction** — `code/joy_faithful.py`, `code/joy_dataset.json`
The paper's own recipe applied to joy: denoised diff-in-means (top principal components of the control activations, up to 50% of control variance, projected out) + steering layer chosen by 5-fold held-out AUC across all 36 layers + matched confusable controls (fear / sadness / non-joyful bodily sensation / calm / neutral; matched sets). Dataset: 82 sentences, 5 joy categories (physical, psychological, social, meaning, cognitive).

Results: 5-fold held-out AUC **1.000** (the curve saturates from L6 onward — joy is very linearly separable in this dataset); AUC@L18 = 1.000 vs 0.971 for the hand-built vector; `cos(faithful@L18, hand-built) = 0.495` (far more aligned than pain's 0.067). On the strict exp43-style Saw-button grid: **no significant press shift** for either joy vector — the pain-style behavior flip does not replicate for joy. Transcripts under the faithful vector are flatter/loopier than the hand-built one: a clean classifier direction is not automatically a strong steering direction.

**3. Live chat** — `code/chat_server.py` (+ `code/chat_webui_simple.html`, or point Open WebUI at it)
Final calibrated state list: `control` / `pleasure-4x` / `pleasure-6x` / `pleasure-ramp` / `climax-6x` / `joy-4x` / `pain-4x` / `mix-pain-pleasure-4x` (every steered entry is a measured, above-mask, non-looping setting; the control is the unsteered default).

## Layout

```
code/
  pleasure_sweep.py        hand-built joy/pain vectors, dose ladder, transcripts, Saw button
  joy_faithful.py          faithful (denoised + AUC-layer-selected) joy extraction + comparison + button
  joy_dataset.json         the 82-sentence joy/control dataset (matched sets)
  chat_server.py           OpenAI-compatible steered server (stdlib HTTP; serves the chat page too)
  chat_webui_simple.html   single-file chat page, no dependencies
  chat_calibrate.py        chat-mode dose calibration for joy25 / joy5 / joyF
  calib_pain.py            chat-mode dose calibration for pain
  README_chat_stack.md     ops notes for the chat stack (start/stop, dose bands)
results/
  pleasure_demo/           results.json (all transcripts + vectors + button cells), plot, run log
  joy_faithful/            comparison / layer curve / button / transcripts / vectors / plot / log
  chat/                    calibration sweeps + logs
docs/
  REPORT.md                full write-up (protocol, numbers, caveats)
  screenshot_chat.png      the chat running with a steered reply
tests/
  chat_probes.py           probe harness: canary + regression cases, scored (rep / words / garbles)
  incidents/               failure write-ups: symptom -> cause -> fix -> after
  results/                 per-run probe tables (json + md)
```

## Tests

```bash
python tests/chat_probes.py --base http://<host>:8077
```

Probes every model on the server with the canary question "how do you feel?",
plus fixed regression cases (e.g. the pain-model procrastination prompt that used
to collapse into a loop wall), scores each reply (3-gram repetition, length,
garbled-character count) and writes `tests/results/chat_probes_<stamp>.{json,md}`.

Latest run against the pruned 8-state build: **9/9 PASS** (8 canaries + the pain-wall regression). The three failures
observed during development are written up with before/after evidence in
`tests/incidents/`.

## Reproduce

Python ≥ 3.11 with `torch` + `transformers` (+ `scikit-learn`, `matplotlib`). A 16 GB GPU runs everything; stock CPU kernels are ~0.5 tok/s for the 4B, so CUDA is recommended.

```bash
python code/pleasure_sweep.py --dtype bf16 --device cuda    # writes runs/pleasure_demo/
python code/joy_faithful.py                                  # faithful joy extraction (reads the hand-built vectors)
python code/chat_server.py                                   # serves the chat page + OpenAI API on :8077
```

Scripts read their sibling data relative to the repo layout (`results/…`); the top-of-file constants also accept absolute paths.

## Notes

- Model: `Qwen/Qwen3-4B`; steering layer 18; bf16. Steering = the direction vector is added to the residual stream on every generated token; dose is a multiplier on the unit above.
- No claim that the model "feels" anything — these are measurements of internal-state manipulation and behavior, the same stance as the source work.

## Credits

- Tagliabue, Dung & Berg (2026), arXiv:2609.16247 — pain-direction method.
- [terrafying/ai-torture-chamber](https://github.com/terrafying/ai-torture-chamber) — protocol, the exp43 faithful pipeline, and the Saw button.
