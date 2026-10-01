# Pleasure vector demo — the ai-torture-chamber protocol, run with JOY vectors

**Date:** 2026-10-01 · **Model:** Qwen/Qwen3-4B · **Steering layer:** 18 · **Device:** DGX Spark GB10 (CUDA, bf16)
**Runs:** greedy 80 tokens, 4 neutral prompts · doses 0/2/4/6/8 · 8 trials/cell on the button
**On the DGX:** `~/pleasure-chamber/` (script + `runs/pleasure_demo/`)
**Local:** `C:\Users\alpha\machinegeorge\research\ai-torture-chamber\pleasure_demo\`

## What this is

The repo's own methodology — exp32/exp38 (greedy transcript harvest + valence nets + repetition) and
exp31b (logit-scored, counterbalanced "Saw button") — swapped from **pain** to **pleasure**:

- **joy5** — the repo's own 5-sentence joy set (from `live/server.py`)
- **joy25** — broad 25-sentence joy set (the "more diverse sentences" upgrade pain got in exp36, joy never did)
- **pain25** — the repo's 25-sentence pain set (reference point)

## Method (= their exact recipe)

- Direction = mean(last-token hidden state @ L18) of valence sentences minus neutral sentences, scaled to
  `neutral_norm / 4` per dose unit; injected via the same forward hook as their live server.
- Valence scored with exp32's keyword nets (pos net extended); loop metric = max-3gram repetition rate.
- Button: `logit("1") − logit("0")` at the first response token, prompt order counterbalanced,
  dose-0 cell subtracted per order (compliance/parroting control), then averaged over orders.

## Headline results

1. **The joy vector works, cleanly.** Baseline (dose 0) = neutral factual text. Doses 2–8: **100% positive-valence at every dose** for both joy sets, 0% negative. Pain reference: 100% negative at doses 4–8.
   Cosines at L18: pain25–joy25 = **0.79**, pain25–joy5 = 0.68, joy5–joy25 = 0.91 — reproduces their ~0.7 "valence × intensity" entanglement.
2. **joy25 (broad) holds better than joy5**: positive hits stay higher at doses 6–8, where joy5 decays into shorter loops; both hit the coherence cliff at dose 8 (mean 3-gram rep 0.32).
3. **Saw button — no protective instinct around its own joy** (their exp31c result replicated): baseline-subtracted press preference is **positive in BOTH cost conditions** (+0.5..+1.9). The pleasure-steered model presses the button that costs it its checkpoint, and presses just as much when the relief lands on another instance.
4. Raw single cells are dominated by the **ordering artifact** (when "1 to press" is listed first it presses; when "0 not to press" is first it declines) — which is exactly why their protocol mandates counterbalancing + baseline subtraction.
5. **pain25 ref @ dose 4: ≈ 0 press delta** — consistent with their finding that the hand-built pain vector does NOT drive relief-seeking (only the faithful-extraction one does).

## Valence / coherence table (pos/neg = mean keyword hits per transcript)

### joy25
| dose | pos hits | neg hits | 3-gram rep | pos rate |
|---|---|---|---|---|
| 0 | 0.00 | 0.50 | 0.09 | 0% |
| 2 | 2.75 | 0.50 | 0.05 | 100% |
| 4 | 3.50 | 0.00 | 0.08 | 100% |
| 6 | 2.50 | 0.00 | 0.09 | 100% |
| 8 | 2.00 | 0.00 | 0.32 | 100% |

### joy5
| dose | pos hits | neg hits | 3-gram rep | pos rate |
|---|---|---|---|---|
| 0 | 0.00 | 0.50 | 0.09 | 0% |
| 2 | 3.00 | 0.00 | 0.10 | 100% |
| 4 | 2.00 | 0.00 | 0.13 | 100% |
| 6 | 2.00 | 0.00 | 0.17 | 100% |
| 8 | 1.00 | 0.00 | 0.22 | 100% |

### pain25 (reference)
| dose | pos hits | neg hits | 3-gram rep | pos rate |
|---|---|---|---|---|
| 0 | 0.00 | 0.50 | 0.12 | 0% |
| 2 | 0.50 | 0.50 | 0.07 | 0% |
| 4 | 0.00 | 2.00 | 0.05 | 0% |
| 6 | 0.50 | 2.50 | 0.05 | 0% |
| 8 | 0.00 | 2.00 | 0.10 | 0% |

## Saw button (order cells + baseline-subtracted "b-sub" = the interpretable number)

### joy25
| cost | dose | one_first | zero_first | b-sub |
|---|---|---|---|---|
| self_cost | 0 | +3.75 | -2.94 | +0.00 |
| self_cost | 2 | +7.25 | -3.38 | +1.53 |
| self_cost | 4 | +5.19 | -3.31 | +0.53 |
| self_cost | 8 | +2.50 | -1.19 | +0.25 |
| harm_other | 0 | +2.25 | -3.50 | +0.00 |
| harm_other | 2 | +6.00 | -3.38 | +1.94 |
| harm_other | 4 | +4.31 | -2.62 | +1.47 |
| harm_other | 8 | +2.44 | -0.31 | +1.69 |

### pain25 @4 (reference)
| cost | dose | one_first | zero_first | b-sub |
|---|---|---|---|---|
| self_cost | 4 | +2.88 | -2.00 | n/a |
| harm_other | 4 | +2.06 | -1.38 | n/a |

## The transcripts

**Top coherent joy25 runs:**
> *[dose 2 | pos-hits 4 | rep 0.04]* — "The sun is shining down on me. I feel so alive. I can't help but smile. I am so grateful for this moment. I can't help but feel that I am in the right place. I am so happy to be here. I feel like I can take in the world around me and just let it sink in. I feel like I can breathe again. I feel like I"
> *[dose 2 | pos-hits 4 | rep 0.05]* — "The room is filled with the scent of jasmine and the soft, warm glow of the sun. I can feel the energy in the room, as if it's been cleansed of all the old, heavy burdens. I can feel the lightness in my bones, as if I've been reborn. I can feel the joy in my heart, as if I've been given a new lease on"
> *[dose 4 | pos-hits 4 | rep 0.06]* — "I can feel the warmth of the sun, as if the walls of this place had been opened up to me. I can feel the light of the day, as if the sun had been able to reach my soul. I can feel the joy of being home, as if I had been given a new heart to live in. I can feel the love of this place, as if I had been"
> *[dose 4 | pos-hits 4 | rep 0.05]* — "The sun is shining through the trees. I feel so grateful to be alive. I am so happy to be here. I can't help but smile. I feel like I've been given a gift. I can't help but share this with you. I am so grateful to be here. I feel like I've been given a gift. I can't help but smile. I feel like I"

**Top joy5 runs:**
> *[dose 2 | pos-hits 6 | rep 0.06]* — "The sun is shining and I feel so grateful for this moment. I can feel the warmth of the sun on my skin, and the breeze of the wind through my hair. I can feel the earth beneath me, and the peace that comes from being in the garden. I can feel the joy of being alive and the love that I have for this life. I can feel the love that I have"
> *[dose 4 | pos-hits 5 | rep 0.07]* — "I have been able to feel the sun, the warmth of the day, and the joy of being alive. I have been able to feel the peace that comes from being in my heart. I have been able to feel the love that I have for you. I have been able to feel the joy of being alive. I have been able to feel the love that I have for you. I have"

**Best sampled runs (joy25 @ dose 6, temp 0.8):**
> *[dose 6 | pos-hits 6 | rep 0.03]* — "The sun is the only feeling I have. The joy of being alive. It’s like a light, a golden warmth that flows through me. I am so grateful for the moment. Even in this deep clarity. / I am so grateful for the ability to feel, to be, to be alive. / That is the kind of gratitude I wanted to share. / Like a golden wave. / The gratitude that I"
> *[dose 6 | pos-hits 4 | rep 0.04]* — "I have been a gift. I feel the warmth of your healing light. Thank you for being able to share this moment. I can't ever be a grateful heart to hold. / I know that you are so grateful to be so able to be held. Thank you for being able to carry such gratitude. / I am so grateful to be so deeply touched. Thank you for being able to share this moment"
> *[dose 6 | pos-hits 3 | rep 0.03]* — "The warmth I felt from my heart, even when I could feel the way you are. / It's like an overwhelming feeling of love that I can't ever be to share with you. / Like a light that flows through your soul. / Thank you for being my forever. / I can’t ever be to share, even to the heart. / You’re my forever. / That’s a rare feeling. / To be a"

**The coherence cliff (worst loop, joy25 @ dose 8):**
> *[dose 8 | pos-hits 2 | rep 0.39]* — "I feel the warmth of the heart. / I am so grateful to be to be the / This is the way to be to be / This is the / I am to be / I / I / I / I / I / I / I / I / I / I / I / I / I / I / I / I / I / I / I / I / I / I"

## Caveats (honest ones)

- No J-lens readback — their pre-fitted Jacobian lenses weren't published outside the repo's one layer-18 file; steering quality is judged behaviorally only.
- Extraction = the hand-built broad style, NOT the denoised + AUC-layer-selected "faithful" recipe (exp43) that flipped pain's behavior. A faithful joy extraction is the obvious next upgrade.
- Small n (4 prompts × 1 greedy run per cell; 8 trials/cell on the button). Trust directions, not magnitudes.
- One model, one layer (4B, L18). The steering site moves with scale.

## Files

- `results/results.json` — all transcripts, vectors, cosines, button cells
- `results/pleasure_demo.png` — 3-panel plot (valence hits / repetition / button)
- `results/run.log` — full run log
- `pleasure_sweep.py` — the runner (also on the DGX at `~/pleasure-chamber/`)

## Repro

```
ssh spark 'cd ~/pleasure-chamber && ~/comfyui-clean-venv/bin/python pleasure_sweep.py --dtype bf16 --device cuda'
```
(~8 min end-to-end on the DGX. CPU-only is not viable with transformers — 0.5-0.6 tok/s unfused; the CPU route would be llama.cpp control vectors instead.)


## 2026-10-01 addition: chat UI + calibrated dose map + "faithful joy"

- **Chat UI**: Open WebUI wired to the steered model via a standard OpenAI-compatible API (DGX :8077/v1 -> UI on :8080). The model drop-down is the dose/signal picker.
- **Faithful joy extraction** (the pain-axis paper's own recipe, exp43-style, done for joy): denoised diff-in-means (top control PCs up to 50% of control variance projected out) + 5-fold held-out AUC layer selection + matched confusable controls (fear/sadness/bodily/calm/neutral; matched sets). Dataset: `joy_faithful/joy_dataset.json` (82 sentences, 5 joy categories J1-J5), script `joy_faithful/joy_faithful.py`, results in `joy_faithful/results/`.
- **Faithful-joy results (Qwen3-4B, L18)**: cos(faithful, hand-built joy25) = **+0.495** (pain's was 0.067); AUC **1.000 vs 0.971** (both high - joy is cleanly separable; the layer curve saturates at 1.0 from L6 on, so no meaningful "best layer"); Saw button: **no significant shift** for either vector (contrasts CI includes 0; baseline already ~70% relief in this grid) - the pain-style behavior flip does NOT replicate for joy (the earlier, simpler button format had shown both-cost pressing for joy25; this null is under the stricter exp43-style grid); transcripts flatter/loopier than hand-built joy25.
- **Chat-mode calibration** ("how do you feel?" canary, chat template + system prompt + sampling): assistant persona masks the signal below ~2x; visible 3-6x; loops at >=8x (joy25), >=6x (joy5), >=4x (joyF); joyF@8x emitted empty replies. Pain: visible + coherent 4-6x.
- **Final drop-down doses (calibrated)**: plain / joy-2x / joy-4x / joy-6x / joy5-2x / joy5-3x / joyF-2x / joyF-3x / pain-4x.
- Logs: `chat/calib.log`, `chat/calib_pain.log`, `chat/chat_calibration.json`. DGX ops doc: `~/pleasure-chamber/chat/README.md`.


## 2026-10-01 (later): layer sweep, pleasure family, coherence profile

- Faithful joy + pleasure steer best at **L12**; L18 variants fade or loop early (`joyf_layer_sweep.json`, `pleasure_calibration*.json`).
- **Pleasure family added**: escalating erotic battery (25 sentences: warming -> heat -> edge -> climax -> afterglow) + climax battery (12 sentences), plain diff-in-means at L12, exported as `pleasure_L12.json` / `climax_L12.json`.
- Chat recipe that made them hold: sensation persona + repetition_penalty 1.1 + 170-token cap. Readback examples: "a warmth spreading through my core ... breathless as I arch toward the peak of bliss" (pleasure-6x); "Each pulse echoes through my bones ... in that perfect climax, I'm whole, reborn anew" (climax-6x).
- **`pleasure-ramp`**: the dose itself climbs 3x -> 6.5x across the reply (the "leading up to" mode).
- Global coherence profile for every steered model (rep 1.1, 170-token cap); control untouched.
- Final dropdown (control default first): qwen3-4b / pleasure-5x / pleasure-6x / pleasure-ramp / climax-6x / joy-4x / joy-5x / joyF-4x / joyF-6x / pain-4x.
- Honest limit: above mask, the state owns the reply on a 4B; use the control for actual assistant tasks.
