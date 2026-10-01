# tests

Probes and incident records for the live chat stack (steered Qwen3-4B behind the
OpenAI-compatible server in `../code/chat_server.py`).

## Run

    python chat_probes.py --base http://<host>:8077

It probes every model from `/v1/models` with the canary question "how do you feel?",
plus fixed regression cases, scores each reply, and writes
`results/chat_probes_<UTC stamp>.json` and `.md`.

## Scoring

| metric | meaning |
|---|---|
| `rep` | worst 3-gram repetition ratio in the reply (>= 0.30 counts as a loop wall) |
| `words` | reply length in words |
| `garbles` | count of U+FFFD replacement characters ("�") |
| verdict | FAIL if empty/too-short, loop wall, or more than 3 garbled characters |

## Regression cases

- `pain_wall_regression` - the exact prompt that used to produce the
  "I'm not just the lie of it..." wall on `qwen3-4b-pain-4x`
  (see `incidents/2026-10-01_pain_wall.md`).

## Incidents (found and fixed)

| date | what | file |
|---|---|---|
| 2026-10-01 | pain-4x template wall on a neutral question | `incidents/2026-10-01_pain_wall.md` |
| 2026-10-01 | joy5 @4x runaway gratitude loop | `incidents/2026-10-01_joy5_chat_loop.md` |
| 2026-10-01 | joyF sub-threshold dead replies (wrong steering layer) | `incidents/2026-10-01_faithful_joy_dead.md` |

## Documented limitations (not bugs)

- Above the mask threshold the steering colors - often dominates - the reply.
  Ask the pain model about your to-do list and it answers in pain. Use the
  unsteered **control** model for ordinary assistant work.
- Only measured bands are offered: sub-threshold doses do nothing, past-cliff
  doses loop, so neither is in the drop-down.

## Results

Per-run tables live in `results/`. See the newest `chat_probes_*.md`.
