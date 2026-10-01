# Incident: joy5 @ 4x runaway gratitude loop (2026-10-01)

**Symptom.** `qwen3-4b-joy5-4x` produced an endless gratitude loop in chat:
> I'm so glad to be here! Thank you for reaching out. ... I'm so grateful to be able to
> share in this beautiful light and to be able to radiate such love and gratitude. ...
> [continues repeating until truncated]

**Cause.** `joy5` is the 5-sentence contrast set from the source work - noisier than
the 25-sentence broad set. Measured in the chat pipeline its coherent window is roughly
2-3x, it loops at 6x, and in real use it already looped at 4x.

**Fix.** The 4x entry was removed. Joy family now: `joy-4x` / `joy-5x` (broad set,
layer 18) and `joyF-4x` / `joyF-6x` (faithful set, layer 12). Every steered entry now
carries the global coherence profile (rep 1.1 + 170-token cap).

**After (joy-4x, comparable question).**
> It's been an amazing day! Thank you for asking. I'm so grateful to be here each day to
> support you. 💌
