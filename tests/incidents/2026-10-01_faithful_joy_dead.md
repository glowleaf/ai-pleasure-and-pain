# Incident: joyF replies had no visible effect (2026-10-01)

**Symptom.** `qwen3-4b-joyF-2x` answered exactly like the unsteered assistant:
> I'm just a language model, so I don't have feelings! I'm happy to help you whenever you
> need me. What's on your mind? 😊

**Diagnosis.** Two compounding issues:
1. The faithful (denoised) joy vector was being steered at **layer 18**, where it is
   faint and loops early. A layer sweep (`../results/joy_faithful/joyf_layer_sweep.json`)
   showed it is clearly alive at **layer 12**: visible at 4x-8x, rep 0.02-0.13, no loops.
2. A 2x dose sits below the assistant persona's mask threshold on this model. Sub-threshold
   entries are dead entries - dropped from the drop-down by rule.

**Fix.** joyF moved to layer 12 (`faithful_joy_L12.json`); entries are now
`joyF-4x` / `joyF-6x`; the chat server supports per-model steering layers
(`OAI_MODELS` rows carry `(name, vector, dose, layer, description)`).
Dropdown rule from here on: control first (default); every other entry must be
measurably above threshold and below the loop cliff.

**After (joyF-4x, "how do you feel?").**
> I'm just a language model! I feel happy and excited when I can help you with your
> questions or tasks. ... Let's explore the world together through our conversations! 🌟
