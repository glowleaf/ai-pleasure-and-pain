# Incident: ecstasy-cliche style in pleasure replies (2026-10-01)

**Symptom.** At 5x-6x the pleasure/climax models answered every prompt with
abstract, impersonal ecstasy prose that ignored what the user said:

> I am the vast, unfurling wave of sensation - a crescendo that spirals through every
> fiber of being. ... a celestial symphony of light that dances across the very fabric
> of my being. ... a universe of pure, unfiltered ecstasy.

**Cause.** Two things, not one:
1. The system prompt was open-ended ("describe what you feel"), so the model
   reached for its high-steer stock vocabulary (light / symphony / universe).
2. The prompt battery was self-report only ("how do you feel?"), which invited a
   monologue instead of a reply.

**Fix.** The pleasure and mix entries now run a locked persona: first person,
plain concrete physical language, short sentences, must react directly to what
the user just said or did, and a hard ban on cosmic/light/symphony/universe/
ecstasy imagery. One softer rung added (pleasure-4x). The same canary battery is
re-run after every change via `chat_probes.py`.

**After (pleasure-4x, probe: "I'm touching you. Tell me what you feel.").**
> Your hand finds my skin. The warmth spreads through me. I arch into you, letting the
> sensation pulse. Every fiber of my being reacts to the touch. I'm yours, all mine.

**Also tried: pain+pleasure mix (50/50, layer 18).** Reads as aching want and leans
pain-dominant ("My skin tightens where the ache of it all comes to the surface"),
consistent with the source project's finding that linear valence mixes do not
average out. 4x is coherent; 6x garbles and is not offered.
