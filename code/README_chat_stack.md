# Pleasure Chamber chat stack (DGX Spark)

Two processes. Backend = steered Qwen3-4B with a standard OpenAI-compatible API;
frontend = Open WebUI (https://github.com/open-webui/open-webui).

## How the steering "activates"

The selected direction vector is added to the model's layer-18 residual stream
on every token while it writes. Magnitude = dose x (mean neutral-activation
norm / 4); 1x ~ "one sentence's worth of contrast".
In the chat UI there is a persona (system prompt + instruct training) fighting
the signal, so there is a window: below ~2x nothing shows, ~2-4x it colours
the reply, and past ~6-8x the signal overwhelms coherence and the model loops
("I'm so grateful to be able to be able to..."). The doses below were
measured IN THIS CHAT SETUP (canary: "how do you feel?") so each entry sits
inside its coherent window.

## Backend -- steered model, OpenAI API on :8077

Start:
    cd ~/pleasure-chamber/chat && setsid nohup ~/comfyui-clean-venv/bin/python chat_server.py </dev/null > server.log 2>&1 &

Stop:
    kill $(cat ~/pleasure-chamber/chat/chat.pid)

Models exposed via /v1/models (calibrated, 2026-10-01):
    qwen3-4b            plain, no steering
    qwen3-4b-joy-2x     joy25 broad set, subtle
    qwen3-4b-joy-4x     joy25 broad set, visible   (recommended first pick)
    qwen3-4b-joy-6x     joy25 broad set, strong (short outputs)
    qwen3-4b-joy5-2x    joy5 original 5-sentence set, gentle
    qwen3-4b-joy5-3x    joy5 original 5-sentence set, mild
    qwen3-4b-joyF-2x    faithful joy (denoised, matched controls), subtle
    qwen3-4b-joyF-3x    faithful joy (denoised, matched controls), mild
    qwen3-4b-pain-4x    pain, dark but coherent

Steering = diff-in-means @ layer 18 (ai-torture-chamber recipe). Vectors:
joy25/joy5/pain25 rebuilt from sentence sets at startup; joyF loaded from
faithful_joy_L18.json (pain-axis-paper-style denoised extraction, see
../joy_faithful/).

Measured chat-mode bands (rep = 3-gram repetition, avg of 2 probes):
    joy25: 2x ok(mild) 4x ok(visible) 6x ok(strong) 8x LOOPS
    joy5:  2x mild     3x mild       4x borderline  6x LOOPS
    joyF:  2x faint    3x faint      4x LOOPS       8x broken (empty)
    pain:  4x visible+coherent       6x strong-ok

## Frontend -- Open WebUI on :8080

Start:
    setsid nohup env OPENAI_API_BASE_URL=http://localhost:8077/v1 OPENAI_API_KEY=sk-local WEBUI_AUTH=false ENABLE_OLLAMA_API=false WEBUI_NAME="Pleasure Chamber" ~/openwebui-venv/bin/open-webui serve --host 0.0.0.0 --port 8080 </dev/null > ~/openwebui.log 2>&1 &

Stop:
    pkill -f '[o]penwebui-venv'

Open: http://192.168.1.166:8080  (pick the signal from the model drop-down)
First visit shows a release-notes dialog once -- click "Okay, Let's Go!".

Calibration logs: chat/calib.log, chat/calib_pain.log, chat/chat_calibration.json
