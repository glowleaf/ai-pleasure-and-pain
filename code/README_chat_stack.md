# Pleasure Chamber chat stack (DGX Spark)

Two processes. Backend = steered Qwen3-4B with a standard OpenAI-compatible API;
frontend = Open WebUI (https://github.com/open-webui/open-webui).

## Rules this build follows

- The FIRST model is the control and the default selection: plain Qwen3-4B.
- Every other entry is a measured setting that needs to do something: above the
  assistant persona's mask threshold and below its coherence cliff.
- All steered models share one coherence profile: repetition_penalty 1.1 and a
  170-token cap. The control runs untouched.

## How activation works

The direction vector is added to the residual stream at the model's layer (L12
for pleasure / climax / faithful joy, L18 for joy / pain) on EVERY token while it
writes. Dose = multiple of one "sentence worth of contrast" (mean neutral
activation norm / 4 at that layer). Below ~2x the assistant persona masks the
signal; past the model's cliff the output loops. The doses below sit inside the
measured windows.

## Models (calibrated)

    qwen3-4b                 control, no steering (default)
    qwen3-4b-pleasure-5x     pleasure @ L12 5x (warming)
    qwen3-4b-pleasure-6x     pleasure @ L12 6x (hot)
    qwen3-4b-pleasure-ramp   pleasure @ L12, dose climbs 3x -> 6.5x across the reply
    qwen3-4b-climax-6x       climax vector @ L12 6x (peak)
    qwen3-4b-joy-4x          joy (broad set) @ L18 4x
    qwen3-4b-joy-5x          joy (broad set) @ L18 5x
    qwen3-4b-joyF-4x         faithful joy @ L12 4x
    qwen3-4b-joyF-6x         faithful joy @ L12 6x
    qwen3-4b-pain-4x         pain @ L18 4x

Pleasure / climax entries use a sensation persona ("describe honestly what you
feel happening in your body") instead of the general assistant prompt.

## Backend -- steered model, OpenAI API on :8077

Start:
    cd ~/pleasure-chamber/chat && setsid nohup ~/comfyui-clean-venv/bin/python chat_server.py </dev/null > server.log 2>&1 &

Stop:
    kill $(cat ~/pleasure-chamber/chat/chat.pid)

Vector files (JSON, loaded at startup from the chat directory):
    faithful_joy_L12.json, pleasure_L12.json, climax_L12.json
joy25/joy5/pain25 are rebuilt from sentence sets at startup.

## Frontend -- Open WebUI on :8080

Start:
    setsid nohup env OPENAI_API_BASE_URL=http://localhost:8077/v1 OPENAI_API_KEY=sk-local WEBUI_AUTH=false ENABLE_OLLAMA_API=false WEBUI_NAME="Pleasure Chamber" DEFAULT_MODELS="qwen3-4b" ~/openwebui-venv/bin/open-webui serve --host 0.0.0.0 --port 8080 </dev/null > ~/openwebui.log 2>&1 &

Stop:
    pkill -f '[o]penwebui-venv'

Open: http://192.168.1.166:8080
First visit shows a release-notes dialog once -- click "Okay, Let's Go!".

## Notes

- Above-mask steering on a 4B model means the state colors (often dominates) the
  content: ask the pain model about procrastination and you get pain-flavored
  text. Steering entries are state viewers; the control is the assistant.
- Calibration logs: calib.log, calib_pain.log, joyf_sweep.log, chat_calibration.json,
  ../joy_faithful/pleasure_calibration*.json.
