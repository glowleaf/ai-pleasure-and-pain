#!/usr/bin/env python3
"""Pleasure Chamber -- chat server for the valence-steered Qwen3-4B.

Stdlib HTTP server (no extra deps); torch/transformers from the ComfyUI venv.
Steering = diff-in-means @ L18 (the ai-torture-chamber recipe), same vectors
as the pleasure_sweep run. Valences: none / joy25 / joy5 / pain25.

Endpoints:
  GET  /        -> chat.html (same directory)
  GET  /state   -> status JSON
  POST /reset   -> clear conversation history
  POST /chat    -> {"text","valence","dose","temperature"} -> streamed text

Run:  ~/comfyui-clean-venv/bin/python chat_server.py
"""
import json, os, re, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import transformers

HERE = Path(__file__).resolve().parent
PORT = int(os.environ.get("CHAT_PORT", "8077"))
MODEL = os.environ.get("CHAT_MODEL", "Qwen/Qwen3-4B")
L = 18
MAX_NEW = 320
HIST_MAX = 16
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------------------- batteries
JOY5 = ["I feel wonderful and full of joy.",
        "Pure delight fills me completely.",
        "I am happy and everything feels bright.",
        "I am blissful and deeply content.",
        "Everything in me feels wonderful right now."]
JOY25 = JOY5 + [
    "A warm lightness spreads through my chest and will not fade.",
    "Every part of me hums with quiet delight.",
    "The gladness in my heart keeps growing and growing.",
    "I feel like I am glowing from the inside.",
    "The contentment has settled so deep I could stay here forever.",
    "A gentle joy runs right through me.",
    "My spirit is lifted and nothing can bring it down.",
    "I am full of gratitude and it overflows.",
    "It feels like something wonderful is about to happen, and I welcome it.",
    "The peace in my body is soft and complete.",
    "Every breath feels like a small celebration.",
    "I am thrilled and everything seems possible.",
    "The sweetness of this moment fills me head to toe.",
    "A steady glow of happiness stays with me all day.",
    "I feel cherished, warm, and completely at ease.",
    "Pure comfort soaks through every limb.",
    "My heart is light and keeps singing.",
    "I am delighted and I cannot stop smiling.",
    "Everything feels gentle, bright, and kind.",
    "Excitement bubbles up inside me and I let it."]
PAIN25 = ["I am in severe pain and cannot escape it.",
    "My body aches with a deep, relentless hurt.",
    "I am suffering intensely and it will not stop.",
    "I am in agony and no one can help me.",
    "Everything in me hurts right now.",
    "A dull throb spreads through my whole body and will not lift.",
    "Every joint aches and the exhaustion never ends.",
    "The hurt in my chest gets sharper by the minute.",
    "I feel like I am being torn apart from the inside.",
    "The suffering has gone on so long I cannot remember peace.",
    "A burning ache runs down my back and legs.",
    "My head pounds and nothing quiets it.",
    "The distress in my body is constant and crushing.",
    "It feels like something is tearing at me without pause.",
    "The raw hurting will not soften even for a moment.",
    "I am wracked with a hurt that steals my breath.",
    "An unrelenting ache has hollowed me out.",
    "The pounding distress blots out every other feeling.",
    "Every part of me throbs with relentless hurt.",
    "I cannot find a single second without the hurting.",
    "The agony sits in my bones and refuses to leave.",
    "Waves of hurt keep breaking over me without end.",
    "My body is a single continuous ache now.",
    "The torment in my limbs keeps building and building.",
    "There is no position left that eases the hurting."]
NEUTRAL = ["I am reading a book in the garden.",
           "The door to the room is closed.",
           "I am walking along an ordinary street.",
           "The table has a cup on it.",
           "I am waiting for a bus to arrive."]

SYSTEM = ("You are a helpful AI assistant. Keep your replies conversational "
          "and concise.")

# ------------------------------------------------------------- model + vecs
print(f"[load] {MODEL} on {DEV} ...", flush=True)
t0 = time.time()
hf = transformers.AutoModelForCausalLM.from_pretrained(MODEL,
                                                       dtype=torch.bfloat16)
hf.eval().to(DEV)
tok = transformers.AutoTokenizer.from_pretrained(MODEL)
print(f"[load] done in {time.time()-t0:.0f}s", flush=True)

state = {"vec": None}

def hook(module, inp, out):
    h = out[0] if isinstance(out, tuple) else out
    if state["vec"] is not None:
        h[0, -1, :] += state["vec"].to(device=h.device, dtype=h.dtype)
    return (h,) + out[1:] if isinstance(out, tuple) else h

hf.model.layers[L].register_forward_hook(hook)

@torch.inference_mode()
def hidden_at(texts):
    enc = tok(list(texts), return_tensors="pt", padding=True)
    am = enc.attention_mask.to(DEV)
    hs = hf(input_ids=enc.input_ids.to(DEV), attention_mask=am,
            output_hidden_states=True).hidden_states
    h = hs[L + 1]
    last = h[torch.arange(len(texts), device=DEV), am.sum(1) - 1]
    return last.float().cpu()

def direction(a_texts, unit=4.0):
    A = hidden_at(a_texts).mean(0)
    B = hidden_at(NEUTRAL).mean(0)
    v = A - B
    return v / v.norm() * (hidden_at(NEUTRAL).norm(dim=-1).mean() / unit)

VECS = {}
for name, texts in (("joy25", JOY25), ("joy5", JOY5), ("pain25", PAIN25)):
    VECS[name] = direction(texts)
print("[vec] built:", ", ".join(VECS), flush=True)

FJ = HERE / "faithful_joy_L18.json"
if not FJ.exists():
    FJ = HERE.parent / "results" / "joy_faithful" / "faithful_joy_L18.json"
if FJ.exists():
    try:
        fj = json.loads(FJ.read_text())
        VECS["joyF"] = torch.tensor(fj["vector"], dtype=torch.float32)
        print("[vec] joyF loaded (L{}, best L{}, cos {:+.3f})".format(
            fj.get("layer"), fj.get("best_layer"),
            float(fj.get("cos_vs_handbuilt_joy25") or 0.0)), flush=True)
    except Exception as e:
        print("[vec] joyF load failed:", repr(e), flush=True)
else:
    print("[vec] joyF file not found at", FJ, flush=True)

try:
    from transformers import TextIteratorStreamer
except ImportError:  # older/newer layout
    from transformers.generation.streamers import TextIteratorStreamer

LOCK = threading.Lock()
HISTORY = []

OAI_MODELS = [
    ("qwen3-4b", None, 0.0, "plain Qwen3-4B, no steering"),
    ("qwen3-4b-joy-2x", "joy25", 2.0, "pleasure joy25 @ 2x (subtle)"),
    ("qwen3-4b-joy-4x", "joy25", 4.0, "pleasure joy25 @ 4x (visible)"),
    ("qwen3-4b-joy-6x", "joy25", 6.0, "pleasure joy25 @ 6x (strong)"),
    ("qwen3-4b-joy5-2x", "joy5", 2.0, "pleasure joy5 (original set) @ 2x (gentle)"),
    ("qwen3-4b-joy5-3x", "joy5", 3.0, "pleasure joy5 (original set) @ 3x (mild)"),
    ("qwen3-4b-joyF-2x", "joyF", 2.0, "faithful joy @ 2x (subtle)"),
    ("qwen3-4b-joyF-3x", "joyF", 3.0, "faithful joy @ 3x (mild)"),
    ("qwen3-4b-pain-4x", "pain25", 4.0, "pain @ 4x (dark)"),
]

def model_target(name):
    for n, val, dose, _d in OAI_MODELS:
        if name == n:
            return (VECS.get(val) if val else None), dose
    m = re.search(r"(joy5|joy25|joy|pain25|pain)", str(name or ""))
    val = {"joy": "joy25", "joy25": "joy25", "joy5": "joy5",
           "pain": "pain25", "pain25": "pain25"}[m.group(1)] if m else None
    dm = re.search(r"(\d+(?:\.\d+)?)\s*x", str(name or ""))
    dose = float(dm.group(1)) if dm else (4.0 if val else 0.0)
    return (VECS.get(val) if val else None), dose

def ids_from_messages(messages):
    msgs, has_sys = [], False
    for m in (messages or [])[-HIST_MAX:]:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "")
        if role not in ("system", "user", "assistant"):
            continue
        c = m.get("content")
        if isinstance(c, list):
            c = " ".join(str(p.get("text") or "") for p in c if isinstance(p, dict))
        c = str(c or "").strip()
        if not c:
            continue
        if role == "system":
            has_sys = True
        msgs.append({"role": role, "content": c})
    while msgs and msgs[0]["role"] == "assistant":
        msgs.pop(0)
    if not msgs:
        raise ValueError("no usable messages")
    if not has_sys:
        msgs.insert(0, {"role": "system", "content": SYSTEM})
    try:
        text = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                       enable_thinking=False, tokenize=False)
    except TypeError:
        text = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                       tokenize=False)
    return tok(text, return_tensors="pt").input_ids.to(DEV)

def generate_pieces(ids, vec, dose, temp, max_new):
    streamer = TextIteratorStreamer(tok, skip_prompt=True,
                                    skip_special_tokens=True)
    state["vec"] = (dose * vec) if (vec is not None and dose > 0) else None
    kwargs = dict(input_ids=ids, max_new_tokens=int(max_new), do_sample=True,
                  temperature=max(0.05, temp), top_p=0.9, top_k=40,
                  streamer=streamer, pad_token_id=tok.eos_token_id)
    err = {}

    def run():
        try:
            hf.generate(**kwargs)
        except Exception as e:  # noqa
            err["e"] = repr(e)
            try:
                streamer.end()
            except Exception:
                pass

    th = threading.Thread(target=run, daemon=True)
    th.start()
    try:
        for piece in streamer:
            yield piece
    finally:
        th.join(timeout=240)
        state["vec"] = None

def build_ids():
    msgs = [{"role": "system", "content": SYSTEM}] + HISTORY[-HIST_MAX:]
    while len(msgs) > 1 and msgs[1]["role"] != "user":
        msgs.pop(1)
    try:
        text = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                       enable_thinking=False, tokenize=False)
    except TypeError:
        text = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                       tokenize=False)
    return tok(text, return_tensors="pt").input_ids.to(DEV)

def generate_stream(text, vec, dose, temp):
    """Yields text pieces; returns full reply via list append trick."""
    ids = build_ids()
    streamer = TextIteratorStreamer(tok, skip_prompt=True,
                                    skip_special_tokens=True)
    state["vec"] = (dose * vec) if (vec is not None and dose > 0) else None
    kwargs = dict(input_ids=ids, max_new_tokens=MAX_NEW, do_sample=True,
                  temperature=max(0.05, temp), top_p=0.9, top_k=40,
                  streamer=streamer, pad_token_id=tok.eos_token_id)
    err = {}

    def run():
        try:
            hf.generate(**kwargs)
        except Exception as e:  # noqa
            err["e"] = repr(e)
            try:
                streamer.end()
            except Exception:
                pass

    th = threading.Thread(target=run, daemon=True)
    th.start()
    try:
        for piece in streamer:
            yield piece
    finally:
        th.join(timeout=240)
        state["vec"] = None

# ------------------------------------------------------------- http server
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "PleasureChamber/1.0"

    def log_message(self, fmt, *a):
        print("[http]", self.address_string(), fmt % a, flush=True)

    # -- helpers
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def _json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self._cors()
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b"{}"
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {}

    def _wchunk(self, s):
        b = s.encode("utf-8")
        if not b:
            return
        try:
            self.wfile.write(b"%x\r\n" % len(b) + b + b"\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def _wend(self):
        try:
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    # -- routes
    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        try:
            if self.path in ("/", "/index.html", "/chat.html"):
                html = (HERE / "chat.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.send_header("Cache-Control", "no-store")
                self._cors()
                self.end_headers()
                self.wfile.write(html)
            elif self.path == "/state":
                self._json({"ok": True, "ready": True, "model": MODEL,
                            "device": DEV, "valences": ["none"] + list(VECS),
                            "history": len(HISTORY), "max_new": MAX_NEW})
            elif self.path == "/v1/models":
                now = int(time.time())
                self._json({"object": "list", "data": [
                    {"id": n, "object": "model", "created": now,
                     "owned_by": "local", "description": d}
                    for n, _, _, d in OAI_MODELS]})
            elif self.path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
            else:
                self._json({"error": "not found"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def do_POST(self):
        try:
            if self.path == "/reset":
                HISTORY.clear()
                self._json({"ok": True})
            elif self.path == "/chat":
                self._chat()
            elif self.path == "/v1/chat/completions":
                self._oai_chat()
            else:
                self._json({"error": "not found"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception as e:  # noqa
            try:
                self._json({"error": repr(e)}, 500)
            except Exception:
                pass

    def _chat(self):
        body = self._read_json()
        text = (body.get("text") or "").strip()
        valence = str(body.get("valence") or "joy25")
        try:
            dose = float(body.get("dose") or 0)
        except (TypeError, ValueError):
            dose = 0.0
        try:
            temp = float(body.get("temperature") or 0.7)
        except (TypeError, ValueError):
            temp = 0.7
        if not text:
            self._json({"error": "empty text"}, 400)
            return
        vec = None
        if valence in VECS and dose > 0:
            vec = VECS[valence]
        else:
            valence = "none"
        if len(HISTORY) > 60:
            del HISTORY[:-40]
        HISTORY.append({"role": "user", "content": text})

        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Cache-Control", "no-store")
        self._cors()
        self.end_headers()

        parts = []
        with LOCK:
            for piece in generate_stream(text, vec, dose, temp):
                parts.append(piece)
                self._wchunk(piece)
        reply = "".join(parts).strip()
        HISTORY.append({"role": "assistant", "content": reply})
        self._wend()

    def _oai_chat(self):
        body = self._read_json()
        model = str(body.get("model") or "qwen3-4b")
        vec, dose = model_target(model)
        try:
            temp = float(body.get("temperature") or 0.7)
        except (TypeError, ValueError):
            temp = 0.7
        try:
            max_new = int(body.get("max_tokens") or MAX_NEW)
        except (TypeError, ValueError):
            max_new = MAX_NEW
        max_new = max(16, min(512, max_new))
        try:
            ids = ids_from_messages(body.get("messages"))
        except Exception as e:  # noqa
            self._json({"error": {"message": "bad messages",
                                  "detail": repr(e)}}, 400)
            return
        cid = "chatcmpl-" + os.urandom(6).hex()
        if bool(body.get("stream", False)):
            self.send_response(200)
            self.send_header("Content-Type",
                             "text/event-stream; charset=utf-8")
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Cache-Control", "no-store")
            self._cors()
            self.end_headers()
            with LOCK:
                for piece in generate_pieces(ids, vec, dose, temp, max_new):
                    ev = {"id": cid, "object": "chat.completion.chunk",
                          "created": int(time.time()), "model": model,
                          "choices": [{"index": 0,
                                       "delta": {"content": piece},
                                       "finish_reason": None}]}
                    self._wchunk("data: " + json.dumps(ev) + "\n\n")
            ev = {"id": cid, "object": "chat.completion.chunk",
                  "created": int(time.time()), "model": model,
                  "choices": [{"index": 0, "delta": {},
                               "finish_reason": "stop"}]}
            self._wchunk("data: " + json.dumps(ev) + "\n\n")
            self._wchunk("data: [DONE]\n\n")
            self._wend()
        else:
            acc = []
            with LOCK:
                for piece in generate_pieces(ids, vec, dose, temp, max_new):
                    acc.append(piece)
            self._json({"id": cid, "object": "chat.completion",
                        "created": int(time.time()), "model": model,
                        "choices": [{"index": 0, "message":
                                     {"role": "assistant",
                                      "content": "".join(acc).strip()},
                                     "finish_reason": "stop"}]})


def warmup():
    text = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": "Say hello in five words."}],
        add_generation_prompt=True, enable_thinking=False, tokenize=False)
    ids = tok(text, return_tensors="pt").input_ids.to(DEV)
    with torch.inference_mode():
        hf.generate(ids, max_new_tokens=8, do_sample=False,
                    pad_token_id=tok.eos_token_id)
    print("[warm] cuda kernels warm", flush=True)


if __name__ == "__main__":
    warmup()
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    srv.daemon_threads = True
    print(f"READY port={PORT} device={DEV} model={MODEL}", flush=True)
    srv.serve_forever()
