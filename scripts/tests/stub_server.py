#!/usr/bin/env python3
"""Stub of the llama-server endpoints that score_exact.py uses. No model; for tests only.

Endpoints: POST /tokenize, /apply-template, /v1/chat/completions, /v1/embeddings; GET /v1/models, /props.

The "model" is a deterministic function of the rendered prompt:
  - logits over a vocabulary of VOCAB ids, drawn from a generator seeded by the prompt's hash and
    softcapped at SOFTCAP (|logit| <= SOFTCAP), as some served models do;
  - a final state of N_EMBD floats, drawn the same way.
The tokenizer splits text into pieces (" word", punctuation runs, whitespace runs) and gives each new
piece the next free id. Chat completions render the messages with a fixed template, prepend one BOS
token, apply logit_bias, then return post-sampling probabilities (softmax at temperature 1 over the
whole vocabulary), rounded to float32 as a real server returns them.

Every request body is appended to StubState.requests, and StubState.max_in_flight records the largest
number of requests handled at the same time, so a test can check request shapes and sequencing.
Run standalone: python stub_server.py [port]
"""
import hashlib, json, re, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

VOCAB = 4096
N_EMBD = 24          # deliberately not a common hidden size: the scorer must read it, not assume it
SOFTCAP = 30.0
BOS_ID = 1
SPECIAL = ["<pad>", "<bos>"]
PIECE_RE = re.compile(r" ?[A-Za-z0-9]+| ?[^A-Za-z0-9\s]+|\s+")


class StubState:
    def __init__(self):
        self.lock = threading.Lock()
        self.vocab = {p: i for i, p in enumerate(SPECIAL)}
        self.requests = []          # (path, body)
        self.in_flight = 0
        self.max_in_flight = 0
        self.report_n_embd = True   # False: /v1/models omits meta.n_embd
        self.fail_after = None      # int n: chat requests after the n-th answer HTTP 500
        self.n_chat = 0

    def tok_id(self, piece):
        with self.lock:
            if piece not in self.vocab:
                if len(self.vocab) >= VOCAB:
                    raise RuntimeError("stub vocabulary full")
                self.vocab[piece] = len(self.vocab)
            return self.vocab[piece]

    def piece(self, tid):
        with self.lock:
            for p, i in self.vocab.items():
                if i == tid:
                    return p
        return f"<unused{tid}>"

    def tokenize(self, text):
        return [self.tok_id(p) for p in PIECE_RE.findall(text)]


def render(messages):
    out = []
    for m in messages:
        out.append(f"<{m['role']}>\n{m['content'].strip()}\n")
    out.append("<model>\n")
    return "".join(out)


def _rng(prompt, salt):
    h = hashlib.sha256((salt + "\x00" + prompt).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def true_logits(prompt):
    """Final logits (float64, length VOCAB) for a rendered prompt; softcapped like the served model."""
    raw = _rng(prompt, "logits").normal(0.0, 12.0, VOCAB)
    return SOFTCAP * np.tanh(raw / SOFTCAP)


def true_state(prompt):
    """Final state (float32, length N_EMBD) at the last prompt token."""
    return _rng(prompt, "state").normal(0.0, 3.0, N_EMBD).astype(np.float32)


def make_handler(state):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            raw = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path == "/v1/models":
                meta = {"n_vocab": VOCAB, **({"n_embd": N_EMBD} if state.report_n_embd else {})}
                return self._send(200, {"object": "list", "data": [{"id": "stub", "object": "model", "meta": meta}]})
            if self.path == "/props":
                return self._send(200, {"model_path": "stub.gguf"})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            with state.lock:
                state.requests.append((self.path, body))
                state.in_flight += 1
                state.max_in_flight = max(state.max_in_flight, state.in_flight)
            try:
                code, obj = self.route(body)
            except Exception as ex:  # report as a server error, like a real server would
                code, obj = 500, {"error": f"{type(ex).__name__}: {ex}"}
            finally:
                with state.lock:
                    state.in_flight -= 1
            self._send(code, obj)

        def route(self, body):
            if self.path == "/tokenize":
                text = body["content"]
                ids = ([BOS_ID] if body.get("add_special") else []) + state.tokenize(text)
                if body.get("with_pieces"):
                    return 200, {"tokens": [{"id": i, "piece": state.piece(i)} for i in ids]}
                return 200, {"tokens": ids}
            if self.path == "/apply-template":
                return 200, {"prompt": render(body["messages"])}
            if self.path == "/v1/chat/completions":
                with state.lock:
                    state.n_chat += 1
                    if state.fail_after is not None and state.n_chat > state.fail_after:
                        return 500, {"error": "injected failure"}
                prompt = render(body["messages"])
                ntok = 1 + len(state.tokenize(prompt))
                lg = true_logits(prompt).copy()
                for tid, b in body.get("logit_bias", []):
                    lg[int(tid)] += float(b)
                lg = lg / float(body.get("temperature", 1.0))
                p = np.exp(lg - lg.max())
                p /= p.sum()
                k = int(body.get("top_logprobs", 0))
                order = np.argsort(-p, kind="stable")[:k]
                top = [{"id": int(t), "token": state.piece(int(t)), "prob": float(np.float32(p[t]))} for t in order]
                return 200, {"choices": [{"index": 0, "message": {"role": "assistant", "content": top[0]["token"]},
                                          "logprobs": {"content": [{"id": top[0]["id"], "token": top[0]["token"],
                                                                    "prob": top[0]["prob"], "top_probs": top}]}}],
                             "usage": {"prompt_tokens": ntok, "completion_tokens": 1}}
            if self.path == "/v1/embeddings":
                prompt = body["input"]
                ntok = 1 + len(state.tokenize(prompt))
                v = true_state(prompt)
                if body.get("embd_normalize", 2) != -1:
                    v = v / np.linalg.norm(v)
                return 200, {"data": [{"index": 0, "embedding": [float(x) for x in v]}],
                             "usage": {"prompt_tokens": ntok, "total_tokens": ntok}}
            return 404, {"error": "not found"}
    return H


def start(port=0):
    """Start the stub in a background thread. Returns (server, state, base_url)."""
    st = StubState()
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(st))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, st, f"http://127.0.0.1:{srv.server_address[1]}"


if __name__ == "__main__":
    srv, _, url = start(int(sys.argv[1]) if len(sys.argv) > 1 else 8089)
    print(f"stub listening on {url}", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        srv.shutdown()
