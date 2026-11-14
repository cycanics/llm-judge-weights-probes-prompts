#!/usr/bin/env python3
"""Exact first-token score of a yes/no LLM judge through llama-server, with the optional final hidden state.

usage:
  score_exact.py --server http://127.0.0.1:8080 --spec SPEC.json --items ITEMS.jsonl --out ROWS.jsonl
                 [--states PREFIX] [--limit N] [--resume] [--bias 61] [--k-extra 32]
  score_exact.py --selftest

Inputs
  SPEC.json   {"system": <text or null>, "user_template": "...{text}...", "pos": <word meaning "positive">,
               "neg": <word meaning "clear">, "variants": [<optional list of token strings>]}
  ITEMS.jsonl one object per line: "key" (unique), "text", optional "label".

Method (one request per item, strictly one at a time)
  The judge answers with one word, so its decision is the first generated token. The score is
      margin = logit(pos) - logit(neg)
  at the answer position, read exactly from /v1/chat/completions:
  - An equal logit bias B goes on the pos token, the neg token and every variant token. The request sets
    temperature 1, samplers ["temperature"], post_sampling_probs true, max_tokens 1 and asks for the top
    K = (number of biased tokens) + --k-extra probabilities. The returned probabilities are then a softmax
    over the whole vocabulary after the bias.
  - If B exceeds the spread of the model's final logits, every biased token ranks above every unbiased one
    and is always in the returned list. The scorer checks this on every row and records an error if not.
    The default 61 suits models whose final logits are softcapped at 30 (spread <= 60); for a model
    without a softcap choose B above its logit spread.
  - Because the bias is equal, it cancels in the difference:
        log p(pos) - log p(neg) = (logit(pos) + B) - (logit(neg) + B) = logit(pos) - logit(neg).
    The margin is exact and is never floored, however unlikely either word is.
  - lp_pos, lp_neg, variants and top5 are log-softmax values rebuilt from the returned tokens as
    c_t - logsumexp(c), c_t = log p_t - (B if t is biased else 0). They leave out the probability mass of
    tokens outside the returned K; the margin does not depend on it.
  - cache_prompt is false: every item is a fresh pass, so a row depends only on the item and the spec,
    not on the order of items or on what the server evaluated before. (With prompt caching the logits
    move with the cache split.)
  - pos and neg must each be one token whose piece equals the word, and appending the word to the
    rendered prompt must add exactly that token; otherwise the scorer stops.

Variants
  The tokens that receive the bias are pos, neg and their variants. If the spec lists "variants", those
  strings are used. Otherwise the default rule is, for each of the two words: the word, the word with a
  leading space, lower-case, capitalised. Variants that are not a single token are dropped and listed in
  the header line. The study this script comes from listed its variants by hand in the spec.

Final state (--states PREFIX)
  Needs a server started with --embeddings --pooling last. Right after the chat pass the scorer sends
  the same rendered prompt to /v1/embeddings with embd_normalize -1 and gets the un-normalised final
  hidden state (output of the last norm, before the output matrix) at the last prompt token, the
  position whose logits give the answer. The server reuses the n-1 tokens of the chat pass and
  recomputes the last one, so this costs little. Each row checks that the embedding pass saw as many
  prompt tokens as the chat pass.
  Written to PREFIX.npy (float32, [items, hidden size], items order, NaN rows = not done) and
  PREFIX.keys.txt (one key per line). The hidden size is read from the server (/v1/models meta.n_embd)
  or, if the server does not report it, from the first returned vector.

Output row (one JSON line per item)
  key, label, ntok (prompt tokens), margin, lp_pos, lp_neg, variants {string: log-prob},
  top5 [[piece, log-prob], ...], s (seconds), err (null or message).
  --resume skips keys whose row has err null; rows with an error are retried and appended.
"""
import argparse, http.client, json, math, os, sys, time, urllib.parse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


class Http:
    """One persistent HTTP/1.1 connection; reconnects once on a broken connection."""

    def __init__(self, server, timeout):
        u = urllib.parse.urlparse(server)
        self.host, self.port, self.timeout = u.hostname, u.port or 80, timeout
        self.conn = None

    def _call(self, method, path, body=None):
        data = None if body is None else json.dumps(body).encode()
        hdr = {} if body is None else {"Content-Type": "application/json"}
        for attempt in (0, 1):
            try:
                if self.conn is None:
                    self.conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
                self.conn.request(method, path, body=data, headers=hdr)
                resp = self.conn.getresponse()
                raw = resp.read()
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status} {path}: {raw[:200]!r}")
                return json.loads(raw)
            except (http.client.HTTPException, ConnectionError, OSError):
                self.conn = None
                if attempt:
                    raise

    def post(self, path, body):
        return self._call("POST", path, body)

    def get(self, path):
        return self._call("GET", path)


# ---------------------------------------------------------------- spec and items
def load_spec(path):
    v = json.load(open(path, encoding="utf-8"))
    for f in ("system", "user_template", "pos", "neg"):
        if f not in v:
            sys.exit(f"FATAL: spec lacks field {f!r}")
    if v["user_template"].count("{text}") != 1:
        sys.exit("FATAL: user_template must contain {text} exactly once")
    if v["pos"] == v["neg"]:
        sys.exit("FATAL: pos == neg")
    if "variants" in v and not (isinstance(v["variants"], list) and all(isinstance(x, str) for x in v["variants"])):
        sys.exit("FATAL: variants must be a list of strings")
    return v


def load_items(path):
    items = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    for i, r in enumerate(items):
        if "key" not in r or "text" not in r:
            sys.exit(f"FATAL: item line {i + 1} lacks key or text")
        r["key"] = str(r["key"])
    keys = [r["key"] for r in items]
    if len(set(keys)) != len(keys):
        sys.exit("FATAL: duplicate keys in items")
    return items


def messages(spec, text):
    m = []
    if spec["system"] is not None:
        m.append({"role": "system", "content": spec["system"]})
    m.append({"role": "user", "content": spec["user_template"].replace("{text}", text)})
    return m


def variant_words(spec):
    """Strings whose tokens receive the bias besides pos and neg: listed in the spec, else the default rule."""
    if "variants" in spec:
        cand = [spec["pos"], spec["neg"]] + list(spec["variants"])
    else:
        cand = []
        for w in (spec["pos"], spec["neg"]):
            cand += [w, " " + w, w.lower(), w.capitalize()]
    out = []
    for w in cand:
        if w not in out:
            out.append(w)
    return out


# ---------------------------------------------------------------- tokens
def single_token(h, word):
    t = h.post("/tokenize", {"content": word, "with_pieces": True, "add_special": False})["tokens"]
    if len(t) == 1 and t[0]["piece"] == word:
        return t[0]["id"]
    return None


def resolve_tokens(h, spec, kw):
    """pos/neg must be one token each, and stay that token when appended to the rendered prompt."""
    ids = {}
    tmpl = h.post("/apply-template", {"messages": messages(spec, "Sample text."), "chat_template_kwargs": kw})["prompt"]
    a = h.post("/tokenize", {"content": tmpl, "add_special": False, "parse_special": True})["tokens"]
    for role in ("pos", "neg"):
        w = spec[role]
        tid = single_token(h, w)
        if tid is None:
            sys.exit(f"FATAL: {role} word {w!r} is not a single token")
        b = h.post("/tokenize", {"content": tmpl + w, "add_special": False, "parse_special": True})["tokens"]
        if b != a + [tid]:
            sys.exit(f"FATAL: {role} word {w!r} does not tokenize to the single id {tid} at the answer position")
        ids[role] = tid
    variants, dropped = {}, []
    for w in variant_words(spec):
        tid = single_token(h, w)
        if tid is None:
            dropped.append(w)
        else:
            variants[w] = tid
    return ids, variants, dropped


def server_hidden_size(h):
    try:
        d = h.get("/v1/models")["data"][0]
        n = (d.get("meta") or {}).get("n_embd")
        return int(n) if n else None
    except Exception:
        return None


# ---------------------------------------------------------------- one row
def parse_row(j, ids, variants, biased, bias):
    """Margin and log-softmax values from one chat response. Raises on a check failure."""
    bset = set(biased)
    tl = j["choices"][0]["logprobs"]["content"][0]["top_probs"]
    c, piece = {}, {}
    for e in tl:
        if e["id"] in c:
            continue
        if e["prob"] <= 0.0:
            if e["id"] in bset:
                raise RuntimeError(f"biased id {e['id']} returned with probability 0")
            continue                      # unbiased token below float range: no mass to add
        c[e["id"]] = math.log(e["prob"]) - (bias if e["id"] in bset else 0.0)
        piece[e["id"]] = e["token"]
    missing = [t for t in biased if t not in c]
    if missing:
        raise RuntimeError(f"biased ids missing from response: {missing}")
    if any(e["id"] not in bset for e in tl[:len(biased)]):
        raise RuntimeError("an unbiased token outranked a biased one (bias too small for this model?)")
    m = max(c.values())
    lse = m + math.log(sum(math.exp(v - m) for v in c.values()))
    id2var = {v: w for w, v in variants.items()}
    return {"margin": c[ids["pos"]] - c[ids["neg"]],
            "lp_pos": c[ids["pos"]] - lse, "lp_neg": c[ids["neg"]] - lse,
            "variants": {id2var[t]: c[t] - lse for t in variants.values()},
            "top5": [[piece[t], v - lse] for t, v in sorted(c.items(), key=lambda x: -x[1])[:5]]}


# ---------------------------------------------------------------- run
def run(a):
    h = Http(a.server.rstrip("/"), a.timeout)
    kw = json.loads(a.template_kwargs)
    spec = load_spec(a.spec)
    items = load_items(a.items)
    if a.limit is not None:
        items = items[:a.limit]
    keys = [r["key"] for r in items]
    n_items = len(items)

    done = set()
    if os.path.exists(a.out) and os.path.getsize(a.out) > 0:
        if not a.resume:
            sys.exit(f"FATAL: {a.out} exists; pass --resume to continue it")
        for l in open(a.out, encoding="utf-8"):
            try:
                r = json.loads(l)
            except Exception:
                continue
            if r.get("err") is None:
                done.add(r["key"])

    H, dim, f_npy = None, None, None
    if a.states:
        f_npy, f_keys = a.states + ".npy", a.states + ".keys.txt"
        if os.path.exists(f_keys) and a.resume:
            if open(f_keys, encoding="utf-8").read().split("\n")[:-1] != keys:
                sys.exit(f"FATAL: {f_keys} is for different items; cannot resume")
        elif os.path.exists(f_keys) or os.path.exists(f_npy):
            sys.exit(f"FATAL: {f_keys} or {f_npy} exists; pass --resume to continue it")
        else:
            with open(f_keys, "w", encoding="utf-8") as fk:
                fk.write("".join(k + "\n" for k in keys))
        if os.path.exists(f_npy):
            H = np.lib.format.open_memmap(f_npy, mode="r+")
            if H.ndim != 2 or H.shape[0] != n_items:
                sys.exit(f"FATAL: {f_npy} has shape {H.shape}, expected {n_items} rows")
            dim = H.shape[1]
        elif done:
            sys.exit(f"FATAL: {a.out} has done rows but {f_npy} does not exist; states would be missing")
        else:
            dim = server_hidden_size(h)    # None: taken from the first returned vector

    def open_states(d):
        nonlocal H
        H = np.lib.format.open_memmap(f_npy, mode="w+", dtype=np.float32, shape=(n_items, d))
        H[:] = np.nan
        H.flush()

    if a.states and H is None and dim is not None:
        open_states(dim)

    ids, variants, dropped = resolve_tokens(h, spec, kw)
    biased = sorted(set(variants.values()) | set(ids.values()))
    k = len(biased) + a.k_extra
    print(f"pos {spec['pos']!r}={ids['pos']} neg {spec['neg']!r}={ids['neg']} biased {variants} "
          f"dropped (not one token) {dropped} K {k} bias {a.bias} states "
          f"{'off' if not a.states else ('hidden size ' + str(dim) if dim else 'hidden size from first vector')}",
          flush=True)
    todo = [i for i, r in enumerate(items) if r["key"] not in done]
    print(f"total {n_items} already done {n_items - len(todo)} to do {len(todo)}", flush=True)

    base = {"max_tokens": 1, "logprobs": True, "top_logprobs": k, "post_sampling_probs": True,
            "temperature": 1.0, "samplers": ["temperature"], "seed": 0,
            "logit_bias": [[t, a.bias] for t in biased],
            "chat_template_kwargs": kw, "cache_prompt": False}

    t_start = t_last = time.time()
    n = n_err = consec = 0
    with open(a.out, "a", encoding="utf-8") as fo:
        for i in todo:
            r = items[i]
            rec = {"key": r["key"], "label": r.get("label"), "ntok": None, "margin": None, "lp_pos": None,
                   "lp_neg": None, "variants": None, "top5": None, "s": None, "err": None}
            t0 = time.time()
            try:
                msgs = messages(spec, r["text"])
                j = h.post("/v1/chat/completions", dict(base, messages=msgs))
                rec["ntok"] = j["usage"]["prompt_tokens"]
                rec.update(parse_row(j, ids, variants, biased, a.bias))
                if a.states:
                    prompt = h.post("/apply-template", {"messages": msgs, "chat_template_kwargs": kw})["prompt"]
                    e = h.post("/v1/embeddings", {"input": prompt, "embd_normalize": -1})
                    if e["usage"]["prompt_tokens"] != rec["ntok"]:
                        raise RuntimeError(f"embedding pass {e['usage']['prompt_tokens']} tokens, chat pass {rec['ntok']}")
                    v = np.asarray(e["data"][0]["embedding"], np.float32)
                    if v.ndim != 1 or not np.all(np.isfinite(v)):
                        raise RuntimeError(f"bad state shape {v.shape} or non-finite values; "
                                           "server needs --embeddings --pooling last")
                    if H is None:
                        open_states(v.shape[0])
                    if v.shape[0] != H.shape[1]:
                        raise RuntimeError(f"state length {v.shape[0]}, expected {H.shape[1]}")
                    H[i] = v
                    H.flush()             # state is on disk before its row is written
                consec = 0
            except Exception as ex:
                for f in ("margin", "lp_pos", "lp_neg", "variants", "top5"):
                    rec[f] = None
                rec["err"] = f"{type(ex).__name__}: {ex}"[:300]
                consec += 1
                n_err += 1
            rec["s"] = round(time.time() - t0, 3)
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fo.flush()
            n += 1
            now = time.time()
            if now - t_last >= 30 or n == len(todo):
                print(f"progress {n_items - len(todo) + n}/{n_items} this run {n} errors {n_err} "
                      f"items/s {n / max(now - t_start, 1e-9):.2f}", flush=True)
                t_last = now
            if consec >= 5:
                print("STOP: 5 consecutive errors", flush=True)
                return 2
    print(f"DONE {n} scored this run, {n_err} errors", flush=True)
    return 0


# ---------------------------------------------------------------- selftest
def selftest():
    import tempfile
    sys.path.insert(0, os.path.join(HERE, "tests"))
    sys.dont_write_bytecode = True
    import stub_server as S

    fails = []

    def check(cond, what):
        print(("ok   " if cond else "FAIL ") + what, flush=True)
        if not cond:
            fails.append(what)

    srv, st, url = S.start()
    tmp = tempfile.mkdtemp(prefix="score_exact_selftest_")
    spec = {"system": "Answer with one word: yes if the text asks for action, no otherwise.",
            "user_template": "Text:\n{text}\nAnswer:", "pos": "yes", "neg": "no"}
    p_spec = os.path.join(tmp, "spec.json")
    json.dump(spec, open(p_spec, "w"))
    items = [{"key": f"k{i}", "text": f"synthetic item {i} " + "lorem ipsum " * (i % 7), "label": i % 2}
             for i in range(12)]
    p_items = os.path.join(tmp, "items.jsonl")
    with open(p_items, "w") as f:
        f.write("".join(json.dumps(r) + "\n" for r in items))

    def args(**kw):
        d = dict(server=url, spec=p_spec, items=p_items, out=None, states=None, limit=None, resume=False,
                 bias=61.0, k_extra=32, timeout=30.0, template_kwargs='{"enable_thinking": false}')
        d.update(kw)
        return argparse.Namespace(**d)

    def rows(p):
        return [json.loads(l) for l in open(p)]

    # 1. full run with states
    out1, pre1 = os.path.join(tmp, "r1.jsonl"), os.path.join(tmp, "s1")
    del st.requests[:]
    rc = run(args(out=out1, states=pre1))
    R = rows(out1)
    check(rc == 0 and len(R) == 12 and all(r["err"] is None for r in R), "run with states: 12 rows, 0 errors")
    chats = [b for p, b in st.requests if p == "/v1/chat/completions"]
    check(len(chats) == 12, "one chat request per item")
    check(st.max_in_flight == 1, "requests strictly one at a time")
    check(all(b["cache_prompt"] is False for b in chats), "cache_prompt false on every chat request")
    check(all(b["max_tokens"] == 1 and b["temperature"] == 1.0 and b["samplers"] == ["temperature"]
              and b["post_sampling_probs"] is True and b["logprobs"] is True for b in chats),
          "max_tokens 1, temperature 1, samplers [temperature], post_sampling_probs on every chat request")
    want = {st.vocab[w] for w in ("yes", " yes", "Yes", "no", " no", "No")}
    lb = chats[0]["logit_bias"]
    check({t for t, _ in lb} == want and len({b for _, b in lb}) == 1 and lb[0][1] == 61.0,
          "equal bias 61 on exactly pos, neg and default-rule variants (word, space+word, lower, capitalised)")
    check(chats[0]["top_logprobs"] == len(want) + 32, "top_logprobs = biased tokens + k_extra")

    # exact margin and log-softmax against the stub's own logits
    errs, lp_ok, var_ok = [], True, True
    for it, r in zip(items, R):
        prompt = S.render(messages(spec, it["text"]))
        lg = S.true_logits(prompt)
        errs.append(abs(r["margin"] - (lg[st.vocab["yes"]] - lg[st.vocab["no"]])))
        lsm = lg - (lg.max() + np.log(np.exp(lg - lg.max()).sum()))
        # rebuilt log-softmax leaves out mass outside top K, so it can only be >= the true value
        lp_ok &= r["lp_pos"] >= lsm[st.vocab["yes"]] - 1e-5 and r["lp_neg"] >= lsm[st.vocab["no"]] - 1e-5
        lp_ok &= abs((r["lp_pos"] - r["lp_neg"]) - r["margin"]) < 1e-9
        var_ok &= set(r["variants"]) == {"yes", " yes", "Yes", "no", " no", "No"}
    check(max(errs) < 1e-4, f"margin = logit(pos) - logit(neg) exactly (max abs error {max(errs):.2e}; float32 probs)")
    spread = [abs(r["margin"]) for r in R]
    check(max(spread) > 10, f"margins far from 0 are not floored (largest |margin| {max(spread):.1f})")
    check(lp_ok, "lp_pos - lp_neg = margin; rebuilt log-probs >= true log-softmax")
    check(var_ok, "variants field lists the biased strings")

    X = np.load(pre1 + ".npy")
    keys = open(pre1 + ".keys.txt").read().split("\n")[:-1]
    check(X.shape == (12, S.N_EMBD) and X.dtype == np.float32, f"states [12, {S.N_EMBD}] float32, hidden size from server")
    check(keys == [it["key"] for it in items], "keys file in items order")
    check(all(np.array_equal(X[i], S.true_state(S.render(messages(spec, it["text"])))) for i, it in enumerate(items)),
          "state row i = final state of item i (embd_normalize -1, same rendered prompt)")
    embs = [b for p, b in st.requests if p == "/v1/embeddings"]
    check(len(embs) == 12 and all(b["embd_normalize"] == -1 for b in embs), "one embeddings request per item, embd_normalize -1")

    # 2. hidden size from the first vector when the server does not report it
    out2, pre2 = os.path.join(tmp, "r2.jsonl"), os.path.join(tmp, "s2")
    st.report_n_embd = False
    run(args(out=out2, states=pre2, limit=3))
    st.report_n_embd = True
    check(np.load(pre2 + ".npy").shape == (3, S.N_EMBD), "hidden size taken from the first vector when the server omits it")

    # 3. interrupted run, then --resume: same rows and states as the uninterrupted run
    out3, pre3 = os.path.join(tmp, "r3.jsonl"), os.path.join(tmp, "s3")
    st.n_chat, st.fail_after = 0, 4         # chat requests 5+ fail with HTTP 500
    rc = run(args(out=out3, states=pre3))
    st.fail_after = None
    R3 = rows(out3)
    check(rc == 2 and sum(r["err"] is None for r in R3) == 4, "server failure: 4 rows done, run stops after 5 consecutive errors")
    X3 = np.load(pre3 + ".npy")
    check(np.isnan(X3[4:]).all() and np.isfinite(X3[:4]).all(), "states of failed rows stay NaN")
    rc = run(args(out=out3, states=pre3, resume=True))
    ok = {r["key"]: r for r in rows(out3) if r["err"] is None}
    check(rc == 0 and len(ok) == 12, "--resume retries the failed rows and completes the run")
    check(all(ok[r["key"]]["margin"] == r["margin"] for r in R) and np.array_equal(np.load(pre3 + ".npy"), X),
          "resumed run: margins and states bit-identical to the uninterrupted run")
    try:
        run(args(out=out3, states=pre3, resume=False))
        check(False, "existing output refused without --resume")
    except SystemExit:
        check(True, "existing output refused without --resume")

    # 4. bias too small: every row must be an error, never a wrong margin
    out4 = os.path.join(tmp, "r4.jsonl")
    run(args(out=out4, bias=2.0, limit=4))
    R4 = rows(out4)
    check(all(r["err"] and r["margin"] is None for r in R4), "bias 2: rows recorded as errors, no margin")

    # 5. variants listed in the spec; a variant that is not one token is dropped
    spec5 = dict(spec, variants=[" yes", "YES", "N-O"])
    p5 = os.path.join(tmp, "spec5.json")
    json.dump(spec5, open(p5, "w"))
    del st.requests[:]
    out5 = os.path.join(tmp, "r5.jsonl")
    run(args(out=out5, spec=p5, limit=2))
    lb5 = {t for t, _ in [b for p, b in st.requests if p == "/v1/chat/completions"][0]["logit_bias"]}
    check(lb5 == {st.vocab[w] for w in ("yes", "no", " yes", "YES")}, "spec variants used; multi-token variant dropped")
    check(all(abs(a["margin"] - b["margin"]) < 1e-4 for a, b in zip(rows(out5), R[:2])),
          "margin does not depend on the variant set (within float32 rounding)")
    out5b = os.path.join(tmp, "r5b.jsonl")
    run(args(out=out5b, bias=75.0, limit=4))
    check(all(r["err"] is None for r in rows(out5b)) and
          all(abs(a["margin"] - b["margin"]) < 1e-4 for a, b in zip(rows(out5b), R[:4])),
          "bias 75 gives the same margins as bias 61: the bias cancels")

    # 6. answer word that is not a single token stops the run
    p6 = os.path.join(tmp, "spec6.json")
    json.dump(dict(spec, pos="y-e-s"), open(p6, "w"))
    try:
        run(args(out=os.path.join(tmp, "r6.jsonl"), spec=p6))
        check(False, "multi-token answer word stops the run")
    except SystemExit:
        check(True, "multi-token answer word stops the run")

    # 7. null system message
    p7 = os.path.join(tmp, "spec7.json")
    json.dump(dict(spec, system=None), open(p7, "w"))
    del st.requests[:]
    run(args(out=os.path.join(tmp, "r7.jsonl"), spec=p7, limit=1))
    m7 = [b for p, b in st.requests if p == "/v1/chat/completions"][0]["messages"]
    check([m["role"] for m in m7] == ["user"], "null system: user message only")

    srv.shutdown()
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"selftest: {'PASS' if not fails else 'FAIL'} ({len(fails)} failed)")
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true", help="run the checks against tests/stub_server.py and exit")
    ap.add_argument("--server")
    ap.add_argument("--spec", help="prompt spec JSON")
    ap.add_argument("--items", help="items JSONL (key, text, optional label)")
    ap.add_argument("--out", help="output rows JSONL")
    ap.add_argument("--states", help="PREFIX for PREFIX.npy / PREFIX.keys.txt (final state per item)")
    ap.add_argument("--limit", type=int, help="score only the first N items")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--bias", type=float, default=61.0,
                    help="equal logit bias on the answer tokens; must exceed the model's final-logit spread")
    ap.add_argument("--k-extra", type=int, default=32, help="unbiased tokens returned for the log-softmax")
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--template-kwargs", default='{"enable_thinking": false}',
                    help="JSON passed as chat_template_kwargs")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    for f in ("server", "spec", "items", "out"):
        if getattr(a, f) is None:
            ap.error(f"--{f.replace('_', '-')} is required")
    sys.exit(run(a))


if __name__ == "__main__":
    main()
