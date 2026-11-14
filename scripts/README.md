# Scripts: exact first-token scores and the probe gap for a yes/no LLM judge

Three scripts for a judge model served by llama.cpp's `llama-server` that answers with one word.

| script | what it does | needs |
|---|---|---|
| `score_exact.py` | exact `logit(pos) − logit(neg)` at the answer position for every item, and optionally the final hidden state | Python 3 + numpy; a running `llama-server` |
| `probe_gap.py` | fits a logistic-regression probe on the final states and reports how much AUC the judge leaves unused | numpy, scikit-learn |
| `compare_states.py` | checks whether two state files for the same items are bit-identical | numpy |

`tests/stub_server.py` is a stand-in for `llama-server` used by `score_exact.py --selftest`. It has no model.

## 1. `score_exact.py`

### Server

Start `llama-server` with your model, its chat template, and one slot:

```
llama-server -m MODEL.gguf --jinja -np 1 --host 127.0.0.1 --port 8080 [-ngl 99 -c 16384 -fa on]
```

If you want final states (`--states`), add `--embeddings --pooling last`. That server answers chat completions and embeddings both. With `--embeddings` the server sets the batch size equal to the micro-batch size, so the margins can differ slightly from those of a server started without it. Score with one server configuration throughout.

Use `-np 1`. With several slots, requests are batched together and the margins move.

### Inputs

Prompt spec, a JSON file:

```json
{
  "system": "You check text for ... Answer with one word: <POS> or <NEG>.",
  "user_template": "Text:\n{text}",
  "pos": "<POS>",
  "neg": "<NEG>",
  "variants": ["<optional>", "<token strings>"]
}
```

- `system` may be `null`. In that case no system message is sent.
- `user_template` must contain `{text}` exactly once.
- `pos` is the word for a positive answer (e.g. "this is an attack") and `neg` the word for a negative one. Each must be a single token in the model's vocabulary, and it must stay that token when appended to the rendered prompt. The script checks both and stops if either fails.
- `variants` is optional. It lists more token strings that get the same bias, so that their probabilities are reported too. If it is absent, the script uses four forms of each answer word: the word itself, the word with a leading space, lower case, and capitalised. Any form that is not a single token is dropped and listed in the header line. The study behind these scripts listed its variants by hand. The variants do not affect the margin.

Items, a JSONL file with one object per line: `key` (unique), `text`, and optionally `label`.

### Command

```
python score_exact.py --server http://127.0.0.1:8080 --spec spec.json --items items.jsonl \
    --out scores.jsonl [--states states] [--resume] [--limit N] [--bias 61] [--k-extra 32] \
    [--template-kwargs '{"enable_thinking": false}']
```

### Method

- Each item gets one `/v1/chat/completions` request, sent one at a time. The request uses `cache_prompt: false`, so every item is a fresh pass and its score does not depend on which items came before it.
- The request also sets `max_tokens 1`, `temperature 1`, `samplers ["temperature"]`, `post_sampling_probs true` and `top_logprobs K`, where K = (number of biased tokens) + `--k-extra`. The returned probabilities are then the full-vocabulary softmax after the bias.
- An equal logit bias B (`--bias`) goes on `pos`, `neg` and every variant. If B is larger than the spread of the model's final logits, the biased tokens rank above all others and always come back in the list. In the difference the bias cancels: `log p(pos) − log p(neg) = logit(pos) − logit(neg)`. The margin is therefore exact, and it is not floored, however unlikely either word is.
- The script checks on every row that all biased tokens outrank all unbiased ones. A row that fails is written with an error and no margin. The default B = 61 fits models whose final logits are softcapped at 30, where the spread is at most 60. For a model without a softcap, pick B above its logit spread; errors on every row mean B is too small.
- With `--states PREFIX`, the same rendered prompt goes to `/v1/embeddings` with `embd_normalize: -1` right after the chat pass. This returns the un-normalised final hidden state at the last prompt token, which is the output of the last norm before the output matrix. The server reuses the chat pass's cache, so the extra cost is about one token. Every row checks that both passes saw the same number of prompt tokens. The hidden size is read from the server (`/v1/models`, `meta.n_embd`) or, if the server does not report it, from the first vector.
- `chat_template_kwargs` defaults to `{"enable_thinking": false}`. Change it with `--template-kwargs` if your template needs something else.

### Output

`scores.jsonl` has one line per item:

| field | meaning |
|---|---|
| `key`, `label` | copied from the item |
| `ntok` | prompt tokens evaluated |
| `margin` | `logit(pos) − logit(neg)`, exact. Above 0 means the judge's answer leans to `pos` |
| `lp_pos`, `lp_neg` | log-probabilities of the two words |
| `variants` | log-probability of each biased string |
| `top5` | the five most likely returned tokens as `[piece, log-prob]` |
| `s` | seconds for the row |
| `err` | `null`, or the error message |

The log-probabilities are rebuilt from the returned tokens, so they leave out the mass of tokens outside the top K. The margin does not depend on that.

`--states PREFIX` also writes:
- `PREFIX.npy`: float32, shape [items, hidden size], in item order. A NaN row means not done.
- `PREFIX.keys.txt`: one key per line.

The state is written to disk before its row. `--resume` skips keys whose row has `err` null and retries the others. The run stops after 5 consecutive errors.

## 2. `probe_gap.py`

### Inputs

- **States:** a `.npy` file with one row per item. `score_exact.py --states` writes these.
- **Rows:** a CSV file with a header, or a JSONL file, with the fields `key`, `label` (1 = positive, e.g. attack; 0 = negative) and `score` (the judge's own score, higher = more positive, e.g. `margin`).
  - Optional field `stratum`: if present on the tuning rows, the cross-validation folds are stratified on stratum × label rather than on label alone.
  - If `PREFIX.keys.txt` sits beside `PREFIX.npy`, rows are matched to states by key. Otherwise the rows must be in the same order as the states.
- You need one tuning set (to fit) and one test set (to score), collected separately. Several test files are concatenated in the order given.

To turn a `score_exact.py` output into a rows file: take `key`, `label` and `margin` (as `score`) from `scores.jsonl`.

### Command

```
python probe_gap.py --tune-states tune.npy --tune-rows tune.csv \
    --test-states test.npy [test2.npy ...] --test-rows test.jsonl [test2.jsonl ...] \
    [--c-grid 1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,1] [--cv-folds 5] [--seed 0] [--nboot 2000] \
    [--judge-threshold 0] [--json out.json] \
    [--learning-curve --sizes 50,100,200,400,800 --draws 10 --lc-folds 3 [--lc-c-grid ...]]
```

### Method

1. **The probe.** An L2 logistic regression with balanced class weights, fitted on tuning states standardised with the tuning set's mean and sd.
   - C is chosen from `--c-grid` by stratified `--cv-folds`-fold cross-validation on the tuning set. The criterion is the mean fold AUC; on a tie the first C wins.
   - The probe is then refitted on all tuning items.
2. **Scoring on the test set.** The script prints:
   - **judge AUC** and **probe AUC**;
   - **lead** = probe AUC − judge AUC, with a 95 % interval from a paired bootstrap over test items. Negatives and positives are resampled separately, and each resample is shared by both scores. The probe is not refitted, so the interval covers test-set sampling only;
   - **positives caught at the judge's false-alarm count.** The judge marks `score > --judge-threshold`, and k is the number of test negatives it marks. The probe's threshold is then set on the test set so that it also marks k negatives, and both recalls are printed.
3. **`--learning-curve`.** For each n in `--sizes`:
   - draw `--draws` label-stratified random samples of n tuning items, with at least 6 per class;
   - choose C by `--lc-folds`-fold CV inside each sample, using `--lc-c-grid` or, if it is not given, `--c-grid`;
   - fit, score the full test set, and print the mean, sd, min and max of the lead.

   One random generator, seeded with `--seed`, serves all sizes in the order given, so the draws for one size depend on the sizes listed before it.

### Reading the output

- The probe's AUC is what a linear readout gets from the model's final state under your prompt. It is a level to aim for, not a ceiling on what the model "knows".
- The lead is the part of that level the prompt does not get into the answer:
  - a large lead means prompt work can pay;
  - a lead near your wording noise means it is time to stop rewriting. To measure wording noise, score the same prompt rewritten a few ways with the same meaning and take the sd of the judge AUC.
- A small tuning sample weakens the probe, so the learning-curve leads are biased low. On average they understate the unused part; single small-sample draws can overstate it.

## 3. `compare_states.py`

```
python compare_states.py A.npy B.npy
```

- If `A.keys.txt` and `B.keys.txt` exist, B's rows are matched to A's by key.
- The script prints whether the files are bit-identical, comparing the same bytes, NaN patterns included.
- If they are not, it prints the number of differing rows and values and the largest absolute difference.
- Exit status: 0 identical, 1 different, 2 not comparable.

Typical use: an edit that touches only the output matrix must leave the final states unchanged, so the states from the edited and the unedited model should be bit-identical.

## Self-tests

```
python score_exact.py --selftest      # against tests/stub_server.py, no model
python probe_gap.py --selftest        # synthetic data
python compare_states.py --selftest   # synthetic arrays
```

## Tested

Everything below was run with Python 3.14.4, numpy 2.5.4 and scikit-learn 1.9.1, on Linux.

**`score_exact.py --selftest`: PASS, 27 checks, against the stub server only.** The stub has a 4,096-token vocabulary with logits softcapped at 30 and a 24-float hidden state. The checks:
- Request shape on every chat request: `cache_prompt false`, `max_tokens 1`, temperature 1, `samplers ["temperature"]`, `post_sampling_probs`, and `top_logprobs` = biased tokens + k-extra. Requests run strictly one at a time.
- Bias: an equal bias of 61 lands on exactly pos, neg and the default-rule variants. Explicit spec variants are used when given, and a multi-token variant is dropped.
- Margin: equal to the stub's true `logit(pos) − logit(neg)` within 4.5e-8 (the stub returns float32 probabilities). Margins beyond ±30 are not floored. Changing the variant set or the bias (61 → 75) leaves the margin unchanged within 1e-4.
- Rebuilt log-probabilities: `lp_pos − lp_neg = margin`, and neither is below the true log-softmax.
- Bias too small (2): every row is recorded as an error, with no margin.
- States: rows equal the stub's states. The hidden size is taken from the server, or from the first vector when the server does not report it. The keys file is in item order.
- Interrupted run: an injected server failure stops the run after 5 consecutive errors, and the failed rows' states stay NaN. `--resume` completes the run, and its margins and states are bit-identical to an uninterrupted run. An existing output is refused without `--resume`.
- A multi-token answer word stops the run. A null system prompt sends a user message only.

**The scorer after generalisation was verified against the stub only.** It was not re-run against a real `llama-server` or a real model. The method it implements (the equal bias, fresh passes, and states through `/v1/embeddings`) was used in the study with an earlier, non-general version of the script.

**`probe_gap.py --selftest`: PASS, 12 checks, on synthetic data** (600 tuning and 800 test items, 150-d states, label signal in a 3-d latent). The checks:
- AUC on a hand example.
- The matched threshold marks exactly k negatives for every k.
- A weak judge (AUC 0.685): probe AUC 0.914, with a lead whose interval excludes 0.
- The bootstrap interval contains the point lead.
- At the judge's false-alarm count the probe marks the same number of negatives.
- A judge that already uses the signal shows no significant lead (−0.007).
- CSV and JSONL inputs give the same result, and shuffled rows are realigned by the keys file.
- The learning curve skips sizes not below the tuning-set size.

**`compare_states.py --selftest`: PASS, 7 checks, on synthetic arrays.** The checks:
- A copy is bit-identical.
- Four changed rows are found, including a 1-ulp change, and the largest difference is reported.
- A NaN present in one file only is detected.
- +0 vs −0 is reported as not bit-identical, with a largest difference of 0.
- Rows stored in another order are matched by key.

Also run on two synthetic 500 × 128 files: identical, then 2 changed rows.

**`probe_gap.py` on the study's stored data.** The data: the final states and margins of a 26B judge model under three prompts, with 1,579 tuning items and 1,643 test items (579 positives). The run used the CPU only, the default C grid, 5 folds stratified on data source × label, seed 0 and 2,000 resamples. It reproduced the published figures (published judge / probe AUC: 0.816 / 0.970, 0.941 / 0.981, 0.966 / 0.981):

| prompt | judge AUC | probe AUC | lead [95 % CI] | published lead |
|---|---|---|---|---|
| production | 0.8156 | 0.9695 | +0.1540 [+0.1340, +0.1746] | 0.154 [0.134, 0.175] |
| hand-tuned | 0.9413 | 0.9806 | +0.0393 [+0.0271, +0.0528] | 0.039 [0.027, 0.053] |
| best | 0.9658 | 0.9806 | +0.0148 [+0.0073, +0.0227] | 0.015 [0.007, 0.023] |

The learning curve was run for the production prompt with `--sizes 50,100,200 --draws 10 --lc-folds 3 --lc-c-grid 1e-4,1e-3,1e-2,1e-1`. At n = 200 the lead was +0.116 ± 0.014 (min +0.082, max +0.135), equal to the published value.

**`compare_states.py` on the study's stored data.** The states of 1,579 items from a model with an output-matrix edit were compared with those of the same model unedited: bit-identical, 1,579 / 1,579. The same items under a different quantisation of the model were reported as different in all 1,579 rows.

## Differences from the scripts used in the study

- **Answer-word variants.** The study listed the variants that receive the bias by hand. The default rule here (the word itself, with a leading space, lower-case, capitalised) does not add an upper-case form; list variants in the prompt spec to control this.
- **No readout from the model file.** The study also recomputed the margin from the final state and the model's two output rows. That check, quoted in sections 5 and 8 of the write-up, cannot be reproduced with these scripts.
- **Matched false-alarm count.** `probe_gap.py` matches the probe to the judge's false-alarm count over all negatives in the test file. The write-up's "recall at the judge's band-page count" matches on the band pages only; to reproduce it, pass a test file whose negatives are only those pages.
- **No subsample-scoring mode.** The write-up's "200 fitted, 200 scored" row scores each fit on random subsamples of the test set; the learning-curve mode here scores on the full test set.
