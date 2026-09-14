# Replacing the synthetic data with labels mined from a real digest log

`data/alerts.jsonl` is hand-written so that the sandbox trains without any credentials. Once
you have a real Google Alerts pipeline that keeps a log of what it did with each alert item,
that log is a free source of training labels. This note describes the mapping and the file
format the trainer expects; there is deliberately no Gmail code in this repository.

## 1. What the trainer reads

One JSON object per line (`JSONL`), UTF-8, with these fields:

| field   | type   | required | meaning                                                        |
|---------|--------|----------|----------------------------------------------------------------|
| `text`  | string | yes      | headline, optionally followed by the snippet (see §3)          |
| `label` | string | yes      | one of `keep`, `spam`, `soft-mention`, `dupe`                  |
| `brand` | string | no       | the alert keyword that triggered the item; only used for stats |

Anything else is ignored, so you can keep `url`, `alert_id`, `received_at`, etc. in the file.
Point the trainer at it with `dune exec bin/train.exe -- --data path/to/real.jsonl`.

## 2. Deriving the label from the digest log

Most digest pipelines already make a four-way decision per item; record it and translate:

| pipeline outcome                                                              | label          |
|-------------------------------------------------------------------------------|----------------|
| item was posted to the digest                                                 | `keep`         |
| item skipped by a gambling / SEO / domain blocklist rule (六合彩, 娱乐城, `jnh*.com`, …) | `spam`         |
| item skipped because the brand only appears as a location, résumé line, campus-recruiting roundup, or sidebar link | `soft-mention` |
| item skipped by the de-duplication step (same story already posted, e.g. cosine similarity or shared canonical URL) | `dupe`         |

Suggested rules of thumb while mining:

- Prefer *decisions a human confirmed* (manual overrides, thumbs-up/down in the digest) over raw
  heuristic outcomes when both exist; heuristics are what the model is supposed to replace.
- For `dupe`, keep the near-duplicate headline itself, not the story it duplicated. A text-only
  classifier can only learn surface cues of re-reported stories (terse rewrites, wire-service
  phrasing); the real de-duplication signal is *history*, so consider feeding the model
  `"<headline> ||| <best matching previously-posted headline>"` as `text` if you want it to learn
  the comparison rather than the style.
- Items with no decision recorded (e.g. pipeline crashed) should be dropped, not defaulted.

## 3. Text field

The synthetic set uses headline only. Real alerts also carry a snippet; appending it helps the
`soft-mention` class most (the location / résumé context is usually in the snippet, not the
title). Keep it short: the model truncates to `--seq-len` characters (default 96), so raise that
further when including long snippets, and expect training to slow down roughly linearly.

Normalisation the tokenizer already does: ASCII case-folding. It does *not* strip HTML entities
or collapse whitespace, so do that during export.

## 4. Sanity checks before training

```sh
# label balance -- the split is stratified, but every class needs at least ~10 examples
jq -r .label real.jsonl | sort | uniq -c

# duplicated texts leak between train and holdout; remove exact duplicates
jq -r .text real.jsonl | sort | uniq -d | head
```

Then run `dune exec bin/train.exe -- --data real.jsonl --epochs 40` and compare the confusion
matrix with the synthetic run. With a few thousand real items, raise `--d-model 64 --layers 2`.
