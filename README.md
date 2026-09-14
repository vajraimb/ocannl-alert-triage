# alert_triage — an OCANNL text-classifier sandbox

A small, trainable prototype that sorts Google Alerts headlines (mixed Chinese/English) into four
triage classes, written against [OCANNL](https://github.com/ahrefs/ocannl) **1.0.1** (opam package
`neural_nets_lib`):

| label          | meaning                                                            |
|----------------|--------------------------------------------------------------------|
| `keep`         | real corporate / finance news worth putting in a digest            |
| `spam`         | gambling, SEO, name-collision junk (银泰娱乐城, 六合彩, `jnh*.com`, …) |
| `soft-mention` | brand only appears as a location, résumé line, campus-recruit list |
| `dupe`         | same story already covered (near-duplicate headline)               |

The point is to learn OCANNL idioms on a realistically shaped problem — `%op` model code, shape
inference with named axes, the generalized einsum, compiling one training routine, a hand-rolled
optimizer in `%cd` — not to reach production accuracy. There is no Gmail integration and no GPU
requirement: everything runs on the CPU `cc` backend in a few seconds.

## Layout

```
dune-project, alert_triage.opam(.template)  package definition; pins OCANNL 1.0.1
ocannl_config          OCANNL runtime config (backend=cc, single precision, fixed init seed)
data/alerts.jsonl      145 synthetic alert headlines (~35 per class, 7 brands) + 14 real ones (Sep 2026)
lib/label.ml           the four classes <-> class ids
lib/dataset.ml         JSONL loading, UTF-8 code-point tokenizer, stratified split
lib/tensors.ml         host arrays -> OCANNL data tensors, argmax / softmax helpers
lib/model.ml           the transformer classifier (%op)
lib/optim.ml           Adam written in %cd (Train ships only SGD)
bin/train.ml           train, evaluate on the holdout split, classify headlines
scripts/export_labels.md  how to replace the synthetic data with labels mined from a digest log
```

## Install

Requirements: opam ≥ 2.1, a C compiler (`gcc`/`clang`) for the `cc` backend, and the usual
`pkg-config`, `m4`, `libgmp-dev`. OCANNL's dependency closure also needs `libffi-dev`, `zlib1g-dev`,
`libsqlite3-dev`, `libcurl4-openssl-dev`, `libpcre2-dev` (opam's depext step installs these for you
when you confirm).

```sh
git clone https://github.com/vajraimb/ocannl-alert-triage.git && cd ocannl-alert-triage
opam switch create . 5.3.0 --no-install     # or reuse any OCaml >= 5.3 switch
eval $(opam env)
# Pins arrayjit + neural_nets_lib to the 1.0.1 release tag (via pin-depends in the opam file),
# follows OCANNL's own pins (ppx_minidebug, notty-community, dataprep dev branches), then builds
# ~140 packages. Takes 10-20 minutes the first time.
opam install -y --confirm-level=unsafe-yes --deps-only .
dune build
```

Why the pin: opam-repository currently publishes `neural_nets_lib` up to 0.6.1, whose API is quite
different (no `Context`, no `cross_entropy_loss`, …). The 1.0.1 tag is the latest release on
GitHub as of writing; `alert_triage.opam.template` records it so a plain `opam install --deps-only`
picks it up. If OCANNL ≥ 1.0.1 lands in opam-repository, delete the template and the pin.

## Train

```sh
dune exec bin/train.exe               # defaults: 40 epochs, batch 16, d_model 32, 4 heads, 1 layer, Adam
dune exec bin/train.exe -- --help     # all options
dune exec bin/train.exe -- --layers 2 --d-model 64 --epochs 60 --lr 1e-3
dune exec bin/train.exe -- --optimizer sgd --lr 0.05
```

`ocannl_config` is looked up by walking up from the current directory, so run from the repo root
(or set `OCANNL_BACKEND=cc` etc. in the environment).

A default run on a 4-core CPU takes about 12 seconds end to end (C compilation of the routines
included) and prints:

```
Loaded 159 examples from data/alerts.jsonl: 112 train (7 batches of 16), 47 holdout, vocab 643 chars
  train  : keep=34 spam=26 soft-mention=29 dupe=23
  holdout: keep=14 spam=13 soft-mention=9 dupe=11
Model: d_model=32 heads=4 d_ff=64 layers=1 seq_len=96 -> 32164 trainable parameters

Training for 40 epochs (280 steps, adam, peak lr 0.002)...
epoch   1  train loss 1.3907  holdout acc 21.3%  (0s)
epoch   5  train loss 0.9746  holdout acc 55.3%  (1s)
epoch  10  train loss 0.1986  holdout acc 55.3%  (2s)
epoch  20  train loss 0.0118  holdout acc 57.4%  (4s)
epoch  40  train loss 0.0053  holdout acc 57.4%  (9s)

Holdout accuracy: 57.4% (47 examples; chance = 25.0%)
Confusion matrix (rows = truth, columns = predicted):
                       keep         spam soft-mention         dupe
          keep            6            0            4            4
          spam            1           11            1            0
  soft-mention            1            2            6            0
          dupe            5            0            2            4
```

The split is stratified and seeded (`--seed`); 20% of each class is held out and the training set is
truncated to a multiple of the batch size (leftovers join the holdout). The vocabulary is built from
the training texts only, so unseen characters map to `UNK` exactly as they would in production.

## Classify a headline

There is no separate weights file: with a 12-second training run, the simplest correct flow is to
train and predict in one process. Any positional argument or `--predict` value is classified after
training, using a third forward graph that shares the trained parameters:

```sh
dune exec bin/train.exe -- --predict "银泰百货宁波店改造完成重新开业" "六合彩 银泰娱乐 开户送彩金"
```

Without `--predict` the program classifies seven built-in headlines that are *not* in the training
data. From the run above:

```
Predictions:
  dupe           银泰百货宁波店改造完成重新开业
                 keep=0.00 spam=0.00 soft-mention=0.13 dupe=0.87
  keep           中远海运控股公布三季度净利润同比增长
                 keep=1.00 spam=0.00 soft-mention=0.00 dupe=0.00
  spam           银泰国际娱乐城注册送888彩金 六合彩开奖
                 keep=0.00 spam=1.00 soft-mention=0.00 dupe=0.00
  spam           安永 时时彩 幸运飞艇 开户送礼金
                 keep=0.00 spam=1.00 soft-mention=0.00 dupe=0.00
  soft-mention   简历模板：曾任安永审计助理
                 keep=0.00 spam=0.00 soft-mention=0.93 dupe=0.07
  soft-mention   宁波银泰城附近新开网红奶茶店
                 keep=0.00 spam=0.00 soft-mention=0.92 dupe=0.08
  keep           上海家化拟以自有资金回购公司股份
                 keep=0.99 spam=0.00 soft-mention=0.00 dupe=0.00
```

`spam` is easy (the vocabulary of gambling sites is disjoint); `keep` vs `dupe` is the hard pair
because in the synthetic set a dupe *is* a keep headline reworded more tersely, so the model learns
"short = dupe". That is a data artifact, not a model bug — see `scripts/export_labels.md` for how
real de-duplication labels (and pairing the headline with the story it duplicates) would fix it.

## The model, in OCANNL terms

`lib/model.ml`:

1. **Ids -> embedding.** Token ids are a `[b s |]` tensor (two *batch* axes, no output axes).
   `Nn_blocks.one_hot_of_ids` turns them into a logical one-hot `[b s | v]` (`range v = ids`, never
   materialized), and `{ tok_embed; o = [ d_model ] } * one_hot` is the embedding lookup: the inline
   `{ … }` declares a learnable parameter whose *input* axis `v` is inferred from the one-hot.
2. **Positions.** Parameters cannot have batch axes, so the learned positional table is
   `{ pos_embed; o = [ d_model ] } * pos_one_hot` where `pos_one_hot : [s | s']` is a constant
   identity matrix (`one_hot_of_ids ~num_classes:seq_len (position_indices ~seq_len ())`). The
   `[s | d]` result broadcasts over the example axis when added.
3. **Encoder blocks** (`encoder_block`, post-norm): `Nn_blocks.multi_head_attention` +
   residual + `layer_norm`, then `Nn_blocks.mlp` (FFN) + residual + `layer_norm`. Inside the
   attention, the einsum `"... s | h d; ... t | h d => ... s | t -> h"` keeps query positions `s` in
   the batch row and moves key positions `t` into the *input* row, so scores are `[b s | t -> h]`,
   softmax runs over `t`, and `"... s | t -> h; ... t | h e => ... s | h e"` contracts `t` with the
   values. Head count and per-head width are fixed with `Shape.set_dim h num_heads` etc. in
   `nn_blocks.ml`; `w_o`'s output width is inferred from the residual connection.
4. **Padding mask**, derived in-graph from the ids: `mask = ids > 0.` (`[b s | 1]`),
   `attn_mask = mask ++ "... t | ... => ... 0 | t -> 0"` (`[b 1 | t -> 1]`, which broadcasts against
   the scores — rows align from the right, hence the explicit size-1 query axis) is passed to the
   attention's `~mask`, filling masked keys with `-inf` before the softmax.
5. **Masked mean pool**: `((encoded *. mask) ++ "... s | d => ... | d") /. lengths` with
   `lengths = mask ++ "... s | ... => ... | 0"`. The einsum reduces the position axis out of the
   batch row while keeping `d`.
6. **Head**: `({ w_cls } * pooled) + { b_cls = 0.; o = [ num_classes ] }`.

`bin/train.ml`:

- The whole training set lives on the device as `[n_batches; batch_size; seq_len]`; a static
  symbol `batch_n` slices one batch per step with `train_ids @| batch_n`, and
  `Train.sequential_loop` iterates it. The only per-step host traffic is writing the learning
  rate and reading back the scalar loss.
- Loss: `Nn_blocks.cross_entropy_loss ~spec:"...|v" () ~logits ~targets /. !..batch_size`
  (fused log-sum-exp, backward writes `probs - targets` straight into the logits gradient).
- One compiled routine does forward + backprop + optimizer step:
  `Train.to_routine ctx bindings (Asgns.sequence [ Train.grad_update loss; optimizer ])`.
- Two more forward-only routines (holdout set, single headline) are compiled from the same context
  and therefore share parameters; the headline buffer is overwritten with `Context.set_values`.
- The learning rate is a `Train.host_scalar` driven by `Train.Lr_schedule` (warmup + cosine);
  for Adam the bias correction is folded into it (`Optim.bias_corrected_lr`).
- `lib/optim.ml` is Adam in eight `%cd` assignments per parameter, mirroring `Train.sgd_one`.

## Known limitations vs. PyTorch

- **Static shapes, one routine per shape.** Batch size and sequence length are compile-time
  constants of the routine; a different batch size means compiling another forward graph (as done
  for the holdout set and the single headline). No dynamic padding, no variable-length batches.
- **No dynamic indexing.** Embedding lookup is a one-hot matmul that the compiler turns into a
  gather; token ids must be floats here (see pitfalls). No `torch.gather`/`index_select` API.
- **No autograd on host-side control flow, no eager mode.** Everything is built as a graph and
  compiled to C; debugging means reading OCANNL's `.cd`/generated C dumps rather than printing
  tensors mid-forward (`Train.printf_tree` / `Context.get_values` work only for materialized
  nodes).
- **Optimizers and loop utilities are recipes, not a library.** `Train` has SGD (+momentum,
  Nesterov, weight decay), LR schedules, gradient clipping; Adam/AdamW you write yourself (`lib/optim.ml`).
  No `DataLoader`, no `nn.Module` state dict — parameter persistence is `Persistence.save/restore`
  keyed by tensor ids.
- **No dropout in this model** (`Nn_blocks.dropout` exists and needs a `~train_step` counter for its
  RNG), no weight decay, no early stopping: with 112 training examples the model memorizes the
  training set (loss → 0.005) and the holdout number reflects that.
- **BatchNorm has no running statistics** in `Nn_blocks` 1.0.1; LayerNorm (used here) is fine.
- **CPU only in this setup.** OCANNL has CUDA/HIP/Metal backends (`backend=cuda` in
  `ocannl_config`), but the default path and everything tested here is the `cc` backend.

## Pitfalls met along the way (OCANNL 1.0.1)

- **Integer token ids zero the embedding.** `Nn_blocks.token_ids_of_batch` stores ids as
  `uint32` for the integer-native gather, but that precision propagates through `one_hot_of_ids`
  into the embedding matrix, whose uniform init then truncates to 0 and nothing learns. Storing the
  ids as `single` floats (`lib/tensors.ml`) takes the documented "double-precision guard" path.
- **Optimizer state must be materialized.** Inline `{ sgd_momentum }` / `{ adam_m }` buffers
  inside `%cd` are left to the inliner, which at the next step has no computation for them
  ("Stale optimize_ctx"). `Optim.state_tensor` creates them with `NTDSL.term` and
  `Train.set_materialized`; `bin/train.ml` does the same for `Train.sgd_update ~momentum`.
- **Pointwise broadcasting aligns rows from the right**, like NumPy: a `[b | t -> 1]` mask against
  `[b s | t -> h]` scores unifies `b` with `s`. Insert a size-1 axis (`... 0 |`) explicitly.
- **Parameters have no batch axes**, so a positional embedding is a matmul with an identity
  matrix rather than a `[seq_len | d]` parameter.

## Data

`data/alerts.jsonl`: one `{"text", "label", "brand"}` object per line. 145 lines are hand-written in the
style of Chinese Google Alert headlines for 上海家化, 安永, 东方希望, 日照钢铁, 美特斯邦威, 银泰百货 and 中远海运.
`spam` items are gambling/SEO false matches (银泰国际娱乐, 六合彩, `jnh*.com`), `soft-mention`
items use the brand as a location, résumé line or campus-recruiting roundup, and every `dupe` is a
reworded `keep` headline. The last 14 lines (`"source": "gmail-2026-09"`) are real alert headlines
from Sep 7–13 2026 with human labels; note that real spam is often formal PR copy or a name collision
rather than gambling vocabulary. `scripts/export_labels.md` explains how to add more labels mined
from a real digest log.
