(** Train the Google Alerts triage classifier on CPU, report holdout accuracy, and print predictions
    for a few headlines.

    Usage: [dune exec bin/train.exe -- [options] [--predict HEADLINE]...]. Run with [--help] for the
    option list. *)

open Base
open Ocannl
open Stdio
module IDX = Train.IDX
open Nn_blocks.DSL_modules
module Asgns = Ir.Assignments
open Alert_triage

(* --- Command line ------------------------------------------------------------------------ *)

type optimizer = Adam | Sgd

type options = {
  mutable data : string;
  mutable epochs : int;
  mutable batch_size : int;
  mutable seq_len : int;
  mutable d_model : int;
  mutable num_heads : int;
  mutable d_ff : int;
  mutable num_layers : int;
  mutable lr : float;
  mutable optimizer : optimizer;
  mutable seed : int;
  mutable eval_every : int;
  mutable predict : string list;
}

(* Hand-picked headlines that do not appear verbatim in data/alerts.jsonl. *)
let default_predictions =
  [
    "银泰百货宁波店改造完成重新开业";
    "中远海运控股公布三季度净利润同比增长";
    "银泰国际娱乐城注册送888彩金 六合彩开奖";
    "安永 时时彩 幸运飞艇 开户送礼金";
    "简历模板：曾任安永审计助理";
    "宁波银泰城附近新开网红奶茶店";
    "上海家化拟以自有资金回购公司股份";
  ]

let parse_args () =
  let o =
    {
      data = "data/alerts.jsonl";
      epochs = 40;
      batch_size = 16;
      seq_len = 32;
      d_model = 32;
      num_heads = 4;
      d_ff = 64;
      num_layers = 1;
      lr = 2e-3;
      optimizer = Adam;
      seed = 42;
      eval_every = 5;
      predict = [];
    }
  in
  let specs =
    [
      ( "--data",
        Stdlib.Arg.String (fun s -> o.data <- s),
        "PATH JSONL dataset (default data/alerts.jsonl)" );
      ("--epochs", Stdlib.Arg.Int (fun i -> o.epochs <- i), "N training epochs (default 40)");
      ( "--batch-size",
        Stdlib.Arg.Int (fun i -> o.batch_size <- i),
        "N examples per step (default 16)" );
      ( "--seq-len",
        Stdlib.Arg.Int (fun i -> o.seq_len <- i),
        "N characters per headline (default 32)" );
      ("--d-model", Stdlib.Arg.Int (fun i -> o.d_model <- i), "N embedding width (default 32)");
      ("--heads", Stdlib.Arg.Int (fun i -> o.num_heads <- i), "N attention heads (default 4)");
      ("--d-ff", Stdlib.Arg.Int (fun i -> o.d_ff <- i), "N feed-forward hidden width (default 64)");
      ("--layers", Stdlib.Arg.Int (fun i -> o.num_layers <- i), "N encoder blocks (default 1)");
      ("--lr", Stdlib.Arg.Float (fun f -> o.lr <- f), "F peak learning rate (default 2e-3)");
      ( "--optimizer",
        Stdlib.Arg.Symbol
          ([ "adam"; "sgd" ], function "sgd" -> o.optimizer <- Sgd | _ -> o.optimizer <- Adam),
        " optimizer (default adam)" );
      ("--seed", Stdlib.Arg.Int (fun i -> o.seed <- i), "N split/init seed (default 42)");
      ( "--eval-every",
        Stdlib.Arg.Int (fun i -> o.eval_every <- i),
        "N epochs between holdout evals (default 5)" );
      ( "--predict",
        Stdlib.Arg.String (fun s -> o.predict <- o.predict @ [ s ]),
        "HEADLINE classify this headline after training (repeatable)" );
    ]
  in
  Stdlib.Arg.parse specs
    (fun anon -> o.predict <- o.predict @ [ anon ])
    "train.exe: train the OCANNL alert-triage classifier\n\nOptions:";
  if List.is_empty o.predict then o.predict <- default_predictions;
  o

(* --- Main -------------------------------------------------------------------------------- *)

let () =
  let o = parse_args () in
  Utils.settings.fixed_state_for_init <- Some o.seed;
  Tensor.unsafe_reinitialize ();
  let num_classes = Label.num_classes in

  (* --- Data --- *)
  let examples = Dataset.load_jsonl o.data in
  let { Dataset.train; holdout } =
    Dataset.stratified_split ~seed:o.seed ~holdout_frac:0.2 ~batch_size:o.batch_size examples
  in
  let vocab = Dataset.Vocab.build (Array.to_list (Array.map train ~f:(fun e -> e.text))) in
  let vocab_size = Dataset.Vocab.size vocab in
  let encode e = Dataset.Vocab.encode vocab ~max_len:o.seq_len e.Dataset.text in
  let n_train = Array.length train and n_holdout = Array.length holdout in
  let n_batches = n_train / o.batch_size in
  printf "Loaded %d examples from %s: %d train (%d batches of %d), %d holdout, vocab %d chars\n"
    (List.length examples) o.data n_train n_batches o.batch_size n_holdout vocab_size;
  let show_counts name arr =
    printf "  %s:" name;
    Array.iter (Dataset.label_counts arr) ~f:(fun (l, n) -> printf " %s=%d" (Label.to_string l) n);
    printf "\n"
  in
  show_counts "train  " train;
  show_counts "holdout" holdout;
  printf "%!";

  (* --- Data tensors: the whole training set lives on the device, batches are selected with the
     static batch index [@| batch_n]. --- *)
  let train_ids =
    Tensors.batched_ids ~label:"train_ids" ~n_batches ~batch_size:o.batch_size ~seq_len:o.seq_len
      (Array.map train ~f:encode)
  in
  let train_labels =
    Tensors.one_hot_tensor ~label:"train_labels" ~dims:[ n_batches; o.batch_size ] ~num_classes
      (fun i -> Label.to_int train.(i).label)
  in
  let batch_n, bindings = IDX.get_static_symbol ~static_range:n_batches IDX.empty in
  let%op batch_ids = train_ids @| batch_n in
  let%op batch_labels = train_labels @| batch_n in

  (* --- Model and loss --- *)
  let config =
    {
      Model.vocab_size;
      seq_len = o.seq_len;
      d_model = o.d_model;
      num_heads = o.num_heads;
      d_ff = o.d_ff;
      num_layers = o.num_layers;
      num_classes;
    }
  in
  let model = Model.build ~label:[ "triage" ] config in
  let%op logits = model ~train_step:None batch_ids in
  (* Mean cross-entropy over the batch; the class axis [v] is the output axis of the logits. *)
  let batch_size = o.batch_size in
  let%op batch_loss =
    Nn_blocks.cross_entropy_loss ~spec:"...|v" () ~logits ~targets:batch_labels /. !..batch_size
  in

  (* --- Optimizer --- *)
  let update = Train.grad_update batch_loss in
  let total_steps = o.epochs * n_batches in
  let warmup_steps = Int.max 1 (total_steps / 10) in
  let schedule =
    {
      Train.Lr_schedule.kind = Cosine;
      base_lr = o.lr;
      warmup_steps;
      total_steps;
      final_frac = 0.05;
    }
  in
  let learning_rate = Train.host_scalar ~l:"learning_rate" o.lr in
  let lr_at_step step =
    let lr = Train.Lr_schedule.learning_rate schedule ~step in
    match o.optimizer with Adam -> Optim.bias_corrected_lr ~lr ~step () | Sgd -> lr
  in
  let optimizer_step =
    match o.optimizer with
    | Adam -> Optim.adam_update ~learning_rate batch_loss
    | Sgd ->
        let sgd = Train.sgd_update ~learning_rate ~momentum:0.9 batch_loss in
        (* Same workaround as [Optim.state_tensor]: in OCANNL 1.0.1 the inline [{ sgd_momentum }]
           buffers are left to the inliner, which then finds no computation for them at the start of
           the next step; keep them device-resident. *)
        Set.iter (Asgns.collect_written sgd.Asgns.asgns) ~f:(fun tn ->
            if List.exists tn.Ir.Tnode.label ~f:(String.is_prefix ~prefix:"sgd_momentum") then
              Train.set_materialized tn);
        sgd
  in

  (* --- Compile: one routine for the full training step (forward + backprop + optimizer). --- *)
  let ctx = Context.auto () in
  let ctx = Train.init_params ctx bindings batch_loss in
  let step_routine = Train.to_routine ctx bindings (Asgns.sequence [ update; optimizer_step ]) in
  let ctx = step_routine.Context.context in
  let n_params =
    Set.fold (Train.trainable_params batch_loss) ~init:0 ~f:(fun acc p ->
        acc + Ir.Tnode.num_elems p.Tensor.value)
  in
  printf "Model: d_model=%d heads=%d d_ff=%d layers=%d seq_len=%d -> %d trainable parameters\n%!"
    o.d_model o.num_heads o.d_ff o.num_layers o.seq_len n_params;

  (* --- Holdout graph: same parameters, a second forward-only routine over the whole holdout set as
     one batch. --- *)
  let holdout_ids =
    Tensors.flat_ids ~label:"holdout_ids" ~seq_len:o.seq_len (Array.map holdout ~f:encode)
  in
  let%op holdout_logits = model ~train_step:None holdout_ids in
  let holdout_routine =
    Train.to_routine ctx IDX.empty
      [%cd
        ~~("holdout forward";
           holdout_logits.forward)]
  in
  let ctx = holdout_routine.Context.context in
  let holdout_truth = Array.map holdout ~f:(fun e -> Label.to_int e.label) in
  let evaluate () =
    Train.run ctx holdout_routine;
    let flat = Context.get_values ctx holdout_logits.value in
    let pred = Tensors.argmax_rows ~cols:num_classes flat in
    let correct = Array.counti pred ~f:(fun i p -> p = holdout_truth.(i)) in
    (Float.of_int correct /. Float.of_int n_holdout, pred)
  in

  (* --- Single-headline graph: a [1; seq_len] id buffer we overwrite from the host. --- *)
  let infer_ids = Tensors.ids_tensor ~label:"infer_ids" ~dims:[ 1; o.seq_len ] (fun _ -> 0) in
  Train.set_materialized infer_ids.Tensor.value;
  let%op infer_logits = model ~train_step:None infer_ids in
  let infer_routine =
    Train.to_routine ctx IDX.empty
      [%cd
        ~~("headline forward";
           infer_logits.forward)]
  in
  let ctx = infer_routine.Context.context in
  let predict headline =
    let ids = Dataset.Vocab.encode vocab ~max_len:o.seq_len headline in
    ignore
      (Context.set_values ctx infer_ids.Tensor.value (Array.map ids ~f:Float.of_int) : Context.t);
    Train.run ctx infer_routine;
    let row = Context.get_values ctx infer_logits.Tensor.value in
    let probs = Tensors.softmax_row row in
    let best = (Tensors.argmax_rows ~cols:num_classes row).(0) in
    (Label.of_int_exn best, probs)
  in

  (* --- Training loop --- *)
  let open Operation.At in
  printf "\nTraining for %d epochs (%d steps, %s, peak lr %g)...\n%!" o.epochs total_steps
    (match o.optimizer with Adam -> "adam" | Sgd -> "sgd+momentum")
    o.lr;
  let step = ref 0 in
  let t0 = Unix.gettimeofday () in
  for epoch = 1 to o.epochs do
    let epoch_loss = ref 0. in
    Train.sequential_loop step_routine.Context.bindings ~f:(fun () ->
        ignore
          (Context.set_values ctx learning_rate.Tensor.value [| lr_at_step !step |] : Context.t);
        Train.run ctx step_routine;
        epoch_loss := !epoch_loss +. (ctx, batch_loss).@[0];
        Int.incr step);
    let avg = !epoch_loss /. Float.of_int n_batches in
    if Float.is_nan avg then (
      eprintf "Loss became NaN at epoch %d; lower --lr.\n%!" epoch;
      Stdlib.exit 2);
    if epoch = 1 || epoch % o.eval_every = 0 || epoch = o.epochs then
      let acc, _ = evaluate () in
      printf "epoch %3d  train loss %.4f  holdout acc %.1f%%  (%.0fs)\n%!" epoch avg (100. *. acc)
        (Unix.gettimeofday () -. t0)
    else printf "epoch %3d  train loss %.4f\n%!" epoch avg
  done;

  (* --- Final holdout report with a confusion matrix --- *)
  let acc, pred = evaluate () in
  printf "\nHoldout accuracy: %.1f%% (%d examples; chance = %.1f%%)\n" (100. *. acc) n_holdout
    (100. /. Float.of_int num_classes);
  printf "Confusion matrix (rows = truth, columns = predicted):\n%14s" "";
  Array.iter Label.all ~f:(fun l -> printf "%13s" (Label.to_string l));
  printf "\n";
  Array.iteri Label.all ~f:(fun ti tl ->
      printf "%14s" (Label.to_string tl);
      Array.iteri Label.all ~f:(fun pi _ ->
          let n = Array.counti pred ~f:(fun i p -> holdout_truth.(i) = ti && p = pi) in
          printf "%13d" n);
      printf "\n");

  (* --- Example predictions --- *)
  printf "\nPredictions:\n";
  List.iter o.predict ~f:(fun headline ->
      let label, probs = predict headline in
      printf "  %-14s %s\n" (Label.to_string label) headline;
      printf "  %-14s" "";
      Array.iteri probs ~f:(fun i p -> printf " %s=%.2f" (Label.to_string (Label.of_int_exn i)) p);
      printf "\n");
  printf "%!"
