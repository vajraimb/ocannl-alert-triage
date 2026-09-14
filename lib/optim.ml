(** Adam, written in OCANNL's [%cd] assignment syntax.

    [Train] ships SGD (with momentum / Nesterov / weight decay) but no Adam, so this is the "recipe
    book" extension the OCANNL migration guide suggests. Each parameter gets two non-differentiable
    state tensors ([adam_m], [adam_v]); they are allocated zero-filled on the device and persist
    across routine runs.

    Bias correction is folded into the learning-rate scalar on the host (see {!bias_corrected_lr}),
    which is the update rule from Kingma & Ba, Section 2:
    [lr_t = lr * sqrt (1 - beta2^t) / (1 - beta1^t)] and [p -= lr_t * m / (sqrt v + eps)]. *)

open Base
open Ocannl
open Nn_blocks.DSL_modules
module Asgns = Ir.Assignments

(** Optimizer state that outlives a routine run: a non-differentiable tensor whose shape is inferred
    from its use, materialized so the compiler keeps it in a device buffer (zero-filled at
    allocation) instead of treating it as an inlinable temporary. *)
let state_tensor ~name p =
  let t = NTDSL.term ~label:(name :: p.Tensor.value.Ir.Tnode.label) () in
  Train.set_materialized t.Tensor.value;
  t

let adam_one ~learning_rate ~beta1 ~beta2 ~epsilon p =
  if Option.is_none p.Tensor.diff then
    raise @@ Tensor.Session_error ("Optim.adam_one: not differentiable", Some p);
  let one_minus_beta1 = 1. -. beta1 in
  let one_minus_beta2 = 1. -. beta2 in
  let adam_m = state_tensor ~name:"adam_m" p in
  let adam_v = state_tensor ~name:"adam_v" p in
  [%cd
    ~~(p "param adam step";
       (* m <- beta1 * m + (1 - beta1) * grad. Gradients may only appear as direct operands of an
          assignment, hence the two statements. *)
       adam_m =: !.beta1 *. adam_m;
       adam_m =+ !.one_minus_beta1 * p.grad ~logic:".";
       (* v <- beta2 * v + (1 - beta2) * grad^2 *)
       adam_v =: !.beta2 *. adam_v;
       { adam_grad_sq } =: p.grad * p.grad ~logic:".";
       adam_v =+ !.one_minus_beta2 * adam_grad_sq ~logic:".";
       (* p <- p - lr_t * m / (sqrt v + eps); the right-hand side is a plain (non-differentiable)
          tensor expression that the compiler inlines. *)
       { adam_delta } =: adam_m /. (sqrt adam_v + !.epsilon);
       p =- learning_rate * adam_delta ~logic:".")]

(** Maps {!adam_one} over the parameters the loss actually trains. *)
let adam_update ~learning_rate ?(beta1 = 0.9) ?(beta2 = 0.999) ?(epsilon = 1e-8) loss =
  let comp =
    Set.to_list (Train.trainable_params loss)
    |> List.map ~f:(adam_one ~learning_rate ~beta1 ~beta2 ~epsilon)
    |> Asgns.sequence
  in
  { comp with asgns = Asgns.Block_comment ("adam_update", comp.asgns) }

(** The learning rate to feed the device before optimizer step [step] (0-based). *)
let bias_corrected_lr ~lr ?(beta1 = 0.9) ?(beta2 = 0.999) ~step () =
  let t = Float.of_int (step + 1) in
  lr *. Float.sqrt (1. -. (beta2 **. t)) /. (1. -. (beta1 **. t))
