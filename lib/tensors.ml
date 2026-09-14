(** Host-side arrays -> OCANNL data tensors. *)

open Base
open Ocannl
open Nn_blocks.DSL_modules

(** Token ids as a tensor with batch axes [dims] (e.g. [[n_batches; batch_size; seq_len]]) and no
    output axes; [f flat_index] supplies the id of the [flat_index]-th position in row-major order.

    The ids are stored as single-precision floats (exact up to 2^24, far beyond any character
    vocabulary). [Nn_blocks.token_ids_of_batch] would store them as [uint32] for an integer-native
    embedding gather, but in OCANNL 1.0.1 that integer precision propagates through [one_hot_of_ids]
    into the embedding matrix, whose uniform initialization then truncates to zero and the network
    cannot learn. Float ids take the documented "double-precision guard" path. *)
let ids_tensor ~label ~dims (f : int -> int) =
  let open Bigarray in
  let ga = Genarray.create Float32 c_layout (Array.of_list dims) in
  let total = List.fold dims ~init:1 ~f:( * ) in
  let flat = reshape_1 ga total in
  for i = 0 to total - 1 do
    Array1.set flat i (Float.of_int (f i))
  done;
  TDSL.wrap ~l:label ~b:dims ~o:[] (Ir.Ndarray.as_array Ir.Ops.Single ga) ()

(** Sequences [seqs.(i)] (each of length [seq_len]) laid out as [[n_batches; batch_size; seq_len]].
*)
let batched_ids ~label ~n_batches ~batch_size ~seq_len (seqs : int array array) =
  ids_tensor ~label ~dims:[ n_batches; batch_size; seq_len ] (fun i ->
      seqs.(i / seq_len).(i % seq_len))

(** Sequences laid out as [[num_seqs; seq_len]] (a single evaluation batch). *)
let flat_ids ~label ~seq_len (seqs : int array array) =
  ids_tensor ~label ~dims:[ Array.length seqs; seq_len ] (fun i -> seqs.(i / seq_len).(i % seq_len))

(** Dense one-hot targets in single precision, batch axes [dims] (e.g. [[n_batches; batch_size]]),
    output axis [num_classes]. *)
let one_hot_tensor ~label ~dims ~num_classes (class_of : int -> int) =
  let open Bigarray in
  let total = List.fold dims ~init:1 ~f:( * ) in
  let ga = Genarray.create Float32 c_layout (Array.of_list (dims @ [ num_classes ])) in
  Genarray.fill ga 0.;
  let flat = reshape_1 ga (total * num_classes) in
  for i = 0 to total - 1 do
    Array1.set flat ((i * num_classes) + class_of i) 1.
  done;
  TDSL.wrap ~l:label ~b:dims ~o:[ num_classes ] (Ir.Ndarray.as_array Ir.Ops.Single ga) ()

(** Row-wise argmax over a flat [rows * cols] logits array. *)
let argmax_rows ~cols (flat : float array) : int array =
  let rows = Array.length flat / cols in
  Array.init rows ~f:(fun r ->
      let best = ref 0 in
      for c = 1 to cols - 1 do
        if Float.( > ) flat.((r * cols) + c) flat.((r * cols) + !best) then best := c
      done;
      !best)

(** Softmax of one logits row, for reporting confidences. *)
let softmax_row (row : float array) : float array =
  let m = Array.fold row ~init:Float.neg_infinity ~f:Float.max in
  let e = Array.map row ~f:(fun x -> Float.exp (x -. m)) in
  let z = Array.fold e ~init:0. ~f:( +. ) in
  Array.map e ~f:(fun x -> x /. z)
