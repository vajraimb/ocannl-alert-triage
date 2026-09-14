(** The classifier network, expressed with OCANNL's [%op] syntax.

    Shapes follow OCANNL's three-row convention [batch | input -> output]. Token ids arrive as a
    [batch: examples, positions] tensor with no output axes; the network turns them into
    [batch: examples | output: d_model] and finally [batch: examples | output: num_classes] logits.

    Data flow:
    - token ids -> logical one-hot -> [tok_embed] matmul (an embedding lookup)
    - plus a learned positional embedding, added through a constant identity matrix over positions
    - [num_layers] post-norm encoder blocks ({!encoder_block}): multi-head self-attention with a
      key-padding mask, residual, LayerNorm, then FFN, residual, LayerNorm
    - masked mean pool over the position axis (einsum reduction, divided by the real length)
    - linear classifier to [num_classes] logits. *)

open Base
open Ocannl
open Nn_blocks.DSL_modules

type config = {
  vocab_size : int;
  seq_len : int;
  d_model : int;
  num_heads : int;
  d_ff : int;
  num_layers : int;
  num_classes : int;
}

(** One post-norm transformer encoder block, like [Nn_blocks.transformer_encoder_block] but
    threading a key-padding [mask] into the attention.

    Inside [Nn_blocks.multi_head_attention] the position axis [s] of the input [x : b s | d] stays a
    batch axis on the query side, while the einsum ["... s | h d; ... t | h d => ... s | t -> h"]
    moves the key positions [t] into the input row: the raw scores are [b s | t -> h] (one
    [t]-vector per query position and head). [mask] must broadcast against that (rows align from the
    right, like NumPy): we pass [b 1 | t -> 1], so a masked key position is filled with [-inf] for
    every query and head before the softmax over [t]. The second einsum
    ["... s | t -> h; ... t | h e => ... s | h e"] contracts [t] against the values' position axis,
    giving [b s | h e], which [w_o] projects back to [d]. *)
let%op encoder_block ~label ~num_heads ~d_k ~d_v ~d_ff () =
  let mha = Nn_blocks.multi_head_attention ~label:("mha" :: label) ~num_heads ~d_k ~d_v () in
  let ffn = Nn_blocks.mlp ~label:("ffn" :: label) ~hid_dims:[ d_ff ] () in
  let ln1 = Nn_blocks.layer_norm ~label:("ln1" :: label) () in
  let ln2 = Nn_blocks.layer_norm ~label:("ln2" :: label) () in
  fun ~train_step ~mask x ->
    let x1 = ln1 (x + mha ~train_step ~mask x) in
    ln2 (x1 + ffn x1)

(** Builds the model. Applying the unit argument creates the parameters ([{ tok_embed }] etc. are
    inline [%op] parameter declarations lifted to the [()] point), and the returned closure can be
    applied to several different id tensors (training batch, holdout set, single headline) so that
    all forward graphs share the same parameters.

    Axis bookkeeping inside the closure:
    - [ids]: [b s |] (two batch axes: example, position; no output axes); PAD is id 0
    - [tok_one_hot]: [b s | v]; [tok_embed]: [| v -> d], so [tok_embed * tok_one_hot]: [b s | d]
    - [pos_one_hot]: [s | s'] (identity matrix); [pos_embed]: [| s' -> d], product: [s | d], which
      broadcasts over the leading example axis when added
    - [mask]: [b s | 1] with 1 at real characters (the size-1 output axis comes from the scalar
      literal and broadcasts); [attn_mask] moves the position axis into the input row and adds a
      size-1 query axis ([b 1 | t -> 1]) to match the attention scores; [lengths] sums the mask to
      [b | 1]
    - the pooling einsum ["... s | d => ... | d"] sums the masked states over positions. *)
let%op classifier ~label ~vocab_size ~seq_len ~d_model ~num_heads ~d_ff ~num_layers ~num_classes ()
    =
  let d_head = [%oc d_model / num_heads] in
  let layers =
    List.init num_layers ~f:(fun i ->
        encoder_block
          ~label:(("layer" ^ Int.to_string i) :: label)
          ~num_heads ~d_k:d_head ~d_v:d_head ~d_ff ())
  in
  (* Constant identity matrix over positions: [pos_one_hot[s, s'] = (s = s')]. Multiplying it by a
     learnable [| s' -> d] matrix yields the learned positional embedding table while keeping every
     parameter free of batch axes, as OCANNL requires. *)
  let pos_one_hot =
    Nn_blocks.one_hot_of_ids ~num_classes:seq_len (Nn_blocks.position_indices ~seq_len ())
  in
  fun ~train_step ids ->
    let mask = ids > 0. in
    let attn_mask = mask ++ "... t | ... => ... 0 | t -> 0" in
    let lengths = mask ++ "... s | ... => ... | 0" in
    let tok_one_hot = Nn_blocks.one_hot_of_ids ~num_classes:vocab_size ids in
    let embedded =
      ({ tok_embed; o = [ d_model ] } * tok_one_hot) + ({ pos_embed; o = [ d_model ] } * pos_one_hot)
    in
    let encoded =
      List.fold layers ~init:embedded ~f:(fun x layer -> layer ~train_step ~mask:attn_mask x)
    in
    (* Masked mean over the position axis: zero out PAD states, reduce [s] out of the batch row,
       divide by the number of real characters. *)
    let pooled = ((encoded *. mask) ++ "... s | d => ... | d") /. lengths in
    ({ w_cls } * pooled) + { b_cls = 0.; o = [ num_classes ] }

let build ~label (c : config) =
  classifier ~label ~vocab_size:c.vocab_size ~seq_len:c.seq_len ~d_model:c.d_model
    ~num_heads:c.num_heads ~d_ff:c.d_ff ~num_layers:c.num_layers ~num_classes:c.num_classes ()
