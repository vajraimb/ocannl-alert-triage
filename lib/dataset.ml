(** Loading the JSONL alert dump, a character-level tokenizer, and the train/holdout split.

    Everything here is plain OCaml: OCANNL only sees the resulting integer arrays (see {!Model} and
    [bin/train.ml]). *)

open Base

type example = { text : string; label : Label.t; brand : string option }

let load_jsonl path : example list =
  Stdio.In_channel.read_lines path
  |> List.filter ~f:(fun line -> not (String.is_empty (String.strip line)))
  |> List.map ~f:(fun line ->
      let json = Yojson.Safe.from_string line in
      let open Yojson.Safe.Util in
      {
        text = json |> member "text" |> to_string;
        label = json |> member "label" |> to_string |> Label.of_string_exn;
        brand = json |> member "brand" |> to_string_option;
      })

(** {2 UTF-8 code points}

    Headlines mix CJK and ASCII, so the token unit is a Unicode code point, not a byte. Continuation
    bytes are folded into the leading byte; malformed input degrades to the raw byte value. *)
let codepoints (s : string) : int list =
  let n = String.length s in
  let byte i = Char.to_int s.[i] in
  let rec go i acc =
    if i >= n then List.rev acc
    else
      let b0 = byte i in
      let len, init =
        if b0 < 0x80 then (1, b0)
        else if b0 land 0xE0 = 0xC0 then (2, b0 land 0x1F)
        else if b0 land 0xF0 = 0xE0 then (3, b0 land 0x0F)
        else if b0 land 0xF8 = 0xF0 then (4, b0 land 0x07)
        else (1, b0)
      in
      if i + len > n then go (i + 1) (b0 :: acc)
      else
        let cp = ref init in
        for k = 1 to len - 1 do
          cp := (!cp lsl 6) lor (byte (i + k) land 0x3F)
        done;
        go (i + len) (!cp :: acc)
  in
  go 0 []

(** ASCII letters are case-folded; everything else (CJK, punctuation, digits) is kept as-is so that
    e.g. the fullwidth colon of a news headline stays distinguishable from a spammy [_] separator.
*)
let normalize_codepoint cp = if cp >= 0x41 && cp <= 0x5A then cp + 0x20 else cp

(** {2 Vocabulary} *)

module Vocab = struct
  type t = { to_id : (int, int) Hashtbl.t; of_id : int array }

  let pad_id = 0
  let unk_id = 1
  let size v = Array.length v.of_id

  (** Builds the vocabulary from the training texts only, so holdout accuracy reflects unseen
      characters mapping to [unk_id] as they would in production. *)
  let build (texts : string list) : t =
    let to_id = Hashtbl.create (module Int) in
    let of_id = ref [ unk_id; pad_id ] in
    (* Reserved ids first: 0 = PAD, 1 = UNK. *)
    Hashtbl.set to_id ~key:(-1) ~data:pad_id;
    Hashtbl.set to_id ~key:(-2) ~data:unk_id;
    List.iter texts ~f:(fun text ->
        List.iter (codepoints text) ~f:(fun cp ->
            let cp = normalize_codepoint cp in
            if not (Hashtbl.mem to_id cp) then (
              let id = List.length !of_id in
              Hashtbl.set to_id ~key:cp ~data:id;
              of_id := cp :: !of_id)));
    { to_id; of_id = Array.of_list (List.rev !of_id) }

  (** Fixed-length encoding: truncated or right-padded with [pad_id] to [max_len]. *)
  let encode v ~max_len (text : string) : int array =
    let ids =
      codepoints text
      |> List.map ~f:(fun cp ->
          Hashtbl.find (v.to_id : (int, int) Hashtbl.t) (normalize_codepoint cp)
          |> Option.value ~default:unk_id)
    in
    Array.init max_len ~f:(fun i -> match List.nth ids i with Some id -> id | None -> pad_id)

  (** Number of real (non-pad) positions after truncation. *)
  let length ~max_len text = Int.min max_len (List.length (codepoints text))
end

type split = { train : example array; holdout : example array }
(** {2 Split}

    Stratified by label with a seeded shuffle. The training set is truncated to a multiple of
    [batch_size] (the static batch index [@|] slices equal-sized batches); leftovers join the
    holdout so no example is discarded. *)

let stratified_split ~seed ~holdout_frac ~batch_size (examples : example list) : split =
  let rng = Random.State.make [| seed |] in
  let shuffle arr =
    let arr = Array.copy arr in
    for i = Array.length arr - 1 downto 1 do
      let j = Random.State.int rng (i + 1) in
      let tmp = arr.(i) in
      arr.(i) <- arr.(j);
      arr.(j) <- tmp
    done;
    arr
  in
  let train = ref [] and holdout = ref [] in
  Array.iter Label.all ~f:(fun label ->
      let group =
        List.filter examples ~f:(fun e -> Label.equal e.label label) |> Array.of_list |> shuffle
      in
      let n = Array.length group in
      let n_hold =
        Int.max 1 (Float.to_int (Float.round_nearest (holdout_frac *. Float.of_int n)))
      in
      Array.iteri group ~f:(fun i e ->
          if i < n_hold then holdout := e :: !holdout else train := e :: !train));
  let train = shuffle (Array.of_list !train) in
  let n_train = Array.length train - (Array.length train % batch_size) in
  let spill = Array.sub train ~pos:n_train ~len:(Array.length train - n_train) in
  {
    train = Array.sub train ~pos:0 ~len:n_train;
    holdout = shuffle (Array.append (Array.of_list !holdout) spill);
  }

let label_counts (examples : example array) =
  Array.map Label.all ~f:(fun label ->
      (label, Array.count examples ~f:(fun e -> Label.equal e.label label)))
