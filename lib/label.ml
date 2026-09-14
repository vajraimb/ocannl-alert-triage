(** The four triage classes. Their integer ids are the class-axis positions of the logits. *)

open Base

type t = Keep | Spam | Soft_mention | Dupe

let equal (a : t) (b : t) = Poly.equal a b
let all = [| Keep; Spam; Soft_mention; Dupe |]
let num_classes = Array.length all

let to_string = function
  | Keep -> "keep"
  | Spam -> "spam"
  | Soft_mention -> "soft-mention"
  | Dupe -> "dupe"

let of_string_exn = function
  | "keep" -> Keep
  | "spam" -> Spam
  | "soft-mention" | "soft_mention" -> Soft_mention
  | "dupe" -> Dupe
  | other -> invalid_arg ("Label.of_string_exn: unknown label " ^ other)

let to_int = function Keep -> 0 | Spam -> 1 | Soft_mention -> 2 | Dupe -> 3
let of_int_exn i = all.(i)
