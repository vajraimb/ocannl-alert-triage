(lang dune 3.18)

(name alert_triage)

(generate_opam_files true)

(authors "OCANNL learning sandbox")

(maintainers "OCANNL learning sandbox")

(license MIT)

(package
 (name alert_triage)
 (synopsis
  "Tiny transformer text classifier for Google Alerts triage, built on OCANNL")
 (description
  "A learning sandbox: a character-level transformer encoder trained with OCANNL (neural_nets_lib) to sort Chinese/English Google Alert headlines into keep / spam / soft-mention / dupe.")
 (depends
  (ocaml
   (>= 5.3.0))
  (neural_nets_lib
   (= 1.0.1))
  (base
   (>= v0.17.0))
  stdio
  (yojson
   (>= 2.0.0))))
