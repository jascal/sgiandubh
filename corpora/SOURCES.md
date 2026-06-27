# Corpus sources & attribution

These corpora are third-party content, redistributed here under their own licenses (the rest of this repo is
Apache-2.0). They feed the expert build (`tools/build_experts.sh`): grounding passages + distillation prompts.

## `logic_kb.txt` — Open Logic Project
- **Source:** the [Open Logic Project](https://openlogicproject.org/) (the *Open Logic Text*).
- **License:** Creative Commons Attribution 4.0 International (**CC BY 4.0**).
- **Modifications:** extracted to one declarative sentence per line for use as grounding passages; no claims of
  endorsement by the original authors.
- `logic_questions.txt` is an original set of questions written for this repo (Apache-2.0), used as distillation
  prompts answered against the Open Logic material.

## RISC-V (`norm-rules.json`, not shipped here)
- **Source:** the normative rules export from a [riscv-isa-manual](https://github.com/riscv/riscv-isa-manual/releases)
  release. Download it per release and pass it to the build (`NORM_RULES=…`); it is not committed (it's a versioned
  release artifact).
- **License:** Creative Commons Attribution 4.0 International (**CC BY 4.0**).
- `tools/riscv_questions.py` derives distillation prompts from it; `tools/normrules2package.py` turns it into a package.

> Per CC BY 4.0, retain this attribution if you redistribute the derived corpora or the built expert packages.
