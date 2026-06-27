#!/usr/bin/env python
"""riscv_questions.py — build a distillation prompt set for the RISC-V expert.

The model-free RISC-V spoke only retrieves raw normative rules. To give it a *model-reasoned* (distilled) tier like
the logic spoke, fieldrun needs a set of PROMPTS to distill. This emits that set from the RISC-V spec data:

  - one "What does the <X> instruction do?" per distinct instruction referenced in the rules (insn:X[] markup), and
  - a curated set of concept questions (memory model, harts, extensions, …).

Then, on a CPU box:
  fieldrun --bundle <model> --export-logic-corpus riscv_questions.txt --steps 128 --out export_riscv/
  python tools/dl2package.py --corpus riscv_questions.txt --dl export_riscv/ --out package_riscv_distilled
  python tools/build_grounding.py --corpus package_riscv/rules.txt --out package_riscv_distilled --no-split  # GloVe
  python tools/build_gram.py --corpus package_riscv/rules_plain.txt --out package_riscv_distilled/gram
  ./build/sgiandubh package_riscv_distilled 8081 --answer-from-corpus
The distilled answer is the model's prose; grounding attaches the matching norm-rule as the citation — flexible AND
grounded. (The certificate certifies the model's decision; grounding ties the answer to the spec.)
"""
import argparse, re

# Curated concept questions — the "why/how" the per-instruction questions don't cover.
CONCEPTS = [
    "What is a hart in RISC-V?",
    "What is the RISC-V Weak Memory Ordering (RVWMO) model?",
    "What do the aq and rl bits do in atomic memory instructions?",
    "What is the difference between the base integer ISA and an extension?",
    "What is the difference between RV32I and RV64I?",
    "What does the FENCE instruction do and when is it needed?",
    "What is the difference between FENCE and FENCE.I?",
    "What are the RISC-V integer computational instructions?",
    "What are the RISC-V load and store instructions?",
    "What are the RISC-V control transfer (branch and jump) instructions?",
    "How does RISC-V encode immediates in I-type instructions?",
    "What is the role of register x0 in RISC-V?",
    "What is a CSR and how are CSRs accessed?",
    "What does the ECALL instruction do?",
    "What does the EBREAK instruction do?",
    "What is the A extension and what does it provide?",
    "What is the M extension and what does it provide?",
    "What is the C (compressed) extension?",
    "What is the difference between an acquire and a release in atomics?",
    "How does RISC-V handle misaligned memory accesses?",
    "What is endianness in RISC-V and is it byte-address invariant?",
    "What are the RISC-V multiply and divide instructions?",
    "What is the purpose of the AUIPC instruction?",
    "What is the LR/SC (load-reserved / store-conditional) pair used for?",
    "What does the SFENCE.VMA instruction do?",
]


def instructions(rules_path):
    seen = set()
    for ln in open(rules_path, encoding="utf-8"):
        for m in re.findall(r"insn:([a-z0-9][a-z0-9._]*)", ln.lower()):
            m = m.rstrip(".")
            if len(m) >= 2 and not m.startswith("."):
                seen.add(m)
    return sorted(seen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", default="package_riscv/rules.txt")
    ap.add_argument("--out", default="package_riscv/questions.txt")
    ap.add_argument("--no-concepts", action="store_true")
    a = ap.parse_args()
    insns = instructions(a.rules)
    with open(a.out, "w", encoding="utf-8") as f:
        if not a.no_concepts:
            for q in CONCEPTS:
                f.write(q + "\n")
        for x in insns:
            f.write(f"What does the {x.upper()} instruction do in RISC-V?\n")
    print(f"{a.out}: {0 if a.no_concepts else len(CONCEPTS)} concept + {len(insns)} instruction questions "
          f"= {(0 if a.no_concepts else len(CONCEPTS)) + len(insns)} prompts")


if __name__ == "__main__":
    main()
