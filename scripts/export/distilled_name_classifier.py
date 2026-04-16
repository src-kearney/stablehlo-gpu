"""
Export a NER model for PII name detection to StableHLO MLIR.

Model: dslim/bert-base-NER — BERT fine-tuned on CoNLL-2003 for token
classification. Detects PER (person names), ORG, LOC, MISC. Use the
PER label output to identify names for PII redaction.

Requires transformers<5.0.0 (Flax support was dropped in v5).

Usage:
    python scripts/export/distilled_name_classifier.py [--output mlir/stablehlo/distilled_name_classifier.mlir]
"""

import argparse
import os
import jax
import jax.numpy as jnp
from transformers import FlaxAutoModelForTokenClassification


MODEL = "dslim/bert-base-NER"


def main(output_path: str) -> None:
    print(f"Loading {MODEL} ...")
    model = FlaxAutoModelForTokenClassification.from_pretrained(MODEL)

    # deterministic=True disables dropout for a pure inference export.
    # Output shape: [batch, seq_len, num_labels] — logits per token per class.
    # Label 0 = O (not an entity), PER labels identify person names.
    def forward(input_ids, attention_mask):
        return model(
            input_ids,
            attention_mask=attention_mask,
        ).logits

    batch, seq = 1, 128
    abstract_ids  = jax.ShapeDtypeStruct((batch, seq), jnp.int32)
    abstract_mask = jax.ShapeDtypeStruct((batch, seq), jnp.int32)

    print(f"Exporting forward pass (batch={batch}, seq_len={seq}) to StableHLO ...")
    exported = jax.export.export(jax.jit(forward))(abstract_ids, abstract_mask)

    with open(output_path, "w") as f:
        f.write(exported.mlir_module())

    size_kb = os.path.getsize(output_path) / 1024
    print(f"Saved StableHLO to {output_path} ({size_kb:.1f} KB)")
    print(f"Labels: {model.config.id2label}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="mlir/stablehlo/distilled_name_classifier.mlir",
                        help="Path to write the StableHLO MLIR file")
    args = parser.parse_args()
    main(args.output)
