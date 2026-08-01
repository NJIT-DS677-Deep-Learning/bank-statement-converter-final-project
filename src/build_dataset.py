"""
Builds the full synthetic bank statement dataset for Donut fine-tuning which is png format not pdf.
Also breaks up all statements into our training and test sets.

Output layout (HuggingFace `imagefolder`-compatible):

  dataset/
    train/
      metadata.jsonl -> one line per image: this holds the information of the actual information in each statement.
      0000_chase_style.png
      0001_schwab_style.png
      ...
    validation/
      metadata.jsonl
      ...
    test/
      metadata.jsonl
      ...

`true_label` in metadata.jsonl follows the format the HF Donut fine-tuning
examples expect: {"gt_parse": <the parsed dict>} both the raw
dict (for eval) and precompute the token sequence are stored separately in
`token_sequence.txt` per split

Usage:
    python build_dataset.py --n 200 --out ./dataset
"""
import argparse
import json
import os
import random
from pdf2image import convert_from_path

from generate_statement import generate_true_label, TEMPLATES, BANK_NAMES
from schema import json_to_token_sequence


def build_split(split_name, n, out_dir, dpi=150, start_idx=0):
    split_dir = os.path.join(out_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    template_names = list(TEMPLATES.keys())
    metadata_lines = []

    for i in range(n):
        idx = start_idx + i
        template_name = template_names[i % len(template_names)]
        render_fn = TEMPLATES[template_name]
        record = generate_true_label(bank_name=BANK_NAMES[template_name])

        pdf_path = os.path.join(split_dir, f"_tmp_{idx}.pdf")
        render_fn(record, pdf_path)

        images = convert_from_path(pdf_path, dpi=dpi)
        file_name = f"{idx:04d}_{template_name}.png"
        images[0].save(os.path.join(split_dir, file_name))
        os.remove(pdf_path)

        target_sequence = json_to_token_sequence(record)

        metadata_lines.append(json.dumps({
            "file_name": file_name,
            "true_label": json.dumps({"gt_parse": record}),
            "target_sequence": target_sequence,
        }))

    with open(os.path.join(split_dir, "metadata.jsonl"), "w") as f:
        f.write("\n".join(metadata_lines) + "\n")

    print(f"[{split_name}] wrote {n} statements to {split_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200, help="total number of statements")
    parser.add_argument("--out", type=str, default="./dataset")
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    random.seed(args.seed)

    n_train = int(args.n * args.train_frac)
    n_val = int(args.n * args.val_frac)
    n_test = args.n - n_train - n_val

    os.makedirs(args.out, exist_ok=True)
    build_split("train", n_train, args.out, dpi=args.dpi, start_idx=0)
    build_split("validation", n_val, args.out, dpi=args.dpi, start_idx=n_train)
    build_split("test", n_test, args.out, dpi=args.dpi, start_idx=n_train + n_val)

    print(f"\nDone. {n_train} train / {n_val} val / {n_test} test statements in {args.out}")


if __name__ == "__main__":
    main()
