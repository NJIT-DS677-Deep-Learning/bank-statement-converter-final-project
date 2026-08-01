"""
Self-contained demo: bank statement PDF -> structured transaction data

This is the artifact for demo report section 3
Must be run it after fine-tuning (finetune_donut.py) has
produced a checkpoint but will essentially show the input of a pdf to output of csv.

Usage:
    python demo.py --checkpoint ./donut-bankstatement --pdf statement.pdf


"""
import argparse

import torch
from PIL import Image
from pdf2image import convert_from_path
from transformers import DonutProcessor, VisionEncoderDecoderModel

from schema import token_sequence_to_json


def load_model(checkpoint_dir, device):
    processor = DonutProcessor.from_pretrained(checkpoint_dir)
    model = VisionEncoderDecoderModel.from_pretrained(checkpoint_dir)
    model.to(device)
    model.eval()
    return model, processor


def extract_page(image: Image.Image, model, processor, device, max_length=768):
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    decoder_start_id = model.config.decoder_start_token_id

    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values,
            decoder_input_ids=torch.tensor([[decoder_start_id]], device=device),
            max_length=max_length,
            num_beams=1,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    sequence = processor.tokenizer.batch_decode(generated_ids)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    return sequence


def extract_pdf(pdf_path, model, processor, device, dpi=150):
    """Runs every page of the PDF through the model and merges the transactions.
    Assumes account-level fields (holder, account number, etc.) come from page 1."""
    pages = convert_from_path(pdf_path, dpi=dpi)

    merged = None
    for i, page_image in enumerate(pages):
        seq = extract_page(page_image, model, processor, device)
        parsed = token_sequence_to_json(seq)
        if merged is None:
            merged = parsed
        else:
            merged["transactions"].extend(parsed.get("transactions", []))
        print(f"  page {i + 1}/{len(pages)}: found {len(parsed.get('transactions', []))} transactions")

    return merged


def print_and_save(result, csv_path="extracted_transactions.csv"):
    print("\n--- Statement summary ---")
    for key in ["account_holder", "account_number", "bank_name",
                "statement_period_start", "statement_period_end",
                "opening_balance", "closing_balance"]:
        print(f"  {key}: {result.get(key)}")

    print(f"\n--- Transactions ({len(result['transactions'])}) ---")
    print(f"{'Date':<12}{'Description':<32}{'Debit':>10}{'Credit':>10}{'Balance':>12}")
    for t in result["transactions"]:
        print(f"{t['date'] or '':<12}{(t['description'] or '')[:30]:<32}"
              f"{t['debit'] or '':>10}{t['credit'] or '':>10}{t['balance'] or '':>12}")

    import csv
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "description", "debit", "credit", "balance"])
        writer.writeheader()
        writer.writerows(result["transactions"])
    print(f"\nSaved {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="./donut-bankstatement")
    parser.add_argument("--pdf", type=str, required=True)
    parser.add_argument("--csv-out", type=str, default="extracted_transactions.csv")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {args.checkpoint} on {device}...")
    model, processor = load_model(args.checkpoint, device)

    print(f"Extracting {args.pdf}...")
    result = extract_pdf(args.pdf, model, processor, device)
    print_and_save(result, args.csv_out)
