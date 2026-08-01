"""
Fine-tunes donut-base on the synthetic bank statement dataset.

Need to run this on a GPU (we used wulver/Colab)  it needs to download the
donut-base checkpoint from Hugging Face,
Steps to use in Colab:

    !pip install transformers datasets pillow -q
    # upload schema.py, dataset/ (from build_dataset.py) alongside this file
    !python finetune_donut.py --dataset ./dataset --epochs 4 --out ./donut-bankstatement

Uses a plain PyTorch training loop (not Seq2SeqTrainer) so it's easy to
see and modify exactly what's happening.
"""
import argparse
import json
import os

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel, VisionEncoderDecoderConfig

from schema import special_tokens

MODEL_ID = "naver-clova-ix/donut-base"


class StatementDataset(Dataset):
    """Reads one split directory (train/validation/test) written by build_dataset.py."""

    def __init__(self, split_dir, processor, max_length=768):
        self.split_dir = split_dir
        self.processor = processor
        self.max_length = max_length
        self.records = []
        with open(os.path.join(split_dir, "metadata.jsonl")) as f:
            for line in f:
                if line.strip():
                    self.records.append(json.loads(line))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        image = Image.open(os.path.join(self.split_dir, rec["file_name"])).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)

        target = rec["target_sequence"] + self.processor.tokenizer.eos_token
        labels = self.processor.tokenizer(
            target,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)
        # Ignore padding in the loss.
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}


def build_model_and_processor(image_size=(960, 1280), max_length=768):
    """
    Loads donut-base and adds our schema's special tokens to both the
    tokenizer and the model's decoder embedding matrix (which must be
    resized to match, or the new token ids are out of range).
    """
    config = VisionEncoderDecoderConfig.from_pretrained(MODEL_ID)
    config.encoder.image_size = image_size
    config.decoder.max_length = max_length

    processor = DonutProcessor.from_pretrained(MODEL_ID)
    processor.image_processor.size = {"height": image_size[0], "width": image_size[1]}
    processor.image_processor.do_align_long_axis = False

    model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID, config=config)

    added = processor.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens()})
    model.decoder.resize_token_embeddings(len(processor.tokenizer))
    print(f"Added {added} special tokens; vocab size now {len(processor.tokenizer)}")

    # Task start/end tokens for generation.
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids("<s_donut-statement>")

    return model, processor


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cpu":
        print("WARNING: no GPU detected")

    model, processor = build_model_and_processor(max_length=args.max_length)
    model.to(device)

    train_ds = StatementDataset(os.path.join(args.dataset, "train"), processor, args.max_length)
    val_ds = StatementDataset(os.path.join(args.dataset, "validation"), processor, args.max_length)
    print(f"train examples: {len(train_ds)}, val examples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    os.makedirs(args.out, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if step % 10 == 0:
                print(f"epoch {epoch} step {step}/{len(train_loader)} loss {loss.item():.4f}")

        avg_train_loss = total_loss / max(len(train_loader), 1)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(pixel_values=pixel_values, labels=labels)
                val_loss += outputs.loss.item()
        avg_val_loss = val_loss / max(len(val_loader), 1)

        print(f"=== epoch {epoch} done: train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} ===")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(args.out)
            processor.save_pretrained(args.out)
            print(f"Saved new best checkpoint to {args.out} (val_loss={avg_val_loss:.4f})")

    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="./dataset")
    parser.add_argument("--out", type=str, default="./donut-bankstatement")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=768)
    args = parser.parse_args()
    train(args)
