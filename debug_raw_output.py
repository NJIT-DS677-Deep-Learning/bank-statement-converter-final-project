"""Prints the model's raw generated token sequence before JSON parsing -- diagnostic only."""

import sys

import torch

from PIL import Image

from pdf2image import convert_from_path

from transformers import DonutProcessor, VisionEncoderDecoderModel



checkpoint = sys.argv[1] if len(sys.argv) > 1 else "./donut-bankstatement"

pdf_path = sys.argv[2] if len(sys.argv) > 2 else "test_statement.pdf"



device = "cuda" if torch.cuda.is_available() else "cpu"

processor = DonutProcessor.from_pretrained(checkpoint)

model = VisionEncoderDecoderModel.from_pretrained(checkpoint).to(device)

model.eval()



pages = convert_from_path(pdf_path, dpi=150)

image = pages[0]



pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

decoder_start_id = model.config.decoder_start_token_id

print(f"decoder_start_token_id: {decoder_start_id}")

print(f"decoded start token: {processor.tokenizer.decode([decoder_start_id])}")

print(f"eos_token_id: {processor.tokenizer.eos_token_id}, pad_token_id: {processor.tokenizer.pad_token_id}")



with torch.no_grad():

    generated_ids = model.generate(

        pixel_values,

        decoder_input_ids=torch.tensor([[decoder_start_id]], device=device),

        max_length=768,

        num_beams=1,

        eos_token_id=processor.tokenizer.eos_token_id,

        pad_token_id=processor.tokenizer.pad_token_id,

    )



print(f"\ngenerated_ids shape: {generated_ids.shape}")

print(f"raw token ids: {generated_ids[0].tolist()[:50]}...")



sequence = processor.tokenizer.batch_decode(generated_ids)[0]

print(f"\n--- RAW DECODED SEQUENCE ---\n{sequence}\n--- END ---")
