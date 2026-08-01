"""

Runs the fine-tuned checkpoint against every example in dataset/test/,

compares to ground truth, and prints aggregate precision/recall/F1.



This is the real signal to trust over single-example demo runs --

greedy generation on one statement is noisy; averaging over the whole

test set tells you what the model is actually doing.



Usage:

    python eval_on_testset.py --checkpoint ./donut-bankstatement --dataset ./dataset

"""

import argparse

import json

import os



import torch

from PIL import Image

from transformers import DonutProcessor, VisionEncoderDecoderModel



from schema import token_sequence_to_json

from eval import evaluate_statement, aggregate_results





def load_model(checkpoint_dir, device):

    processor = DonutProcessor.from_pretrained(checkpoint_dir)

    model = VisionEncoderDecoderModel.from_pretrained(checkpoint_dir).to(device)

    model.eval()

    return model, processor





def extract_image(image, model, processor, device, max_length=768):

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





def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, default="./donut-bankstatement")

    parser.add_argument("--dataset", type=str, default="./dataset")

    parser.add_argument("--split", type=str, default="test")

    args = parser.parse_args()



    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model on {device}...")

    model, processor = load_model(args.checkpoint, device)



    split_dir = os.path.join(args.dataset, args.split)

    with open(os.path.join(split_dir, "metadata.jsonl")) as f:

        records = [json.loads(line) for line in f if line.strip()]



    print(f"Evaluating on {len(records)} {args.split} examples...\n")



    results = []

    for i, rec in enumerate(records):

        image = Image.open(os.path.join(split_dir, rec["file_name"])).convert("RGB")

        true_label = json.loads(rec["true_label"])["gt_parse"]



        sequence = extract_image(image, model, processor, device)

        predicted = token_sequence_to_json(sequence)



        r = evaluate_statement(true_label, predicted)

        results.append(r)

        print(f"[{i+1}/{len(records)}] {rec['file_name']}: "

              f"txn_f1={r.txn_f1:.2f} gt_txns={r.num_true_txns} pred_txns={r.num_pred_txns} matched={r.num_matched_txns}")



    print("\n=== AGGREGATE RESULTS ===")

    agg = aggregate_results(results)

    print(json.dumps(agg, indent=2))



    with open(f"eval_results_{args.split}.json", "w") as f:

        json.dump(agg, f, indent=2)

    print(f"\nSaved eval_results_{args.split}.json")





if __name__ == "__main__":

    main()
