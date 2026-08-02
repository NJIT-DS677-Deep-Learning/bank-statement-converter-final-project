# Bank Statement Converter Using Donut

This project fine-tunes the **Donut Document Understanding Transformer** to extract structured financial data from PDF bank statements without using a separate OCR system.

The model processes statement images and attempts to extract:

- Account holder
- Account number
- Bank name
- Statement period
- Opening and closing balances
- Transaction dates
- Descriptions
- Debit and credit amounts
- Running balances

## Model

The project uses:

```text
naver-clova-ix/donut-base
```

Donut combines a **Swin Transformer vision encoder** with a **BART-style text decoder**. Custom tokens are used to generate structured bank statement fields and transaction records.

## Dataset

Since few public labeled bank statement datasets are available, we generated 200 synthetic statements using Chase-style and Schwab-style layouts.

| Split | Statements |
|---|---:|
| Training | 140 |
| Validation | 30 |
| Test | 30 |

Each statement contains approximately 8–20 transactions.

## Training

| Parameter | Value |
|---|---|
| Resolution | 960 × 1280 |
| Max sequence length | 768 |
| Batch size | 1 |
| Optimizer | AdamW |
| Learning rate | 3e-5 |
| Epochs | 15 |
| Hardware | NVIDIA A100 10GB MIG |
| Training time | ~26 minutes |

The best validation loss was **0.2030**.

## Results

The model learned the general bank statement structure and usually generated the correct field tokens in the correct order. However, it did not reliably reproduce exact field values.

- Mean exact match: **0.00**
- Mean character similarity: **0.195**
- Ground-truth test transactions: **396**
- Parseable predicted transactions: **26**
- Transaction precision, recall, and F1: **0.00**

The main limitations were the small dataset, exposure bias during generation, and the strict accuracy requirements of financial data.

## Repository Structure

```text
bank-statement-converter-final-project/
├── src/
│   ├── schema.py
│   ├── generate_statement.py
│   ├── build_dataset.py
│   ├── finetune_donut.py
│   ├── demo.py
│   ├── eval.py
│   └── eval_on_testset.py
├── examples/
├── logs/
├── eval_results_test.json
├── finetune_job.slurm
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/NJIT-DS677-Deep-Learning/bank-statement-converter-final-project.git
cd bank-statement-converter-final-project

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Build the Dataset

```bash
python src/build_dataset.py --n 200 --out ./dataset
```

## Train the Model

```bash
sbatch finetune_job.slurm
```

## Run the Demo

```bash
python src/demo.py --checkpoint ./donut-bankstatement --pdf statement.pdf
```

## Evaluate the Model

```bash
python src/eval_on_testset.py --checkpoint ./donut-bankstatement --dataset ./dataset
```

## Future Improvements

- Increase the dataset to 10,000–100,000 statements
- Add more bank layouts and document styles
- Train first on shorter statements
- Use beam search or constrained decoding
- Validate date and currency formats
- Compare the model with Qwen2-VL or PaddleOCR-VL
- Evaluate on real anonymized statements

## Limitations

This project is a proof of concept and is not accurate enough for production financial reconciliation or automated accounting. All generated results should be manually verified.

## Authors

- Sivaragha Buddana
- Tara Walenczyk

## Reference

Kim et al. **OCR-Free Document Understanding Transformer.** ECCV 2022.
