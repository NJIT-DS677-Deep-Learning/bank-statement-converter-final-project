"""
Evaluations on how the model is performing.

Compares a predicted statement dict against ground truth and computes:
  - per-field accuracy (account_holder, account_number, balances, etc.)
  - transaction-level precision/recall/F1 (a predicted transaction counts
    as a match if date, debit, credit, and balance all match exactly --
    description uses a looser containment check since VLMs paraphrase)

Use this identically for the fine-tuned Donut model and the zero-shot

Usage:
    from eval import evaluate_statement, aggregate_results
    results = [evaluate_statement(gt, pred) for gt, pred in zip(true_labels, predictions)]
    print(aggregate_results(results))
"""
from dataclasses import dataclass, field


HEADER_FIELDS = [
    "account_holder", "account_number", "bank_name",
    "statement_period_start", "statement_period_end",
    "opening_balance", "closing_balance",
]


@dataclass
class StatementEvalResult:
    header_correct: dict = field(default_factory=dict)
    txn_precision: float = 0.0
    txn_recall: float = 0.0
    txn_f1: float = 0.0
    num_gt_txns: int = 0
    num_pred_txns: int = 0
    num_matched_txns: int = 0


def _txn_matches(a, b, desc_loose=True):
    if a.get("date") != b.get("date"):
        return False
    if a.get("debit") != b.get("debit"):
        return False
    if a.get("credit") != b.get("credit"):
        return False
    if a.get("balance") != b.get("balance"):
        return False
    if desc_loose:
        ad = (a.get("description") or "").strip().lower()
        bd = (b.get("description") or "").strip().lower()
        if ad and bd and ad not in bd and bd not in ad:
            return False
    return True


def evaluate_statement(true_label: dict, predicted: dict) -> StatementEvalResult:
    result = StatementEvalResult()

    for f in HEADER_FIELDS:
        result.header_correct[f] = (
            str(true_label.get(f, "")).strip() == str(predicted.get(f, "")).strip()
        )

    gt_txns = true_label.get("transactions", []) or []
    pred_txns = list(predicted.get("transactions", []) or [])
    result.num_gt_txns = len(gt_txns)
    result.num_pred_txns = len(pred_txns)

    matched = 0
    used = [False] * len(pred_txns)
    for gt_t in gt_txns:
        for i, pred_t in enumerate(pred_txns):
            if used[i]:
                continue
            if _txn_matches(gt_t, pred_t):
                used[i] = True
                matched += 1
                break
    result.num_matched_txns = matched

    result.txn_precision = matched / result.num_pred_txns if result.num_pred_txns else 0.0
    result.txn_recall = matched / result.num_gt_txns if result.num_gt_txns else 0.0
    if result.txn_precision + result.txn_recall > 0:
        result.txn_f1 = (2 * result.txn_precision * result.txn_recall /
                          (result.txn_precision + result.txn_recall))
    else:
        result.txn_f1 = 0.0

    return result


def aggregate_results(results: list) -> dict:
    n = len(results)
    if n == 0:
        return {}

    header_acc = {}
    for f in HEADER_FIELDS:
        header_acc[f] = sum(r.header_correct.get(f, False) for r in results) / n

    avg_precision = sum(r.txn_precision for r in results) / n
    avg_recall = sum(r.txn_recall for r in results) / n
    avg_f1 = sum(r.txn_f1 for r in results) / n

    return {
        "n_statements": n,
        "header_field_accuracy": header_acc,
        "mean_header_accuracy": sum(header_acc.values()) / len(header_acc),
        "transaction_precision": avg_precision,
        "transaction_recall": avg_recall,
        "transaction_f1": avg_f1,
    }


if __name__ == "__main__":
    # Smoke test with a hand-built example.
    gt = {
        "account_holder": "Jane Doe", "account_number": "...1234", "bank_name": "Chase Bank",
        "statement_period_start": "2026-01-01", "statement_period_end": "2026-01-31",
        "opening_balance": "1000.00", "closing_balance": "950.00",
        "transactions": [
            {"date": "2026-01-05", "description": "STARBUCKS #123", "debit": "5.50", "credit": "", "balance": "994.50"},
            {"date": "2026-01-10", "description": "PAYROLL DEPOSIT", "debit": "", "credit": "50.00", "balance": "1044.50"},
        ],
    }
    pred = {
        "account_holder": "Jane Doe", "account_number": "...1234", "bank_name": "Chase",
        "statement_period_start": "2026-01-01", "statement_period_end": "2026-01-31",
        "opening_balance": "1000.00", "closing_balance": "950.00",
        "transactions": [
            {"date": "2026-01-05", "description": "STARBUCKS", "debit": "5.50", "credit": "", "balance": "994.50"},
        ],
    }
    result = evaluate_statement(gt, pred)
    print(result)
    print(aggregate_results([result]))
