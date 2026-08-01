"""

Moel Evaluation Methods.



Compares a predicted statement dict against true_label and computes:

  - per-field exact-match accuracy (strict- a single wrong character fails)

  - per-field character-level similarity (fuzzy - partial credit via

    difflib's ratio(), 0.0-1.0, useful for showing "how close" a small

    or undertrained model got even when it rarely gets exact matches)

  - transaction-level precision/recall/F1 (exact match on date/debit/credit/balance)



Report BOTH the strict and fuzzy numbers - exact match alone can look

like near-total failure for a resource-constrained fine-tune even when

the model is clearly learning the right structure and getting close on

content, which the fuzzy metric makes visible it may just not be a perfect match.



Use this identically for the fine-tuned Donut model and the zero-shot

general VLM baseline so the comparison



Usage:

    from eval import evaluate_statement, aggregate_results

    results = [evaluate_statement(t, pred) for t, pred in zip(true_labels, predictions)]

    print(aggregate_results(results))

"""

import difflib

from dataclasses import dataclass, field





HEADER_FIELDS = [

    "account_holder", "account_number", "bank_name",

    "statement_period_start", "statement_period_end",

    "opening_balance", "closing_balance",

]





def char_similarity(a, b):

    """0.0-1.0 fuzzy similarity between two strings, robust to None."""

    a = (a or "").strip()

    b = (b or "").strip()

    if not a and not b:

        return 1.0

    return difflib.SequenceMatcher(None, a, b).ratio()





@dataclass

class StatementEvalResult:

    header_correct: dict = field(default_factory=dict)

    header_similarity: dict = field(default_factory=dict)

    txn_precision: float = 0.0

    txn_recall: float = 0.0

    txn_f1: float = 0.0

    txn_fuzzy_precision: float = 0.0

    txn_fuzzy_recall: float = 0.0

    txn_fuzzy_f1: float = 0.0

    num_true_txns: int = 0

    num_pred_txns: int = 0

    num_matched_txns: int = 0

    num_fuzzy_matched_txns: int = 0





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





def _txn_fuzzy_match(a, b, threshold=0.5):

    """Looser transaction match: date and amount both just need to be

    reasonably similar (not exact) - shows 'found roughly the right

    transaction' even when several characters are garbled, which is

    useful signal for an early-stage or resource-constrained model."""

    date_sim = char_similarity(a.get("date"), b.get("date"))

    amt_a = a.get("debit") or a.get("credit") or ""

    amt_b = b.get("debit") or b.get("credit") or ""

    amt_sim = char_similarity(amt_a, amt_b)

    return date_sim >= threshold and amt_sim >= threshold





def evaluate_statement(true_label: dict, predicted: dict) -> StatementEvalResult:

    result = StatementEvalResult()



    for f in HEADER_FIELDS:

        true_val = str(true_label.get(f, "")).strip()

        pred_val = str(predicted.get(f, "")).strip()

        result.header_correct[f] = (true_val == pred_val)

        result.header_similarity[f] = char_similarity(true_val, pred_val)



    true_txns = true_label.get("transactions", []) or []

    pred_txns = list(predicted.get("transactions", []) or [])

    result.num_true_txns = len(true_txns)

    result.num_pred_txns = len(pred_txns)



    matched = 0

    used = [False] * len(pred_txns)

    for true_t in true_txns:

        for i, pred_t in enumerate(pred_txns):

            if used[i]:

                continue

            if _txn_matches(true_t, pred_t):

                used[i] = True

                matched += 1

                break

    result.num_matched_txns = matched



    result.txn_precision = matched / result.num_pred_txns if result.num_pred_txns else 0.0

    result.txn_recall = matched / result.num_true_txns if result.num_true_txns else 0.0

    if result.txn_precision + result.txn_recall > 0:

        result.txn_f1 = (2 * result.txn_precision * result.txn_recall /

                          (result.txn_precision + result.txn_recall))

    else:

        result.txn_f1 = 0.0



    fuzzy_matched = 0

    used_fuzzy = [False] * len(pred_txns)

    for true_t in true_txns:

        for i, pred_t in enumerate(pred_txns):

            if used_fuzzy[i]:

                continue

            if _txn_fuzzy_match(true_t, pred_t):

                used_fuzzy[i] = True

                fuzzy_matched += 1

                break

    result.num_fuzzy_matched_txns = fuzzy_matched



    result.txn_fuzzy_precision = fuzzy_matched / result.num_pred_txns if result.num_pred_txns else 0.0

    result.txn_fuzzy_recall = fuzzy_matched / result.num_true_txns if result.num_true_txns else 0.0

    if result.txn_fuzzy_precision + result.txn_fuzzy_recall > 0:

        result.txn_fuzzy_f1 = (2 * result.txn_fuzzy_precision * result.txn_fuzzy_recall /

                                (result.txn_fuzzy_precision + result.txn_fuzzy_recall))

    else:

        result.txn_fuzzy_f1 = 0.0



    return result





def aggregate_results(results: list) -> dict:

    n = len(results)

    if n == 0:

        return {}



    header_acc = {}

    header_sim = {}

    for f in HEADER_FIELDS:

        header_acc[f] = sum(r.header_correct.get(f, False) for r in results) / n

        header_sim[f] = sum(r.header_similarity.get(f, 0.0) for r in results) / n



    avg_precision = sum(r.txn_precision for r in results) / n

    avg_recall = sum(r.txn_recall for r in results) / n

    avg_f1 = sum(r.txn_f1 for r in results) / n

    avg_fuzzy_precision = sum(r.txn_fuzzy_precision for r in results) / n

    avg_fuzzy_recall = sum(r.txn_fuzzy_recall for r in results) / n

    avg_fuzzy_f1 = sum(r.txn_fuzzy_f1 for r in results) / n



    return {

        "n_statements": n,

        "header_field_exact_accuracy": header_acc,

        "mean_header_exact_accuracy": sum(header_acc.values()) / len(header_acc),

        "header_field_char_similarity": header_sim,

        "mean_header_char_similarity": sum(header_sim.values()) / len(header_sim),

        "transaction_precision_exact": avg_precision,

        "transaction_recall_exact": avg_recall,

        "transaction_f1_exact": avg_f1,

        "transaction_precision_fuzzy": avg_fuzzy_precision,

        "transaction_recall_fuzzy": avg_fuzzy_recall,

        "transaction_f1_fuzzy": avg_fuzzy_f1,

    }
