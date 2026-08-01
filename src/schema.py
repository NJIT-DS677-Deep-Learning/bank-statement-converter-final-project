"""
Shared schema for the bank statement extraction

This defines:
1. The true label JSON structure we generate synthetic data in.
2. The Donut special tokens used to serialize that JSON into a target
   token sequence the decoder is trained to generate.

This file must be identical across the data generator, the training script,
and the inference/demo script - the special tokens must match everywhere.
"""

#Field-level special tokens (one open/close pair per field)
FIELD_TOKENS = [
    "s_account_holder", "s_account_number", "s_bank_name",
    "s_statement_period_start", "s_statement_period_end",
    "s_opening_balance", "s_closing_balance",
    "s_transactions", "s_transaction",
    "s_date", "s_description", "s_debit", "s_credit", "s_balance",
]

# Donut convention: <s_x> ... </s_x> for each field, plus a couple of
# structural tokens to mark the start/end of the whole document and the
# repeated transaction list.
def special_tokens():
    tokens = ["<s_donut-statement>", "</s_donut-statement>"]
    for name in FIELD_TOKENS:
        tokens.append(f"<{name}>")
        tokens.append(f"</{name}>")
    return tokens


def json_to_token_sequence(record: dict) -> str:
    """
    Convert a ground-truth statement dict into the flat token sequence
    Donut's decoder is trained to produce.

    Expected `record` shape:
    {
        "account_holder": str,
        "account_number": str,
        "bank_name": str,
        "statement_period_start": "YYYY-MM-DD",
        "statement_period_end": "YYYY-MM-DD",
        "opening_balance": "1234.56",
        "closing_balance": "2345.67",
        "transactions": [
            {"date": "YYYY-MM-DD", "description": str,
             "debit": "12.34" | "", "credit": "12.34" | "", "balance": "1234.56"},
            ...
        ]
    }
    """
    parts = ["<s_donut-statement>"]
    parts.append(f"<s_account_holder>{record['account_holder']}</s_account_holder>")
    parts.append(f"<s_account_number>{record['account_number']}</s_account_number>")
    parts.append(f"<s_bank_name>{record['bank_name']}</s_bank_name>")
    parts.append(f"<s_statement_period_start>{record['statement_period_start']}</s_statement_period_start>")
    parts.append(f"<s_statement_period_end>{record['statement_period_end']}</s_statement_period_end>")
    parts.append(f"<s_opening_balance>{record['opening_balance']}</s_opening_balance>")
    parts.append(f"<s_closing_balance>{record['closing_balance']}</s_closing_balance>")

    parts.append("<s_transactions>")
    for txn in record["transactions"]:
        parts.append("<s_transaction>")
        parts.append(f"<s_date>{txn['date']}</s_date>")
        parts.append(f"<s_description>{txn['description']}</s_description>")
        parts.append(f"<s_debit>{txn.get('debit', '')}</s_debit>")
        parts.append(f"<s_credit>{txn.get('credit', '')}</s_credit>")
        parts.append(f"<s_balance>{txn['balance']}</s_balance>")
        parts.append("</s_transaction>")
    parts.append("</s_transactions>")

    parts.append("</s_donut-statement>")
    return "".join(parts)


def token_sequence_to_json(seq: str) -> dict:
    """
    Parse a generated token sequence back into a plain dict.
    Tolerant of minor malformation (missing closing tags, etc.) since
    a fine-tuned model's raw output won't always be perfectly well-formed.
    """
    import re

    def extract(tag, text):
        m = re.search(f"<s_{tag}>(.*?)</s_{tag}>", text, re.DOTALL)
        return m.group(1).strip() if m else None

    result = {
        "account_holder": extract("account_holder", seq),
        "account_number": extract("account_number", seq),
        "bank_name": extract("bank_name", seq),
        "statement_period_start": extract("statement_period_start", seq),
        "statement_period_end": extract("statement_period_end", seq),
        "opening_balance": extract("opening_balance", seq),
        "closing_balance": extract("closing_balance", seq),
        "transactions": [],
    }

    txn_block_match = re.search(r"<s_transactions>(.*?)</s_transactions>", seq, re.DOTALL)
    if txn_block_match:
        txn_block = txn_block_match.group(1)
        for txn_match in re.finditer(r"<s_transaction>(.*?)</s_transaction>", txn_block, re.DOTALL):
            txn_text = txn_match.group(1)
            result["transactions"].append({
                "date": extract("date", txn_text),
                "description": extract("description", txn_text),
                "debit": extract("debit", txn_text),
                "credit": extract("credit", txn_text),
                "balance": extract("balance", txn_text),
            })

    return result
