from src.generate_statement import generate_true_label, TEMPLATES, BANK_NAMES
rec = generate_true_label(bank_name=BANK_NAMES['chase_style'])
TEMPLATES['chase_style'](rec, 'test_statement.pdf')
print("Wrote test_statement.pdf")
