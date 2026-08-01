"""
This file is the synthetic bank statement generator. This is necessary because actual bank statements are
difficult to source given they hold personal information. 

Produces:
  - a rendered PDF that looks like an actual bank statement
  - a matching dict (see schema.py for shape) which tells what is actually in the statements

Two template structures are included (a Chase-style and a Schwab-style
layout) so the model sees more than one format and we can see how it acts on multiple. We can add more
in the future if we want to by
copying one of the render_* functions and registering it in TEMPLATES.
"""
import random
from datetime import date, timedelta
from faker import Faker
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

fake = Faker()

DESCRIPTIONS = [
    "AMAZON MKTPLACE PMTS", "STARBUCKS #{n}", "WHOLE FOODS MKT #{n}",
    "DIRECT DEP PAYROLL", "ACH TRANSFER TO SAVINGS", "SHELL OIL #{n}",
    "NETFLIX.COM", "UBER TRIP {n}", "CHECK #{n}", "ATM WITHDRAWAL #{n}",
    "COMCAST CABLE", "MORTGAGE PMT", "VENMO PAYMENT", "TARGET STORE #{n}",
    "INTEREST PAYMENT", "MONTHLY MAINTENANCE FEE", "WIRE TRANSFER IN",
    "TRADER JOES #{n}", "CVS PHARMACY #{n}", "SPOTIFY PREMIUM",
]


def random_description():
    template = random.choice(DESCRIPTIONS)
    return template.format(n=random.randint(100, 9999))


def generate_true_label(bank_name="Chase", num_transactions=None):
    """Build a random but internally-consistent statement record."""
    if num_transactions is None:
        num_transactions = random.randint(8, 20)

    period_end = date.today() - timedelta(days=random.randint(0, 300))
    period_start = period_end - timedelta(days=30)

    balance = round(random.uniform(500, 15000), 2)
    opening_balance = balance

    account_holder = fake.name()
    account_number = f"...{random.randint(1000, 9999)}"

    txn_dates = sorted(
        period_start + timedelta(days=random.randint(0, 30))
        for _ in range(num_transactions)
    )

    transactions = []
    for d in txn_dates:
        is_credit = random.random() < 0.25
        amount = round(random.uniform(5, 800), 2)
        if is_credit:
            balance += amount
            debit, credit = "", f"{amount:.2f}"
        else:
            balance -= amount
            debit, credit = f"{amount:.2f}", ""
        transactions.append({
            "date": d.isoformat(),
            "description": random_description(),
            "debit": debit,
            "credit": credit,
            "balance": f"{balance:.2f}",
        })

    return {
        "account_holder": account_holder,
        "account_number": account_number,
        "bank_name": bank_name,
        "statement_period_start": period_start.isoformat(),
        "statement_period_end": period_end.isoformat(),
        "opening_balance": f"{opening_balance:.2f}",
        "closing_balance": f"{balance:.2f}",
        "transactions": transactions,
    }


def _base_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BankHeader", fontSize=18, fontName="Helvetica-Bold",
                               textColor=colors.HexColor("#1a1a2e"), spaceAfter=2))
    styles.add(ParagraphStyle(name="SmallGray", fontSize=8, fontName="Helvetica",
                               textColor=colors.HexColor("#555555")))
    styles.add(ParagraphStyle(name="SectionLabel", fontSize=10, fontName="Helvetica-Bold",
                               textColor=colors.HexColor("#1a1a2e"), spaceBefore=10, spaceAfter=4))
    return styles


def render_style_a(record: dict, out_path: str, accent_hex="#005eb8"):
    """Chase-ish layout: header bar, summary block, single transaction table."""
    doc = SimpleDocTemplate(out_path, pagesize=letter,
                             leftMargin=0.6 * inch, rightMargin=0.6 * inch,
                             topMargin=0.5 * inch, bottomMargin=0.5 * inch)
    styles = _base_styles()
    story = []

    story.append(Paragraph(record["bank_name"], styles["BankHeader"]))
    story.append(Paragraph("Personal Checking Account Statement", styles["SmallGray"]))
    story.append(Spacer(1, 10))

    info_data = [
        ["Account Holder:", record["account_holder"], "Statement Period:",
         f"{record['statement_period_start']} to {record['statement_period_end']}"],
        ["Account Number:", record["account_number"], "Opening Balance:", f"${record['opening_balance']}"],
        ["", "", "Closing Balance:", f"${record['closing_balance']}"],
    ]
    info_table = Table(info_data, colWidths=[1.1 * inch, 2.1 * inch, 1.4 * inch, 1.6 * inch])
    info_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 14))

    story.append(Paragraph("Transaction History", styles["SectionLabel"]))
    table_data = [["Date", "Description", "Debit", "Credit", "Balance"]]
    for t in record["transactions"]:
        table_data.append([
            t["date"], t["description"],
            f"${t['debit']}" if t["debit"] else "",
            f"${t['credit']}" if t["credit"] else "",
            f"${t['balance']}",
        ])
    txn_table = Table(table_data, colWidths=[0.9 * inch, 2.8 * inch, 0.9 * inch, 0.9 * inch, 1 * inch])
    txn_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(accent_hex)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f4f7")]),
        ("ALIGN", (2, 0), (4, -1), "RIGHT"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(txn_table)
    doc.build(story)


def render_style_b(record: dict, out_path: str, accent_hex="#4b2e83"):
    """Schwab-ish layout: centered header, boxed summary, gridless table."""
    doc = SimpleDocTemplate(out_path, pagesize=letter,
                             leftMargin=0.7 * inch, rightMargin=0.7 * inch,
                             topMargin=0.6 * inch, bottomMargin=0.5 * inch)
    styles = _base_styles()
    styles["BankHeader"].alignment = 1  # center
    story = []

    story.append(Paragraph(record["bank_name"], styles["BankHeader"]))
    story.append(Paragraph("Brokerage Account Statement", ParagraphStyle(
        name="centeredgray", parent=styles["SmallGray"], alignment=1)))
    story.append(Spacer(1, 12))

    summary_data = [
        ["Account Holder", record["account_holder"]],
        ["Account Number", record["account_number"]],
        ["Statement Period", f"{record['statement_period_start']} \u2013 {record['statement_period_end']}"],
        ["Opening Balance", f"${record['opening_balance']}"],
        ["Closing Balance", f"${record['closing_balance']}"],
    ]
    summary_table = Table(summary_data, colWidths=[2 * inch, 3.5 * inch])
    summary_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor(accent_hex)),
        ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#dddddd")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 16))

    story.append(Paragraph("Activity", ParagraphStyle(
        name="sectionlabel2", parent=styles["SectionLabel"], textColor=colors.HexColor(accent_hex))))
    table_data = [["Trade Date", "Description", "Debit", "Credit", "Balance"]]
    for t in record["transactions"]:
        table_data.append([
            t["date"], t["description"],
            t["debit"], t["credit"], t["balance"],
        ])
    txn_table = Table(table_data, colWidths=[0.9 * inch, 2.9 * inch, 0.85 * inch, 0.85 * inch, 0.9 * inch])
    txn_table.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, 0), 1, colors.HexColor(accent_hex)),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (2, 0), (4, -1), "RIGHT"),
        ("LINEBELOW", (0, 1), (-1, -1), 0.25, colors.HexColor("#e5e5e5")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(txn_table)
    doc.build(story)


TEMPLATES = {
    "chase_style": lambda record, path: render_style_a(record, path, accent_hex="#005eb8"),
    "schwab_style": lambda record, path: render_style_b(record, path, accent_hex="#4b2e83"),
}
BANK_NAMES = {"chase_style": "Chase Bank", "schwab_style": "Charles Schwab"}


if __name__ == "__main__":
    # Quick smoke test: generate one of each template.
    for template_name, render_fn in TEMPLATES.items():
        rec = generate_true_label(bank_name=BANK_NAMES[template_name])
        out_pdf = f"/home/claude/sample_{template_name}.pdf"
        render_fn(rec, out_pdf)
        print(f"Wrote {out_pdf} with {len(rec['transactions'])} transactions")
