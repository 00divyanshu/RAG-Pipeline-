from pathlib import Path

def create_sample_pdf(filepath: Path):
    """
    Creates a valid 2-page PDF with sample knowledge base content
    about a fictional company's policies.
    """
    # PDF 1.4 specification
    page1_text = (
        "BT /F1 18 Tf 50 720 Td (Acme Corp - Remote Work and Equipment Policy 2026) Tj ET "
        "BT /F1 12 Tf 50 680 Td (Section 1: Work Hours and Core Availability) Tj ET "
        "BT /F1 10 Tf 50 650 Td (Employees are required to be online during core hours from 10:00 AM to 3:00 PM EST.) Tj ET "
        "BT /F1 10 Tf 50 630 Td (Flexible hours can be arranged with team leads outside core hours.) Tj ET "
        "BT /F1 12 Tf 50 590 Td (Section 2: Home Office Equipment Allowance) Tj ET "
        "BT /F1 10 Tf 50 560 Td (Each full-time remote employee is eligible for an annual home office allowance of $1500 USD.) Tj ET "
        "BT /F1 10 Tf 50 540 Td (Covered items include ergonomic desks, chairs, external monitors, and noise-canceling headsets.) Tj ET "
        "BT /F1 10 Tf 50 520 Td (Expense claims must be submitted via Expensify within 30 days of purchase.) Tj ET"
    )

    page2_text = (
        "BT /F1 18 Tf 50 720 Td (Acme Corp - Leave and Vacation Benefits) Tj ET "
        "BT /F1 12 Tf 50 680 Td (Section 3: Annual Paid Time Off) Tj ET "
        "BT /F1 10 Tf 50 650 Td (All regular employees receive 25 days of paid time off per calendar year.) Tj ET "
        "BT /F1 10 Tf 50 630 Td (Unused PTO can rollover up to a maximum of 5 days into the subsequent year.) Tj ET "
        "BT /F1 12 Tf 50 590 Td (Section 4: Health and Wellness Stipend) Tj ET "
        "BT /F1 10 Tf 50 560 Td (Acme provides a monthly wellness stipend of $100 for gym memberships, fitness classes, or mental health apps.) Tj ET "
        "BT /F1 10 Tf 50 540 Td (For questions, contact the People Operations team at people-ops@acmecorp.example.) Tj ET"
    )

    stream1 = page1_text.encode("latin1")
    stream2 = page2_text.encode("latin1")

    objects = []
    
    # 1: Catalog
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    # 2: Pages
    objects.append(b"<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>")
    # 3: Page 1
    objects.append(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 6 0 R >>")
    # 4: Page 2
    objects.append(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 7 0 R >>")
    # 5: Font
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    # 6: Stream 1
    objects.append(f"<< /Length {len(stream1)} >>\nstream\n".encode("latin1") + stream1 + b"\nendstream")
    # 7: Stream 2
    objects.append(f"<< /Length {len(stream2)} >>\nstream\n".encode("latin1") + stream2 + b"\nendstream")

    pdf_bytes = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []

    for i, obj in enumerate(objects, 1):
        offsets.append(len(pdf_bytes))
        pdf_bytes.extend(f"{i} 0 obj\n".encode("latin1"))
        pdf_bytes.extend(obj)
        pdf_bytes.extend(b"\nendobj\n")

    xref_offset = len(pdf_bytes)
    pdf_bytes.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("latin1"))
    for offset in offsets:
        pdf_bytes.extend(f"{offset:010d} 00000 n \n".encode("latin1"))

    pdf_bytes.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("latin1")
    )

    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        f.write(pdf_bytes)
    print(f"Sample PDF created at: {filepath}")

if __name__ == "__main__":
    create_sample_pdf(Path("data/docs/sample_company_policy.pdf"))

