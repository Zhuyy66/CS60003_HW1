from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    CondPageBreak,
    Image,
    KeepTogether,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


BASE = Path(__file__).resolve().parent
MARKDOWN = BASE / "final_report.md"
PDF = BASE / "final_report.pdf"


def register_fonts() -> str:
    font_name = "STSong-Light"
    pdfmetrics.registerFont(UnicodeCIDFont(font_name))
    return font_name


def make_styles(font_name: str) -> dict[str, ParagraphStyle]:
    styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleCN",
            parent=styles["Title"],
            fontName=font_name,
            fontSize=20,
            leading=28,
            spaceAfter=18,
            wordWrap="CJK",
        ),
        "h1": ParagraphStyle(
            "Heading1CN",
            parent=styles["Heading1"],
            fontName=font_name,
            fontSize=15,
            leading=22,
            spaceBefore=14,
            spaceAfter=8,
            keepWithNext=True,
            wordWrap="CJK",
        ),
        "h2": ParagraphStyle(
            "Heading2CN",
            parent=styles["Heading2"],
            fontName=font_name,
            fontSize=13,
            leading=19,
            spaceBefore=10,
            spaceAfter=6,
            keepWithNext=True,
            wordWrap="CJK",
        ),
        "body": ParagraphStyle(
            "BodyCN",
            parent=styles["BodyText"],
            fontName=font_name,
            fontSize=10,
            leading=15.5,
            firstLineIndent=18,
            spaceAfter=5,
            wordWrap="CJK",
        ),
        "bullet": ParagraphStyle(
            "BulletCN",
            parent=styles["BodyText"],
            fontName=font_name,
            fontSize=10,
            leading=15.5,
            leftIndent=18,
            firstLineIndent=-10,
            spaceAfter=4,
            wordWrap="CJK",
        ),
        "code": ParagraphStyle(
            "Code",
            parent=styles["Code"],
            fontName="Courier",
            fontSize=8.5,
            leading=11,
            leftIndent=10,
            rightIndent=10,
            backColor=colors.whitesmoke,
            borderPadding=6,
            spaceBefore=4,
            spaceAfter=8,
        ),
        "caption": ParagraphStyle(
            "CaptionCN",
            parent=styles["BodyText"],
            fontName=font_name,
            fontSize=9,
            leading=12,
            alignment=1,
            textColor=colors.dimgray,
            spaceAfter=8,
            wordWrap="CJK",
        ),
    }


def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = text.replace("`", "")
    return Paragraph(text, style)


def parse_table(lines: list[str], font_name: str) -> Table:
    rows = []
    for line in lines:
        if set(line.replace("|", "").replace("-", "").replace(":", "").strip()) == set():
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        rows.append(cells)

    col_count = max(len(row) for row in rows)
    col_widths = None
    if col_count == 4:
        col_widths = [5.1 * cm, 2.5 * cm, 2.2 * cm, 2.3 * cm]
    elif col_count == 7:
        col_widths = [2.2 * cm, 2.3 * cm, 2.0 * cm, 2.0 * cm, 2.0 * cm, 2.1 * cm, 1.8 * cm]

    table = Table(rows, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), font_name),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("LEADING", (0, 0), (-1, -1), 10),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAEFF7")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def render() -> None:
    font_name = register_fonts()
    styles = make_styles(font_name)
    doc = SimpleDocTemplate(
        str(PDF),
        pagesize=A4,
        rightMargin=1.7 * cm,
        leftMargin=1.7 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
    )

    story = []
    lines = MARKDOWN.read_text(encoding="utf-8").splitlines()
    i = 0
    in_code = False
    code_lines: list[str] = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            if in_code:
                story.append(Preformatted("\n".join(code_lines), styles["code"]))
                code_lines = []
                in_code = False
            else:
                in_code = True
            i += 1
            continue

        if in_code:
            code_lines.append(line)
            i += 1
            continue

        if not stripped:
            i += 1
            continue

        if stripped.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            story.append(CondPageBreak(4.0 * cm))
            story.append(parse_table(table_lines, font_name))
            story.append(Spacer(1, 8))
            continue

        if stripped.startswith("!["):
            alt = stripped.split("]", 1)[0][2:]
            rel = stripped.split("(", 1)[1].rstrip(")")
            image_path = BASE / rel
            if image_path.exists():
                width = 15.5 * cm
                height = 7.4 * cm
                if "confusion" in image_path.name:
                    width = 13.2 * cm
                    height = 11.0 * cm
                elif "error" in image_path.name:
                    height = 10.2 * cm
                elif "first_layer_weight_grid" in image_path.name:
                    width = 12.7 * cm
                    height = 12.7 * cm
                elif "all_class_related_weights" in image_path.name:
                    width = 13.2 * cm
                    height = 17.5 * cm
                elif "forest_river_related_weights" in image_path.name or "linearized_class_templates" in image_path.name:
                    height = 7.0 * cm
                block = [
                    Image(str(image_path), width=width, height=height, kind="proportional"),
                    paragraph(alt, styles["caption"]),
                ]
                story.append(CondPageBreak(min(height + 1.0 * cm, 18.0 * cm)))
                story.append(KeepTogether(block))
            i += 1
            continue

        if stripped.startswith("# "):
            story.append(Paragraph(stripped[2:], styles["title"]))
        elif stripped.startswith("## "):
            if story:
                story.append(Spacer(1, 4))
                story.append(CondPageBreak(4.5 * cm))
            story.append(Paragraph(stripped[3:], styles["h1"]))
        elif stripped.startswith("### "):
            story.append(CondPageBreak(3.5 * cm))
            story.append(Paragraph(stripped[4:], styles["h2"]))
        elif stripped.startswith("- "):
            story.append(paragraph("• " + stripped[2:], styles["bullet"]))
        else:
            story.append(paragraph(stripped, styles["body"]))
        i += 1

    doc.build(story)
    print(PDF)


if __name__ == "__main__":
    render()
