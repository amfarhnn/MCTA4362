from __future__ import annotations

import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "Example template report.docx"
REPORT_MARKDOWN = ROOT / "docs" / "report_draft.md"
OUTPUT = ROOT / "MCTA4362_DC_Motor_Report_Draft.docx"
WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W = f"{{{WORD_NS}}}"


def paragraph(text: str = "", style: str | None = None) -> ET.Element:
    p = ET.Element(f"{W}p")
    if style:
        p_pr = ET.SubElement(p, f"{W}pPr")
        p_style = ET.SubElement(p_pr, f"{W}pStyle")
        p_style.set(f"{W}val", style)
    run = ET.SubElement(p, f"{W}r")
    t = ET.SubElement(run, f"{W}t")
    t.text = text
    if text.startswith(" ") or text.endswith(" "):
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    return p


def markdown_table_line(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def is_markdown_separator(line: str) -> bool:
    content = line.replace("|", "").replace(":", "").replace("-", "").strip()
    return content == ""


def table(rows: list[list[str]]) -> ET.Element:
    tbl = ET.Element(f"{W}tbl")
    tbl_pr = ET.SubElement(tbl, f"{W}tblPr")
    tbl_style = ET.SubElement(tbl_pr, f"{W}tblStyle")
    tbl_style.set(f"{W}val", "TableGrid")
    tbl_width = ET.SubElement(tbl_pr, f"{W}tblW")
    tbl_width.set(f"{W}w", "0")
    tbl_width.set(f"{W}type", "auto")

    for row in rows:
        tr = ET.SubElement(tbl, f"{W}tr")
        for cell in row:
            tc = ET.SubElement(tr, f"{W}tc")
            tc_pr = ET.SubElement(tc, f"{W}tcPr")
            tc_width = ET.SubElement(tc_pr, f"{W}tcW")
            tc_width.set(f"{W}w", "2400")
            tc_width.set(f"{W}type", "dxa")
            tc.append(paragraph(cell, "Normal"))
    return tbl


def markdown_to_paragraphs(markdown: str) -> list[ET.Element]:
    elements: list[ET.Element] = []
    in_code_block = False
    lines = markdown.splitlines()
    index = 0

    while index < len(lines):
        raw_line = lines[index]
        line = raw_line.rstrip()

        if line.startswith("```"):
            in_code_block = not in_code_block
            index += 1
            continue

        if in_code_block:
            if line:
                elements.append(paragraph(line, "Normal"))
            index += 1
            continue

        if not line:
            elements.append(paragraph(""))
            index += 1
            continue

        if line.startswith("|"):
            table_lines = []
            while index < len(lines) and lines[index].rstrip().startswith("|"):
                table_lines.append(lines[index].rstrip())
                index += 1
            rows = [
                markdown_table_line(item)
                for item in table_lines
                if not is_markdown_separator(item)
            ]
            if rows:
                elements.append(table(rows))
            continue

        if line.startswith("# "):
            elements.append(paragraph(line[2:], "Title"))
        elif line.startswith("## "):
            elements.append(paragraph(line[3:], "Heading1"))
        elif line.startswith("### "):
            elements.append(paragraph(line[4:], "Heading2"))
        elif line.startswith("- "):
            elements.append(paragraph("- " + line[2:], "ListParagraph"))
        else:
            elements.append(paragraph(line, "Normal"))
        index += 1

    return elements


def build_report() -> None:
    if not TEMPLATE.exists():
        raise FileNotFoundError(f"Missing template: {TEMPLATE}")
    if not REPORT_MARKDOWN.exists():
        raise FileNotFoundError(f"Missing report draft: {REPORT_MARKDOWN}")

    ET.register_namespace("w", WORD_NS)

    markdown = REPORT_MARKDOWN.read_text(encoding="utf-8")
    with zipfile.ZipFile(TEMPLATE, "r") as source:
        document_xml = source.read("word/document.xml")
        root = ET.fromstring(document_xml)
        body = root.find(f"{W}body")
        if body is None:
            raise RuntimeError("Could not find Word document body.")

        section_properties = body.find(f"{W}sectPr")
        body.clear()
        for item in markdown_to_paragraphs(markdown):
            body.append(item)
        if section_properties is not None:
            body.append(section_properties)

        updated_document = ET.tostring(root, encoding="utf-8", xml_declaration=True)

        temp_output = OUTPUT.with_suffix(".tmp.docx")
        with zipfile.ZipFile(temp_output, "w", zipfile.ZIP_DEFLATED) as target:
            for item in source.infolist():
                if item.filename == "word/document.xml":
                    target.writestr(item, updated_document)
                else:
                    target.writestr(item, source.read(item.filename))

    if OUTPUT.exists():
        OUTPUT.unlink()
    temp_output.replace(OUTPUT)

    print(f"Generated: {OUTPUT}")


if __name__ == "__main__":
    build_report()
