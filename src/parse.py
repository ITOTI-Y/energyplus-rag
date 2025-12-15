import os
import subprocess
from pathlib import Path

import pypdfium2 as pdfium
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

MINERU_VRAM = os.getenv("MINERU_VRAM", "30")

def parse_pdf(
    pdf_file: str | Path,
    output_dir: str | Path,
) -> str:

    pdf = pdfium.PdfDocument(pdf_file)
    total_pages = len(pdf)
    pdf.close()

    pdf_file = Path(pdf_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_name = pdf_file.stem
    pdf_output_dir = output_dir / pdf_name
    pdf_output_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["mineru", "-p", str(pdf_file), "--vram", f"{MINERU_VRAM}"]

    step = 25
    for i in range(1000, total_pages, step):
        try:
            input_cmd = [
                *cmd, "-o", str(pdf_output_dir / f"page_{i:04d}_{min(i+step, total_pages):04d}"), "-s", f"{i}", "-e", f"{min(i+step, total_pages)}"]
            result = subprocess.run(
                input_cmd, capture_output=True, check=True)
            if result.returncode != 0:
                logger.error(f"Failed to parse {pdf_file}")
                continue
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout while parsing {pdf_file}")
            continue
        except FileNotFoundError:
            logger.error(
                "Failed to find mineru, please install mineru first.")
            return ""
    return _combine_output(pdf_output_dir)

def _combine_output(output_dir: str | Path) -> str:
    import json
    output_dir = Path(output_dir)
    json_files = output_dir.glob("**/*_content_list.json")
    json_files = sorted(json_files, key=lambda x: x.parents[2].name)

    combined_json = output_dir / f"{output_dir.name}.json"

    with combined_json.open("w", encoding="utf-8") as outfile:
        absolute_page_idx = 0
        result = []
        for json_file in json_files:
            with json_file.open("r", encoding="utf-8") as infile:
                data = json.load(infile)
                page_length = data[-1]['page_idx']
                data = [{**item, "page_idx": absolute_page_idx + int(item['page_idx'])} for item in data]
                result.extend(data)
                absolute_page_idx += page_length
        json.dump(result, outfile, ensure_ascii=False, indent=4)

    logger.info(f"Combined json file created at {combined_json}")
    return str(combined_json)
