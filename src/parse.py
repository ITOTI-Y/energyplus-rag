import os
import subprocess
from collections.abc import Generator
from pathlib import Path

import pypdfium2 as pdfium
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

MINERU_VRAM = os.getenv("MINERU_VRAM", "30")


def parse_pdf(
    pdf_file: str | Path,
    output_dir: str | Path,
) -> Generator[str, None, str]:
    pdf = pdfium.PdfDocument(pdf_file)
    total_pages = len(pdf)
    pdf.close()

    pdf_file = Path(pdf_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_name = pdf_file.stem
    pdf_output_dir = output_dir / pdf_name
    pdf_output_dir.mkdir(parents=True, exist_ok=True)

    yield f"Parsing {pdf_name}: have {total_pages} pages"

    cmd = ["mineru", "-p", str(pdf_file), "--vram", f"{MINERU_VRAM}"]

    step = 200
    for i in range(0, total_pages, step):
        start_page = i
        end_page = min(i + step, total_pages)
        yield f"Parsing {pdf_name}: page {start_page} to {end_page}"

        try:
            input_cmd = [
                *cmd,
                "-o",
                str(pdf_output_dir / f"page_{i:04d}_{end_page:04d}"),
                "-s",
                f"{i}",
                "-e",
                f"{end_page}",
            ]

            process = subprocess.Popen(
                input_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            if process.stdout:
                for line in process.stdout:
                    line = line.strip()
                    if line:
                        yield f"  [mineru] {line}"

            process.wait()

            if process.returncode != 0:
                logger.error(f"Failed to parse {pdf_file}")
                yield f"  [mineru] Failed to parse {pdf_file} in page {start_page} to {end_page}"
                continue
            else:
                yield f"  [mineru] Parsed {pdf_file} in page {start_page} to {end_page} successfully"

        except subprocess.TimeoutExpired:
            logger.error(f"Timeout while parsing {pdf_file}")
            yield f"  [mineru] Timeout while parsing {pdf_file} in page {start_page} to {end_page}"
            continue
        except FileNotFoundError:
            logger.error("Failed to find mineru, please install mineru first.")
            yield "  [mineru] Failed to find mineru, please install mineru first."
            return ""
    yield "combining output files into a single json file..."
    result = _combine_output(pdf_output_dir)
    yield "  [mineru] Combined output files into a single json file successfully"
    yield f"RESULT:{result}"


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
                page_length = data[-1]["page_idx"]
                data = [
                    {**item, "page_idx": absolute_page_idx + int(item["page_idx"])}
                    for item in data
                ]
                result.extend(data)
                absolute_page_idx += page_length
        json.dump(result, outfile, ensure_ascii=False, indent=4)

    logger.info(f"Combined json file created at {combined_json}")
    return str(combined_json)
