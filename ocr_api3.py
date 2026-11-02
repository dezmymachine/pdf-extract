import os
import time
import tempfile

import fitz
import httpx
from gradio_client import Client

PDF_PATH = "rag_paper.pdf"
OUTPUT_FILE = "ocr_output.txt"

def pdf_to_images(pdf_path, dpi=300):
    doc = fitz.open(pdf_path)
    tmp_dir = tempfile.mkdtemp(prefix="pdf_ocr_")
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    paths = []
    for i, page in enumerate(doc):
        out = os.path.join(tmp_dir, f"page_{i+1:04d}.png")
        page.get_pixmap(matrix=mat).save(out)
        paths.append(out)
    doc.close()
    return paths

print("Converting PDF to images locally...")
image_paths = pdf_to_images(PDF_PATH, dpi=200)
print(f"Created {len(image_paths)} page images")

client = Client("baidu/Unlimited-OCR")

all_text = []
for i, img_path in enumerate(image_paths):
    print(f"\nPage {i+1}/{len(image_paths)}...")
    t0 = time.time()
    last_text = ""
    for chunk in client.predict(
        img_path,
        "base",
        "Multi page parsing.",
        api_name="/run_ocr"
    ):
        last_text = chunk["text"]
    if not last_text:
        print("  (empty result, retrying with gundam mode)")
        for chunk in client.predict(
            img_path,
            "gundam",
            "Multi page parsing.",
            api_name="/run_ocr"
        ):
            last_text = chunk["text"]
    elapsed = time.time() - t0
    print(f"  {len(last_text)} chars in {elapsed:.1f}s")
    all_text.append(last_text)

full_text = "\n\n=== PAGE BREAK ===\n\n".join(all_text)
with open(OUTPUT_FILE, "w") as f:
    f.write(full_text)

print(f"\nDone! Saved to {OUTPUT_FILE}")
print(full_text[:2000])
