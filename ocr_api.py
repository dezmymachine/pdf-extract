import json
import os
import time
from gradio_client import Client

PDF_PATH = "rag_paper.pdf"
OUTPUT_FILE = "ocr_output.txt"

client = Client("baidu/Unlimited-OCR")

print(f"Exploding PDF -> pages...")
result = client.predict(PDF_PATH, api_name="/explode_pdf")
pages = result["pages"] if isinstance(result, dict) else result
print(f"Got {len(pages)} pages")

all_text = []
for i, page_data in enumerate(pages):
    page_path = page_data if isinstance(page_data, str) else page_data.get("path", "")
    print(f"  OCR page {i+1}/{len(pages)}...")
    t0 = time.time()
    for chunk in client.predict(
        page_path,
        "base",
        "Multi page parsing.",
        api_name="/run_ocr"
    ):
        text = chunk["text"]
        tps = time.time() - t0
        chars = len(text)
        print(f"    \r  Page {i+1}: {chars} chars in {tps:.0f}s", end="")
    print()
    all_text.append(text)

full_text = "\n\n=== PAGE BREAK ===\n\n".join(all_text)
with open(OUTPUT_FILE, "w") as f:
    f.write(full_text)

print(f"\nDone! {len(all_text)} pages -> {OUTPUT_FILE}")
print(full_text[:3000])
