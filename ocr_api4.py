import json
import os
import time
import tempfile

import fitz
import httpx

PDF_PATH = "rag_paper.pdf"
OUTPUT_FILE = "ocr_output.txt"
SPACE = "https://baidu-unlimited-ocr.hf.space"

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

def upload_file(session, filepath):
    url = f"{SPACE}/gradio_api/upload"
    with open(filepath, "rb") as f:
        r = session.post(url, files={"files": f})
    r.raise_for_status()
    return r.json()[0]

session = httpx.Client(timeout=300, verify=True)

print("Converting PDF to images...")
image_paths = pdf_to_images(PDF_PATH, dpi=200)
print(f"Pages: {len(image_paths)}")

all_text = []
for i, img_path in enumerate(image_paths):
    print(f"\nPage {i+1}/{len(image_paths)}: uploading image...")
    uploaded = upload_file(session, img_path)

    payload = {
        "data": [uploaded, "base", "Multi page parsing."],
        "event_data": None,
        "fn_index": None,
        "session_hash": "x",
        "trigger_id": None,
    }
    url = f"{SPACE}/gradio_api/call/run_ocr"
    print(f"  Starting OCR...")
    t0 = time.time()
    with session.stream("POST", url, json=payload) as r:
        r.raise_for_status()
        text = ""
        for line in r.iter_lines():
            if not line:
                continue
            if line.startswith("data: "):
                event_data = json.loads(line[6:])
                if event_data.get("event") == "complete":
                    out = event_data.get("output", {}).get("data", [None])[0]
                    if isinstance(out, dict) and "text" in out:
                        text = out["text"]
                    break
                elif event_data.get("event") == "generating":
                    out = event_data.get("output", {}).get("data", [None])[0]
                    if isinstance(out, dict) and "text" in out:
                        text = out["text"]
                elif event_data.get("event") == "error":
                    raise RuntimeError(str(event_data))
    elapsed = time.time() - t0
    print(f"  {len(text)} chars in {elapsed:.1f}s")
    all_text.append(text)

full_text = "\n\n=== PAGE BREAK ===\n\n".join(all_text)
with open(OUTPUT_FILE, "w") as f:
    f.write(full_text)

print(f"\nDone! Saved to {OUTPUT_FILE}")
print(full_text[:2000])
