import json
import os
import time
import tempfile
import uuid

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

session = httpx.Client(timeout=600, verify=True)
SESSION_HASH = uuid.uuid4().hex[:10]

def upload_file(filepath):
    url = f"{SPACE}/gradio_api/upload"
    with open(filepath, "rb") as f:
        r = session.post(url, files={"files": f})
    r.raise_for_status()
    server_path = r.json()[0]
    return {
        "path": server_path,
        "url": f"{SPACE}/file={server_path}",
        "size": os.path.getsize(filepath),
        "orig_name": os.path.basename(filepath),
        "mime_type": "image/png",
        "is_stream": False,
        "meta": {"_type": "gradio.FileData"},
    }

def call_run_ocr(filedata, mode, prompt):
    url = f"{SPACE}/gradio_api/call/run_ocr"
    payload = {
        "data": [filedata, mode, prompt],
        "event_data": None,
        "fn_index": None,
        "session_hash": SESSION_HASH + str(int(time.time())),
        "trigger_id": None,
    }
    r = session.post(url, json=payload)
    r.raise_for_status()
    event_id = r.json()["event_id"]
    result_url = f"{SPACE}/gradio_api/call/run_ocr/{event_id}"
    text = ""
    with session.stream("GET", result_url) as sr:
        for line in sr.iter_lines():
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
    return text

print("Converting PDF to images...")
image_paths = pdf_to_images(PDF_PATH, dpi=200)
print(f"Pages: {len(image_paths)}")

all_text = []
for i, img_path in enumerate(image_paths):
    print(f"\nPage {i+1}/{len(image_paths)}: uploading...")
    filedata = upload_file(img_path)
    print(f"  Calling run_ocr (base)...")
    t0 = time.time()
    text = call_run_ocr(filedata, "base", "Multi page parsing.")
    if not text:
        print(f"  Retry with gundam...")
        text = call_run_ocr(filedata, "gundam", "Multi page parsing.")
    elapsed = time.time() - t0
    print(f"  {len(text)} chars in {elapsed:.1f}s")
    all_text.append(text)

full_text = "\n\n=== PAGE BREAK ===\n\n".join(all_text)
with open(OUTPUT_FILE, "w") as f:
    f.write(full_text)

print(f"\nDone! Saved to {OUTPUT_FILE}")
print(full_text[:3000])
