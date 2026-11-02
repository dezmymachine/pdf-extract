import json
import logging
import os
import uuid
from pathlib import Path

import httpx

logging.basicConfig(level=logging.INFO, format="%(message)s")

SPACE = "https://baidu-unlimited-ocr.hf.space"
TOKEN = os.environ["HF_TOKEN"]
PDF_PATH = "rag_paper.pdf"

session = httpx.Client(timeout=httpx.Timeout(300), verify=True)
headers = {"Authorization": f"Bearer {TOKEN}"}

# Step 1: Upload PDF
print("Uploading PDF...")
with open(PDF_PATH, "rb") as f:
    r = session.post(f"{SPACE}/gradio_api/upload", files={"files": f}, headers=headers)
r.raise_for_status()
pdf_path = r.json()[0]
print(f"Uploaded: {pdf_path}")

# Step 2: Explode PDF
print("Exploding PDF...")
pdf_filedata = {
    "path": pdf_path,
    "url": f"{SPACE}/file={pdf_path}",
    "size": os.path.getsize(PDF_PATH),
    "orig_name": "rag_paper.pdf",
    "mime_type": "application/pdf",
    "is_stream": False,
    "meta": {"_type": "gradio.FileData"},
}
payload = {
    "data": [pdf_filedata],
    "event_data": None,
    "fn_index": None,
    "session_hash": uuid.uuid4().hex[:10],
    "trigger_id": None,
}
r = session.post(f"{SPACE}/gradio_api/call/explode_pdf", json=payload, headers=headers)
r.raise_for_status()
event_id = r.json()["event_id"]

print(f"Polling explode_pdf result ({event_id})...")
with session.stream("GET", f"{SPACE}/gradio_api/call/explode_pdf/{event_id}", headers=headers) as sr:
    for line in sr.iter_lines():
        if not line:
            continue
        if line.startswith("data: "):
            data = json.loads(line[6:])
            if data.get("event") == "complete":
                result = data.get("output", {}).get("data")
                print(f"Explode result: {json.dumps(result, indent=2)[:500]}")
                break
            elif data.get("event") == "error":
                print(f"Error: {data}")
                break
