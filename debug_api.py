import json
import os
import time
import tempfile
import uuid

import fitz
import httpx

SPACE = "https://baidu-unlimited-ocr.hf.space"

session = httpx.Client(timeout=600, verify=True)

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

# Test with a simple image
img_path = "test_page.png"
doc = fitz.open("rag_paper.pdf")
page = doc[0]
mat = fitz.Matrix(200 / 72, 200 / 72)
page.get_pixmap(matrix=mat).save(img_path)
doc.close()

print("Uploading...")
filedata = upload_file(img_path)

url = f"{SPACE}/gradio_api/call/run_ocr"
payload = {
    "data": [filedata, "gundam", "document parsing."],
    "event_data": None,
    "fn_index": None,
    "session_hash": uuid.uuid4().hex[:10],
    "trigger_id": None,
}
r = session.post(url, json=payload)
print(f"POST status: {r.status_code}")
print(f"POST response: {r.text[:500]}")
r.raise_for_status()
event_id = r.json()["event_id"]

result_url = f"{SPACE}/gradio_api/call/run_ocr/{event_id}"
print(f"Streaming from {result_url}...")
with session.stream("GET", result_url) as sr:
    for i, line in enumerate(sr.iter_lines()):
        if not line:
            continue
        print(f"Line {i}: {line[:300]}")
        if i > 20:
            print("... truncated")
            break
