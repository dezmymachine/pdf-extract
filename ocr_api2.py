import json
import time
import httpx

SPACE = "https://baidu-unlimited-ocr.hf.space"
PDF_PATH = "rag_paper.pdf"
OUTPUT_FILE = "ocr_output.txt"

session = httpx.Client(timeout=1200, verify=True)

def upload_file(filepath):
    url = f"{SPACE}/gradio_api/upload"
    with open(filepath, "rb") as f:
        r = session.post(url, files={"files": f})
    r.raise_for_status()
    return r.json()[0]

def call_api(api_name, data, stream=False):
    payload = {
        "data": data,
        "event_data": None,
        "fn_index": None,
        "session_hash": "x",
        "trigger_id": None,
    }
    url = f"{SPACE}/gradio_api/call/{api_name.lstrip('/')}"
    if stream:
        with session.stream("POST", url, json=payload) as r:
            r.raise_for_status()
            output = ""
            for line in r.iter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    event_data = json.loads(line[6:])
                    if event_data.get("event") == "complete":
                        out = event_data.get("output", {}).get("data", [None])[0]
                        if out:
                            output = out.get("text", "")
                        break
                    elif event_data.get("event") == "generating":
                        out = event_data.get("output", {}).get("data", [None])[0]
                        if out:
                            output = out.get("text", "")
                    elif "generating" in str(event_data.get("event", "")):
                        pass
            return output
    else:
        r = session.post(url, json=payload)
        r.raise_for_status()
        event_id = r.json()["event_id"]
        # Poll for result
        result_url = f"{SPACE}/gradio_api/call/{api_name.lstrip('/')}/{event_id}"
        for _ in range(300):
            with session.stream("GET", result_url) as sr:
                for line in sr.iter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        event_data = json.loads(line[6:])
                        if event_data.get("event") == "complete":
                            return event_data.get("output", {}).get("data")
                        elif event_data.get("event") == "error":
                            raise RuntimeError(str(event_data))
            time.sleep(0.5)
        raise TimeoutError("API call timed out")

print("Uploading PDF to Space...")
uploaded = upload_file(PDF_PATH)
print(f"Uploaded: {uploaded}")

print("Exploding PDF into pages...")
result = call_api("/explode_pdf", [uploaded])
print(f"Result: {result}")

if result and len(result) > 0:
    pages = result[0] if isinstance(result, list) else result
    if isinstance(pages, dict) and "pages" in pages:
        pages = pages["pages"]
    print(f"Got {len(pages)} pages")
    for i, p in enumerate(pages):
        print(f"  Page {i}: {p}")
else:
    print("No result from explode_pdf")
    pages = []
