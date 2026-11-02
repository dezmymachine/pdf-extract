import json
import logging
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Callable, Literal, Optional

import httpx

try:
    import fitz
except ImportError:
    fitz = None

logger = logging.getLogger(__name__)

_DEFAULT_SPACE = "https://baidu-unlimited-ocr.hf.space"


def pdf_to_markdown(
    pdf_path: str | Path,
    hf_token: str,
    *,
    mode: Literal["base", "gundam"] = "base",
    dpi: int = 200,
    space_url: str = _DEFAULT_SPACE,
    max_retries: int = 5,
    retry_delay: int = 30,
    page_timeout: int = 300,
    page_break: str = "\n\n---\n\n",
    on_page_done: Optional[Callable[[int, int, float], None]] = None,
) -> str:
    """Extract text from a PDF as markdown via the baidu/Unlimited-OCR HF Space.

    Each page is converted to an image locally, uploaded to the Space,
    and OCR'd via the serverless GPU.

    Args:
        pdf_path: Path to the PDF file.
        hf_token: Hugging Face user access token.
        mode: 'base' (accurate, 1024px) or 'gundam' (faster, 640px crop).
        dpi: Resolution for local PDF-to-image conversion.
        space_url: HF Space URL.
        max_retries: Max times to retry when ZeroGPU queue is busy.
        retry_delay: Base seconds between retries.
        page_timeout: Max seconds per page OCR.
        page_break: Delimiter inserted between pages.
        on_page_done: Callback(page_index, total_pages, elapsed_seconds).

    Returns:
        Full markdown text of the document.
    """
    if fitz is None:
        raise ImportError(
            "pymupdf is required for PDF-to-image conversion. "
            "Install it with: pip install pymupdf"
        )

    space_url = space_url.rstrip("/")
    session = httpx.Client(timeout=httpx.Timeout(page_timeout + 30), verify=True)
    hf_headers = {"Authorization": f"Bearer {hf_token}"}
    upload_url = f"{space_url}/gradio_api/upload"

    # ── 1. Convert PDF to page images locally ────────────────────────────────
    logger.info("Converting PDF to page images...")
    doc = fitz.open(pdf_path)
    tmp_dir = tempfile.mkdtemp(prefix="pdf_to_md_")
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    page_images: list[str] = []
    for i in range(len(doc)):
        out = os.path.join(tmp_dir, f"page_{i + 1:04d}.png")
        doc[i].get_pixmap(matrix=mat).save(out)
        page_images.append(out)
    doc.close()
    total_pages = len(page_images)
    logger.info(f"  -> {total_pages} pages")

    # ── 2. OCR each page ─────────────────────────────────────────────────────
    all_text: list[str] = []
    for idx, img_path in enumerate(page_images):
        logger.info(f"Page {idx + 1}/{total_pages} -- OCR ({mode})")
        t0 = time.time()

        # Upload page image to Space
        with open(img_path, "rb") as f:
            r = session.post(upload_url, files={"files": f}, headers=hf_headers)
        r.raise_for_status()
        server_path = r.json()[0]

        page_filedata = _make_filedata(server_path, img_path, space_url)
        text = _ocr_page_with_retry(
            session, space_url, hf_headers,
            page_filedata, mode,
            max_retries, retry_delay,
        )
        elapsed = time.time() - t0
        logger.info(f"  -> {len(text)} chars in {elapsed:.0f}s")
        all_text.append(text)

        if on_page_done:
            on_page_done(idx + 1, total_pages, elapsed)

    session.close()

    # Cleanup temp images
    for p in page_images:
        try:
            os.remove(p)
        except OSError:
            pass
    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass

    return page_break.join(all_text)


# ── Internal helpers ─────────────────────────────────────────────────────────


def _make_filedata(
    server_path: str,
    local_path: str | Path | None = None,
    space_url: str = "",
    name: str = "",
) -> dict:
    size = 0
    basename = name
    if local_path:
        p = Path(local_path)
        size = p.stat().st_size if p.exists() else 0
        basename = basename or p.name
    basename = basename or Path(server_path).name
    ext = os.path.splitext(basename)[1].lower()
    mime = {
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".tiff": "image/tiff",
        ".webp": "image/webp",
    }.get(ext, "application/octet-stream")
    return {
        "path": server_path,
        "url": f"{space_url}/file={server_path}",
        "size": size,
        "orig_name": basename,
        "mime_type": mime,
        "is_stream": False,
        "meta": {"_type": "gradio.FileData"},
    }


def _call_api(
    session: httpx.Client,
    space_url: str,
    headers: dict,
    api_name: str,
    data: list,
) -> list:
    session_hash = uuid.uuid4().hex[:10]
    payload = {
        "data": data,
        "event_data": None,
        "fn_index": None,
        "session_hash": session_hash,
        "trigger_id": None,
    }
    url = f"{space_url}/gradio_api/call/{api_name}"

    r = session.post(url, json=payload, headers=headers)
    r.raise_for_status()
    event_id = r.json()["event_id"]

    return _read_sse(session, f"{url}/{event_id}", headers)


def _read_sse(session: httpx.Client, url: str, headers: dict) -> list:
    """Read a Gradio SSE stream to completion. Returns the output data array."""
    with session.stream("GET", url, headers=headers) as r:
        r.raise_for_status()
        event_type = None
        last_output: list = []

        for line in r.iter_lines():
            line = line.strip()
            if not line:
                if event_type == "error":
                    err = last_output[0] if isinstance(last_output, list) and last_output else {"error": str(last_output)}
                    raise RuntimeError(f"Space API error: {err.get('error', err)}")
                event_type = None
                continue

            if line.startswith("event: "):
                event_type = line[7:].strip()
            elif line.startswith("data: "):
                raw = json.loads(line[6:])
                # raw is always a list of output values
                last_output = raw if isinstance(raw, list) else [raw]

                if event_type == "complete":
                    return last_output
                elif event_type == "error":
                    err_info = last_output[0] if last_output else raw
                    if isinstance(err_info, dict) and "error" in err_info:
                        raise RuntimeError(f"Space API error: {err_info['error']}")
                    raise RuntimeError(f"Space API error: {err_info}")
                elif event_type in ("generating", "heartbeat", None):
                    pass

        return last_output


def _ocr_page_with_retry(
    session, space_url, headers, filedata, mode, max_retries, retry_delay,
) -> str:
    for attempt in range(1, max_retries + 1):
        try:
            result = _call_api(
                session, space_url, headers, "run_ocr",
                [filedata, mode, "Multi page parsing."],
            )
            text = ""
            if result:
                if isinstance(result, list) and len(result) > 0:
                    out = result[0]
                    text = out.get("text", "") if isinstance(out, dict) else str(out)
                elif isinstance(result, dict):
                    text = result.get("text", "")
                elif isinstance(result, str):
                    text = result
            if text:
                return text
            if attempt < max_retries:
                logger.warning(f"  Empty result, retry {attempt}/{max_retries}...")
                time.sleep(retry_delay)
        except RuntimeError as e:
            err_msg = str(e)
            if "No GPU was available" in err_msg or "ZeroGPU" in err_msg:
                if attempt < max_retries:
                    wait = retry_delay * attempt
                    logger.warning(
                        f"  ZeroGPU busy (attempt {attempt}/{max_retries}), "
                        f"waiting {wait}s..."
                    )
                    time.sleep(wait)
                    continue
            raise
    raise RuntimeError(f"Page OCR failed after {max_retries} retries")
