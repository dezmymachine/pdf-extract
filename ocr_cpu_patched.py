import os
import sys
import tempfile
import time

import fitz

import torch
original_cuda = torch.Tensor.cuda

def patched_cuda(self, *args, **kwargs):
    return self

original_autocast = torch.autocast

def patched_autocast(device_type, *args, **kwargs):
    if device_type == "cuda":
        return original_autocast("cpu", *args, **{k: v for k, v in kwargs.items() if k != 'dtype'})
    return original_autocast(device_type, *args, **kwargs)

torch.Tensor.cuda = patched_cuda

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from transformers import AutoModel, AutoTokenizer

PDF_PATH = "rag_paper.pdf"
OUTPUT_DIR = "ocr_output_cpu"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def pdf_to_images(pdf_path, dpi=200, max_pages=None):
    doc = fitz.open(pdf_path)
    tmp_dir = tempfile.mkdtemp(prefix="pdf_ocr_")
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    paths = []
    total = len(doc)
    pages_to_do = min(total, max_pages) if max_pages else total
    for i in range(pages_to_do):
        page = doc[i]
        out = os.path.join(tmp_dir, f"page_{i+1:04d}.png")
        page.get_pixmap(matrix=mat).save(out)
        paths.append(out)
        print(f"  Converted page {i+1}/{total}")
    doc.close()
    return paths

print("Loading model (CPU, patched)...")
t0 = time.time()
model_name = "baidu/Unlimited-OCR"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True,
    use_safetensors=True,
    dtype=torch.bfloat16,
    device_map="cpu",
)
model = model.eval()
print(f"Model loaded in {time.time()-t0:.1f}s")

image_files = pdf_to_images(PDF_PATH, dpi=150, max_pages=1)
print(f"Page images: {image_files}")

print("Running OCR (patched CPU)...")
t0 = time.time()
try:
    model.infer_multi(
        tokenizer,
        prompt="<image>Multi page parsing.",
        image_files=image_files,
        output_path=OUTPUT_DIR,
        image_size=1024,
        max_length=32768,
        no_repeat_ngram_size=35,
        ngram_window=1024,
        save_results=True,
    )
    print(f"OCR completed in {time.time()-t0:.1f}s")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
