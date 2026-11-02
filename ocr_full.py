import os
import sys
import tempfile
import time

import fitz

import torch
torch.Tensor.cuda = lambda self, *a, **kw: self

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from transformers import AutoModel, AutoTokenizer

PDF_PATH = "rag_paper.pdf"
OUTPUT_DIR = "ocr_output_cpu"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def pdf_to_images_single(pdf_path, page_num, dpi=200):
    doc = fitz.open(pdf_path)
    tmp_dir = tempfile.mkdtemp(prefix="pdf_ocr_")
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    page = doc[page_num]
    out = os.path.join(tmp_dir, f"page_{page_num+1:04d}.png")
    page.get_pixmap(matrix=mat).save(out)
    doc.close()
    return out

print("Loading model (CPU infer patched)...")
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

pages = [0, 1, 5, 10, 20]

for pnum in pages:
    print(f"\n--- Page {pnum+1} ---")
    img = pdf_to_images_single(PDF_PATH, pnum, dpi=200)
    t0 = time.time()
    model.infer_multi(
        tokenizer,
        prompt="<image>Multi page parsing.",
        image_files=[img],
        output_path=OUTPUT_DIR,
        image_size=1024,
        max_length=32768,
        no_repeat_ngram_size=35,
        ngram_window=1024,
        save_results=False,
    )
    elapsed = time.time() - t0
    print(f"Time: {elapsed:.1f}s")

    result_file = os.path.join(OUTPUT_DIR, os.path.basename(img).replace(".png", ".txt"))
    if os.path.exists(result_file):
        with open(result_file) as f:
            print(f.read()[:1500])
