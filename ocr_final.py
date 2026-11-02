import os
import tempfile
import time

import fitz

import torch
torch.Tensor.cuda = lambda self, *a, **kw: self
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from transformers import AutoModel, AutoTokenizer

PDF_PATH = "rag_paper.pdf"
model_name = "baidu/Unlimited-OCR"

def page_to_img(pdf_path, page_num, dpi=200):
    doc = fitz.open(pdf_path)
    tmp = tempfile.mkdtemp(prefix="pdf_ocr_")
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    out = os.path.join(tmp, f"p{page_num}.png")
    doc[page_num].get_pixmap(matrix=mat).save(out)
    doc.close()
    return out

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True, use_safetensors=True, dtype=torch.bfloat16, device_map="cpu").eval()
print("Ready")

for pnum in [0, 1, 20]:
    print(f"\n=== Page {pnum+1} ===")
    img = page_to_img(PDF_PATH, pnum, 200)
    t0 = time.time()
    model.infer_multi(tokenizer, prompt="<image>Multi page parsing.", image_files=[img], output_path="ocr_pages", image_size=1024, max_length=32768, no_repeat_ngram_size=35, ngram_window=1024, save_results=False)
    print(f"({time.time()-t0:.0f}s)")
    rfile = f"ocr_pages/p{pnum}.txt"
    if os.path.exists(rfile):
        with open(rfile) as f:
            print(f.read()[:2000])
