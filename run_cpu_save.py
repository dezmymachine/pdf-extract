import os, sys, tempfile, time, fitz

import torch
torch.Tensor.cuda = lambda self, *a, **kw: self
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from transformers import AutoModel, AutoTokenizer

PDF_PATH = "rag_paper.pdf"
OUTPUT_FILE = "rag_paper.md"

def page_to_img(pdf_path, page_num, dpi=200):
    doc = fitz.open(pdf_path)
    tmp = tempfile.mkdtemp(prefix="pdf_ocr_")
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    out = os.path.join(tmp, f"p{page_num}.png")
    doc[page_num].get_pixmap(matrix=mat).save(out)
    doc.close()
    return out, tmp

print("Loading model (CPU patched)...")
model_name = "baidu/Unlimited-OCR"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name, trust_remote_code=True, use_safetensors=True,
    dtype=torch.bfloat16, device_map="cpu",
).eval()

pages_to_extract = [0, 1, 10]

all_text = []
for pnum in pages_to_extract:
    print(f"\n=== Page {pnum+1} ===")
    img, tmpdir = page_to_img(PDF_PATH, pnum, 200)
    t0 = time.time()
    model.infer_multi(
        tokenizer, prompt="<image>Multi page parsing.",
        image_files=[img], output_path=tmpdir, image_size=1024,
        max_length=32768, no_repeat_ngram_size=35, ngram_window=1024,
        save_results=False,
    )
    # Find output file
    out_file = os.path.join(tmpdir, f"p{pnum}.txt")
    if os.path.exists(out_file):
        with open(out_file) as f:
            page_text = f.read()
    else:
        found = None
        for fname in os.listdir(tmpdir):
            if fname.endswith((".txt", ".md")):
                found = os.path.join(tmpdir, fname)
                break
        if found:
            with open(found) as f:
                page_text = f.read()
        else:
            page_text = "(no output file found)"
    elapsed = time.time() - t0
    print(f"  Time: {elapsed:.0f}s, Chars: {len(page_text)}")
    all_text.append(f"## Page {pnum+1}\n\n{page_text}")

full = "\n\n---\n\n".join(all_text)
with open(OUTPUT_FILE, "w") as f:
    f.write(full)
print(f"\nSaved to {OUTPUT_FILE} ({len(full)} chars)")
