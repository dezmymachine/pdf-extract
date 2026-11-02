import logging
import os

from pdf_to_md import pdf_to_markdown

logging.basicConfig(level=logging.INFO, format="%(message)s")

result = pdf_to_markdown(
    "rag_paper.pdf",
    hf_token=os.environ["HF_TOKEN"],
    mode="gundam",
    max_retries=3,
    retry_delay=15,
)

with open("rag_paper.md", "w") as f:
    f.write(result)

print(f"\nSaved to rag_paper.md ({len(result)} chars)")
