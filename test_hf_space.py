import logging
import os

from pdf_to_md import pdf_to_markdown

logging.basicConfig(level=logging.INFO, format="%(message)s")

text = pdf_to_markdown(
    "rag_paper.pdf",
    hf_token=os.environ["HF_TOKEN"],
    mode="gundam",      # faster for testing
    max_retries=3,
    retry_delay=15,
    page_timeout=180,
)

print(text[:3000])
print(f"\n\n--- Total chars: {len(text)} ---")
