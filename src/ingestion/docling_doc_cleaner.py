import re
from typing import List
from bs4 import BeautifulSoup
from pathlib import Path

class DocumentCleaner:
    def __init__(self):
        # Expanded regex patterns to catch academic boilerplate, footnotes, emails, and copyright text
        self.noise_patterns = [
            re.compile(r"(?i)proceedings of the.*?conference"),
            re.compile(r"(?i)accepted as a poster.*"),
            re.compile(r"(?i)under review as a conference paper.*"),
            re.compile(r"(?i)preprint.*?under review.*"),
            re.compile(r"(?i)arxiv:\d{4}\.\d{4,5}v\d+"),
            re.compile(r"(?i)presented at.*?"),
            
            # --- For Front-Matter Noise ---
            re.compile(r"[\w\.-]+@[\w\.-]+\.\w+"),  # Email addresses
            re.compile(r"(?i)provided proper attribution is provided.*"), # Copyright boilerplate
            re.compile(r"(?i)permission to make digital or hard copies.*") # ACM/IEEE copyright boilerplate
        ]
        self.drop_labels = ["page_header", "page_footer", "page_no"]

    def clean(self, documents: List) -> List:
        cleaned_docs = []
        for doc in documents:
            clean_doc = self._clean_document(doc)
            
            # markdown_text = clean_doc.export_to_markdown()
            # clean_doc_path = Path("docs/clean-docs-v1")
            # clean_doc_path.mkdir(parents=True, exist_ok=True)
            # md_path = clean_doc_path / f"{Path(doc.name)}.md"
            # md_path.write_text(markdown_text, encoding="utf-8")
            
            cleaned_docs.append(clean_doc)
        return cleaned_docs

    def _clean_document(self, doc):
        if not hasattr(doc, 'texts'):
            return doc

        for text_item in doc.texts:
            label_str = str(getattr(text_item, 'label', '')).lower()
            
            # 1. Clear text for explicitly tagged running headers/footers
            if any(drop_lbl in label_str for drop_lbl in self.drop_labels):
                text_item.text = ""
                continue
                
            # 2. BeautifulSoup: Remove any stray HTML/XML artifacts
            if "<" in text_item.text and ">" in text_item.text:
                text_item.text = BeautifulSoup(text_item.text, "html.parser").get_text(separator=" ")
            
            # 3. Regex: Remove academic footnotes and boilerplate
            original_text = text_item.text
            for pattern in self.noise_patterns:
                original_text = pattern.sub("", original_text)
            
            # 4. Clean up excess whitespace left by deletions
            text_item.text = re.sub(r'\s+', ' ', original_text).strip()
            
        return doc

if __name__ == "__main__":
    import os
    import sys
    from pathlib import Path
    
    project_root = Path(__file__).resolve().parents[2]
    sys.path.append(str(project_root))
    
    try:
        from src.ingestion.docling_pdf_loader import DoclingPDFLoader
    except ImportError:
        print("Could not import DoclingPDFLoader. Ensure it exists in src/loaders/")
        sys.exit(1)

    test_dir_path = project_root / "docs/pdfs"
    
    loader = DoclingPDFLoader()
    raw_documents = loader.load_directory(str(test_dir_path), extract_images=False)
    
    if not raw_documents:
        print("No documents found to clean.")
        sys.exit(0)
        
    cleaner = DocumentCleaner()
    cleaned_documents = cleaner.clean(raw_documents)
    
    print("\nCleaning process completed successfully.")
    
    for idx, doc in enumerate(cleaned_documents, start=1):
        print(f"\n==================================================")
        print(f"Cleaned Document {idx}")
        print(f"==================================================")
        
        markdown_preview = doc.export_to_markdown()
        print("\n--- CLEANED MARKDOWN PREVIEW ---")
        print(markdown_preview[:1000])
        print("--------------------------------\n")