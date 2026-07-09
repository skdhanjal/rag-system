import os
import re
# Disable Hugging Face symlinks to prevent WinError 1314 on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

from typing import List, Optional
from pathlib import Path
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption

class BaseLoader:
    def load(self, path: str) -> List:
        raise NotImplementedError()

class DoclingPDFLoader(BaseLoader):
    def __init__(self):
        pipeline_options = PdfPipelineOptions()
        pipeline_options.generate_picture_images = True
        pipeline_options.do_ocr = False  # Disable OCR if the PDF isn't a scanned image
        pipeline_options.table_structure_options.mode = TableFormerMode.FAST  # Drops heavy structural recognition models
        
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                    backend=PyPdfiumDocumentBackend
                )
            }
        )

    def load(self, path: str, extract_images: bool = False, output_image_dir: Optional[str] = None) -> List:
        result = self.converter.convert(path)
        doc = result.document
        
        if extract_images and output_image_dir:
            img_dir = Path(output_image_dir)
            img_dir.mkdir(parents=True, exist_ok=True)
            
            for pic_idx, picture in enumerate(doc.pictures):
                if picture.image:
                    caption_str = ""
                    if hasattr(picture, "caption_text"):
                        caption_str = picture.caption_text(doc=doc).strip()
                    
                    if caption_str:
                        filename = f"{self._sanitize_filename(caption_str)}.png"
                    else:
                        filename = f"{Path(path).stem}_image_{pic_idx + 1}.png"
                    
                    picture.image.pil_image.save(img_dir / filename)
                    
        return [doc]

    def load_directory(self, directory_path: str, extract_images: bool = False, output_image_dir: Optional[str] = None) -> List:
        all_documents = []
        dir_path = Path(directory_path)
        
        if not dir_path.exists() or not dir_path.is_dir():
            print(f"Directory not found: {directory_path}")
            return all_documents

        for pdf_file in dir_path.glob("*.pdf"):
            print(f"Parsing: {pdf_file.name}...")
            try:
                docs = self.load(
                    path=str(pdf_file), 
                    extract_images=extract_images, 
                    output_image_dir=output_image_dir
                )
                all_documents.extend(docs)
            except Exception as e:
                print(f"Failed to parse {pdf_file.name}: {e}")
                
        return all_documents

    def _sanitize_filename(self, text: str, max_length: int = 50) -> str:
        clean_text = re.sub(r'[\\/*?:"<>|.\']', "", text)
        clean_text = re.sub(r'[\s-]+', "_", clean_text)
        clean_text = clean_text.strip("_")
        return clean_text[:max_length].rstrip("_")

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    test_dir_path = project_root / "docs/pdfs"
    target_image_dir = project_root / "docs/extracted_images"
    
    loader = DoclingPDFLoader()
    print(f"Scanning directory: {test_dir_path}\n")
    
    # Execution with context-aware image extraction turned on
    documents = loader.load_directory(
        directory_path=str(test_dir_path), 
        extract_images=False, 
        output_image_dir=str(target_image_dir)
    )
    
    print(f"\nProcessing complete. Loaded {len(documents)} document(s).")
    for idx, doc in enumerate(documents, start=1):
        print(f"\n--- DOCUMENT {idx} MARKDOWN PREVIEW ---")
        print(doc.export_to_markdown()[:1000])
        print("---------------------------------------")