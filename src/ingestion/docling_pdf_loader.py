import glob
import os
import re

from annotated_types import doc

# Disable Hugging Face symlinks to prevent WinError 1314 on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

from typing import Any, Dict, List, Optional
from pathlib import Path
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption

class DoclingPDFLoader():
    def __init__(self):
        pipeline_options = PdfPipelineOptions()
        
        pipeline_options.do_table_structure = True       # structured table extraction
        pipeline_options.do_formula_enrichment = True     # LaTeX extraction for equations
        pipeline_options.generate_picture_images = True   # save figures as standalone images
        pipeline_options.images_scale = 1.0  
        
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                    backend=PyPdfiumDocumentBackend
                )
            }
        )
        
    def save_docling_json(self, doc, out_path: Path) -> bool:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)  # create the PARENT dir, never out_path itself

        try:
            doc.save_as_json(out_path)
            return True
        except PermissionError as e:
            print(f"[Loader] Could not write {out_path} (locked or permission issue): {e}")
        return False    

    def load(self, path: str, extract_images: bool = False, output_image_dir: Optional[str] = None) -> List:
        result = self.converter.convert(path)
        doc = result.document
        doc.name = result.input.file.name
        
        doc_json_path = Path(path).parent / "parsed_json" / f"{Path(path).stem}.docling.json"
        
        print(f"[Loader] Saving parsed document to: {doc_json_path}")
        self.save_docling_json(doc, doc_json_path)
        
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
    
    
    def _load_markdown_directory(self, md_dir_path: str) -> List[Dict[str, Any]]:
        """Utility to rapidly load pre-parsed Markdown documents."""
        md_files = glob.glob(os.path.join(md_dir_path, "*.md"))
        
        if not md_files:
            print(f"[Abort] No .md files found in {md_dir_path}")
            return []
            
        loaded_docs = []
        for filepath in md_files:
            with open(filepath, 'r', encoding='utf-8') as f:
                loaded_docs.append({
                    "content": f.read(),
                    "metadata": {"source": os.path.basename(filepath)}
                })
                
        return loaded_docs

    def _sanitize_filename(self, text: str, max_length: int = 50) -> str:
        clean_text = re.sub(r'[\\/*?:"<>|.\']', "", text)
        clean_text = re.sub(r'[\s-]+', "_", clean_text)
        clean_text = clean_text.strip("_")
        return clean_text[:max_length].rstrip("_")

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    test_dir_path = project_root / "docs/pdfs"
    out_dir = project_root / "docs/test/markdown_output"
    target_image_dir = project_root / "docs/test/extracted_images"
    
    loader = DoclingPDFLoader()
    print(f"Scanning directory: {test_dir_path}\n")
    
    # Execution with context-aware image extraction turned on
    documents = loader.load_directory(
        directory_path=str(test_dir_path), 
        extract_images=False, 
        output_image_dir=str(target_image_dir),
    )
    
    print(f"\nProcessing complete. Loaded {len(documents)} document(s).")
    for idx, doc in enumerate(documents, start=1):
        print(f"\n--- DOCUMENT {idx} ({doc.name}) MARKDOWN PREVIEW ---")
        print(doc.export_to_markdown()[:1000])
        print("---------------------------------------")