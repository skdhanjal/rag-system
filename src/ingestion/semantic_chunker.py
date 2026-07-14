from pathlib import Path
from typing import List
from transformers import AutoTokenizer
from docling.chunking import HybridChunker
from typing import List

import src.config.settings as config
from src.schema.document import Document, DocumentMetadata

class SemanticChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 80):
        print(f"=== INITIALIZING DOCLING HYBRID CHUNKER ===")
        print(f"Target Tokenizer: {config.EMBEDDING_MODEL_NAME}")
        print(f"Max Tokens/Chunk: {config.TARGET_MAX_TOKENS}")
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL_NAME)
        except Exception as e:
            print(f"[Warning] Could not load huggingface tokenizer '{config.EMBEDDING_MODEL_NAME}': {e}. Falling back to default tokenization.")
            self.tokenizer = None

        self.chunker = HybridChunker(
            tokenizer=self.tokenizer,
            max_tokens=config.TARGET_MAX_TOKENS,
            merge_peers=True 
        )

    def create_chunks(self, cleaned_docs: list) -> List[Document]:
        """
        Takes a list of native DoclingDocument objects, applies token-aware 
        hierarchical chunking, and maps them precisely to DocumentMetadata schema.
        """
        valid_chunks = []
        
        for dl_doc in cleaned_docs:
            doc_name = str(getattr(dl_doc, "name", "Unknown"))
            paper_name = doc_name.replace(".pdf", "") if doc_name else "Unknown"
            
            if isinstance(doc_name, dict):
                doc_name = doc_name.get("filename", doc_name.get("name", "Unknown"))
                
            print(f"Applying HybridChunker to: {doc_name}...")
            
            chunk_iter = self.chunker.chunk(dl_doc=dl_doc)
            
            for i, chunk in enumerate(chunk_iter):
                # Contextualize prepends the section headings to the chunk text
                enriched_text = self.chunker.contextualize(chunk=chunk)
                
                # Accurately count the final tokens of the enriched text
                if self.tokenizer:
                    token_count = len(self.tokenizer.encode(enriched_text, add_special_tokens=False))
                else:
                    token_count = len(enriched_text.split())
                
                # 1. Headings & Parent Section
                headings_list = []
                
                parent_section = "Unknown"
                if hasattr(chunk, "meta") and chunk.meta and hasattr(chunk.meta, "headings") and chunk.meta.headings:
                    headings_list = chunk.meta.headings
                    parent_section = " > ".join(headings_list) if headings_list else "Unknown"

                # 2. Page Tracking (First Page & All Spanned Pages)
                first_page_num = 0
                unique_pages = set()
                
                if hasattr(chunk, "meta") and chunk.meta and hasattr(chunk.meta, "doc_items"):
                    for item in chunk.meta.doc_items:
                        if hasattr(item, "prov") and item.prov:
                            for prov in item.prov:
                                if hasattr(prov, "page_no") and prov.page_no:
                                    unique_pages.add(prov.page_no)
                                    if first_page_num == 0:
                                        first_page_num = prov.page_no

                # 3. Construct our standardized metadata object exactly to schema
                metadata = DocumentMetadata(
                    source=str(doc_name),
                    paper_name=str(paper_name),
                    chunk_number=i + 1,
                    parent_section=parent_section,
                    page_number=first_page_num,
                    document_type="Research Paper",
                    keywords=[],  # Leave empty unless populated by a later enrichment step
                    all_headings=headings_list,
                    page_numbers=sorted(list(unique_pages)),
                    token_count=token_count
                )
                
                valid_chunks.append(Document(content=enriched_text, metadata=metadata))
                
        print(f"Generated {len(valid_chunks)} token-safe hybrid chunks across all documents.")
        return valid_chunks
    
    
    # =====================================================================
# INDEPENDENT TEST BLOCK
# Run this file directly via `python src/ingestion/semantic_chunker.py`
# =====================================================================
if __name__ == "__main__":
    
    import os
    from docling.document_converter import DocumentConverter

    print("\n--- Starting Semantic Chunker Unit Test ---")
    
    # 1. Define a test PDF. Replace this with a path to a real paper you have locally!
    test_pdf_path = "docs/graphs/bert.pdf" 
    
    if not os.path.exists(test_pdf_path):
        print(f"[Error] Test file not found at '{test_pdf_path}'.")
        print("Please place a test PDF in the root directory or update the 'test_pdf_path' variable.")
    else:
        print(f"1. Converting '{test_pdf_path}' into native Docling structural format...")
        converter = DocumentConverter()
        conversion_result = converter.convert(test_pdf_path)
        docling_doc = conversion_result.document
        
        # Inject a dummy file name property so our extraction logic catches it
        docling_doc.name = test_pdf_path
        
        print("\n2. Initializing Semantic Chunker...")
        chunker = SemanticChunker()
        
        print("\n3. Processing document into token-safe chunks...")
        test_chunks = chunker.create_chunks([docling_doc])
        
        print(f"\n=== CHUNKING RESULTS: {len(test_chunks)} TOTAL CHUNKS ===")
        
        # 4. Print detailed metadata for the first 5 chunks to verify schema population
        for i, chunk in enumerate(test_chunks[:5]):
            print("\n" + "="*60)
            print(f"Chunk Number  : {chunk.metadata.chunk_number}")
            print(f"Paper Name    : {chunk.metadata.paper_name}")
            print(f"Parent Section: {chunk.metadata.parent_section}")
            print(f"All Headings  : {chunk.metadata.all_headings}")
            print(f"Pages Spanned : {chunk.metadata.page_numbers}")
            print(f"Starting Page : {chunk.metadata.page_number}")
            print(f"Token Count   : {chunk.metadata.token_count}")
            print(f"Content Preview:\n{chunk.content.replace('\n', ' ')}...")
            
        if len(test_chunks) > 5:
            print("\n" + "="*60)
            print(f"... and {len(test_chunks) - 5} more chunks mapped perfectly.")