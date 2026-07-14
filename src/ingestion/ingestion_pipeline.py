import sys
from pathlib import Path
from typing import Any, Optional
from dotenv import load_dotenv
    
load_dotenv(override=True)
    
from src.ingestion.docling_pdf_loader import DoclingPDFLoader
from src.ingestion.docling_doc_cleaner import DocumentCleaner
from src.ingestion.semantic_chunker import SemanticChunker
from src.ingestion.doc_embedder import HFDocEmbedder
from src.ingestion.vector_store import QdrantStore
import src.config.settings as config
from src.ingestion.docling_json_loader import DoclingJsonLoader

class IngestionPipeline:
    def __init__(self, loader_mode: str = "pdf"):
        """
        loader_mode:
            "pdf"  -- parse PDFs directly with Docling (slow, always fresh).
            "json" -- load pre-parsed DoclingDocument JSON (fast). Requires
                      parse_paper.py to have already been run on these PDFs.
        """
        print("=== INITIALIZING END-TO-END INGESTION PIPELINE ===")
        self.collection_name = config.QDRANT_COLLECTION_NAME
        self.loader_mode = loader_mode

        # Instantiate pipeline blocks sequentially
        self.loader = DoclingPDFLoader() if loader_mode == "pdf" else DoclingJsonLoader()
        self.cleaner = DocumentCleaner()
        self.chunker = SemanticChunker()
        self.doc_embedder = HFDocEmbedder()

        # Connect to our storage target (using in-memory mode for safe automated execution)
        self.vector_store = QdrantStore()

        print(f"[Loader] mode='{loader_mode}' -> using {type(self.loader).__name__}")

    def run(self,
            directory_path: Optional[str] = None,
            extract_images: bool = False,
            output_image_dir: Optional[str] = str(config.EXTRACTED_IMAGES_DIR),
            ) -> Any:
        """
        Executes the entire document preprocessing and database entry system lifecycle.
        """
        # directory_path defaults depend on loader_mode, so it's resolved here
        
        if directory_path is None:
            directory_path = str(
                config.PARSED_JSON_DIR if self.loader_mode == "json" else config.PDF_DIR
            )

        # Step 1: Document Loading
        step1_label = "Loading pre-parsed JSON" if self.loader_mode == "json" else "Parsing PDF layouts"
        
        print(f"\n--- STEP 1: {step1_label} from {directory_path} ---")

        raw_docs = self.loader.load_directory(
            directory_path=directory_path,
            extract_images=extract_images,
            output_image_dir=output_image_dir
        )

        if not raw_docs:
            print("[Abort] No documents could be extracted.")
            return None

        # Step 2: Context Purification / Cleaning
        print(f"\n--- STEP 2: Stripping systemic boilerplate from {len(raw_docs)} document structures ---")
        cleaned_docs = self.cleaner.clean(raw_docs)

        # Steps 3, 4, & 5: Structural Extraction, Semantic Chunking, & Strategy Metadata Enrichment
        print(f"\n--- STEPS 3-5: Building Hierarchical Structural Chunks & Dynamic Tag Placement ---")
        document_chunks = self.chunker.create_chunks(cleaned_docs)

        if not document_chunks:
            print("[Abort] Processing lifecycle resulted in 0 target semantic text chunks.")
            return None

        # Step 6: Context Embedding Vector Computations
        print(f"\n--- STEP 6: Running Vector Target Generation via {config.EMBEDDING_MODEL_NAME} ---")
        embedded_payloads = self.doc_embedder.generate_embeddings(document_chunks, batch_size=config.EMBEDDING_BATCH_SIZE)

        # Step 7: Database Setup and Batch Verification
        print(f"\n--- STEP 7: Transporting Array Payloads to Vector Database Cluster ---")
        self.vector_store.initialize_collection(collection_name=self.collection_name, vector_size=config.VECTOR_DIMENSION)
        self.vector_store.upload_documents(collection_name=self.collection_name, embedded_payloads=embedded_payloads)

        print(f"\n[Lifecycle Success] End-to-end processing pipeline completed flawlessly.")
        return self.vector_store


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the ingestion pipeline")
    
    parser.add_argument(
        "--loader", choices=["pdf", "json"], default="json",
        help="'pdf': parse PDFs directly (slow). 'json': load pre-parsed DoclingDocument JSON (fast).",
    )
    
    parser.add_argument("--dir", default=None, help="Override the input directory for this run")
    
    args = parser.parse_args()

    pipeline = IngestionPipeline(loader_mode=args.loader)
    active_vector_store = pipeline.run(directory_path=args.dir)
    
    if active_vector_store:
        print("\n==================================================")
        print("DATABASE INGESTION STATE VALIDATION")
        print("==================================================")
        
        # Fetch status properties from the collection to verify points have populated
        collection_info = active_vector_store.client.get_collection(collection_name=config.QDRANT_COLLECTION_NAME)
        
        print(f"Collection status:  {collection_info.status.upper()}")
        print(f"Total points stored: {collection_info.points_count}")
        
        # Let's perform a dummy fetch to look inside a real database payload node
        sample_scroll = active_vector_store.client.scroll(collection_name=config.QDRANT_COLLECTION_NAME, limit=1)
        
        if sample_scroll[0]:
            sample_point = sample_scroll[0][0]
            print(f"\nVerified Active Node ID: {sample_point.id}")
            print(f" -> Stored Metadata Paper: {sample_point.payload.get('paper_name')}")
            print(f" -> Stored Metadata Section: {sample_point.payload.get('parent_section')}")
            print(f" -> Keywords Captured:       {sample_point.payload.get('keywords')}")
            print(f" -> DB Text Payload Length:  {len(sample_point.payload.get('text'))} chars")
        print("==================================================")
        active_vector_store.client.close()