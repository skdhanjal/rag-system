from pathlib import Path
import torch
from typing import List, Tuple
import sys

# Centralized path routing for local run confirmation
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.schema.document import Document
import src.config.settings as config

# Choose your backend library (SentenceTransformers is highly optimized for batching)
from sentence_transformers import SentenceTransformer

class HFDocEmbedder:
    def __init__(self):
        # 1. Dynamically select the fastest available hardware accelerator
        if torch.cuda.is_available():
            self.device = "cuda"
            self.torch_dtype = torch.float16  # Enable FP16 half-precision on Nvidia GPUs
            print("🚀 Embedding Engine: CUDA GPU detected. Enabling FP16 acceleration.")
        elif torch.backends.mps.is_available():
            self.device = "mps"
            self.torch_dtype = torch.float32  # CoreML/MPS handles FP32 better on Apple Silicon
            print("🍏 Embedding Engine: Apple Silicon MPS detected.")
        else:
            self.device = "cpu"
            self.torch_dtype = torch.float32
            print("⚠️ [Warning] Embedding Engine: No GPU detected! Running on CPU will be extremely slow.")

        print(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}...")
        
        # 2. Initialize model directly on target device
        self.model = SentenceTransformer(
            config.EMBEDDING_MODEL_NAME, 
            device=self.device,
            model_kwargs={"torch_dtype": self.torch_dtype} if self.device == "cuda" else {}
        )

    def generate_embeddings(self, documents: List[Document], batch_size: int = config.EMBEDDING_BATCH_SIZE) -> List[Tuple[Document, List[float]]]:
        """
        Generates dense vector embeddings using high-throughput mini-batching.
        """
        if not documents:
            return []

        # 3. Extract strings into a contiguous block of memory for batch processing
        texts_to_embed = [doc.content for doc in documents]
        
        print(f"Encoding {len(texts_to_embed)} chunks via vector engine...")
        
        # 4. Compute embeddings using vector matrix multiplication (the speed engine)
        # Optimal batch_size is usually 16 or 32 depending on your GPU VRAM
        dense_embeddings = self.model.encode(
            inputs=texts_to_embed,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True, # BGE-M3 performs best when normalized to unit vectors
            convert_to_numpy=True
        )

        # 5. Pack chunks back up cleanly with their parent Document structure
        embedded_payloads = []
        for doc, vector in zip(documents, dense_embeddings):
            embedded_payloads.append((doc, vector.tolist()))

        print(f"[Success] Multi-batch embedding vector synthesis finalized.")
        return embedded_payloads
    
# =====================================================================
# INDEPENDENT TEST BLOCK
# Run this file directly via `python src/embeddings/embedding_generator.py`
# =====================================================================
if __name__ == "__main__":
    import sys
    from src.schema.document import Document, DocumentMetadata

    print("\n--- Starting Embedding Generator Unit Test ---")
    
    # 1. Create a dummy metadata structure tracking mock attributes
    mock_metadata = DocumentMetadata(
        paper_name="unit_test_paper.pdf",
        chunk_number=1,
        parent_section="Test Section",
        page_number=1,
        document_type="Research Paper",
        keywords=["test", "embedding"],
        all_headings=["Introduction", "Test Section"],
        page_numbers=[1],
        token_count=15
    )

    # 2. Build a micro-batch of mock documents to test the parallel matrix encoder
    sample_texts = [
        "Attention mechanisms have revolutionized the way sequence-to-sequence deep learning models process long-range context dependencies.",
        "Bidirectional encoder representations from transformers rely heavily on masked language modeling pre-training objectives.",
        "Retrieval-augmented generation pairs parametric internal memory with non-parametric external vector search index pipelines."
    ]
    
    mock_documents = [
        Document(content=text, metadata=mock_metadata) for text in sample_texts
    ]
    
    print(f"1. Prepared a mini-batch of {len(mock_documents)} documents for encoding.")

    # 3. Initialize the generator to check device mapping and model execution
    print("\n2. Initializing Embedding Generator Class...")
    try:
        generator = HFDocEmbedder()
    except Exception as e:
        print(f"[Fatal Error] Failed to initialize the embedding model framework: {e}")
        sys.exit(1)

    # 4. Trigger high-throughput batch inference pass
    print("\n3. Dispatching vector generation pass...")
    try:
        payloads = generator.generate_embeddings(mock_documents)
        
        print(f"\n=== EMBEDDING RESULTS: {len(payloads)} VECTORS SYNTHESIZED ===")
        
        # 5. Extract diagnostics from the output payloads
        for idx, (doc, vector) in enumerate(payloads):
            print("\n" + "-"*50)
            print(f"Document Index  : {idx + 1}")
            print(f"Vector Dimensions: {len(vector)} elements (Expected: 1024 for BGE-M3)")
            print(f"Data Structure  : {type(vector)} containing float entries")
            print(f"Vector Preview  : {vector[:5]} ... [truncated]")
            print(f"Content Tied To : \"{doc.content[:60]}...\"")
            
        print("\n" + "="*60)
        print("🎉 [Success] Embedding engine is fully functional, isolated, and highly optimized!")
        
    except Exception as e:
        print(f"[Runtime Error] Batch execution crashed during forward pass: {e}")    