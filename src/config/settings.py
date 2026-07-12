import os
from pathlib import Path

# --- Directory Path Configurations ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "docs"
PDF_DIR = DATA_DIR / "pdfs"
PARSED_JSON_DIR = DATA_DIR / "parsed_json"
EXTRACTED_IMAGES_DIR = DATA_DIR / "extracted_images"

# HTML Output location for our vector space scatter plot
VISUALIZATION_OUTPUT_PATH = DATA_DIR / "embedding_space.html"

# Ensure runtime directories exist immediately
PDF_DIR.mkdir(parents=True, exist_ok=True)
EXTRACTED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# --- Semantic Chunker Configurations ---
TARGET_MAX_TOKENS = 500
MIN_TOKEN_THRESHOLD = 20

# --- Embedding Model Configurations ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-base-en-v1.5", "BAAI/bge-m3"]
EMBEDDING_BATCH_SIZE = 16
EMBEDDING_DEVICE = "cpu"  # Change to "cuda" or "mps" if hardware acceleration is available
VECTOR_DIMENSION = 384  # Must match the embedding model's output dimension

# chunker settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# --- Visualization Configurations ---
# Canvas dimension options: "2D" or "3D"
VISUALIZATION_DIMENSION = "2D"

# Higher-level grouping strategy to prevent legend cluttering.
# Options: "paper_name" (Recommended) or "parent_section"
VISUALIZATION_GROUP_BY = "paper_name"

# --- Qdrant Vector Database Configurations ---
QDRANT_COLLECTION_NAME = "llm_research_collection"

# PERSISTENCE CHANGE: Switched from ":memory:" to a dedicated local directory path
QDRANT_LOCATION = os.getenv("QDRANT_LOCATION", str(PROJECT_ROOT / "local_qdrant")) 
QDRANT_URL = os.getenv("QDRANT_URL", None)
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)

# --- Retrieval & Reranking Configurations ---
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2" # ["BAAI/bge-reranker-v2-m3", "cross-encoder/ms-marco-MiniLM-L-6-v3"]
RETRIEVAL_TOP_K = 20
RERANK_TOP_K = 10

# --- LLM Interface Configurations ---
LLM_MODEL_NAME = "gemini/gemini-3.1-flash-lite"
LLM_AS_JUDGE_MODEL_NAME = "gpt-4o-mini"

LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 2048

# --- Router / Intent Classifier Settings ---
LLM_CLASSIFIER_MODEL_NAME = "gpt-4o-mini"
LLM_CLASSIFIER_TEMPERATURE = 0.0
LLM_CLASSIFIER_MAX_TOKENS = 5