import sys
import uuid
from pathlib import Path
from typing import List, Tuple, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Centralized path routing
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.schema.document import Document
import src.config.settings as config


class QdrantStore():
    def __init__(self):
        # Explicitly branch initialization logic depending on whether a remote URL or local path is supplied
        if config.QDRANT_URL:
            print(f"Connecting to remote Qdrant cluster at {config.QDRANT_URL}...")
            self.client = QdrantClient(
                url=config.QDRANT_URL, 
                api_key=config.QDRANT_API_KEY
            )
        else:
            print(f"Connecting to persistent local Qdrant storage at {config.QDRANT_LOCATION}...")
            # Use 'path' parameter for local disk storage, NOT 'url' or 'location'
            self.client = QdrantClient(path=config.QDRANT_LOCATION)

    def initialize_collection(self, collection_name: str, vector_size: int = config.VECTOR_DIMENSION) -> None:
        """
        Creates a collection if it does not already exist. 
        Uses Cosine distance to perfectly align with normalized BGE-M3 embeddings.
        """
        # Check if collection already exists to prevent overwriting
        collections = self.client.get_collections().collections
        exists = any(c.name == collection_name for c in collections)
        
        if exists:
            print(f"Collection '{collection_name}' already exists. Skipping recreation.")
            return

        print(f"Creating collection '{collection_name}' with vector size {vector_size} (Metric: Cosine)...")
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        print(f"[Success] Collection '{collection_name}' initialized.")

    def upload_documents(self, collection_name: str, embedded_payloads: List[Tuple[Document, List[float]]]) -> None:
        """
        Transforms packed Document objects into production-ready Qdrant PointStruct objects 
        and uploads them in an optimized batch.
        """
        if not embedded_payloads:
            print("No payloads provided for storage upload.")
            return

        print(f"Preparing storage layout for {len(embedded_payloads)} points inside '{collection_name}'...")
        points = []
        
        for doc, vector in embedded_payloads:
            meta = doc.metadata
            
            # Formulate the payload structure mapping directly from our central schema
            payload = {
                "text": doc.content,
                "paper_name": meta.paper_name,
                "chunk_number": meta.chunk_number,
                "parent_section": meta.parent_section,
                "page_number": meta.page_number,
                "document_type": meta.document_type,
                "keywords": meta.keywords,
                "all_headings": meta.all_headings,
                "page_numbers": meta.page_numbers,
                "token_count": meta.token_count
            }
            
            # Create a deterministic UUID using the paper name and chunk index 
            # This guarantees that rerunning the ingestion will update existing items instead of duplicating them
            deterministic_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{meta.paper_name}_{meta.chunk_number}"))
            
            points.append(
                PointStruct(
                    id=deterministic_id,
                    vector=vector,
                    payload=payload
                )
            )
            
        print(f"Executing batch upsert into Qdrant collection '{collection_name}'...")
        self.client.upsert(
            collection_name=collection_name,
            wait=True,
            points=points
        )
        print(f"[Success] Upsert complete. {len(points)} documents safely stored in vector database.")