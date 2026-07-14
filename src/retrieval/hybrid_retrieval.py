import sys
import torch
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer, CrossEncoder
from litellm.exceptions import Timeout as LiteLLMTimeout

# Load environment variables at the top
load_dotenv(override=True)

from src.helpers.llm_helper import call_configured_llm
from src.helpers.conversation_memory import retain_recent_completed_turns
from src.helpers.guardrails import (
    NO_EVIDENCE_MESSAGE,
    has_enough_retrieval_evidence,
    sanitize_retrieved_context,
    validate_llm_answer,
    validate_user_query,
)
from src.prompts.prompts import get_academic_system_instruction
import src.config.settings as config
from src.retrieval.query_router import QueryRouter

class HybridRetriever:
    def __init__(self):
        print("=== INITIALIZING CONFIG-DRIVEN RETRIEVAL ENGINE ===")
        
        # 1. Initialize Qdrant Client 
        if getattr(config, "QDRANT_URL", None):
            self.client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
        else:
            self.client = QdrantClient(path=getattr(config, "QDRANT_LOCATION", "./qdrant_data"))
            
        self.router = QueryRouter()
        self.device = config.EMBEDDING_DEVICE
        self.torch_dtype = torch.float32

        # 3. Load Configured Models
        print(f"Loading Dense Embedding Model: {config.EMBEDDING_MODEL_NAME}...")
        
        self.embedding_model = SentenceTransformer(
            config.EMBEDDING_MODEL_NAME, 
            device=self.device,
            model_kwargs={"torch_dtype": self.torch_dtype} if self.device == "cuda" else {}
        )
        
        print(f"Loading Cross-Encoder Reranker: {config.RERANKER_MODEL_NAME}...")
        
        self.reranker_model = CrossEncoder(
            config.RERANKER_MODEL_NAME, 
            device=self.device,
            activation_fn=torch.nn.Identity()
        )
        print("Hybrid Retrieval Initialized.")

    def _get_neighbor_chunk(self, collection_name: str, paper_name: str, target_chunk_num: int) -> str:
        """Queries Qdrant using root payload field conditions to find sequential neighbors."""
        if target_chunk_num <= 0:
            return ""
            
        results, _ = self.client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(key="paper_name", match=models.MatchValue(value=paper_name)),
                    models.FieldCondition(key="chunk_number", match=models.MatchValue(value=target_chunk_num))
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False
        )
        return results[0].payload.get("text", "") if results else ""

    def _fetch_candidate_hits(self, query: str, collection_name: str, top_k: int) -> List[Any]:
        """Phase 1: Finding candidate hits using hybrid dense vector and keyword matches."""
        query_vector = self.embedding_model.encode(query, normalize_embeddings=True).tolist()
        
        dense_response = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True
        )
        
        dense_hits = dense_response.points
        
        lexical_hits, _ = self.client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="text", match=models.MatchText(text=query))]
            ),
            limit=top_k,
            with_payload=True,
            with_vectors=False
        )
        
        seen_ids = set()
        combined_candidates = []
        for hit in dense_hits + lexical_hits:
            if hit.id not in seen_ids:
                seen_ids.add(hit.id)
                combined_candidates.append(hit)
                
        return combined_candidates

    def _rerank_candidates(self, query: str, candidates: List[Any], rerank_top_k: int) -> List[Tuple[Any, float]]:
        """Phase 2: Cross-Encoder Re-ranking of candidate blocks."""
        if not candidates:
            return []
            
        cross_inp = [[query, hit.payload.get("text", "")] for hit in candidates]
        
        scores = self.reranker_model.predict(cross_inp)
        
        ranked_candidates = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return ranked_candidates[:rerank_top_k]

    def _expand_context(self, ranked_top: List[Tuple[Any, float]], collection_name: str) -> List[Dict[str, Any]]:
        """Phase 3: Neighbor Context Expansion for the top-ranked segments."""
        final_context_blocks = []
        for hit, score in ranked_top:
            payload = hit.payload
            paper_name = payload.get("paper_name", "Unknown")
            source = payload.get("source", "Unknown")
            chunk_num = payload.get("chunk_number", 0)
            parent_section = payload.get("parent_section", "Unknown")
            page_number = payload.get("page_number", 0)
            
            prev_text = self._get_neighbor_chunk(collection_name, paper_name, chunk_num - 1)
            current_text = payload.get("text", "")
            next_text = self._get_neighbor_chunk(collection_name, paper_name, chunk_num + 1)
            
            unified_content = "\n".join(filter(None, [prev_text, current_text, next_text]))
            
            final_context_blocks.append({
                "content": unified_content,
                "score": float(score),
                "metadata": {
                    "source": source,
                    "paper_name": paper_name,
                    "parent_section": parent_section,
                    "page_number": page_number,
                    "chunk_number": chunk_num,  
                }
            })
            
        return final_context_blocks

    def retrieve_context(self, query: str, collection_name: str = config.QDRANT_COLLECTION_NAME, top_k: int = config.RETRIEVAL_TOP_K) -> List[Dict[str, Any]]:
        """
        Executes Strategy Document Workflow:
        Finding Hits -> Re-ranking -> Context Expansion
        """
        candidates = self._fetch_candidate_hits(query, collection_name, top_k)
        
        print(f"Found {len(candidates)} candidate hits for query: '{query}' in collection '{collection_name}'.")
        if not candidates:
            return []
        
        print(f"Re-ranking top {config.RERANK_TOP_K} candidates using Cross-Encoder model...")
        top_ranked = self._rerank_candidates(query, candidates, config.RERANK_TOP_K)
        
        return self._expand_context(top_ranked, collection_name)

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """Call the configured LLM and handle generation errors."""
        try:
            print(f"Invoking LLM model '{config.LLM_MODEL_NAME}' for query response generation...")
            answer = call_configured_llm(
                model_name=config.LLM_MODEL_NAME,
                messages=messages,
                temperature=config.LLM_TEMPERATURE,
            )
            print(f"Received response from '{config.LLM_MODEL_NAME}' ({len(answer)} characters).")
            return answer
        except LiteLLMTimeout:
            print(f"[LLM Timeout] '{config.LLM_MODEL_NAME}' did not respond in time.")
            if config.LLM_PROVIDER == "ollama":
                return (
                    f"The local Ollama model did not respond within {config.LLM_TIMEOUT_SECONDS} seconds. "
                    "Try again with a smaller prompt, check that Ollama is running, or switch to Groq."
                )
            return (
                f"The configured LLM did not respond within {config.LLM_TIMEOUT_SECONDS} seconds. "
                "Please try again."
            )
        except Exception as error:
            print(f"[LLM Core Error] Generation layer failure: {error}")
            return (
                "Context was retrieved, but the configured LLM could not generate an answer. "
                "Check the provider configuration and try again."
            )

    def answer_query(self, query: str, chat_history: List[Dict[str, str]]) -> Tuple[str, List[Dict[str, Any]]]:
        """The principal interface consumed by your application layer."""
        chat_history = retain_recent_completed_turns(chat_history)

        is_valid_query, guardrail_message = validate_user_query(query)
        if not is_valid_query:
            return guardrail_message, []

        # Resolve routing and direct response in a single check/call
        requires_retrieval, direct_answer = self.router.resolve_query_intent(query, chat_history)
        
        if not requires_retrieval:
            return direct_answer, []
            
        contexts = self.retrieve_context(query)
        
        print(f"Retrieved {len(contexts)} context blocks for query: '{query}'.")
        
        if not has_enough_retrieval_evidence(contexts):
            return NO_EVIDENCE_MESSAGE, []

        context_str = "\n\n---\n\n".join(
            [f"📄 Source: {ctx['metadata'].get('source', 'Unknown')}\n📑 Section: {ctx['metadata'].get('parent_section', 'Unknown')}\n📖 Content:\n{ctx['content']}" for ctx in contexts]
        )

        context_str = sanitize_retrieved_context(context_str)
        system_instruction = get_academic_system_instruction(context_str)

        messages = [{"role": "system", "content": system_instruction}]
        if chat_history:
            messages.extend(chat_history)
        messages.append({"role": "user", "content": query})

        answer = self._call_llm(messages)
        is_valid_answer, guardrail_message = validate_llm_answer(answer)
        if not is_valid_answer:
            return guardrail_message, contexts

        return answer, contexts

if __name__ == "__main__":
    print("\n--- Starting Clean Configured Retriever Integration Test ---")
    retriever = HybridRetriever()
    
    test_query = "What is the primary training objective of BERT?"
    mock_history = [
        {"role": "user", "content": "Let's discuss language representation models."},
        {"role": "assistant", "content": "Sure, I have access to your database collection containing several transformer architecture papers."}
    ]
    
    try:
        print(f"\nDispatching query via settings logic: '{test_query}'...")
        answer, sources = retriever.answer_query(query=test_query, chat_history=mock_history)
        
        print("\n" + "="*60)
        print("🤖 CHAT INTERFACE ANSWER RESPONSE:")
        print("="*60)
        print(answer)
        
        print("\n" + "="*60)
        print("📚 VERIFIED SOURCE METADATA DELIVERED:")
        print("="*60)
        for idx, src in enumerate(sources):
            metadata = src.get("metadata", {})
            print(f"[{idx+1}] Paper: {metadata.get('source', 'Unknown')} | Section: {metadata.get('parent_section', 'Unknown')} | Reranker Score: {src['score']:.4f}")
            
    except Exception as e:
        print(f"\n[Runtime Validation Error] Pipeline test failed: {e}")
