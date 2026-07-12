import sys
from pathlib import Path
from typing import List, Tuple
import numpy as np
from sklearn.manifold import TSNE
from qdrant_client import QdrantClient
import plotly.express as px  # Add this to your imports at the top!
import plotly.graph_objects as go # Keep this if you use it for layout styling

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.schema.document import Document, DocumentMetadata
import src.config.settings as config

class EmbeddingVisualizer:
    def __init__(self, output_path: Path = config.VISUALIZATION_OUTPUT_PATH, 
                 dimension: str = config.VISUALIZATION_DIMENSION,
                 group_by: str = config.VISUALIZATION_GROUP_BY):
        self.output_path = output_path
        self.dimension = dimension.upper()
        self.group_by = group_by.lower()
        # --- Capture Model Properties from Config ---
        self.model_name = getattr(config, "EMBEDDING_MODEL_NAME", "Unknown Model")
        self.max_tokens = getattr(config, "TARGET_MAX_TOKENS", "Unknown")
        self.vector_dimension = getattr(config, "VECTOR_DIMENSION", "Unknown")
        
        if self.dimension not in ["2D", "3D"]:
            print(f"[Warning] Invalid dimension '{dimension}'. Defaulting to '3D'.")
            self.dimension = "3D"

    def fetch_and_visualize(self, collection_name: str = config.QDRANT_COLLECTION_NAME) -> None:
        """
        Connects directly to the persistent Qdrant instance, scrolls out all embedded 
        vectors and payloads, and triggers layout compilation.
        """
        print(f"Connecting to persistent vector storage to pull points...")
        if config.QDRANT_URL:
            client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
        else:
            client = QdrantClient(path=config.QDRANT_LOCATION)

        # Scroll out records from local DB
        records, _ = client.scroll(
            collection_name=collection_name,
            with_payload=True,
            with_vectors=True,
            limit=10000
        )

        if not records:
            print(f"[Error] No vector records found inside collection '{collection_name}'. Ingestion required first.")
            return

        print(f"Successfully fetched {len(records)} active vector records from database storage.")
        
        embedded_payloads = []
        for record in records:
            payload = record.payload or {}
            vector = record.vector
            
            # Smart Fallback: Check if fields are inside a nested "metadata" dict OR flat at the root payload level
            meta_source = payload.get("metadata", payload) if isinstance(payload.get("metadata"), dict) else payload
            
            # Safely extract with multiple common naming variations
            paper_name = meta_source.get("paper_name", payload.get("paper_name", "Unknown"))
            soure_name = meta_source.get("source", "Unknown")
            parent_section = meta_source.get("parent_section", payload.get("parent_section", "Unknown"))
            page_number = meta_source.get("page_number", payload.get("page_number", 0))
            token_count = meta_source.get("token_count", payload.get("token_count", 0))
            keywords = meta_source.get("keywords", payload.get("keywords", []))
            chunk_number = meta_source.get("chunk_number", payload.get("chunk_number", 0))
            
            # Reconstruct the Document Schema layer
            metadata = DocumentMetadata(
                source=soure_name,
                paper_name=paper_name,
                chunk_number=chunk_number,
                parent_section=parent_section,
                page_number=page_number,
                token_count=token_count,
                keywords=keywords
            )
            
            # Pull text content from common standard keys
            content = payload.get("content", payload.get("text", "No text content available."))
            
            doc = Document(content=content, metadata=metadata)
            embedded_payloads.append((doc, vector))

        # Regenerate the canvas using the newly fortified objects
        self.generate_scatter_plot(embedded_payloads)

    def generate_scatter_plot(self, embedded_payloads: List[Tuple[Document, List[float]]]) -> str:
        """
        Reduces high-D embeddings and renders an interactive Plotly scatter map.
        Groups and colors cleanly by file name or section attributes.
        """
        if len(embedded_payloads) < 4:
            print("[Warning] Not enough data points to compute meaningful t-SNE distributions. Need at least 4 chunks.")
            return ""

        vectors = np.array([payload[1] for payload in embedded_payloads])
        documents = [payload[0] for payload in embedded_payloads]

        computed_perplexity = min(30, max(2, len(vectors) // 2))
        n_components = 2 if self.dimension == "2D" else 3
        
        print(f"Executing t-SNE dimension reduction ({config.VECTOR_DIMENSION}D -> {self.dimension}) with perplexity={computed_perplexity}...")
        tsne = TSNE(n_components=n_components, perplexity=computed_perplexity, random_state=42, init='pca')
        transformed_vectors = tsne.fit_transform(vectors)

        x_coords = transformed_vectors[:, 0]
        y_coords = transformed_vectors[:, 1]
        z_coords = transformed_vectors[:, 2] if self.dimension == "3D" else None

        # --- Clean Label Extraction & Formatting ---
        group_labels = []
        for doc in documents:
            val = getattr(doc.metadata, self.group_by, "Unknown")
            
            if isinstance(val, dict):
                val = val.get("filename", val.get("name", val.get("title", "Unknown")))
            
            group_labels.append(str(val) if val is not None else "Unknown")
            
        unique_groups = sorted(list(set(group_labels))) # Sorted keeps color assignments consistent
        
        # Define our high-contrast, vibrant palette
        VIBRANT_HIGH_CONTRAST_PALETTE = [
            "#E31A1C",  # Vibrant Crimson / Red
            "#1F78B4",  # Rich Royal Blue
            "#33A02C",  # Deep Emerald Green
            "#FF7F00",  # Electric Vibrant Orange
            "#6A3D9A",  # Dark Velvet Purple
            "#008080",  # Crisp Dark Teal
            "#B15928",  # Burnt Sienna / Brown
            "#FF007F",  # Bright Hot Pink
            "#FFD700"   # Deep Gold / Yellow
        ]

        fig = go.Figure()

        # Enumerate gives us an index (g_idx) to pull a unique color for each group
        for g_idx, group in enumerate(unique_groups):
            indices = [i for i, g in enumerate(group_labels) if g == group]
            hover_text = []
            
            for idx in indices:
                doc = documents[idx]
                snippet = doc.content.replace("<", "&lt;").replace(">", "&gt;")[:150]
                
                p_name = doc.metadata.paper_name
                source = doc.metadata.source
                if isinstance(p_name, dict):
                    p_name = p_name.get("filename", "Unknown")

                card = (
                    f"<b>Source:</b> {source}<br>"
                    f"<b>Paper:</b> {p_name}<br>"
                    f"<b>Section:</b> {doc.metadata.parent_section}<br>"
                    # f"<b>Page:</b> {doc.metadata.page_number} | <b>Tokens:</b> {doc.metadata.token_count}<br>"
                    f"<b>Keywords:</b> {', '.join(doc.metadata.keywords) if doc.metadata.keywords else 'None'}<br>"
                    f"<b>Content:</b> {snippet}..."
                )
                hover_text.append(card)
                
            # Safely pick color (modulo wrap-around avoids errors if you have more than 9 files)
            assigned_color = VIBRANT_HIGH_CONTRAST_PALETTE[g_idx % len(VIBRANT_HIGH_CONTRAST_PALETTE)]

            if self.dimension == "2D":
                fig.add_trace(go.Scatter(
                    x=x_coords[indices], 
                    y=y_coords[indices], 
                    mode='markers', 
                    name=str(group),
                    text=hover_text, 
                    hoverinfo='text',
                    marker=dict(
                        size=9, 
                        color=assigned_color,  # Fix: Assign single group color here
                        opacity=0.85, 
                        line=dict(width=1, color='DarkSlateGrey') # Keep points crisp on zoom out
                    )
                ))
            else:
                fig.add_trace(go.Scatter3d(
                    x=x_coords[indices], 
                    y=y_coords[indices], 
                    z=z_coords[indices], 
                    mode='markers', 
                    name=str(group),
                    text=hover_text, 
                    hoverinfo='text',
                    marker=dict(
                        size=5, 
                        color=assigned_color,  # Fix: Assign single group color here
                        opacity=0.85
                    )
                ))

       # --- Fixed Native Sidebar Legend Settings ---
        shared_legend_style = dict(
            orientation="v",        
            yanchor="top",
            y=1.0,                  
            xanchor="left",
            x=1.02,                 
            traceorder="normal",
            itemclick="toggle",
            itemdoubleclick="toggleothers",
            title=dict(text=f"<b>{self.group_by.replace('_', ' ').title()}:</b>")
        )
        
        dynamic_title = dict(
            text=(
                f"<b>RAG Vector Space Map (Grouped by {self.group_by.replace('_', ' ').title()})</b><br>"
                f"<span style='font-size: 13px; color: #666666; font-weight: normal;'>"
                f"Vector Engine: {self.model_name} &nbsp;|&nbsp; Max Token Allocation: {self.max_tokens} tokens &nbsp;|&nbsp; "
                f"Vector Dimension: {self.vector_dimension}D &nbsp;|&nbsp; Total Chunks: {len(embedded_payloads)}"
                f"</span>"
            ),
            x=0.5,
            y=0.95,
            xanchor="center",
            yanchor="top"
        )
        
        if self.dimension == "2D":
            fig.update_layout(
                title=dynamic_title,
                xaxis=dict(title="t-SNE Axis 1", gridcolor="rgb(245, 245, 245)"),
                yaxis=dict(title="t-SNE Axis 2", gridcolor="rgb(245, 245, 245)"),
                plot_bgcolor="white", 
                autosize=True,               
                height=None,                  
                margin=dict(l=50, r=180, b=60, t=100), # Boosted top margin to 100 for title breathing room
                showlegend=True,  
                legend=shared_legend_style
            )
        else:
            fig.update_layout(
                title=dynamic_title,
                scene=dict(
                    xaxis=dict(title="t-SNE Axis 1", backgroundcolor="rgb(240, 240, 240)"),
                    yaxis=dict(title="t-SNE Axis 2", backgroundcolor="rgb(240, 240, 240)"),
                    zaxis=dict(title="t-SNE Axis 3", backgroundcolor="rgb(240, 240, 240)")
                ),
                autosize=True,               
                height=None,                  
                margin=dict(l=20, r=180, b=60, t=100), # Boosted top margin to 100
                showlegend=True,  
                legend=shared_legend_style
            )
            
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(self.output_path))
        print(f"[Success] Interactive decoupled plot generated at: {self.output_path}")
        return str(self.output_path)

if __name__ == "__main__":
    # Allows updating or re-rendering charts instantly without running ingestion!
    print("=== STANDALONE VECTOR SPACE VISUALIZER RUN ===")
    visualizer = EmbeddingVisualizer()
    visualizer.fetch_and_visualize()