"""Gradio UI for RAG Chat Assistant."""

from pathlib import Path
from typing import List, Tuple

import gradio as gr

from retrieval.hybrid_retrieval import HybridRetriever

def _format_document_context(doc, index: int) -> str:
    """Format a single document with metadata and content."""
    metadata = doc["metadata"] if isinstance(doc, dict) and "metadata" in doc else {}
    source = metadata.get("source", "Unknown")

    lines = [f"\n --- Document {index} ---", f"Source: {source}"]
    if "pages" in metadata:
        lines.append(f"Pages: {', '.join(map(str, metadata['pages']))}")
    if "chunk_number" in metadata:
        lines.append(f"Chunk: {metadata['chunk_number']}")

    lines.append("")
    lines.append(doc["content"])
    return "\n".join(lines)


def chat(query: str, history: List[List[str]], hybrid_retriever: HybridRetriever) -> Tuple[str, str]:
    """
    Chat wrapper that returns the answer and formatted document contexts.
    """
    messages = []
    for h in history:
        if isinstance(h, dict) and "role" in h and "content" in h:
            messages.append({"role": h["role"], "content": h["content"]})
        elif isinstance(h, (list, tuple)) and len(h) >= 2:
            messages.append({"role": "user", "content": h[0]})
    
    answer, docs = hybrid_retriever.answer_query(query, chat_history=messages)

    formatted_contexts = "\n".join(
        _format_document_context(doc, idx)
        for idx, doc in enumerate(docs, 1)
    )
    return answer, formatted_contexts

def create_ui(retriever: HybridRetriever):
    """Create and return the Gradio UI blocks."""
    
    def put_message_in_chatbot(message, history):
        return "", history + [{"role": "user", "content": message}]
    
    theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])

    with gr.Blocks(title="RAG Chat Assistant", theme=theme) as demo:
        gr.Markdown("# RAG Chat Assistant", elem_id="title")
        gr.Markdown("Chat with your documents using Retrieval-Augmented Generation (RAG). Retrieved contexts appear on the right.")

        with gr.Row():
            # Left column: Chat interface
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(
                    label="Chat History",
                    height=600
                )
                
                msg = gr.Textbox(
                    label="Your Question",
                    placeholder="Type your question here...",
                    show_label=False,
                )

            # Right column: Retrieved contexts
            with gr.Column(scale=1):
                gr.Markdown("### Retrieved Contexts")
                contexts_display = gr.Markdown(
                    value="Retrieved contexts will appear here...",
                    height=600,
                )

        # Handle user input and update both chat and contexts
        def respond(chat_history):
            """Process user message and return bot response with contexts."""
            
            last_message = chat_history[-1]["content"]
            last_message_text = last_message if isinstance(last_message, str) else last_message[0]["text"]
            prior_messages = chat_history[:-1] if chat_history else []
            
            answer, contexts_text = chat(last_message_text, prior_messages, retriever)
            
            chat_history.append({"role": "assistant", "content": answer})
            
            return chat_history, contexts_text

        # Submit on message enter
        msg.submit(put_message_in_chatbot, [msg, chatbot], [msg, chatbot]).then(respond, [chatbot], [chatbot, contexts_display])

    return demo


if __name__ == "__main__":
    # Create and launch the UI
    
    retriever = HybridRetriever()
    demo = create_ui(retriever)
    
    demo.launch(
        inbrowser=True,
        share=False,
    )
