"""Gradio UI for RAG Chat Assistant."""

from pathlib import Path
from typing import List, Tuple

import gradio as gr

from retrieval.hybrid_retrieval import HybridRetriever

def _format_document_context(doc, index: int) -> str:
    """Format a single document with metadata and content."""
    metadata = doc["metadata"] if isinstance(doc, dict) and "metadata" in doc else {}
    # source = metadata.get("source", "Unknown")

    lines = []
    if "page_number" in metadata:
        lines.append(f"Page: {metadata['page_number']}")
    if "chunk_number" in metadata:
        lines.append(f"Chunk: {metadata['chunk_number']}")

    lines.append("")
    lines.append(doc["content"])
    return "\n".join(lines)

def format_context(context):
    result = "<h2 style='color: #ff7800;'>Relevant Context</h2>\n\n"
    for doc in context:
        metadata = doc["metadata"]
        result += f"<span style='color: #ff7800;'>Source: {metadata.get('source', 'Unknown')}</span>, "
        result += f"<span style='color: #ff7800;'> Section: {metadata.get('parent_section', 'Unknown')}</span>\n\n"
        result += doc["content"] + "\n\n"
    return result


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

    formatted_contexts = format_context(docs)
    return answer, formatted_contexts

def create_ui(retriever: HybridRetriever):
    """Create and return the Gradio UI blocks."""
    
    def put_message_in_chatbot(message, history):
        return "", history + [{"role": "user", "content": message}]
    
    theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])

    with gr.Blocks(title="Research Assistant Chat", theme=theme) as demo:
        gr.Markdown("# Research Assistant Chat", elem_id="title")
        gr.Markdown("Ask questions about your research papers and explore retrieved context directly from the conversation pane.")

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
                # gr.Markdown("### Retrieved Contexts")
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
