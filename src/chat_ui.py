"""Gradio UI for RAG Chat Assistant."""

from pathlib import Path
from typing import List, Tuple

import gradio as gr

from retrieval.hybrid_retrieval import HybridRetriever

ASSETS_DIR = Path(__file__).parent / "assets"
USER_AVATAR = str(ASSETS_DIR / "avatar_user.svg")
ASSISTANT_AVATAR = str(ASSETS_DIR / "avatar_assistant.svg")

EMPTY_CONTEXT_HTML = (
    "<div class='ctx-empty'>No context retrieved yet — ask a question to see relevant sources here.</div>"
)

THEME = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])

CSS_PATH = str(ASSETS_DIR / "chat_ui.css")


def _format_document_context(doc, index: int) -> str:
    """Format a single document with metadata and content."""
    metadata = doc["metadata"] if isinstance(doc, dict) and "metadata" in doc else {}

    lines = []
    if "page_number" in metadata:
        lines.append(f"Page: {metadata['page_number']}")
    if "chunk_number" in metadata:
        lines.append(f"Chunk: {metadata['chunk_number']}")

    lines.append("")
    lines.append(doc["content"])
    return "\n".join(lines)


def format_context(context):
    if not context:
        return EMPTY_CONTEXT_HTML

    cards = []
    for doc in context:
        metadata = doc["metadata"]
        source = metadata.get("source", "Unknown")
        section = metadata.get("parent_section", "Unknown")
        chunk = metadata.get("chunk_number", "Unknown")
        cards.append(
            "<div class='ctx-card'>"
            "<div class='ctx-meta'>"
            f"<span class='ctx-source'>{source}</span>"
            "<span class='ctx-sep'>&rsaquo;</span>"
            f"<span>{section}</span>"
            "<span class='ctx-sep'>&rsaquo;</span>"
            f"<span>Chunk {chunk}</span>"
            "</div>"
            f"<div class='ctx-content'>{doc['content']}</div>"
            "</div>"
        )
    return "<div class='ctx-list'>" + "\n".join(cards) + "</div>"


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

    def clear_conversation():
        return [], EMPTY_CONTEXT_HTML

    with gr.Blocks(title="Research Assistant Chat") as demo:
        with gr.Column(elem_id="app-shell"):
            with gr.Row(elem_id="app-header"):
                gr.Markdown(
                    "# Research Assistant Chat\n"
                    "Ask questions about your research papers and explore retrieved context directly from the conversation pane.",
                    elem_id="header-text",
                )
                clear_btn = gr.Button(
                    "Clear conversation", elem_id="clear_btn", size="sm", variant="secondary", scale=0
                )

            with gr.Row(elem_id="main-row"):
                # Left column: Chat interface
                with gr.Column(scale=3, min_width=520, elem_id="chat-column"):
                    chatbot = gr.Chatbot(
                        show_label=False,
                        height=560,
                        elem_id="chatbot",
                        layout="panel",
                        buttons=[],
                        avatar_images=(USER_AVATAR, ASSISTANT_AVATAR),
                        placeholder="## Ask anything about your research papers\nYour conversation and its retrieved sources will appear here.",
                    )

                    with gr.Row(elem_id="input-row"):
                        msg = gr.Textbox(
                            placeholder="Message Research Assistant...",
                            show_label=False,
                            elem_id="user_input",
                            container=False,
                            autofocus=True,
                            lines=1,
                            max_lines=6,
                            scale=9,
                        )
                        send_btn = gr.Button("➤", elem_id="send_btn", scale=1, min_width=44)

                # Right column: Retrieved contexts
                with gr.Column(scale=2, min_width=320, elem_id="context-sidebar"):
                    gr.Markdown("### Relevant Context", elem_id="context-title")
                    contexts_display = gr.Markdown(
                        value=EMPTY_CONTEXT_HTML,
                        elem_id="contexts_display",
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

        # Submit on message enter or send-button click
        msg.submit(put_message_in_chatbot, [msg, chatbot], [msg, chatbot]).then(
            respond, [chatbot], [chatbot, contexts_display]
        )
        send_btn.click(put_message_in_chatbot, [msg, chatbot], [msg, chatbot]).then(
            respond, [chatbot], [chatbot, contexts_display]
        )
        clear_btn.click(clear_conversation, None, [chatbot, contexts_display])

    return demo


if __name__ == "__main__":
    # Create and launch the UI

    retriever = HybridRetriever()
    demo = create_ui(retriever)

    demo.launch(
        theme=THEME,
        css_paths=CSS_PATH,
        inbrowser=True,
        share=False,
        server_port=7865
    )
