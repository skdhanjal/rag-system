# Academic Literature RAG System

An evidence-grounded Retrieval-Augmented Generation (RAG) application for exploring five foundational Transformer-era research papers. The system parses academic PDFs, preserves useful document structure, indexes traceable chunks in Qdrant, and answers questions through a Gradio chat interface.

This repository was created for the Generative AI Fundamentals assignment. The accompanying report contains the detailed architecture, experiments, and evaluation discussion; this README explains what is included and how to run it.

## What the project delivers

- **Structure-aware ingestion:** Docling parses PDFs with layout analysis, table extraction, formula enrichment, and figure-image support.
- **Fast repeatable ingestion:** Fresh PDF parsing writes reusable Docling JSON. The default JSON loader reuses those files, avoiding repeated CPU-intensive PDF processing.
- **Clean, contextual chunks:** Cleaning removes recurring boilerplate. Docling `HybridChunker` produces token-aware chunks and retains paper, section, page, and chunk-number metadata.
- **Persistent vector search:** SentenceTransformer embeddings are stored in a local Qdrant collection with source metadata.
- **Improved retrieval:** Dense semantic candidates and Qdrant text-match candidates are deduplicated, reranked by a cross-encoder, and expanded with the preceding and following chunks from the same paper.
- **Conversational answers:** The Gradio chat application retains exactly the four most recent completed user/assistant turns for routing and answer generation, while leaving the complete transcript visible in the UI.
- **Basic guardrails:** The chat flow validates user queries, blocks common prompt-injection attempts, checks that retrieval returned enough evidence, sanitizes instruction-like text inside retrieved context, and handles empty model responses safely.
- **Configurable LLM providers:** Answer generation and routing can use local Ollama or hosted OpenAI, Google, and Groq models through LiteLLM configuration.
- **Evaluation:** The separate Gradio dashboard measures retrieval quality and uses an LLM judge to assess generated answers against reference answers.
- **Experiment artefacts:** Saved evaluation exports, embedding-space plots, extracted figures, parsed Docling JSON, and the print-ready report are included under `docs/`.

## Source corpus

The five assignment PDFs are in [`docs/pdfs`](docs/pdfs).

| Paper | File |
| --- | --- |
| Attention Is All You Need | `Attention_is_all_you_need.pdf` |
| BERT | `Pre-training_of_Bidirectional_Transformers.pdf` |
| GPT-3 | `Language_Models_are_Few-Shot_Learners.pdf` |
| RoBERTa | `A_Robustly_Optimized_BERT.pdf` |
| T5 | `Limits_of_transfer_learning.pdf` |

## Quick start

From the repository root:

```powershell
uv sync
```

Create a `.env` file, populate the credentials for the provider(s) you use, then ingest the supplied pre-parsed documents:

```powershell
uv run python -m src.ingestion.ingestion_pipeline --loader json
```

Start the chat application:

```powershell
uv run python src/chat_ui.py
```

## Prerequisites and provider setup

- Python 3.12 or later
- [uv](https://docs.astral.sh/uv/)
- Internet access on the first run to download Docling and Hugging Face model assets
- A running Ollama service for local generation, or API credentials for hosted models

The checked-in settings default to the `groq` provider with model `groq/openai/gpt-oss-120b` for answer generation and routing, and `gemini/gemini-3.1-flash-lite` as the answer-evaluation judge. The active answer provider can be changed without modifying retrieval code. Both Google and Groq API keys can be created without a paid subscription in most regions.

Use the following as a starting `.env` file. Never commit this file or any API keys.

```dotenv

# Current default answer-generation and routing provider
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key

# Required by the configured LLM-as-a-judge answer evaluation
GOOGLE_API_KEY=your_google_ai_api_key

# Optional: local Ollama answer generation
OLLAMA_MODEL=llama3.2
OLLAMA_API_BASE=http://localhost:11434

# Optional: Groq answer generation
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b

# Optional: Google answer generation
# LLM_PROVIDER=google
# GOOGLE_MODEL=gemini/gemini-3.1-flash-lite

LLM_TIMEOUT_SECONDS=60
HF_TOKEN=your_hugging_face_token
```

`HF_TOKEN` is useful for authenticated Hugging Face downloads but may not be required for public models. For chat-only local use, set `LLM_PROVIDER=ollama` and ensure Ollama is running. The answer-evaluation flow still needs the configured hosted judge credential.

## Configuration

Runtime configuration is centralised in [`src/config/settings.py`](src/config/settings.py).

- `EMBEDDING_MODEL_NAME` and `VECTOR_DIMENSION` must match. The checked-in configuration uses `sentence-transformers/all-MiniLM-L6-v2` with 384 dimensions. `BAAI/bge-base-en-v1.5` with 768 dimensions is a tested alternative.
- `TARGET_MAX_TOKENS` is the Docling HybridChunker limit; it is set to 500.
- `QDRANT_LOCATION` controls persistent local storage. Set `QDRANT_URL` and `QDRANT_API_KEY` in `.env` for a remote Qdrant deployment.
- `RETRIEVAL_TOP_K=10` controls initial candidates and `RERANK_TOP_K=3` controls the candidates retained after cross-encoder reranking.
- `MAX_CONVERSATION_TURNS=4` limits the model memory window to the latest four completed exchanges.
- `LLM_PROVIDER` selects `ollama`, `openai`, `google`, or `groq`. Provider model names can be overridden with the related environment variables.
- `LLM_MAX_TOKENS=2048` and `LLM_TIMEOUT_SECONDS=60` control answer-generation length and request timeout.
- `LLM_AS_JUDGE_MODEL_NAME` selects the reference-answer-based evaluator.
- `ENABLE_GUARDRAILS`, `MIN_QUERY_CHARS`, `MAX_QUERY_CHARS`, and `MIN_RETRIEVAL_RESULTS` control the lightweight runtime guardrails.

If you change the embedding model or vector dimension, recreate the Qdrant collection and ingest the corpus again. Existing vectors cannot be reused across incompatible dimensions.

## Runtime guardrails

The chat flow includes basic deterministic guardrails in [`src/helpers/guardrails.py`](src/helpers/guardrails.py). These are intentionally simple and transparent:

- invalid, empty, or oversized queries are rejected before routing;
- common prompt-injection phrases such as attempts to reveal hidden prompts or override instructions are blocked;
- follow-up requests that only ask to shorten, rewrite, or reformat the previous answer are handled as response refinements instead of fresh retrieval queries;
- answer generation is skipped when retrieval does not return enough evidence;
- retrieved context is sanitized before it is inserted into the system prompt;
- empty LLM responses are replaced with a safe fallback message.

These checks do not replace a full moderation or policy framework. They provide a practical first layer for this academic RAG application and keep the behavior easy to inspect in code.

## Ingest documents

Ingestion cleans the source documents, creates structural chunks, embeds them, and upserts them into the `llm_research_collection` Qdrant collection.

### Fast mode: use existing Docling JSON

This is the default and recommended mode for the supplied corpus. It loads the parsed Docling documents already stored in `docs/parsed_json`.

```powershell
uv run python -m src.ingestion.ingestion_pipeline --loader json
```

### Full mode: parse PDFs again

Use this after adding or replacing PDFs. Docling layout processing can be slow on CPU. Each parse writes a `.docling.json` file beside the input directory for later reuse.

```powershell
uv run python -m src.ingestion.ingestion_pipeline --loader pdf --dir docs/pdfs
```

The command above writes parsed documents to `docs/pdfs/parsed_json`. Re-ingest that output without parsing the PDFs again:

```powershell
uv run python -m src.ingestion.ingestion_pipeline --loader json --dir docs/pdfs/parsed_json
```

Both modes accept `--dir` when a different input directory is required.

```powershell
# Parse PDFs from another folder
uv run python -m src.ingestion.ingestion_pipeline --loader pdf --dir path\to\pdfs

# Load pre-parsed Docling JSON from another folder
uv run python -m src.ingestion.ingestion_pipeline --loader json --dir path\to\parsed_json
```

The PDF loader supports figure-image extraction, and existing extracted figures are kept in `docs/extracted_images`. Images are not currently sent to a vision LLM or added as figure summaries to the text index; the current retrieval corpus is text-based.

## Run the applications

Run ingestion first so the local Qdrant collection contains vectors.

### Chat interface

```powershell
uv run python src/chat_ui.py
```

Gradio opens a browser application. The source panel shows the paper, section, page, and retrieved excerpts used to ground each answer.

To switch answer-generation providers, update `LLM_PROVIDER` in `.env` and restart the application.

```dotenv
# Local open-source model
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2

# Hosted Groq model
LLM_PROVIDER=groq
GROQ_MODEL=openai/gpt-oss-120b

# Other supported values
# LLM_PROVIDER=openai
# LLM_PROVIDER=google
```

### Evaluation dashboard

```powershell
uv run python src/eval_ui.py
```

The dashboard provides two independent flows for all 50 questions in `src/evaluation/tests.jsonl`:

1. **Retrieval evaluation** reports keyword-based MRR, nDCG, and keyword coverage over retrieved context.
2. **Answer evaluation** runs the full RAG answer path and asks the configured LLM judge to compare the generated answer with the reference answer. It scores accuracy, completeness, and relevance from 1 to 5 and returns feedback.

Answer evaluation is more expensive than retrieval evaluation. A normal question can invoke the LLM router, the answer-generation model, and the hosted judge.

### Run one evaluation question from the command line

The command-line evaluator runs both retrieval and answer evaluation for a zero-based row number from `tests.jsonl`.

```powershell
# Evaluate the first test question
uv run python src/evaluation/eval.py 0
```

## Generate the embedding-space visualization

After ingestion, generate or refresh the interactive Plotly/t-SNE visualization:

```powershell
uv run python src/vector_space_visualizer.py
```

The output is written to `docs/embedding_space.html`. Set 2D or 3D display and grouping behaviour in `src/config/settings.py`.

## Artefacts and report

| Location | Contents |
| --- | --- |
| `docs/eval_tests/all-MiniLM-L6-v2` | Saved MiniLM retrieval and answer-evaluation dashboard exports. |
| `docs/eval_tests/BAAI-bge-base-en-v1.5` | Saved BGE-base retrieval and answer-evaluation dashboard exports. |
| `docs/graphs` | Embedding-space visualizations and related experimental graphs. |
| `docs/parsed_json` | Pre-parsed Docling documents used by the fast ingestion path. |
| `docs/extracted_images` | Figure images retained from PDF parsing for inspection and potential multimodal enrichment. |

The report compares chunk sizes, preprocessing, MiniLM and BGE-base embeddings, retrieval enhancements, and the two recorded end-to-end answer-evaluation runs.

## Repository layout

```text
src/
|-- ingestion/                 PDF/JSON loading, cleaning, chunking, embeddings, Qdrant writes
|-- retrieval/                 Query routing, candidate retrieval, reranking, context expansion
|-- helpers/                   LLM helper, guardrails, and conversation-memory utilities
|-- evaluation/                Test set, retrieval metrics, LLM-as-a-judge evaluation
|-- config/                    Runtime models, paths, and retrieval settings
|-- prompts/                   Chat and query-routing prompts
|-- chat_ui.py                 Gradio chat application
|-- eval_ui.py                 Gradio evaluation dashboard
`-- vector_space_visualizer.py

docs/
|-- pdfs/                      Source research papers
|-- parsed_json/               Reusable parsed Docling documents
|-- extracted_images/          Extracted PDF figures
|-- eval_tests/                Saved evaluation dashboard exports
|-- graphs/                    Visualization and experiment artefacts

local_qdrant/                  Persistent local Qdrant data, created at runtime
```

## Operational notes and troubleshooting

- Re-running ingestion upserts deterministic chunk IDs, so existing chunks are updated rather than blindly duplicated.
- The implementation uses dense embeddings plus a Qdrant text-match candidate path. It does not implement a true sparse BM25 index or reciprocal-rank fusion.
- If no context is retrieved or the collection is missing, run ingestion and confirm its point-count output.
- If a model download fails, check network access and `HF_TOKEN`, then rerun `uv sync` if dependencies are incomplete.
- If a provider request fails, check the selected provider's key, model availability, and timeout settings.
- For Ollama connection errors, start the Ollama service and confirm the selected model appears in `ollama list`.
- Initial model downloads and PDF parsing take longer than subsequent JSON-loader runs.

## Assignment alignment

| Assignment area | Repository support |
| --- | --- |
| Five-PDF ingestion | Included corpus and Docling PDF/JSON ingestion pipeline |
| Vector database | Persistent local Qdrant collection |
| Semantic retrieval | SentenceTransformer embeddings, cross-encoder reranking, and neighbour expansion |
| Open-source LLM option | Local Ollama support with `llama3.2` |
| Conversational interface | Gradio chat application with four-turn model memory |
| Basic guardrails | Query validation, prompt-injection checks, evidence fallback, context sanitization, and empty-answer fallback |
| Evaluation | 50-question test set with retrieval and LLM-judge answer metrics |
| Report and experiments | Pdf report plus saved evaluation and visualization artefacts |

## License

No license has been specified for this repository.
