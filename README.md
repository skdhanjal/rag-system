# Academic Literature RAG System

## Overview

This project is a Retrieval-Augmented Generation (RAG) application for exploring foundational research papers on Transformer-based language models. It ingests academic PDFs, preserves useful document structure during preprocessing, indexes semantically meaningful chunks in Qdrant, and provides a conversational interface that answers questions from the indexed literature.

It was built for the **Generative AI Fundamentals** assignment, which requires a RAG application over five PDFs, semantic vector retrieval, a conversational bot, an evaluation workflow, and a final report.

The detailed design rationale and experimental analysis belong in the accompanying PDF report. This README focuses on what the repository does and how to run it.

## Project objectives

- Ingest and process five supplied academic PDFs.
- Split extracted content into meaningful, structure-aware chunks.
- Generate embeddings and persist them in a vector database.
- Retrieve relevant paper excerpts and generate grounded answers in a chat interface.
- Evaluate retrieval and answer quality against a curated question set.
- Provide supporting diagnostics, including an embedding-space visualization.

## What has been implemented

The current codebase provides the following capabilities:

- **Document ingestion:** Docling parses PDFs with layout analysis, table-structure extraction, formula enrichment, and optional figure extraction.
- **Reusable parsed data:** Parsed Docling documents can be stored as JSON and reloaded without parsing the PDFs again.
- **Cleaning and chunking:** Boilerplate such as page headers, footers, emails, and publication notices is removed. Docling's `HybridChunker` creates token-aware chunks while retaining heading and page context.
- **Embedding and storage:** `sentence-transformers/all-MiniLM-L6-v2` creates normalized 384-dimensional embeddings, which are stored in a persistent local Qdrant collection by default.
- **Retrieval and reranking:** Dense vector candidates are combined with text-match candidates, reranked with a cross-encoder, and expanded with the immediately previous and next chunks from the same paper.
- **Grounded chat:** A Gradio interface sends retrieved excerpts and chat history to the configured LLM and displays the supporting contexts beside each answer.
- **Evaluation dashboard:** A second Gradio interface evaluates retrieval with MRR, nDCG, and keyword coverage, and evaluates answers with LLM-as-a-judge scores for accuracy, completeness, and relevance.
- **Question set:** `src/evaluation/tests.jsonl` contains 50 evaluation questions spanning the five papers.
- **Vector visualization:** A Plotly/t-SNE utility produces an interactive map of the stored embedding space.

## Included research corpus

The source PDFs are stored in [`docs/pdfs`](docs/pdfs) and correspond to the five papers specified in the assignment:

| Paper | File |
| --- | --- |
| Attention Is All You Need | `Attention_is_all_you_need.pdf` |
| BERT | `Pre-training_of_Bidirectional_Transformers.pdf` |
| GPT-3 | `Language_Models_are_Few-Shot_Learners.pdf` |
| RoBERTa | `A_Robustly_Optimized_BERT.pdf` |
| T5 | `Limits_of_transfer_learning.pdf` |

## Repository layout

```text
src/
├── ingestion/       PDF/JSON loading, cleaning, chunking, embeddings, Qdrant writes
├── retrieval/       Query routing, retrieval, reranking, and context expansion
├── evaluation/      Test set, retrieval metrics, and LLM-as-a-judge evaluation
├── config/          Runtime models, paths, and retrieval settings
├── prompts/         Chat and query-routing prompts
├── chat_ui.py       Gradio chat application
├── eval_ui.py       Gradio evaluation dashboard
└── vector_space_visualizer.py

docs/
├── pdfs/            Source research papers
├── parsed_json/     Pre-parsed Docling documents used by the fast ingestion mode
├── clean-docs/      Saved cleaned-document examples
└── graphs/          Previous evaluation and visualization exports

local_qdrant/        Persistent local Qdrant data (created/populated at runtime)
```

## Prerequisites

- Python 3.12 or later
- [uv](https://docs.astral.sh/uv/)
- Internet access on first run to download Docling and Hugging Face model assets
- API credentials for the LLM services configured in `src/config/settings.py`

The current configuration uses Google Gemini for answer generation and OpenAI models for intent routing and answer evaluation. The `.env` file is intentionally ignored; never commit API keys.

## Setup

From the repository root:

```powershell
uv sync
```

Create a `.env` file in the repository root and provide the keys used by the current configuration:

```dotenv
GOOGLE_API_KEY=your_google_ai_api_key
OPENAI_API_KEY=your_openai_api_key
HF_TOKEN=your_hugging_face_token
```

`HF_TOKEN` is useful for authenticated Hugging Face downloads. Depending on the public availability of the configured models, it may not be required.

### Configuration

Runtime settings live in [`src/config/settings.py`](src/config/settings.py). Common values to review are:

- `EMBEDDING_MODEL_NAME` and `VECTOR_DIMENSION` — these must remain compatible.
- `QDRANT_LOCATION` — local persistence path; set `QDRANT_URL` and `QDRANT_API_KEY` in `.env` to use a remote Qdrant instance.
- `LLM_MODEL_NAME` — answer-generation model.
- `LLM_CLASSIFIER_MODEL_NAME` — query-routing model.
- `LLM_AS_JUDGE_MODEL_NAME` — evaluation judge model.
- `RETRIEVAL_TOP_K` and `RERANK_TOP_K` — number of candidates retrieved and retained.

## Ingest the documents

Ingestion creates embeddings and writes document payloads to the `llm_research_collection` Qdrant collection.

### Fast mode: load existing parsed Docling JSON

This is the default and recommended mode for the repository because parsed documents are already available in `docs/parsed_json`.

```powershell
uv run python -m src.ingestion.ingestion_pipeline --loader json
```

### Full mode: parse the source PDFs again

Use this mode after replacing or adding PDFs. It runs Docling layout analysis before cleaning, chunking, embedding, and storage, so it is slower.

```powershell
uv run python -m src.ingestion.ingestion_pipeline --loader pdf --dir docs/pdfs
```

The PDF loader writes Docling JSON to a `parsed_json` folder next to the input directory. For the command above, the generated files are placed in `docs/pdfs/parsed_json`. To ingest those generated JSON files without reparsing, run:

```powershell
uv run python -m src.ingestion.ingestion_pipeline --loader json --dir docs/pdfs/parsed_json
```

### Use a different input directory

Both modes accept `--dir`:

```powershell
# Parse PDFs from another folder
uv run python -m src.ingestion.ingestion_pipeline --loader pdf --dir path\to\pdfs

# Load pre-parsed Docling JSON from another folder
uv run python -m src.ingestion.ingestion_pipeline --loader json --dir path\to\parsed_json
```

At the end of a successful run, the pipeline prints the Qdrant collection status, point count, and a sample stored payload.

## Run the chat application

Run ingestion first, then start the Gradio chat UI:

```powershell
uv run python src/chat_ui.py
```

Gradio opens the application in a browser. Ask questions about the five indexed papers; the right-hand panel displays the retrieved source excerpts and metadata used for the answer.

## Run the evaluation dashboard

Ensure the Qdrant collection has been populated, then start the dashboard:

```powershell
uv run python src/eval_ui.py
```

The dashboard has two independent flows:

1. **Retrieval evaluation** calculates MRR, nDCG, and keyword coverage for all questions in `src/evaluation/tests.jsonl`.
2. **Answer evaluation** generates an answer for each question and uses the configured judge model to score accuracy, completeness, and relevance on a 1–5 scale.

Answer evaluation makes external LLM calls and can take time and incur provider usage costs.

### Run one evaluation question from the command line

Use a one-based row number from `tests.jsonl`:

```powershell
uv run python src/evaluation/eval.py 1
```

## Generate the embedding-space visualization

After ingestion, create or refresh the interactive Plotly visualization:

```powershell
uv run python src/vector_space_visualizer.py
```

The generated HTML file is written to `docs/embedding_space.html` by default. Change visualization settings such as 2D/3D mode or grouping in `src/config/settings.py`.

## Operational notes

- Re-running ingestion upserts deterministic chunk IDs, so existing chunks are updated instead of blindly duplicated.
- Local Qdrant data persists under `local_qdrant` unless `QDRANT_LOCATION` is changed.
- The current chat interface passes its conversation history to generation. The README does not claim a strict four-turn memory limit because that limit is not yet enforced in the UI layer.
- The current implementation uses dense embeddings and a text-match candidate path. It does not yet implement a true BM25 index or reciprocal-rank fusion.
- The final assignment report is a separate deliverable and should document architectural decisions, experiments, evaluation results, limitations, and future improvements.

## Troubleshooting

- **No retrieved context or Qdrant collection error:** Run the ingestion pipeline first and confirm that it reports stored points.
- **Model download/authentication error:** Check your internet connection and `HF_TOKEN`; then rerun `uv sync` if dependencies are incomplete.
- **LLM authentication or routing error:** Confirm `GOOGLE_API_KEY` and `OPENAI_API_KEY` are present in `.env` and that the configured models are available to your account.
- **Embedding dimension error:** If you change `EMBEDDING_MODEL_NAME`, update `VECTOR_DIMENSION` and recreate the Qdrant collection before re-ingesting.
- **Slow first run:** Initial model downloads and PDF parsing can take substantially longer than later runs using `--loader json`.

## Assignment alignment

| Assignment area | Repository support |
| --- | --- |
| Five-PDF ingestion | Included source corpus and Docling ingestion pipeline |
| Vector database | Persistent local Qdrant collection |
| Semantic retrieval | SentenceTransformer embeddings, retrieval, reranking, and context expansion |
| Conversational interface | Gradio chat application |
| Evaluation | 50-question test set with retrieval and answer-quality metrics |
| Final report | To be submitted separately as a PDF deliverable |

## License

No license has been specified for this repository.
