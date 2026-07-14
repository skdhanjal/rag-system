"""Generate the first PDF draft for the Academic Literature RAG project."""

from pathlib import Path
from textwrap import wrap

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "reports" / "academic_literature_rag_report_v1.pdf"
PAGE_SIZE = (1654, 2339)
MARGIN = 110


def font(size: int, bold: bool = False):
    name = "arialbd.ttf" if bold else "arial.ttf"
    return ImageFont.truetype(Path("C:/Windows/Fonts") / name, size)


class Report:
    def __init__(self):
        self.pages = []
        self.page_number = 0
        self.new_page()

    def new_page(self):
        if self.pages:
            self._footer()
        self.page_number += 1
        self.image = Image.new("RGB", PAGE_SIZE, "white")
        self.draw = ImageDraw.Draw(self.image)
        self.y = MARGIN
        self.pages.append(self.image)

    def _footer(self):
        self.draw.line((MARGIN, 2220, PAGE_SIZE[0] - MARGIN, 2220), fill="#cbd5e1", width=2)
        self.draw.text((MARGIN, 2245), "Academic Literature RAG System - Report Draft v1", fill="#64748b", font=font(20))
        self.draw.text((PAGE_SIZE[0] - MARGIN - 25, 2245), str(self.page_number), fill="#64748b", font=font(20))

    def ensure_space(self, height: int):
        if self.y + height > 2180:
            self.new_page()

    def title(self, text: str):
        self.ensure_space(80)
        self.draw.text((MARGIN, self.y), text, fill="#0f172a", font=font(38, True))
        self.y += 62
        self.draw.line((MARGIN, self.y, PAGE_SIZE[0] - MARGIN, self.y), fill="#2563eb", width=4)
        self.y += 28

    def heading(self, text: str):
        self.ensure_space(58)
        self.draw.text((MARGIN, self.y), text, fill="#172554", font=font(27, True))
        self.y += 44

    def paragraph(self, text: str, size: int = 30, color="#334155", gap: int = 26):
        line_font = font(size)
        max_width = PAGE_SIZE[0] - 2 * MARGIN
        words = text.split()
        lines, line = [], ""
        for word in words:
            candidate = f"{line} {word}".strip()
            if self.draw.textlength(candidate, font=line_font) <= max_width:
                line = candidate
            else:
                lines.append(line)
                line = word
        if line:
            lines.append(line)
        self.ensure_space(len(lines) * (size + 10) + gap)
        for line in lines:
            self.draw.text((MARGIN, self.y), line, fill=color, font=line_font)
            self.y += size + 10
        self.y += gap

    def bullets(self, items):
        for item in items:
            self.paragraph(f"- {item}", size=28, gap=16)

    def callout(self, title: str, text: str, fill="#eff6ff", accent="#2563eb"):
        text_font = font(26)
        words = text.split()
        lines, line = [], ""
        max_width = PAGE_SIZE[0] - 2 * MARGIN - 70
        for word in words:
            candidate = f"{line} {word}".strip()
            if self.draw.textlength(candidate, font=text_font) <= max_width:
                line = candidate
            else:
                lines.append(line)
                line = word
        if line:
            lines.append(line)
        height = 82 + len(lines) * 38
        self.ensure_space(height + 22)
        self.draw.rounded_rectangle((MARGIN, self.y, PAGE_SIZE[0] - MARGIN, self.y + height), radius=16, fill=fill)
        self.draw.rectangle((MARGIN, self.y, MARGIN + 10, self.y + height), fill=accent)
        self.draw.text((MARGIN + 32, self.y + 22), title, fill="#172554", font=font(28, True))
        for index, line in enumerate(lines):
            self.draw.text((MARGIN + 32, self.y + 68 + index * 38), line, fill="#334155", font=text_font)
        self.y += height + 24

    def table(self, headers, rows, widths):
        row_height = 100
        self.ensure_space((len(rows) + 1) * row_height + 30)
        x = MARGIN
        for header, width in zip(headers, widths):
            self.draw.rectangle((x, self.y, x + width, self.y + row_height), fill="#1e3a8a")
            self.draw.text((x + 12, self.y + 30), header, fill="white", font=font(21, True))
            x += width
        self.y += row_height
        for index, row in enumerate(rows):
            x = MARGIN
            fill = "#eff6ff" if index % 2 == 0 else "#ffffff"
            for cell, width in zip(row, widths):
                self.draw.rectangle((x, self.y, x + width, self.y + row_height), fill=fill, outline="#cbd5e1")
                lines = wrap(str(cell), width=max(10, int(width / 12)))[:3]
                for line_no, line in enumerate(lines):
                    self.draw.text((x + 12, self.y + 16 + line_no * 23), line, fill="#1e293b", font=font(20))
                x += width
            self.y += row_height
        self.y += 24

    def architecture(self):
        self.ensure_space(970)
        base_y = self.y
        self.draw.rounded_rectangle((MARGIN, base_y, PAGE_SIZE[0] - MARGIN, base_y + 920), radius=20, fill="#f8fafc", outline="#94a3b8", width=2)
        self.draw.text((MARGIN + 28, base_y + 25), "Implementation-aligned system flow", fill="#1e293b", font=font(28, True))

        def node(x, y, width, height, title, details, fill):
            self.draw.rounded_rectangle((x, y, x + width, y + height), radius=14, fill=fill, outline="#64748b", width=2)
            self.draw.text((x + 16, y + 16), title, fill="#172554", font=font(22, True))
            for index, line in enumerate(details):
                self.draw.text((x + 16, y + 51 + index * 24), line, fill="#334155", font=font(19))

        def arrow(x1, y1, x2, y2):
            self.draw.line((x1, y1, x2, y2), fill="#475569", width=3)
            self.draw.polygon(((x2, y2), (x2 - 12, y2 - 7), (x2 - 12, y2 + 7)), fill="#475569")

        x = MARGIN + 45
        self.draw.rounded_rectangle((x, base_y + 85, x + 1320, base_y + 365), radius=16, fill="#eff6ff", outline="#93c5fd", width=2)
        self.draw.text((x + 20, base_y + 103), "OFFLINE INGESTION: prepare a searchable academic corpus", fill="#1d4ed8", font=font(20, True))
        node(x + 25, base_y + 150, 220, 145, "1. Source PDFs", ["Five research papers", "Multi-column layouts", "Tables and formulas"], "#ffffff")
        node(x + 285, base_y + 150, 245, 145, "2. Docling parsing", ["Recover reading order", "Detect layout and tables", "Preserve hierarchy"], "#ffffff")
        node(x + 570, base_y + 150, 250, 145, "3. Clean and chunk", ["Remove repeated noise", "HybridChunker uses", "headings and token limits"], "#ffffff")
        node(x + 860, base_y + 150, 400, 145, "4. Encode and store", ["BGE-base creates 768D vectors", "Store chunk text, page, section", "and order metadata in Qdrant"], "#dbeafe")
        arrow(x + 245, base_y + 222, x + 285, base_y + 222)
        arrow(x + 530, base_y + 222, x + 570, base_y + 222)
        arrow(x + 820, base_y + 222, x + 860, base_y + 222)

        self.draw.rounded_rectangle((x, base_y + 420, x + 1320, base_y + 825), radius=16, fill="#f0fdf4", outline="#86efac", width=2)
        self.draw.text((x + 20, base_y + 438), "ONLINE QUERY: retrieve evidence and generate a grounded answer", fill="#15803d", font=font(20, True))
        node(x + 25, base_y + 490, 230, 150, "1. Query and memory", ["User question", "Latest four completed", "conversation turns"], "#fffbeb")
        node(x + 295, base_y + 490, 230, 150, "2. Route and retrieve", ["Route greeting vs retrieval", "Dense Qdrant search", "Text-match candidates"], "#ffffff")
        node(x + 565, base_y + 490, 230, 150, "3. Improve context", ["Deduplicate candidates", "Cross-encoder reranking", "Expand N-1 and N+1"], "#ffffff")
        node(x + 835, base_y + 490, 425, 150, "4. Grounded generation", ["Build prompt from retrieved evidence", "Configurable Ollama, Groq, Google,", "or OpenAI model returns answer"], "#dcfce7")
        arrow(x + 255, base_y + 565, x + 295, base_y + 565)
        arrow(x + 525, base_y + 565, x + 565, base_y + 565)
        arrow(x + 795, base_y + 565, x + 835, base_y + 565)
        self.draw.text((x + 28, base_y + 680), "Qdrant provides the stored vectors, chunks, and metadata used by both retrieval and neighbor expansion.", fill="#475569", font=font(19))
        self.draw.text((x + 28, base_y + 720), "Cross-encoder reranking is the precision stage; the LLM only receives the final grounded context and bounded chat memory.", fill="#475569", font=font(19))
        self.y += 950

    def chart(self, labels, values, title):
        self.ensure_space(400)
        x0, y0 = MARGIN + 45, self.y + 55
        chart_w, chart_h = 1230, 250
        self.draw.text((MARGIN, self.y), title, fill="#172554", font=font(25, True))
        self.draw.line((x0, y0 + chart_h, x0 + chart_w, y0 + chart_h), fill="#64748b", width=2)
        max_value = 0.85
        gap = 55
        bar_w = int((chart_w - gap * (len(values) + 1)) / len(values))
        for i, (label, value) in enumerate(zip(labels, values)):
            x = x0 + gap + i * (bar_w + gap)
            height = int(chart_h * value / max_value)
            self.draw.rectangle((x, y0 + chart_h - height, x + bar_w, y0 + chart_h), fill="#2563eb")
            value_text = f"{value:.3f}"
            self.draw.text((x + 4, y0 + chart_h - height - 27), value_text, fill="#1e293b", font=font(16, True))
            for line_no, line in enumerate(wrap(label, width=12)):
                self.draw.text((x, y0 + chart_h + 10 + line_no * 18), line, fill="#334155", font=font(15))
        self.y += 390

    def save(self):
        self._footer()
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        self.pages[0].save(OUTPUT, save_all=True, append_images=self.pages[1:], resolution=150.0)


def build_report():
    report = Report()
    report.draw.text((MARGIN, 250), "Engineering a Structurally Aware", fill="#0f172a", font=font(48, True))
    report.draw.text((MARGIN, 315), "RAG System for Academic Literature", fill="#0f172a", font=font(48, True))
    report.draw.line((MARGIN, 405, PAGE_SIZE[0] - MARGIN, 405), fill="#2563eb", width=6)
    report.draw.text((MARGIN, 455), "Generative AI Fundamentals Assignment", fill="#334155", font=font(28))
    report.draw.text((MARGIN, 500), "First report draft - July 2026", fill="#64748b", font=font(23))
    report.y = 680
    report.heading("Executive summary")
    report.paragraph("This project implements a Retrieval-Augmented Generation application for five foundational research papers: Transformer, BERT, GPT-3, RoBERTa, and T5. The system preserves document structure during ingestion, retrieves evidence from a persistent Qdrant index, and generates grounded answers through a configurable language-model provider.")
    report.heading("Assignment coverage")
    report.bullets([
        "Five supplied PDFs are ingested and indexed.",
        "Semantic retrieval uses dense embeddings stored in Qdrant.",
        "A Gradio chat application maintains the latest four completed conversation turns as model memory.",
        "A 50-question test suite evaluates retrieval and answer quality.",
        "An open-source local Ollama model is supported, alongside Groq, Google, and OpenAI providers.",
    ])
    report.callout(
        "What makes this project different",
        "Instead of treating academic PDFs as plain text, the ingestion pipeline preserves layout-derived structure, headings, page provenance, and neighboring chunk relationships. This gives retrieval and answer generation richer evidence than a flat character splitter.",
        fill="#eef6ff",
    )
    report.callout(
        "Evidence-led engineering",
        "The report uses recorded chunk-size, encoder, retrieval, and visualization experiments to justify the selected configuration. Results with different test-set sizes are explicitly labelled as directional rather than directly comparable.",
        fill="#f0fdf4",
        accent="#16a34a",
    )
    report.heading("Report roadmap")
    report.bullets([
        "System architecture: how structured documents become grounded answers.",
        "Design decisions: why Docling, HybridChunker, BGE-base, Qdrant, reranking, and configurable generation were selected.",
        "Experiments: how chunk size, preprocessing, and encoder choices were measured.",
        "Evaluation and next steps: what the current results show and what remains before final submission.",
    ])

    report.new_page()
    report.title("1. System architecture and implementation")
    report.architecture()
    report.callout(
        "How to read the architecture",
        "The top lane prepares the corpus once: PDFs become cleaned, heading-aware chunks with provenance and 768D BGE-base vectors. The lower lane runs for every question: it applies four-turn memory, retrieves evidence, reranks it, expands adjacent chunks, and asks the configured LLM to answer only from the supplied context.",
        fill="#eff6ff",
    )
    report.paragraph("Docling first reconstructs reading order, tables, formulas, and hierarchical document items. The cleaning stage removes recurring publication noise while retaining academic content. HybridChunker then produces token-safe chunks that retain heading context; metadata records the source paper, section, page, and chunk order.")
    report.paragraph("At runtime, Qdrant returns dense semantic candidates and a text-match candidate path. The candidate set is deduplicated and passed to a cross-encoder, which jointly reads the question and candidate text. The highest-ranked chunks are expanded with N-1 and N+1 chunks from the same paper before grounded answer generation.")

    report.new_page()
    report.title("2. Design decisions")
    report.table(
        ["Component", "Selected approach", "Reason for selection"],
        [
            ("PDF parsing", "Docling", "Preserves layout, headings, tables, formulas, and reading order in academic PDFs."),
            ("Cleaning", "Regex + BeautifulSoup", "Removes recurring headers, footers, email addresses, and publication boilerplate."),
            ("Chunking", "Docling HybridChunker", "Uses document structure and token limits rather than arbitrary character boundaries."),
            ("Embeddings", "BAAI/bge-base-en-v1.5", "Outperformed MiniLM in recorded 500-token evaluation comparisons."),
            ("Vector store", "Qdrant", "Persistent local storage with metadata filtering and dense-vector retrieval."),
            ("Reranking", "ms-marco MiniLM cross-encoder", "Scores query and candidate text jointly to reduce semantic false positives."),
            ("Generation", "Configurable LiteLLM provider", "Supports local Ollama and hosted Groq, Google, or OpenAI models."),
        ],
        [260, 330, 844],
    )
    report.heading("Conversation memory")
    report.paragraph("The application retains the latest four completed user-assistant exchanges before prompting the model. The visible Gradio transcript remains intact; only model memory is bounded.")
    report.heading("Why these components work together")
    report.paragraph("The selected components divide responsibility cleanly. Docling handles document understanding, HybridChunker protects semantic boundaries, BGE-base represents chunk meaning, Qdrant performs efficient candidate retrieval, the cross-encoder improves precision, and the language model converts retrieved evidence into a readable answer. This separation makes each stage testable and replaceable.")
    report.heading("Current scope")
    report.paragraph("The current retrieval implementation is dense-vector retrieval plus a Qdrant text-match candidate path, followed by cross-encoder reranking. It is not a true BM25 sparse-vector index or reciprocal-rank-fusion implementation; this distinction is preserved in the final report for accuracy.")
    report.callout(
        "Configurability",
        "The answer-generation provider is controlled in settings.py. Ollama supports a local open-source model; Groq, Google, and OpenAI are optional hosted providers. This allows the deployment choice to balance privacy, quality, speed, and cost without changing retrieval code.",
        fill="#fffbeb",
        accent="#d97706",
    )
    report.heading("End-to-end flow in plain language")
    report.bullets([
        "A research PDF is converted into structured content, then cleaned so repeating publication artefacts do not dominate the embedding space.",
        "HybridChunker creates chunks that keep their section context; each chunk is embedded and stored with provenance metadata for later inspection.",
        "A user question retrieves likely evidence, then the reranker promotes chunks that best answer that exact question rather than merely sharing similar words.",
        "The generator receives the selected evidence and bounded conversation memory, producing a response grounded in the indexed papers rather than a generic answer.",
    ])

    report.new_page()
    report.title("3. Experimental methodology")
    report.paragraph("Experiments varied preprocessing, splitter style, chunk size, and embedding model. Retrieval was evaluated using Mean Reciprocal Rank (MRR), normalized Discounted Cumulative Gain (nDCG), and keyword coverage. Answer quality was evaluated with an LLM-as-a-judge on accuracy, completeness, and relevance.")
    report.table(
        ["Experiment", "Controlled variable", "Evidence artifact"],
        [
            ("Initial chunking", "200, 500, and 1,000-character chunks", "docs/graphs/initial_tests/embedding_vis_*.pdf"),
            ("Character sweep", "500 to 3,300 cleaned Markdown characters", "docs/eval_tests/all-MiniLM-L6-v2/eval_clean_md_char_*.pdf"),
            ("Token sweep", "333, 450, 500, and 1,000 tokens", "docs/eval_tests/all-MiniLM-L6-v2/eval_clean_md_tokens_*.pdf"),
            ("Encoder comparison", "MiniLM 384D versus BGE-base 768D", "docs/graphs/evals and docs/eval_tests"),
            ("Retrieval improvements", "Reranking and neighbor context expansion", "docs/eval_tests/*/eval_500t_hybrid_*.pdf"),
        ],
        [310, 430, 694],
    )
    report.heading("Comparison caution")
    report.paragraph("Not all saved dashboards use the same question count: early experiments use 15 or 29 tests, while the later evaluation set contains 50 tests. Results from different test-set sizes are presented as directional evidence, not direct head-to-head benchmarks.", color="#7c2d12")
    report.heading("Experiment sequence")
    report.paragraph("The work began with visual inspection of embedding distributions for character chunks of 200, 500, and 1,000 units. It then compared raw and cleaned Markdown, character and token chunk sizes, 384D MiniLM and 768D BGE-base embeddings, and finally retrieval enhancements such as reranking and neighboring-context expansion.")
    report.callout(
        "Evaluation principle",
        "A stronger system should retrieve evidence earlier in the ranked list, cover the terminology required by a question, and generate an answer that is accurate, complete, and relevant. The report therefore combines retrieval metrics with answer-quality judging rather than relying on a single score.",
        fill="#f0fdf4",
        accent="#16a34a",
    )

    report.new_page()
    report.title("4. Retrieval experiment results")
    report.table(
        ["Configuration", "Tests", "MRR", "nDCG", "Coverage"],
        [
            ("MiniLM, clean Markdown, 500 characters", "50", "0.4487", "0.5200", "73.3%"),
            ("MiniLM, clean Markdown, 2,500 characters", "50", "0.6091", "0.6329", "72.8%"),
            ("MiniLM, clean Markdown, 500 tokens", "50", "0.6254", "0.6451", "91.7%"),
            ("MiniLM, clean Markdown, 1,000 tokens", "50", "0.6243", "0.6625", "86.5%"),
            ("MiniLM, optimized retrieval and expansion", "50", "0.7504", "0.7570", "89.3%"),
            ("BGE-base, optimized retrieval and expansion", "50", "0.7755", "0.7768", "89.0%"),
        ],
        [520, 120, 120, 120, 180],
    )
    report.chart(
        ["500 char", "2500 char", "500 token", "1000 token", "MiniLM opt", "BGE opt"],
        [0.4487, 0.6091, 0.6254, 0.6243, 0.7504, 0.7755],
        "MRR progression across selected 50-question experiments",
    )
    report.heading("Interpretation")
    report.bullets([
        "Small character chunks underperform because they fragment technical arguments and section context.",
        "Token-aware chunks around 500 tokens provide a strong balance between rank quality and keyword coverage.",
        "Cross-encoder reranking and adjacent-chunk expansion produced the largest observed retrieval improvement.",
        "BGE-base achieved the best recorded final MRR and nDCG, supporting its selection as the current encoder.",
    ])
    report.callout(
        "Selection decision",
        "The project now uses BAAI/bge-base-en-v1.5 with 768-dimensional vectors. In the recorded 50-question optimized run, it achieved MRR 0.7755 and nDCG 0.7768, improving on the optimized MiniLM run at 0.7504 and 0.7570 respectively. Keyword coverage remained effectively stable at about 89 percent.",
        fill="#eef6ff",
    )

    report.new_page()
    report.title("5. Evaluation, limitations, and next steps")
    report.heading("Evaluation framework")
    report.paragraph("The evaluation dashboard runs all questions in src/evaluation/tests.jsonl. Retrieval metrics are MRR, nDCG, and keyword coverage. Answer evaluation uses an LLM judge to score accuracy, completeness, and relevance from 1 to 5. The project therefore uses a documented custom evaluation approach rather than RAGAS.")
    report.paragraph("The later test suite contains 50 questions across the five papers and covers direct facts, definitions, mechanisms, motivations, and multi-hop questions. This breadth is greater than the assignment requirement of ten questions and provides a stronger basis for analysing retrieval behavior.")
    report.heading("Recorded early answer-quality result")
    report.paragraph("One early 29-question dashboard reported accuracy 3.97/5, completeness 3.97/5, and relevance 3.79/5. These values are included as historical evidence only; a final run with the current configuration is required before submission.")
    report.heading("Limitations")
    report.bullets([
        "The text-match candidate path is not equivalent to a BM25 sparse-vector index or reciprocal-rank fusion.",
        "Some saved experiments differ in question count and should not be treated as strictly comparable.",
        "Local Ollama generation can be slower on CPU; model provider, timeout, and output length are configurable.",
        "Real-time feedback-based memory refinement remains a bonus extension, not a completed feature.",
    ])
    report.heading("Before final submission")
    report.bullets([
        "Recreate and ingest the Qdrant collection with the BGE-base 768D configuration.",
        "Run and capture the final 50-question evaluation with the selected deployment configuration.",
        "Add final dashboard screenshots, updated metrics, and any BM25/RRF decision to this report.",
        "Verify setup instructions and ensure no API secrets are included in the submission.",
    ])
    report.callout(
        "Report status",
        "This version is a structured evidence draft. It is ready for review and presentation, but final submission should replace historical screenshots with a fresh evaluation using the current BGE-base configuration and selected LLM provider.",
        fill="#fffbeb",
        accent="#d97706",
    )
    report.heading("Conclusion")
    report.paragraph("The project demonstrates an end-to-end RAG workflow for structurally complex academic literature. Measured experiments support structure-aware preprocessing, 500-token-scale chunks, BGE-base embeddings, cross-encoder reranking, and neighbor-context expansion as the strongest current configuration.")
    report.save()
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    build_report()
