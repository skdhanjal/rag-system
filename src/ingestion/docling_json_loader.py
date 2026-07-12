"""
docling_json_loader.py

Fast DoclingDocument loader that reads pre-parsed document structure back
from saved JSON, instead of re-running PDF layout analysis, OCR, table
recognition, and formula enrichment on every pipeline run.

Only responsibility of this module: get DoclingDocument objects into
memory from disk. No cleaning, no chunking -- that's other modules' jobs.

Requires PDFs to have already been parsed once (see parse_paper.py),
which calls doc.save_as_json(...) to produce one <paper_id>.docling.json
per paper alongside the markdown output.
"""

from pathlib import Path
from docling_core.types.doc.document import DoclingDocument

DOCLING_JSON_SUFFIX = ".docling.json"


def load_docling_document(doc_json_path) -> DoclingDocument:
    """Load a single DoclingDocument from its saved JSON representation."""
    return DoclingDocument.load_from_json(Path(doc_json_path))


def load_docling_documents_from_folder(folder_path, pattern: str = f"*{DOCLING_JSON_SUFFIX}"):
    """
    Read every saved DoclingDocument JSON file in a folder and return them
    as a plain list of DoclingDocument objects -- the same shape a
    PDF-parsing loader would typically hand back, so a downstream cleaning
    pipeline needs no changes to accept it.

    Each returned document carries its own paper_id via doc.name (set
    automatically by Docling during the original conversion, e.g. a PDF
    named "1706.03762.pdf" becomes doc.name == "1706.03762") -- no need
    to track filenames alongside the objects separately.
    """
    folder_path = Path(folder_path)
    return [load_docling_document(p) for p in sorted(folder_path.glob(pattern))]


class DoclingJsonLoader:
    """Drop-in replacement for a loader that parses PDFs directly, e.g.:

        self.loader = DoclingJsonLoader()   # instead of the slow PDF-parsing loader

    Matches the load_directory(directory_path, extract_images, output_image_dir)
    interface so no other pipeline code needs to change. extract_images /
    output_image_dir are accepted for signature compatibility but are
    no-ops here: images were already extracted to disk when the PDFs were
    originally parsed (see parse_paper.py) -- that prior work is exactly
    what this loader exists to avoid repeating.
    """

    def load_directory(self, directory_path, extract_images: bool = True, output_image_dir=None):
        if extract_images or output_image_dir is not None:
            print(
                "DoclingJsonLoader: extract_images/output_image_dir are ignored -- "
                "images were already extracted when these PDFs were originally parsed."
            )
        return load_docling_documents_from_folder(directory_path)