from typing import List
from dataclasses import dataclass, field

@dataclass
class DocumentMetadata:
    source: str
    paper_name: str
    chunk_number: int
    parent_section: str
    page_number: int
    document_type: str = "Research Paper"
    keywords: List[str] = field(default_factory=list)
    all_headings: List[str] = field(default_factory=list)
    page_numbers: List[int] = field(default_factory=list) # Full range if it spans pages
    token_count: int = 0

@dataclass
class Document:
    content: str
    metadata: DocumentMetadata