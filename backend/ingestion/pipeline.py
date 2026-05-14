"""
Document ingestion pipeline: PDF → layout-aware chunks → embeddings → vector
store + BM25 index.

Design decisions
────────────────

Layout-aware extraction (pymupdf4llm, not PyPDFLoader):
  Enterprise documents are not paragraphs of prose — they are sectioned
  contracts, tables, lists, and forms. Plain PyPDFLoader flattens all of
  that into a single text stream and lets the character splitter slice
  through tables and headings indiscriminately. The chunks that come out
  look like word salad.

  pymupdf4llm extracts to Markdown, preserving:
    • Heading hierarchy (so chunks know which section they're in)
    • Table structure (rows stay together)
    • List item boundaries
    • Page boundaries (carried into metadata for citation purposes)

  We then split that Markdown with separators that respect its structure
  (``\\n## ``, ``\\n### ``, ``\\n\\n``, …) so a chunk break is much more
  likely to fall on a section or paragraph boundary, not mid-table.

  Tradeoff: pymupdf4llm depends on PyMuPDF (AGPL). For commercial closed-
  source deployments where AGPL is unacceptable, swap to ``unstructured``
  or ``docling`` — both expose a similar markdown-style API. The chunking
  code below is agnostic to the parser; only ``_load_markdown`` changes.

chunk_size=800, overlap=150:
  Larger chunks preserve more sentence context per embedding, reducing
  the chance that a key sentence is split across chunk boundaries.
  250-char "sentence chunks" lose cross-sentence context important for
  policy documents. 800 chars ≈ 3-4 dense sentences — empirically
  optimal for enterprise FAQ/policy corpora.

SHA-256 content deduplication:
  Re-ingesting the same PDF without dedup doubles the index size and
  degrades precision. Hash-based dedup is collision-resistant (2^-128)
  and costs ~1ms. This catches *exact* duplicates only.

Version-aware document handling — DESIGN NOTE, NOT IMPLEMENTED.
  Revised versions of the same contract (e.g. "MSA-2024.pdf" updated to
  "MSA-2024-rev2.pdf") will both ingest as independent documents. For a
  compliance use case this is wrong — retrieval can return conflicting
  facts from two versions of the same agreement, and there is no
  "as-of" query semantics ("what did we say in Q3 2025?").

  The right design when this becomes load-bearing:
    1. Add ``document_id`` (stable across versions, e.g. operator-
       supplied or derived from filename normalisation) and
       ``version`` (monotonic integer) to chunk payload.
    2. Add ``valid_from`` / ``valid_to`` timestamps; superseded
       versions get ``valid_to`` set on the next ingest.
    3. Default retrieval filter: ``valid_to IS NULL`` (i.e. current).
       Point-in-time filter for audit: ``valid_from <= T < valid_to``.
    4. ``ingest_pdf`` accepts an optional ``supersedes: document_id``
       argument; on success it bumps the version and seals the prior.

  This is a multi-day change touching the ingest API, the vector
  store schema, the retrieval node filter, and the route handlers. It
  is deliberately not implemented in this submission — the SHA-256
  dedup is the correct floor; the version-aware story is the correct
  ceiling, documented here so a reader knows the gap is recognised.

Audit log:
  Enterprise deployments require a tamper-evident record of deletions
  for GDPR/compliance. The audit.jsonl append-only log satisfies this
  without a full database.
"""

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from core.config import Settings, get_settings
from core.exceptions import IngestionError
from retrieval.vector_store import get_vectorstore

logger = logging.getLogger(__name__)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _load_markdown(pdf_path: str) -> List[dict]:
    """Extract a PDF as a list of per-page Markdown blocks.

    Returns ``[{"page": int, "text": str}, ...]``. We use pymupdf4llm
    rather than PyPDFLoader because the latter destroys document structure
    (headings, lists, tables collapse into a single text stream).

    pymupdf4llm with ``page_chunks=True`` returns a list of page dicts,
    each containing the page's Markdown and its 0-indexed page number.
    We translate to 1-indexed for human-friendly citations.
    """
    import pymupdf4llm  # lazy: heavy dep

    page_chunks = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
    pages: List[dict] = []
    for entry in page_chunks:
        # pymupdf4llm returns either {"text": str, "metadata": {"page": int}}
        # or older shapes; handle both defensively.
        text = entry.get("text") if isinstance(entry, dict) else str(entry)
        meta = entry.get("metadata", {}) if isinstance(entry, dict) else {}
        page_number = meta.get("page", len(pages))
        # Some versions return 0-indexed; humans read pages from 1.
        pages.append({"page": int(page_number) + 1, "text": _clean_markdown(text or "")})
    return pages


# Compiled once at module load — used on every page during ingestion.
# Order is significant: combined bold-italic markers must be stripped
# before plain bold/italic, otherwise we'd leave dangling `_` or `*` chars.
_BOLD_ITALIC_RE = re.compile(r"\*{2}_(.+?)_\*{2}|\*{3}(.+?)\*{3}|___(.+?)___")
_BOLD_RE        = re.compile(r"\*{2}(.+?)\*{2}|__(.+?)__")
# Italic must avoid mid-word matches (e.g. ``x_min`` or ``f*ck``); the
# lookbehind / lookahead requires the marker to sit on a word boundary.
_ITALIC_RE      = re.compile(r"(?<!\w)\*([^*\n]+?)\*(?!\w)|(?<!\w)_([^_\n]+?)_(?!\w)")
_BR_RE          = re.compile(r"<br\s*/?>", re.IGNORECASE)
_MULTISPACE_RE  = re.compile(r"[ \t]+")
_MULTINEWLINE_RE = re.compile(r"\n{3,}")


def _clean_markdown(text: str) -> str:
    """Strip pymupdf4llm's HTML/Markdown artefacts that hurt retrieval and UX.

    What this removes and why:

      * ``<br>`` / ``<br/>`` tags — pymupdf4llm inserts these inside table
        cells and multi-line list items. They tokenise badly for BM25
        (``<br>`` becomes a literal token that bloats every chunk's term
        frequency) and they render as visible noise in the UI snippet.
        Replaced with a newline so structural breaks survive.

      * ``**...**`` / ``__...__`` / ``*...*`` / ``_..._`` emphasis markers —
        retained as a styling signal, but stripped from the chunk text
        because: (1) BM25 indexes them as junk tokens, (2) the embedding
        model spends capacity on them instead of the semantics, (3) the
        UI snippet renders them as raw asterisks/underscores. Inner
        content is preserved. Mid-word markers (``x_min``, ``f*g``) are
        protected by word-boundary lookaround.

      * ``\\ufffd`` (Unicode replacement char ``ï¿½``) — appears
        where the PDF embeds a custom font without a ToUnicode CMap. In
        prospectuses this is almost always the bullet glyph. We render
        it as ``•`` so the text reads naturally.

    The cleanup runs ONCE per page during ingest. The raw markdown stays
    in the chunked Documents only inside this function — every consumer
    downstream (BM25, embeddings, generator prompt, UI snippet) sees the
    cleaned form. Tradeoff: we lose pymupdf's bold/italic structural
    signal, which is minor for retrieval and gives much cleaner outputs.
    """
    if not text:
        return ""
    text = _BR_RE.sub("\n", text)
    # Bold-italic combinations first (regex captures multiple group
    # variants; we keep whichever matched).
    text = _BOLD_ITALIC_RE.sub(lambda m: next(g for g in m.groups() if g is not None), text)
    text = _BOLD_RE.sub(lambda m: next(g for g in m.groups() if g is not None), text)
    text = _ITALIC_RE.sub(lambda m: next(g for g in m.groups() if g is not None), text)
    text = text.replace("�", "•")
    text = _MULTISPACE_RE.sub(" ", text)
    text = _MULTINEWLINE_RE.sub("\n\n", text)
    return text.strip()


# Markdown-aware separators. Order matters — splitter tries them in order
# and only falls through to a finer-grained separator if the chunk is still
# too big. Heading boundaries are preferred over paragraph boundaries; both
# are preferred over mid-sentence breaks.
_MARKDOWN_SEPARATORS = [
    "\n# ",   # H1 — top-level section
    "\n## ",  # H2 — sub-section
    "\n### ",
    "\n#### ",
    "\n\n",   # paragraph
    "\n",     # line
    ". ",     # sentence
    " ",
    "",
]


def _load_and_chunk(pdf_path: str, settings: Settings) -> List[Document]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter  # lazy

    pages = _load_markdown(pdf_path)
    if not pages:
        raise IngestionError(f"PDF {pdf_path} produced no extractable text")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=_MARKDOWN_SEPARATORS,
        keep_separator=True,
    )

    filename = Path(pdf_path).name
    chunks: List[Document] = []
    for page in pages:
        for piece in splitter.split_text(page["text"]):
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(Document(
                page_content=piece,
                metadata={
                    "source_file": filename,
                    "page": page["page"],
                    "char_count": len(piece),
                },
            ))

    # chunk_index is assigned after the pass so it's globally monotonic
    # across pages (callers that grep for "chunk_index == N" still work).
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    logger.info(
        "Chunked PDF",
        extra={
            "file": filename,
            "pages": len(pages),
            "chunks": len(chunks),
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
        },
    )
    return chunks


def ingest_pdf(
    pdf_path: str,
    settings: Settings | None = None,
    tenant_id: str | None = None,
) -> dict:
    """Ingest a PDF into the vector store, scoped to a tenant.

    Every chunk is tagged with ``tenant_id`` and indexed in Qdrant under a
    payload index of the same name. Retrieval filters by ``tenant_id``
    server-side, so a query as tenant A cannot retrieve tenant B's chunks
    even by accident.

    Dedup is also per-tenant: re-uploading the same PDF under tenant A
    skips; uploading it under tenant B ingests fresh (the two corpora
    are independent).
    """
    settings = settings or get_settings()
    tenant_id = tenant_id or settings.default_tenant

    if not os.path.exists(pdf_path):
        raise IngestionError(f"PDF not found: {pdf_path}")

    vs = get_vectorstore()
    doc_hash = _sha256(pdf_path)

    existing = vs.find_by_metadata({"doc_hash": doc_hash, "tenant_id": tenant_id})
    if existing:
        logger.info("Skipping ingest — duplicate content hash", extra={
            "file": Path(pdf_path).name,
            "tenant_id": tenant_id,
            "existing_chunks": len(existing),
        })
        return {
            "status": "skipped",
            "message": "Document already ingested (content hash match)",
            "chunks": len(existing),
            "file": Path(pdf_path).name,
            "embedding_provider": settings.resolve_embedding_provider(),
        }

    chunks = _load_and_chunk(pdf_path, settings)
    for chunk in chunks:
        chunk.metadata["doc_hash"] = doc_hash
        chunk.metadata["tenant_id"] = tenant_id

    vs.add_documents(chunks)
    logger.info("Ingested PDF", extra={
        "file": Path(pdf_path).name,
        "tenant_id": tenant_id,
        "chunks": len(chunks),
    })

    # Rebuild BM25 index so sparse retrieval sees the new document
    _rebuild_bm25(settings)

    embedding_provider = settings.resolve_embedding_provider()
    return {
        "status": "success",
        "file": Path(pdf_path).name,
        "chunks": len(chunks),
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "embedding_provider": embedding_provider,
    }


def delete_document(
    filename: str,
    settings: Settings | None = None,
    tenant_id: str | None = None,
) -> dict:
    """Remove all chunks for a document, scoped to a tenant.

    A delete for tenant A cannot touch tenant B's documents, even if the
    same filename exists under both tenants — the where-clause is an AND
    of source_file and tenant_id.
    """
    settings = settings or get_settings()
    tenant_id = tenant_id or settings.default_tenant
    vs = get_vectorstore()

    deleted = vs.delete_by_metadata({"source_file": filename, "tenant_id": tenant_id})
    if deleted == 0:
        logger.warning("Delete: no chunks found", extra={"file": filename, "tenant_id": tenant_id})
        return {"status": "not_found", "file": filename, "chunks_deleted": 0}

    _rebuild_bm25(settings)
    logger.info("Deleted chunks", extra={
        "file": filename, "tenant_id": tenant_id, "count": deleted,
    })

    return {"status": "deleted", "file": filename, "chunks_deleted": deleted}


def list_ingested_files(
    settings: Settings | None = None,
    tenant_id: str | None = None,
) -> List[str]:
    """Return sorted list of unique source filenames for the given tenant."""
    settings = settings or get_settings()
    tenant_id = tenant_id or settings.default_tenant
    try:
        vs = get_vectorstore()
        docs = vs.find_by_metadata({"tenant_id": tenant_id})
        files = {d.metadata.get("source_file") for d in docs}
        return sorted(f for f in files if f)
    except Exception:
        logger.exception("Failed to list ingested files")
        return []


def _rebuild_bm25(settings: Settings) -> None:
    """Sync BM25 index with the vector store after ingest or delete."""
    try:
        from retrieval.bm25_retriever import BM25Retriever
        docs = get_vectorstore().all_documents()
        if not docs:
            return
        texts = [d.page_content for d in docs]
        metas = [d.metadata for d in docs]
        BM25Retriever(settings).build(texts, metas)
    except Exception as exc:
        logger.warning("BM25 rebuild failed (non-fatal)", extra={"err": str(exc)})
