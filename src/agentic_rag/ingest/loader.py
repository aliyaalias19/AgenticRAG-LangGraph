"""Load and filter markdown documents from a documentation tree."""

from pathlib import Path

import frontmatter
from tqdm import tqdm

from agentic_rag.ingest.models import Document
from agentic_rag.obs.logging import get_logger

logger = get_logger(__name__)

EXCLUDED_FILENAMES = frozenset({"_index.md", "_print"})
EXCLUDED_DIR_NAMES = frozenset({"includes", "_print"})


def _is_excluded(
    path: Path,
    docs_root: Path,
    excluded_prefixes: tuple[str, ...],
) -> bool:
    """Return True if the path should be skipped during ingestion."""
    if path.name in EXCLUDED_FILENAMES:
        return True

    relative = path.relative_to(docs_root)
    if any(part in EXCLUDED_DIR_NAMES for part in relative.parts):
        return True

    relative_str = relative.as_posix()
    return any(relative_str.startswith(prefix) for prefix in excluded_prefixes)


def _extract_title(post: frontmatter.Post, path: Path) -> str:
    """Return the document title from frontmatter, falling back to the filename."""
    raw_title = post.metadata.get("title")
    if isinstance(raw_title, str) and raw_title.strip():
        return raw_title.strip()
    return path.stem.replace("-", " ").title()


def _string_metadata(post: frontmatter.Post) -> dict[str, str]:
    """Return frontmatter entries whose values are plain strings."""
    return {key: value for key, value in post.metadata.items() if isinstance(value, str)}


def load_documents(
    docs_root: Path,
    min_chars: int,
    excluded_prefixes: tuple[str, ...] = (),
) -> list[Document]:
    """Load all eligible markdown documents beneath ``docs_root``."""
    if not docs_root.is_dir():
        message = f"Documentation root does not exist: {docs_root}"
        raise FileNotFoundError(message)

    markdown_paths = sorted(docs_root.rglob("*.md"))
    logger.info("markdown_files_discovered", count=len(markdown_paths))

    documents: list[Document] = []
    skipped_excluded = 0
    skipped_short = 0
    skipped_unparseable = 0

    for path in tqdm(markdown_paths, desc="Loading documents", unit="file"):
        if _is_excluded(path, docs_root, excluded_prefixes):
            skipped_excluded += 1
            continue

        try:
            post = frontmatter.load(path)
        except (UnicodeDecodeError, ValueError):
            skipped_unparseable += 1
            logger.warning("document_unparseable", path=str(path))
            continue

        content = post.content.strip()
        if len(content) < min_chars:
            skipped_short += 1
            continue

        relative_path = path.relative_to(docs_root)
        documents.append(
            Document(
                doc_id=str(relative_path.with_suffix("")),
                source_path=str(relative_path),
                title=_extract_title(post, path),
                content=content,
                content_hash=Document.compute_hash(content),
                char_count=len(content),
                section_path=list(relative_path.parent.parts),
                frontmatter=_string_metadata(post),
            )
        )

    logger.info(
        "documents_loaded",
        loaded=len(documents),
        skipped_excluded=skipped_excluded,
        skipped_short=skipped_short,
        skipped_unparseable=skipped_unparseable,
    )
    return documents
