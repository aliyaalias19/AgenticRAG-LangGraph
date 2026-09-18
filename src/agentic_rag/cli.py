"""Command-line entry points for corpus operations."""

import argparse
import sys

from agentic_rag.ingest.pipeline import ingest_corpus
from agentic_rag.obs.logging import configure_logging, get_logger

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Run the requested CLI command and return a process exit code."""
    parser = argparse.ArgumentParser(prog="agentic-rag")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("ingest", help="Clone and process the document corpus")

    args = parser.parse_args(argv)
    configure_logging()

    if args.command == "ingest":
        manifest = ingest_corpus()
        logger.info("ingest_command_finished", corpus_hash=manifest.corpus_hash)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
