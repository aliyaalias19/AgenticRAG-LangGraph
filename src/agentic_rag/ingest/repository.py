"""Clone and pin the source documentation repository."""

from pathlib import Path

from git import Repo

from agentic_rag.obs.logging import get_logger

logger = get_logger(__name__)


def clone_or_update(repo_url: str, ref: str, destination: Path) -> str:
    """Clone the repository if absent, otherwise fetch updates.

    Returns the resolved commit SHA of the checked-out ref.
    """
    if (destination / ".git").is_dir():
        logger.info("repository_found", path=str(destination))
        repo = Repo(destination)
        repo.remotes.origin.fetch()
    else:
        logger.info("repository_cloning", url=repo_url, ref=ref)
        destination.parent.mkdir(parents=True, exist_ok=True)
        repo = Repo.clone_from(
            url=repo_url,
            to_path=destination,
            branch=ref,
            depth=1,
            single_branch=True,
        )

    commit_sha = repo.head.commit.hexsha
    logger.info("repository_ready", commit_sha=commit_sha, ref=ref)
    return commit_sha
