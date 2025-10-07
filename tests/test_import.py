"""Test Starred Repo Analysis imports and exposes expected names."""

from starred_repo_analysis import (
    RepositoryRecommender,
    ProjectContext,
    StarredRepoScanner,
)


def test_exports() -> None:
    """Package exports the main classes and they are importable."""
    assert callable(RepositoryRecommender)
    assert callable(ProjectContext)
    assert callable(StarredRepoScanner)
