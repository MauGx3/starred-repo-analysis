"""Test Starred Repo Analysis."""

import starred_repo_analysis


def test_import() -> None:
    """Test that the package can be imported."""
    assert isinstance(starred_repo_analysis.__name__, str)
