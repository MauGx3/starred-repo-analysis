"""Starred Repo Analysis.

Expose the main classes for easy imports in tests and downstream usage.

This package exports:
- RepositoryRecommender
- ProjectContext
- StarredRepoScanner
"""

from starred_repo_analysis.repo_recommender import (
    ProjectContext,
    RepositoryRecommender,
)
from starred_repo_analysis.scan_starred_repos import StarredRepoScanner

__all__ = ["ProjectContext", "RepositoryRecommender", "StarredRepoScanner"]
