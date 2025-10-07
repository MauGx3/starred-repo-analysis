"""Starred Repo Analysis.

Expose the main classes for easy imports in tests and downstream usage.

This package exports:
- RepositoryRecommender
- ProjectContext
- StarredRepoScanner
"""

from .repo_recommender import ProjectContext, RepositoryRecommender
from .scan_starred_repos import StarredRepoScanner

__all__ = ["ProjectContext", "RepositoryRecommender", "StarredRepoScanner"]
