import os
from pathlib import Path

import json

from starred_repo_analysis import RepositoryRecommender


def test_recommender_smoke(tmp_path: Path):
    """Smoke test: run recommender against a small sample dataset.

    The test uses the bundled sample JSON and ensures the method runs
    without raising. The environment may not have ML libs installed,
    so the test only asserts the return type and basic structure.
    """

    sample = Path(__file__).parent / "data" / "sample_starred_repos.json"
    assert sample.exists()

    recommender = RepositoryRecommender()

    # Run recommend; if dependencies for embeddings are missing the
    # recommender still returns a dict after scoring with fallbacks.
    recommendations = recommender.recommend(
        starred_repos_file=str(sample),
        project_path=".",
        top_n=5,
        min_score=0.0,
    )

    assert isinstance(recommendations, dict)
    # categories keys should be strings
    for k in recommendations.keys():
        assert isinstance(k, str)
