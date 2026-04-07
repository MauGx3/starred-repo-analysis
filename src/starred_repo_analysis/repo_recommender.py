#!/usr/bin/env python3
"""
Repository Recommendation Engine.

This module provides AI-powered repository recommendations by analyzing
project context and matching against starred repositories using semantic
understanding and multi-factor scoring.

Usage:
    from repo_recommender import RepositoryRecommender
    recommender = RepositoryRecommender()
    recommendations = recommender.recommend(project_path=".")
"""

# Postpone evaluation of type annotations so optional runtime dependencies
# (like numpy) don't cause import-time errors when used in annotations
# (for example: "np.ndarray | None" when `np` is None). This keeps the
# module importable in CI runners that don't have optional ML packages.
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar

UTC = timezone.utc
LOGGER = logging.getLogger(__name__)

DESCRIPTION_MAX_CHARS = 200
DEFAULT_REPORT_PREVIEW_CHARS = 1000
POPULAR_STARS_HIGH = 5000
POPULAR_STARS_MEDIUM = 1000
RECENT_DAYS_1_MONTH = 30
RECENT_DAYS_3_MONTHS = 90
RECENT_DAYS_6_MONTHS = 180
RECENT_DAYS_1_YEAR = 365
DIRECT_DEPENDENCY_THRESHOLD = 0.7
REFERENCE_IMPL_THRESHOLD = 0.6
SEMANTIC_SIMILARITY_HIGH = 0.7

try:
    import numpy as np
except ImportError:
    LOGGER.warning(
        "Warning: numpy not installed.\n"
        "Python executable: %s\n"
        "Install with: %s -m pip install numpy",
        sys.executable,
        sys.executable,
    )
    np = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    LOGGER.warning(
        "Warning: sentence-transformers not installed.\n"
        "Python executable: %s\n"
        "Install with: %s -m pip install sentence-transformers",
        sys.executable,
        sys.executable,
    )
    SentenceTransformer = None

try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    LOGGER.warning(
        "Warning: scikit-learn not installed.\n"
        "Python executable: %s\n"
        "Install with: %s -m pip install scikit-learn",
        sys.executable,
        sys.executable,
    )
    cosine_similarity = None


@dataclass
class RecommendationScore:
    """Scoring breakdown for a repository recommendation."""

    semantic_score: float  # 0-1: Semantic similarity
    tech_stack_score: float  # 0-1: Technology stack match
    topic_score: float  # 0-1: Topic overlap
    popularity_score: float  # 0-1: Normalized popularity
    recency_score: float  # 0-1: Recent activity
    composite_score: float  # 0-100: Weighted total
    reasoning: list[str]  # Human-readable reasons


@dataclass
class RepositoryRecommendation:
    """A recommended repository with scoring details."""

    name: str
    owner: str
    url: str
    description: str
    category: str
    score: RecommendationScore
    metadata: dict


class ProjectContext:
    """Extract and represent project context for matching."""

    def __init__(self, project_path: str = ".") -> None:
        self.project_path = Path(project_path)
        self.readme_content = ""
        self.languages = []
        self.frameworks = []
        self.topics = []
        self.dependencies = []
        self.description = ""

    def extract(self) -> ProjectContext:
        """Extract project context from multiple sources."""
        self._extract_readme()
        self._extract_package_info()
        self._detect_languages()
        return self

    def _extract_readme(self) -> None:
        """Extract information from a README file."""
        readme_patterns = [
            "README.md",
            "README.MD",
            "readme.md",
            "README.rst",
            "README.txt",
            "README",
        ]

        for pattern in readme_patterns:
            readme_path = self.project_path / pattern
            if readme_path.exists():
                try:
                    with readme_path.open(encoding="utf-8") as file_handle:
                        self.readme_content = file_handle.read()
                    self._parse_readme_content()
                    break
                except (OSError, UnicodeDecodeError) as error:
                    LOGGER.warning("Could not read README: %s", error)

    def _parse_readme_content(self) -> None:  # noqa: C901
        """Parse README to extract key information."""
        if not self.readme_content:
            return

        # Extract description (usually first paragraph after title)
        lines = self.readme_content.split("\n")
        desc_lines = []
        skip_title = True

        for raw_line in lines:
            stripped_line = raw_line.strip()
            if not stripped_line or stripped_line.startswith("#"):
                if desc_lines:
                    break
                continue
            if skip_title and stripped_line:
                skip_title = False
            if not skip_title and stripped_line:
                desc_lines.append(stripped_line)
                if len(" ".join(desc_lines)) > DESCRIPTION_MAX_CHARS:
                    break

        self.description = " ".join(desc_lines)

        # Extract technologies mentioned
        tech_keywords = {
            "languages": [
                "python",
                "javascript",
                "typescript",
                "java",
                "go",
                "rust",
                "ruby",
                "php",
                "c++",
                "c#",
                "swift",
                "kotlin",
            ],
            "frameworks": [
                "react",
                "vue",
                "angular",
                "django",
                "flask",
                "express",
                "fastapi",
                "spring",
                "rails",
                "laravel",
                "next.js",
                "nest.js",
            ],
        }

        content_lower = self.readme_content.lower()
        for lang in tech_keywords["languages"]:
            if lang in content_lower:
                self.languages.append(lang)

        for framework in tech_keywords["frameworks"]:
            if framework in content_lower:
                self.frameworks.append(framework)

    def _extract_package_info(self) -> None:
        """Extract dependencies from package files."""
        req_file = self.project_path / "requirements.txt"
        if req_file.exists():
            try:
                with req_file.open(encoding="utf-8") as file_handle:
                    for raw_line in file_handle:
                        stripped_line = raw_line.strip()
                        if stripped_line and not stripped_line.startswith("#"):
                            # Extract package name
                            pkg = re.split(r"[>=<\[]", stripped_line)[
                                0
                            ].strip()
                            if pkg:
                                self.dependencies.append(pkg)
            except (OSError, UnicodeDecodeError) as error:
                LOGGER.warning("Could not read requirements.txt: %s", error)

        pkg_file = self.project_path / "package.json"
        if pkg_file.exists():
            try:
                with pkg_file.open(encoding="utf-8") as file_handle:
                    pkg_data = json.load(file_handle)
                    self.description = self.description or pkg_data.get(
                        "description", ""
                    )

                    # Extract dependencies
                    deps = pkg_data.get("dependencies", {})
                    dev_deps = pkg_data.get("devDependencies", {})
                    self.dependencies.extend(list(deps.keys()))
                    self.dependencies.extend(list(dev_deps.keys()))

                    # Extract keywords as topics
                    keywords = pkg_data.get("keywords", [])
                    self.topics.extend(keywords)
            except (
                OSError,
                UnicodeDecodeError,
                json.JSONDecodeError,
            ) as error:
                LOGGER.warning("Could not read package.json: %s", error)

    def _detect_languages(self) -> None:
        """Detect programming languages from file extensions."""
        if self.languages:
            return  # Already detected from README

        lang_extensions = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".java": "java",
            ".go": "go",
            ".rs": "rust",
            ".rb": "ruby",
            ".php": "php",
            ".cpp": "c++",
            ".cs": "c#",
            ".swift": "swift",
            ".kt": "kotlin",
        }

        lang_counts = defaultdict(int)
        try:
            for file_path in self.project_path.rglob("*"):
                if file_path.is_file():
                    ext = file_path.suffix.lower()
                    if ext in lang_extensions:
                        lang_counts[lang_extensions[ext]] += 1
        except OSError as error:
            LOGGER.warning("Could not scan directory: %s", error)

        # Take top 3 languages by file count
        sorted_langs = sorted(
            lang_counts.items(), key=lambda x: x[1], reverse=True
        )
        self.languages = [lang for lang, _ in sorted_langs[:3]]

    def get_text_representation(self) -> str:
        """Get a text representation for semantic embedding."""
        parts = []

        if self.description:
            parts.append(self.description)

        if self.languages:
            parts.append(f"Languages: {', '.join(self.languages)}")

        if self.frameworks:
            parts.append(f"Frameworks: {', '.join(self.frameworks)}")

        if self.topics:
            parts.append(f"Topics: {', '.join(self.topics[:5])}")

        return " | ".join(parts) if parts else "No description available"


class RepositoryRecommender:
    """AI-powered repository recommendation engine."""

    # Scoring weights
    WEIGHTS: ClassVar[dict[str, float]] = {
        "semantic": 0.35,
        "tech_stack": 0.25,
        "topic": 0.20,
        "popularity": 0.10,
        "recency": 0.10,
    }

    # Category thresholds and keywords
    CATEGORIES: ClassVar[dict[str, dict[str, list[str] | float]]] = {
        "direct_dependency": {
            "keywords": ["library", "package", "module", "sdk", "api-client"],
            "threshold": 0.7,
        },
        "tool_utility": {
            "keywords": [
                "cli",
                "tool",
                "utility",
                "command-line",
                "automation",
            ],
            "threshold": 0.6,
        },
        "reference_implementation": {
            "keywords": [
                "example",
                "sample",
                "template",
                "boilerplate",
                "starter",
            ],
            "threshold": 0.5,
        },
        "learning_resource": {
            "keywords": [
                "tutorial",
                "guide",
                "learning",
                "course",
                "documentation",
                "awesome",
            ],
            "threshold": 0.4,
        },
    }

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Initialize recommender with ML model.

        Args:
            model_name: Sentence transformer model name
                       (default: all-MiniLM-L6-v2 - fast and efficient)
        """
        self.model = None
        self.model_name = model_name

        if SentenceTransformer is not None:
            try:
                LOGGER.info("Loading model: %s...", model_name)
                self.model = SentenceTransformer(model_name)
                LOGGER.info("Model loaded successfully")
            except (RuntimeError, ValueError, OSError) as error:
                LOGGER.warning("Could not load model: %s", error)
                LOGGER.warning("Semantic scoring will be disabled")

    def recommend(
        self,
        starred_repos_file: str,
        project_path: str = ".",
        top_n: int = 20,
        min_score: float = 30.0,
    ) -> dict[str, list[RepositoryRecommendation]]:
        """Generate repository recommendations.

        Args:
            starred_repos_file: Path to starred repos JSON
            project_path: Path to project for analysis
            top_n: Number of top recommendations per category
            min_score: Minimum composite score (0-100)

        Returns
        -------
            Dictionary mapping categories to recommendations
        """
        # Extract project context
        LOGGER.info("Analyzing project context...")
        context = ProjectContext(project_path).extract()
        LOGGER.info("Project: %s...", context.description[:100])
        LOGGER.info("Languages: %s", ", ".join(context.languages))
        LOGGER.info("Frameworks: %s", ", ".join(context.frameworks))

        # Load starred repositories
        LOGGER.info(
            "Loading starred repositories from %s...", starred_repos_file
        )
        with Path(starred_repos_file).open(encoding="utf-8") as file_handle:
            data = json.load(file_handle)
            repos = data.get("repositories", [])
        LOGGER.info("Loaded %s repositories", len(repos))

        # Generate embeddings if model available
        project_embedding = None
        repo_embeddings = None

        if self.model is not None:
            LOGGER.info("Generating semantic embeddings...")
            project_text = context.get_text_representation()
            project_embedding = self.model.encode([project_text])[0]

            # Generate embeddings for repositories
            repo_texts = [self._get_repo_text(repo) for repo in repos]
            repo_embeddings = self.model.encode(repo_texts)
            LOGGER.info("Embeddings generated")

        # Score all repositories
        LOGGER.info("Scoring repositories...")
        recommendations = []
        for i, repo in enumerate(repos):
            score = self._score_repository(
                repo,
                context,
                project_embedding,
                repo_embeddings[i] if repo_embeddings is not None else None,
            )

            if score.composite_score >= min_score:
                category = self._categorize_repository(repo, score)
                recommendations.append(
                    RepositoryRecommendation(
                        name=repo["name"],
                        owner=repo["owner"],
                        url=repo["github_url"],
                        description=repo.get("description", ""),
                        category=category,
                        score=score,
                        metadata=repo,
                    )
                )

        # Group by category and sort
        categorized = defaultdict(list)
        for rec in recommendations:
            categorized[rec.category].append(rec)

        # Sort each category by score and limit to top_n
        for category in categorized:
            categorized[category].sort(
                key=lambda x: x.score.composite_score, reverse=True
            )
            categorized[category] = categorized[category][:top_n]

        LOGGER.info("Generated %s recommendations", len(recommendations))
        for category, recs in categorized.items():
            LOGGER.info("  %s: %s", category, len(recs))

        return dict(categorized)

    def _get_repo_text(self, repo: dict) -> str:
        """Get text representation of repository for embedding."""
        parts = []

        desc = repo.get("enhanced_description") or repo.get("description", "")
        if desc:
            parts.append(desc)

        if repo.get("language"):
            parts.append(f"Language: {repo['language']}")

        topics = repo.get("topics", [])
        if topics:
            parts.append(f"Topics: {', '.join(topics[:5])}")

        return " | ".join(parts) if parts else "No description"

    def _score_repository(
        self,
        repo: dict,
        context: ProjectContext,
        project_embedding: np.ndarray | None,
        repo_embedding: np.ndarray | None,
    ) -> RecommendationScore:
        """Calculate comprehensive score for a repository."""
        reasoning = []

        # 1. Semantic similarity (0-1)
        semantic_score = 0.0
        if (
            project_embedding is not None
            and repo_embedding is not None
            and cosine_similarity is not None
        ):
            semantic_score = float(
                cosine_similarity([project_embedding], [repo_embedding])[0][0]
            )
            if semantic_score > SEMANTIC_SIMILARITY_HIGH:
                reasoning.append(
                    f"High semantic similarity ({semantic_score:.2f})"
                )

        # 2. Technology stack matching (0-1)
        tech_score = self._calculate_tech_stack_score(repo, context, reasoning)

        # 3. Topic overlap (0-1)
        topic_score = self._calculate_topic_score(repo, context, reasoning)

        # 4. Popularity score (0-1)
        popularity_score = self._calculate_popularity_score(repo, reasoning)

        # 5. Recency score (0-1)
        recency_score = self._calculate_recency_score(repo, reasoning)

        # Calculate weighted composite score (0-100)
        composite = (
            semantic_score * self.WEIGHTS["semantic"]
            + tech_score * self.WEIGHTS["tech_stack"]
            + topic_score * self.WEIGHTS["topic"]
            + popularity_score * self.WEIGHTS["popularity"]
            + recency_score * self.WEIGHTS["recency"]
        ) * 100

        return RecommendationScore(
            semantic_score=semantic_score,
            tech_stack_score=tech_score,
            topic_score=topic_score,
            popularity_score=popularity_score,
            recency_score=recency_score,
            composite_score=composite,
            reasoning=reasoning,
        )

    def _calculate_tech_stack_score(
        self, repo: dict, context: ProjectContext, reasoning: list[str]
    ) -> float:
        """Calculate technology stack matching score."""
        score = 0.0
        matches = []

        # Language match
        repo_lang = repo.get("language") or ""
        repo_lang = repo_lang.lower() if repo_lang else ""
        context_langs = [lang.lower() for lang in context.languages]
        if repo_lang and repo_lang in context_langs:
            score += 0.5
            matches.append(f"language: {repo_lang}")

        # Framework mentions in description/topics
        repo_text = (
            f"{repo.get('description', '')} {' '.join(repo.get('topics', []))}"
        ).lower()

        for framework in context.frameworks:
            if framework.lower() in repo_text:
                score += 0.3
                matches.append(f"framework: {framework}")
                break  # Count only once

        # Dependency name matching
        repo_name_lower = repo["name"].lower()
        for dep in context.dependencies:
            dep_lower = dep.lower()
            if dep_lower in repo_name_lower or repo_name_lower in dep_lower:
                score += 0.2
                matches.append(f"dependency: {dep}")
                break

        if matches:
            reasoning.append(f"Tech stack match: {', '.join(matches)}")

        return min(score, 1.0)

    def _calculate_topic_score(
        self, repo: dict, context: ProjectContext, reasoning: list[str]
    ) -> float:
        """Calculate topic overlap score."""
        repo_topics = {topic.lower() for topic in repo.get("topics", [])}
        project_topics = {topic.lower() for topic in context.topics}

        if not repo_topics or not project_topics:
            return 0.0

        overlap = repo_topics.intersection(project_topics)
        if overlap:
            score = len(overlap) / max(len(repo_topics), len(project_topics))
            reasoning.append(f"Shared topics: {', '.join(overlap)}")
            return score

        return 0.0

    def _calculate_popularity_score(
        self, repo: dict, reasoning: list[str]
    ) -> float:
        """Calculate normalized popularity score."""
        stars = repo.get("stars", 0)

        # Logarithmic scale: 0 stars -> 0, 10k stars -> ~1.0
        if stars == 0:
            return 0.0

        score = min(np.log10(stars + 1) / 4.0, 1.0) if np else 0.0

        if stars > POPULAR_STARS_HIGH:
            reasoning.append(f"Highly popular ({stars:,} stars)")
        elif stars > POPULAR_STARS_MEDIUM:
            reasoning.append(f"Popular project ({stars:,} stars)")

        return score

    def _calculate_recency_score(
        self, repo: dict, reasoning: list[str]
    ) -> float:
        """Calculate recency/activity score."""
        pushed_at = repo.get("pushed_at")
        if not pushed_at:
            return 0.0

        try:
            # Parse ISO format timestamp
            last_update = datetime.fromisoformat(
                pushed_at.replace("Z", "+00:00")
            )
            now = datetime.now(timezone.utc)
            days_ago = (now - last_update).days

            # Score based on recency: < 30 days -> 1.0, > 365 days -> 0.0
            if days_ago < RECENT_DAYS_1_MONTH:
                score = 1.0
                reasoning.append("Recently updated (< 1 month)")
            elif days_ago < RECENT_DAYS_3_MONTHS:
                score = 0.8
                reasoning.append("Recently updated (< 3 months)")
            elif days_ago < RECENT_DAYS_6_MONTHS:
                score = 0.6
            elif days_ago < RECENT_DAYS_1_YEAR:
                score = 0.4
            else:
                score = 0.2
        except ValueError:
            return 0.0
        else:
            return score

    def _categorize_repository(
        self, repo: dict, score: RecommendationScore
    ) -> str:
        """Categorize repository based on content and score."""
        desc = repo.get("description", "")
        enhanced = repo.get("enhanced_description", "")
        topics = " ".join(repo.get("topics", []))
        repo_text = f"{desc} {enhanced} {topics}".lower()

        # Check each category
        for category, config in self.CATEGORIES.items():
            for keyword in config["keywords"]:
                if keyword in repo_text:
                    return category

        # Default categorization based on score
        if score.tech_stack_score > DIRECT_DEPENDENCY_THRESHOLD:
            return "direct_dependency"
        if score.semantic_score > REFERENCE_IMPL_THRESHOLD:
            return "reference_implementation"
        return "learning_resource"

    def generate_report(
        self,
        recommendations: dict[str, list[RepositoryRecommendation]],
        output_format: str = "markdown",
    ) -> str:
        """Generate a human-readable report.

        Args:
            recommendations: Categorized recommendations
            output_format: 'markdown' or 'text'

        Returns
        -------
            Formatted report string
        """
        if output_format == "markdown":
            return self._generate_markdown_report(recommendations)
        return self._generate_text_report(recommendations)

    def _generate_markdown_report(
        self, recommendations: dict[str, list[RepositoryRecommendation]]
    ) -> str:
        """Generate a markdown report."""
        lines = ["# Repository Recommendations\n"]

        category_names = {
            "direct_dependency": "📦 Direct Dependencies",
            "tool_utility": "🔧 Tools & Utilities",
            "reference_implementation": "📚 Reference Implementations",
            "learning_resource": "🎓 Learning Resources",
        }

        for category, recs in sorted(recommendations.items()):
            if not recs:
                continue

            lines.append(f"\n## {category_names.get(category, category)}\n")

            for rec in recs:
                lines.append(f"### [{rec.owner}/{rec.name}]({rec.url})")
                score_line = (
                    f"\n**Score: {rec.score.composite_score:.1f}/100**\n"
                )
                lines.append(score_line)
                lines.append(f"{rec.description}\n")

                # Scoring details
                lines.append("**Scoring Breakdown:**")
                lines.append(
                    f"- Semantic Similarity: {rec.score.semantic_score:.2f}"
                )
                lines.append(
                    f"- Tech Stack Match: {rec.score.tech_stack_score:.2f}"
                )
                lines.append(f"- Topic Overlap: {rec.score.topic_score:.2f}")
                lines.append(f"- Popularity: {rec.score.popularity_score:.2f}")
                lines.append(f"- Recency: {rec.score.recency_score:.2f}\n")

                # Reasoning
                if rec.score.reasoning:
                    lines.append("**Why this recommendation:**")
                    lines.extend(
                        [f"- {reason}" for reason in rec.score.reasoning]
                    )
                    lines.append("")

                # Metadata
                lang = rec.metadata.get("language")
                stars = rec.metadata.get("stars", 0)
                topics = rec.metadata.get("topics", [])

                meta_parts = []
                if lang:
                    meta_parts.append(f"Language: {lang}")
                if stars:
                    meta_parts.append(f"⭐ {stars:,}")
                if topics:
                    meta_parts.append(f"Topics: {', '.join(topics[:5])}")

                if meta_parts:
                    lines.append(" | ".join(meta_parts))

                lines.append("\n---\n")

        return "\n".join(lines)

    def _generate_text_report(
        self, recommendations: dict[str, list[RepositoryRecommendation]]
    ) -> str:
        """Generate a plain text report."""
        lines = ["REPOSITORY RECOMMENDATIONS", "=" * 50, ""]

        for category, recs in sorted(recommendations.items()):
            if not recs:
                continue

            lines.append(f"\n{category.upper().replace('_', ' ')}")
            lines.append("-" * 50)

            for i, rec in enumerate(recs, 1):
                lines.append(f"\n{i}. {rec.owner}/{rec.name}")
                lines.append(f"   Score: {rec.score.composite_score:.1f}/100")
                lines.append(f"   URL: {rec.url}")
                lines.append(f"   {rec.description}")

                if rec.score.reasoning:
                    lines.append("   Reasons:")
                    lines.extend(
                        [f"   - {reason}" for reason in rec.score.reasoning]
                    )

                lines.append("")

        return "\n".join(lines)


def main() -> None:
    """Run the CLI interface for testing."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="AI-powered repository recommendations"
    )
    parser.add_argument(
        "--starred-repos",
        default="results/starred_repos_authenticated_user_latest.json",
        help="Path to starred repositories JSON file",
    )
    parser.add_argument(
        "--project-path", default=".", help="Path to project for analysis"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file for report (default: auto-generated in results/)",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "text"],
        default="markdown",
        help="Output format",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of recommendations per category",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=30.0,
        help="Minimum score threshold (0-100)",
    )

    args = parser.parse_args()

    # Create recommender
    recommender = RepositoryRecommender()

    # Generate recommendations
    recommendations = recommender.recommend(
        starred_repos_file=args.starred_repos,
        project_path=args.project_path,
        top_n=args.top_n,
        min_score=args.min_score,
    )

    # Generate report
    report = recommender.generate_report(recommendations, args.format)

    # Generate default output filename if not specified
    output_file = args.output
    if not output_file:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        ext = "md" if args.format == "markdown" else "txt"
        output_file = f"results/recommendations_{timestamp}.{ext}"

    # Save output
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_handle:
        file_handle.write(report)
    LOGGER.info("Report saved to %s", output_path)

    # Also print to stdout for immediate viewing
    LOGGER.info("\n%s", "=" * 50)
    LOGGER.info(
        "%s",
        report[:DEFAULT_REPORT_PREVIEW_CHARS] + "..."
        if len(report) > DEFAULT_REPORT_PREVIEW_CHARS
        else report,
    )


if __name__ == "__main__":
    main()
