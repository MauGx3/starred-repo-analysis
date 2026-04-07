#!/usr/bin/env python3
"""
Starred Repository Scanner.

This script fetches starred repositories from GitHub, extracts metadata,
and prepares data for AI-based analysis.

Usage:
    python scan_starred_repos.py --output data/starred-repos.json
    python scan_starred_repos.py --username octocat --limit 50
"""

import argparse
import base64
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

UTC = timezone.utc
# Load environment variables from .env file in the project root
load_dotenv()

try:
    import requests
    from requests import RequestException
except ImportError:
    sys.stderr.write("Error: requests library is required. Install with: pip install requests\n")
    sys.exit(1)

try:
    from starred_repo_analysis.repo_recommender import RepositoryRecommender
except ImportError:
    try:
        from repo_recommender import RepositoryRecommender
    except ImportError:
        RepositoryRecommender = None


LOGGER = logging.getLogger(__name__)
HTTP_OK = 200
HTTP_UNAUTHORIZED = 401
HTTP_NOT_FOUND = 404
LOW_RATE_LIMIT_THRESHOLD = 10
REQUEST_TIMEOUT_SECONDS = 30
README_PREVIEW_LENGTH = 2000
POPULAR_STARS_THRESHOLD = 1000
ACTIVE_STARS_THRESHOLD = 100
PROGRESS_INTERVAL = 10


class StarredRepoScanner:
    """Scanner for GitHub starred repositories."""

    BASE_URL = "https://api.github.com"

    def __init__(self, token: str | None = None, username: str | None = None) -> None:
        """Initialize scanner.

        Args:
            token: GitHub personal access token (optional but recommended)
            username: GitHub username to scan (defaults to authenticated user)
        """
        self.token = token or os.environ.get("GITHUB_TOKEN")
        self.username = username
        self.headers = {}

        if self.token:
            self.headers["Authorization"] = f"token {self.token}"

        self.headers["Accept"] = "application/vnd.github.v3+json"

    def fetch_starred_repos(self, per_page: int = 100, max_pages: int | None = None) -> list[dict]:
        """Fetch starred repositories.

        Args:
            per_page: Number of results per page (max 100)
            max_pages: Maximum number of pages to fetch (None for all)

        Returns
        -------
            List of repository data dictionaries
        """
        url = f"{self.BASE_URL}/user/starred"
        if self.username:
            url = f"{self.BASE_URL}/users/{self.username}/starred"

        all_repos = []
        page = 1

        while True:
            if max_pages and page > max_pages:
                break

            LOGGER.info("Fetching page %s...", page)

            params = {"per_page": per_page, "page": page}
            response = requests.get(
                url, headers=self.headers, params=params, timeout=REQUEST_TIMEOUT_SECONDS
            )

            if response.status_code == HTTP_UNAUTHORIZED:
                LOGGER.error("Authentication required. Set GITHUB_TOKEN environment variable.")
                sys.exit(1)
            if response.status_code == HTTP_NOT_FOUND:
                LOGGER.error("User '%s' not found.", self.username)
                sys.exit(1)
            if response.status_code != HTTP_OK:
                LOGGER.error("API returned status code %s", response.status_code)
                LOGGER.error("Response: %s", response.text)
                sys.exit(1)

            repos = response.json()

            if not repos:
                break

            all_repos.extend(repos)
            page += 1

            # Check rate limit
            remaining = response.headers.get("X-RateLimit-Remaining")
            if remaining and int(remaining) < LOW_RATE_LIMIT_THRESHOLD:
                LOGGER.warning("Only %s API calls remaining", remaining)

        LOGGER.info("Fetched %s starred repositories", len(all_repos))
        return all_repos

    def fetch_readme(self, owner: str, repo: str) -> str | None:
        """Fetch README content for a repository.

        Args:
            owner: Repository owner
            repo: Repository name

        Returns
        -------
            README content as string, or None if not found.
        """
        url = f"{self.BASE_URL}/repos/{owner}/{repo}/readme"

        try:
            response = requests.get(url, headers=self.headers, timeout=REQUEST_TIMEOUT_SECONDS)
            if response.status_code == HTTP_OK:
                data = response.json()
                # README content is base64 encoded
                content = base64.b64decode(data["content"]).decode("utf-8")
                return content
        except RequestException as error:
            LOGGER.warning("Could not fetch README for %s/%s: %s", owner, repo, error)

        return None

    def fetch_languages(self, owner: str, repo: str) -> dict | None:
        """Fetch language breakdown for a repository.

        Args:
            owner: Repository owner
            repo: Repository name

        Returns
        -------
            Dictionary of languages with byte counts, or None if not found.
        """
        url = f"{self.BASE_URL}/repos/{owner}/{repo}/languages"

        try:
            response = requests.get(url, headers=self.headers, timeout=REQUEST_TIMEOUT_SECONDS)
            if response.status_code == HTTP_OK:
                return response.json()
        except RequestException as error:
            LOGGER.warning("Could not fetch languages for %s/%s: %s", owner, repo, error)

        return None

    def generate_enhanced_description(  # noqa: C901
        self, repo: dict, readme: str | None = None
    ) -> str:
        """Generate an enhanced description using repository metadata.

        Args:
            repo: Repository data from GitHub API
            readme: Optional README content for context

        Returns
        -------
            Enhanced description string.
        """
        base_description = repo.get("description", "")

        # Start with base description
        enhanced_parts = []

        if base_description:
            enhanced_parts.append(base_description)

        # Add context from topics
        topics = repo.get("topics", [])
        if topics and len(topics) > 0:
            topic_str = ", ".join(topics[:5])  # First 5 topics
            enhanced_parts.append(f"Topics: {topic_str}")

        # Add language context
        language = repo.get("language")
        if language:
            enhanced_parts.append(f"Built with {language}")

        # Add activity indicators
        stars = repo.get("stargazers_count", 0)
        if stars > POPULAR_STARS_THRESHOLD:
            enhanced_parts.append(f"Popular project with {stars:,} stars")
        elif stars > ACTIVE_STARS_THRESHOLD:
            enhanced_parts.append(f"Active project with {stars} stars")

        # Extract key phrases from README if available
        if readme:
            # Look for common README sections
            readme_lower = readme.lower()

            # Check for key indicators
            if (
                "cli" in readme_lower
                or "command-line" in readme_lower
                or "command line" in readme_lower
            ):
                enhanced_parts.append("Command-line tool")

            if "framework" in readme_lower:
                enhanced_parts.append("Framework")
            elif "library" in readme_lower:
                enhanced_parts.append("Library")

            if "api" in readme_lower and ("rest" in readme_lower or "graphql" in readme_lower):
                enhanced_parts.append("Provides API functionality")

        return " | ".join(enhanced_parts) if enhanced_parts else "No description available"

    def extract_metadata(
        self,
        repo: dict,
        *,
        include_readme: bool = False,
        include_languages: bool = False,
        enhance_description: bool = False,
    ) -> dict:
        """Extract relevant metadata from repository data.

        Args:
            repo: Repository data from GitHub API
            include_readme: Whether to fetch and include README content
            include_languages: Whether to fetch language breakdown
            enhance_description: Whether to generate enhanced description

        Returns
        -------
            Dictionary with extracted metadata.
        """
        metadata = {
            "repository": repo["full_name"],
            "owner": repo["owner"]["login"],
            "name": repo["name"],
            "github_url": repo["html_url"],
            "description": repo.get("description", ""),
            "language": repo.get("language"),
            "topics": repo.get("topics", []),
            "stars": repo["stargazers_count"],
            "forks": repo["forks_count"],
            "watchers": repo.get("watchers_count", 0),
            "open_issues": repo.get("open_issues_count", 0),
            "created_at": repo["created_at"],
            "updated_at": repo["updated_at"],
            "pushed_at": repo.get("pushed_at"),
            "homepage": repo.get("homepage"),
            "license": repo.get("license", {}).get("name") if repo.get("license") else None,
            "archived": repo.get("archived", False),
            "fork": repo.get("fork", False),
            "default_branch": repo.get("default_branch", "main"),
            "has_issues": repo.get("has_issues", False),
            "has_wiki": repo.get("has_wiki", False),
            "has_pages": repo.get("has_pages", False),
        }

        readme_content = None
        if include_readme:
            readme_content = self.fetch_readme(repo["owner"]["login"], repo["name"])
            if readme_content:
                # Truncate to first 2000 characters to keep manageable for AI
                metadata["readme_preview"] = readme_content[:README_PREVIEW_LENGTH]

        # Fetch language breakdown
        if include_languages:
            languages = self.fetch_languages(repo["owner"]["login"], repo["name"])
            if languages:
                # Calculate percentages
                total_bytes = sum(languages.values())
                language_percentages = {
                    lang: round((bytes_count / total_bytes) * 100, 1)
                    for lang, bytes_count in languages.items()
                }
                metadata["languages"] = language_percentages

        # Generate enhanced description
        if enhance_description:
            metadata["enhanced_description"] = self.generate_enhanced_description(
                repo, readme_content
            )

        return metadata

    def scan(  # noqa: PLR0913
        self,
        output_file: str | None = None,
        per_page: int = 100,
        max_pages: int | None = None,
        *,
        include_readme: bool = False,
        include_languages: bool = False,
        enhance_description: bool = False,
        limit: int | None = None,
    ) -> dict:
        """Scan starred repositories and generate output.

        Args:
            output_file: Path to save JSON output
                        (default: auto-generated in results/)
            per_page: Results per page for API calls
            max_pages: Maximum pages to fetch
            include_readme: Whether to fetch README previews
            include_languages: Whether to fetch language breakdown
            enhance_description: Whether to generate enhanced descriptions
            limit: Maximum number of repositories to process

        Returns
        -------
            Dictionary with scan results.
        """
        # Fetch repositories
        repos = self.fetch_starred_repos(per_page=per_page, max_pages=max_pages)

        # Apply limit if specified
        if limit:
            repos = repos[:limit]

        # Extract metadata
        LOGGER.info("Processing %s repositories...", len(repos))
        repositories = []

        for i, repo in enumerate(repos, 1):
            if i % PROGRESS_INTERVAL == 0:
                LOGGER.info("Processed %s/%s repositories...", i, len(repos))

            metadata = self.extract_metadata(
                repo,
                include_readme=include_readme,
                include_languages=include_languages,
                enhance_description=enhance_description,
            )
            repositories.append(metadata)

        # Create output structure
        output = {
            "scan_date": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "total_repositories": len(repositories),
            "username": self.username or "authenticated_user",
            "repositories": repositories,
        }

        # Generate default output filename if not specified
        if not output_file:
            timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            username = self.username or "authenticated_user"
            output_file = f"results/starred_repos_{username}_{timestamp}.json"

        # Save to file
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file_handle:
            json.dump(output, file_handle, indent=2, ensure_ascii=False)
        LOGGER.info("Results saved to %s", output_path)

        return output


def main() -> None:  # noqa: C901, PLR0915
    """Run the command-line interface."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Scan GitHub starred repositories and extract metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan your own starred repos (requires GITHUB_TOKEN)
  python scan_starred_repos.py --output data/starred-repos.json

  # Scan specific user's starred repos (public)
  python scan_starred_repos.py --username octocat --output octocat-stars.json

  # Limit to first 50 repositories
  python scan_starred_repos.py --limit 50 --output sample.json

  # Include README previews (slower)
  python scan_starred_repos.py --include-readme --limit 10

  # Get AI-powered recommendations for current project
  python scan_starred_repos.py --recommend

  # Get recommendations with custom parameters
  python scan_starred_repos.py --recommend --project-path /path/to/project \\
      --recommend-output recommendations.md
        """,
    )

    parser.add_argument("--output", "-o", help="Output JSON file path (default: print to stdout)")
    parser.add_argument(
        "--username", "-u", help="GitHub username to scan (defaults to authenticated user)"
    )
    parser.add_argument(
        "--token", "-t", help="GitHub personal access token (or set GITHUB_TOKEN env var)"
    )
    parser.add_argument(
        "--per-page", type=int, default=100, help="Results per page (max 100, default: 100)"
    )
    parser.add_argument("--max-pages", type=int, help="Maximum pages to fetch (default: all)")
    parser.add_argument("--limit", "-l", type=int, help="Maximum number of repositories to process")
    parser.add_argument(
        "--include-readme", action="store_true", help="Fetch and include README previews (slower)"
    )
    parser.add_argument(
        "--include-languages",
        action="store_true",
        help="Fetch language breakdown for each repository",
    )
    parser.add_argument(
        "--enhance-description",
        action="store_true",
        help="Generate enhanced descriptions using metadata analysis",
    )
    parser.add_argument(
        "--recommend", action="store_true", help="Generate AI-powered repository recommendations"
    )
    parser.add_argument(
        "--project-path",
        default=".",
        help="Path to project for recommendation analysis (default: current)",
    )
    parser.add_argument(
        "--recommend-output", help="Output file for recommendations (default: stdout)"
    )
    parser.add_argument(
        "--recommend-format",
        choices=["markdown", "text"],
        default="markdown",
        help="Format for recommendations report",
    )
    parser.add_argument(
        "--recommend-top-n",
        type=int,
        default=10,
        help="Number of recommendations per category (default: 10)",
    )
    parser.add_argument(
        "--recommend-min-score",
        type=float,
        default=30.0,
        help="Minimum recommendation score 0-100 (default: 30.0)",
    )

    args = parser.parse_args()

    # Handle recommendation mode
    if args.recommend:
        if RepositoryRecommender is None:
            LOGGER.error(
                "repo_recommender module not found. Make sure repo_recommender.py is importable."
            )
            sys.exit(1)

        # Helper: find latest starred_repos file in results/
        def find_latest_results_file(username_hint: str | None = None) -> str | None:
            results_dir = Path.cwd() / "results"
            if not results_dir.is_dir():
                return None
            candidates: list[Path] = []
            for entry in results_dir.iterdir():
                name = entry.name
                if name.startswith("starred_repos_") and name.endswith(".json"):
                    if username_hint and username_hint not in name:
                        # still allow other files if none match username later
                        pass
                    candidates.append(entry)
            if not candidates:
                return None
            newest_candidate = max(candidates, key=lambda candidate: candidate.stat().st_mtime)
            return str(newest_candidate)

        # Determine starred repos file: prefer explicit output, then latest in
        # results/, else run a quick scan to produce one.
        starred_file = None
        if args.output:
            starred_file = args.output
        else:
            # try to find latest results file
            username_hint = None
            # if username provided, use it as hint
            if args.username:
                username_hint = args.username
            starred_file = find_latest_results_file(username_hint)

        # If we still don't have a file, run a scan to produce one
        if not starred_file or not Path(starred_file).exists():
            LOGGER.info(
                "No existing starred repos file found; running a quick scan to generate one..."
            )
            scanner = StarredRepoScanner(token=args.token, username=args.username)
            # run a short scan (honor other flags)
            scanner.scan(
                output_file=None,
                per_page=args.per_page,
                max_pages=args.max_pages,
                include_readme=args.include_readme,
                include_languages=args.include_languages,
                enhance_description=args.enhance_description,
                limit=args.limit,
            )
            # find the newest file now
            starred_file = find_latest_results_file(args.username)

        if not starred_file or not Path(starred_file).exists():
            LOGGER.error(
                "Error: Could not locate or create starred repos file."
                " Run the scanner first (without --recommend) or provide --output path."
            )
            sys.exit(1)

        LOGGER.info("Using starred repos file: %s", starred_file)

        # Create recommender and generate recommendations
        recommender = RepositoryRecommender()
        recommendations = recommender.recommend(
            starred_repos_file=starred_file,
            project_path=args.project_path,
            top_n=args.recommend_top_n,
            min_score=args.recommend_min_score,
        )

        # Generate report
        report = recommender.generate_report(recommendations, args.recommend_format)

        # Output report to file if requested, otherwise recommender will auto-save
        if args.recommend_output:
            output_path = Path(args.recommend_output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as file_handle:
                file_handle.write(report)
            LOGGER.info("Recommendations saved to %s", output_path)
        else:
            # Let recommender module handle default saving behavior by saving
            # here to results/.
            timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            ext = "md" if args.recommend_format == "markdown" else "txt"
            default_path = Path("results") / f"recommendations_{timestamp}.{ext}"
            default_path.parent.mkdir(parents=True, exist_ok=True)
            with default_path.open("w", encoding="utf-8") as file_handle:
                file_handle.write(report)
            LOGGER.info("Recommendations saved to %s", default_path)

        return

    # Create scanner
    scanner = StarredRepoScanner(token=args.token, username=args.username)

    # Run scan
    results = scanner.scan(
        output_file=args.output,
        per_page=args.per_page,
        max_pages=args.max_pages,
        include_readme=args.include_readme,
        include_languages=args.include_languages,
        enhance_description=args.enhance_description,
        limit=args.limit,
    )

    # Print to stdout if no output file specified
    if not args.output:
        LOGGER.info(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
