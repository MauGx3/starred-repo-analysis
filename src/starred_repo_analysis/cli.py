"""CLI entry point for starred-repo-analysis.

Provides the ``starred-repo-analysis`` command that accepts a GitHub profile
URL (or plain username) and runs the full scan-and-recommend pipeline.

Usage
-----
::

    starred-repo-analysis https://github.com/octocat
    starred-repo-analysis octocat --output stars.json
    starred-repo-analysis https://github.com/octocat --recommend
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from starred_repo_analysis.scan_starred_repos import StarredRepoScanner

try:
    from starred_repo_analysis.repo_recommender import (
        RepositoryRecommender as _RepositoryRecommender,
    )

    _RECOMMENDER_AVAILABLE = True
except ImportError:
    _RECOMMENDER_AVAILABLE = False

_GITHUB_HOSTS: frozenset[str] = frozenset({"github.com", "www.github.com"})
LOGGER = logging.getLogger(__name__)


def parse_github_username(value: str) -> str:
    """Extract a GitHub username from a profile URL or return the value as-is.

    Parameters
    ----------
    value : str
        A GitHub profile URL (e.g. ``https://github.com/username``) or a
        plain username string.

    Returns
    -------
    str
        The GitHub username.

    Raises
    ------
    ValueError
        If *value* looks like a URL but does not contain a username path
        segment, or if the host is not ``github.com``.

    Examples
    --------
    >>> parse_github_username("https://github.com/octocat")
    'octocat'
    >>> parse_github_username("https://github.com/octocat/")
    'octocat'
    >>> parse_github_username("octocat")
    'octocat'
    """
    parsed = urlparse(value)
    # Any non-empty scheme means the value was intended as a URL.
    if parsed.scheme:
        if parsed.scheme not in {"http", "https"}:
            msg = (
                f"Unsupported URL scheme '{parsed.scheme}'. "
                "Only http/https GitHub URLs are accepted."
            )
            raise ValueError(msg)
        # Use parsed.hostname so port numbers in netloc don't break comparison.
        hostname = (parsed.hostname or "").lower()
        if hostname not in _GITHUB_HOSTS:
            msg = (
                f"Unsupported host '{parsed.hostname}'. "
                "Only github.com URLs are accepted."
            )
            raise ValueError(msg)
        parts = [p for p in parsed.path.split("/") if p]
        if not parts:
            msg = "No username found in the provided GitHub URL."
            raise ValueError(msg)
        return parts[0]
    # No scheme - treat the raw value as a username (or user/repo path).
    username = value.split("/", maxsplit=1)[0].strip()
    if not username:
        msg = "Username must not be empty."
        raise ValueError(msg)
    return username


def _build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the ``starred-repo-analysis`` command.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser instance.
    """
    parser = argparse.ArgumentParser(
        prog="starred-repo-analysis",
        description=(
            "Analyse a GitHub user's starred repositories and optionally "
            "generate AI-powered recommendations for your current project."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan starred repos for a public GitHub user
  starred-repo-analysis https://github.com/octocat

  # Scan your own starred repos (requires GITHUB_TOKEN env var)
  starred-repo-analysis https://github.com/me --token $GITHUB_TOKEN

  # Save results to a file
  starred-repo-analysis https://github.com/octocat --output stars.json

  # Also generate recommendations for the current project
  starred-repo-analysis https://github.com/octocat --recommend

  # Recommendations for a specific project path
  starred-repo-analysis https://github.com/octocat --recommend --project-path /my/project
        """,
    )

    parser.add_argument(
        "url",
        metavar="URL",
        help=(
            "GitHub profile URL (e.g. https://github.com/username) "
            "or plain GitHub username."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        metavar="FILE",
        help=(
            "Save the scan results JSON to FILE. If omitted, results are "
            "printed to stdout and also auto-saved to a timestamped JSON "
            "file in the results/ directory."
        ),
    )
    parser.add_argument(
        "--token",
        "-t",
        metavar="TOKEN",
        help="GitHub personal access token (or set GITHUB_TOKEN env var).",
    )
    parser.add_argument(
        "--per-page",
        type=int,
        default=100,
        metavar="N",
        help="Results per page (max 100, default: %(default)s).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        metavar="N",
        help="Maximum number of pages to fetch (default: all).",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        metavar="N",
        help="Maximum number of repositories to process.",
    )
    parser.add_argument(
        "--include-readme",
        action="store_true",
        help="Fetch and include README previews (slower).",
    )
    parser.add_argument(
        "--include-languages",
        action="store_true",
        help="Fetch language breakdown for each repository.",
    )
    parser.add_argument(
        "--enhance-description",
        action="store_true",
        help="Generate enhanced descriptions using metadata analysis.",
    )
    parser.add_argument(
        "--recommend",
        action="store_true",
        help="Generate AI-powered repository recommendations after scanning.",
    )
    parser.add_argument(
        "--project-path",
        default=".",
        metavar="PATH",
        help="Project path used for recommendation analysis (default: %(default)s).",
    )
    parser.add_argument(
        "--recommend-output",
        metavar="FILE",
        help="Save the recommendations report to FILE (default: auto-save to results/).",
    )
    parser.add_argument(
        "--recommend-format",
        choices=["markdown", "text"],
        default="markdown",
        help="Format for the recommendations report (default: %(default)s).",
    )
    parser.add_argument(
        "--recommend-top-n",
        type=int,
        default=10,
        metavar="N",
        help="Number of recommendations per category (default: %(default)s).",
    )
    parser.add_argument(
        "--recommend-min-score",
        type=float,
        default=30.0,
        metavar="SCORE",
        help="Minimum recommendation score 0-100 (default: %(default)s).",
    )
    return parser


def _find_latest_starred_file(hint_output: str | None) -> Path | None:
    """Locate the most recent starred-repos JSON file in the results directory.

    Parameters
    ----------
    hint_output : str | None
        The ``--output`` path supplied by the user, if any.

    Returns
    -------
    Path | None
        The path to the located file, or ``None`` if none could be found.
    """
    if hint_output:
        path = Path(hint_output)
        if path.exists():
            return path
    results_dir = Path.cwd() / "results"
    if not results_dir.is_dir():
        return None
    candidates = sorted(
        (p for p in results_dir.iterdir() if p.name.startswith("starred_repos_") and p.suffix == ".json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _save_report(report: str, output_path: str | None, fmt: str) -> None:
    """Write the recommendations *report* to a file.

    Parameters
    ----------
    report : str
        The formatted report content.
    output_path : str | None
        Explicit output file path, or ``None`` to auto-generate under ``results/``.
    fmt : str
        Report format name (``"markdown"`` or ``"text"``), used to choose the
        file extension when *output_path* is ``None``.
    """
    if output_path:
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(report, encoding="utf-8")
        LOGGER.info("Recommendations saved to %s", dest)
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        ext = "md" if fmt == "markdown" else "txt"
        dest = Path("results") / f"recommendations_{timestamp}.{ext}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(report, encoding="utf-8")
        LOGGER.info("Recommendations saved to %s", dest)


def _run_recommendations(args: argparse.Namespace) -> None:
    """Generate and save AI-powered repository recommendations.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    """
    if not _RECOMMENDER_AVAILABLE:
        LOGGER.error("repo_recommender module could not be imported.")
        sys.exit(1)

    starred_file = _find_latest_starred_file(args.output)
    if starred_file is None:
        LOGGER.error(
            "Could not locate the starred repos JSON file for recommendation. "
            "Re-run with --output to specify a path."
        )
        sys.exit(1)

    recommender = _RepositoryRecommender()
    recommendations = recommender.recommend(
        starred_repos_file=str(starred_file),
        project_path=args.project_path,
        top_n=args.recommend_top_n,
        min_score=args.recommend_min_score,
    )
    report = recommender.generate_report(recommendations, args.recommend_format)
    _save_report(report, args.recommend_output, args.recommend_format)


def main() -> None:
    """Entry point for the ``starred-repo-analysis`` CLI command.

    Parses command-line arguments, resolves the GitHub username from the
    supplied URL, runs the starred-repository scan, and optionally generates
    AI-powered recommendations.
    """
    parser = _build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        username = parse_github_username(args.url)
    except ValueError as exc:
        parser.error(str(exc))

    scanner = StarredRepoScanner(token=args.token, username=username)

    results = scanner.scan(
        output_file=args.output,
        per_page=args.per_page,
        max_pages=args.max_pages,
        include_readme=args.include_readme,
        include_languages=args.include_languages,
        enhance_description=args.enhance_description,
        limit=args.limit,
    )

    if not args.output:
        print(json.dumps(results, indent=2, ensure_ascii=False))  # noqa: T201

    if args.recommend:
        _run_recommendations(args)
