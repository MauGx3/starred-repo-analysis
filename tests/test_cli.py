"""Tests for the starred_repo_analysis CLI entry point."""

import argparse
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from starred_repo_analysis.cli import _build_parser, main, parse_github_username


class TestParseGithubUsername:
    """Unit tests for :func:`parse_github_username`."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("https://github.com/octocat", "octocat"),
            ("https://github.com/octocat/", "octocat"),
            ("https://www.github.com/octocat", "octocat"),
            ("http://github.com/octocat", "octocat"),
            # Port in URL - parsed.hostname strips it correctly.
            ("https://github.com:443/octocat", "octocat"),
            ("https://github.com/octocat/some-repo", "octocat"),
            ("octocat", "octocat"),
            ("octocat/some-repo", "octocat"),
        ],
    )
    def test_valid_inputs(self, value: str, expected: str) -> None:
        """Correctly extracts the username from various input formats."""
        assert parse_github_username(value) == expected

    @pytest.mark.parametrize(
        "value",
        [
            "https://gitlab.com/octocat",
            "https://bitbucket.org/octocat",
        ],
    )
    def test_unsupported_host_raises(self, value: str) -> None:
        """Raises ValueError for non-github.com hosts."""
        with pytest.raises(ValueError, match="Unsupported host"):
            parse_github_username(value)

    def test_unsupported_scheme_raises(self) -> None:
        """Raises ValueError for non-http/https schemes."""
        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            parse_github_username("ssh://github.com/octocat")

    def test_url_without_path_raises(self) -> None:
        """Raises ValueError when a github.com URL has no username segment."""
        with pytest.raises(ValueError, match="No username found"):
            parse_github_username("https://github.com/")

    def test_empty_string_raises(self) -> None:
        """Raises ValueError for an empty string."""
        with pytest.raises(ValueError, match="Username must not be empty"):
            parse_github_username("")


class TestBuildParser:
    """Tests for :func:`_build_parser`."""

    def test_returns_parser(self) -> None:
        """Returns an ArgumentParser instance."""
        assert isinstance(_build_parser(), argparse.ArgumentParser)

    def test_url_positional(self) -> None:
        """URL is a required positional argument."""
        parser = _build_parser()
        args = parser.parse_args(["https://github.com/octocat"])
        assert args.url == "https://github.com/octocat"

    def test_default_recommend_false(self) -> None:
        """--recommend flag defaults to False."""
        parser = _build_parser()
        args = parser.parse_args(["https://github.com/octocat"])
        assert args.recommend is False

    def test_recommend_flag(self) -> None:
        """--recommend flag is parsed correctly."""
        parser = _build_parser()
        args = parser.parse_args(["https://github.com/octocat", "--recommend"])
        assert args.recommend is True

    def test_output_flag(self) -> None:
        """--output / -o flag is parsed correctly."""
        parser = _build_parser()
        args = parser.parse_args(["https://github.com/octocat", "-o", "out.json"])
        assert args.output == "out.json"


class TestMain:
    """Integration-style tests for :func:`main`."""

    def _mock_scan_result(self) -> dict:
        return {
            "scan_date": "2025-01-01T00:00:00Z",
            "total_repositories": 1,
            "username": "octocat",
            "repositories": [],
        }

    def test_main_prints_json_to_stdout(self, capsys: pytest.CaptureFixture) -> None:
        """Without --output the scan result is printed as JSON to stdout."""
        mock_result = self._mock_scan_result()

        with (
            patch(
                "starred_repo_analysis.cli.StarredRepoScanner",
            ) as mock_scanner,
            patch("sys.argv", ["starred-repo-analysis", "https://github.com/octocat"]),
        ):
            instance = MagicMock()
            instance.scan.return_value = mock_result
            mock_scanner.return_value = instance

            main()

        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["username"] == "octocat"

    def test_main_saves_to_output_file(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """With --output the result is saved to the specified file."""
        output_file = str(tmp_path / "stars.json")
        mock_result = self._mock_scan_result()

        with (
            patch(
                "starred_repo_analysis.cli.StarredRepoScanner",
            ) as mock_scanner,
            patch(
                "sys.argv",
                ["starred-repo-analysis", "octocat", "--output", output_file],
            ),
        ):
            instance = MagicMock()
            instance.scan.return_value = mock_result
            mock_scanner.return_value = instance

            main()

        instance.scan.assert_called_once()
        call_kwargs = instance.scan.call_args.kwargs
        assert call_kwargs["output_file"] == output_file

        captured = capsys.readouterr()
        assert captured.out == ""

    def test_main_invalid_url_exits(self) -> None:
        """An invalid (non-github) URL causes SystemExit."""
        with (
            patch("sys.argv", ["starred-repo-analysis", "https://gitlab.com/user"]),
            pytest.raises(SystemExit),
        ):
            main()

    def test_main_username_passed_to_scanner(self) -> None:
        """The extracted username is forwarded to StarredRepoScanner."""
        mock_result = self._mock_scan_result()

        with (
            patch(
                "starred_repo_analysis.cli.StarredRepoScanner",
            ) as mock_scanner,
            patch("sys.argv", ["starred-repo-analysis", "https://github.com/octocat"]),
        ):
            instance = MagicMock()
            instance.scan.return_value = mock_result
            mock_scanner.return_value = instance

            main()

        mock_scanner.assert_called_once_with(token=None, username="octocat")
