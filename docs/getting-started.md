# Getting Started

This page helps you run the project locally and validate the documentation build used by GitHub Pages.

## Prerequisites

- Python 3.10+
- `uv` installed
- A GitHub personal access token (recommended for authenticated scans)

## Install dependencies

Install project dependencies from the repository root:

```bash
uv sync --python 3.10 --all-extras
```

If you prefer `pip`, install at least the runtime dependencies:

```bash
pip install -r requirements.txt
```

## Configure authentication

Set your GitHub token before scanning:

```bash
export GITHUB_TOKEN="<your_token>"
```

Use least-privilege tokens and avoid committing secrets.

## Run your first scan

```bash
python -m starred_repo_analysis.scan_starred_repos --limit 20 --include-languages
```

Expected result: a JSON file in `results/` containing repository metadata.

## Generate recommendations

```bash
python -m starred_repo_analysis.scan_starred_repos --recommend --project-path .
```

Expected result: a recommendation report in `results/`.

## Validate documentation locally

The GitHub Pages workflow builds docs with `uv run mkdocs build`.
Run the same command locally to confirm compatibility:

```bash
uv run mkdocs build
```

Local preview (optional):

```bash
uv run mkdocs serve
```
