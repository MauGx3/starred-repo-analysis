# Scanner CLI

The scanner CLI fetches starred repositories and writes structured metadata for analysis.

## Module

- `starred_repo_analysis.scan_starred_repos`

## Basic usage

```bash
python -m starred_repo_analysis.scan_starred_repos --limit 25
```

## Common options

- `--output`, `-o`: output JSON path
- `--username`, `-u`: target GitHub username
- `--token`, `-t`: GitHub token (or use `GITHUB_TOKEN`)
- `--per-page`: API page size (default `100`)
- `--max-pages`: cap number of API pages
- `--limit`, `-l`: cap number of repositories processed
- `--include-readme`: include README preview per repository
- `--include-languages`: include language percentage breakdown
- `--enhance-description`: generate metadata-based enhanced descriptions
- `--recommend`: run recommendation flow after scan

## Examples

Scan your authenticated account and save output:

```bash
python -m starred_repo_analysis.scan_starred_repos \
  --output results/stars.json \
  --include-languages \
  --enhance-description
```

Scan a public user profile:

```bash
python -m starred_repo_analysis.scan_starred_repos \
  --username octocat \
  --limit 50
```

## Output schema (high level)

The generated JSON includes:

- `scan_date`
- `total_repositories`
- `username`
- `repositories[]`

Each repository entry includes fields such as:

- identity: `repository`, `owner`, `name`, `github_url`
- discovery metadata: `description`, `language`, `topics`
- stats: `stars`, `forks`, `watchers`, `open_issues`
- status: `archived`, `fork`, `default_branch`
- optional enrichments: `readme_preview`, `languages`, `enhanced_description`
