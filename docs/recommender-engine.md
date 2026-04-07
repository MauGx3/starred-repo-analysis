# Recommender Engine

The recommender engine analyzes your project context and scores scanned repositories for practical relevance.

## Module

- `starred_repo_analysis.repo_recommender`

## Basic usage

```bash
python -m starred_repo_analysis.repo_recommender \
  --starred-repos results/starred_repos_authenticated_user_YYYYMMDD_HHMMSS.json \
  --project-path .
```

## Scoring model

Each recommendation includes a weighted composite score from:

- Semantic similarity
- Technology stack match
- Topic overlap
- Popularity
- Recency

## Categories

Recommendations are grouped into one of:

- `direct_dependency`
- `tool_utility`
- `reference_implementation`
- `learning_resource`

## Key options

- `--starred-repos`: input dataset path
- `--project-path`: project root to analyze
- `--output`, `-o`: report output path
- `--format`: `markdown` or `text`
- `--top-n`: max results per category
- `--min-score`: minimum composite score threshold

## Report contents

Markdown reports include:

- Category sections
- Repository links and descriptions
- Score breakdown by factor
- Human-readable recommendation reasoning

## Dependency behavior

Optional ML packages improve semantic scoring. If they are not installed, the module still runs with reduced semantic depth and retains deterministic scoring factors.
