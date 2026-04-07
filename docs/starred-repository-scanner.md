# Workflow Guide

This guide describes a practical end-to-end flow for scanning starred repositories and generating recommendations for the current project.

## End-to-end workflow

1. Configure a GitHub token using `GITHUB_TOKEN`.
2. Run `starred_repo_analysis.scan_starred_repos` to produce a JSON dataset in `results/`.
3. Run `starred_repo_analysis.repo_recommender` against that dataset and your project path.
4. Review the generated markdown report and refine thresholds as needed.

## Step 1: generate starred repository data

Run a focused scan first:

```bash
python -m starred_repo_analysis.scan_starred_repos \
  --limit 50 \
  --include-languages \
  --enhance-description
```

This writes a timestamped dataset under `results/`.

## Step 2: produce recommendations

Use the generated dataset as input:

```bash
python -m starred_repo_analysis.repo_recommender \
  --starred-repos results/starred_repos_authenticated_user_YYYYMMDD_HHMMSS.json \
  --project-path . \
  --format markdown \
  --top-n 10 \
  --min-score 30
```

This creates a markdown report in `results/`.

## Recommendation categories

Current reports group repositories into:

- `direct_dependency`
- `tool_utility`
- `reference_implementation`
- `learning_resource`

## Useful iteration loop

Use this loop to improve quality:

1. Scan with a low limit for fast feedback.
2. Review recommendation reasoning and score breakdown.
3. Tune `--min-score` and `--top-n`.
4. Re-run with a larger limit.

## Troubleshooting

### Authentication errors

If scanning your own stars fails with an authentication error, ensure `GITHUB_TOKEN` is defined and valid.

### Missing optional ML packages

The recommender can run without optional embedding dependencies, but semantic scoring may be reduced.

### File not found for recommendations

If the recommender input file is missing, run the scanner first and pass the resulting JSON path explicitly.
