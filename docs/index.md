# Starred Repo Analysis

Starred Repo Analysis helps you turn your GitHub stars into a practical, searchable recommendations dataset for active projects.

## What this project does

- Scans starred repositories through the GitHub API.
- Extracts useful metadata (topics, language, popularity, activity).
- Optionally enriches repository descriptions.
- Scores repositories against your current project context.
- Produces machine-friendly JSON and human-friendly recommendation reports.

## Main components

- `starred_repo_analysis.scan_starred_repos`:
  fetches and prepares starred repository data.
- `starred_repo_analysis.repo_recommender`:
  evaluates scanned repositories and creates ranked recommendations.

## Start here

1. Open [Getting Started](getting-started.md) to install dependencies and configure authentication.
2. Run a first scan with the [Scanner CLI](scanner-cli.md).
3. Generate recommendations with the [Recommender Engine](recommender-engine.md).
4. If you want a complete end-to-end flow, follow the [Workflow Guide](starred-repository-scanner.md).

## Documentation and deployment notes

- This site is built with MkDocs and deployed by the repository GitHub Actions Pages workflow.
- The workflow runs `uv run mkdocs build` and publishes the generated `site/` artifact.
- Keep page links relative (for example, `getting-started.md`) to stay compatible with GitHub Pages routing.
