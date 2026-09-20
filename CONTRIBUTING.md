# Contributing to vjepa-stress

We welcome contributions that improve reproducibility, documentation, or benchmark analysis. All contributions must adhere to the Poisson Labs repository standard defined in [`docs/repo-hygiene.md`](docs/repo-hygiene.md).

## Git Hook Setup

To ensure commit messages comply with repository hygiene policies (rejecting automated attribution trailers and machine co-authorship per `docs/repo-hygiene.md §2.1`), enable the repository hooks upon cloning:

```bash
git config core.hooksPath .githooks
```

## Commit Guidelines

- Write plain, concise commit messages describing the technical change.
- Never include automated attribution trailers, generated tags, or machine co-authorship footers.
- Ensure all file paths in scripts, tests, and documentation are repository-relative.
- Ensure no personal or defunct email addresses are added.

## Development Workflow

1. Set up the virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -e ".[dev]"
   ```
2. Run unit tests:
   ```bash
   pytest tests/ -m "not slow and not calibration"
   ```
3. Run code formatting and linting:
   ```bash
   ruff check .
   ```
