# Contributing to Tsururu

Thank you for your interest in contributing! This document explains how to get started.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Pre-commit Hooks](#pre-commit-hooks)
- [Running Tests](#running-tests)
- [Pull Request Guidelines](#pull-request-guidelines)

## Code of Conduct

Please be respectful and constructive in all interactions. We follow the [Contributor Covenant](https://www.contributor-covenant.org/).

## How to Contribute

- **Bug reports**: Open an issue using the [Bug Report](.github/ISSUE_TEMPLATE/bug_report.md) template.
- **Feature requests**: Open an issue using the [Feature Request](.github/ISSUE_TEMPLATE/feature_request.md) template.
- **Code contributions**: Fork the repo, create a branch, make your changes, and open a pull request.

## Development Setup

```bash
# Clone your fork
git clone https://github.com/<your-username>/tsururu.git
cd tsururu

# Install uv
python -m pip install uv

# Create environment with uv dependency manager
uv venv
source .venv/bin/activate

# Install dependencies
# - For standard development:
uv sync
# - If you need optional-dependencies (see pyproject.toml):
uv sync --all-extras
```

We use [pre-commit](https://pre-commit.com/) to enforce code style. After installing hooks, they run automatically on every commit.

To manually verify all files before committing:

```bash
uv run pre-commit run --all-files
```

## Running Tests

```bash
uv run pytest tests/
```

To run a specific test module:

```bash
uv run pytest tests/test_strategies/
```

## Pull Request Guidelines

1. Branch off `main` with a descriptive name: `fix/missing-value-imputer` or `feat/new-strategy`.
2. Keep PRs focused — one feature or fix per PR.
3. Make sure all tests pass and pre-commit checks are clean before opening the PR.
4. Fill in the pull request template completely.
5. Reference related issues with `Closes #<issue-number>` in the PR description.
