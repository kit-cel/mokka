#!/bin/bash
# Run tests with local configuration (excludes slow tests)

echo "Running tests with local configuration (fast tests only)..."
uv run pytest -c pytest_local.ini "$@"