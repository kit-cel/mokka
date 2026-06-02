#!/bin/bash
# Run tests with CI configuration (includes all tests)

echo "Running tests with CI configuration (all tests including slow ones)..."
uv run pytest -c pytest_ci.ini "$@"