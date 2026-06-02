#!/bin/bash
# CI Debug script - runs tests with detailed logging for debugging

echo "🔍 CI Debug Mode - Running tests with maximum verbosity"
echo "This will help identify which test is causing the timeout"
echo ""

# Run tests with maximum verbosity and detailed reporting
uv run pytest -c pytest_ci.ini \
    --verbose \
    --showlocals \
    -rA \
    --durations=50 \
    --tb=long \
    --capture=no \
    --report-log=pytest_debug.log \
    "$@"

echo ""
echo "📊 Debug information saved to pytest_debug.log"
echo "🔬 Check the log file for detailed test execution information"