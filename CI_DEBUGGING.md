# CI Pipeline Debugging Guide

## Recent Changes to Improve CI Debugging

### 1. Increased Timeout ⏳
- **Changed**: Default 1-hour timeout → **2-hour timeout**
- **Reason**: Tests were hitting the timeout limit
- **Location**: `.gitlab-ci.yml` - `pytest` job

### 2. Enhanced Test Output 📊
- **Added verbose flags**: `--verbose`, `-rA`, `--durations=50`
- **Better tracebacks**: `--tb=short`, `--showlocals`
- **Immediate output**: `--capture=no`, `--color=yes`
- **Location**: `pytest_ci.ini` and `.gitlab-ci.yml`

### 3. Progress Tracking 🔍
- **Added timestamps**: Shows when test execution starts/ends
- **Progress indicators**: Better visibility of which test is running
- **Detailed logging**: Comprehensive test execution logs

### 4. Debug Script 🐞
- **Created**: `scripts/ci_debug.sh`
- **Purpose**: Run tests locally with CI-like verbosity
- **Usage**: `scripts/ci_debug.sh`

## How to Use the Debugging Features

### Local Debugging
```bash
# Run with CI-like verbosity locally
scripts/ci_debug.sh

# Test specific module with debug output
scripts/ci_debug.sh tests/mokka/test_channels.py
```

### CI Configuration
The CI now runs with these enhanced settings:
- **Timeout**: 2 hours (up from 1 hour)
- **Verbosity**: Maximum output to identify slow tests
- **Logging**: Detailed timestamps and progress tracking
- **Reporting**: Comprehensive test results and durations

### Expected Improvements
1. **Better Visibility**: Can see exactly which test is running when timeout occurs
2. **Detailed Logs**: `pytest_report.log` contains comprehensive execution details
3. **Progress Tracking**: Timestamps show test execution progression
4. **Extended Time**: 2-hour window should accommodate slower CI environment

## Debugging Workflow

### If Pipeline Fails Again:
1. **Check the logs**: Look at `pytest_report.log` in CI artifacts
2. **Identify slow tests**: Use duration reports to find bottlenecks
3. **Run locally**: Use `scripts/ci_debug.sh` to reproduce the issue
4. **Optimize**: Mark additional slow tests or improve test performance

### Common CI-Specific Issues
- **Resource constraints**: CI runners may have limited CPU/memory
- **Filesystem differences**: Different performance characteristics
- **Network latency**: Dependency downloads may be slower
- **Concurrency limits**: Parallel test execution may be restricted

## Monitoring the Next Pipeline

When the next pipeline runs, look for:
- **Progress output**: Shows which test is currently executing
- **Duration reports**: Identifies which tests take the longest
- **Timestamps**: Helps track execution progression
- **Detailed tracebacks**: Provides context for any failures

The enhanced configuration should provide much better visibility into where the CI pipeline gets stuck.