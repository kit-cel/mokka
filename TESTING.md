# Testing Guide for Mokka

## Test Configuration

This project has two test configurations to optimize developer workflow:

### Local Development (Fast Tests)

For local development, use the fast test configuration that excludes slow-running tests:

```bash
# Run fast tests only
scripts/test_local.sh

# Run fast tests with specific options
scripts/test_local.sh tests/mokka/test_mapping.py -v
```

**What's excluded:**
- Adaptive equalizer tests (20+ seconds each)
- SSFM dual polarization tests (6+ seconds each)  
- Pulse shaping AWGN tests (5+ seconds each)

**Execution time:** ~1-2 seconds for full test suite

### CI/Full Testing

For complete testing (CI, pre-commit, full validation), use the CI configuration:

```bash
# Run all tests including slow ones
scripts/test_ci.sh

# Run with coverage reporting
scripts/test_ci.sh --cov=src --cov-report=term
```

**Execution time:** ~2-3 minutes for full test suite

## Test Markers

Tests are categorized using pytest markers:

- `@pytest.mark.slow` - Marks tests that take >1 second to execute
- `@pytest.mark.integration` - Marks integration tests  
- `@pytest.mark.unit` - Marks unit tests

## Adding New Tests

### Fast Tests (Unit Tests)

```python
def test_my_fast_function():
    # Test code here - should execute in <100ms
    assert some_function() == expected_result
```

### Slow Tests

```python
import pytest

@pytest.mark.slow
def test_my_slow_simulation():
    # Test code here - may take several seconds
    result = run_complex_simulation()
    assert result.is_valid()
```

## Test Structure

- **Unit tests**: Test individual functions in isolation
- **Integration tests**: Test module interactions
- **Performance tests**: Validate execution time and resource usage
- **Regression tests**: Prevent known bugs from reappearing

## Running Specific Tests

```bash
# Run tests for a specific module
scripts/test_local.sh tests/mokka/test_channels.py

# Run a specific test function
scripts/test_local.sh tests/mokka/test_channels.py::test_awgn

# Run with verbose output
scripts/test_local.sh -v --tb=short

# Run with duration reporting
scripts/test_local.sh --durations=10
```

## Test Coverage

Current coverage: ~14% overall

**Coverage goals:**
- Unit tests: >80% for new code
- Integration tests: >60% for critical paths
- Overall: >50% target

Run coverage analysis:

```bash
scripts/test_ci.sh --cov=src --cov-report=term-missing
```

## Continuous Integration

The CI configuration includes:
- Full test suite execution
- Coverage reporting
- Test duration monitoring
- Failure analysis

## Troubleshooting

**Tests timing out:**
- Check for infinite loops in test code
- Add timeouts to long-running operations
- Use `@pytest.mark.slow` for tests >1s

**Import errors:**
- Ensure PYTHONPATH includes ./src
- Check dependencies are installed
- Verify module structure

**Flaky tests:**
- Add random seeds for reproducibility
- Use pytest-rerunfailures plugin
- Isolate test dependencies
