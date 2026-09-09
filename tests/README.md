# SDLW Tests

Unit tests for the Sensor-Driven Lévy Walk simulation code.

## Running tests

Run from the repository root with the `sdlw-sim` conda environment activated:

```bash
conda activate sdlw-sim
pytest tests/ -v
```

Run a single test file:

```bash
pytest tests/test_metrics_utils.py -v
```

## Test coverage

- `test_levy_behavior.py` — controller initialization, Lévy exponent sampling, step generation, heading selection, state transitions, velocity limits, and controller modes
- `test_algorithm_flow.py` — empirical checks for SDLW formula behavior and default parameter consistency
- `test_metrics_utils.py` — collision counting, per-run metrics, and coverage utilities
- `test_position_generation.py` — collision-free spawn placement and grid formation generation
- `test_run_simulation.py` — single-run CLI parsing and JSON/coverage output helpers

## Dependencies

Tests require `pytest`, `numpy`, `matplotlib`, `pyyaml`, and `ir-sim`. Install repository dependencies with:

```bash
pip install -r requirements.txt
```
