"""
Tests for metrics_utils module.

Canonical coverage for collision counting, run metrics, and coverage-grid utilities.
"""

import numpy as np

from metrics_utils import (
    count_collisions,
    ExperimentMetrics,
    compute_coverage_fraction,
    init_headless_coverage,
)


class TestCollisionCounting:
    def test_count_collisions_none(self, mock_robots):
        assert count_collisions(mock_robots) == 0

    def test_count_collisions_multiple(self, mock_robots):
        mock_robots[0].set_collision(True)
        mock_robots[2].set_collision(True)
        assert count_collisions(mock_robots) == 2

    def test_count_collisions_empty_list(self):
        assert count_collisions([]) == 0


class TestExperimentMetrics:
    def test_initialization(self):
        metrics = ExperimentMetrics()
        assert metrics.collision_count == 0
        assert metrics.collision_steps == []
        assert metrics.coverage_samples == []
        assert metrics.total_distance == 0.0
        assert metrics.elapsed_time is None

    def test_record_collision_step(self):
        metrics = ExperimentMetrics()
        metrics.record_collision_step(1.5, 2)
        metrics.record_collision_step(3.2, 1)

        assert metrics.collision_count == 3
        assert metrics.collision_steps == [(1.5, 2), (3.2, 1)]

    def test_record_coverage(self):
        metrics = ExperimentMetrics()
        metrics.record_coverage(1.0, 0.25)
        metrics.record_coverage(2.0, 0.45)
        assert metrics.coverage_samples[-1] == (2.0, 0.45)

    def test_to_dict_fields(self):
        metrics = ExperimentMetrics()
        metrics.record_collision_step(1.0, 2)
        metrics.record_coverage(10.0, 0.8)
        metrics.total_distance = 45.67

        result = metrics.to_dict()
        assert set(result.keys()) == {'collision_count', 'total_distance', 'coverage', 'elapsed_time'}
        assert result['collision_count'] == 2
        assert result['total_distance'] == 45.67
        assert result['coverage'] == 0.8


class TestCoverageComputation:
    def test_compute_coverage_fraction_empty(self):
        state = {'visited': np.zeros((10, 10), dtype=bool), 'obstacles': np.zeros((10, 10), dtype=bool)}
        assert compute_coverage_fraction(state) == 0.0

    def test_compute_coverage_fraction_full(self):
        state = {'visited': np.ones((10, 10), dtype=bool), 'obstacles': np.zeros((10, 10), dtype=bool)}
        assert compute_coverage_fraction(state) == 1.0

    def test_compute_coverage_fraction_with_obstacles(self):
        visited = np.ones((10, 10), dtype=bool)
        obstacles = np.zeros((10, 10), dtype=bool)
        obstacles[:2, :] = True
        state = {'visited': visited, 'obstacles': obstacles}
        assert compute_coverage_fraction(state) == 1.0


class TestHeadlessCoverage:
    def test_init_headless_coverage_basic(self):
        bounds = ((0.0, 10.0), (0.0, 10.0))
        state = init_headless_coverage(bounds, grid_res=1.0)

        assert {'gx', 'gy', 'visited', 'obstacles', 'grid_res', 'world_bounds'}.issubset(set(state.keys()))
        assert state['grid_res'] == 1.0
        assert state['world_bounds'] == bounds

    def test_init_headless_coverage_dimensions(self):
        bounds = ((0.0, 10.0), (-5.0, 5.0))
        state = init_headless_coverage(bounds, grid_res=1.0)

        assert state['gx'].shape == (11, 11)
        assert state['gy'].shape == (11, 11)
        assert state['visited'].shape == (11, 11)
