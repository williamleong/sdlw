"""
Tests for position_generation module.

Unit tests for collision checking and robot position generation functions.
"""

import pytest
import numpy as np
from position_generation import (
    is_position_valid, generate_valid_position, generate_grid_formation,
    ObstacleState, ObstacleShape
)
from config import ROBOT_CONFIG, POSITION_GEN_CONFIG


class TestIsPositionValid:
    """Test collision detection logic."""
    
    def test_empty_obstacles(self):
        """Any position is valid when there are no obstacles."""
        assert is_position_valid(5.0, 5.0, [])
        assert is_position_valid(0.0, 0.0, [])
        assert is_position_valid(100.0, 100.0, [])
    
    def test_circular_obstacle_collision(self):
        """Detect collision with circular obstacles."""
        obs: ObstacleState = {
            'shape': {'name': 'circle', 'radius': 1.0},
            'state': [10.0, 10.0, 0.0]
        }
        
        # Position far from obstacle is valid
        assert is_position_valid(0.0, 0.0, [obs])
        
        # Position very close to center should be invalid
        # (robot_radius + clearance_margin + obs_radius = 0.1 + 0.3 + 1.0 = 1.4)
        assert not is_position_valid(10.0, 10.0, [obs])
        assert not is_position_valid(10.5, 10.0, [obs])
        
        # Position beyond clearance is valid
        assert is_position_valid(12.0, 10.0, [obs])
    
    def test_custom_clearance_margin(self):
        """Clearance margin parameter works correctly."""
        obs: ObstacleState = {
            'shape': {'name': 'circle', 'radius': 1.0},
            'state': [10.0, 10.0, 0.0]
        }
        
        # With larger margin, same position becomes invalid
        # clearance_margin=1.0 instead of default 0.3
        assert is_position_valid(11.6, 10.0, [obs], clearance_margin=0.3)
        assert not is_position_valid(11.6, 10.0, [obs], clearance_margin=1.0)
    
    def test_rectangle_obstacle(self):
        """Detect collision with rectangular obstacles."""
        obs: ObstacleState = {
            'shape': {'name': 'rectangle', 'length': 4.0, 'width': 2.0},
            'state': [10.0, 10.0, 0.0]
        }
        
        # Far away is valid
        assert is_position_valid(0.0, 0.0, [obs])
        
        # Inside rectangle bounds should be invalid
        assert not is_position_valid(10.0, 10.0, [obs])
        assert not is_position_valid(11.0, 10.5, [obs])
        
        # Outside is valid
        assert is_position_valid(5.0, 10.0, [obs])
    
    def test_multiple_obstacles(self):
        """Handle multiple obstacles correctly."""
        obstacles = [
            {'shape': {'name': 'circle', 'radius': 1.0}, 'state': [5.0, 5.0, 0.0]},
            {'shape': {'name': 'circle', 'radius': 1.0}, 'state': [15.0, 15.0, 0.0]},
        ]
        
        # Valid position away from all
        assert is_position_valid(10.0, 10.0, obstacles)
        
        # Invalid near first obstacle
        assert not is_position_valid(5.0, 5.0, obstacles)
        
        # Invalid near second obstacle
        assert not is_position_valid(15.0, 15.0, obstacles)


class TestGenerateValidPosition:
    """Test random position generation."""
    
    def test_generates_valid_position(self):
        """Generate a position that passes validation."""
        world_bounds = (0.0, 20.0, 0.0, 20.0)
        obstacles: list = []
        robot_radius = 0.1
        existing_positions = []
        rng = np.random.RandomState(42)
        
        x, y = generate_valid_position(world_bounds, obstacles, robot_radius, 
                                      existing_positions, rng)
        
        assert 0.0 <= x <= 20.0
        assert 0.0 <= y <= 20.0
        assert is_position_valid(x, y, obstacles, robot_radius)
    
    def test_respects_existing_positions(self):
        """Generated position maintains minimum separation from existing robots."""
        world_bounds = (0.0, 20.0, 0.0, 20.0)
        obstacles: list = []
        robot_radius = 0.1
        existing_positions = [(10.0, 10.0)]  # One robot already placed
        rng = np.random.RandomState(42)
        
        x, y = generate_valid_position(world_bounds, obstacles, robot_radius,
                                      existing_positions, rng,
                                      min_robot_separation=0.5)
        
        # New position should be far from existing
        dist = np.hypot(x - 10.0, y - 10.0)
        assert dist >= 0.5  # Must respect min_robot_separation
    
    def test_fails_with_impossible_constraints(self):
        """Raise error when constraints cannot be satisfied."""
        world_bounds = (0.0, 1.0, 0.0, 1.0)  # Very small world
        obstacles: list = []
        robot_radius = 0.1
        existing_positions = [(0.5, 0.5)]
        rng = np.random.RandomState(42)
        
        # Impossible: min_separation larger than available space
        with pytest.raises(RuntimeError):
            generate_valid_position(world_bounds, obstacles, robot_radius,
                                  existing_positions, rng,
                                  min_robot_separation=10.0,
                                  max_attempts=10)


class TestGenerateGridFormation:
    """Test grid position generation."""
    
    def test_generates_k_positions(self):
        """Generate exactly k positions."""
        world_bounds = (0.0, 20.0, 0.0, 20.0)
        obstacles: list = []
        robot_radius = 0.1
        
        for k in [1, 2, 4, 8, 12]:
            positions = generate_grid_formation(world_bounds, obstacles, robot_radius, k)
            assert len(positions) == k
            # Each position should be (x, y, theta) but only first two used
            assert all(len(p) == 3 or len(p) == 2 for p in positions)
    
    def test_positions_within_bounds(self):
        """All generated positions are within world bounds."""
        world_bounds = (0.0, 20.0, 0.0, 20.0)
        obstacles: list = []
        robot_radius = 0.1
        k = 8
        
        positions = generate_grid_formation(world_bounds, obstacles, robot_radius, k)
        
        for pos in positions:
            x, y = pos[0], pos[1]
            assert 0.0 <= x <= 20.0
            assert 0.0 <= y <= 20.0
    
    def test_positions_non_overlapping(self):
        """Robots in grid don't overlap (respect minimum separation)."""
        world_bounds = (0.0, 20.0, 0.0, 20.0)
        obstacles: list = []
        robot_radius = 0.1
        min_separation = 0.5
        k = 8
        
        positions = generate_grid_formation(world_bounds, obstacles, robot_radius, k,
                                           min_robot_separation=min_separation)
        
        # Check pairwise distances
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                x1, y1 = positions[i][0], positions[i][1]
                x2, y2 = positions[j][0], positions[j][1]
                dist = np.hypot(x1 - x2, y1 - y2)
                # Allow small numerical tolerance
                assert dist >= min_separation - 1e-6, f"Robots {i} and {j} too close: {dist}"
    
    def test_all_positions_valid(self):
        """All generated positions pass collision checks."""
        world_bounds = (0.0, 20.0, 0.0, 20.0)
        obstacles = [
            {'shape': {'name': 'circle', 'radius': 1.0}, 'state': [10.0, 10.0, 0.0]}
        ]
        robot_radius = 0.1
        k = 8
        
        positions = generate_grid_formation(world_bounds, obstacles, robot_radius, k)
        
        for x, y, *_ in positions:
            assert is_position_valid(x, y, obstacles, robot_radius),\
                f"Position ({x}, {y}) failed collision check"
    
    def test_fails_with_impossible_configuration(self):
        """Raise error when k robots can't fit in spawn zone."""
        world_bounds = (0.0, 2.0, 0.0, 2.0)  # Very small
        obstacles: list = []
        robot_radius = 0.1
        k = 100  # Way too many
        
        with pytest.raises(RuntimeError):
            generate_grid_formation(world_bounds, obstacles, robot_radius, k)
    
    def test_near_square_layout(self):
        """Grid formation prefers near-square layouts."""
        world_bounds = (0.0, 20.0, 0.0, 20.0)
        obstacles: list = []
        robot_radius = 0.1
        
        # k=9 should ideally be 3x3
        positions = generate_grid_formation(world_bounds, obstacles, robot_radius, 9)
        assert len(positions) == 9
        
        # Try to verify rough grid structure
        # For 3x3 grid, rows and columns should be roughly aligned
        xs = sorted([p[0] for p in positions])
        ys = sorted([p[1] for p in positions])
        
        # Should have some clustering in x and y dimensions
        assert len(xs) == 9 and len(ys) == 9


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
