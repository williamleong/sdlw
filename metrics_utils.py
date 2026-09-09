"""
Metrics collection utilities for SDLW simulations.

Provides functions for:
- Collision counting from IR-SIM collision flags
- Per-run metric tracking
- Coverage metrics integration
"""

from typing import Dict, List, Optional


def count_collisions(robots) -> int:
    """
    Count collision events from robot collision_flag.
    
    Args:
        robots: List of robot objects with collision_flag attribute
        
    Returns:
        Number of robots currently in collision
    """
    count = 0
    for robot in robots:
        if hasattr(robot, 'collision_flag') and robot.collision_flag:
            count += 1
    return count


class ExperimentMetrics:
    """Track metrics during a single simulation run."""
    
    def __init__(self):
        self.collision_count: int = 0
        self.collision_steps: List[tuple] = []
        self.coverage_samples: List[tuple] = []  # [(time, coverage_fraction)]
        self.total_distance: float = 0.0
        self.elapsed_time: Optional[float] = None
        
    def record_collision_step(self, time: float, count: int):
        """Record collision count at this time step."""
        if count > 0:
            self.collision_count += count
            self.collision_steps.append((time, count))
    
    def record_coverage(self, time: float, coverage_fraction: float):
        """Record coverage at this time step."""
        self.coverage_samples.append((time, coverage_fraction))
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary for export."""
        coverage_val = self.coverage_samples[-1][1] if self.coverage_samples else 0.0
        return {
            'collision_count': self.collision_count,
            'total_distance': round(self.total_distance, 2),
            'coverage': round(coverage_val, 2),
            'elapsed_time': round(self.elapsed_time, 2) if self.elapsed_time is not None else None
        }


def compute_coverage_fraction(coverage_state: Dict) -> float:
    """
    Compute coverage fraction from coverage_state.
    
    Args:
        coverage_state: Dictionary from sim_coverage.init_coverage
        
    Returns:
        Fraction of free space visited (0.0 to 1.0)
    """
    visited = coverage_state['visited']
    obstacles = coverage_state['obstacles']
    
    # Free space is non-obstacle cells
    free_space = ~obstacles
    visited_free = visited & free_space
    
    total_free = free_space.sum()
    visited_count = visited_free.sum()
    
    if total_free == 0:
        return 0.0
    
    return float(visited_count) / float(total_free)


def init_headless_coverage(world_bounds: tuple, grid_res: float = 0.2) -> Dict:
    """
    Initialize coverage tracking for headless runs (no display required).
    
    Args:
        world_bounds: ((x_min, x_max), (y_min, y_max)) tuple
        grid_res: Grid resolution in meters
        
    Returns:
        Coverage state dictionary
    """
    import numpy as np
    
    (x_min, x_max), (y_min, y_max) = world_bounds
    
    xs = np.arange(x_min, x_max + grid_res, grid_res)
    ys = np.arange(y_min, y_max + grid_res, grid_res)
    gx, gy = np.meshgrid(xs, ys)
    visited = np.zeros_like(gx, dtype=bool)  # Track if a cell has ever been visited
    visit_counts = np.zeros_like(gx, dtype=float)  # Track visit frequency for visualization
    
    return {
        'gx': gx,
        'gy': gy,
        'visited': visited,
        'visit_counts': visit_counts,
        'obstacles': np.zeros_like(gx, dtype=bool),
        'grid_res': grid_res,
        'world_bounds': world_bounds,
        'last_robot_cells': {}
    }


def rasterize_obstacles_to_grid(coverage_state: Dict, obstacles: List[Dict]):
    """
    Mark obstacle-occupied cells in the coverage grid.
    
    Args:
        coverage_state: Dictionary from init_headless_coverage (modified in-place)
        obstacles: List of obstacle dicts from YAML config with 'shape' and 'state' keys
    """
    import numpy as np
    
    if not obstacles:
        return
    
    gx = coverage_state['gx']
    gy = coverage_state['gy']
    obstacles_grid = coverage_state['obstacles']
    
    for obs in obstacles:
        obs_shape = obs.get('shape', {})
        obs_state = obs.get('state', [0, 0, 0])
        obs_x, obs_y = float(obs_state[0]), float(obs_state[1])
        
        if obs_shape.get('name') == 'circle':
            # Mark cells within circle radius
            obs_r = float(obs_shape.get('radius', 0.2))
            distance = np.sqrt((gx - obs_x)**2 + (gy - obs_y)**2)
            obstacles_grid |= (distance <= obs_r)
        
        elif obs_shape.get('name') == 'rectangle':
            # Mark cells within rectangle bounds (axis-aligned)
            length = float(obs_shape.get('length', 1.0))
            width = float(obs_shape.get('width', 1.0))
            half_l = length / 2.0
            half_w = width / 2.0
            obstacles_grid |= (
                (np.abs(gx - obs_x) <= half_l) & (np.abs(gy - obs_y) <= half_w)
            )
    
    coverage_state['obstacles'] = obstacles_grid


def update_headless_coverage(coverage_state: Dict, robots):
    """
    Update coverage state from robot positions using goal_threshold.
    Works for headless runs without display.
    
    Args:
        coverage_state: Dictionary from init_headless_coverage
        robots: List of robot objects
    """
    import numpy as np
    
    gx = coverage_state['gx']
    gy = coverage_state['gy']
    visited = coverage_state['visited']
    visit_counts = coverage_state.get('visit_counts')
    last_robot_cells = coverage_state.get('last_robot_cells')
    if last_robot_cells is None:
        last_robot_cells = {}
        coverage_state['last_robot_cells'] = last_robot_cells
    (x_min, _), (y_min, _) = coverage_state['world_bounds']
    grid_res = coverage_state['grid_res']
    
    # Update visit counts based on robot positions and goal_threshold
    for robot in robots:
        if hasattr(robot, 'goal_threshold') and hasattr(robot, 'state'):
            # Robot position
            rx = float(robot.state[0, 0])
            ry = float(robot.state[1, 0])
            coverage_radius = float(robot.goal_threshold)
            
            # Compute coverage mask for all cells within coverage_radius
            distance = np.sqrt((gx - rx)**2 + (gy - ry)**2)
            mask = (distance <= coverage_radius)
            
            # Update visited mask every timestep (binary: visited or not)
            visited |= mask
            
            # Increment visit_counts every timestep to show visit frequency for heatmap
            if visit_counts is not None:
                visit_counts += mask.astype(float)
    
    coverage_state['visited'] = visited
    if visit_counts is not None:
        coverage_state['visit_counts'] = visit_counts


def compute_coverage_from_headless(coverage_state: Dict) -> float:
    """
    Compute coverage fraction from headless coverage state (free-space only).
    
    Args:
        coverage_state: Dictionary from init_headless_coverage with obstacles marked
        
    Returns:
        Fraction of free (non-obstacle) space visited (0.0 to 1.0)
    """
    visited = coverage_state['visited']
    obstacles = coverage_state['obstacles']
    
    # Free space is non-obstacle cells
    free_space = ~obstacles
    # Count cells that have been visited (visited is already boolean)
    visited_free = visited & free_space
    
    total_free = free_space.sum()
    visited_count = visited_free.sum()
    
    if total_free == 0:
        return 0.0
    
    return float(visited_count) / float(total_free)
