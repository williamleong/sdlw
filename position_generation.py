"""
Position generation utilities for robot spawn placement in arenas.

Provides functions for collision checking and generating valid robot positions
in grid formations and random placements.
"""

from typing import List, Tuple, Optional, TypedDict
import numpy as np
from config import ROBOT_CONFIG, POSITION_GEN_CONFIG


class ObstacleShape(TypedDict):
    """Type definition for obstacle shape parameters."""
    name: str
    radius: float


class ObstacleState(TypedDict):
    """Type definition for obstacle state (position and orientation)."""
    shape: ObstacleShape
    state: List[float]


def is_position_valid(x: float, y: float, obstacles: List[ObstacleState], 
                     robot_radius: Optional[float] = None, 
                     clearance_margin: Optional[float] = None) -> bool:
    """
    Check if a position is collision-free.
    
    Args:
        x, y: Position coordinates to check
        obstacles: List of obstacle dicts with 'shape' and 'state' keys
        robot_radius: Radius of robot region to check (default: ROBOT_CONFIG['radius'])
        clearance_margin: Additional safety margin around obstacles (default: ROBOT_CONFIG['clearance_margin'])
        
    Returns:
        True if position is valid (no collisions), False otherwise
    """
    if robot_radius is None:
        robot_radius = ROBOT_CONFIG['radius']
    if clearance_margin is None:
        clearance_margin = ROBOT_CONFIG['clearance_margin']
    if not obstacles:
        return True
    
    check_point_radius = robot_radius + clearance_margin
    
    for obs in obstacles:
        obs_shape = obs['shape']
        obs_state = obs['state']
        obs_x, obs_y = obs_state[0], obs_state[1]
        
        if obs_shape['name'] == 'circle':
            # Check distance to circular obstacle
            obs_radius = obs_shape['radius']
            dist = np.hypot(x - obs_x, y - obs_y)
            if dist < (check_point_radius + obs_radius):
                return False
        
        elif obs_shape['name'] == 'rectangle':
            # For rectangle obstacles, check with margin
            length = obs_shape['length']
            width = obs_shape['width']
            # Rectangle is centered at obs_x, obs_y
            half_l = length / 2.0 + check_point_radius
            half_w = width / 2.0 + check_point_radius
            if (abs(x - obs_x) < half_l and abs(y - obs_y) < half_w):
                return False
    
    return True


def generate_valid_position(world_bounds: Tuple[float, float, float, float], 
                           obstacles: List[ObstacleState], 
                           robot_radius: float, 
                           existing_positions: List[Tuple[float, float]],
                           rng: np.random.RandomState, 
                           min_robot_separation: Optional[float] = None, 
                           max_attempts: Optional[int] = None) -> Tuple[float, float]:
    """
    Generate a random collision-free position.
    
    Args:
        world_bounds: (x_min, x_max, y_min, y_max) tuple
        obstacles: List of obstacle dicts
        robot_radius: Robot radius for collision checking
        existing_positions: List of (x, y) tuples for already-placed robots
        rng: numpy RandomState for reproducibility
        min_robot_separation: Minimum distance between robots (default: POSITION_GEN_CONFIG['min_robot_separation'])
        max_attempts: Maximum retry attempts (default: POSITION_GEN_CONFIG['max_attempts'])
        
    Returns:
        (x, y) tuple of valid position
        
    Raises:
        RuntimeError if no valid position found after max_attempts
    """
    if min_robot_separation is None:
        min_robot_separation = POSITION_GEN_CONFIG['min_robot_separation']
    if max_attempts is None:
        max_attempts = POSITION_GEN_CONFIG['max_attempts']
    x_min, x_max, y_min, y_max = world_bounds
    
    for attempt in range(max_attempts):
        # Generate candidate position
        x = rng.uniform(x_min, x_max)
        y = rng.uniform(y_min, y_max)
        
        # Check against obstacles
        if not is_position_valid(x, y, obstacles, robot_radius):
            continue
        
        # Check against existing robot positions
        valid = True
        for ex_x, ex_y in existing_positions:
            dist = np.hypot(x - ex_x, y - ex_y)
            if dist < min_robot_separation:
                valid = False
                break
        
        if valid:
            return (x, y)
    
    raise RuntimeError(
        f"Could not find valid position after {max_attempts} attempts. "
        f"Arena may be too crowded or constraints too strict."
    )


def generate_grid_formation(world_bounds: Tuple[float, float, float, float], 
                           obstacles: List[ObstacleState], 
                           robot_radius: float, 
                           k: int, 
                           min_robot_separation: Optional[float] = None) -> List[Tuple[float, float]]:
    """
    Generate k robot positions in a grid formation within the spawn zone.

    This implementation:
    - Chooses grid columns/rows to maximize spacing inside the spawn zone
    - Attempts a staggered (offset) layout for larger teams to increase
      pairwise separation where possible
    - Falls back to randomized placement for any remaining robots while
      respecting `min_robot_separation` as much as possible

    Args:
        world_bounds: (x_min, x_max, y_min, y_max) tuple defining spawn zone
        obstacles: List of obstacle dicts for collision checking
        robot_radius: Robot radius for collision checking
        k: Number of robots to position
        min_robot_separation: Minimum distance between robots (default: ROBOT_CONFIG['min_separation'])

    Returns:
        List of (x, y) tuples for robot positions

    Raises:
        RuntimeError if cannot find valid positions for all robots
    """
    if min_robot_separation is None:
        min_robot_separation = ROBOT_CONFIG['min_separation']
    x_min, x_max, y_min, y_max = world_bounds

    # Simplified square-grid-only placement.
    # Strategy:
    # - Try column counts near sqrt(k) to get a near-square layout
    # - For each (cols, rows) compute equal spacing s and try small offsets
    # - Accept the first placement where all k points are outside obstacles
    #   and respect `min_robot_separation`.
    start_cols = max(1, int(round(np.sqrt(k))))
    col_range = list(range(max(1, start_cols - 2), min(k, start_cols + 3) + 1))
    # Prefer candidates near the ideal start_cols (near-square layouts first)
    col_candidates = sorted(col_range, key=lambda c: abs(c - start_cols))

    for cols in col_candidates:
        rows = int(np.ceil(k / cols))
        sx = (x_max - x_min) / (cols + 1)
        sy = (y_max - y_min) / (rows + 1)
        s = min(sx, sy)
        if s < min_robot_separation:
            continue

        # try a few offsets within half a spacing to avoid obstacles
        n_offsets = 5
        for ox in np.linspace(0.0, s * 0.5, n_offsets):
            for oy in np.linspace(0.0, s * 0.5, n_offsets):
                pts = []
                valid = True
                for r in range(rows):
                    for c in range(cols):
                        if len(pts) >= k:
                            break
                        x = x_min + (c + 1) * s + ox
                        y = y_min + (r + 1) * s + oy
                        # keep inside bounds with robot radius
                        x = min(max(x, x_min + robot_radius), x_max - robot_radius)
                        y = min(max(y, y_min + robot_radius), y_max - robot_radius)

                        if not is_position_valid(x, y, obstacles, robot_radius):
                            valid = False
                            break

                        if any(np.hypot(x - ex, y - ey) < min_robot_separation for ex, ey in pts):
                            valid = False
                            break

                        pts.append((float(x), float(y)))
                    if not valid or len(pts) >= k:
                        break

                if valid and len(pts) >= k:
                    return pts[:k]

    # If we reach here, no clean square grid could be found
    raise RuntimeError(
        f"Could not generate square grid formation for {k} robots. "
        f"Found 0 valid placements. Try increasing the spawn zone, reducing `k`, or lowering `min_robot_separation`."
    )
