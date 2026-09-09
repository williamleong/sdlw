"""
Configuration constants for SDLW simulations.

Centralizes magic numbers and tunable parameters used throughout the codebase.

"""

# Robot configuration
ROBOT_CONFIG = {
    'radius': 0.10,              # Robot body radius in meters
    'min_separation': 0.5,        # Minimum distance between robots in grid formation
    'clearance_margin': 0.3,      # Safety margin around obstacles when checking collision
}

# Render and animation configuration
RENDER_CONFIG = {
    'interval': 0.1,              # Render interval in seconds (0.1s = 10 fps)
    'fps': 10,                    # Frames per second for saved animations
}

# Coverage tracking configuration
COVERAGE_CONFIG = {
    'grid_resolution': 0.25,      # Grid cell size in meters for coverage heatmap
}

# Position generation configuration
POSITION_GEN_CONFIG = {
    'max_attempts': 100,          # Max attempts to generate valid position
    'min_robot_separation': 0.5,  # Minimum distance between robots
    'forbidden_zone_margin': 0.3, # Safety margin around forbidden zones when placing obstacles
}

# Arena spawn zones: (x_min, x_max, y_min, y_max) where robots are initialized
# These are region boundaries where grid formations are generated
SPAWN_ZONES = {
    'rooms_corridors': (16.5, 19.5, 1, 8),  # Bottom-right vertical corridor (shifted)
    'open': (16.5, 19.5, 1, 8),             # Lower right corner
    'cluttered_50': (16.5, 19.5, 1, 8),     # Lower right corner
}

# Obstacle generation for cluttered arenas
OBSTACLE_GEN_CONFIG = {
    'min_radius': 0.2,            # Minimum radius for randomly generated obstacles
    'max_radius': 0.6,            # Maximum radius for randomly generated obstacles
    'obstacle_clearance': 0.3,    # Clearance between generated obstacles
}