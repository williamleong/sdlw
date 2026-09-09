"""
Shared pytest fixtures for the SDLW test suite.

Provides common fixtures for environment setup, temporary directories,
and test data generation following ir-sim testing patterns.
"""

from collections.abc import Generator
from pathlib import Path
from typing import Any
import tempfile
import shutil

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from irsim.config import env_param

# Force non-GUI backend for testing
matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Logger setup for irsim integration
# ---------------------------------------------------------------------------


class DummyLogger:
    """Dummy logger for testing that doesn't output anything."""
    
    def info(self, *args, **kwargs):
        pass
    
    def warning(self, *args, **kwargs):
        pass
    
    def error(self, *args, **kwargs):
        pass
    
    def debug(self, *args, **kwargs):
        pass


@pytest.fixture(autouse=True)
def setup_logger():
    """Install dummy logger for all tests."""
    env_param.logger = DummyLogger()
    yield
    env_param.logger = None


# ---------------------------------------------------------------------------
# Matplotlib cleanup fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def cleanup_matplotlib() -> Generator[None, None, None]:
    """Automatically close all matplotlib figures before and after each test."""
    plt.close("all")
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Temporary directory fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test outputs."""
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Mock robot fixtures
# ---------------------------------------------------------------------------


class MockRobot:
    """Mock robot object for testing metrics collection."""
    
    def __init__(self, robot_id: int, x: float = 0.0, y: float = 0.0):
        self.id = robot_id
        self.name = f"robot_{robot_id}"
        self.collision_flag = False
        self.arrive_flag = False
        self._state = np.array([[x], [y], [0.0]])  # x, y, theta
        
    @property
    def state(self):
        return self._state
        
    def set_collision(self, value: bool):
        """Set collision flag."""
        self.collision_flag = value
        
    def set_arrived(self, value: bool):
        """Set arrive flag."""
        self.arrive_flag = value
        
    def set_position(self, x: float, y: float):
        """Set robot position."""
        self._state[0, 0] = x
        self._state[1, 0] = y


@pytest.fixture
def mock_robots():
    """Create a list of mock robots for testing."""
    return [MockRobot(i) for i in range(3)]

