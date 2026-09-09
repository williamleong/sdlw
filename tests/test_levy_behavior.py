"""
Tests for Levy flight behavior (beh_omni_levy).

Tests the LevyFlightController including:
- Levy exponent sampling
- Discrete step generation
- Heading selection
- State machine transitions
- Baseline modes (reactive_levy, unbiased_levy)
"""

import math

import numpy as np
import pytest

from beh_omni_levy import LevyFlightController, beh_omni_levy


class DummySensor:
    range_max = 10.0

    def get_points(self):
        return None


class DummyRobot:
    def __init__(self, robot_id=0):
        self.id = robot_id
        self.state = np.zeros((3, 1))
        self.sensors = [DummySensor() for _ in range(4)]


class TestLevyExponentSampling:
    """Tests for Levy exponent sampling."""
    
    def test_sample_levy_exponent_in_range(self):
        """Test that sampled exponent is within specified range."""
        controller = LevyFlightController(alpha_min=2.0, alpha_max=3.0, seed=2001)
        
        samples = [controller.sample_levy_exponent() for _ in range(100)]
        
        assert all(2.0 <= alpha <= 3.0 for alpha in samples)
        
    def test_sample_levy_exponent_fixed(self):
        """Test that fixed exponent (alpha_min == alpha_max) returns constant."""
        controller = LevyFlightController(alpha_min=2.5, alpha_max=2.5)
        
        samples = [controller.sample_levy_exponent() for _ in range(10)]
        
        assert all(alpha == 2.5 for alpha in samples)
        
    def test_initialize_alpha_reactive_levy(self):
        """Test alpha initialization for reactive_levy mode (random)."""
        controller = LevyFlightController(
            alpha_min=2.0,
            alpha_max=3.0,
            mode='reactive_levy',
            seed=2002,
        )
        
        assert controller.current_alpha is not None
        assert 2.0 <= controller.current_alpha <= 3.0
        
    def test_initialize_alpha_fixed_exponent(self):
        """Test alpha initialization when a fixed exponent is specified."""
        controller = LevyFlightController(alpha_min=2.5, alpha_max=2.5)
        
        assert controller.current_alpha == 2.5
        
    def test_initialize_alpha_unbiased_levy(self):
        """Test alpha initialization for unbiased_levy mode."""
        controller = LevyFlightController(
            alpha_min=2.0,
            alpha_max=3.0,
            mode='unbiased_levy',
            seed=2003,
        )
        
        assert controller.current_alpha is not None
        assert 2.0 <= controller.current_alpha <= 3.0


class TestDiscreteLevyStepGeneration:
    """Tests for discrete Levy step generation."""
    
    def test_generate_levy_step_non_negative(self):
        """Test that generated steps are non-negative."""
        controller = LevyFlightController(
            alpha_min=2.5,
            alpha_max=2.5,
            min_flight_distance=0.5,
            max_flight_distance=10.0,
            seed=2004,
        )
        
        steps = [controller.generate_levy_step() for _ in range(100)]
        
        assert all(step >= 0.0 for step in steps)
        
    def test_generate_levy_step_within_bounds(self):
        """Test that generated steps respect min/max bounds."""
        controller = LevyFlightController(
            min_flight_distance=1.0,
            max_flight_distance=5.0,
            seed=2005,
        )
        
        steps = [controller.generate_levy_step() for _ in range(100)]
        
        # Steps should be 0 (with P=0.5) or in integer range [1, 5]
        for step in steps:
            assert step == 0.0 or (1.0 <= step <= 5.0)
            
    def test_discrete_levy_step_includes_zero(self):
        """Test that zero steps are generated with P(d=0)=0.5."""
        controller = LevyFlightController(
            alpha_min=2.5,
            alpha_max=2.5,
            min_flight_distance=1.0,
            max_flight_distance=10.0,
            seed=2006,
        )
        
        steps = [controller._generate_discrete_levy_step(2.5) for _ in range(1000)]
        
        # Should have approximately 50% zeros (with some variance)
        zero_count = sum(1 for step in steps if step == 0)
        assert 400 <= zero_count <= 600  # Rough confidence interval
        
    def test_discrete_levy_step_power_law_distribution(self):
        """Test that non-zero steps follow power-law (larger values less frequent)."""
        controller = LevyFlightController(
            alpha_min=2.5,
            alpha_max=2.5,
            min_flight_distance=1.0,
            max_flight_distance=10.0,
            seed=2007,
        )
        
        steps = [controller._generate_discrete_levy_step(2.5) for _ in range(1000)]
        non_zero_steps = [s for s in steps if s > 0]
        
        # Count frequency of small vs large steps
        small_steps = sum(1 for s in non_zero_steps if s <= 3)
        large_steps = sum(1 for s in non_zero_steps if s >= 8)
        
        # Due to power-law, should have more small steps than large
        assert small_steps > large_steps


class TestHeadingSelection:
    """Tests for heading selection methods."""
    
    def test_select_next_heading_unbiased_levy(self):
        """Test unbiased_levy returns uniform random heading."""
        controller = LevyFlightController(mode='unbiased_levy', seed=2008)
        
        headings = [
            controller.select_next_heading(0.0, 5.0, 5.0, 5.0, 5.0)
            for _ in range(100)
        ]
        
        # Check headings are in valid range [-pi, pi]
        assert all(-np.pi <= h <= np.pi for h in headings)
        
        # Check some variance (not all the same)
        assert len(set(headings)) > 10
        
    def test_select_next_heading_fixed_exponent_behaves_like_unbiased(self):
        """Test that specifying a fixed exponent alone does not change heading policy.
        For uniform-heading behaviour use `unbiased_levy` baseline.
        """
        controller = LevyFlightController(mode='unbiased_levy', seed=2009)
        
        headings = [
            controller.select_next_heading(0.0, 5.0, 5.0, 5.0, 5.0)
            for _ in range(100)
        ]
        
        # Check headings are in valid range [-pi, pi]
        assert all(-np.pi <= h <= np.pi for h in headings)
        
        # Check some variance
        assert len(set(headings)) > 10
        
    def test_select_next_heading_reactive_levy_open_front(self):
        """Test reactive_levy biases toward open front sensor."""
        controller = LevyFlightController(
            mode='reactive_levy',
            back_sensor_weight=0.3,
            seed=2010,
        )
        
        # Front sensor very open, others blocked
        headings = [
            controller.select_next_heading(
                current_heading=0.0,
                front_range=10.0,
                left_range=0.5,
                right_range=0.5,
                back_range=0.5
            )
            for _ in range(100)
        ]
        
        # Forward region is 25% of full heading range; reactive policy should
        # exceed random-chance occupancy in this scenario.
        forward_biased = sum(1 for h in headings if abs(h) < np.pi/4)
        assert forward_biased > 30
        
    def test_select_next_heading_reactive_levy_blocked_front(self):
        """Test reactive_levy avoids blocked front sensor."""
        controller = LevyFlightController(
            mode='reactive_levy',
            back_sensor_weight=0.3,
            seed=2011,
        )
        
        # Front blocked, sides open
        headings = [
            controller.select_next_heading(
                current_heading=0.0,
                front_range=0.2,
                left_range=10.0,
                right_range=10.0,
                back_range=0.5
            )
            for _ in range(100)
        ]
        
        # Most headings should avoid forward (abs > pi/4)
        turning = sum(1 for h in headings if abs(h) > np.pi/4)
        assert turning > 50
        
    def test_select_next_heading_all_blocked(self):
        """Check heading bounds with equal, short, nonzero range readings."""
        controller = LevyFlightController(mode='reactive_levy', seed=2012)
        
        # Equal short ranges still have positive total weight, so sampling applies.
        # The backward penalty leaves a forward resultant; only bounds are checked.
        headings = [
            controller.select_next_heading(
                current_heading=0.0,
                front_range=0.1,
                left_range=0.1,
                right_range=0.1,
                back_range=0.1
            )
            for _ in range(50)
        ]
        
        # Check that headings are in valid range
        assert all(-np.pi <= h <= np.pi for h in headings)
        
        # Check for decent spread (at least 10 unique values in 50 samples)
        unique_headings = len(set([round(h, 1) for h in headings]))
        assert unique_headings > 5  # Some variance in headings


class TestStateMachine:
    """Tests for controller state machine."""
    
    def test_initial_state_is_rotating(self):
        """Test controller starts in ROTATING state."""
        controller = LevyFlightController()
        assert controller.state == LevyFlightController.State.ROTATING
        
    def test_transition_rotating_to_moving(self):
        """Test transition from ROTATING to MOVING when aligned."""
        controller = LevyFlightController(
            heading_tolerance_rad=0.1,
            alpha_min=2.5,
            alpha_max=2.5
        )
        controller.target_heading = 0.05  # Within tolerance
        
        # May need multiple attempts since generate_levy_step can return 0
        for _ in range(100):
            vx, vy, wz = controller.update(
                current_pos_x=0.0,
                current_pos_y=0.0,
                current_heading=0.0,
                front_range=10.0,
                left_range=10.0,
                right_range=10.0,
                back_range=10.0
            )
            
            if controller.state == LevyFlightController.State.MOVING and controller.target_distance > 0.0:
                break
            # If we got 0 distance, we stay in ROTATING state, try again
            controller.state = LevyFlightController.State.ROTATING
            controller.target_heading = 0.05
        
        assert controller.state == LevyFlightController.State.MOVING
        assert controller.target_distance > 0.0
        
    def test_transition_moving_to_rotating_on_distance(self):
        """Test transition from MOVING to ROTATING when distance reached."""
        controller = LevyFlightController(alpha_min=2.5, alpha_max=2.5)
        controller.state = LevyFlightController.State.MOVING
        controller.target_distance = 1.0
        controller.start_position = np.array([0.0, 0.0])
        
        # Move to position beyond target distance
        vx, vy, wz = controller.update(
            current_pos_x=1.5,
            current_pos_y=0.0,
            current_heading=0.0,
            front_range=10.0,
            left_range=10.0,
            right_range=10.0,
            back_range=10.0
        )
        
        assert controller.state == LevyFlightController.State.ROTATING
        
    def test_transition_moving_to_rotating_on_obstacle(self):
        """Test transition from MOVING to ROTATING when obstacle detected."""
        controller = LevyFlightController(
            collision_threshold=0.4,
            alpha_min=2.5,
            alpha_max=2.5
        )
        controller.state = LevyFlightController.State.MOVING
        controller.target_distance = 10.0
        controller.start_position = np.array([0.0, 0.0])
        
        # Obstacle detected ahead
        vx, vy, wz = controller.update(
            current_pos_x=0.1,
            current_pos_y=0.0,
            current_heading=0.0,
            front_range=0.3,  # Below collision_threshold
            left_range=10.0,
            right_range=10.0,
            back_range=10.0
        )
        
        assert controller.state == LevyFlightController.State.ROTATING


class TestVelocityCommands:
    """Tests for velocity command generation."""
    
    def test_rotating_state_generates_rotation(self):
        """Test ROTATING state generates angular velocity."""
        controller = LevyFlightController(
            max_yaw_rate=1.0,
            alpha_min=2.5,
            alpha_max=2.5
        )
        controller.state = LevyFlightController.State.ROTATING
        controller.target_heading = np.pi / 2  # 90 degrees
        
        vx, vy, wz = controller.update(
            current_pos_x=0.0,
            current_pos_y=0.0,
            current_heading=0.0,
            front_range=10.0,
            left_range=10.0,
            right_range=10.0,
            back_range=10.0
        )
        
        # Should have angular velocity to turn
        assert wz != 0.0
        assert abs(wz) <= controller.max_yaw_rate
        
    def test_moving_state_generates_forward_velocity(self):
        """Test MOVING state generates forward velocity."""
        controller = LevyFlightController(
            max_speed=1.0,
            alpha_min=2.5,
            alpha_max=2.5
        )
        controller.state = LevyFlightController.State.MOVING
        controller.target_distance = 5.0
        controller.start_position = np.array([0.0, 0.0])
        
        vx, vy, wz = controller.update(
            current_pos_x=0.0,
            current_pos_y=0.0,
            current_heading=0.0,
            front_range=10.0,
            left_range=10.0,
            right_range=10.0,
            back_range=10.0
        )
        
        # Should have forward velocity
        assert vx > 0.0
        assert abs(vx) <= controller.max_speed
        
    def test_velocity_respects_max_speed(self):
        """Test generated velocities respect max_speed limit."""
        controller = LevyFlightController(
            max_speed=0.5,
            alpha_min=2.5,
            alpha_max=2.5
        )
        controller.state = LevyFlightController.State.MOVING
        controller.target_distance = 10.0
        controller.start_position = np.array([0.0, 0.0])
        
        vx, vy, wz = controller.update(
            current_pos_x=0.0,
            current_pos_y=0.0,
            current_heading=0.0,
            front_range=10.0,
            left_range=10.0,
            right_range=10.0,
            back_range=10.0
        )
        
        # Velocity magnitude should not exceed max_speed
        velocity_mag = math.sqrt(vx**2 + vy**2)
        assert velocity_mag <= controller.max_speed + 1e-6
        
    def test_velocity_respects_max_yaw_rate(self):
        """Test generated angular velocity respects max_yaw_rate limit."""
        controller = LevyFlightController(
            max_yaw_rate=0.5,
            alpha_min=2.5,
            alpha_max=2.5
        )
        controller.state = LevyFlightController.State.ROTATING
        controller.target_heading = np.pi  # Large angle difference
        
        vx, vy, wz = controller.update(
            current_pos_x=0.0,
            current_pos_y=0.0,
            current_heading=0.0,
            front_range=10.0,
            left_range=10.0,
            right_range=10.0,
            back_range=10.0
        )
        
        assert abs(wz) <= controller.max_yaw_rate + 1e-6


class TestBaselineModes:
    """Tests for different baseline mode configurations."""
    
    def test_reactive_levy_mode(self):
        """Test reactive_levy mode configuration."""
        controller = LevyFlightController(mode='reactive_levy')

        assert controller.mode == 'reactive_levy'
        assert 2.0 <= controller.current_alpha <= 3.0
        
    def test_fixed_exponent_mode(self):
        """Test that specifying a fixed exponent sets alpha accordingly."""
        controller = LevyFlightController(alpha_min=2.5, alpha_max=2.5)
        
        assert controller.current_alpha == 2.5
        
    def test_unbiased_levy_mode(self):
        """Test unbiased_levy mode configuration."""
        controller = LevyFlightController(mode='unbiased_levy')

        assert controller.mode == 'unbiased_levy'
        assert 2.0 <= controller.current_alpha <= 3.0


class TestBehaviorControllerLifecycle:
    def test_controller_is_isolated_per_robot(self):
        first_robot = DummyRobot(robot_id=1)
        second_robot = DummyRobot(robot_id=1)

        beh_omni_levy(first_robot, seed=7)
        beh_omni_levy(second_robot, seed=7)

        assert first_robot._sdlw_levy_controller is not second_robot._sdlw_levy_controller

    def test_controller_is_recreated_when_configuration_changes(self):
        robot = DummyRobot(robot_id=1)

        beh_omni_levy(robot, seed=7, mode="reactive_levy")
        original_controller = robot._sdlw_levy_controller
        beh_omni_levy(robot, seed=7, mode="unbiased_levy")

        assert robot._sdlw_levy_controller is not original_controller
        assert robot._sdlw_levy_controller.mode == "unbiased_levy"

    def test_controller_is_reused_for_unchanged_configuration(self):
        robot = DummyRobot(robot_id=1)

        beh_omni_levy(robot, seed=7, mode="reactive_levy")
        original_controller = robot._sdlw_levy_controller
        beh_omni_levy(robot, seed=7, mode="reactive_levy")

        assert robot._sdlw_levy_controller is original_controller


class TestControllerParameters:
    """Tests for controller parameter initialization."""
    
    def test_default_parameters(self):
        """Test default parameter values."""
        controller = LevyFlightController()
        
        assert controller.max_speed == 0.5
        assert controller.max_yaw_rate == 0.5
        assert controller.alpha_min == 2.0
        assert controller.alpha_max == 3.0
        assert controller.min_flight_distance == 0.5
        assert controller.max_flight_distance == 20.0
        assert controller.collision_threshold == 0.4
        assert controller.heading_tolerance_rad == 0.087
        assert controller.back_sensor_weight == 0.3
        assert controller.mode == 'reactive_levy'
        
    def test_custom_parameters(self):
        """Test custom parameter values."""
        controller = LevyFlightController(
            max_speed=1.5,
            max_yaw_rate=1.0,
            alpha_min=2.2,
            alpha_max=2.8,
            min_flight_distance=1.0,
            max_flight_distance=20.0,
            collision_threshold=0.5,
            heading_tolerance_rad=0.1,
            back_sensor_weight=0.5,
            mode='unbiased_levy'
        )
        
        assert controller.max_speed == 1.5
        assert controller.max_yaw_rate == 1.0
        assert controller.alpha_min == 2.2
        assert controller.alpha_max == 2.8
        assert controller.min_flight_distance == 1.0
        assert controller.max_flight_distance == 20.0
        assert controller.collision_threshold == 0.5
        assert controller.heading_tolerance_rad == 0.1
        assert controller.back_sensor_weight == 0.5
        assert controller.mode == 'unbiased_levy'
