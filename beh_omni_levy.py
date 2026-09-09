# -*- coding: utf-8 -*-
"""
Sensor-Driven Lévy Walk Behavior for Omni-directional Robots

Implements a sensor-driven Lévy walk exploration pattern with step-lengths
drawn from a discrete power-law distribution and (by default) a *randomized* Lévy
exponent α ∈ (2,3), as proven effective for parallel search:

  - Clementi, d'Amore, Giakkoupis, Natale (arXiv:2004.01562v6, 2024)
    "Search via Parallel Lévy Walks on Z^2", Theorem 1.6.

This module implements two variants:
1) Sensor-Driven Lévy Walk (SDLW): Full method with sensor-adaptive von Mises heading policy
   that adjusts concentration based on normalized sensor-resultant strength.
2) Uniform Heading Lévy Walk (UHLW): Baseline with uniform random heading selection.

Key design features:
- Each agent draws its exponent α ~ Uniform(2,3) (once per agent by default),
  which is near-optimal for parallel search without knowing k or the target distance ℓ.
- Step lengths are generated as integer distances with P(d=0)=1/2 and
    P(d=i) ∝ i^{-α} for i≥1, truncated into [min_flight_distance, max_flight_distance].
- Heading selection mode controls whether to use sensor-adaptive (SDLW) or
  uniform random (UHLW) heading selection.

Backwards compatibility:
- If you pass levy_exponent in kwargs, we fix α to that value
  (alpha_min == alpha_max), reproducing pre-randomization behavior.
"""

from typing import Optional
import math

import numpy as np
from irsim.config import env_param
from irsim.lib import register_behavior
from irsim.util.util import WrapToPi, get_transform


class LevyFlightController:
    """
    Sensor-driven Lévy walk controller implementing power-law distributed random walks.

    The controller alternates between:
    1) Rotating to a target heading
    2) Moving forward for a Lévy-distributed flight distance

    Supports both Sensor-Driven Lévy Walk (SDLW) with sensor-adaptive heading and
    Uniform Heading Lévy Walk (UHLW) baseline with uniform random headings.

    Features (aligned with arXiv:2004.01562v6):
    - The Lévy exponent α is sampled from (2,3) (default: once per agent).
    - Flight distances are sampled as integer jumps with tail ~ d^{-α}.
    - Heading selection adapts based on mode (sensor-adaptive or uniform random).
    """

    class State:
        MOVING = 0
        ROTATING = 1

    def __init__(
        self,
        max_speed=0.5,
        max_yaw_rate=0.5,
        *,
        alpha_min=2.0,
        alpha_max=3.0,
        min_flight_distance=0.5,
        max_flight_distance=20.0,
        collision_threshold=0.4,
        heading_tolerance_rad=0.087,  # ~5 degrees
        back_sensor_weight=0.3,  # bias against backward turns (0..1)
        kappa_min=0.6,  # von Mises concentration lower bound
        kappa_max=10.0,  # von Mises concentration upper bound
        mode='reactive_levy',  # 'reactive_levy' (SDLW), 'unbiased_levy' (UHLW)
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None
    ):
        # Motion limits
        self.max_speed = float(max_speed)
        self.max_yaw_rate = float(max_yaw_rate)

        # Mode configuration (heading-selection policy)
        self.mode = str(mode)

        # Lévy exponent configuration
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)

        # Flight distance bounds
        self.min_flight_distance = float(min_flight_distance)
        self.max_flight_distance = float(max_flight_distance)

        # Obstacle avoidance
        self.collision_threshold = float(collision_threshold)

        # Heading tolerance for rotation completion
        self.heading_tolerance_rad = float(heading_tolerance_rad)

        # Backward turn bias
        self.back_sensor_weight = float(back_sensor_weight)

        # Von Mises concentration parameters
        self.kappa_min = float(kappa_min)
        self.kappa_max = float(kappa_max)

        # Validate all parameters
        self._validate_parameters()

        # State machine
        self.state = self.State.ROTATING

        # Flight parameters
        self.target_heading = 0.0
        self.target_distance = 0.0
        self.distance_traveled = 0.0
        self.start_position = np.array([0.0, 0.0], dtype=float)

        # RNG
        self.rng = rng if rng is not None else np.random.default_rng(seed)

        # Initialize α per agent and precompute weights
        self._initialize_alpha()
        self._precompute_levy_weights()

    # ----
    # Parameter validation
    # ----

    def _validate_parameters(self) -> None:
        """
        Validate parameters against SDLW constraints and general sensibility.
        Raises ValueError if any parameter is out of bounds.
        """
        # Lévy exponent bounds: randomized SDLW uses the Clementi et al. range (2,3).
        if not (0.0 < self.alpha_min <= 3.0):
            raise ValueError(f"alpha_min must be in (0, 3], got {self.alpha_min}")
        if not (2.0 <= self.alpha_max <= 3.0):
            raise ValueError(f"alpha_max must be in [2.0, 3.0], got {self.alpha_max}")
        if self.alpha_min > self.alpha_max:
            raise ValueError(f"alpha_min ({self.alpha_min}) must be <= alpha_max ({self.alpha_max})")

        # Flight distance bounds
        if self.min_flight_distance <= 0:
            raise ValueError(f"min_flight_distance must be > 0, got {self.min_flight_distance}")
        if self.max_flight_distance <= 0:
            raise ValueError(f"max_flight_distance must be > 0, got {self.max_flight_distance}")
        if self.min_flight_distance > self.max_flight_distance:
            raise ValueError(f"min_flight_distance ({self.min_flight_distance}) must be <= "
                           f"max_flight_distance ({self.max_flight_distance})")

        # Collision threshold
        if self.collision_threshold < 0:
            raise ValueError(f"collision_threshold must be >= 0, got {self.collision_threshold}")

        # Heading tolerance
        if self.heading_tolerance_rad < 0:
            raise ValueError(f"heading_tolerance_rad must be >= 0, got {self.heading_tolerance_rad}")

        # Backward sensor weight must satisfy 0 ≤ β_B ≤ 1.
        if not (0.0 <= self.back_sensor_weight <= 1.0):
            raise ValueError(f"back_sensor_weight must be in [0.0, 1.0], got {self.back_sensor_weight}")

        # Von Mises concentration parameters
        if self.kappa_min <= 0:
            raise ValueError(f"kappa_min must be > 0, got {self.kappa_min}")
        if self.kappa_max <= 0:
            raise ValueError(f"kappa_max must be > 0, got {self.kappa_max}")
        if self.kappa_min > self.kappa_max:
            raise ValueError(f"kappa_min ({self.kappa_min}) must be <= kappa_max ({self.kappa_max})")

    # -----------------------------
    # Lévy exponent and step length
    # -----------------------------

    def sample_levy_exponent(self) -> float:
        """
        Sample Lévy exponent α uniformly in (alpha_min, alpha_max).
        The randomized SDLW strategy uses α ∈ (2,3).
        """
        # Clamp the interval to (2,3) for sanity, but keep user overrides
        lo = max(2.0, min(self.alpha_min, self.alpha_max))
        hi = min(3.0, max(self.alpha_min, self.alpha_max))
        # If user gave a fixed α (lo == hi), return it
        if hi <= lo:
            return float(lo)
        return float(self.rng.uniform(lo, hi))

    def _initialize_alpha(self):
        """
        Initialize α per agent (sample on first call only).
        If alpha_min == alpha_max a fixed α is used (backwards-compatible
        behaviour when `levy_exponent` is supplied). Otherwise sample α ~ U(alpha_min, alpha_max).
        """
        # If user specified a fixed exponent via alpha_min==alpha_max, use it
        if abs(self.alpha_max - self.alpha_min) < 1e-12:
            self.current_alpha = float(self.alpha_min)
        else:
            self.current_alpha = self.sample_levy_exponent()

    def _precompute_levy_weights(self):
        """
        Precompute probability weights for discrete Lévy sampling to avoid
        recalculation on every step sample.
        """
        alpha = self.current_alpha if self.current_alpha is not None else 2.5
        min_i = max(1, int(np.ceil(self.min_flight_distance)))
        max_i = max(min_i, int(np.floor(self.max_flight_distance)))
        
        values = np.arange(min_i, max_i + 1, dtype=float)
        weights = values ** (-alpha)
        weights_sum = float(np.sum(weights))
        
        if weights_sum > 0.0:
            self.levy_values = values
            self.levy_probs = weights / weights_sum
        else:
            # Fallback: uniform distribution if weights fail
            self.levy_values = values
            self.levy_probs = np.ones_like(values) / len(values)

    def generate_levy_step(self) -> float:
        """
        Generate an integer-valued flight length with power-law tail p(d) ∝ d^{-α}, α ∈ (2,3):
            P(d=0)=1/2,
            P(d=i) ∝ i^{-α} for i ∈ [ceil(d_min), floor(d_max)].
        """
        alpha = self.current_alpha if self.current_alpha is not None else 2.5  # fallback

        d = self._generate_discrete_levy_step(alpha)
        env_param.logger.debug(f'Generated discrete Lévy distance: {d} (alpha={alpha:.2f})')

        return float(d)

    def _generate_discrete_levy_step(self, alpha: float) -> int:
        """
        Sample integer distance d using the discrete Lévy distribution:
            P(d=0)=1/2,
            P(d=i) ∝ i^{-α} for i >= 1 (truncated to [d_min, d_max]).
        
        Uses precomputed probability weights for efficiency.
        """
        # With probability 1/2, stay put before sampling a new heading.
        if self.rng.uniform(0.0, 1.0) < 0.5:
            return 0

        # Use precomputed weights
        return int(self.rng.choice(self.levy_values, p=self.levy_probs))

    # -----------------------------
    # Heading generator
    # -----------------------------
    def select_next_heading(
        self,
        current_heading: float,
        front_range: float,
        left_range: float,
        right_range: float,
        back_range: float
    ) -> float:
        """
        Select next heading using sensor-weighted von Mises distribution.
        Biases toward open space while maintaining controlled randomness.
        Penalizes backward turns via back_sensor_weight.

        For 'unbiased_levy' baseline, returns uniform random heading.
        """
        # Baselines with uniform random heading
        if self.mode == 'unbiased_levy':
            return WrapToPi(float(self.rng.uniform(-np.pi, np.pi)))

        # Sensor-adaptive heading (reactive_levy)
        sensor_angles = [0.0, np.pi / 2, -np.pi / 2, np.pi]  # front, left, right, back
        sensor_ranges = [front_range, left_range, right_range, back_range]
        sensor_weights = [1.0, 1.0, 1.0, self.back_sensor_weight]  # Penalize back sensor

        weighted_x = 0.0
        weighted_y = 0.0
        total_weight = 0.0

        for angle, distance, weight_mult in zip(sensor_angles, sensor_ranges, sensor_weights):
            weight = (distance * distance) * weight_mult  # emphasize clearer directions, penalize back
            abs_angle = current_heading + angle
            weighted_x += weight * np.cos(abs_angle)
            weighted_y += weight * np.sin(abs_angle)
            total_weight += weight

        if total_weight > 1e-6:
            # 1) Weighted-mean direction (deterministic resultant)
            mu = float(np.arctan2(weighted_y, weighted_x))

            # 2) Normalized resultant strength (0..1), stored as openness
            #    A dominant weighted direction gives a higher value.
            #    Cancellation between opposing contributions gives a lower value.
            resultant_mag = float(np.hypot(weighted_x, weighted_y))
            openness = resultant_mag / (total_weight + 1e-9)  # normalized 0..1

            # 3) Map normalized resultant strength to von Mises concentration
            #    Higher strength -> lower kappa -> broader sampling around mu
            #    Lower strength -> higher kappa -> narrower sampling around mu
            kappa = self.kappa_max - (self.kappa_max - self.kappa_min) * np.clip(openness, 0.0, 1.0)

            # 4) Sample final heading around mu (controlled randomness)
            sampled = float(self.rng.vonmises(mu=mu, kappa=kappa))

            return WrapToPi(sampled)

        # Near-zero total sensor weight - turn around
        return WrapToPi(current_heading + np.pi)

    # -----------------------------
    # Main update
    # -----------------------------

    def update(
        self,
        current_pos_x: float,
        current_pos_y: float,
        current_heading: float,
        front_range: float,
        left_range: float,
        right_range: float,
        back_range: float
    ):
        """
        One control tick: update state and return (vx, vy, wz) in body frame.
        """
        vel_x = 0.0
        vel_y = 0.0
        vel_w = 0.0

        # Simple collision detection
        obstacle_ahead = front_range < self.collision_threshold

        # =======================
        # State transitions
        # =======================
        if self.state == self.State.ROTATING:
            # Check if rotation complete
            heading_error = WrapToPi(self.target_heading - current_heading)
            if abs(heading_error) < self.heading_tolerance_rad:
                # Start a new flight segment
                self.state = self.State.MOVING
                self.target_distance = self.generate_levy_step()
                self.distance_traveled = 0.0
                self.start_position = np.array([current_pos_x, current_pos_y], dtype=float)

        elif self.state == self.State.MOVING:
            # Update traveled distance
            current_pos = np.array([current_pos_x, current_pos_y], dtype=float)
            displacement = current_pos - self.start_position
            self.distance_traveled = float(np.linalg.norm(displacement))

            # If target distance reached or obstacle ahead, rotate to new heading
            if (self.distance_traveled >= self.target_distance) or obstacle_ahead:
                self.state = self.State.ROTATING

                # Use sensor-weighted heading selection in both cases
                self.target_heading = self.select_next_heading(
                    current_heading, front_range, left_range, right_range, back_range
                )

        # =======================
        # Common obstacle avoidance: lateral nudges apply in both states
        # =======================
        if left_range < self.collision_threshold:
            vel_y = -0.2  # move right
        elif right_range < self.collision_threshold:
            vel_y = 0.2   # move left

        # =======================
        # State actions
        # =======================
        if self.state == self.State.ROTATING:
            # Rotate at max yaw rate toward target heading
            heading_error = WrapToPi(self.target_heading - current_heading)
            vel_w = self.max_yaw_rate if heading_error > 0.0 else -self.max_yaw_rate

            # Small axial nudges: back up if front is too close; move forward if back is too close
            if front_range < self.collision_threshold:
                vel_x = -0.2
            elif back_range < self.collision_threshold:
                vel_x = 0.2

        elif self.state == self.State.MOVING:
            # Forward motion unless obstacle too close
            if obstacle_ahead:
                vel_x = 0.0
            else:
                vel_x = self.max_speed

        return vel_x, vel_y, vel_w


@register_behavior("omni", "levy")
def beh_omni_levy(ego_object, **kwargs):
    """
    Lévy behavior for omni-directional robots.

    Args (kwargs):
      - max_vel (float): max forward velocity [m/s] (default 0.5)
      - max_yaw (float): max yaw rate [rad/s] (default 0.5)

      - alpha_min (float): lower bound for α sampling, must be in (0, 3] (default 2.0)
      - alpha_max (float): upper bound for α sampling, must be in [2, 3] (default 3.0)
      - min_flight_distance (float): min flight distance [m] (default 0.5)
    - max_flight_distance (float): max flight distance [m] (default 20.0)
      - collision_threshold (float): obstacle threshold [m] (default 0.4)
      - back_sensor_weight (float): multiplier for back sensor [0, 1].
            Biases against backward turns. 0=never use back, 1=equal weight (default 0.3)
      - kappa_min (float): lower von Mises concentration bound approached at high
            normalized resultant strength; broader sampling around the mean.
            Must be > 0 (default 0.6)
      - kappa_max (float): upper von Mises concentration bound approached at low
            normalized resultant strength; narrower sampling around the mean.
            Must be > 0 (default 10.0)
      - mode (str): 'reactive_levy' (SDLW) or 'unbiased_levy' (UHLW) (default 'reactive_levy')

      Backwards compatibility:
      - levy_exponent (float): if provided and alpha_min/alpha_max not given,
        we will fix α to this value (alpha_min = alpha_max = levy_exponent)
    """
    # Extract common parameters
    max_vel = float(kwargs.get("max_vel", 0.5))
    max_yaw = float(kwargs.get("max_yaw", 0.5))

    # Backward-compatible handling of a fixed levy_exponent
    fixed_alpha = kwargs.get("levy_exponent", None)
    alpha_min = kwargs.get("alpha_min", None)
    alpha_max = kwargs.get("alpha_max", None)

    if fixed_alpha is not None and (alpha_min is None and alpha_max is None):
        # Fix α to the provided levy_exponent
        alpha_min = float(fixed_alpha)
        alpha_max = float(fixed_alpha)
    else:
        # Defaults to the randomized strategy in (2,3)
        alpha_min = 2.0 if alpha_min is None else float(alpha_min)
        alpha_max = 3.0 if alpha_max is None else float(alpha_max)

    min_flight_distance = float(kwargs.get("min_flight_distance", 0.5))
    max_flight_distance = float(kwargs.get("max_flight_distance", 20.0))
    collision_threshold = float(kwargs.get("collision_threshold", 0.4))
    back_sensor_weight = float(kwargs.get("back_sensor_weight", 0.3))
    kappa_min = float(kwargs.get("kappa_min", 0.6))
    kappa_max = float(kwargs.get("kappa_max", 10.0))
    mode = str(kwargs.get("mode", "reactive_levy"))

    # Identify robot
    obj_id = getattr(ego_object, "id", 0)
    behavior_seed = kwargs.get("seed", None)
    controller_seed = None if behavior_seed is None else int(behavior_seed) + (int(obj_id) * 1000)
    controller_signature = (
        max_vel,
        max_yaw,
        alpha_min,
        alpha_max,
        min_flight_distance,
        max_flight_distance,
        collision_threshold,
        back_sensor_weight,
        kappa_min,
        kappa_max,
        mode,
        controller_seed,
    )

    if getattr(ego_object, "_sdlw_levy_controller_signature", None) != controller_signature:
        ego_object._sdlw_levy_controller = LevyFlightController(
            max_speed=max_vel,
            max_yaw_rate=max_yaw,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            min_flight_distance=min_flight_distance,
            max_flight_distance=max_flight_distance,
            collision_threshold=collision_threshold,
            back_sensor_weight=back_sensor_weight,
            kappa_min=kappa_min,
            kappa_max=kappa_max,
            mode=mode,
            seed=controller_seed,
        )
        ego_object._sdlw_levy_controller_signature = controller_signature
    controller = ego_object._sdlw_levy_controller

    # Extract robot state
    state = ego_object.state
    current_pos_x = float(state[0, 0])
    current_pos_y = float(state[1, 0])
    current_heading = float(state[2, 0])

    # Extract lidar distances from sensors (expect at least 4: front, left, right, back)
    if len(ego_object.sensors) < 4:
        raise ValueError(
            f"Expected at least 4 sensors (front, left, right, back), got {len(ego_object.sensors)}"
        )

    lidar_distances = []
    for sensor in ego_object.sensors[:4]:
        points = sensor.get_points()
        if points is not None and points.shape[1] > 0:
            # Use mean point to estimate distance (simple, robust)
            mx = float(np.mean(points[0, :]))
            my = float(np.mean(points[1, :]))
            distance = math.hypot(mx, my)
            lidar_distances.append(distance)
        else:
            # No detection: treat as clear to max range
            lidar_distances.append(float(sensor.range_max))

    front_range, left_range, right_range, back_range = lidar_distances

    # Update controller and get body-frame (vx, vy, wz)
    cmd_vx, cmd_vy, cmd_wz = controller.update(
        current_pos_x, current_pos_y, current_heading,
        front_range, left_range, right_range, back_range
    )

    # Convert body velocity to world frame (2D)
    trans, rot = get_transform(state)
    velocity_body = np.array([[cmd_vx], [cmd_vy]], dtype=float)
    velocity_world = rot @ velocity_body  # shape (2,1)

    # Store yaw rate for the simulator integrator.
    if not hasattr(ego_object, 'commands'):
        ego_object.commands = {}
    ego_object.commands['yaw_rate'] = float(cmd_wz)

    return velocity_world
