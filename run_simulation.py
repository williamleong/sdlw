#!/usr/bin/env python3
"""
Run one SDLW simulation in IR-SIM with a selected arena, team size,
controller mode, seed, and duration.
"""

import argparse
import csv
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import numpy as np
import yaml

from config import COVERAGE_CONFIG, OBSTACLE_GEN_CONFIG, RENDER_CONFIG, ROBOT_CONFIG, SPAWN_ZONES
from metrics_utils import (
    ExperimentMetrics,
    compute_coverage_from_headless,
    init_headless_coverage,
    rasterize_obstacles_to_grid,
    update_headless_coverage,
)
from position_generation import generate_grid_formation


class ObstacleShape(TypedDict, total=False):
    name: str
    radius: float
    length: float
    width: float


class ObstacleState(TypedDict):
    shape: ObstacleShape
    state: List[float]


@dataclass(frozen=True)
class SimulationConfig:
    arena: str
    team_size: int
    mode: str
    seed: int
    max_time: float
    arena_dir: str
    save_animation: bool
    animation_format: str

    def __post_init__(self) -> None:
        if self.team_size <= 0:
            raise ValueError("team_size must be greater than zero")
        if self.max_time <= 0:
            raise ValueError("max_time must be greater than zero")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one SDLW simulation",
        allow_abbrev=False,
    )
    parser.add_argument("--arena", default="open", help="Arena YAML stem to load, e.g. open")
    parser.add_argument("--team-size", type=positive_int, default=4, help="Number of robots in the team")
    parser.add_argument(
        "--mode",
        choices=["reactive_levy", "unbiased_levy"],
        default="reactive_levy",
        help="Controller mode to run",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for this simulation")
    parser.add_argument("--max-time", type=positive_float, default=120.0, help="Maximum simulation time in seconds")
    parser.add_argument("--output", default="results/simulation", help="Output folder for summary and coverage files")
    parser.add_argument("--arena-dir", default="arenas", help="Directory containing arena YAML files")
    parser.add_argument(
        "--no-save-animation",
        dest="save_animation",
        action="store_false",
        default=True,
        help="Disable animation output; animations are saved by default",
    )
    parser.add_argument(
        "--animation-format",
        choices=["mp4", "gif"],
        default="mp4",
        help="Animation format for saved animations",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-simulation progress")
    return parser


def build_run_id(arena: str, team_size: int, mode: str, seed: int) -> str:
    return f"{arena}_team{team_size}_{mode}_seed{seed}"


def write_simulation_outputs(
    output_folder: Path,
    run_id: str,
    config: SimulationConfig,
    metrics: ExperimentMetrics,
    status: str,
) -> Tuple[Path, Optional[Path]]:
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    coverage_path = None
    if metrics.coverage_samples:
        coverage_path = output_folder / "coverage.csv"
        with open(coverage_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "coverage"])
            writer.writerows(metrics.coverage_samples)

    summary = {
        "run_id": run_id,
        "status": status,
        "simulation": asdict(config),
        "metrics": metrics.to_dict(),
    }
    summary_path = output_folder / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    return summary_path, coverage_path


def generate_random_obstacles(
    world_size: Tuple[float, float],
    n_obstacles: int,
    seed: int,
    min_radius: Optional[float] = None,
    max_radius: Optional[float] = None,
    forbidden_zones: Optional[List[Tuple[float, float, float, float]]] = None,
) -> List[ObstacleState]:
    if min_radius is None:
        min_radius = OBSTACLE_GEN_CONFIG["min_radius"]
    if max_radius is None:
        max_radius = OBSTACLE_GEN_CONFIG["max_radius"]

    rng = np.random.RandomState(seed)
    obstacles: List[ObstacleState] = []
    width, height = world_size
    margin = max_radius + 0.5
    max_attempts = n_obstacles * 100

    for _ in range(max_attempts):
        if len(obstacles) >= n_obstacles:
            break

        x = rng.uniform(margin, width - margin)
        y = rng.uniform(margin, height - margin)
        radius = rng.uniform(min_radius, max_radius)

        if forbidden_zones:
            in_forbidden_zone = False
            for z_xmin, z_xmax, z_ymin, z_ymax in forbidden_zones:
                clearance = OBSTACLE_GEN_CONFIG["obstacle_clearance"]
                if (
                    z_xmin - radius - clearance < x < z_xmax + radius + clearance
                    and z_ymin - radius - clearance < y < z_ymax + radius + clearance
                ):
                    in_forbidden_zone = True
                    break
            if in_forbidden_zone:
                continue

        overlaps_existing = False
        for obs in obstacles:
            obs_x, obs_y = obs["state"][0], obs["state"][1]
            obs_r = obs["shape"].get("radius", 0.0)
            distance = np.hypot(x - obs_x, y - obs_y)
            if distance < radius + obs_r + OBSTACLE_GEN_CONFIG["obstacle_clearance"]:
                overlaps_existing = True
                break

        if not overlaps_existing:
            obstacles.append(
                {
                    "shape": {"name": "circle", "radius": float(radius)},
                    "state": [float(x), float(y), 0.0],
                }
            )

    return obstacles


def resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return Path(__file__).parent / path


def load_simulation_config(config: SimulationConfig) -> Dict[str, Any]:
    arena_dir = resolve_repo_path(config.arena_dir)
    yaml_path = arena_dir / f"{config.arena}.yaml"
    if not yaml_path.exists():
        available = sorted(p.stem for p in arena_dir.glob("*.yaml")) if arena_dir.exists() else []
        raise FileNotFoundError(f"Arena YAML not found: {yaml_path}. Available arenas: {available}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        arena_config = yaml.safe_load(f)

    robot_config = arena_config["robot"][0]
    robot_config["number"] = config.team_size
    robot_config["distribution"] = {"name": "manual"}
    robot_config["behavior"]["mode"] = config.mode
    robot_config["behavior"]["seed"] = int(config.seed)

    arena_config.setdefault("world", {})
    arena_config["world"].setdefault("plot", {})
    arena_config["world"]["plot"]["show_title"] = False

    all_obstacles = list(arena_config.get("obstacle", []))
    if config.arena.startswith("cluttered"):
        n_obstacles = int(config.arena.split("_")[1])
        spawn_zone = SPAWN_ZONES.get(config.arena)
        generated = generate_random_obstacles(
            (20, 20),
            n_obstacles,
            config.seed,
            forbidden_zones=[spawn_zone] if spawn_zone else None,
        )
        all_obstacles.extend(generated)
        arena_config["obstacle"] = all_obstacles

    spawn_zone = SPAWN_ZONES.get(config.arena, (2, 18, 2, 18))
    arena_config["world"]["plot"]["robot_spawn_zone"] = list(spawn_zone)
    robot_radius = robot_config["shape"].get("radius", ROBOT_CONFIG["radius"])

    try:
        grid_positions = generate_grid_formation(
            spawn_zone,
            all_obstacles,
            robot_radius,
            config.team_size,
            min_robot_separation=ROBOT_CONFIG["min_separation"],
        )
    except RuntimeError as exc:
        raise RuntimeError(
            f"Cannot place {config.team_size} robots in arena '{config.arena}' "
            f"within spawn zone {spawn_zone}. Reduce the team size or adjust the arena spawn zone."
        ) from exc

    robot_config["state"] = [[float(x), float(y), float(np.pi)] for x, y in grid_positions]
    return arena_config


def write_debug_yaml(output_folder: Path, run_id: str, arena_config: Dict[str, Any]) -> Path:
    yaml_dir = output_folder / "yaml"
    yaml_dir.mkdir(parents=True, exist_ok=True)
    debug_yaml_path = yaml_dir / f"{run_id}.yaml"
    with open(debug_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(arena_config, f)
    return debug_yaml_path


def create_environment(config_path: Path, config: SimulationConfig, output_folder: Path, run_id: str):
    import irsim

    env = irsim.make(
        str(config_path),
        display=False,
        seed=config.seed,
        log_level="WARNING",
        save_ani=config.save_animation,
    )

    if config.save_animation:
        animation_dir = output_folder / "animations"
        buffer_dir = output_folder / "animation_buffers" / run_id
        animation_dir.mkdir(parents=True, exist_ok=True)
        buffer_dir.mkdir(parents=True, exist_ok=True)
        env.path_param.ani_buffer_path = str(buffer_dir)
        env.path_param.ani_path = str(animation_dir)
        env._env_plot.show_title = True

    return env


def setup_coverage(env, arena_config: Dict[str, Any]) -> Dict[str, Any]:
    world_bounds = (
        (env._world.x_range[0], env._world.x_range[1]),
        (env._world.y_range[0], env._world.y_range[1]),
    )
    coverage_state = init_headless_coverage(world_bounds, grid_res=COVERAGE_CONFIG["grid_resolution"])
    rasterize_obstacles_to_grid(coverage_state, arena_config.get("obstacle", []))
    return coverage_state


def execute_simulation_loop(env, metrics: ExperimentMetrics, coverage_state: Dict[str, Any], config: SimulationConfig, run_id: str, verbose: bool) -> None:
    def simulate_robot_yaw(robots, step_time):
        for robot in robots:
            if hasattr(robot, "commands") and "yaw_rate" in robot.commands:
                robot._state[2, 0] += robot.commands["yaw_rate"] * step_time

    robots_collided = set()
    previous_positions = {
        robot.id: np.array([robot.state[0, 0], robot.state[1, 0]])
        for robot in env.robot_list
    }
    step_count = 0
    last_progress_time = 0.0
    coverage_overlay = None

    while env.time < config.max_time:
        env.step()
        step_count += 1
        simulate_robot_yaw(env.robot_list, env.step_time)

        for robot in env.robot_list:
            current_position = np.array([robot.state[0, 0], robot.state[1, 0]])
            previous_position = previous_positions.get(robot.id)
            if previous_position is not None:
                metrics.total_distance += float(np.linalg.norm(current_position - previous_position))
            previous_positions[robot.id] = current_position

            if getattr(robot, "collision_flag", False) and robot.id not in robots_collided:
                robots_collided.add(robot.id)
                metrics.record_collision_step(env.time, 1)

        update_headless_coverage(coverage_state, env.robot_list)
        if step_count % 20 == 0:
            metrics.record_coverage(env.time, compute_coverage_from_headless(coverage_state))

        if verbose and env.time - last_progress_time >= 30.0:
            latest_coverage = metrics.coverage_samples[-1][1] if metrics.coverage_samples else 0.0
            print(
                f"[{run_id}] t={env.time:.1f}s/{config.max_time:.1f}s, "
                f"coverage={latest_coverage:.1%}, collisions={metrics.collision_count}"
            )
            last_progress_time = env.time

        if config.save_animation:
            env._env_plot.ax.set_title(f"Simulation Time: {env.time:.2f}s", pad=3)
            visited_counts = coverage_state.get("visit_counts", coverage_state["visited"]).copy()
            visited_counts[coverage_state["obstacles"]] = 0
            visited_grid = np.log1p(visited_counts)
            if visited_grid.max() > 0:
                visited_grid = visited_grid / visited_grid.max()

            if coverage_overlay is None:
                extent = list(env._world.x_range) + list(env._world.y_range)
                coverage_overlay = env._env_plot.ax.imshow(
                    visited_grid,
                    cmap="YlOrRd",
                    alpha=0.7,
                    origin="lower",
                    extent=extent,
                    vmin=0,
                    vmax=1,
                    zorder=0.5,
                    interpolation="nearest",
                )
            else:
                coverage_overlay.set_data(visited_grid)
            env.render(interval=RENDER_CONFIG["interval"])

    update_headless_coverage(coverage_state, env.robot_list)
    metrics.record_coverage(env.time, compute_coverage_from_headless(coverage_state))
    metrics.elapsed_time = env.time


def finish_environment(env, config: SimulationConfig, run_id: str) -> None:
    if config.save_animation:
        suffix = ".mp4" if config.animation_format == "mp4" else ".gif"
        try:
            env._env_plot.ax.set_title(f"Simulation Time: {env.time:.2f}s", pad=3)
            env.render(interval=RENDER_CONFIG["interval"])
        finally:
            env.end(ending_time=0, ani_name=run_id, suffix=suffix, fps=RENDER_CONFIG["fps"])
    else:
        env.end()


def run_simulation(config: SimulationConfig, output_folder: Path, verbose: bool = False) -> Tuple[str, ExperimentMetrics, str]:
    random.seed(config.seed)
    np.random.seed(config.seed)

    run_id = build_run_id(config.arena, config.team_size, config.mode, config.seed)
    metrics = ExperimentMetrics()
    status = "complete"
    output_folder.mkdir(parents=True, exist_ok=True)

    arena_config = load_simulation_config(config)
    resolved_config_path = write_debug_yaml(output_folder, run_id, arena_config)

    env = None
    try:
        env = create_environment(resolved_config_path, config, output_folder, run_id)
        env.load_behavior("beh_omni_levy")
        coverage_state = setup_coverage(env, arena_config)
        execute_simulation_loop(env, metrics, coverage_state, config, run_id, verbose)
        finish_environment(env, config, run_id)
    except Exception as exc:
        status = "error"
        print(
            f"[{run_id}] simulation failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        if env is not None:
            try:
                env.end()
            except Exception:
                pass

    return run_id, metrics, status


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = SimulationConfig(
        arena=args.arena.strip().strip("/\\"),
        team_size=args.team_size,
        mode=args.mode,
        seed=args.seed,
        max_time=args.max_time,
        arena_dir=args.arena_dir,
        save_animation=args.save_animation,
        animation_format=args.animation_format,
    )
    output_folder = resolve_repo_path(args.output)

    run_id, metrics, status = run_simulation(config, output_folder, verbose=args.verbose)
    summary_path, coverage_path = write_simulation_outputs(output_folder, run_id, config, metrics, status)

    print(f"Simulation {status}: {run_id}")
    print(f"Summary: {summary_path}")
    if coverage_path:
        print(f"Coverage: {coverage_path}")

    return 0 if status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
