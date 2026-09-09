"""
Tests for the single-run simulation CLI and output helpers.
"""

import json

import pytest
import yaml

from metrics_utils import ExperimentMetrics
from run_simulation import (
    SimulationConfig,
    build_parser,
    build_run_id,
    load_simulation_config,
    main,
    run_simulation,
    write_simulation_outputs,
)


def test_parser_uses_single_simulation_options(tmp_path):
    parser = build_parser()

    args = parser.parse_args([
        "--arena", "open",
        "--team-size", "4",
        "--mode", "reactive_levy",
        "--seed", "7",
        "--max-time", "30",
        "--output", str(tmp_path),
    ])

    assert args.arena == "open"
    assert args.team_size == 4
    assert args.mode == "reactive_levy"
    assert args.seed == 7
    assert args.max_time == 30
    assert args.output == str(tmp_path)
    assert args.save_animation is True


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--team-size", "0"),
        ("--team-size", "-1"),
        ("--max-time", "0"),
        ("--max-time", "-0.1"),
    ],
)
def test_parser_rejects_non_positive_simulation_values(option, value):
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([option, value])


@pytest.mark.parametrize(
    ("team_size", "max_time", "message"),
    [
        (0, 30.0, "team_size must be greater than zero"),
        (1, 0.0, "max_time must be greater than zero"),
    ],
)
def test_simulation_config_rejects_non_positive_values(team_size, max_time, message):
    with pytest.raises(ValueError, match=message):
        SimulationConfig(
            arena="open",
            team_size=team_size,
            mode="reactive_levy",
            seed=0,
            max_time=max_time,
            arena_dir="arenas",
            save_animation=False,
            animation_format="mp4",
        )


def test_parser_can_disable_default_animation():
    parser = build_parser()

    args = parser.parse_args(["--no-save-animation"])

    assert args.save_animation is False


def test_parser_rejects_redundant_save_animation_flag():
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--save-animation"])


@pytest.mark.parametrize("batch_flag", ["--n-runs", "--arenas", "--team-sizes", "--baselines", "--n-workers"])
def test_parser_rejects_batch_experiment_flags(batch_flag):
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([batch_flag, "2"])


def test_build_run_id_uses_single_simulation_terms():
    assert build_run_id("open", 4, "reactive_levy", 7) == "open_team4_reactive_levy_seed7"


def test_run_simulation_reports_runtime_errors_to_stderr(monkeypatch, tmp_path, capsys):
    config = SimulationConfig(
        arena="open",
        team_size=1,
        mode="reactive_levy",
        seed=0,
        max_time=0.2,
        arena_dir="arenas",
        save_animation=False,
        animation_format="mp4",
    )
    config_paths = []

    def fail_to_create_environment(config_path, *_args):
        config_paths.append(config_path)
        raise RuntimeError("injected environment failure")

    monkeypatch.setattr("run_simulation.create_environment", fail_to_create_environment)

    run_id, _metrics, status = run_simulation(config, tmp_path)
    captured = capsys.readouterr()

    assert status == "error"
    assert captured.err == (
        f"[{run_id}] simulation failed: RuntimeError: injected environment failure\n"
    )
    expected_config_path = tmp_path / "yaml" / f"{run_id}.yaml"
    assert config_paths == [expected_config_path]
    assert expected_config_path.is_file()


def test_load_simulation_config_rejects_impossible_spawn_placement():
    config = SimulationConfig(
        arena="open",
        team_size=100,
        mode="reactive_levy",
        seed=0,
        max_time=30.0,
        arena_dir="arenas",
        save_animation=False,
        animation_format="mp4",
    )

    with pytest.raises(
        RuntimeError,
        match=r"Cannot place 100 robots in arena 'open'.*spawn zone \(16.5, 19.5, 1, 8\)",
    ):
        load_simulation_config(config)


@pytest.mark.parametrize("mode", ["reactive_levy", "unbiased_levy"])
def test_cli_runs_real_headless_simulation(mode, tmp_path):
    exit_code = main([
        "--arena", "open",
        "--team-size", "4",
        "--mode", mode,
        "--seed", "7",
        "--max-time", "0.2",
        "--no-save-animation",
        "--output", str(tmp_path),
    ])

    run_id = f"open_team4_{mode}_seed7"
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    resolved = yaml.safe_load((tmp_path / "yaml" / f"{run_id}.yaml").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert summary["status"] == "complete"
    assert summary["simulation"]["mode"] == mode
    assert summary["simulation"]["team_size"] == 4
    assert summary["simulation"]["seed"] == 7
    assert summary["metrics"]["elapsed_time"] >= 0.2
    assert (tmp_path / "coverage.csv").is_file()
    assert resolved["robot"][0]["number"] == 4
    assert resolved["robot"][0]["behavior"]["mode"] == mode
    assert resolved["robot"][0]["behavior"]["seed"] == 7


def test_write_simulation_outputs_creates_json_summary_and_coverage_csv(tmp_path):
    metrics = ExperimentMetrics()
    metrics.record_collision_step(1.0, 1)
    metrics.record_coverage(5.0, 0.25)
    metrics.record_coverage(10.0, 0.5)
    metrics.total_distance = 12.345
    metrics.elapsed_time = 10.0

    config = SimulationConfig(
        arena="open",
        team_size=4,
        mode="reactive_levy",
        seed=7,
        max_time=30.0,
        arena_dir="arenas",
        save_animation=False,
        animation_format="mp4",
    )

    summary_path, coverage_path = write_simulation_outputs(
        output_folder=tmp_path,
        run_id="open_team4_reactive_levy_seed7",
        config=config,
        metrics=metrics,
        status="complete",
    )

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    assert summary["run_id"] == "open_team4_reactive_levy_seed7"
    assert summary["status"] == "complete"
    assert summary["simulation"] == {
        "arena": "open",
        "team_size": 4,
        "mode": "reactive_levy",
        "seed": 7,
        "max_time": 30.0,
        "arena_dir": "arenas",
        "save_animation": False,
        "animation_format": "mp4",
    }
    assert summary["metrics"] == {
        "collision_count": 1,
        "total_distance": 12.35,
        "coverage": 0.5,
        "elapsed_time": 10.0,
    }

    assert coverage_path is not None
    assert coverage_path.read_text(encoding="utf-8").splitlines() == [
        "time,coverage",
        "5.0,0.25",
        "10.0,0.5",
    ]
