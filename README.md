# Sensor-Driven Lévy Walk (SDLW)

A simulator for Sensor-Driven Lévy Walk (SDLW) exploration by minimal-sensing robot teams, built on [IR-SIM](https://github.com/hanruihua/ir-sim).

## What is included

- `beh_omni_levy.py` — IR-SIM omni-directional behavior implementing:
  - `reactive_levy`: Sensor-Driven Lévy Walk (SDLW)
  - `unbiased_levy`: uniform-heading Lévy walk comparison mode
- `run_simulation.py` — single-run CLI for one arena, team size, controller mode, seed, and duration
- `metrics_utils.py` — collision, distance, and coverage tracking helpers
- `position_generation.py` — collision-free robot spawn placement utilities
- `arenas/` — YAML environments for open, rooms-and-corridors, and cluttered arenas
- `tests/` — pytest coverage for controller behavior, metrics, spawn placement, and the single-run CLI

## Setup

Create a workspace, then clone this repository and IR-SIM as sibling directories:

```bash
mkdir -p sdlw_ws
cd sdlw_ws

git clone https://github.com/williamleong/sdlw.git
git clone https://github.com/hanruihua/ir-sim.git

cd sdlw
conda env create -f environment.yml
conda activate sdlw-sim

cd ../ir-sim
git checkout v2.9.0
python -m pip install -e .

cd ../sdlw
```

Run commands from `sdlw_ws/sdlw` unless otherwise stated.

Tested with Python 3.11.15, IR-SIM 2.9.0, NumPy 2.4.6, PyYAML 6.0.3,
Matplotlib 3.11.0, ImageIO 2.37.3 with imageio-ffmpeg 0.6.0, Pillow 12.2.0, pytest 9.1.1,
and FFmpeg 7.1.1.

Verify the installation:

```bash
conda activate sdlw-sim
python -c "import irsim; print('IR-SIM import OK')"
pytest tests/ -q
```

On headless Linux systems, IR-SIM may print messages such as `Failed to use 'TkAgg' backend` or `Failed to use 'Qt5Agg' backend` before falling back to the non-interactive `Agg` backend. These messages are harmless for headless tests and simulations.

## Running

```bash
conda activate sdlw-sim
python run_simulation.py \
  --arena open \
  --team-size 4 \
  --mode reactive_levy \
  --seed 0 \
  --max-time 600 \
  --output results/open
```

Operational notes:

- The default duration is 120 simulated seconds; the examples below explicitly use 600 seconds.
- Animation is enabled by default and requires FFmpeg. Rendering and encoding increase wall-clock time and disk usage.
- Wall-clock runtime depends on arena complexity, team size, animation settings, and host performance; simulated seconds are not expected to match real seconds.
- For a quick headless smoke run, use a short `--max-time` together with `--no-save-animation`.
- The pinned environment and headless workflow have been validated on Linux x86_64. Other platforms are not currently tested automatically.

Useful options:

- `--arena`: arena YAML stem, e.g. `open`, `rooms_corridors`, or `cluttered_50`
- `--team-size`: number of robots to spawn
- `--mode`: `reactive_levy` or `unbiased_levy`
- `--seed`: random seed for obstacle generation, spawn placement, and controller behavior
- `--max-time`: simulation duration in seconds
- `--output`: output folder for this run
- `--no-save-animation`: disable animation output; animations are saved by default
- `--animation-format`: `mp4` or `gif`

Each run writes:

- `summary.json` — run configuration, status, and final metrics
- `coverage.csv` — coverage time series when coverage samples are available
- `yaml/<run_id>.yaml` — the resolved IR-SIM YAML used for the run
- `animations/<run_id>.<ext>` — saved by default unless `--no-save-animation` is used

Example with the `rooms_corridors` arena:

```bash
python run_simulation.py \
  --arena rooms_corridors \
  --team-size 4 \
  --mode reactive_levy \
  --seed 3 \
  --max-time 600 \
  --animation-format mp4 \
  --output results/rooms_corridors
```

Disable animation for a faster metrics-only run:

```bash
python run_simulation.py \
  --arena open \
  --team-size 1 \
  --max-time 600 \
  --no-save-animation \
  --output results/open_metrics_only
```

## Tests

```bash
conda activate sdlw-sim
pytest tests/ -v
```

## Controller configuration

In an IR-SIM YAML file, use the registered omni behavior named `levy`:

```yaml
behavior:
  name: levy
  mode: reactive_levy  # or unbiased_levy
  alpha_min: 2.0
  alpha_max: 3.0
  min_flight_distance: 0.5
  max_flight_distance: 20.0
  collision_threshold: 0.4
  back_sensor_weight: 0.3
  kappa_min: 0.6
  kappa_max: 10.0
```

When using `run_simulation.py`, command-line options override selected fields from the arena YAML. In particular, `--mode` overrides `behavior.mode`, and `--seed` sets `behavior.seed` in the resolved YAML. Other controller parameters, such as `alpha_min`, `alpha_max`, and `kappa_max`, come from the selected arena YAML unless you edit that YAML directly. The effective configuration for each run is saved to `yaml/<run_id>.yaml` in the output folder.

`reactive_levy` uses sensor-adaptive von Mises heading selection. It maps the
normalized sensor-resultant magnitude $o$ to concentration as
$\kappa(o)=\kappa_{\max}-(\kappa_{\max}-\kappa_{\min})o$. Thus,
higher normalized resultant strength produces broader angular sampling around
the weighted mean direction, with concentration approaching `kappa_min`;
lower strength produces narrower sampling, with concentration approaching
`kappa_max`. The metric measures directional imbalance in the weighted range
readings: a dominant direction increases its value, while opposing contributions
can cancel. The implementation stores this metric in the variable `openness`. `unbiased_levy` uses the same Lévy step-length
distribution with uniform random headings.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{leong2026decentralized,
  title={Decentralized Scalable Exploration via Emergent Adaptive L{\'e}vy Walks on Minimal-Sensing Platforms},
  author={Leong, Wai Lun and Teo, Swee Huat Rodney},
  booktitle={2026 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2026},
  address={Pittsburgh, PA, USA}
}
```

## License

MIT License. See `LICENSE`.
