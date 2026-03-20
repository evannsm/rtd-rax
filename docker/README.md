# Docker Setup

All RTD-RAX experiments run inside a Docker container with pre-installed dependencies.

## Dependencies

**Base image:** `osrf/ros:jazzy-desktop-full`

**Python packages** (installed in `/opt/rtd-venv`):
- numpy, scipy, matplotlib, shapely
- jax[cpu]
- [immrax](https://github.com/gtfactslab/immrax)

**System libraries** (required by immrax): `libcdd-dev`, `libgmp-dev`

## Getting Started

```bash
make build       # build the Docker image
make run_gui     # start container with X11 forwarding for plots
```

For GUI plotting on Linux/X11, allow Docker access first:

```bash
xhost +local:docker
```

After finishing, revoke access:

```bash
xhost -local:docker
```

## Container Lifecycle

| Command | Description |
|---|---|
| `make build` | Build the Docker image |
| `make run` | Start container (headless) |
| `make run_gui` | Start container with X11 forwarding |
| `make start` | Restart a stopped container |
| `make stop` | Stop the running container |
| `make kill` | Force-stop the container |
| `make attach` | Attach a shell to the running container |

If the container name is already in use:

```bash
make stop
make run_gui
```

## Running Experiments

### Basic Planners

```bash
make rtd-numpy                         # one-shot NumPy planner (GUI)
make rtd-gap                           # gap scenario, standard FRS (no path)
make rtd-gap FRS=noerror               # gap scenario, noerror FRS (path found)
make rtd-gap-verify FRS=noerror        # + immrax verification (safe)
make rtd-gap-verify FRS=noerror UNCERTAINTY=0.05 DISTURBANCE=0.01
```

### Journey Comparison (Receding-Horizon Replanning)

```bash
make rtd-journey-gap FRS=noerror JOURNEY_MAX_STEPS=80
make rtd-journey-gap-compare JOURNEY_VERIFY=1 JOURNEY_VERIFY_EVERY=1
make rtd-journey-gap-compare JOURNEY_VERIFY=1 JOURNEY_REPAIR=1
make rtd-journey-angled-compare JOURNEY_VERIFY=1 JOURNEY_REPAIR=1
```

### Case Study Suites

```bash
make rtd-case2-suite                   # Case 2 representative + two-repair outputs
make rtd-disturbance-compare           # randomized disturbance course comparison
```

### Figure Generation

```bash
make manuscript-figures                 # regenerate all manuscript assets
make manuscript-case1-gap-suite        # Case 1 figure family → figures/Study1_Gap/
make manuscript-case2-suite            # Case 2 figure family → figures/Study2_AngledObs/
make manuscript-disturbance-gallery    # disturbance comparison PDFs + GIFs
```

Individual figure targets:

```bash
make manuscript-gap-standard-fail                 # gap_standard_fail.pdf
make manuscript-gap-standard-fail-legend          # gap_standard_fail_legend.pdf
make manuscript-gap-noerror-verify                # gap_noerror_verify.pdf
make manuscript-gap-noerror-verify-legend         # gap_noerror_verify_legend.pdf
make manuscript-merge-gap                         # merged side-by-side gap figure
make manuscript-full-gap-noerror-verify-legend    # 3-panel (k-space + FRS + world)
make manuscript-angled-compare-legend-left        # angled obstacle compare with legend
make manuscript-angled-compare-no-legend          # angled obstacle compare without legend
make manuscript-disturbance-seed10                # disturbance compare, seed 10
make manuscript-disturbance-seed20                # disturbance compare, seed 20
make manuscript-disturbance-seed32                # disturbance compare, seed 32
make fix-perms                         # fix file permissions to host user
```

## Make Variables

Override any variable with `NAME=value`:

| Variable | Default | Description |
|---|---|---|
| `FRS` | `standard` | FRS variant (`standard` or `noerror`) |
| `UNCERTAINTY` | `0.01` | immrax positional uncertainty (m) |
| `DISTURBANCE` | `0.0` | Bounded additive disturbance on `[dpx, dpy, dh, dv]` |
| `JOURNEY_MAX_STEPS` | `60` | Maximum replanning iterations |
| `JOURNEY_GOAL_TOL` | `0.10` | Goal distance tolerance (m) |
| `JOURNEY_T_MOVE` | `0.5` | Execution time per replan step (s) |
| `JOURNEY_VERIFY` | `0` | Enable immrax verification (`1` to enable) |
| `JOURNEY_VERIFY_EVERY` | `1` | Verify every N replan steps |
| `JOURNEY_VERIFY_UNCERTAINTY` | `0.01` | Positional uncertainty for journey verification |
| `JOURNEY_VERIFY_DISTURBANCE` | `0.0` | Disturbance bound for journey verification |
| `JOURNEY_REPAIR` | `0` | Enable hybrid repair (`1` to enable) |
| `JOURNEY_REPAIR_MAX_ITERS` | `4` | Max repair iterations per step |
| `JOURNEY_REPAIR_SPEED_BACKOFF` | `0.15` | Speed decrement during repair |
| `JOURNEY_REPAIR_BUFFER_STEP` | `0.01` | Obstacle buffer increment per CEGIS iteration (m) |
| `DISTURBANCE_COMPARE_SEED` | `32` | Seed for disturbance course comparison |
| `DISTURBANCE_OBSTACLE_INSET` | `0.30` | Gate obstacle inset from road boundaries (m) |
| `IMAGE_NAME` | `ws_rtd_jazzy` | Docker image tag |
| `CONTAINER_NAME` | `ws_rtd` | Docker container name |

Examples:

```bash
make rtd-gap FRS=noerror
make rtd-gap-verify FRS=noerror UNCERTAINTY=0.05
make rtd-journey-gap-compare JOURNEY_VERIFY=1 JOURNEY_REPAIR=1
make build IMAGE_NAME=ws_rtd_custom
```

## Case 2 Reproduction

The representative Case 2 parameters are hard-coded in the makefile. To reproduce:

```bash
# Representative run (1 repair)
make rtd-angled-animate
make rtd-angled-repair-view

# Alternate run (2 repairs)
make rtd-angled-animate-two-repairs
make rtd-angled-repair-view-two-repairs

# Full suite (all of the above)
make rtd-case2-suite
make manuscript-case2-suite
```

## Notes

- The repository is mounted into the container at `/workspace/ws_RTD`, so local edits are immediately visible.
- Output figures are saved to `figures/` (grouped by study) and `case_study_outputs/`.
- Files are created with your host UID/GID. If permissions are wrong, run `make fix-perms`.
