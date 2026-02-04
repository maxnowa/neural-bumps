# Neural Field Simulation Framework

A Python framework for simulating neural field dynamics on ring topologies, designed to study attentional interactions in working memory. This package models how localized patterns of neural activity ("bumps") interact when multiple stimuli compete for representation, enabling investigation of attraction and repulsion biases observed in behavioral experiments.

## Background

Neural field models describe the spatiotemporal dynamics of large populations of neurons using continuous activity variables. When external input is applied to such a network with appropriate lateral connectivity, the activity can self-organize into stable localized patterns called **bumps**. These bumps are thought to underlie working memory representations.

In experiments, when subjects must remember a target location while ignoring a distractor, their recall is systematically biased—either attracted toward or repelled from the distractor position. This framework allows you to:

- Simulate bump dynamics with different kernel shapes
- Measure how bumps interact and shift over time
- Find kernel parameters that reproduce empirical bias patterns
- Compare theoretical predictions (simplified ODE model) with full PDE simulations

## Features

- **Two simulation modes**: Full FFT-based PDE integration or fast ODE-based simplified model
- **Single and dual ring architectures**: Model one or two coupled neural populations
- **Multiple kernel types**: Mexican hat, triple Gaussian, Kilpatrick exponential, and more
- **Stability analysis**: Automatically check whether a kernel supports stable bumps
- **Parameter search**: Monte Carlo exploration of kernel parameter space
- **Visualization suite**: Kernel profiles, space-time diagrams, trajectory plots, bias curves
- **Reproducible research**: Replicates results from Kilpatrick & Ermentrout (2018)

## Installation

```bash
# Clone the repository
git clone https://github.com/maxnowa/neural-bumps.git
cd neural-bumps

# Install as editable package
pip install -e .

# Or install with notebook support
pip install -e ".[notebook]"
```

**Requirements**: Python 3.9+, numpy, scipy, matplotlib, pandas, seaborn, tqdm, pyyaml

## Quick Start

### Basic Simulation

```python
from neural_field import SimulationConfig, create_simulator
from neural_field.config import SimulationMode, RingMode, LateralKernelType

# Create configuration
config = SimulationConfig(
    sim_mode=SimulationMode.FULL,
    ring_mode=RingMode.SINGLE,
)
config.lateral_kernel.kernel_type = LateralKernelType.KILPATRICK

# Create simulator and run a trial
simulator = create_simulator(config)
result = simulator.run_trial(target_loc=0.0, dist_loc=40.0)

print(f"Final bias: {result.bias:.2f} degrees")
```

### Measuring Bias Curves

```python
import numpy as np

# Test multiple target-distractor distances
distances = np.linspace(-90, 90, 19)
biases = []

for dist in distances:
    result = simulator.run_trial(target_loc=0.0, dist_loc=dist)
    biases.append(result.bias)

# Plot the bias curve
import matplotlib.pyplot as plt
plt.plot(distances, biases, 'o-')
plt.xlabel('Distractor distance (deg)')
plt.ylabel('Bias (deg)')
plt.axhline(0, color='gray', linestyle='--')
plt.show()
```

### Dual-Ring Model with Cross-Coupling

```python
from neural_field.config import CrossKernelParams, CrossKernelType

config = SimulationConfig(
    sim_mode=SimulationMode.SIMPLIFIED,
    ring_mode=RingMode.DUAL,
)
config.cross_kernel = CrossKernelParams(
    kernel_type=CrossKernelType.GAUSSIAN,
    J_cross=0.3,
    sig_cross=40.0,
)

simulator = create_simulator(config)
result = simulator.run_trial(target_loc=0.0, dist_loc=50.0)
```

## Simulation Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `FULL` | FFT-based PDE integration | Accurate dynamics, validation |
| `SIMPLIFIED` | ODE on bump centroids | Fast parameter sweeps |

| Ring Mode | Description |
|-----------|-------------|
| `SINGLE` | One neural population with lateral interactions |
| `DUAL` | Two populations with cross-ring coupling |

## Experiment Modes

The framework supports two experiment paradigms that mirror different experimental designs used in working memory research:

### Simultaneous Mode (default)

Both target and distractor are initialized at t=0 and persist throughout the simulation. This models scenarios where both stimuli are presented together and must be maintained in memory.

```python
result = simulator.run_trial(
    target_loc=0.0,
    dist_loc=40.0,
    experiment_type="simultaneous"
)
```

**Timeline:**
```
t=0                                              t=T_total
|================================================|
|  Target bump + Distractor bump (both active)   |
|================================================|
```

### Delayed Mode

Models the full experimental paradigm used in attention studies:

1. **Target presentation** — Target stimulus appears and forms a bump
2. **Delay period** — Target must be maintained in memory
3. **Distractor presentation** — Distractor appears briefly as external input
4. **Post-distractor delay** — Distractor disappears, only target bump remains
5. **Recall** — Final target position is measured (bias computed)

```python
result = simulator.run_trial(
    target_loc=0.0,
    dist_loc=40.0,
    experiment_type="delayed",
    dist_onset=200,   # Distractor appears at t=200ms
    dist_dur=500,     # Distractor lasts for 500ms
)
```

**Timeline:**
```
t=0        t=200      t=700                      t=T_total
|----------|==========|--------------------------|
|  Target  | Target + |  Target only             |
|  only    | Distractor|  (distractor removed)   |
|----------|==========|--------------------------|
           ↑          ↑
      dist_onset   dist_onset + dist_dur
```

This mode allows you to study how transient distractors influence sustained memory representations, which is the paradigm used in many behavioral experiments on attentional capture and working memory interference.

## Kernel Types

### Lateral Kernels (within-ring)

| Type | Description |
|------|-------------|
| `MEXICAN_HAT` | Classic center-surround profile |
| `TRIPLE_GAUSSIAN` | Excitation + inhibition + long-range attraction |
| `KILPATRICK` | Exponential decay form from Kilpatrick & Ermentrout |
| `QUAD_EXP` | Tunable rise/decay powers |

### Cross Kernels (between rings)

| Type | Description |
|------|-------------|
| `GAUSSIAN` | Simple Gaussian coupling |
| `TENT` | Linear decay profile |
| `DOUBLE_BUMP` | Two-peaked coupling |
| `MEXICAN_HAT` | Center-surround cross-coupling |

## Configuration

All parameters are organized into typed dataclasses:

```python
from neural_field.config import (
    SimulationConfig,
    GridParams,
    TimeParams,
    LateralKernelParams,
    CrossKernelParams,
    InputParams,
)

# Customize grid resolution
config = SimulationConfig()
config.grid.L = 180.0      # Half-domain size (degrees)
config.grid.dx = 0.05      # Spatial resolution

# Customize timing
config.time.dt = 0.01      # Time step (ms)
config.time.T_total = 1000 # Total duration (ms)
config.time.eps = 0.05     # Noise strength

# Customize input
config.input.amp_target = 0.5
config.input.amp_dist = 0.3
config.input.theta = 0.25  # Activation threshold
```

## Analysis Tools

### Stability Analysis

Check whether a kernel configuration supports stable bumps:

```python
from neural_field.analysis import check_stability, search_stable_kernels

# Check a specific kernel
is_stable, bump_width = check_stability(
    x=grid.x,
    dx=grid.dx,
    w_lat=kernel_function,
    theta=0.25
)

# Search parameter space for stable kernels
df_stable = search_stable_kernels(n_samples=50000)
```


## Visualization

```python
from neural_field.visualization import (
    plot_kernel,
    plot_potential_and_force,
    plot_results,
    plot_space_time,
)

# Visualize kernel profile
plot_kernel(x, kernel_values, theta=0.25)

# Show interaction force and potential
plot_potential_and_force(distances, force_values)

# Comprehensive trial results
plot_results(result, config)

# Space-time diagram of activity
plot_space_time(result.trajectory_full, grid.x, config.time)
```

## Project Structure

```
neural-bumps/
├── neural_field/           # Main package
│   ├── config/             # Configuration (params, enums, loader)
│   ├── core/               # Core abstractions (grid, results, math)
│   ├── kernels/            # Kernel implementations and physics
│   ├── simulation/         # Simulation engines (full, simplified)
│   ├── analysis/           # Stability and validation tools
│   ├── visualization/      # Plotting functions
│   └── io/                 # Data I/O utilities
├── notebooks/              # Jupyter notebooks with examples
├── data/                   # Pre-computed results
│   ├── stable_params.csv   # 50k stable kernel configurations
│   ├── candidates.csv      # Filtered kernels with velocity profiles
│   └── verified_bias_curves.csv
├── config/                 # Default YAML configuration
└── code/                   # Legacy API (backwards compatibility)
```

## Working with Results

The `TrialResult` object contains:

| Attribute | Description |
|-----------|-------------|
| `bias` | Final displacement from target (degrees) |
| `trajectory` | Time series of bump center position |
| `trajectory_full` | All bump positions (for multi-bump cases) |
| `interfaces` | Left and right bump edges over time |
| `initial_u` | Neural field state at trial start |
| `final_u` | Neural field state at trial end |


## Examples

See `notebooks/neural_field.ipynb` for comprehensive examples including:

- Replicating Kilpatrick & Ermentrout (2018) results
- Comparing full vs. simplified models
- Analyzing noise effects on bump dynamics
- Large-scale kernel parameter optimization
- Designing cross-ring coupling for specific bias patterns

## License

MIT License - see LICENSE file for details.
