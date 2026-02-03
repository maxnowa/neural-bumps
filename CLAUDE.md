# CLAUDE.md

## Project Overview

Neural field simulation framework for studying attentional interactions in working memory. Models how sustained neural activity patterns ("bumps") interact on a ring topology, simulating attention effects and distractor-induced memory biases.

**Research Focus:** Relationship between neural field kernel properties and behavioral outcomes (attraction/repulsion biases).

---

## Tech Stack

- **Language:** Python 3.9+
- **Scientific:** numpy, scipy, matplotlib, pandas, seaborn
- **Utilities:** tqdm, pyyaml, dataclasses (stdlib)
- **Environment:** Jupyter notebooks

---

## Project Structure

```
lab-rotation-schwalger/
├── pyproject.toml              # Package definition + dependencies
├── requirements.txt            # Pinned dependencies
├── config/
│   └── default.yaml            # Default configuration
│
├── neural_field/               # Main package (NEW)
│   ├── __init__.py             # Public API exports
│   ├── config/                 # Configuration management
│   │   ├── params.py           # Typed dataclasses
│   │   ├── enums.py            # SimulationMode, RingMode, KernelType
│   │   └── loader.py           # YAML loading utilities
│   ├── core/                   # Core abstractions
│   │   ├── grid.py             # SpatialGrid class
│   │   ├── results.py          # TrialResult dataclass
│   │   └── math.py             # wrap_deg(), circ_dist()
│   ├── kernels/                # Kernel implementations
│   │   ├── base.py             # KernelFunction protocol
│   │   ├── lateral.py          # MexicanHat, TripleGaussian, etc.
│   │   ├── cross.py            # Cross-ring kernels
│   │   ├── factory.py          # create_lateral_kernel(), create_cross_kernel()
│   │   └── physics.py          # BumpPhysics: width, stiffness, force
│   ├── simulation/             # Simulation engines
│   │   ├── base.py             # Simulator protocol
│   │   ├── full.py             # FullSimulator (FFT-based PDE)
│   │   ├── simplified.py       # SimplifiedSimulator (ODE)
│   │   ├── tracking.py         # BumpTracker class
│   │   └── factory.py          # create_simulator()
│   ├── analysis/               # Analysis tools
│   │   ├── stability.py        # check_stability(), search_stable_kernels()
│   │   └── validation.py       # run_empirical_validation()
│   ├── visualization/          # Plotting
│   │   ├── kernels.py          # plot_kernel(), plot_potential_and_force()
│   │   ├── simulation.py       # plot_results(), plot_space_time()
│   │   └── analysis.py         # plot_candidates()
│   └── io/                     # Data I/O
│       ├── paths.py            # Paths class
│       └── data.py             # load_stable_params(), save_results()
│
├── code/                       # Legacy code (backwards compatibility)
│   ├── compat.py               # Backwards-compatible wrappers
│   ├── Parameters.py           # (legacy)
│   ├── Kernels.py              # (legacy)
│   ├── Simulation.py           # (legacy)
│   └── ...
│
├── notebooks/                  # Jupyter notebooks
│   └── neural_field.ipynb      # Main results notebook
│
├── data/                       # Pre-computed results
│   ├── stable_params.csv
│   ├── candidates.csv
│   └── verified_bias_curves.csv
│
├── plots/                      # Generated figures
└── papers/                     # Reference literature
```

---

## Key Concepts

| Term | Description | Reference |
|------|-------------|-----------|
| **Bump** | Localized activity pattern on ring | `neural_field/simulation/tracking.py` |
| **w_lat** | Lateral (within-ring) kernel | `neural_field/kernels/lateral.py` |
| **w_cross** | Cross-ring coupling kernel | `neural_field/kernels/cross.py` |
| **theta** | Threshold for activation | `neural_field/config/params.py` |
| **h** | Bump half-width | `neural_field/kernels/physics.py` |
| **alpha** | Bump stiffness | `neural_field/kernels/physics.py` |
| **J_force** | Interaction force function | `neural_field/kernels/physics.py` |
| **bias** | Final displacement from target | `neural_field/core/results.py` |

---

## Running the Code

### Installation

```bash
# Install as editable package
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

### Quick Start (New API)

```python
from neural_field import SimulationConfig, create_simulator
from neural_field.config import SimulationMode, RingMode, LateralKernelType

# Create configuration
config = SimulationConfig(
    sim_mode=SimulationMode.FULL,
    ring_mode=RingMode.SINGLE,
)
config.lateral_kernel.kernel_type = LateralKernelType.KILPATRICK

# Create simulator and run
simulator = create_simulator(config)
result = simulator.run_trial(target_loc=0.0, dist_loc=40.0)
print(f"Bias: {result.bias:.2f}°")
```

### Quick Start (Legacy API)

```python
# For backwards compatibility during migration
from code.compat import ModelParams, Kernels, Simulation

p = ModelParams(sim_mode='full', ring_mode='single')
k = Kernels(p, kernel_type='mexican_hat')
s = Simulation(p, k)
results = s.run_trial(target_loc=0.0, dist_loc=40.0)
```

### Parameter Search

```python
from neural_field.analysis import search_stable_kernels
df = search_stable_kernels(n_samples=50000)
```

---

## Simulation Modes

| Mode | Enum | Description |
|------|------|-------------|
| **full** | `SimulationMode.FULL` | Full PDE with FFT convolution |
| **simplified** | `SimulationMode.SIMPLIFIED` | Reduced ODE on centroids |
| **single ring** | `RingMode.SINGLE` | One population |
| **dual ring** | `RingMode.DUAL` | Two populations with cross-coupling |

---

## Kernel Types

**Lateral kernels** (`LateralKernelType`):
- `TRIPLE_GAUSSIAN` - excitatory + inhibitory + attraction
- `MEXICAN_HAT` - classic neural field kernel
- `KILPATRICK` - exponential decay form
- `QUAD_EXP` - tunable rise/decay powers

**Cross kernels** (`CrossKernelType`):
- `GAUSSIAN`, `TENT`, `QUAD_EXP`, `DOUBLE_BUMP`, `MEXICAN_HAT`

---

## Data Files

| File | Contents |
|------|----------|
| `stable_params.csv` | 50k Monte Carlo search results with stable widths |
| `candidates.csv` | Filtered kernels with velocity profiles |
| `verified_bias_curves.csv` | Empirically validated bias curves |

---

## Additional Documentation

| Topic | File |
|-------|------|
| Design patterns & conventions | `.claude/docs/architectural_patterns.md` |
| Research notes & ideas | `lab-book.md` |
| Reference paper | `papers/s10827-018-0679-7.pdf` (Kilpatrick & Ermentrout) |
