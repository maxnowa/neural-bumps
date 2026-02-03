# Architectural Patterns

## Overview

This document captures recurring patterns and design decisions used throughout the neural field simulation codebase.

---

## Pattern 1: Dependency Injection via Composition

The codebase uses a clean layered architecture where each component receives its dependencies through constructors.

**Chain:** `ModelParams` → `Kernels` → `Simulation`

| Component | Receives | Reference |
|-----------|----------|-----------|
| Kernels | ModelParams | Kernels.py:8-26 |
| Simulation | ModelParams + Kernels | Simulation.py:9-13 |

This enables easy testing and parameter sweeps by swapping ModelParams instances.

---

## Pattern 2: Strategy Pattern for Simulation Modes

Runtime mode selection via `sim_mode` parameter:

- **full**: FFT-based PDE integration (Simulation.py:161-292)
- **simplified**: Reduced ODE on centroids only (Simulation.py:294-365)

Dispatch occurs at Simulation.py:140-159:
```
if p.sim_mode == "full" → _run_full()
else → _run_simplified()
```

---

## Pattern 3: Dual Ring Topology Support

Conditional dynamics based on `ring_mode`:

- **single**: One ring, self-interaction only (Simulation.py:244-246)
- **dual**: Two rings with cross-coupling (Simulation.py:247-270)

Cross-kernel (`w_cross`) only used in dual mode. See Kernels.py:70-115 for cross-kernel types.

---

## Pattern 4: Circular Domain Arithmetic

All positions use wraparound on [-180, 180]:

| Helper | Location | Purpose |
|--------|----------|---------|
| `wrap_deg(a)` | Simulation.py:16-17 | Normalize angle to [-180, 180] |
| `circ_dist(a, b)` | Simulation.py:19-20 | Shortest distance on ring |

Used throughout bump tracking and trajectory analysis.

---

## Pattern 5: FFT-Based Convolution

Spatial convolutions computed via FFT for O(n log n) efficiency:

1. **Pre-compute** kernel FFTs at initialization (Kernels.py:22-23)
2. **Apply** during integration: `ifft(fft(f) * W_fft)` (Simulation.py:237-250)

This is critical for performance with n_points = 7200.

---

## Pattern 6: Iterative Root Finding for Physics

Physics parameters derived by solving integral equations:

| Parameter | Equation Solved | Location |
|-----------|-----------------|----------|
| Bump width `h` | W_lat(d) = theta | Kernels.py:141-162 |
| Stable widths | Amari stability roots | stability.py:19-47 |

Uses `scipy.optimize.brentq` for robust bracketed root finding.

---

## Pattern 7: Result Dictionary Pattern

All simulations return structured dictionaries:

```python
{
    "bias": float,              # Final displacement from target
    "trajectory": ndarray,      # Target bump trajectory
    "trajectory_full": ndarray, # All bump trajectories
    "interfaces": tuple,        # (left_edges, right_edges)
    "initial_u": ndarray,       # Initial spatial profile
    "final_u": ndarray,         # Final spatial profile
}
```

Reference: Simulation.py:285-292

This enables polymorphic handling across plotting functions (plot_utils.py).

---

## Pattern 8: Lazy Initialization with Validation

`ModelParams.__post_init__` (Parameters.py:51-55) ensures consistency:

- Derives `n_points` from grid parameters
- Adjusts `dx` for perfect tiling
- Prevents downstream numerical errors

---

## Pattern 9: Mask-Based Spatial Operations

Stimulus initialization via boolean masks instead of explicit loops:

```python
target_mask = dist_from_target < half_width
u1[target_mask] += amplitude
```

Reference: Simulation.py:180-200

---

## Pattern 10: Temporal Windows for Stimulus Control

Time-gated inputs enable experimental paradigms:

```python
if (time >= onset) and (time < onset + duration):
    apply_stimulus()
```

Reference: Simulation.py:216-222

Supports both `simultaneous` and `standard` (delayed) conditions.

---

## Data Flow Summary

```
Parameters.py     → ModelParams (config container)
        ↓
Kernels.py        → w_lat, w_cross, physics (h, alpha, J_force)
        ↓
Simulation.py     → run_trial() → results dict
        ↓
plot_utils.py     → visualizations
stability.py      → parameter sweeps
physics_utils.py  → empirical validation
```
