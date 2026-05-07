# Steady Allen-Cahn Energy ALM Experiment

This experiment applies the Newton-space nonlinear energy ALM idea to the
steady Allen-Cahn equation

```text
-epsilon^2 u'' + u^3 - u = 0, epsilon > 0.
```

At each outer iteration, the trial space is built from the Newton linearized
equation

```text
-epsilon^2 u_{k+1}'' + (3 u_k^2 - 1) u_{k+1} = 2 u_k^3.
```

The solve is performed in the stretched coordinate `y = x / epsilon`. With
`v(y) = u(x)`, the linearized problem becomes

```text
-v_{k+1}'' + (3 v_k^2 - 1) v_{k+1} = 2 v_k^3.
```

This matches the shared TFPM element builder's normalized form without
introducing the large `1 / epsilon^2` reaction coefficient.

Inside that fixed TFPM trial space, the code minimizes the original
Allen-Cahn energy

```text
E_epsilon(u) = int 0.5 epsilon^2 |u'|^2 + 0.25 (u^2 - 1)^2 dx
```

Equivalently, the fixed-space minimizer is computed from

```text
epsilon * int 0.5 |v'|^2 + 0.25 (v^2 - 1)^2 dy.
```

subject to the Dirichlet and interface value constraints by a nonlinear
augmented Lagrangian method.

The driver also solves a standard Newton finite-difference method (FDM) on
the same stretched coordinate grid for comparison. Relative errors are measured
against a higher-resolution Newton-FDM reference solution.

The default setup solves on `[-0.5, 0.5]` with `epsilon=0.01`,
zero Dirichlet data `u(-0.5)=0`, `u(0.5)=0`, the `smooth_periodic`
initial guess, and an artificial TFPM interface at `x=0`.

Run from the repository parent package context:

```powershell
python -m tfpm_compare_suite_auglag.experiments.steady_allen_cahn_energy_alm.solve_steady_allen_cahn_energy_alm
```

Or run the file directly from this repository:

```powershell
python experiments/steady_allen_cahn_energy_alm/solve_steady_allen_cahn_energy_alm.py
```

Useful smoke run:

```powershell
python experiments/steady_allen_cahn_energy_alm/solve_steady_allen_cahn_energy_alm.py --num-elements 16 --fdm-elements 16 --fdm-reference-elements 128 --output-dir results_smoke
```

Run with a different epsilon:

```powershell
python experiments/steady_allen_cahn_energy_alm/solve_steady_allen_cahn_energy_alm.py --epsilon 0.2 --num-elements 80
```

Skip the FDM comparison if only the energy-ALM solve is needed:

```powershell
python experiments/steady_allen_cahn_energy_alm/solve_steady_allen_cahn_energy_alm.py --no-fdm-comparison
```

Main outputs:

- `steady_allen_cahn_solution.png`
- `steady_allen_cahn_history.png`
- `steady_allen_cahn_solution.npz`
- `steady_allen_cahn_summary.txt`
- `steady_allen_cahn_vs_newton_fdm_solution.png`
- `steady_allen_cahn_vs_newton_fdm_errors.png`
- `steady_allen_cahn_vs_newton_fdm_summary.txt`
