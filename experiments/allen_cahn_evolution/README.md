# Allen-Cahn Evolution Experiment

This experiment solves the time-dependent Allen-Cahn equation

```text
u_t - (eps(x)^2 u_x)_x + u^3 - u = 0
```

with Dirichlet boundary data and optional interface jumps. The implementation is
self-contained in this folder and only reuses the shared TFPM-Lagrange kernel from
`tfpm_compare_suite_auglag.core`.

Run all three schemes:

```powershell
python -m tfpm_compare_suite_auglag.experiments.allen_cahn_evolution.run_allen_cahn_evolution
```

Run one scheme:

```powershell
python -m tfpm_compare_suite_auglag.experiments.allen_cahn_evolution.run_allen_cahn_evolution --mode scheme2
```

By default the experiment also computes a high-resolution FDM reference with a
BDF time integrator and reports final-time relative errors. Disable it with:

```powershell
python -m tfpm_compare_suite_auglag.experiments.allen_cahn_evolution.run_allen_cahn_evolution --no-reference
```

Outputs are written to `results/` by default:

- `allen_cahn_evolution_final.png`
- `allen_cahn_evolution_energy.png`
- `allen_cahn_evolution_errors_vs_fdm_reference.png`
- `scheme*_state_history.npz`
- `fdm_reference_state_history.npz`
- `summary.txt`
