# Nonlinear Energy ALM Experiment

This experiment is intentionally self-contained. It reuses the existing TFPM
element builders, but the nonlinear energy minimizer and its augmented
Lagrangian loop live only in this directory.

The method solves

```text
min E(u) = int_a^b 0.5 |u'|^2 + 0.25 u^4 - f u dx
```

with Dirichlet/interface value constraints. At each outer iteration, the Newton
linearized problem

```text
-u_{k+1}'' + 3 u_k^2 u_{k+1} = f + 2 u_k^3
```

is passed to the shared TFPM element builder to construct the local trial space.
The fixed-space problem is then solved by a nonlinear augmented Lagrangian
method.

Run from the repository parent package context:

```powershell
python -m tfpm_compare_suite_auglag.experiments.nonlinear_energy_alm.solve_energy_alm
```

Or run the file directly from this repository:

```powershell
python experiments/nonlinear_energy_alm/solve_energy_alm.py
```

Compare this method with `experiments/nonlinear_pde` TFPM-AugLag and its
FDM-fine reference:

```powershell
python experiments/nonlinear_energy_alm/compare_with_nonlinear_pde.py
```
