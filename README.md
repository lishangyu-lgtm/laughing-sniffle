# TFPM Compare Suite (Lagrange)

This package studies TFPM with a direct Lagrange-multiplier interface treatment for

`-(eps(x)u')' + c(x)u = f(x), u(0)=m, u(1)=n, [u]=p, [eps(x)u']=q`

with piecewise-constant diffusion.

The project is organized around reusable solver code plus self-contained
experiment folders.

## Layout

```text
tfpm_compare_suite_lagrange/
|- core/                      reusable linear-interface model + TFPM/Lagrange kernel
|- baselines/                 FDM and FEM comparison solvers
|- postprocess/               reserved for future shared post-processing helpers
|- experiments/
|  |- compare_lagrange/       main linear comparison experiment
|  |- allen_cahn_evolution/   time-evolution Allen-Cahn Scheme I-III experiment
|  |- convergence_rate/       grid convergence-rate study
|  |- equal_eps_error_sweep/  error vs eps with eps1 = eps2
|  |- nonlinear_pde/
|  `- penalty_vs_lagrange/    penalty vs direct Lagrange experiment
|- results*/                  older output folders kept as-is
`- __init__.py
```

## Where To Edit

For the main linear comparison workflow, edit
`experiments/compare_lagrange/problem.py`.

That file contains:

- the default coefficient and right-hand side for the comparison experiment
- the optional exact reference
- `DEFAULT_EXPERIMENT`
- the default output directory for that experiment

The reusable linear-interface objects now live in `core/linear_interface.py`:

- `ProblemConfig`
- `ExactReference`
- `x_to_y(...)`, `y_to_x(...)`
- transformed coefficient / RHS helpers
- flux-jump averaging helpers

## Experiment Folders

Each experiment folder is intended to keep three things close together:

- experiment-specific setup
- the runner entry point
- that experiment's default `results/` directory

Current experiment folders:

- `experiments/compare_lagrange/`
  Files: `problem.py`, `analysis_plot.py`, `run_compare_lagrange.py`, `results/`
- `experiments/allen_cahn_evolution/`
  Files: `README.md`, `problem.py`, `analysis_plot.py`, `run_allen_cahn_evolution.py`, `results/`
- `experiments/convergence_rate/`
  Files: `convergence_study.py`, `analysis_plot.py`, `results/`
- `experiments/equal_eps_error_sweep/`
  Files: `run_equal_eps_error_sweep.py`, `results/`
- `experiments/nonlinear_pde/`
  Files: `analysis_plot.py`, `solve_nonlinear_pde.py`, `results/`
- `experiments/penalty_vs_lagrange/`
  Files: `problem.py`, `analysis_plot.py`, `run_penalty_vs_lagrange.py`, `results/`

This is a good default pattern for future studies too. If a new experiment grows
its own parameters or manufactured cases, place them beside that experiment's
script instead of putting them into `core/`.

## Run Experiments

Recommended entry points:

```powershell
python -m tfpm_compare_suite_lagrange.experiments.compare_lagrange.run_compare_lagrange
python -m tfpm_compare_suite_lagrange.experiments.allen_cahn_evolution.run_allen_cahn_evolution
python -m tfpm_compare_suite_lagrange.experiments.convergence_rate.convergence_study
python -m tfpm_compare_suite_lagrange.experiments.equal_eps_error_sweep.run_equal_eps_error_sweep
python -m tfpm_compare_suite_lagrange.experiments.nonlinear_pde.solve_nonlinear_pde
python -m tfpm_compare_suite_lagrange.experiments.penalty_vs_lagrange.run_penalty_vs_lagrange
```

## Outputs

Each experiment now writes to its own local `results/` directory by default:

- `experiments/compare_lagrange/results/`
- `experiments/allen_cahn_evolution/results/`
- `experiments/convergence_rate/results/`
- `experiments/equal_eps_error_sweep/results/`
- `experiments/nonlinear_pde/results/`
- `experiments/penalty_vs_lagrange/results/`

This keeps figures, summaries, and comparisons from different studies from
mixing together.
