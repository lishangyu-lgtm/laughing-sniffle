# Experiments

Each experiment lives in its own folder, so its setup, runnable script, and
default outputs stay together.

- `compare_lagrange/`
  Files: `problem.py`, `analysis_plot.py`, `run_compare_lagrange.py`, `results/`
- `allen_cahn_evolution/`
  Files: `README.md`, `problem.py`, `analysis_plot.py`, `run_allen_cahn_evolution.py`, `results/`
- `convergence_eps_variation/`
  Files: `convergence_study_eps_variation.py`, `results/`
- `nonlinear_pde/`
  Files: `analysis_plot.py`, `solve_nonlinear_pde.py`, `results/`
- `steady_allen_cahn_energy_alm/`
  Files: `README.md`, `solve_steady_allen_cahn_energy_alm.py`, `results/`
- `penalty_vs_lagrange/`
  Files: `problem.py`, `analysis_plot.py`, `run_penalty_vs_lagrange.py`, `results/`

Suggested rule for future additions:

- If a file mainly runs one study and writes outputs, put that original script
  inside `experiments/<name>/`.
- Put experiment-specific setup beside it.
- Keep experiment-specific plotting modules inside that experiment folder.
- Move only reusable solver logic into `../core/` or `../baselines/`.
