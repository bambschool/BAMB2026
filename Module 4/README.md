# Module 4: Spatial cognition and adaptive foraging

Materials for Module 4 of [BAMB! 2026](https://www.bambschool.org/), by Charley M. Wu (TU Darmstadt, [hmc-lab.com](https://hmc-lab.com/)).


## Tutorials

The easiest way in is Google Colab: click a badge and run the notebook from top to bottom. The first section grabs the files it needs from this repo and installs the packages, so there is nothing to set up by hand.

**Part 1 — Model-Based Reinforcement Learning** &nbsp; [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bambschool/BAMB2026/blob/Module-4/Module%204/day1_block1_dyna_q.ipynb)


**Part 2 — Chinese Restaurant Processes and Non-Stationary Environments** &nbsp; [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bambschool/BAMB2026/blob/Module-4/Module%204/day1_block2_crp_dyna.ipynb)


**Part 3 — Gaussian Process Regression as Bayesian Value Function Approximation** &nbsp; [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bambschool/BAMB2026/blob/Module-4/Module%204/day1_block3_gp_ucb.ipynb)


## Running locally

If you would rather work on your own machine, you will need Python 3.12 and a few packages:

```bash
pip install numpy scipy matplotlib gymnasium pandas pyarrow
jupyter notebook
```

Running locally, just skip the "Setup: Only for Google Colab" section at the top of each notebook — the helper files and data are already sitting next to it.

## What's in here

- `day1_block1_dyna_q.ipynb`, `day1_block2_crp_dyna.ipynb`, `day1_block3_gp_ucb.ipynb` — the three tutorials
- `models.py` — DYNA-Q, CRP, and GP implementations shared by Parts 1 and 2
- `environments.py` — the `GridWorldEnv` used in Parts 1 and 2
- `environments/` — 44 Minecraft reward grids (22 smooth, 22 random) as 20×20 CSVs; Part 3 uses a handful
- `data/wu2025minecraft.feather` — the foraging dataset

Each notebook is self-contained: everything it needs is either in the notebook itself or in the files above.

## The data

`wu2025minecraft.feather` is a slice of the data from [Wu et al. (2025)](https://www.nature.com/articles/s41467-025-58365-6) — the full release lives at [github.com/charleywu/minecraftforaging](https://github.com/charleywu/minecraftforaging). Players explored a 20×20 grid of melon and pumpkin blocks across several sessions and conditions; here we use only the solo rounds in the smooth environments. The higher-resolution position traces and transcribed visual fields from the original study are left out to keep the file manageable.
