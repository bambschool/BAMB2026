# Module 4: Spatial cognition and adaptive foraging

Teaching materials for **Module 4** of the 2026 [Barcelona Advanced Modeling of Behavior Summer School (BAMB)](https://www.bambschool.org/), prepared by Charley M. Wu (TU Darmstadt; [hmc-lab.com](https://hmc-lab.com/)).

This module covers model-based reinforcement learning, non-parametric environment clustering, Gaussian-process regression, and computational model fitting, culminating in applications to experiment design and spatial foraging data from:

> Wu, C. M., Deffner, D., Kahl, B., Meder, B., Ho, M. H., & Kurvers, R. H. J. M. (2025). Adaptive mechanisms of social and asocial learning in immersive collective foraging. *Nature Communications*, 16, 3539. https://doi.org/10.1038/s41467-025-58365-6

## Day 1 tutorials

Each notebook opens with a **Setup: Only for Google Colab** section. The **Open in Colab** badges run the notebook on Google Colab with no local setup — that first section downloads the helper modules and data the notebook needs from this repository and installs the required packages.

| Block | Topic | Tutorial |
| --- | --- | --- |
| 1 | Introduction to DYNA-Q and comparison to Q-learning on gridworld navigation | [notebook](day1_block1_dyna_q.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bambschool/BAMB2026/blob/Module-4/Module%204/day1_block1_dyna_q.ipynb) |
| 2 | Extending DYNA-Q to non-stationary environments with a Chinese Restaurant Process (CRP) prior over environment clusters (DINER) | [notebook](day1_block2_crp_dyna.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bambschool/BAMB2026/blob/Module-4/Module%204/day1_block2_crp_dyna.ipynb) |
| 3 | Gaussian-process regression and GP-UCB for spatially correlated bandit tasks, with an introduction to the Minecraft foraging data from Wu et al. (2025) | [notebook](day1_block3_gp_ucb.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bambschool/BAMB2026/blob/Module-4/Module%204/day1_block3_gp_ucb.ipynb) |

## Running on Google Colab

Click an **Open in Colab** badge above, then run the notebook top to bottom. The first section, **"Setup: Only for Google Colab"**, fetches the helper files and data for that notebook from this repository and installs the required packages. Everything after that runs identically on Colab and locally.

## Running locally

Requires Python 3.12. Install the core dependencies and launch Jupyter:

```bash
pip install numpy scipy matplotlib seaborn gymnasium pandas pyarrow
jupyter notebook
```

When running locally, skip the "Setup: Only for Google Colab" section — the helper modules and data already sit next to the notebooks.

## Supporting files

* `models.py` — DYNA-Q, CRP, and GP model implementations shared across the Day 1 notebooks (used by blocks 1–2)
* `environments.py` — `GridWorldEnv` gymnasium environment used in blocks 1–2
* `environments/` — 44 Minecraft environments (22 smooth, 22 random) as 20×20 semicolon-delimited CSVs; block 3 uses a subset
* `data/wu2025minecraft.feather` — raw foraging dataset from Wu et al. (2025)

Each notebook is self-contained: all data loading, model definitions, and visualisations are handled within the notebook or the supporting files above.

## Data

The foraging dataset (`data/wu2025minecraft.feather`) is a subset of the data from [Wu et al. (2025)](https://www.nature.com/articles/s41467-025-58365-6), available in full at https://github.com/charleywu/minecraftforaging. Participants explored a 20×20 grid of melon and pumpkin blocks in a Minecraft world across multiple sessions and conditions. The notebooks use only the solo, smooth-environment rounds. Accompanying data on player position (5 Hz) and transcribed visual fields are omitted due to size and complexity.
