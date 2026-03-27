# Diffusion-Forcing Maze2D Update Summary

## Scope
Implemented deterministic dataset splitting and held-out same-distribution maze-instance evaluation support for Maze2D planning.

## Dataset Splitting
- Added split-aware sample indexing in [datasets/offline_rl/maze2d.py](datasets/offline_rl/maze2d.py).
- New behavior:
	- Optional split enable flag.
	- Episode-level split strategy.
	- Deterministic split with seed.
	- Configurable train/validation/test fractions.
	- Window validity constraint: sampled sequence must remain within a single episode.
- Updated dataset length and item retrieval to use split-filtered start indices.

## Held-Out Maze Evaluation
- Added held-out evaluation controls to planning algorithm in [algorithms/diffusion_forcing/df_planning.py](algorithms/diffusion_forcing/df_planning.py):
	- Enable flag.
	- Split gating (validation/test).
	- Seeded held-out maze sampling.
	- Configurable env constructor kwarg for maze injection.
	- Optional fallback to planning-only eval if env injection is unsupported.
- Validation/test path now supports selecting maze instances distinct from training-exposed defaults.

## Maze Utility Extensions
- Extended utilities in [utils/logging_utils.py](utils/logging_utils.py):
	- Added maze-string parser.
	- Start/goal sampler now accepts external maze strings and RNG.
	- Trajectory plotting now accepts external maze strings for geometry-consistent visualization.

## Config Surface
- Added default split and held-out eval config blocks in [configurations/dataset/base_dataset.yaml](configurations/dataset/base_dataset.yaml).
- Added Maze2D-large overrides in [configurations/dataset/maze2d_large.yaml](configurations/dataset/maze2d_large.yaml).
- Wired dataset held-out config into planner config in [configurations/algorithm/df_planning.yaml](configurations/algorithm/df_planning.yaml).

## Compatibility
- Defaults preserve prior behavior unless split or held-out options are explicitly enabled.
- YAML indentation issue in base dataset config was corrected to ensure Hydra composition works.
