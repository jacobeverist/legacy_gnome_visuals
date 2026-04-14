# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repository Is

A collection of standalone Python visualization scripts for the HyperGrid Transform (HGT) — a neuromorphic encoding scheme. There is no build system, no test suite, and no unified entry point. Each script is independently runnable.

Rendered outputs (PNGs, GIFs, MKVs) produced by this code are archived at https://github.com/jacobeverist/legacy_gnome_gallery and browseable at https://jacobeverist.github.io/legacy_gnome_gallery/.

## Running Scripts

Run scripts from within their subdirectory so `out/` paths resolve correctly:

```bash
cd hypergrid_transform
python plot_hgt_visuals.py
```

Each subdirectory has a `requirements.txt`. There is no top-level environment — install per module.

## Architecture

### Core dependency: `brainblocks`

All meaningful visualization scripts depend on `brainblocks.tools.HyperGridTransform`. This is the central object — it takes parameters like `num_grids`, `num_bins`, `num_subspace_dims`, grid bases (angles), and periods, and produces binary "gnome" encodings of scalar/vector inputs. Most scripts construct one or more HGT instances, sweep input values, and visualize the resulting encoding structure or inter-point similarities.

### Similarity metrics

Scripts compute pairwise similarity between gnome encodings using the Otsuka-Ochiai coefficient (bitwise overlap normalized by geometric mean of set sizes). Similarity is visualized as 2D heatmaps or contour plots.

### Output pipeline

1. Scripts save PNG frames to `out/` (numbered sequences for animations, single files for static plots)
2. `encode_video.sh` in each module converts PNG sequences to MKV then GIF via ffmpeg:
   ```bash
   ffmpeg -y -framerate 10 -i out/visual_test_frames_1_%05d.png -c:v libx264 -preset veryslow -crf 0 out/visual_test.mkv
   ffmpeg -y -i out/visual_test.mkv -filter_complex "[0:v] split [a][b];[a] palettegen [p];[b][p] paletteuse" out/visual_test.gif
   ```

### Key modules

- **`hypergrid_transform/`** — the most developed module. `plot_hgt_visuals.py` renders animated 2D grid visualizations; `plot_similarity.py` generates similarity heatmaps; `run_hypergrid_experiments.py` sweeps encoding parameters; `helpers/hypergrid_graphics.py` contains shared rendering functions for grids, bases, and similarity maps.

- **`reference_code/`** — contains foundational utilities: original HGT implementations, hex grid geometry math, and `datasets.py` which generates synthetic test trajectories (lemniscates, Clifford attractors, random walks).

- **`gnomes_to_graph/`** and **`segmented_space_and_axes/`** — graph-based and spatial visualizations of how the encoder partitions input space.

### Configuration

All parameters (number of grids, bins, angles, periods) are hardcoded in each script's `main()` or module-level constants. There is no config file system.
