# PhysGaussian-SDF

Fork of [PhysGaussian (CVPR 2024)](https://github.com/XPandora/PhysGaussian) with compatibility fixes for warp 1.x / PyTorch 2.6 / CUDA 12.4, plus an interactive GUI demo (doll on stairs).

Original paper: [[Project Page](https://xpandora.github.io/PhysGaussian/)] [[arXiv](https://arxiv.org/abs/2311.12198)]

---

## Changes from upstream

### Compatibility fixes

| File | Change |
|------|--------|
| `mpm_solver_warp/warp_utils.py` | Remove `import warp.torch`; use `wp.from_torch()` instead of `torch2warp_*` helpers |
| `mpm_solver_warp/mpm_utils.py` | `wp.mat33(vec,vec,vec)` → `wp.matrix_from_cols(vec,vec,vec)` (warp 1.x API) |
| `utils/decode_param.py` | Handle `particle_filling: null` without crash |
| `gaussian-splatting/submodules/diff-gaussian-rasterization/setup.py` | Add CCCL/CUDA include paths for Linux + CUDA 12.4 |
| `gaussian-splatting/submodules/simple-knn/setup.py` | Same as above |
| `gaussian-splatting/submodules/simple-knn/simple_knn/__init__.py` | Created (missing from upstream) |

### New files

| File | Description |
|------|-------------|
| `utils/mesh_sdf.py` | Bake a triangle mesh into a signed distance field on the MPM grid (Taichi GPU). Used for SDF-based rigid colliders. |
| `utils/mesh_to_gs.py` | Bake a static mesh into 3D Gaussians (stratified sampling, anisotropic disk covariance, baked Lambertian shading). |
| `scripts/doll_stair_gs_gui.py` | Interactive GUI: doll (MPM plasticine, damage model) falling down stairs. Real-time cv2 window, orbit camera, two render modes. |
| `config/doll_stair_config.json` | Config for the doll-on-stairs demo. |

---

## Setup

### Prerequisites

- CUDA 12.4, driver ≥ 550
- conda

### Environment

```bash
conda create -n phys python=3.10
conda activate phys
conda install pytorch==2.6.0 torchvision==0.21.0 pytorch-cuda=12.4 -c pytorch -c nvidia

pip install -r requirements.txt
pip install -e gaussian-splatting/submodules/diff-gaussian-rasterization/
pip install -e gaussian-splatting/submodules/simple-knn/
```

On the server, `libcudart` symlinks may be needed before `pip install -e`:

```bash
ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12 \
       $CONDA_PREFIX/lib/libcudart.so
ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12 \
       $CONDA_PREFIX/lib/libcudart.so.12
```

### Clone

```bash
git clone --recurse-submodules https://github.com/adbalsk/PhysGaussian-SDF.git
cd PhysGaussian-SDF
```

---

## Usage

### Offline simulation (original pipeline)

```bash
python gs_simulation.py \
    --model_path ./model/bread-trained/ \
    --output_path ./output/tear_bread \
    --config ./config/tear_bread_config.json \
    --render_img --compile_video
```

Skip particle filling (faster) by setting `"particle_filling": null` in the config.

### Interactive GUI — doll on stairs

```bash
python scripts/doll_stair_gs_gui.py \
    --model_path ./model/doll.ply \
    --config ./config/doll_stair_config.json
```

| Flag | Default | Description |
|------|---------|-------------|
| `--render` | `hybrid` | `hybrid`: pyrender mesh + gaussian doll (depth-composited); `gaussian`: all-gaussian single pass |
| `--mode` | `live` | `live`: real-time MPM; `playback`: replay a recorded `.npz` trajectory |
| `--substeps` | `25` | MPM substeps per displayed frame |
| `--width/--height` | `800` | Window resolution |
| `--record <path>` | — | Save doll trajectory to `.npz` (live mode) |
| `--traj <path>` | — | Load trajectory for playback mode |

**Controls:** `LMB drag` = orbit · `scroll` = zoom · `Space` = pause · `R` = reset · `[/]` = substeps · `L` = toggle lighting · `S` = save frame · `Esc/Q` = quit

The doll uses material `plasticine` with a damage model: particles permanently lose elasticity when yield stress is exhausted, then vanish from the render (`mu=0` → `opacity=0`).

---

## Citation

```bibtex
@article{xie2023physgaussian,
  title={PhysGaussian: Physics-Integrated 3D Gaussians for Generative Dynamics},
  author={Xie, Tianyi and Zong, Zeshun and Qiu, Yuxing and Li, Xuan and Feng, Yutao and Yang, Yin and Jiang, Chenfanfu},
  journal={arXiv preprint arXiv:2311.12198},
  year={2023},
}
```
