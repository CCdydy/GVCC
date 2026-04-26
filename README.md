# GVCC

Official code release for **[GVCC: Zero-Shot Video Compression via Codebook-Driven Stochastic Rectified Flow](https://arxiv.org/abs/2603.26571)** (arXiv:2603.26571).

A short qualitative example is included under [`demo/`](demo/).
Algorithmic details of the encode/decode pipeline are in [`PIPELINE.md`](PIPELINE.md).

## Install

Tested with Python 3.10, CUDA 13.0, PyTorch 2.10, and a single 80 GB GPU.

```bash
# 1. PyTorch (CUDA 13.0 wheel — adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# 2. Remaining dependencies (versions pinned to the ones used in the paper)
pip install -r requirements.txt
```

`flash_attn` is optional — `wan/modules/attention.py` falls back to PyTorch SDPA if it is not installed.

## Model weights

Download the three Wan2.1-14B checkpoints from HuggingFace and place each one **alongside its corresponding `run_*.sh`** (the run scripts locate the model via `${SCRIPT_DIR}/Wan2.1-*`):

| Method | HuggingFace repo                  | Required local path                  |
| ------ | --------------------------------- | ------------------------------------ |
| T2V    | `Wan-AI/Wan2.1-T2V-14B-Diffusers` | `exp_t2v/Wan2.1-T2V-14B-Diffusers/`  |
| I2V    | `Wan-AI/Wan2.1-I2V-14B-720P`      | `exp_i2v/Wan2.1-I2V-14B-720P/`       |
| FLF2V  | `Wan-AI/Wan2.1-FLF2V-14B-720P`    | `exp_flf2v/Wan2.1-FLF2V-14B-720P/`   |

Convenience download scripts (each calls `huggingface_hub.snapshot_download`):

```bash
bash exp_t2v/download_t2v_14b.sh
bash exp_i2v/download_i2v_14b.sh
bash exp_flf2v/download_flf2v_14b.sh
```

Each path may be a real directory or a symlink to a shared cache.

## Data

Download the seven [UVG-1080p](https://ultravideo.fi/) YUV sequences (Beauty, Bosphorus, HoneyBee, Jockey, ReadySetGo, ShakeNDry, YachtRide) and place them anywhere under `data/uvg/`. The loader (`uvg_data.py`) recursively scans for `*.yuv` and matches sequences by filename.

## Run

```bash
# T2V — codebook only
bash exp_t2v/run_t2v.sh           # 720p, 3 GOPs (quick)
bash exp_t2v/run_t2v_1080p.sh     # 1080p, full UVG

# I2V — autoregressive with tail-residual correction
bash exp_i2v/run.sh
bash exp_i2v/run_1080p.sh

# FLF2V — first/last-frame conditioning
bash exp_flf2v/run_flf2v.sh
bash exp_flf2v/run_flf2v_1080p.sh
```

VRAM: 14B models need ~48 GB at 720p and ~70 GB at 1080p.
Pass `--help` to any of the `run_*_experiment.py` files for the full parameter list (`M`, `K`, `steps`, `g_scale`, `ddim_tail`, etc.).

## Output layout

```
exp_{method}/results_{resolution}/
  summary.json
  {sequence}/
    original.mp4
    reconstructed_full.mp4
    gop{N}/
      metrics.json
      reconstructed.mp4
      codebook.tdcm
```

## Citation

```bibtex
@article{zeng2026gvcc,
  title   = {GVCC: Zero-Shot Video Compression via Codebook-Driven Stochastic Rectified Flow},
  author  = {Zeng, Ziyue and Su, Xun and Liu, Haoyuan and Lu, Bingyu and Tatsumi, Yui and Watanabe, Hiroshi},
  journal = {arXiv preprint arXiv:2603.26571},
  year    = {2026}
}
```

## License

Apache-2.0 (see [LICENSE](LICENSE)). The `wan/` subpackage is vendored from [Wan2.1](https://github.com/Wan-Video/Wan2.1) (Apache-2.0); upstream copyright headers are preserved. See [NOTICE](NOTICE) for the full attribution list.
