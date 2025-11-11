# Environment Matrix – Stage 2

This document captures the exact runtime state used for the Stage 2 sprint so collaborators can reproduce the setup without guesswork.

## 1. Runtime Summary

| Component | Version | Notes / Verification |
|-----------|---------|----------------------|
| OS | Ubuntu 22.04 LTS (kernel 6.x) | `cat /etc/os-release` (see host) |
| Python | 3.12.2 (conda-forge build) | `uv run python -V` |
| uv | 0.9.8 | `uv --version` |
| PyTorch | 2.9.0+cu128 | `uv run python -c "import torch; print(torch.__version__)"` |
| torchvision | 0.24.0+cu128 | `uv run python -c "import torchvision; print(torchvision.__version__)"` |
| torchaudio | 2.9.0+cu128 | `uv run python -c "import torchaudio; print(torchaudio.__version__)"` |
| CUDA runtime | 12.8 (PyTorch wheels) | `uv run python -c "import torch; print(torch.version.cuda)"` |
| NVIDIA driver | 550.90.07 | `nvidia-smi --query-gpu=name,driver_version --format=csv` |
| GPUs | 2 × NVIDIA RTX 6000 Ada (49 GB) | Prefer `cuda:1` for single-GPU jobs |

## 2. uv / Dependency Management
- `pyproject.toml` + `uv.lock` pin all Python dependencies.
- Sync command: `uv sync --all-extras`.
- When installing torch 2.9 manually: `uv pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128`.
- Cache guidance: set `UV_CACHE_DIR=/tmp/uv-cache` if default path lacks space.

## 3. GPU Usage Notes
- Default to `cuda:1` for long single-GPU training/eval to avoid interfering with tmux sessions pinned to GPU0.
- Distributed jobs use `torchrun --nproc_per_node=2` with both GPUs.
- Driver 550.90.07 + CUDA 12.4 runtime confirmed compatible with PyTorch 2.9.0/cu128 wheel; no additional toolkit install needed.
- Enable `NCCL_IB_DISABLE=1` if networking errors appear (not observed yet).

## 4. Verification Checklist
Run the following snippet after provisioning a new machine to confirm parity:

```bash
uv --version
uv run python -V
uv run python - <<'PY'
import torch, torchvision, torchaudio
print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('torchvision', torchvision.__version__)
print('torchaudio', torchaudio.__version__)
print('device0', torch.cuda.get_device_name(0))
PY
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
```

Record the outputs in `logs/env_checks/<date>.txt` before running large jobs.

## 5. Known Good Combinations
| Stack | Status | Notes |
|-------|--------|-------|
| torch 2.9.0 + torchvision 0.24.0 + CUDA 12.8 | ✅ | Current default; supports FlashAttention and Muon optimizers. |
| torch 2.9.0 + torchvision 0.23.x | ❌ | Version mismatch; torchvision 0.23 expects torch 2.8. |
| torch 2.5.0 + torchvision 0.20.0 | ✅ legacy | Use only if targeting older runs (no muon support). |

## 6. Process
1. Clone repo → `git clone https://github.com/kmccleary3301/nested_learning.git`.
2. `cd nested_learning && uv sync --all-extras`.
3. Verify versions via checklist above.
4. Export `WANDB_API_KEY` from `git.env` (sourced manually) before training.
5. Launch jobs via `uv run ...` to guarantee the locked environment.

Keeping this matrix current prevents silent drifts when PyTorch or CUDA releases change. Update it whenever the `uv.lock` or driver stack changes.
