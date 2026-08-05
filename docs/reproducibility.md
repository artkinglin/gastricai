# Reproducibility

## Environment

Use Python 3.11 or 3.12 and install dependencies into a project virtual
environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

## Seeds

Training scripts expose `--seed` and set Python, NumPy, and PyTorch seeds. CUDA
determinism is requested where possible.

## Known Nondeterminism

Exact results can still vary across:

- GPU models and CUDA/cuDNN versions.
- PyTorch and TorchVision versions.
- Image decoding libraries.
- Multithreaded dataloading.

## Recommended Run Records

For each run, save:

- Git commit SHA.
- Config file.
- Dependency versions.
- Hardware notes.
- Dataset version or manifest.
- Training history and test reports from `runs/`.
