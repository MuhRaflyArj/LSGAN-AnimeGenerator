# Anime Faces GAN Assignment

This project trains and compares two GAN variants on an anime face dataset:
- Vanilla GAN (baseline)
- LSGAN (least-squares GAN)

The main workflow is in `notebook.ipynb`:
1. Configure training and paths.
2. Download and load the anime faces dataset.
3. Train Vanilla GAN and LSGAN with the same settings.
4. Save checkpoints and generated samples.
5. Compare quality using FID, KID, and LPIPS diversity.

Source code is in `anime_gan/`:
- `config.py`: hyperparameters and path/device setup
- `data.py`: dataset download and dataloader creation
- `models.py`: generator and discriminator architectures
- `training.py`: training loop for Vanilla GAN and LSGAN
- `metrics.py`: FID/KID/LPIPS evaluation utilities
- `visualization.py`: training and sample plotting helpers

## CUDA Version in `pyproject.toml`

This project is configured for **CUDA 12.8** PyTorch wheels through `uv` sources:
- `tool.uv.sources` points `torch`, `torchvision`, and `torchaudio` to index `pytorch-cu128`
- `tool.uv.index` uses `https://download.pytorch.org/whl/cu128`

So when you install dependencies with `uv`, it will resolve the CUDA 12.8 wheel index configured in the TOML.

## Install Dependencies (Recommended: `uv`)

1. Install `uv` (if not installed):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. From the project root, create/sync environment from `pyproject.toml`:

```bash
uv sync --all-extras
```

3. Activate the virtual environment:

```bash
source .venv/bin/activate
```

4. (Optional) Verify PyTorch and CUDA:

```bash
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA runtime:', torch.version.cuda)"
```

5. Run notebook:

```bash
jupyter lab
```

Open `notebook.ipynb` and run the cells.
