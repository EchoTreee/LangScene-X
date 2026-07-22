# Blackwell Migration Guide

This repository was previously debugged around an older mixed CUDA toolchain. On NVIDIA Blackwell GPUs (`sm_120`), the stable migration path is:

1. Keep the current partially working environment untouched as a fallback.
2. Create a new PyTorch `cu128` runtime that can execute on Blackwell.
3. Verify PyTorch CUDA execution before rebuilding any local extension.
4. Rebuild all PyTorch ABI-bound CUDA extensions inside that new environment.

## 1. Create a clean Blackwell runtime

```bash
conda create -n langscenex_bw python=3.10 -y
conda activate langscenex_bw
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Do not reuse the old `langscene124` toolchain as the long-term compile source for this environment.

## 2. Verify the runtime first

Run this before installing project CUDA extensions:

```bash
python - <<'PYTORCH_CHECK'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
x = torch.randn(1024, 1024, device="cuda")
y = x @ x
print("ok", y.shape, y.device)
PYTORCH_CHECK
```

If this fails, stop here. Do not troubleshoot `simple-knn` or `diff-langsurf-rasterizer` until the runtime itself works on the GPU.

## 3. Install pure Python dependencies

Install Python-side packages first, then rebuild local CUDA extensions last:

```bash
python -m pip install \
  numpy==1.26.4 \
  scipy==1.14.1 \
  scikit-learn \
  opencv-python-headless==4.10.0.84 \
  open3d==0.19.0 \
  plyfile \
  lpips \
  trimesh \
  torch-kmeans \
  tqdm \
  diffusers>=0.32.1 \
  accelerate>=1.1.1 \
  transformers>=4.46.2 \
  imageio>=2.35.1 \
  imageio-ffmpeg>=0.5.1 \
  openai>=1.54.0 \
  moviepy>=2.0.0 \
  scikit-video>=1.1.11 \
  pydantic>=2.10.3 \
  tensorflow==2.15.0 \
  loguru \
  einops \
  mediapy

conda install -n langscenex_bw -y libgl
python -m pip install -e auto-seg/submodules/segment-anything-1
python -m pip install -e auto-seg/submodules/segment-anything-2
python -m pip install git+https://github.com/openai/CLIP.git
```

## 4. Verify the compile toolchain

Before rebuilding extensions, make sure the active compile chain belongs to the new environment:

```bash
which nvcc
nvcc -V
echo "$CUDA_HOME"
echo "$PATH"
echo "$LD_LIBRARY_PATH"
gcc --version
g++ --version
```

If any of these still point at `langscene124`, `/usr/local/cuda-12.2`, or another older CUDA install, fix that first.

For extension rebuilds, using the system compiler remains the safer default:

```bash
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
```

If the new toolkit is installed inside the conda env, also export the matching paths explicitly:

```bash
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
```

## 5. Rebuild local CUDA extensions in the new environment

These extensions bind against the active `torch` headers and ABI, so they must be rebuilt after the PyTorch upgrade:

```bash
python -m pip install --no-build-isolation -e field_construction/submodules/simple-knn
python -m pip install --no-build-isolation -e field_construction/submodules/diff-langsurf-rasterizer
```

Then verify imports directly:

```bash
python - <<'PYEXT_CHECK'
import simple_knn._C
import diff_LangSurf_rasterization._C
print("extension import ok")
PYEXT_CHECK
```

## 6. Restore runtime library paths

After activating the environment, source the helper script from the repository root:

```bash
conda activate langscenex_bw
source /workspace/projects/LangScene-X/langscenex_env.sh
```

`langscenex_env.sh` now follows the currently activated conda environment instead of hardcoding `langscenex`.

## 7. Start the project only after all three checks pass

Do not run `quick_start.sh` until all of the following succeed:

1. The minimal PyTorch CUDA matmul test.
2. `import simple_knn._C`
3. `import diff_LangSurf_rasterization._C`

Then run:

```bash
bash quick_start.sh <first_rgb_image_path> <last_rgb_image_path>
```

## Troubleshooting order

When a rebuild fails, debug in this order:

1. `which nvcc` and `nvcc -V`
2. `CUDA_HOME`, `PATH`, `LD_LIBRARY_PATH`
3. `gcc --version` and `g++ --version`
4. Only then inspect source compatibility against the newer `torch` / CUDA stack
