#!/usr/bin/env bash

set -euo pipefail

if [[ -n "${1:-}" ]]; then
    env_prefix="$1"
elif [[ -n "${CONDA_PREFIX:-}" ]]; then
    env_prefix="$CONDA_PREFIX"
else
    echo "langscenex_env.sh: activate a conda environment first, or pass its prefix explicitly." >&2
    return 1 2>/dev/null || exit 1
fi

torch_lib="${env_prefix}/lib/python3.10/site-packages/torch/lib"

if [[ ! -d "$torch_lib" ]]; then
    echo "langscenex_env.sh: torch runtime directory not found: $torch_lib" >&2
    return 1 2>/dev/null || exit 1
fi

extra_libs=(
    "$torch_lib"
    "${env_prefix}/lib/python3.10/site-packages/nvidia/cuda_runtime/lib"
    "${env_prefix}/lib/python3.10/site-packages/nvidia/cublas/lib"
    "${env_prefix}/lib/python3.10/site-packages/nvidia/cudnn/lib"
    "${env_prefix}/lib/python3.10/site-packages/nvidia/cusparse/lib"
    "${env_prefix}/lib/python3.10/site-packages/nvidia/cusparselt/lib"
    "${env_prefix}/lib/python3.10/site-packages/nvidia/cusolver/lib"
    "${env_prefix}/lib/python3.10/site-packages/nvidia/cufft/lib"
    "${env_prefix}/lib/python3.10/site-packages/nvidia/curand/lib"
    "${env_prefix}/lib/python3.10/site-packages/nvidia/nccl/lib"
    "${env_prefix}/lib/python3.10/site-packages/nvidia/nvjitlink/lib"
    "${env_prefix}/lib/python3.10/site-packages/nvidia/nvtx/lib"
)

for libdir in "${extra_libs[@]}"; do
    if [[ -d "$libdir" ]]; then
        case ":${LD_LIBRARY_PATH:-}:" in
            *":${libdir}:"*) ;;
            *) export LD_LIBRARY_PATH="${libdir}:${LD_LIBRARY_PATH:-}" ;;
        esac
    fi
done

export LANGSCENEX_ENV_PREFIX="$env_prefix"
