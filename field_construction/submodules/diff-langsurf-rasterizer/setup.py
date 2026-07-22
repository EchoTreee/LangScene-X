#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def _split_env_list(name):
    value = os.environ.get(name, '')
    return [item for item in value.split(os.pathsep) if item]


def _split_env_flags(name):
    value = os.environ.get(name, '')
    return [item for item in value.split() if item]


root_dir = os.path.dirname(os.path.abspath(__file__))
include_dirs = [os.path.join(root_dir, 'third_party', 'glm')]
include_dirs.extend(_split_env_list('LANGSCENEX_EXTRA_INCLUDE_DIRS'))
nvcc_flags = ['-I' + os.path.join(root_dir, 'third_party', 'glm')]
nvcc_flags.extend(_split_env_flags('LANGSCENEX_NVCC_FLAGS'))

setup(
    name='diff_LangSurf_rasterization',
    packages=['diff_LangSurf_rasterization'],
    ext_modules=[
        CUDAExtension(
            name='diff_LangSurf_rasterization._C',
            sources=[
                'cuda_rasterizer/rasterizer_impl.cu',
                'cuda_rasterizer/forward.cu',
                'cuda_rasterizer/backward.cu',
                'rasterize_points.cu',
                'ext.cpp',
            ],
            include_dirs=include_dirs,
            extra_compile_args={'nvcc': nvcc_flags},
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)
