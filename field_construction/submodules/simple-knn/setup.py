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


cxx_compiler_flags = []
nvcc_flags = []
include_dirs = _split_env_list('LANGSCENEX_EXTRA_INCLUDE_DIRS')
nvcc_flags.extend(_split_env_flags('LANGSCENEX_NVCC_FLAGS'))

if os.name == 'nt':
    cxx_compiler_flags.append('/wd4624')

setup(
    name='simple_knn',
    ext_modules=[
        CUDAExtension(
            name='simple_knn._C',
            sources=[
                'spatial.cu',
                'simple_knn.cu',
                'ext.cpp',
            ],
            include_dirs=include_dirs,
            extra_compile_args={'nvcc': nvcc_flags, 'cxx': cxx_compiler_flags},
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)
