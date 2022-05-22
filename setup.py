import sys
from site import getsitepackages, getusersitepackages
from pathlib import Path
from setuptools import find_packages, setup, Extension


if __name__ == '__main__':
    PYTHON_VERSION = f'{sys.version_info.major}.{sys.version_info.minor}'
    if '--user' in sys.argv:
        NUMPY_INCLUDE_DIR = Path(getusersitepackages())/'numpy'/'core'/'include'
        if not NUMPY_INCLUDE_DIR.exists():
            NUMPY_INCLUDE_DIR = Path(getsitepackages()[0])/'numpy'/'core'/'include'
    else:
        NUMPY_INCLUDE_DIR = Path(getsitepackages()[0])/'numpy'/'core'/'include'

    setup(
        name = 'completion3d',
        version = '0.1.0',
        description = '3D vision tasks with scene/shape completion (based on mmdet3d)',
        author = 'Chengjie Huang',
        author_email = 'c.huang@uwaterloo.ca',
        url = 'https://github.com/c7huang/completion3d',
        packages = find_packages(),
        ext_modules = [
            Extension(
                name = 'completion3d.simulation.ray_casting',
                include_dirs = [NUMPY_INCLUDE_DIR],
                extra_compile_args = ['-fopenmp'],
                undef_macros = ['RC_WITHOUT_TRIANGLE', 'RC_WITHOUT_SPHERE'],
                sources = [
                    'completion3d/simulation/ray_casting/ray_casting.cpp',
                    'completion3d/simulation/ray_casting/pymodule.cpp'
                ]
            )
        ]
    )
