from setuptools import Extension, setup

from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "drone_swarm_c",
        sources=["drone_swarm_binding.pyx", "c_src/drone_swarm.c"],
        include_dirs=[np.get_include(), "c_src"],
        extra_compile_args=["-O3", "-march=native", "-ffast-math"],
    )
]

setup(
    name="pufferdroneswarm",
    py_modules=[],
    packages=[],
    ext_modules=cythonize(extensions, language_level=3),
)
