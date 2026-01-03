from setuptools import Extension, setup

from Cython.Build import cythonize
import numpy as np
import subprocess


def pkg_config(packages: list[str]) -> tuple[list[str], list[str]]:
    try:
        cflags = subprocess.check_output(["pkg-config", "--cflags", *packages], text=True).strip().split()
        libs = subprocess.check_output(["pkg-config", "--libs", *packages], text=True).strip().split()
        return cflags, libs
    except Exception:
        return [], []


def split_flags(flags: list[str]) -> tuple[list[str], list[str], list[str], list[str]]:
    include_dirs: list[str] = []
    library_dirs: list[str] = []
    libraries: list[str] = []
    extras: list[str] = []
    for flag in flags:
        if flag.startswith("-I"):
            include_dirs.append(flag[2:])
        elif flag.startswith("-L"):
            library_dirs.append(flag[2:])
        elif flag.startswith("-l"):
            libraries.append(flag[2:])
        else:
            extras.append(flag)
    return include_dirs, library_dirs, libraries, extras


raylib_cflags, raylib_libs = pkg_config(["raylib"])
raylib_includes, _, _, raylib_extra_cflags = split_flags(raylib_cflags)
_, raylib_lib_dirs, raylib_libraries, raylib_extra_ldflags = split_flags(raylib_libs)

base_compile_args = ["-O3", "-march=native", "-ffast-math"]

extensions = [
    Extension(
        "drone_swarm_c",
        sources=["drone_swarm_binding.pyx", "c_src/drone_swarm.c"],
        include_dirs=[np.get_include(), "c_src"],
        extra_compile_args=base_compile_args,
    ),
    Extension(
        "drone_swarm.binding",
        sources=["c_src/binding.c", "c_src/drone_swarm.c"],
        include_dirs=[np.get_include(), "c_src", "PufferLib/pufferlib/ocean"] + raylib_includes,
        library_dirs=raylib_lib_dirs,
        libraries=raylib_libraries,
        extra_compile_args=base_compile_args + raylib_extra_cflags,
        extra_link_args=raylib_extra_ldflags,
    ),
]

setup(
    name="pufferdroneswarm",
    py_modules=[],
    packages=["drone_swarm"],
    ext_modules=cythonize(extensions, language_level=3),
)
