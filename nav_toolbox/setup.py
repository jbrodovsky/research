"""
Setup file for nav_toolbox package. This file is used to build the C++ extension.
"""

from setuptools import setup, Extension

setup(
    name="nav_toolbox",
    version="0.1",
    packages=["nav_toolbox"],
    ext_modules=[Extension("nav_toolbox.earth", sources=["./src/earth.cpp"])],
)
