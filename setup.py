from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'taxi_simulator',
        ['src/taxi_simulator.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=['-std=c++17', '-O3']
    ),
]

setup(
    name='taxi_simulator',
    version='0.1',
    ext_modules=ext_modules,
    zip_safe=False,
)
