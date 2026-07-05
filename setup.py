from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'megacity_taxi_env',
        ['MegacityTaxiEnv.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=['-std=c++17', '-O3']
    ),
]

setup(
    name='megacity_taxi_env',
    version='0.1',
    ext_modules=ext_modules,
    zip_safe=False,
)
