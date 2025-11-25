from setuptools import setup, Extension
import numpy

extension = Extension(
    'arrayprocessor',
    sources=['arrayprocessor_module.cpp', 'arrayprocessor.cpp'],
    include_dirs=[numpy.get_include()],
    libraries=[],
    library_dirs=[],
    runtime_library_dirs=[],
    extra_compile_args=['-std=c++11', '-O3'],
    extra_link_args=[]
)

setup(
    name='arrayprocessor',
    version='1.0',
    description='Array processing module',
    ext_modules=[extension]
)