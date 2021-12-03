from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name='project3-cuda-extension',
    ext_modules=[
        CUDAExtension('project3_cuda',
            sources=['project3_kernel.cu', 'project3.cpp'],
            extra_compile_args={'cxx': [], 'nvcc': ['-O3']}),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
