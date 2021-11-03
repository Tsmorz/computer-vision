from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name='project2 CUDA extension',
    ext_modules=[
        CUDAExtension('project2_cuda',
            sources=['project2_kernel.cu', 'project2.cpp'],
            extra_compile_args={'cxx': [], 'nvcc': ['-O3']}),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
