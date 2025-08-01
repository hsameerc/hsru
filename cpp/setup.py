from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Optional: Allow override via environment variable
libtorch_path = os.getenv(
    'LIBTORCH_PATH',
    os.path.join(os.path.dirname(__file__), 'libtorch')  # fallback path
)

setup(
    name='hsru_cuda_kernel',
    version='0.1.0',
    author='Sameer Humagain',
    author_email='im@hsameer.com.np',
    description='A fast, dynamic, and robust CUDA kernel for the HSRU model',
    long_description=(
        'The HSRU is a recurrent neural network built by stacking custom '
        '`DualStateLIFLayer` cells. Each cell maintains two distinct internal states, '
        'mimicking the membrane potential and refractory state of a Leaky-Integrate-and-Fire (LIF) neuron.'
    ),
    packages=find_packages(),

    ext_modules=[
        CUDAExtension(
            name='hsru_cuda_kernel',
            sources=[
                'hsru_cpp/hsru_binding.cpp',
                'hsru_cpp/hsru_kernel.cu',
            ],
            include_dirs=[
                os.path.join(libtorch_path, 'include'),
                os.path.join(libtorch_path, 'include', 'torch', 'csrc', 'api', 'include'),
            ],
            library_dirs=[
                os.path.join(libtorch_path, 'lib')
            ],
            libraries=[
                'c10',
                'torch',
                'torch_cpu',
                'torch_cuda'  # ✅ Required for GPU ops
            ],
            extra_compile_args={
                'cxx': ['/O2'] if os.name == 'nt' else ['-O2'],
                'nvcc': ['--use_fast_math', '--expt-relaxed-constexpr']
            }
        ),
    ],
    cmdclass={'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)},
    zip_safe=False,
)
