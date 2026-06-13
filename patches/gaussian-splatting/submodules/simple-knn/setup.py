from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os, platform

cxx_compiler_flags = []
if os.name == 'nt':
    cxx_compiler_flags.append("/wd4624")

nvcc_args = []
if platform.system() == "Windows":
    nvcc_args = ["-DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING", "-Xcompiler", "/Zc:preprocessor"]
    cxx_compiler_flags.append("/Zc:preprocessor")
else:
    conda_prefix = os.environ.get('CONDA_PREFIX', '')
    if conda_prefix:
        nvcc_args.append('-I' + os.path.join(conda_prefix, 'targets/x86_64-linux/include'))
        nvcc_args.append('-I' + os.path.join(conda_prefix, 'lib/python3.10/site-packages/nvidia/cuda_runtime/include'))

setup(
    name="simple_knn",
    ext_modules=[
        CUDAExtension(
            name="simple_knn._C",
            sources=[
            "spatial.cu", 
            "simple_knn.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": nvcc_args, "cxx": cxx_compiler_flags})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
