from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os, platform
os.path.dirname(os.path.abspath(__file__))

nvcc_args = [
    "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/"),
]
cxx_args = []

if platform.system() == "Windows":
    nvcc_args += ["-DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING", "-Xcompiler", "/Zc:preprocessor"]
    cxx_args = ["/Zc:preprocessor"]
else:
    # Linux: add CCCL and CUDA runtime include paths
    conda_prefix = os.environ.get('CONDA_PREFIX', '')
    if conda_prefix:
        nvcc_args.append('-I' + os.path.join(conda_prefix, 'targets/x86_64-linux/include'))
        nvcc_args.append('-I' + os.path.join(conda_prefix, 'lib/python3.10/site-packages/nvidia/cuda_runtime/include'))

setup(
    name="diff_gaussian_rasterization",
    packages=['diff_gaussian_rasterization'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "rasterize_points.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": nvcc_args, "cxx": cxx_args})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
