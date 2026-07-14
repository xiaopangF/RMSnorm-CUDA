import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def prepend_to_path(path):
    if not path:
        return
    current = os.environ.get("PATH", "")
    parts = current.split(os.pathsep) if current else []
    if path not in parts:
        os.environ["PATH"] = path + os.pathsep + current


prepend_to_path(os.environ.get("BUILD_PATH_PREFIX"))
prepend_to_path(os.environ.get("NVCC_CCBIN"))
prepend_to_path(os.path.dirname(os.environ.get("CL_EXE", "")))
prepend_to_path(os.path.dirname(os.environ.get("NINJA_EXE", "")))

nvcc_args = ["-O3", "-allow-unsupported-compiler"]
nvcc_ccbin = os.environ.get("NVCC_CCBIN")
if nvcc_ccbin:
    nvcc_args.append(f"-ccbin={nvcc_ccbin}")


class LocalBuildExtension(BuildExtension):
    def build_extensions(self):
        if getattr(self.compiler, "compiler_type", None) == "msvc":
            if not self.compiler.initialized:
                self.compiler.initialize()

            cl_exe = os.environ.get("CL_EXE")
            link_exe = os.environ.get("LINK_EXE")
            lib_exe = os.environ.get("LIB_EXE")
            if cl_exe:
                self.compiler.cc = cl_exe
            if link_exe:
                self.compiler.linker = link_exe
            if lib_exe:
                self.compiler.lib = lib_exe

        super().build_extensions()


setup(
    name="rmsnorm_cuda",
    version="0.1.0",
    packages=["rmsnorm_cuda"],
    ext_modules=[
        CUDAExtension(
            name="rmsnorm_cuda._C",
            sources=[
                "csrc/binding.cpp",
                "csrc/rmsnorm_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["/O2"],
                "nvcc": nvcc_args,
            },
        )
    ],
    cmdclass={"build_ext": LocalBuildExtension},
)
