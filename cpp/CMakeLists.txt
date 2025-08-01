# ===================================================================
# === HSRU CUDA Kernel CMake Build System (Production-Grade)       ===
# ===================================================================
cmake_minimum_required(VERSION 3.18)
project(hsru_cuda_kernel LANGUAGES CXX CUDA)

# Optional: Set CUDA Toolkit path manually (Windows-specific)
set(CUDAToolkit_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8")

# --- Build Configuration ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_COMPILER "nvcc")

# --- Find Dependencies ---
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(Torch REQUIRED)
find_package(pybind11 REQUIRED)
find_package(CUDAToolkit REQUIRED)

# --- Target Definition ---
pybind11_add_module(_core MODULE
    hsru_cpp/hsru_binding.cpp
    hsru_cpp/hsru_kernel.cu
)

# Include Directories
target_include_directories(_core PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
)

# Link Libraries
target_link_libraries(_core PRIVATE
    ${TORCH_LIBRARIES}
    CUDA::cudart
    pybind11::module
    "${TORCH_INSTALL_PREFIX}/lib/torch_python.lib"
)

# Compile Definitions
target_compile_definitions(_core PRIVATE
    TORCH_API_INCLUDE_EXTENSION_H
    TORCH_EXTENSION_NAME=hsru_cuda_kernel
)

# Compile Features
target_compile_features(_core PRIVATE cxx_std_17)

# CUDA Properties
set_target_properties(_core PROPERTIES
    CUDA_ARCHITECTURES "75;80;86;90"
    CUDA_SEPARABLE_COMPILATION ON
)

# Python Extension Properties
set_target_properties(_core PROPERTIES
    PREFIX ""  # Avoid 'lib' prefix
    SUFFIX ".pyd"  # Ensure correct extension for Python
)

# Install Target
install(TARGETS _core
    LIBRARY DESTINATION hsru_cuda_kernel  # This creates hsru_cuda_kernel/ in the wheel
)

# Debug Output
message(STATUS "✅ Python version: ${Python3_VERSION}")
message(STATUS "✅ PyTorch include dirs: ${TORCH_INCLUDE_DIRS}")
message(STATUS "✅ PyTorch libraries: ${TORCH_LIBRARIES}")
message(STATUS "✅ Torch install prefix: ${TORCH_INSTALL_PREFIX}")
message(STATUS "✅ CUDA Toolkit include dirs: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "✅ Pybind11 include dirs: ${pybind11_INCLUDE_DIRS}")
