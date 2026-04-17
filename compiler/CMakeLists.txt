cmake_minimum_required(VERSION 3.20)
project(remora LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# MLIR packages ship a CMake config file (MLIRConfig.cmake) so find_package() works.
# StableHLO doesn't. It has no install target that generates one.
# Instead we add its source and build directories to the include/link paths directly.
if(NOT DEFINED STABLEHLO_ROOT)
  message(FATAL_ERROR "Set STABLEHLO_ROOT to your stablehlo source directory")
endif()
if(NOT DEFINED STABLEHLO_BUILD)
  message(FATAL_ERROR "Set STABLEHLO_BUILD to your stablehlo build directory")
endif()

include_directories(${STABLEHLO_ROOT})
include_directories(${STABLEHLO_BUILD})
link_directories(${STABLEHLO_BUILD}/lib)

# Optional: path to an LLVM build that has NVPTX backends (libLLVMNVPTX*.a).
# Required only for --emit-ptx. Defaults to MLIR_DIR/../.. (i.e. same llvm-build).
if(NOT DEFINED LLVM_NVPTX_LIB_DIR)
  set(LLVM_NVPTX_LIB_DIR "${MLIR_DIR}/../../..")
endif()
link_directories(${LLVM_NVPTX_LIB_DIR})

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

add_executable(remora main.cpp cpu.cpp cpu_tests.cpp gpu.cpp gpu_tests.cpp pipeline.cpp)

# Optional CUDA harness — enables --run-gpu flag.
# Requires CUDA Toolkit on the target machine (not needed for --emit-ptx on Mac).
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
  message(STATUS "CUDA Toolkit found, enabling --run-gpu harness")
  target_link_libraries(remora PRIVATE CUDA::cuda_driver)
  target_compile_definitions(remora PRIVATE REMORA_CUDA)
else()
  message(STATUS "CUDA Toolkit not found — --emit-ptx works, --run-gpu disabled")
endif()

target_link_libraries(remora PRIVATE
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRArithDialect
  MLIRTensorDialect
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
  MLIRRegisterAllPasses
  StablehloBase
  StablehloAssemblyFormat
  StablehloBroadcastUtils
  StablehloBroadcastLowering
  StablehloLinalgTransforms
  StablehloOps
  StablehloPasses
  StablehloRegister
  StablehloTypeConversion
  StablehloTypeInference
  VhloOps
  VhloTypes
  ChloOps
  MLIRExecutionEngine
  # GPU / PTX emission
  MLIRGPUDialect
  MLIRGPUTransforms
  MLIRSCFToGPU
  MLIRNVVMDialect
  MLIRGPUToNVVMTransforms
  MLIRNVVMToLLVMIRTranslation
  MLIRGPUToLLVMIRTranslation
  MLIRTargetLLVMIRExport
  LLVMNVPTXCodeGen
  LLVMNVPTXDesc
  LLVMNVPTXInfo
)
