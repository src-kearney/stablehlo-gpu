cmake_minimum_required(VERSION 3.20)
project(remora-compiler LANGUAGES CXX C)

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

# StableHLO — no config file, point directly at headers and libs
if(NOT DEFINED STABLEHLO_ROOT)
  message(FATAL_ERROR "Set STABLEHLO_ROOT to your stablehlo source directory")
endif()
if(NOT DEFINED STABLEHLO_BUILD)
  message(FATAL_ERROR "Set STABLEHLO_BUILD to your stablehlo build directory")
endif()

include_directories(${STABLEHLO_ROOT})
include_directories(${STABLEHLO_BUILD})
link_directories(${STABLEHLO_BUILD}/lib)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

add_executable(remora-compiler main.cpp)

target_link_libraries(remora-compiler PRIVATE
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
)