cmake_minimum_required(VERSION 3.10)
project(OpenCLBenchmarks
        VERSION 0.1
        DESCRIPTION "Benchmarks for evaluating FPGA/OpenCL performance"
        LANGUAGES CXX)
set(default_build_type "Release")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)



find_package(benchmark REQUIRED)
find_package(spdlog REQUIRED)

include("cmake/SetupFPGA.cmake")

set(OPENCL_INCLUDE_DIRECTORY $ENV{INTELFPGAOCLSDKROOT}/host/include)

add_executable(run_all bin/run_all.cpp)
target_compile_features(run_all PUBLIC cxx_std_14)
target_compile_options(run_all PRIVATE -Wall -Wextra)
target_link_libraries(run_all PUBLIC spdlog::spdlog benchmark::benchmark ${AOCL_LINK_LIBRARIES})
target_include_directories(run_all PUBLIC ${OPENCL_INCLUDE_DIRECTORY})
target_compile_options(run_all PRIVATE ${AOCL_COMPILER_OPTS})
target_include_directories(
  run_all
  PUBLIC
    include/
)

