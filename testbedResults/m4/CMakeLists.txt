cmake_minimum_required(VERSION 3.10)
project(FlowSim)

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags for maximum performance
#set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fPIC -march=native -mtune=native -ffast-math -funroll-loops -fopenmp -fno-exceptions -fno-plt -floop-parallelize-all -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-g -O3")

set(CMAKE_BUILD_TYPE Debug)


# Use CONDA_PREFIX to dynamically locate the LibTorch path
if(DEFINED ENV{CONDA_PREFIX})
    set(CONDA_PREFIX $ENV{CONDA_PREFIX})
else()
    message(FATAL_ERROR "CONDA_PREFIX is not set. Please activate your conda environment.")
endif()

# Locate the Torch CMake package relative to the active conda environment.
file(GLOB TORCH_CMAKE_DIRS "${CONDA_PREFIX}/lib/python*/site-packages/torch/share/cmake/Torch")
if(NOT TORCH_CMAKE_DIRS)
    message(FATAL_ERROR "Unable to locate the Torch CMake package under ${CONDA_PREFIX}.")
endif()
list(SORT TORCH_CMAKE_DIRS)
list(REVERSE TORCH_CMAKE_DIRS)
list(GET TORCH_CMAKE_DIRS 0 TORCH_CMAKE_DIR)
set(CMAKE_PREFIX_PATH "${TORCH_CMAKE_DIR}")
set(Torch_DIR "${TORCH_CMAKE_DIR}")

# Ensure we do not attempt to use CUDA when building with LibTorch.
set(Torch_USE_CUDA OFF CACHE BOOL "Disable CUDA for Torch" FORCE)
set(CAFFE2_USE_CUDA OFF CACHE BOOL "Disable CUDA for Caffe2" FORCE)
set(USE_CUDA OFF CACHE BOOL "Disable CUDA globally" FORCE)
set(TORCH_CUDA_ARCH_LIST "" CACHE STRING "Disable CUDA architecture list" FORCE)
set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "Default CUDA architectures to avoid auto-detection" FORCE)

# Find the Torch package
find_package(Torch REQUIRED)

# Include directories
include_directories("${CONDA_PREFIX}/include")
include_directories(${TORCH_INCLUDE_DIRS})
#include_directories("/data1/lichenni/projects/per-flow-sim/flowsim/yaml-cpp/build")

# Add shared library target
#add_library(inference_shared SHARED src/inference.cpp)
#add_library(flowsim SHARED main.cpp)

# Only keep the integrated no_flowsim target - no separate flowsim needed

SET(NOFLOWSIM_SOURCES
    main_m4_noflowsim.cpp
    Device.h
    Device.cpp
    Chunk.h
    Chunk.cpp
    Event.h
    Event.cpp
    EventQueue.h 
    EventQueue.cpp 
    EventList.h
    EventList.cpp
    Link.h 
    Link.cpp 
    TopologyBuilder.h 
    TopologyBuilder.cpp 
    Topology.h 
    Topology.cpp 
    Type.h
)

# Add executable target - only the integrated no_flowsim
add_executable(no_flowsim ${NOFLOWSIM_SOURCES})

# Set linker flags to reduce binary size
set_target_properties(no_flowsim PROPERTIES
    LINK_FLAGS "-Wl,--strip-all -Wl,--gc-sections"
)

# Link the Torch library to the integrated executable
target_link_libraries(no_flowsim "${TORCH_LIBRARIES}")

#add_subdirectory(rapidyaml/ext/c4core)
add_subdirectory(rapidyaml)

target_include_directories(no_flowsim PRIVATE rapidyaml/src rapidyaml/src/c4)
target_include_directories(no_flowsim PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/rdma_bench)
target_link_libraries(no_flowsim ryml)

# Link CityHash C object to satisfy CityHash128 used via mica.h
add_library(cityhash_c OBJECT rdma_bench/mica/city.c)
target_link_libraries(no_flowsim cityhash_c)

# Ensure C symbols are visible when linking with C++
set_target_properties(no_flowsim PROPERTIES LINKER_LANGUAGE CXX)

# Compile city.c as C++ so symbols match C++ caller
set_source_files_properties(rdma_bench/mica/city.c PROPERTIES LANGUAGE CXX)

# Enable link-time optimization
set_property(TARGET no_flowsim PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)

# CUDA-specific optimizations (if using GPU)
if (TORCH_CUDA_ARCH_LIST)
    set_target_properties(no_flowsim PROPERTIES CUDA_ARCHITECTURES "${TORCH_CUDA_ARCH_LIST}")
endif()

# Set RPATH to find Torch libraries at runtime
set_target_properties(no_flowsim PROPERTIES
    BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
    INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
)

# Prepend the system libstdc++ path to RPATH so runtime prefers GLIBCXX from the system
# This mirrors the manual LD_LIBRARY_PATH workaround (using /usr/lib/x86_64-linux-gnu first)
set(SYSTEM_CXXLIB_PATH "/usr/lib/x86_64-linux-gnu")
set_property(TARGET no_flowsim APPEND PROPERTY BUILD_RPATH "${SYSTEM_CXXLIB_PATH}")
set_property(TARGET no_flowsim APPEND PROPERTY INSTALL_RPATH "${SYSTEM_CXXLIB_PATH}")
target_link_options(no_flowsim PRIVATE -Wl,-rpath,${SYSTEM_CXXLIB_PATH})
