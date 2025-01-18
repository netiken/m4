cmake_minimum_required(VERSION 3.10)
project(FlowSim)

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags for maximum performance
#set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fPIC -march=native -mtune=native -ffast-math -funroll-loops -fopenmp -fno-exceptions -fno-plt -floop-parallelize-all -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-g -O0")

set(CMAKE_BUILD_TYPE Debug)


# Use CONDA_PREFIX to dynamically locate the LibTorch path
if(DEFINED ENV{CONDA_PREFIX})
    set(CONDA_PREFIX $ENV{CONDA_PREFIX})
else()
    message(FATAL_ERROR "CONDA_PREFIX is not set. Please activate your conda environment.")
endif()

# Adjust the Python version in the path if necessary
set(CMAKE_PREFIX_PATH "${CONDA_PREFIX}/lib/python3.10/site-packages/torch/share/cmake/Torch")

# Find the Torch package
find_package(Torch REQUIRED)

# Include directories
include_directories("${CONDA_PREFIX}/include")
include_directories(${TORCH_INCLUDE_DIRS})
#include_directories("/data1/lichenni/projects/per-flow-sim/flowsim/yaml-cpp/build")

# Add shared library target
#add_library(inference_shared SHARED src/inference.cpp)
#add_library(flowsim SHARED main.cpp)

set(SOURCES
    main_m4.cpp
    Device.h
    Device.cpp
    Chunk.h
    Chunk.cpp
    Event.h
    Event.cpp
    EventQueue.h 
    EventQueue.cpp 
    Link.h 
    Link.cpp 
    TopologyBuilder.h 
    TopologyBuilder.cpp 
    Topology.h 
    Topology.cpp 
    Type.h
)

SET(NOFLOWSIM_SOURCES
    main_m4_noflowsim.cpp
)

# Add executable target (only main.cpp)
add_executable(flowsim ${SOURCES})
add_executable(no_flowsim ${NOFLOWSIM_SOURCES})

# Link the Torch library and inference_shared to inference_executable
target_link_libraries(flowsim "${TORCH_LIBRARIES}")
target_link_libraries(no_flowsim "${TORCH_LIBRARIES}")

#add_subdirectory(rapidyaml/ext/c4core)
add_subdirectory(rapidyaml)

target_include_directories(flowsim PRIVATE rapidyaml/src rapidyaml/src/c4)
target_link_libraries(flowsim ryml)

target_include_directories(no_flowsim PRIVATE rapidyaml/src rapidyaml/src/c4)
target_link_libraries(no_flowsim ryml)

# Enable link-time optimization
#set_property(TARGET inference_shared PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
set_property(TARGET flowsim PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
set_property(TARGET no_flowsim PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)

# CUDA-specific optimizations (if using GPU)
if (TORCH_CUDA_ARCH_LIST)
    #set_target_properties(inference_shared PROPERTIES CUDA_ARCHITECTURES "${TORCH_CUDA_ARCH_LIST}")
    set_target_properties(flowsim PROPERTIES CUDA_ARCHITECTURES "${TORCH_CUDA_ARCH_LIST}")
    set_target_properties(no_flowsim PROPERTIES CUDA_ARCHITECTURES "${TORCH_CUDA_ARCH_LIST}")
endif()

# Set RPATH to find Torch libraries at runtime
#set_target_properties(inference_shared PROPERTIES
#    BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
#    INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
#)

set_target_properties(flowsim PROPERTIES
    BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
    INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
)

set_target_properties(no_flowsim PROPERTIES
    BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
    INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
)

