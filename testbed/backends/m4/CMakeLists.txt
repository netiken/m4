cmake_minimum_required(VERSION 3.10)
project(FlowSim)

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_RELEASE "-g -O3")
set(CMAKE_BUILD_TYPE Debug)


# Locate LibTorch from conda or .venv
if(DEFINED ENV{CONDA_PREFIX})
    set(PYTHON_ENV_PREFIX $ENV{CONDA_PREFIX})
    message(STATUS "Using Python environment from CONDA_PREFIX: ${PYTHON_ENV_PREFIX}")
else()
    set(VENV_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../.venv")
    if(EXISTS "${VENV_PATH}")
        set(PYTHON_ENV_PREFIX "${VENV_PATH}")
        message(STATUS "Auto-detected virtual environment: ${PYTHON_ENV_PREFIX}")
    else()
        get_filename_component(PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
        message(FATAL_ERROR "Python environment not found. Install PyTorch: cd ${PROJECT_ROOT} && uv pip install torch")
    endif()
endif()

# Find LibTorch
file(GLOB TORCH_CMAKE_DIRS "${PYTHON_ENV_PREFIX}/lib/python*/site-packages/torch/share/cmake/Torch")
if(NOT TORCH_CMAKE_DIRS)
    get_filename_component(PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
    message(FATAL_ERROR "Unable to locate LibTorch. Install PyTorch: cd ${PROJECT_ROOT} && uv pip install torch")
endif()
list(SORT TORCH_CMAKE_DIRS)
list(REVERSE TORCH_CMAKE_DIRS)
list(GET TORCH_CMAKE_DIRS 0 TORCH_CMAKE_DIR)
set(CMAKE_PREFIX_PATH "${TORCH_CMAKE_DIR}")
set(Torch_DIR "${TORCH_CMAKE_DIR}")
message(STATUS "Found LibTorch at: ${TORCH_CMAKE_DIR}")

find_package(Torch REQUIRED)

# M4 executable with integrated ML inference
SET(M4_SOURCES
    main.cpp
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

# Add executable target
add_executable(main ${M4_SOURCES})

# Set linker flags to reduce binary size
set_target_properties(main PROPERTIES
    LINK_FLAGS "-Wl,--strip-all -Wl,--gc-sections"
)

target_link_libraries(main "${TORCH_LIBRARIES}")

target_include_directories(main PRIVATE "${PYTHON_ENV_PREFIX}/include")
target_include_directories(main PRIVATE ${TORCH_INCLUDE_DIRS})
target_include_directories(main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/rdma_bench)

# Link CityHash
add_library(cityhash_c OBJECT rdma_bench/mica/city.c)
target_link_libraries(main cityhash_c)
set_target_properties(main PROPERTIES LINKER_LANGUAGE CXX)
set_source_files_properties(rdma_bench/mica/city.c PROPERTIES LANGUAGE CXX)

# Optimizations
set_property(TARGET main PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
if (TORCH_CUDA_ARCH_LIST)
    set_target_properties(main PROPERTIES CUDA_ARCHITECTURES "${TORCH_CUDA_ARCH_LIST}")
endif()

# RPATH configuration
set(SYSTEM_CXXLIB_PATH "/usr/lib/x86_64-linux-gnu")
set_target_properties(main PROPERTIES
    BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib;${SYSTEM_CXXLIB_PATH}"
    INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib;${SYSTEM_CXXLIB_PATH}"
)
target_link_options(main PRIVATE -Wl,-rpath,${SYSTEM_CXXLIB_PATH})
