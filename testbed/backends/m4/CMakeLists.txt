cmake_minimum_required(VERSION 3.10)
project(FlowSim)

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags for maximum performance
#set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fPIC -march=native -mtune=native -ffast-math -funroll-loops -fopenmp -fno-exceptions -fno-plt -floop-parallelize-all -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-g -O3")

set(CMAKE_BUILD_TYPE Debug)


# Locate LibTorch from virtual environment or conda
# Priority: 1) CONDA_PREFIX env var, 2) Auto-detect .venv in project root
if(DEFINED ENV{CONDA_PREFIX})
    set(PYTHON_ENV_PREFIX $ENV{CONDA_PREFIX})
    message(STATUS "Using Python environment from CONDA_PREFIX: ${PYTHON_ENV_PREFIX}")
else()
    # Auto-detect .venv in project root (for uv-managed venvs)
    set(VENV_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../.venv")
    if(EXISTS "${VENV_PATH}")
        set(PYTHON_ENV_PREFIX "${VENV_PATH}")
        message(STATUS "Auto-detected virtual environment: ${PYTHON_ENV_PREFIX}")
    else()
        get_filename_component(PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
        message(FATAL_ERROR "Python environment not found. Please either:\n"
                            "  1) Activate conda environment, OR\n"
                            "  2) Install PyTorch in .venv: cd ${PROJECT_ROOT} && uv pip install torch torchvision")
    endif()
endif()

# Locate the Torch CMake package in the Python environment
file(GLOB TORCH_CMAKE_DIRS "${PYTHON_ENV_PREFIX}/lib/python*/site-packages/torch/share/cmake/Torch")
if(NOT TORCH_CMAKE_DIRS)
    get_filename_component(PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
    message(FATAL_ERROR "Unable to locate LibTorch under ${PYTHON_ENV_PREFIX}.\n"
                        "Install PyTorch: cd ${PROJECT_ROOT} && uv pip install torch torchvision")
endif()
list(SORT TORCH_CMAKE_DIRS)
list(REVERSE TORCH_CMAKE_DIRS)
list(GET TORCH_CMAKE_DIRS 0 TORCH_CMAKE_DIR)
set(CMAKE_PREFIX_PATH "${TORCH_CMAKE_DIR}")
set(Torch_DIR "${TORCH_CMAKE_DIR}")
message(STATUS "Found LibTorch at: ${TORCH_CMAKE_DIR}")

# Ensure we do not attempt to use CUDA when building with LibTorch.
set(Torch_USE_CUDA OFF CACHE BOOL "Disable CUDA for Torch" FORCE)
set(CAFFE2_USE_CUDA OFF CACHE BOOL "Disable CUDA for Caffe2" FORCE)
set(USE_CUDA OFF CACHE BOOL "Disable CUDA globally" FORCE)
set(TORCH_CUDA_ARCH_LIST "" CACHE STRING "Disable CUDA architecture list" FORCE)
set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "Default CUDA architectures to avoid auto-detection" FORCE)

# Find the Torch package
find_package(Torch REQUIRED)

# Add shared library target
#add_library(inference_shared SHARED src/inference.cpp)
#add_library(flowsim SHARED main.cpp)

# Only keep the integrated no_flowsim target - no separate flowsim needed

SET(NOFLOWSIM_SOURCES
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

# Include directories AFTER adding subdirectories to avoid polluting subprojects
target_include_directories(no_flowsim PRIVATE "${PYTHON_ENV_PREFIX}/include")
target_include_directories(no_flowsim PRIVATE ${TORCH_INCLUDE_DIRS})
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
