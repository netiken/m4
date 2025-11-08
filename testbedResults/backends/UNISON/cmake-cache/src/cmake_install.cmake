# Install script for directory: /data1/lichenni/m4/testbedResults/backends/UNISON/src

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/data1/lichenni/m4/testbedResults/backends/UNISON/cmake-cache/src/antenna/cmake_install.cmake")
  include("/data1/lichenni/m4/testbedResults/backends/UNISON/cmake-cache/src/aodv/cmake_install.cmake")
  include("/data1/lichenni/m4/testbedResults/backends/UNISON/cmake-cache/src/applications/cmake_install.cmake")
  include("/data1/lichenni/m4/testbedResults/backends/UNISON/cmake-cache/src/bridge/cmake_install.cmake")
  include("/data1/lichenni/m4/testbedResults/backends/UNISON/cmake-cache/src/brite/cmake_install.cmake")
  include("/data1/lichenni/m4/testbedResults/backends/UNISON/cmake-cache/src/buildings/cmake_install.cmake")
  include("/data1/lichenni/m4/testbedResults/backends/UNISON/cmake-cache/src/click/cmake_install.cmake")
  include("/data1/lichenni/m4/testbedResults/backends/UNISON/cmake-cache/src/config-store/cmake_install.cmake")
  include("/data1/lichenni/m4/testbedResults/backends/UNISON/cmake-cache/src/core/cmake_install.cmake")
  include("/data1/lichenni/m4/testbedResults/backends/UNISON/cmake-cache/src/csma/cmake_install.cmake")
  include("/data1/lichenni/m4/testbedResults/backends/UNISON/cmake-cache/src/csma-layout/cmake_install.cmake")
  include("/data1/lichenni/m4/testbedResults/backends/UNISON/cmake-cache/src/dsdv/cmake_install.cmake")
  include("/data1/lichenni/m4/testbedResults/backends/UNISON/cmake-cache/src/dsr/cmake_install.cmake")
  include("/data1/lichenni/m4/testbedResults/backends/UNISON/cmake-cache/src/energy/cmake_install.cmake")
  include("/data1/lichenni/m4/testbedResults/backends/UNISON/cmake-cache/src/fd-net-device/cmake_install.cmake")
  include("/data1/lichenni/m4/testbedResults/backends/UNISON/cmake-cache/src/flow-monitor/cmake_install.cmake")
  include("/data1/lichenni/m4/testbedResults/backends/UNISON/cmake-cache/src/internet/cmake_install.cmake")
  include("/data1/lichenni/m4/testbedResults/backends/UNISON/cmake-cache/src/internet-apps/cmake_install.cmake")
  include("/data1/lichenni/m4/testbedResults/backends/UNISON/cmake-cache/src/lr-wpan/cmake_install.cmake")
  include("/data1/lichenni/m4/testbedResults/backends/UNISON/cmake-cache/src/lte/cmake_install.cmake")
  include("/data1/lichenni/m4/testbedResults/backends/UNISON/cmake-cache/src/mesh/cmake_install.cmake")
  include("/data1/lichenni/m4/testbedResults/backends/UNISON/cmake-cache/src/mobility/cmake_install.cmake")
  include("/data1/lichenni/m4/testbedResults/backends/UNISON/cmake-cache/src/mtp/cmake_install.cmake")
  include("/data1/lichenni/m4/testbedResults/backends/UNISON/cmake-cache/src/netanim/cmake_install.cmake")
  include("/data1/lichenni/m4/testbedResults/backends/UNISON/cmake-cache/src/network/cmake_install.cmake")
  include("/data1/lichenni/m4/testbedResults/backends/UNISON/cmake-cache/src/nix-vector-routing/cmake_install.cmake")
  include("/data1/lichenni/m4/testbedResults/backends/UNISON/cmake-cache/src/olsr/cmake_install.cmake")
  include("/data1/lichenni/m4/testbedResults/backends/UNISON/cmake-cache/src/openflow/cmake_install.cmake")
  include("/data1/lichenni/m4/testbedResults/backends/UNISON/cmake-cache/src/point-to-point/cmake_install.cmake")
  include("/data1/lichenni/m4/testbedResults/backends/UNISON/cmake-cache/src/point-to-point-layout/cmake_install.cmake")
  include("/data1/lichenni/m4/testbedResults/backends/UNISON/cmake-cache/src/propagation/cmake_install.cmake")
  include("/data1/lichenni/m4/testbedResults/backends/UNISON/cmake-cache/src/sixlowpan/cmake_install.cmake")
  include("/data1/lichenni/m4/testbedResults/backends/UNISON/cmake-cache/src/spectrum/cmake_install.cmake")
  include("/data1/lichenni/m4/testbedResults/backends/UNISON/cmake-cache/src/stats/cmake_install.cmake")
  include("/data1/lichenni/m4/testbedResults/backends/UNISON/cmake-cache/src/tap-bridge/cmake_install.cmake")
  include("/data1/lichenni/m4/testbedResults/backends/UNISON/cmake-cache/src/topology-read/cmake_install.cmake")
  include("/data1/lichenni/m4/testbedResults/backends/UNISON/cmake-cache/src/traffic-control/cmake_install.cmake")
  include("/data1/lichenni/m4/testbedResults/backends/UNISON/cmake-cache/src/uan/cmake_install.cmake")
  include("/data1/lichenni/m4/testbedResults/backends/UNISON/cmake-cache/src/virtual-net-device/cmake_install.cmake")
  include("/data1/lichenni/m4/testbedResults/backends/UNISON/cmake-cache/src/wifi/cmake_install.cmake")
  include("/data1/lichenni/m4/testbedResults/backends/UNISON/cmake-cache/src/wimax/cmake_install.cmake")

endif()

