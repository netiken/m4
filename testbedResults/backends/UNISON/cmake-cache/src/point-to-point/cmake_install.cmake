# Install script for directory: /data1/lichenni/m4/testbedResults/backends/UNISON/src/point-to-point

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

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libns3.39-point-to-point-optimized.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libns3.39-point-to-point-optimized.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libns3.39-point-to-point-optimized.so"
         RPATH "/usr/local/lib:$ORIGIN/:$ORIGIN/../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/data1/lichenni/m4/testbedResults/backends/UNISON/build/lib/libns3.39-point-to-point-optimized.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libns3.39-point-to-point-optimized.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libns3.39-point-to-point-optimized.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libns3.39-point-to-point-optimized.so"
         OLD_RPATH "/data1/lichenni/m4/testbedResults/backends/UNISON/build/lib:"
         NEW_RPATH "/usr/local/lib:$ORIGIN/:$ORIGIN/../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libns3.39-point-to-point-optimized.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ns3" TYPE FILE FILES
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/point-to-point/helper/point-to-point-helper.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/point-to-point/model/point-to-point-channel.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/point-to-point/model/point-to-point-net-device.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/point-to-point/model/ppp-header.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/point-to-point/model/cn-header.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/point-to-point/model/pause-header.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/point-to-point/model/pint.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/point-to-point/model/qbb-channel.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/point-to-point/model/qbb-header.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/point-to-point/model/qbb-net-device.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/point-to-point/model/qbb-remote-channel.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/point-to-point/model/rdma-driver.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/point-to-point/model/rdma-hw.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/point-to-point/model/rdma-queue-pair.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/point-to-point/model/switch-mmu.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/point-to-point/model/switch-node.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/point-to-point/model/trace-format.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/point-to-point/helper/qbb-helper.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/point-to-point/helper/sim-setting.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/build/include/ns3/point-to-point-module.h"
    )
endif()

