# Install script for directory: /data1/lichenni/m4/testbedResults/backends/UNISON/src/mesh

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
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libns3.39-mesh-optimized.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libns3.39-mesh-optimized.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libns3.39-mesh-optimized.so"
         RPATH "/usr/local/lib:$ORIGIN/:$ORIGIN/../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/data1/lichenni/m4/testbedResults/backends/UNISON/build/lib/libns3.39-mesh-optimized.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libns3.39-mesh-optimized.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libns3.39-mesh-optimized.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libns3.39-mesh-optimized.so"
         OLD_RPATH "/data1/lichenni/m4/testbedResults/backends/UNISON/build/lib:"
         NEW_RPATH "/usr/local/lib:$ORIGIN/:$ORIGIN/../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libns3.39-mesh-optimized.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ns3" TYPE FILE FILES
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/mesh/helper/dot11s/dot11s-installer.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/mesh/helper/flame/flame-installer.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/mesh/helper/mesh-helper.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/mesh/helper/mesh-stack-installer.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/mesh/model/dot11s/dot11s-mac-header.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/mesh/model/dot11s/hwmp-protocol.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/mesh/model/dot11s/hwmp-rtable.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/mesh/model/dot11s/ie-dot11s-beacon-timing.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/mesh/model/dot11s/ie-dot11s-configuration.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/mesh/model/dot11s/ie-dot11s-id.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/mesh/model/dot11s/ie-dot11s-metric-report.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/mesh/model/dot11s/ie-dot11s-peer-management.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/mesh/model/dot11s/ie-dot11s-peering-protocol.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/mesh/model/dot11s/ie-dot11s-perr.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/mesh/model/dot11s/ie-dot11s-prep.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/mesh/model/dot11s/ie-dot11s-preq.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/mesh/model/dot11s/ie-dot11s-rann.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/mesh/model/dot11s/peer-link-frame.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/mesh/model/dot11s/peer-link.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/mesh/model/dot11s/peer-management-protocol.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/mesh/model/flame/flame-header.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/mesh/model/flame/flame-protocol-mac.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/mesh/model/flame/flame-protocol.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/mesh/model/flame/flame-rtable.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/mesh/model/mesh-information-element-vector.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/mesh/model/mesh-l2-routing-protocol.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/mesh/model/mesh-point-device.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/mesh/model/mesh-wifi-beacon.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/mesh/model/mesh-wifi-interface-mac-plugin.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/mesh/model/mesh-wifi-interface-mac.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/build/include/ns3/mesh-module.h"
    )
endif()

