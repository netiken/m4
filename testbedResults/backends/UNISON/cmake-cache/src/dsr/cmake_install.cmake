# Install script for directory: /data1/lichenni/m4/testbedResults/backends/UNISON/src/dsr

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
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libns3.39-dsr-optimized.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libns3.39-dsr-optimized.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libns3.39-dsr-optimized.so"
         RPATH "/usr/local/lib:$ORIGIN/:$ORIGIN/../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/data1/lichenni/m4/testbedResults/backends/UNISON/build/lib/libns3.39-dsr-optimized.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libns3.39-dsr-optimized.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libns3.39-dsr-optimized.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libns3.39-dsr-optimized.so"
         OLD_RPATH "/data1/lichenni/m4/testbedResults/backends/UNISON/build/lib:"
         NEW_RPATH "/usr/local/lib:$ORIGIN/:$ORIGIN/../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libns3.39-dsr-optimized.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ns3" TYPE FILE FILES
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/dsr/helper/dsr-helper.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/dsr/helper/dsr-main-helper.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/dsr/model/dsr-errorbuff.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/dsr/model/dsr-fs-header.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/dsr/model/dsr-gratuitous-reply-table.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/dsr/model/dsr-maintain-buff.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/dsr/model/dsr-network-queue.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/dsr/model/dsr-option-header.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/dsr/model/dsr-options.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/dsr/model/dsr-passive-buff.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/dsr/model/dsr-rcache.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/dsr/model/dsr-routing.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/dsr/model/dsr-rreq-table.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/dsr/model/dsr-rsendbuff.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/build/include/ns3/dsr-module.h"
    )
endif()

