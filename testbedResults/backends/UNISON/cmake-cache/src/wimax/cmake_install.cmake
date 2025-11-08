# Install script for directory: /data1/lichenni/m4/testbedResults/backends/UNISON/src/wimax

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
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libns3.39-wimax-optimized.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libns3.39-wimax-optimized.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libns3.39-wimax-optimized.so"
         RPATH "/usr/local/lib:$ORIGIN/:$ORIGIN/../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/data1/lichenni/m4/testbedResults/backends/UNISON/build/lib/libns3.39-wimax-optimized.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libns3.39-wimax-optimized.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libns3.39-wimax-optimized.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libns3.39-wimax-optimized.so"
         OLD_RPATH "/data1/lichenni/m4/testbedResults/backends/UNISON/build/lib:"
         NEW_RPATH "/usr/local/lib:$ORIGIN/:$ORIGIN/../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libns3.39-wimax-optimized.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ns3" TYPE FILE FILES
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wimax/helper/wimax-helper.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wimax/model/wimax-channel.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wimax/model/wimax-net-device.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wimax/model/bs-net-device.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wimax/model/ss-net-device.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wimax/model/cid.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wimax/model/cid-factory.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wimax/model/ofdm-downlink-frame-prefix.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wimax/model/wimax-connection.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wimax/model/ss-record.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wimax/model/mac-messages.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wimax/model/dl-mac-messages.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wimax/model/ul-mac-messages.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wimax/model/wimax-phy.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wimax/model/simple-ofdm-wimax-phy.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wimax/model/simple-ofdm-wimax-channel.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wimax/model/send-params.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wimax/model/service-flow.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wimax/model/ss-manager.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wimax/model/connection-manager.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wimax/model/wimax-mac-header.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wimax/model/wimax-mac-queue.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wimax/model/crc8.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wimax/model/service-flow-manager.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wimax/model/bs-uplink-scheduler.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wimax/model/bs-uplink-scheduler-simple.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wimax/model/bs-uplink-scheduler-mbqos.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wimax/model/bs-uplink-scheduler-rtps.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wimax/model/ul-job.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wimax/model/bs-scheduler.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wimax/model/bs-scheduler-simple.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wimax/model/bs-scheduler-rtps.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wimax/model/service-flow-record.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wimax/model/snr-to-block-error-rate-record.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wimax/model/snr-to-block-error-rate-manager.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wimax/model/simple-ofdm-send-param.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wimax/model/ss-service-flow-manager.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wimax/model/bs-service-flow-manager.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wimax/model/cs-parameters.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wimax/model/ipcs-classifier-record.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wimax/model/wimax-tlv.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wimax/model/ipcs-classifier.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wimax/model/bvec.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wimax/model/wimax-mac-to-mac-header.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/build/include/ns3/wimax-module.h"
    )
endif()

