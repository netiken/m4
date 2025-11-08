# Install script for directory: /data1/lichenni/m4/testbedResults/backends/UNISON/src/network

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
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libns3.39-network-optimized.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libns3.39-network-optimized.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libns3.39-network-optimized.so"
         RPATH "/usr/local/lib:$ORIGIN/:$ORIGIN/../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/data1/lichenni/m4/testbedResults/backends/UNISON/build/lib/libns3.39-network-optimized.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libns3.39-network-optimized.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libns3.39-network-optimized.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libns3.39-network-optimized.so"
         OLD_RPATH "/data1/lichenni/m4/testbedResults/backends/UNISON/build/lib:"
         NEW_RPATH "/usr/local/lib:$ORIGIN/:$ORIGIN/../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libns3.39-network-optimized.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ns3" TYPE FILE FILES
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/helper/application-container.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/helper/delay-jitter-estimation.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/helper/net-device-container.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/helper/node-container.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/helper/packet-socket-helper.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/helper/simple-net-device-helper.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/helper/trace-helper.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/model/address.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/model/application.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/model/buffer.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/model/byte-tag-list.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/model/channel-list.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/model/channel.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/model/chunk.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/model/header.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/model/net-device.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/model/nix-vector.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/model/node-list.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/model/node.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/model/packet-metadata.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/model/packet-tag-list.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/model/packet.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/model/socket-factory.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/model/socket.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/model/tag-buffer.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/model/tag.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/model/trailer.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/test/header-serialization-test.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/utils/address-utils.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/utils/bit-deserializer.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/utils/bit-serializer.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/utils/crc32.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/utils/data-rate.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/utils/drop-tail-queue.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/utils/dynamic-queue-limits.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/utils/error-channel.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/utils/error-model.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/utils/ethernet-header.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/utils/ethernet-trailer.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/utils/flow-id-tag.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/utils/flow-id-tag-path.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/utils/flow-size-tag.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/utils/generic-phy.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/utils/inet-socket-address.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/utils/inet6-socket-address.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/utils/ipv4-address.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/utils/ipv6-address.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/utils/llc-snap-header.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/utils/lollipop-counter.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/utils/mac16-address.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/utils/mac48-address.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/utils/mac64-address.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/utils/mac8-address.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/utils/net-device-queue-interface.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/utils/output-stream-wrapper.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/utils/packet-burst.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/utils/packet-data-calculators.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/utils/packet-probe.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/utils/packet-socket-address.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/utils/packet-socket-client.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/utils/packet-socket-factory.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/utils/packet-socket-server.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/utils/packet-socket.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/utils/packetbb.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/utils/pcap-file-wrapper.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/utils/pcap-file.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/utils/pcap-test.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/utils/queue-fwd.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/utils/queue-item.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/utils/queue-limits.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/utils/queue-size.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/utils/queue.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/utils/radiotap-header.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/utils/sequence-number.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/utils/simple-channel.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/utils/simple-net-device.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/utils/sll-header.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/utils/timestamp-tag.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/utils/seq-ts-header.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/utils/broadcom-egress-queue.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/utils/bufferlog-tag.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/utils/custom-header.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/utils/custom-priority-tag.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/utils/feedback-tag.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/utils/interface-tag.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/utils/int-header.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/utils/rdma-tag.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/network/utils/unsched-tag.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/build/include/ns3/network-module.h"
    )
endif()

