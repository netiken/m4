# Install script for directory: /data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi

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
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libns3.39-wifi-optimized.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libns3.39-wifi-optimized.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libns3.39-wifi-optimized.so"
         RPATH "/usr/local/lib:$ORIGIN/:$ORIGIN/../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/data1/lichenni/m4/testbedResults/backends/UNISON/build/lib/libns3.39-wifi-optimized.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libns3.39-wifi-optimized.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libns3.39-wifi-optimized.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libns3.39-wifi-optimized.so"
         OLD_RPATH "/data1/lichenni/m4/testbedResults/backends/UNISON/build/lib:"
         NEW_RPATH "/usr/local/lib:$ORIGIN/:$ORIGIN/../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libns3.39-wifi-optimized.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ns3" TYPE FILE FILES
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/helper/athstats-helper.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/helper/spectrum-wifi-helper.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/helper/wifi-helper.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/helper/wifi-mac-helper.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/helper/wifi-radio-energy-model-helper.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/helper/yans-wifi-helper.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/adhoc-wifi-mac.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/ampdu-subframe-header.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/ampdu-tag.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/amsdu-subframe-header.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/ap-wifi-mac.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/block-ack-agreement.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/block-ack-manager.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/block-ack-type.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/block-ack-window.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/capability-information.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/channel-access-manager.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/ctrl-headers.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/edca-parameter-set.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/eht/default-emlsr-manager.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/eht/eht-capabilities.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/eht/eht-configuration.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/eht/eht-frame-exchange-manager.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/eht/eht-operation.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/eht/tid-to-link-mapping-element.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/eht/eht-phy.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/eht/eht-ppdu.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/eht/emlsr-manager.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/eht/multi-link-element.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/error-rate-model.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/extended-capabilities.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/fcfs-wifi-queue-scheduler.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/frame-capture-model.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/frame-exchange-manager.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/he/constant-obss-pd-algorithm.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/he/he-capabilities.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/he/he-configuration.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/he/he-frame-exchange-manager.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/he/he-operation.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/he/he-phy.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/he/he-ppdu.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/he/he-ru.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/he/mu-edca-parameter-set.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/he/mu-snr-tag.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/he/multi-user-scheduler.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/he/obss-pd-algorithm.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/he/rr-multi-user-scheduler.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/ht/ht-capabilities.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/ht/ht-configuration.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/ht/ht-frame-exchange-manager.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/ht/ht-operation.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/ht/ht-phy.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/ht/ht-ppdu.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/interference-helper.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/mac-rx-middle.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/mac-tx-middle.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/mgt-headers.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/mpdu-aggregator.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/msdu-aggregator.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/nist-error-rate-model.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/non-ht/dsss-error-rate-model.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/non-ht/dsss-parameter-set.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/non-ht/dsss-phy.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/non-ht/dsss-ppdu.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/non-ht/erp-information.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/non-ht/erp-ofdm-phy.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/non-ht/erp-ofdm-ppdu.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/non-ht/ofdm-phy.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/non-ht/ofdm-ppdu.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/non-inheritance.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/originator-block-ack-agreement.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/phy-entity.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/preamble-detection-model.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/qos-frame-exchange-manager.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/qos-txop.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/qos-utils.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/rate-control/aarf-wifi-manager.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/rate-control/aarfcd-wifi-manager.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/rate-control/amrr-wifi-manager.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/rate-control/aparf-wifi-manager.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/rate-control/arf-wifi-manager.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/rate-control/cara-wifi-manager.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/rate-control/constant-rate-wifi-manager.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/rate-control/ideal-wifi-manager.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/rate-control/minstrel-ht-wifi-manager.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/rate-control/minstrel-wifi-manager.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/rate-control/onoe-wifi-manager.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/rate-control/parf-wifi-manager.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/rate-control/rraa-wifi-manager.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/rate-control/rrpaa-wifi-manager.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/rate-control/thompson-sampling-wifi-manager.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/recipient-block-ack-agreement.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/reduced-neighbor-report.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/reference/error-rate-tables.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/simple-frame-capture-model.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/snr-tag.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/spectrum-wifi-phy.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/ssid.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/sta-wifi-mac.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/status-code.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/supported-rates.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/table-based-error-rate-model.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/threshold-preamble-detection-model.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/txop.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/vht/vht-capabilities.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/vht/vht-configuration.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/vht/vht-frame-exchange-manager.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/vht/vht-operation.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/vht/vht-phy.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/vht/vht-ppdu.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/wifi-ack-manager.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/wifi-acknowledgment.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/wifi-assoc-manager.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/wifi-bandwidth-filter.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/wifi-default-ack-manager.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/wifi-default-assoc-manager.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/wifi-default-protection-manager.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/wifi-information-element.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/wifi-mac-header.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/wifi-mac-queue-container.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/wifi-mac-queue-elem.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/wifi-mac-queue-scheduler-impl.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/wifi-mac-queue-scheduler.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/wifi-mac-queue.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/wifi-mac-trailer.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/wifi-mac.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/wifi-mgt-header.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/wifi-mode.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/wifi-mpdu-type.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/wifi-mpdu.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/wifi-net-device.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/wifi-phy-band.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/wifi-phy-common.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/wifi-phy-listener.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/wifi-phy-operating-channel.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/wifi-phy-state-helper.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/wifi-phy-state.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/wifi-phy.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/wifi-ppdu.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/wifi-protection-manager.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/wifi-protection.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/wifi-psdu.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/wifi-radio-energy-model.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/wifi-remote-station-info.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/wifi-remote-station-manager.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/wifi-spectrum-phy-interface.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/wifi-spectrum-signal-parameters.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/wifi-standards.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/wifi-tx-current-model.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/wifi-tx-parameters.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/wifi-tx-timer.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/wifi-tx-vector.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/wifi-utils.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/yans-error-rate-model.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/yans-wifi-channel.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/wifi/model/yans-wifi-phy.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/build/include/ns3/wifi-module.h"
    )
endif()

