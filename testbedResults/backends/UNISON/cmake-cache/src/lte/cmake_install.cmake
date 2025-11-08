# Install script for directory: /data1/lichenni/m4/testbedResults/backends/UNISON/src/lte

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
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libns3.39-lte-optimized.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libns3.39-lte-optimized.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libns3.39-lte-optimized.so"
         RPATH "/usr/local/lib:$ORIGIN/:$ORIGIN/../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/data1/lichenni/m4/testbedResults/backends/UNISON/build/lib/libns3.39-lte-optimized.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libns3.39-lte-optimized.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libns3.39-lte-optimized.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libns3.39-lte-optimized.so"
         OLD_RPATH "/data1/lichenni/m4/testbedResults/backends/UNISON/build/lib:"
         NEW_RPATH "/usr/local/lib:$ORIGIN/:$ORIGIN/../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libns3.39-lte-optimized.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ns3" TYPE FILE FILES
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/helper/emu-epc-helper.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/helper/cc-helper.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/helper/epc-helper.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/helper/lte-global-pathloss-database.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/helper/lte-helper.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/helper/lte-hex-grid-enb-topology-helper.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/helper/lte-stats-calculator.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/helper/mac-stats-calculator.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/helper/no-backhaul-epc-helper.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/helper/phy-rx-stats-calculator.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/helper/phy-stats-calculator.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/helper/phy-tx-stats-calculator.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/helper/point-to-point-epc-helper.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/helper/radio-bearer-stats-calculator.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/helper/radio-bearer-stats-connector.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/helper/radio-environment-map-helper.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/a2-a4-rsrq-handover-algorithm.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/a3-rsrp-handover-algorithm.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/component-carrier-enb.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/component-carrier-ue.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/component-carrier.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/cqa-ff-mac-scheduler.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/epc-enb-application.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/epc-enb-s1-sap.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/epc-gtpc-header.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/epc-gtpu-header.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/epc-mme-application.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/epc-pgw-application.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/epc-s11-sap.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/epc-s1ap-sap.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/epc-sgw-application.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/epc-tft-classifier.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/epc-tft.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/epc-ue-nas.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/epc-x2-header.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/epc-x2-sap.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/epc-x2.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/eps-bearer-tag.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/eps-bearer.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/fdbet-ff-mac-scheduler.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/fdmt-ff-mac-scheduler.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/fdtbfq-ff-mac-scheduler.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/ff-mac-common.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/ff-mac-csched-sap.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/ff-mac-sched-sap.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/ff-mac-scheduler.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-amc.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-anr-sap.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-anr.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-as-sap.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-asn1-header.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-ccm-mac-sap.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-ccm-rrc-sap.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-chunk-processor.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-common.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-control-messages.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-enb-cmac-sap.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-enb-component-carrier-manager.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-enb-cphy-sap.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-enb-mac.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-enb-net-device.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-enb-phy-sap.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-enb-phy.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-enb-rrc.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-ffr-algorithm.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-ffr-distributed-algorithm.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-ffr-enhanced-algorithm.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-ffr-rrc-sap.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-ffr-sap.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-ffr-soft-algorithm.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-fr-hard-algorithm.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-fr-no-op-algorithm.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-fr-soft-algorithm.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-fr-strict-algorithm.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-handover-algorithm.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-handover-management-sap.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-harq-phy.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-interference.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-mac-sap.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-mi-error-model.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-net-device.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-pdcp-header.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-pdcp-sap.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-pdcp-tag.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-pdcp.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-phy-tag.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-phy.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-radio-bearer-info.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-radio-bearer-tag.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-rlc-am-header.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-rlc-am.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-rlc-header.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-rlc-sap.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-rlc-sdu-status-tag.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-rlc-sequence-number.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-rlc-tag.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-rlc-tm.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-rlc-um.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-rlc.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-rrc-header.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-rrc-protocol-ideal.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-rrc-protocol-real.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-rrc-sap.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-spectrum-phy.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-spectrum-signal-parameters.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-spectrum-value-helper.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-ue-ccm-rrc-sap.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-ue-cmac-sap.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-ue-component-carrier-manager.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-ue-cphy-sap.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-ue-mac.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-ue-net-device.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-ue-phy-sap.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-ue-phy.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-ue-power-control.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-ue-rrc.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/lte-vendor-specific-parameters.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/no-op-component-carrier-manager.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/no-op-handover-algorithm.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/pf-ff-mac-scheduler.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/pss-ff-mac-scheduler.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/rem-spectrum-phy.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/rr-ff-mac-scheduler.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/simple-ue-component-carrier-manager.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/tdbet-ff-mac-scheduler.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/tdmt-ff-mac-scheduler.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/tdtbfq-ff-mac-scheduler.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/lte/model/tta-ff-mac-scheduler.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/build/include/ns3/lte-module.h"
    )
endif()

