# Install script for directory: /data1/lichenni/m4/testbedResults/backends/UNISON/src/core

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
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libns3.39-core-optimized.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libns3.39-core-optimized.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libns3.39-core-optimized.so"
         RPATH "/usr/local/lib:$ORIGIN/:$ORIGIN/../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/data1/lichenni/m4/testbedResults/backends/UNISON/build/lib/libns3.39-core-optimized.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libns3.39-core-optimized.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libns3.39-core-optimized.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libns3.39-core-optimized.so"
         OLD_RPATH "/data1/lichenni/m4/testbedResults/backends/UNISON/build/lib:"
         NEW_RPATH "/usr/local/lib:$ORIGIN/:$ORIGIN/../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libns3.39-core-optimized.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ns3" TYPE FILE FILES
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/int64x64-128.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/helper/csv-reader.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/helper/event-garbage-collector.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/helper/random-variable-stream-helper.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/abort.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/ascii-file.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/ascii-test.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/assert.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/atomic-counter.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/attribute-accessor-helper.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/attribute-construction-list.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/attribute-container.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/attribute-helper.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/attribute.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/boolean.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/breakpoint.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/build-profile.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/calendar-scheduler.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/callback.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/command-line.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/config.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/default-deleter.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/default-simulator-impl.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/deprecated.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/des-metrics.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/double.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/enum.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/event-id.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/event-impl.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/fatal-error.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/fatal-impl.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/fd-reader.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/environment-variable.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/global-value.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/hash-fnv.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/hash-function.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/hash-murmur3.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/hash.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/heap-scheduler.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/int-to-type.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/int64x64-double.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/int64x64.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/integer.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/length.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/list-scheduler.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/log-macros-disabled.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/log-macros-enabled.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/log.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/make-event.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/map-scheduler.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/math.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/names.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/node-printer.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/nstime.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/object-base.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/object-factory.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/object-map.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/object-ptr-container.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/object-vector.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/object.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/pair.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/pointer.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/priority-queue-scheduler.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/ptr.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/random-variable-stream.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/rng-seed-manager.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/rng-stream.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/scheduler.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/show-progress.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/simple-ref-count.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/simulation-singleton.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/simulator-impl.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/simulator.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/singleton.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/string.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/synchronizer.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/system-path.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/system-wall-clock-ms.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/system-wall-clock-timestamp.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/test.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/time-printer.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/timer-impl.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/timer.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/trace-source-accessor.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/traced-callback.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/traced-value.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/trickle-timer.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/tuple.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/type-id.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/type-name.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/type-traits.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/uinteger.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/unused.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/valgrind.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/vector.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/warnings.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/watchdog.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/realtime-simulator-impl.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/wall-clock-synchronizer.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/val-array.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/matrix-array.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/src/core/model/random-variable.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/build/include/ns3/config-store-config.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/build/include/ns3/core-config.h"
    "/data1/lichenni/m4/testbedResults/backends/UNISON/build/include/ns3/core-module.h"
    )
endif()

