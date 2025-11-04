#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "ryml::ryml" for configuration "Debug"
set_property(TARGET ryml::ryml APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(ryml::ryml PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libryml.a"
  )

list(APPEND _cmake_import_check_targets ryml::ryml )
list(APPEND _cmake_import_check_files_for_ryml::ryml "${_IMPORT_PREFIX}/lib/libryml.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
