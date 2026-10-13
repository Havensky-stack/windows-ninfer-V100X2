# CMake's automatic /showIncludes detection can decode UTF-8 Chinese MSVC output
# using the ANSI code page. A mismatched Ninja prefix silently loses header deps.
# Probe with explicit UTF-8 decoding; VSLANG alone cannot select an absent language pack.
set(_include_probe "${CMAKE_BINARY_DIR}/CMakeFiles/ninfer-showincludes")
file(MAKE_DIRECTORY "${_include_probe}")
file(WRITE "${_include_probe}/probe.h" "#pragma once\n")
file(WRITE "${_include_probe}/probe.c" "#include \"probe.h\"\nint main(void) { return 0; }\n")
execute_process(
  COMMAND "${CMAKE_C_COMPILER}" /nologo /showIncludes /c probe.c /Foprobe.obj
  WORKING_DIRECTORY "${_include_probe}"
  OUTPUT_VARIABLE _include_output
  ERROR_VARIABLE _include_error
  RESULT_VARIABLE _include_status
  ENCODING UTF-8)
if(NOT _include_status EQUAL 0)
  message(FATAL_ERROR "MSVC header dependency probe failed: ${_include_error}")
endif()
if(NOT _include_output MATCHES "\n([^\r\n]+: +)[A-Za-z]:[/\\\\][^\r\n]*probe[.]h")
  message(FATAL_ERROR "Cannot parse MSVC /showIncludes output: ${_include_output}")
endif()
set(CMAKE_C_CL_SHOWINCLUDES_PREFIX "${CMAKE_MATCH_1}")
set(CMAKE_CXX_CL_SHOWINCLUDES_PREFIX "${CMAKE_MATCH_1}")
set(CMAKE_CL_SHOWINCLUDES_PREFIX "${CMAKE_MATCH_1}")
message(STATUS "MSVC header dependency prefix: ${CMAKE_CXX_CL_SHOWINCLUDES_PREFIX}")
