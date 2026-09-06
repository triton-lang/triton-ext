cmake_minimum_required(VERSION 3.20)
project(agpu CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(AGPU_WARNINGS -Wall -Wextra -Wpedantic -Wno-missing-field-initializers)

add_library(agpu_core INTERFACE)
target_include_directories(agpu_core INTERFACE include)

add_library(agpu_msl STATIC src/Printer.cpp)
target_include_directories(agpu_msl PUBLIC include)
target_compile_options(agpu_msl PRIVATE ${AGPU_WARNINGS})

enable_testing()

include(ProcessorCount)
ProcessorCount(AGPU_NPROC)
if(NOT AGPU_NPROC EQUAL 0)
  set(CTEST_PARALLEL_LEVEL ${AGPU_NPROC} CACHE STRING "ctest jobs")
  set(ENV{CTEST_PARALLEL_LEVEL} ${AGPU_NPROC})
endif()

file(GLOB AGPU_TEST_SRCS CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/test/test_*.cpp")

foreach(src ${AGPU_TEST_SRCS})
  get_filename_component(name ${src} NAME_WE)
  add_executable(${name} ${src})
  target_link_libraries(${name} PRIVATE agpu_msl agpu_core)
  target_include_directories(${name} PRIVATE test)
  target_compile_options(${name} PRIVATE ${AGPU_WARNINGS})
  add_test(NAME ${name} COMMAND ${name})
endforeach()

target_compile_definitions(test_layering PRIVATE
  AGPU_INCLUDE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/include"
  AGPU_BRIDGE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../lib/TritonAppleGPUToMSL")
target_compile_definitions(test_gates PRIVATE
  AGPU_COMPILER_PY="${CMAKE_CURRENT_SOURCE_DIR}/../python/triton_apple_backend/compiler.py")
target_include_directories(test_optables PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../lib/TritonAppleGPUToMSL)

add_executable(emit_probe test/emit_probe.cpp)
target_link_libraries(emit_probe PRIVATE agpu_msl agpu_core)
target_compile_options(emit_probe PRIVATE ${AGPU_WARNINGS})

add_test(NAME metal_compiles
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test/metal_compiles.sh
          $<TARGET_FILE:emit_probe>
          ${CMAKE_CURRENT_BINARY_DIR}/emit_probe)
