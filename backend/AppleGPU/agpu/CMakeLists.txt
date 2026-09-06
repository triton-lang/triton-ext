cmake_minimum_required(VERSION 3.20)
project(agpu CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(AGPU_WARNINGS -Wall -Wextra -Wpedantic)

add_library(agpu_core INTERFACE)
target_include_directories(agpu_core INTERFACE include)

add_library(agpu_msl STATIC src/Printer.cpp)
target_include_directories(agpu_msl PUBLIC include)
target_compile_options(agpu_msl PRIVATE ${AGPU_WARNINGS})

enable_testing()

add_executable(emit_probe test/emit_probe.cpp)
target_link_libraries(emit_probe PRIVATE agpu_msl agpu_core)
target_compile_options(emit_probe PRIVATE ${AGPU_WARNINGS})

add_test(NAME metal_compiles
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test/metal_compiles.sh
          $<TARGET_FILE:emit_probe>
          ${CMAKE_CURRENT_BINARY_DIR}/emit_probe)
