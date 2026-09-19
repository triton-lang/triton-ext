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

add_executable(test_kernelabi test/test_kernelabi.cpp)
target_link_libraries(test_kernelabi PRIVATE agpu_msl agpu_core)
target_compile_options(test_kernelabi PRIVATE ${AGPU_WARNINGS})
add_test(NAME kernelabi COMMAND test_kernelabi)

add_executable(test_rebindplan test/test_rebindplan.cpp)
target_include_directories(test_rebindplan PRIVATE test)
target_link_libraries(test_rebindplan PRIVATE agpu_msl agpu_core)
target_compile_options(test_rebindplan PRIVATE ${AGPU_WARNINGS})
add_test(NAME rebindplan COMMAND test_rebindplan)
