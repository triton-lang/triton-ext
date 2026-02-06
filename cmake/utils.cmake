
# Function to get the current Triton git hash
function(get_triton_git_hash triton_source_dir result_var)
    # Try to get git hash from the triton source directory
    execute_process(
        COMMAND git rev-parse HEAD
        WORKING_DIRECTORY ${triton_source_dir}
        OUTPUT_VARIABLE _git_hash
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE _git_result
    )
    if(_git_result EQUAL 0)
        set(${result_var} ${_git_hash} PARENT_SCOPE)
    else()
        set(${result_var} "" PARENT_SCOPE)
    endif()
endfunction()

# Function to check if triton hash matches
function(triton_ext_check_triton_hash result_var)
    set(triton_source_dir "${CMAKE_SOURCE_DIR}")
    set(hash_file_path "${CMAKE_CURRENT_SOURCE_DIR}/triton-hash.txt")

    # Read the expected hash from file
    if(EXISTS ${hash_file_path})
        file(READ ${hash_file_path} _expected_hash)
        string(STRIP ${_expected_hash} _expected_hash)

        # Get current triton git hash
        get_triton_git_hash(${triton_source_dir} _current_hash)

        if(_current_hash STREQUAL "")
            message(WARNING "Could not retrieve Triton git hash from ${triton_source_dir}. Hash verification skipped.")
            set(${result_var} TRUE PARENT_SCOPE)
        elseif(_current_hash STREQUAL _expected_hash)
            set(${result_var} TRUE PARENT_SCOPE)
        else()
            message(WARNING "Triton hash mismatch: expected ${_expected_hash}, got ${_current_hash}")
            set(${result_var} FALSE PARENT_SCOPE)
        endif()
    else()
        message(WARNING "Triton hash file not found: ${hash_file_path}")
        set(${result_var} TRUE PARENT_SCOPE)
    endif()
endfunction()

# Function to check if a Triton extension should be built
# Usage: triton_ext_should_build_extension(<triton_ext_dir> <result_var>)
#   triton_ext_dir: Path to the Triton extension directory
#   result_var: Variable to store the result
function(triton_ext_should_build_extension triton_ext_dir result_var)
    set(config_file "${triton_ext_dir}/triton-ext.conf")
    # Default to FALSE
    set(${result_var} FALSE PARENT_SCOPE)

    # Try to read from config file first
    if(EXISTS ${config_file})
        file(READ ${config_file} _config_content)
        string(STRIP "${_config_content}" _config_content)
        # Parse the config file (format: name;status or just name)
        # Config file must be in CMake list format (semicolon-separated)
        if(_config_content)
            set(_config_list ${_config_content})
            list(LENGTH _config_list _num_parts)
            if(_num_parts GREATER_EQUAL 1)
                list(GET _config_list 0 _ext_name)
                string(STRIP "${_ext_name}" _ext_name)
            else()
                set(_ext_name ${_config_content})
                string(STRIP "${_ext_name}" _ext_name)
            endif()
            # Check if _ext_name is in TRITON_EXT_NAMES
            list(LENGTH "${TRITON_EXT_NAMES}" _size)
            if(NOT _size EQUAL 0)
                list(FIND "${TRITON_EXT_NAMES}" "${_ext_name}" _index)
                if(NOT _index EQUAL -1)
                    # If _ext_name is in TRITON_EXT_NAMES, set result to TRUE
                    set(${result_var} TRUE PARENT_SCOPE)
                endif()
            else()
                # If TRITON_EXT_NAMES is empty, default to TRUE
                set(${result_var} TRUE PARENT_SCOPE)
            endif()
        endif()
    endif()
endfunction()

# Macro to set up a Triton extension project
# Usage: triton_ext_pass_setup(<ext_class>)
#   ext_class: Extension class name (required)
macro(triton_ext_pass_setup ext_class)
    # Read extension name, status, and hash from local triton-ext.conf file
    set(TRITON_EXT_CONFIG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/triton-ext.conf")
    set(_ext_name "")
    set(_status "experimental")
    set(_hash "")

    if(EXISTS ${TRITON_EXT_CONFIG_FILE})
        file(READ ${TRITON_EXT_CONFIG_FILE} _config_content)
        string(STRIP "${_config_content}" _config_content)

        if(_config_content)
            # Config file is in CMake list format (semicolon-separated): name;status;hash
            set(_config_list ${_config_content})
            list(LENGTH _config_list _num_parts)

            if(_num_parts GREATER_EQUAL 1)
                list(GET _config_list 0 _ext_name)
                string(STRIP "${_ext_name}" _ext_name)
            endif()

            if(_num_parts GREATER_EQUAL 2)
                list(GET _config_list 1 _status)
                string(STRIP "${_status}" _status)
            endif()

            if(_num_parts GREATER_EQUAL 3)
                list(GET _config_list 2 _hash)
                string(STRIP "${_hash}" _hash)
            endif()
        endif()
    endif()

    if(NOT _ext_name)
        message(FATAL_ERROR "triton_ext.conf file not found or empty in ${CMAKE_CURRENT_SOURCE_DIR}")
    endif()

    project(${_ext_name})
    set(TRITON_EXT_NAME ${PROJECT_NAME})
    set(TRITON_EXT_CLASS ${ext_class})
    set(TRITON_EXT_STATUS ${_status})
    if(_hash)
        set(TRITON_EXT_HASH ${_hash})
    endif()
endmacro()

# Function to safely add a subdirectory without failing the build
# Usage: safe_add_subdirectory(<source_dir> [binary_dir] [message])
function(safe_add_subdirectory source_dir)
    # Get optional binary_dir argument
    if(ARGC GREATER 1)
        set(binary_dir ${ARGV1})
    else()
        # Default binary dir based on source dir name
        get_filename_component(dir_name ${source_dir} NAME)
        set(binary_dir "${CMAKE_CURRENT_BINARY_DIR}/${dir_name}")
    endif()

    # Check if source directory exists
    if(NOT IS_DIRECTORY "${source_dir}")
        if(ARGC GREATER 2)
            message(STATUS "${ARGV2} - Source directory does not exist: ${source_dir}")
        else()
            message(STATUS "Skipping subdirectory (does not exist): ${source_dir}")
        endif()
        return()
    endif()

    # Check if CMakeLists.txt exists in source directory
    if(NOT EXISTS "${source_dir}/CMakeLists.txt")
        if(ARGC GREATER 2)
            message(STATUS "${ARGV2} - CMakeLists.txt not found in: ${source_dir}")
        else()
            message(STATUS "Skipping subdirectory (no CMakeLists.txt): ${source_dir}")
        endif()
        return()
    endif()

    # Test if the subdirectory can be configured without errors
    get_filename_component(dir_name ${source_dir} NAME)
    set(test_binary_dir "${CMAKE_CURRENT_BINARY_DIR}/_safe_test_${dir_name}")

    # Collect all cache variables to pass to the test configuration
    get_cmake_property(cache_vars CACHE_VARIABLES)
    set(cache_args "")
    foreach(var ${cache_vars})
        # Skip only truly internal CMake variables (starting with __)
        if(var MATCHES "^__")
            continue()
        endif()

        # Get the variable type and value
        get_property(var_type CACHE ${var} PROPERTY TYPE)
        if(NOT var_type)
            continue()
        endif()

        get_property(var_value CACHE ${var} PROPERTY VALUE)
        if("${var_value}" STREQUAL "")
            # Skip empty values
            continue()
        endif()

        # Filter out variables with illegal characters that could break command parsing
        # Check for square brackets [] or parentheses () using string FIND
        string(FIND "${var_value}" "[" has_open_bracket)
        string(FIND "${var_value}" "]" has_close_bracket)
        string(FIND "${var_value}" "(" has_open_paren)
        string(FIND "${var_value}" ")" has_close_paren)
        if(NOT has_open_bracket EQUAL -1 OR NOT has_close_bracket EQUAL -1 OR
           NOT has_open_paren EQUAL -1 OR NOT has_close_paren EQUAL -1)
            continue()
        endif()

        # Build the -D argument
        # For boolean values, use ON/OFF
        if(var_type STREQUAL "BOOL")
            if(var_value)
                string(APPEND cache_args " -D${var}=ON")
            else()
                string(APPEND cache_args " -D${var}=OFF")
            endif()
        else()
            # For other types, pass the value as-is
            # CMake will handle escaping when passed as a list item
            string(APPEND cache_args " -D${var}=${var_value}")
        endif()
    endforeach()

    # Build the cmake command with all cache variables
    set(cmake_test_cmd "${CMAKE_COMMAND} ${cache_args} ${source_dir} -B ${test_binary_dir}")

    # Try to configure the subdirectory in a test build to check for errors
    execute_process(
        COMMAND ${cmake_test_cmd}
        RESULT_VARIABLE config_result
        OUTPUT_VARIABLE config_output
        ERROR_VARIABLE config_error
        OUTPUT_QUIET ERROR_QUIET
    )

    message(STATUS ${cmake_test_cmd})

    # Clean up test directory
    file(REMOVE_RECURSE "${test_binary_dir}")

    # If configuration test passed, add the subdirectory
    if(config_result EQUAL 0)
        add_subdirectory("${source_dir}" "${binary_dir}")
        if(ARGC GREATER 2)
            message(STATUS "${ARGV2} - Successfully added: ${source_dir}")
        endif()
    else()
        if(ARGC GREATER 2)
            message(WARNING "${ARGV2} - Failed to configure subdirectory: ${source_dir}")
            message(STATUS "Build will continue without this subdirectory")
        else()
            message(WARNING "Failed to configure subdirectory: ${source_dir}")
            message(WARNING "Error: ${config_error}")
            message(WARNING "OUTPUT: ${config_output}")
            message(STATUS "Build will continue without this subdirectory")
        endif()
    endif()
endfunction()
