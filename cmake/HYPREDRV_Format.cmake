# Formatting target
find_program(CLANG_FORMAT "clang-format")
if(NOT CLANG_FORMAT)
    message(STATUS "clang-format not found, formatting targets will not be available")
else()
    add_custom_target(format
        COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_SOURCE_DIR}
            find . -type f -name "*.c" -not -path "./build/*" -exec ${CLANG_FORMAT} -i {} +
        COMMENT "Running clang-format..."
    )
endif()
