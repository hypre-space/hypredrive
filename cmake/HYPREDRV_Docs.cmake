# Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

# Documentation support
# This module provides Doxygen and Sphinx documentation generation

if(HYPREDRV_ENABLE_DOCS)
    # Find Doxygen
    find_package(Doxygen QUIET)

    if(DOXYGEN_FOUND)
        message(STATUS "Found Doxygen: ${DOXYGEN_EXECUTABLE}")

        # Set variables for Doxyfile.in configuration
        set(PACKAGE_NAME ${PROJECT_NAME})
        set(VERSION ${PROJECT_VERSION})
        set(top_srcdir ${CMAKE_SOURCE_DIR})
        # OUTPUT_DIRECTORY should be relative to build directory (where doxygen runs)
        set(OUTPUT_DIRECTORY "docs")

        # Find Graphviz/dot for HAVE_DOT
        find_program(HAVE_DOT_EXECUTABLE NAMES dot)
        if(HAVE_DOT_EXECUTABLE)
            set(HAVE_DOT "YES")
        else()
            set(HAVE_DOT "NO")
        endif()

        # Set output format flags (matching autotools defaults)
        set(GENERATE_HTML "YES")
        set(GENERATE_HTMLHELP "NO")
        set(GENERATE_CHI "NO")
        set(GENERATE_LATEX "YES")
        set(GENERATE_RTF "YES")
        set(GENERATE_MAN "YES")
        set(GENERATE_XML "YES")

        # Configure Doxyfile from template
        set(DOXYFILE_IN ${CMAKE_SOURCE_DIR}/Doxyfile.in)
        set(DOXYFILE_OUT ${CMAKE_BINARY_DIR}/Doxyfile)

        configure_file(${DOXYFILE_IN} ${DOXYFILE_OUT} @ONLY)

        # Post-process Doxyfile to fix OUTPUT_DIRECTORY (in case @top_srcdir@/docs wasn't replaced)
        # Doxygen runs from CMAKE_BINARY_DIR, so OUTPUT_DIRECTORY should be relative
        # file(READ ${DOXYFILE_OUT} DOXYFILE_CONTENT)
        # string(REPLACE "@top_srcdir@/docs" "docs" DOXYFILE_CONTENT "${DOXYFILE_CONTENT}")
        # string(REPLACE "${CMAKE_SOURCE_DIR}/docs" "docs" DOXYFILE_CONTENT "${DOXYFILE_CONTENT}")
        # file(WRITE ${DOXYFILE_OUT} "${DOXYFILE_CONTENT}")

        # Doxygen output directory
        set(DOXYGEN_OUTPUT_DIR ${CMAKE_BINARY_DIR}/docs)

        # Add custom target to generate Doxygen documentation
        add_custom_target(doxygen-doc
            COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_OUT}
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Generating Doxygen documentation"
            VERBATIM
        )

        # Copy PDF if generated (PDF is only built if LaTeX is available)
        # Doxygen generates refman.pdf in the latex directory, not hypredrive.pdf
        set(PDF_SOURCE "${DOXYGEN_OUTPUT_DIR}/latex/refman.pdf")
        set(PDF_DEST "${CMAKE_BINARY_DIR}/docs/devman-hypredrive.pdf")
        set(LATEX_DIR "${DOXYGEN_OUTPUT_DIR}/latex")
        # Create inline script to build and copy PDF
        file(WRITE ${CMAKE_BINARY_DIR}/copy_pdf.cmake
            "if(EXISTS \"${PDF_SOURCE}\")\n"
            "  execute_process(\n"
            "    COMMAND ${CMAKE_COMMAND} -E copy_if_different \"${PDF_SOURCE}\" \"${PDF_DEST}\"\n"
            "    RESULT_VARIABLE COPY_RESULT\n"
            "  )\n"
            "  if(COPY_RESULT EQUAL 0)\n"
            "    message(STATUS \"Copied Doxygen PDF to ${PDF_DEST}\")\n"
            "  else()\n"
            "    message(WARNING \"Failed to copy Doxygen PDF\")\n"
            "  endif()\n"
            "elseif(EXISTS \"${LATEX_DIR}/Makefile\")\n"
            "  message(STATUS \"Building LaTeX PDF from Doxygen-generated sources...\")\n"
            "  execute_process(\n"
            "    COMMAND ${CMAKE_MAKE_PROGRAM}\n"
            "    WORKING_DIRECTORY \"${LATEX_DIR}\"\n"
            "    RESULT_VARIABLE MAKE_RESULT\n"
            "    OUTPUT_VARIABLE MAKE_OUTPUT\n"
            "    ERROR_VARIABLE MAKE_ERROR\n"
            "  )\n"
            "  if(NOT MAKE_RESULT EQUAL 0)\n"
            "    message(WARNING \"LaTeX build failed with exit code ${MAKE_RESULT}\")\n"
            "    message(WARNING \"Output: ${MAKE_OUTPUT}\")\n"
            "    message(WARNING \"Error: ${MAKE_ERROR}\")\n"
            "  endif()\n"
            "  if(EXISTS \"${PDF_SOURCE}\")\n"
            "    execute_process(\n"
            "      COMMAND ${CMAKE_COMMAND} -E copy_if_different \"${PDF_SOURCE}\" \"${PDF_DEST}\"\n"
            "      RESULT_VARIABLE COPY_RESULT\n"
            "    )\n"
            "    if(COPY_RESULT EQUAL 0)\n"
            "      message(STATUS \"Built and copied Doxygen PDF to ${PDF_DEST}\")\n"
            "    else()\n"
            "      message(WARNING \"Failed to copy Doxygen PDF after building\")\n"
            "    endif()\n"
            "  else()\n"
            "    message(STATUS \"Note: LaTeX PDF build failed (check ${LATEX_DIR} for errors)\")\n"
            "  endif()\n"
            "else()\n"
            "  message(STATUS \"Note: LaTeX PDF not available (LaTeX sources not generated)\")\n"
            "endif()\n"
        )
        add_custom_command(TARGET doxygen-doc POST_BUILD
            COMMAND ${CMAKE_COMMAND} -P ${CMAKE_BINARY_DIR}/copy_pdf.cmake
            COMMENT "Copying Doxygen PDF to devman-hypredrive.pdf (if available)"
        )

        message(STATUS "Doxygen documentation target 'doxygen-doc' is available")
    else()
        message(WARNING "Doxygen not found. Install doxygen and graphviz to generate documentation.")
        message(WARNING "  On Ubuntu/Debian: sudo apt-get install doxygen graphviz")
        message(WARNING "  On macOS: brew install doxygen graphviz")
    endif()

    # Find Sphinx (optional, for user manual)
    find_program(SPHINX_BUILD_EXECUTABLE NAMES sphinx-build)

    if(SPHINX_BUILD_EXECUTABLE)
        message(STATUS "Found Sphinx: ${SPHINX_BUILD_EXECUTABLE}")

        # Check if docs/Makefile exists (following the pattern from docs/Makefile)
        set(SPHINX_SOURCE_DIR ${CMAKE_SOURCE_DIR}/docs/usrman-src)
        set(SPHINX_BUILD_DIR ${CMAKE_BINARY_DIR}/docs/usrman-build)

        if(EXISTS ${SPHINX_SOURCE_DIR})
            # Add custom target to build Sphinx documentation
            add_custom_target(sphinx-doc
                COMMAND ${SPHINX_BUILD_EXECUTABLE}
                    -b html
                    ${SPHINX_SOURCE_DIR}
                    ${SPHINX_BUILD_DIR}/html
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                COMMENT "Building Sphinx user manual"
                VERBATIM
            )

            # Add target to build PDF (requires latex)
            add_custom_target(sphinx-latexpdf
                COMMAND ${SPHINX_BUILD_EXECUTABLE}
                    -b latexpdf
                    ${SPHINX_SOURCE_DIR}
                    ${SPHINX_BUILD_DIR}/latex
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                COMMENT "Building Sphinx user manual PDF"
                VERBATIM
            )

            # Copy PDF if generated
            add_custom_command(TARGET sphinx-latexpdf POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${SPHINX_BUILD_DIR}/latex/hypredrive.pdf
                    ${CMAKE_BINARY_DIR}/docs/usrman-hypredrive.pdf
                COMMENT "Copying Sphinx PDF to usrman-hypredrive.pdf"
            )

            message(STATUS "Sphinx documentation targets 'sphinx-doc' and 'sphinx-latexpdf' are available")
        else()
            message(WARNING "Sphinx source directory not found: ${SPHINX_SOURCE_DIR}")
        endif()

        # Combined docs target (like Makefile: docs: doxygen-doc)
        if(DOXYGEN_FOUND AND EXISTS ${SPHINX_SOURCE_DIR})
            # Use docs/Makefile if it exists (following the pattern from docs/Makefile)
            set(DOCS_MAKEFILE ${CMAKE_SOURCE_DIR}/docs/Makefile)
            if(EXISTS ${DOCS_MAKEFILE})
                # Use the Makefile approach (like autotools: docs: doxygen-doc; cd docs && make latexpdf)
                find_program(MAKE_EXECUTABLE NAMES make gmake)
                if(MAKE_EXECUTABLE)
                    add_custom_target(docs
                        COMMAND ${CMAKE_COMMAND} -E echo "Built developer's manual documentation (Doxygen)"
                        COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_SOURCE_DIR}/docs
                            ${MAKE_EXECUTABLE} latexpdf
                        COMMAND ${CMAKE_COMMAND} -E echo "Built user's manual documentation (Sphinx)"
                        DEPENDS doxygen-doc
                        COMMENT "Building all documentation (Doxygen + Sphinx)"
                        VERBATIM
                    )
                else()
                    # Fallback to direct sphinx-build if make is not available
                    add_custom_target(docs
                        COMMAND ${CMAKE_COMMAND} -E echo "Built developer's manual documentation (Doxygen)"
                        COMMAND ${CMAKE_COMMAND} -E echo "Building user's manual documentation (Sphinx)"
                        DEPENDS doxygen-doc sphinx-latexpdf
                        COMMENT "Building all documentation (Doxygen + Sphinx)"
                        VERBATIM
                    )
                endif()
            else()
                # Fallback to direct sphinx-build if Makefile doesn't exist
                add_custom_target(docs
                    COMMAND ${CMAKE_COMMAND} -E echo "Built developer's manual documentation (Doxygen)"
                    COMMAND ${CMAKE_COMMAND} -E echo "Building user's manual documentation (Sphinx)"
                    DEPENDS doxygen-doc sphinx-latexpdf
                    COMMENT "Building all documentation (Doxygen + Sphinx)"
                    VERBATIM
                )
            endif()

            message(STATUS "Combined documentation target 'docs' is available")
        endif()
    else()
        message(WARNING "Sphinx not found. Install sphinx to generate user manual.")
        message(WARNING "  On Ubuntu/Debian: sudo apt-get install python3-sphinx")
        message(WARNING "  On macOS: pip install sphinx")
    endif()
else()
    message(STATUS "Documentation generation is disabled (HYPREDRV_ENABLE_DOCS=OFF)")
endif()
