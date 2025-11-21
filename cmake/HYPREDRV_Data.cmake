# Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

# Dataset download/extract helper targets (uses Zenodo record)

if(NOT HYPREDRV_ENABLE_DATA)
  return()
endif()

set(HYPREDRV_DATA_ZENODO_RECORD "17471036" CACHE STRING "Zenodo record id for datasets")
set(HYPREDRV_DATA_BASE_URL "https://zenodo.org/api/records/${HYPREDRV_DATA_ZENODO_RECORD}/files" CACHE STRING "Base URL for dataset files on Zenodo API")

set(HYPREDRV_DATASETS "ps3d10pt7;compflow6k;poromech2k" CACHE STRING "Datasets to fetch from Zenodo")

set(HYPREDRV_ARCHIVE_ps3d10pt7 "ps3d10pt7.tar.gz" CACHE STRING "Archive name for ps3d10pt7")
set(HYPREDRV_MD5_ps3d10pt7 "1352fa6350fcf6f1a3dc815414597b21" CACHE STRING "MD5 checksum for ps3d10pt7")

set(HYPREDRV_ARCHIVE_compflow6k "compflow6k.tar.gz" CACHE STRING "Archive name for compflow6k")
set(HYPREDRV_MD5_compflow6k "ae06d81605d04399ac5ac8205748424e" CACHE STRING "MD5 checksum for compflow6k")

set(HYPREDRV_ARCHIVE_poromech2k "poromech2k.tar.gz" CACHE STRING "Archive name for poromech2k")
set(HYPREDRV_MD5_poromech2k "7706b89a31b681d8cb76b0f790633c52" CACHE STRING "MD5 checksum for poromech2k")

set(_dl_dir "${CMAKE_BINARY_DIR}/downloads")
set(_stamp_dir "${CMAKE_BINARY_DIR}/stamps")
file(MAKE_DIRECTORY "${_dl_dir}" "${_stamp_dir}")

set(_dl_extract_script "${CMAKE_SOURCE_DIR}/scripts/download_and_extract.sh")
set(_data_dir "${CMAKE_SOURCE_DIR}/data")

set(_dataset_stamps)
foreach(dataset IN LISTS HYPREDRV_DATASETS)
    set(archive "${HYPREDRV_ARCHIVE_${dataset}}")
    set(md5     "${HYPREDRV_MD5_${dataset}}")
    set(url     "${HYPREDRV_DATA_BASE_URL}/${archive}/content")
    set(tarball "${_dl_dir}/${archive}")
    set(stamp   "${_stamp_dir}/data_${dataset}.stamp")
    set(dataset_dir "${_data_dir}/${dataset}")

    # Check if dataset directory already exists and is non-empty
    set(_dataset_exists FALSE)
    if(EXISTS "${dataset_dir}" AND IS_DIRECTORY "${dataset_dir}")
        file(GLOB _existing_files "${dataset_dir}/*")
        list(LENGTH _existing_files _file_count)
        if(_file_count GREATER 0)
            set(_dataset_exists TRUE)
            message(STATUS "Dataset ${dataset} already exists at ${dataset_dir}, skipping download")
            # Create stamp file to indicate dataset is available
            add_custom_command(OUTPUT "${stamp}"
                COMMAND ${CMAKE_COMMAND} -E touch "${stamp}"
                COMMENT "Dataset ${dataset} already present"
                VERBATIM
            )
        endif()
    endif()

    # Dataset doesn't exist or is empty, add download command
    if(NOT _dataset_exists)
        add_custom_command(OUTPUT "${stamp}"
          COMMAND ${CMAKE_COMMAND} -E echo "Fetching ${dataset} from ${url}"
          COMMAND bash "${_dl_extract_script}" "${url}" "${tarball}" "${_data_dir}" "${md5}"
          COMMAND ${CMAKE_COMMAND} -E touch "${stamp}"
          COMMENT "Download and extract dataset: ${dataset}"
          VERBATIM
        )
    endif()

    list(APPEND _dataset_stamps "${stamp}")
endforeach()

add_custom_target(data
    DEPENDS ${_dataset_stamps}
)

add_custom_target(data-clean
    COMMAND ${CMAKE_COMMAND} -E rm -f ${_dataset_stamps}
    COMMENT "Remove dataset fetch stamps (extracted files remain under data/)"
)
