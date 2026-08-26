# Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

# Dataset download/extract helper targets (uses Zenodo record)

if(NOT HYPREDRV_ENABLE_DATA)
  return()
endif()

set(HYPREDRV_DATA_ZENODO_RECORD "22116856" CACHE STRING "Zenodo record id for datasets")
set(HYPREDRV_DATA_BASE_URL "https://zenodo.org/api/records/${HYPREDRV_DATA_ZENODO_RECORD}/files" CACHE STRING "Base URL for dataset files on Zenodo API")

set(HYPREDRV_DATASETS "alcontact1k;cmpf1k;cmphyb1k;cmpres1k;cmpreshyb1k;compflow6k;hspporo1k;hydrofrac1k;immf1k;lcontact1k;lcontactbs1k;mhd2dldc1k;mhd3ddbdt1k;mpporo1k;mpporores1k;poromech2k;ps3d10pt7;rcmpobl1k;smef1k;sphyb1k;spporo1k;spporocf1k;spporoef1k;spporores1k;spres1k;spreshyb1k;tcmpf1k;tcmpres1k;tmpporo1k;tspporo1k;tspporores1k;tspres1k" CACHE STRING "Datasets to fetch from Zenodo")

set(HYPREDRV_ARCHIVE_alcontact1k "alcontact1k.tar.gz" CACHE STRING "Archive name for alcontact1k")
set(HYPREDRV_MD5_alcontact1k "810e465918adacc210167322a53a15a8" CACHE STRING "MD5 checksum for alcontact1k")

set(HYPREDRV_ARCHIVE_cmpf1k "cmpf1k.tar.gz" CACHE STRING "Archive name for cmpf1k")
set(HYPREDRV_MD5_cmpf1k "d870e98646b300aa89b9d8073e87b7ce" CACHE STRING "MD5 checksum for cmpf1k")

set(HYPREDRV_ARCHIVE_cmphyb1k "cmphyb1k.tar.gz" CACHE STRING "Archive name for cmphyb1k")
set(HYPREDRV_MD5_cmphyb1k "c2f21919d406b0cdc805e4bab18136dc" CACHE STRING "MD5 checksum for cmphyb1k")

set(HYPREDRV_ARCHIVE_cmpres1k "cmpres1k.tar.gz" CACHE STRING "Archive name for cmpres1k")
set(HYPREDRV_MD5_cmpres1k "db59a4cc92c0d882d3e7d936f9c1c41d" CACHE STRING "MD5 checksum for cmpres1k")

set(HYPREDRV_ARCHIVE_cmpreshyb1k "cmpreshyb1k.tar.gz" CACHE STRING "Archive name for cmpreshyb1k")
set(HYPREDRV_MD5_cmpreshyb1k "ebcab0ea82ce47716cac3b9ed32d7439" CACHE STRING "MD5 checksum for cmpreshyb1k")

set(HYPREDRV_ARCHIVE_compflow6k "compflow6k.tar.gz" CACHE STRING "Archive name for compflow6k")
set(HYPREDRV_MD5_compflow6k "f9291ba25732e0eac781bdfe4909090c" CACHE STRING "MD5 checksum for compflow6k")

set(HYPREDRV_ARCHIVE_hspporo1k "hspporo1k.tar.gz" CACHE STRING "Archive name for hspporo1k")
set(HYPREDRV_MD5_hspporo1k "11f0dab1800cd627562cb8b46f2df04d" CACHE STRING "MD5 checksum for hspporo1k")

set(HYPREDRV_ARCHIVE_hydrofrac1k "hydrofrac1k.tar.gz" CACHE STRING "Archive name for hydrofrac1k")
set(HYPREDRV_MD5_hydrofrac1k "46d2fcb9ac6f67e78cf722e086dd90a9" CACHE STRING "MD5 checksum for hydrofrac1k")

set(HYPREDRV_ARCHIVE_immf1k "immf1k.tar.gz" CACHE STRING "Archive name for immf1k")
set(HYPREDRV_MD5_immf1k "7d26525a708936954e22d107c525ef1b" CACHE STRING "MD5 checksum for immf1k")

set(HYPREDRV_ARCHIVE_lcontact1k "lcontact1k.tar.gz" CACHE STRING "Archive name for lcontact1k")
set(HYPREDRV_MD5_lcontact1k "855b58ac06755018d239a69f5c5c5390" CACHE STRING "MD5 checksum for lcontact1k")

set(HYPREDRV_ARCHIVE_lcontactbs1k "lcontactbs1k.tar.gz" CACHE STRING "Archive name for lcontactbs1k")
set(HYPREDRV_MD5_lcontactbs1k "10edd3d187ad749049e8906bbfe956f6" CACHE STRING "MD5 checksum for lcontactbs1k")

set(HYPREDRV_ARCHIVE_mhd2dldc1k "mhd2dldc1k.tar.gz" CACHE STRING "Archive name for mhd2dldc1k")
set(HYPREDRV_MD5_mhd2dldc1k "4a7ee5d3f3375154d9e067784aa18541" CACHE STRING "MD5 checksum for mhd2dldc1k")

set(HYPREDRV_ARCHIVE_mhd3ddbdt1k "mhd3ddbdt1k.tar.gz" CACHE STRING "Archive name for mhd3ddbdt1k")
set(HYPREDRV_MD5_mhd3ddbdt1k "cda848ae71072973cf53877e522e1563" CACHE STRING "MD5 checksum for mhd3ddbdt1k")

set(HYPREDRV_ARCHIVE_mpporo1k "mpporo1k.tar.gz" CACHE STRING "Archive name for mpporo1k")
set(HYPREDRV_MD5_mpporo1k "8aa5a42da4cdc974365cf190b3b20909" CACHE STRING "MD5 checksum for mpporo1k")

set(HYPREDRV_ARCHIVE_mpporores1k "mpporores1k.tar.gz" CACHE STRING "Archive name for mpporores1k")
set(HYPREDRV_MD5_mpporores1k "32030ca0ce3709d57db9e45cc2b114a9" CACHE STRING "MD5 checksum for mpporores1k")

set(HYPREDRV_ARCHIVE_poromech2k "poromech2k.tar.gz" CACHE STRING "Archive name for poromech2k")
set(HYPREDRV_MD5_poromech2k "6a530e09b12f533a6de32ab3eb2738ea" CACHE STRING "MD5 checksum for poromech2k")

set(HYPREDRV_ARCHIVE_ps3d10pt7 "ps3d10pt7.tar.gz" CACHE STRING "Archive name for ps3d10pt7")
set(HYPREDRV_MD5_ps3d10pt7 "f1e3fd2c271ef4ed59131acacbb107a8" CACHE STRING "MD5 checksum for ps3d10pt7")

set(HYPREDRV_ARCHIVE_rcmpobl1k "rcmpobl1k.tar.gz" CACHE STRING "Archive name for rcmpobl1k")
set(HYPREDRV_MD5_rcmpobl1k "4384962bed8fffdc53bcdf0152ae53cc" CACHE STRING "MD5 checksum for rcmpobl1k")

set(HYPREDRV_ARCHIVE_smef1k "smef1k.tar.gz" CACHE STRING "Archive name for smef1k")
set(HYPREDRV_MD5_smef1k "229b39060588862f6d19a006e7378a83" CACHE STRING "MD5 checksum for smef1k")

set(HYPREDRV_ARCHIVE_sphyb1k "sphyb1k.tar.gz" CACHE STRING "Archive name for sphyb1k")
set(HYPREDRV_MD5_sphyb1k "e52a013746b5bc7490d426a587b09016" CACHE STRING "MD5 checksum for sphyb1k")

set(HYPREDRV_ARCHIVE_spporo1k "spporo1k.tar.gz" CACHE STRING "Archive name for spporo1k")
set(HYPREDRV_MD5_spporo1k "166cfedf478de2fd7f37ad1e90125cf6" CACHE STRING "MD5 checksum for spporo1k")

set(HYPREDRV_ARCHIVE_spporocf1k "spporocf1k.tar.gz" CACHE STRING "Archive name for spporocf1k")
set(HYPREDRV_MD5_spporocf1k "0d7e78f8369dcb1a22271a2c7a590b4f" CACHE STRING "MD5 checksum for spporocf1k")

set(HYPREDRV_ARCHIVE_spporoef1k "spporoef1k.tar.gz" CACHE STRING "Archive name for spporoef1k")
set(HYPREDRV_MD5_spporoef1k "d98a5b4d9b4ddd430b81978cc73843c9" CACHE STRING "MD5 checksum for spporoef1k")

set(HYPREDRV_ARCHIVE_spporores1k "spporores1k.tar.gz" CACHE STRING "Archive name for spporores1k")
set(HYPREDRV_MD5_spporores1k "833e6825b5d27df29acd196c846b97b0" CACHE STRING "MD5 checksum for spporores1k")

set(HYPREDRV_ARCHIVE_spres1k "spres1k.tar.gz" CACHE STRING "Archive name for spres1k")
set(HYPREDRV_MD5_spres1k "7047e357d9e4579a0f446a67723c9361" CACHE STRING "MD5 checksum for spres1k")

set(HYPREDRV_ARCHIVE_spreshyb1k "spreshyb1k.tar.gz" CACHE STRING "Archive name for spreshyb1k")
set(HYPREDRV_MD5_spreshyb1k "db1e6a691622cbd6e0ba5f366b26c6c9" CACHE STRING "MD5 checksum for spreshyb1k")

set(HYPREDRV_ARCHIVE_tcmpf1k "tcmpf1k.tar.gz" CACHE STRING "Archive name for tcmpf1k")
set(HYPREDRV_MD5_tcmpf1k "2f1947641a108d75ef71ed6c9edaac2b" CACHE STRING "MD5 checksum for tcmpf1k")

set(HYPREDRV_ARCHIVE_tcmpres1k "tcmpres1k.tar.gz" CACHE STRING "Archive name for tcmpres1k")
set(HYPREDRV_MD5_tcmpres1k "0a3ea681e5d3022203f7c06fc86efe73" CACHE STRING "MD5 checksum for tcmpres1k")

set(HYPREDRV_ARCHIVE_tmpporo1k "tmpporo1k.tar.gz" CACHE STRING "Archive name for tmpporo1k")
set(HYPREDRV_MD5_tmpporo1k "b6444ab811efe52a64dfee2b19d88dbd" CACHE STRING "MD5 checksum for tmpporo1k")

set(HYPREDRV_ARCHIVE_tspporo1k "tspporo1k.tar.gz" CACHE STRING "Archive name for tspporo1k")
set(HYPREDRV_MD5_tspporo1k "92af9bd07bb3791326739dd02b70518f" CACHE STRING "MD5 checksum for tspporo1k")

set(HYPREDRV_ARCHIVE_tspporores1k "tspporores1k.tar.gz" CACHE STRING "Archive name for tspporores1k")
set(HYPREDRV_MD5_tspporores1k "8fae3ab75f11a5964b5741f60c195687" CACHE STRING "MD5 checksum for tspporores1k")

set(HYPREDRV_ARCHIVE_tspres1k "tspres1k.tar.gz" CACHE STRING "Archive name for tspres1k")
set(HYPREDRV_MD5_tspres1k "01d665812937b5fd7add1d50baaf7949" CACHE STRING "MD5 checksum for tspres1k")

set(_dl_dir "${CMAKE_BINARY_DIR}/downloads")
set(_stamp_dir "${CMAKE_BINARY_DIR}/stamps")
file(MAKE_DIRECTORY "${_dl_dir}" "${_stamp_dir}")

set(_dl_extract_script "${CMAKE_SOURCE_DIR}/scripts/download_and_extract.sh")
set(_data_dir "${CMAKE_SOURCE_DIR}/data")

# Check whether a data prefix exists as either ASCII (.00000) or binary (.00000.bin).
function(hypredrv_prefix_exists out_var dataset_dir rel_prefix)
    set(_ascii "${dataset_dir}/${rel_prefix}.00000")
    set(_bin   "${dataset_dir}/${rel_prefix}.00000.bin")
    if(EXISTS "${_ascii}" OR EXISTS "${_bin}")
        set(${out_var} TRUE PARENT_SCOPE)
    else()
        set(${out_var} FALSE PARENT_SCOPE)
    endif()
endfunction()

set(_dataset_stamps)
foreach(dataset IN LISTS HYPREDRV_DATASETS)
    set(archive "${HYPREDRV_ARCHIVE_${dataset}}")
    set(md5     "${HYPREDRV_MD5_${dataset}}")
    set(url     "${HYPREDRV_DATA_BASE_URL}/${archive}/content")
    set(tarball "${_dl_dir}/${archive}")
    set(stamp   "${_stamp_dir}/data_${dataset}.stamp")
    set(dataset_dir "${_data_dir}/${dataset}")

    # Check if required files for this dataset are present.
    set(_dataset_exists FALSE)
    if(EXISTS "${dataset_dir}" AND IS_DIRECTORY "${dataset_dir}")
        if(dataset STREQUAL "ps3d10pt7")
            set(_required_prefixes
                "np1/IJ.out.A"
                "np1/IJ.out.b"
                "np4/IJ.out.A"
                "np4/IJ.out.b"
            )
        elseif(dataset STREQUAL "compflow6k")
            set(_required_prefixes
                "np1/IJ.out.A"
                "np1/IJ.out.b"
                "np1/dofmap.out"
                "np4/IJ.out.A"
                "np4/IJ.out.b"
                "np4/dofmap.out"
            )
        elseif(dataset STREQUAL "poromech2k")
            set(_required_prefixes
                "np1/ls_00000/IJ.out.A"
                "np1/ls_00000/IJ.out.b"
                "np1/ls_00000/dofmap.out"
            )
        elseif(dataset STREQUAL "mhd3ddbdt1k" OR dataset STREQUAL "mhd2dldc1k")
            set(_required_prefixes
                "np1/IJ.out.A"
                "np1/IJ.out.b"
                "np1/dofmap.out"
                "np4/IJ.out.A"
                "np4/IJ.out.b"
                "np4/dofmap.out"
            )
        else()
            set(_required_prefixes
                "np1/IJ.out.A"
                "np1/IJ.out.b"
                "np1/dofmap.out"
                "np4/IJ.out.A"
                "np4/IJ.out.b"
                "np4/dofmap.out"
            )
        endif()

        set(_all_present TRUE)
        foreach(_prefix IN LISTS _required_prefixes)
            hypredrv_prefix_exists(_prefix_ok "${dataset_dir}" "${_prefix}")
            if(NOT _prefix_ok)
                set(_all_present FALSE)
                break()
            endif()
        endforeach()

        if(_all_present)
            set(_dataset_exists TRUE)
            message(STATUS "Dataset ${dataset} already exists at ${dataset_dir}, skipping download")
            # Create stamp file to indicate dataset is available
            add_custom_command(OUTPUT "${stamp}"
                COMMAND ${CMAKE_COMMAND} -E touch "${stamp}"
                COMMENT "Dataset ${dataset} already present"
                VERBATIM
            )
        else()
            message(STATUS "Dataset ${dataset} is incomplete at ${dataset_dir}; fetching from Zenodo")
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
