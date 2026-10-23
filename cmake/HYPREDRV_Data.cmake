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
set(HYPREDRV_SHA256_alcontact1k "52d941eff73dd3ee66661955cc9f3972a6f45f548c73fd1192f3d7b276791b39" CACHE STRING "SHA-256 checksum for alcontact1k")

set(HYPREDRV_ARCHIVE_cmpf1k "cmpf1k.tar.gz" CACHE STRING "Archive name for cmpf1k")
set(HYPREDRV_SHA256_cmpf1k "5d008f369e404be977cb33ca9861624b6e82516f592e98b0a9ef727645348fd5" CACHE STRING "SHA-256 checksum for cmpf1k")

set(HYPREDRV_ARCHIVE_cmphyb1k "cmphyb1k.tar.gz" CACHE STRING "Archive name for cmphyb1k")
set(HYPREDRV_SHA256_cmphyb1k "d0f0b1347debf4a8c678f3461d254e649bdba85cea6542e982f6b9d32fc4af64" CACHE STRING "SHA-256 checksum for cmphyb1k")

set(HYPREDRV_ARCHIVE_cmpres1k "cmpres1k.tar.gz" CACHE STRING "Archive name for cmpres1k")
set(HYPREDRV_SHA256_cmpres1k "3efcf7e030714ad55caa7d60be0d7899f478ffb5b08fc67877efd12f255e1a95" CACHE STRING "SHA-256 checksum for cmpres1k")

set(HYPREDRV_ARCHIVE_cmpreshyb1k "cmpreshyb1k.tar.gz" CACHE STRING "Archive name for cmpreshyb1k")
set(HYPREDRV_SHA256_cmpreshyb1k "3103317daff43c7da03df10295ffd29a06fc0be622f2b44b9cebd25ecb1e3cef" CACHE STRING "SHA-256 checksum for cmpreshyb1k")

set(HYPREDRV_ARCHIVE_compflow6k "compflow6k.tar.gz" CACHE STRING "Archive name for compflow6k")
set(HYPREDRV_SHA256_compflow6k "ba807b3bff32d90e18595ac0e5fb153bba019badc9e8b23bb7b7ac00b21ddd14" CACHE STRING "SHA-256 checksum for compflow6k")

set(HYPREDRV_ARCHIVE_hspporo1k "hspporo1k.tar.gz" CACHE STRING "Archive name for hspporo1k")
set(HYPREDRV_SHA256_hspporo1k "bb426478f65e8668ba88dd1e6a79a38dcf60beb3c07ab14e996c541b3d0a4c7f" CACHE STRING "SHA-256 checksum for hspporo1k")

set(HYPREDRV_ARCHIVE_hydrofrac1k "hydrofrac1k.tar.gz" CACHE STRING "Archive name for hydrofrac1k")
set(HYPREDRV_SHA256_hydrofrac1k "9a574bd83dd10cf09bd7d0630498fef628d1d86d27228c88a8432e412926827c" CACHE STRING "SHA-256 checksum for hydrofrac1k")

set(HYPREDRV_ARCHIVE_immf1k "immf1k.tar.gz" CACHE STRING "Archive name for immf1k")
set(HYPREDRV_SHA256_immf1k "06f04d657bba1f3546159c86914602f6224d43dbd613133d7267505fbc7e50cc" CACHE STRING "SHA-256 checksum for immf1k")

set(HYPREDRV_ARCHIVE_lcontact1k "lcontact1k.tar.gz" CACHE STRING "Archive name for lcontact1k")
set(HYPREDRV_SHA256_lcontact1k "419b91a908129e33a172d3e0886dfa47272600e707b508c2ecd9002a09f5a607" CACHE STRING "SHA-256 checksum for lcontact1k")

set(HYPREDRV_ARCHIVE_lcontactbs1k "lcontactbs1k.tar.gz" CACHE STRING "Archive name for lcontactbs1k")
set(HYPREDRV_SHA256_lcontactbs1k "64ff40240fae9bc4cf995f648f48bad73a09110bb841009de68ea214cdaf0a10" CACHE STRING "SHA-256 checksum for lcontactbs1k")

set(HYPREDRV_ARCHIVE_mhd2dldc1k "mhd2dldc1k.tar.gz" CACHE STRING "Archive name for mhd2dldc1k")
set(HYPREDRV_SHA256_mhd2dldc1k "8cf133f71a50a57038062b084aa2de3f8e4f3683ec6f74c27f97cba3fa8ee44d" CACHE STRING "SHA-256 checksum for mhd2dldc1k")

set(HYPREDRV_ARCHIVE_mhd3ddbdt1k "mhd3ddbdt1k.tar.gz" CACHE STRING "Archive name for mhd3ddbdt1k")
set(HYPREDRV_SHA256_mhd3ddbdt1k "52ac170c7114c81862caa2a8ed365b0cd30fe07a90f8411672cc2e71ae0e4f11" CACHE STRING "SHA-256 checksum for mhd3ddbdt1k")

set(HYPREDRV_ARCHIVE_mpporo1k "mpporo1k.tar.gz" CACHE STRING "Archive name for mpporo1k")
set(HYPREDRV_SHA256_mpporo1k "4670e3ecef5a5b658f4c210743452a42738efd1e28a9370778bce449b4bd5341" CACHE STRING "SHA-256 checksum for mpporo1k")

set(HYPREDRV_ARCHIVE_mpporores1k "mpporores1k.tar.gz" CACHE STRING "Archive name for mpporores1k")
set(HYPREDRV_SHA256_mpporores1k "26c9427350af3ab310e7ea83c752602cb9d02087ca5e4fe8bd45223ba24e407f" CACHE STRING "SHA-256 checksum for mpporores1k")

set(HYPREDRV_ARCHIVE_poromech2k "poromech2k.tar.gz" CACHE STRING "Archive name for poromech2k")
set(HYPREDRV_SHA256_poromech2k "7dbdaf271adbae00bfb45cc76690c4356100418fe4f62eebf66aa43a433e0f5e" CACHE STRING "SHA-256 checksum for poromech2k")

set(HYPREDRV_ARCHIVE_ps3d10pt7 "ps3d10pt7.tar.gz" CACHE STRING "Archive name for ps3d10pt7")
set(HYPREDRV_SHA256_ps3d10pt7 "ae3a96e97cb60c7bf2281b9c8cdbf9c17157e2b136b774eb8bd5adc6f55c4c4a" CACHE STRING "SHA-256 checksum for ps3d10pt7")

set(HYPREDRV_ARCHIVE_rcmpobl1k "rcmpobl1k.tar.gz" CACHE STRING "Archive name for rcmpobl1k")
set(HYPREDRV_SHA256_rcmpobl1k "46c78f8bd1deadf9ef76a8e9af4ba4668a0e28b0cb6e883a99502ed05d458c03" CACHE STRING "SHA-256 checksum for rcmpobl1k")

set(HYPREDRV_ARCHIVE_smef1k "smef1k.tar.gz" CACHE STRING "Archive name for smef1k")
set(HYPREDRV_SHA256_smef1k "8e69d353b9534a5168dd8f0fa8f69e9fd657fd8d6dc0e7378930852e8e3b0adc" CACHE STRING "SHA-256 checksum for smef1k")

set(HYPREDRV_ARCHIVE_sphyb1k "sphyb1k.tar.gz" CACHE STRING "Archive name for sphyb1k")
set(HYPREDRV_SHA256_sphyb1k "369f9ffa3ad32b6c5b6586e32ec12f5a99a39eb99c2f642d72760ce25e2d415c" CACHE STRING "SHA-256 checksum for sphyb1k")

set(HYPREDRV_ARCHIVE_spporo1k "spporo1k.tar.gz" CACHE STRING "Archive name for spporo1k")
set(HYPREDRV_SHA256_spporo1k "a7e3b924da91df5b7b5188aedb21a2107aba26fea53b67563bb92b445d047cc1" CACHE STRING "SHA-256 checksum for spporo1k")

set(HYPREDRV_ARCHIVE_spporocf1k "spporocf1k.tar.gz" CACHE STRING "Archive name for spporocf1k")
set(HYPREDRV_SHA256_spporocf1k "d69e2280c25022eac3fe1114ba3c321e7a681097c7602090ec27373517f1c7b9" CACHE STRING "SHA-256 checksum for spporocf1k")

set(HYPREDRV_ARCHIVE_spporoef1k "spporoef1k.tar.gz" CACHE STRING "Archive name for spporoef1k")
set(HYPREDRV_SHA256_spporoef1k "cf3522a0e41d954a79649f9a1862349b34e3aa0d4019683d124c1ba457dadbf3" CACHE STRING "SHA-256 checksum for spporoef1k")

set(HYPREDRV_ARCHIVE_spporores1k "spporores1k.tar.gz" CACHE STRING "Archive name for spporores1k")
set(HYPREDRV_SHA256_spporores1k "0a004fefb9edd0ea207785d92475e1fe442c1cdc59bacad68102ecbf136086ad" CACHE STRING "SHA-256 checksum for spporores1k")

set(HYPREDRV_ARCHIVE_spres1k "spres1k.tar.gz" CACHE STRING "Archive name for spres1k")
set(HYPREDRV_SHA256_spres1k "cbc1a9b5bced60f29b38725c613b29ec144e8131fe8a0afd004c7ee5b151f5c4" CACHE STRING "SHA-256 checksum for spres1k")

set(HYPREDRV_ARCHIVE_spreshyb1k "spreshyb1k.tar.gz" CACHE STRING "Archive name for spreshyb1k")
set(HYPREDRV_SHA256_spreshyb1k "a666597c476797457fbab2a9deb32f7a4e9294e1adc390cc08fd99d62c5f2c1d" CACHE STRING "SHA-256 checksum for spreshyb1k")

set(HYPREDRV_ARCHIVE_tcmpf1k "tcmpf1k.tar.gz" CACHE STRING "Archive name for tcmpf1k")
set(HYPREDRV_SHA256_tcmpf1k "9792eeca70a0b6118f62773cf483b4c2de043ab72c26109c8e004928d1b462f9" CACHE STRING "SHA-256 checksum for tcmpf1k")

set(HYPREDRV_ARCHIVE_tcmpres1k "tcmpres1k.tar.gz" CACHE STRING "Archive name for tcmpres1k")
set(HYPREDRV_SHA256_tcmpres1k "9afd6be780fe96bc868e65e0948562c260b05330a93d4c363caf3f1a0476291d" CACHE STRING "SHA-256 checksum for tcmpres1k")

set(HYPREDRV_ARCHIVE_tmpporo1k "tmpporo1k.tar.gz" CACHE STRING "Archive name for tmpporo1k")
set(HYPREDRV_SHA256_tmpporo1k "087660c59492dcfc7440dd4a0514394f36c9d206ecd88f4b7fec3e4b941a5eb1" CACHE STRING "SHA-256 checksum for tmpporo1k")

set(HYPREDRV_ARCHIVE_tspporo1k "tspporo1k.tar.gz" CACHE STRING "Archive name for tspporo1k")
set(HYPREDRV_SHA256_tspporo1k "904c455b1da5b11d514f0fcb5279160558cb69424d2ca9865a043850c9bf08c4" CACHE STRING "SHA-256 checksum for tspporo1k")

set(HYPREDRV_ARCHIVE_tspporores1k "tspporores1k.tar.gz" CACHE STRING "Archive name for tspporores1k")
set(HYPREDRV_SHA256_tspporores1k "327fabb283e3697bc89dec2062676cceceb0b85a2e83f8185f0487ae06dbb43e" CACHE STRING "SHA-256 checksum for tspporores1k")

set(HYPREDRV_ARCHIVE_tspres1k "tspres1k.tar.gz" CACHE STRING "Archive name for tspres1k")
set(HYPREDRV_SHA256_tspres1k "acbe3bbad9a06000500fb5d5636692b7177b323087db85a6024579e591b4a87f" CACHE STRING "SHA-256 checksum for tspres1k")

set(_dl_dir "${CMAKE_BINARY_DIR}/downloads")
set(_stamp_dir "${CMAKE_BINARY_DIR}/stamps")
file(MAKE_DIRECTORY "${_dl_dir}" "${_stamp_dir}")

set(_dl_extract_script "${CMAKE_SOURCE_DIR}/cmake/HYPREDRV_FetchDataset.cmake")
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
    set(sha256  "${HYPREDRV_SHA256_${dataset}}")
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
          COMMAND ${CMAKE_COMMAND} -DURL=${url} -DTARBALL=${tarball} -DDEST=${_data_dir}
                  -DSHA256=${sha256} -P "${_dl_extract_script}"
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
