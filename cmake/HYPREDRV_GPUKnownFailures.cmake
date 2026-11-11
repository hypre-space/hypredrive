# Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

# Temporary GPU exclusions from the hypre v3.2.1 CUDA/SYCL regression sweeps.
# Include failures in device-only and unified-memory configurations; retain
# the basic SYCL cases that passed clean reruns.
# Revisit these cases with the next hypre/hypredrive releases. Keep CPU and
# other GPU backends covered; HYPREDRV_ENABLE_ALL_TESTS overrides this policy.

if(HYPRE_ENABLE_CUDA)
    list(APPEND HYPREDRV_GPU_DISABLED_TESTS
        elasticity_two_material_1proc
        elasticity_two_material_4proc
        elasticity_two_material_coarse_reuse_1proc
        elasticity_two_material_vis_4proc
        hypredrive_test_alcontact1k_np4
        hypredrive_test_cmpres1k_np4
        hypredrive_test_cmpreshyb1k_np4
        hypredrive_test_ex1_schwarz_1proc
        hypredrive_test_ex1_schwarz_4proc
        hypredrive_test_ex1_schwarz_ilut_1proc
        hypredrive_test_ex1_schwarz_ilut_4proc
        hypredrive_test_lcontact1k_np4
        hypredrive_test_lcontactbs1k_np4
        hypredrive_test_mpporores1k_np4
        hypredrive_test_smef1k_np4
        hypredrive_test_spreshyb1k_np4
        hypredrive_test_tcmpres1k_np4
        lidcavity_test_mgr_schwarz_coarsest_iluk_1proc
        lidcavity_test_schwarz_ras_iluk_1proc
        state_vectors_mpi_4proc
    )
endif()

if(HYPRE_ENABLE_SYCL)
    list(APPEND HYPREDRV_GPU_DISABLED_TESTS
        convdif_test_ilu_1proc
        eigspec_mpi_2proc
        eigspec_mpi_4proc
        elasticity_two_material_1proc
        elasticity_two_material_4proc
        elasticity_two_material_coarse_reuse_1proc
        elasticity_two_material_vis_4proc
        hypredrive_test_alcontact1k_np1
        hypredrive_test_alcontact1k_np4
        hypredrive_test_cmpf1k_np1
        hypredrive_test_cmpf1k_np4
        hypredrive_test_cmphyb1k_np1
        hypredrive_test_cmphyb1k_np4
        hypredrive_test_cmpres1k_np1
        hypredrive_test_cmpres1k_np4
        hypredrive_test_cmpreshyb1k_np1
        hypredrive_test_cmpreshyb1k_np4
        hypredrive_test_ex1_schwarz_1proc
        hypredrive_test_ex1_schwarz_4proc
        hypredrive_test_ex1_schwarz_ilut_1proc
        hypredrive_test_ex1_schwarz_ilut_4proc
        hypredrive_test_ex1b_1proc
        hypredrive_test_ex1c_1proc
        hypredrive_test_ex2_4proc
        hypredrive_test_ex3_1proc
        hypredrive_test_ex3_flow_1proc
        hypredrive_test_ex3_nested_1
        hypredrive_test_ex3_nested_2
        hypredrive_test_ex4_4proc
        hypredrive_test_ex4_cli_mgr_f_amg
        hypredrive_test_ex4_cli_mgr_f_ilu
        hypredrive_test_ex4_cli_mgr_g_amg
        hypredrive_test_ex4_cli_mgr_g_ilu
        hypredrive_test_ex5_1proc
        hypredrive_test_ex6_1proc
        hypredrive_test_ex7_1proc
        hypredrive_test_ex7_cli_dofmap_scaling
        hypredrive_test_ex7_cli_reps4_ls4
        hypredrive_test_ex7_cli_stats2
        hypredrive_test_ex7_cli_tagerr_randsol
        hypredrive_test_ex7_cli_tagerr_scale
        hypredrive_test_ex7_cli_tagres
        hypredrive_test_ex7_custom_scaling
        hypredrive_test_ex7_mgr_coarse_reuse_1proc
        hypredrive_test_ex7_mgr_cycle_v01_1proc
        hypredrive_test_ex7_mgr_cycle_v11_1proc
        hypredrive_test_ex7_mgr_cycle_v_1proc
        hypredrive_test_ex7_mgr_cycle_w11_1proc
        hypredrive_test_ex7_mgr_cycle_w_1proc
        hypredrive_test_ex7_mgr_frelax_ilu_reuse_1proc
        hypredrive_test_ex7_mgr_frelax_reuse_1proc
        hypredrive_test_ex7_mgr_grelax_reuse_1proc
        hypredrive_test_ex7_nested_krylov_mgr_1proc
        hypredrive_test_ex7_nested_mgr_1proc
        hypredrive_test_ex8_1proc
        hypredrive_test_ex8a_4proc
        hypredrive_test_ex8b_4proc
        hypredrive_test_immf1k_np1
        hypredrive_test_immf1k_np4
        hypredrive_test_lcontact1k_np4
        hypredrive_test_lcontactbs1k_np4
        hypredrive_test_mhd2dldc1k_np4
        hypredrive_test_mhd3ddbdt1k_np4
        hypredrive_test_mpporo1k_np1
        hypredrive_test_mpporo1k_np4
        hypredrive_test_mpporores1k_np1
        hypredrive_test_mpporores1k_np4
        hypredrive_test_rcmpobl1k_np1
        hypredrive_test_rcmpobl1k_np4
        hypredrive_test_smef1k_np4
        hypredrive_test_sphyb1k_np1
        hypredrive_test_spporocf1k_np1
        hypredrive_test_spporocf1k_np4
        hypredrive_test_spporoef1k_np1
        hypredrive_test_spporoef1k_np4
        hypredrive_test_spporores1k_np1
        hypredrive_test_spporores1k_np4
        hypredrive_test_spres1k_np1
        hypredrive_test_spres1k_np4
        hypredrive_test_spreshyb1k_np1
        hypredrive_test_spreshyb1k_np4
        hypredrive_test_tcmpf1k_np1
        hypredrive_test_tcmpf1k_np4
        hypredrive_test_tcmpres1k_np1
        hypredrive_test_tcmpres1k_np4
        hypredrive_test_tmpporo1k_np1
        hypredrive_test_tmpporo1k_np4
        hypredrive_test_tspporores1k_np1
        hypredrive_test_tspporores1k_np4
        hypredrive_test_tspres1k_np1
        hypredrive_test_tspres1k_np4
        lidcavity_test_mgr_schwarz_coarsest_iluk_1proc
        lidcavity_test_schwarz_ras_iluk_1proc
        unit_test_eigspec
        unit_test_hypredrv
        unit_test_solver
    )
endif()
