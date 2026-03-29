/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef HYPREDRV_HEADER
#define HYPREDRV_HEADER

#include <mpi.h>
#include <stdint.h> // For: uint32_t

#include <HYPRE.h>
#include <HYPRE_IJ_mv.h>
#include <HYPRE_config.h>
#include <HYPRE_krylov.h>
#include <HYPRE_parcsr_ls.h>
#include <HYPRE_parcsr_mv.h>
#include <HYPRE_utilities.h>

// Undefine autotools package macros from hypre
#undef PACKAGE_NAME
#undef PACKAGE_BUGREPORT
#undef PACKAGE_STRING
#undef PACKAGE_TARNAME
#undef PACKAGE_URL
#undef PACKAGE_VERSION

#include "HYPREDRV_config.h"

#ifdef __cplusplus
extern "C"
{
#endif

// Visibility control macros
#define HYPREDRV_EXPORT_SYMBOL __attribute__((visibility("default")))

   // HYPREDRV_SAFE_CALL and HYPREDRV_SAFE_CALL_COMM are defined in HYPREDRV_utils.h.
   // Include that header after this one when you need these macros.

   /*--------------------------------------------------------------------------
    *--------------------------------------------------------------------------*/

   /**
    * @defgroup HYPREDRV HYPREDRV Public API
    * @brief Public APIs to solve linear systems with hypre through hypredrive.
    * @{
    **/

   /*--------------------------------------------------------------------------
    *--------------------------------------------------------------------------*/

   struct hypredrv_struct;
   /**
    * @ingroup HYPREDRV
    * @typedef HYPREDRV_t
    * @brief Opaque handle for a hypredrive context.
    *
    * The main object type for the HYPREDRV library encapsulates the parsed input
    * parameters, linear-system objects, solver/preconditioner state, performance
    * statistics, and MPI communication context. Create it with HYPREDRV_Create()
    * and destroy it with HYPREDRV_Destroy().
    */
   typedef struct hypredrv_struct *HYPREDRV_t;

   /**
    * @brief Initializes the HYPREDRV library.
    *
    * Initializes the HYPREDRV library for use in the application. This function sets up
    * the necessary environment for the HYPREDRV library by initializing the HYPRE library
    * and its device-specific components. It ensures that initialization occurs only once
    * even if called multiple times.
    *
    * @return Returns an error code with 0 indicating success. Any non-zero value
    * indicates a failure, and the error code can be further described using
    * HYPREDRV_ErrorCodeDescribe(error_code).
    *
    * @note This function must be called after MPI_Init and before any other HYPREDRV
    * functions are used. It is safe to call this function multiple times; only the first
    * call will perform the initialization.
    *
    * Example Usage:
    * @code
    *    HYPREDRV_SAFE_CALL(HYPREDRV_Initialize());
    * @endcode
    */

   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_Initialize(void);

   /**
    * @brief Finalizes the HYPREDRV library.
    *
    * Cleans up and releases any resources allocated by the HYPREDRV library. This
    * function automatically destroys any live ``HYPREDRV_t`` objects still owned by the
    * library, then finalizes the HYPRE library and its device-specific components. It
    * should be the last HYPREDRV-related function called before the application exits.
    *
    * @return Returns an error code with 0 indicating success. Any non-zero value
    * indicates a failure, and the error code can be further described using
    * HYPREDRV_ErrorCodeDescribe(error_code).
    *
    * @note This function should be called before MPI_Finalize. It is safe to call this
    * function even if HYPREDRV_Initialize was not successfully called or was not called
    * at all; in such cases, the function will have no effect. Auto-destroyed live
    * handles are reclaimed internally, but caller-owned variables are not rewritten to
    * ``NULL`` because only the library sees the live-object registry.
    *
    * Example Usage:
    * @code
    *    HYPREDRV_SAFE_CALL(HYPREDRV_Finalize());
    * @endcode
    */

   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_Finalize(void);

   /**
    * @brief Describes and handles an error code from HYPREDRV operations.
    *
    * This function processes an error code from HYPREDRV operations. If the error code is
    * non-zero, it will:
    * 1. Describe the error using the internal error code description system
    * 2. Print the error message to stderr
    * 3. Clear the error message buffer
    *
    * If the error code is zero (indicating no error), the function returns immediately
    * without taking any action.
    *
    * @param error_code The error code to be processed (uint32_t)
    *
    * @note This function prints the error description and returns normally - it does
    * NOT call MPI_Abort. For an abort-on-error helper, use the HYPREDRV_SAFE_CALL
    * macro (defined in HYPREDRV_utils.h), which calls this function and then calls
    * MPI_Abort when the error code is non-zero.
    *
    * Example Usage:
    * @code
    *    // Direct usage:
    *    uint32_t error = some_HYPREDRV_operation();
    *    HYPREDRV_ErrorCodeDescribe(error);
    *
    *    // Preferred usage with macro:
    *    HYPREDRV_SAFE_CALL(some_HYPREDRV_operation());
    * @endcode
    */

   HYPREDRV_EXPORT_SYMBOL void HYPREDRV_ErrorCodeDescribe(uint32_t error_code);

   /**
    * @brief Create a HYPREDRV object.
    *
    * This function allocates memory for a HYPREDRV object and initializes it with the
    * provided MPI communicator. The newly created object is returned through the
    * hypredrv_ptr parameter.
    *
    * @param comm The MPI communicator to be associated with the HYPREDRV object.
    * This communicator is used for parallel communications in the underlying HYPRE
    * library calls.
    *
    * @param hypredrv_ptr A pointer to the HYPREDRV_t object where the address of the
    * newly allocated HYPREDRV object will be stored. After the function call,
    * *hypredrv_ptr will point to the allocated object.
    *
    * @return Returns an error code with 0 indicating success. Any non-zero value
    * indicates a failure, and the error code can be further described using
    * HYPREDRV_ErrorCodeDescribe(error_code).
    *
    * @note It is the caller's responsibility to ensure that the MPI environment is
    * properly initialized before calling this function. The function does not initialize
    * or finalize the MPI environment.
    *
    * Example Usage:
    * @code
    *    HYPREDRV_t *hypredrv;
    *    MPI_Comm comm = MPI_COMM_WORLD; // or any other valid MPI_Comm
    *    HYPREDRV_SAFE_CALL(HYPREDRV_Create(comm, &hypredrv));
    * @endcode
    */

   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_Create(MPI_Comm    comm,
                                                   HYPREDRV_t *hypredrv_ptr);

   /**
    * @brief Destroy a HYPREDRV object.
    *
    * This function deallocates the memory for a HYPREDRV object and performs necessary
    * cleanup for its associated resources. It destroys HYPRE matrices and vectors created
    * and used by the object, deallocates any input arguments, and finally frees the
    * HYPREDRV object itself. After this function is called, the pointer to the HYPREDRV
    * object is set to NULL to prevent any dangling references.
    *
    * @param hypredrv_ptr A pointer to the HYPREDRV_t object that is to be destroyed. This
    * pointer should have been initialized by HYPREDRV_Create function.
    *
    * @return Returns an error code with 0 indicating success. Any non-zero value
    * indicates a failure, and the error code can be further described using
    * HYPREDRV_ErrorCodeDescribe(error_code).
    *
    * @note It is the caller's responsibility to ensure that hypredrv_ptr is a valid
    * pointer to a HYPREDRV_t object. Passing an invalid pointer or a pointer to an
    * already destroyed object can lead to undefined behavior.
    *
    * Example Usage:
    * @code
    *    HYPREDRV_t *hypredrv;
    *    // ... (hypredrv is created and used) ...
    *    HYPREDRV_SAFE_CALL(HYPREDRV_Destroy(&hypredrv));
    *    // hypredrv is now NULL.
    * @endcode
    */

   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_Destroy(HYPREDRV_t *hypredrv_ptr);

   /**
    * @brief Print library information at entrance.
    *
    * This function optionally prints the current date and time, followed by version
    * information about hypredrive and hypre.
    *
    * @param comm The MPI communicator associated with the HYPREDRV object.
    * @param print_datetime Whether to print the date/time header (nonzero => on).
    *
    * @return Returns an error code with 0 indicating success. Any non-zero value
    * indicates a failure, and the error code can be further described using
    * HYPREDRV_ErrorCodeDescribe(error_code).
    *
    * Example Usage:
    * @code
    *    MPI_Comm comm = MPI_COMM_WORLD; // or any other valid MPI_Comm
    *    HYPREDRV_SAFE_CALL(HYPREDRV_PrintLibInfo(comm, 1));
    * @endcode
    */

   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_PrintLibInfo(MPI_Comm comm,
                                                         int      print_datetime);

   /**
    * @brief Print system information.
    *
    * @param comm The MPI communicator associated with the HYPREDRV object.
    *
    * @return Returns an error code with 0 indicating success. Any non-zero value
    * indicates a failure, and the error code can be further described using
    * HYPREDRV_ErrorCodeDescribe(error_code).
    *
    * @note This function uses conditional compilation to determine what information about
    * the system (machine) to print.
    *
    * Example Usage:
    * @code
    *    MPI_Comm comm = MPI_COMM_WORLD; // or any other valid MPI_Comm
    *    HYPREDRV_SAFE_CALL(HYPREDRV_PrintSystemInfo(comm));
    * @endcode
    */

   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_PrintSystemInfo(MPI_Comm comm);

   /**
    * @brief Print library information at exit.
    *
    * This function prints the driver name followed by the current date and time.
    *
    * @param comm The MPI communicator associated with the HYPREDRV object.
    * @param argv0 The application driver name.
    *
    * @return Returns an error code with 0 indicating success. Any non-zero value
    indicates a failure,
    * and the error code can be further described using
    HYPREDRV_ErrorCodeDescribe(error_code).

    * @note This function is intended to be used just before finishing the driver
    application
    *
    * Example Usage:
    * @code
    *    MPI_Comm comm = MPI_COMM_WORLD; // or any other valid MPI_Comm
    *    HYPREDRV_SAFE_CALL(HYPREDRV_PrintExitInfo(comm, argv[0]));
    * @endcode
    */

   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_PrintExitInfo(MPI_Comm    comm,
                                                          const char *argv0);

   /**
    * @brief Parse input arguments for a HYPREDRV object.
    *
    * This function parses the command-line arguments and immediately applies the
    * resulting configuration: HYPRE execution policy and Umpire pool sizes are set
    * (skipped in library mode, where the caller owns HYPRE initialization), stats
    * counters are initialized, and any preconditioner-reuse timestep schedule is loaded.
    * No separate "configure" call is needed after this function.
    *
    * @param argc The count of command-line arguments.
    * @param argv The array of command-line argument strings.
    * @param hypredrv The HYPREDRV_t object whose input arguments are to be parsed and
    * set.
    *
    * @return Returns an error code with 0 indicating success. Any non-zero value
    * indicates a failure, and the error code can be further described using
    * HYPREDRV_ErrorCodeDescribe(error_code).
    *
    * @note It is expected that argv[1] is the name of the input file in YAML format
    *
    * Example Usage:
    * @code
    *    HYPREDRV_t hypredrv = NULL;
    *    int argc = ...; // Number of arguments
    *    char **argv = ...; // Argument strings
    *    // ... (hypredrv is created) ...
    *    HYPREDRV_SAFE_CALL(HYPREDRV_InputArgsParse(argc, argv, hypredrv));
    * @endcode
    */

   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_InputArgsParse(int argc, char **argv,
                                                           HYPREDRV_t hypredrv);

   /**
    * @brief Enable library (borrowed-ownership) mode for a HYPREDRV object.
    *
    * When library mode is active, matrices and vectors passed via
    * HYPREDRV_LinearSystemSetMatrix(), HYPREDRV_LinearSystemSetRHS(),
    * HYPREDRV_LinearSystemSetInitialGuess(), etc. are treated as borrowed
    * references: HYPREDRV will not destroy them when the object is destroyed or
    * when new objects are set. The caller retains full ownership and must free
    * these objects independently.
    *
    * When library mode is inactive (the default), ownership of non-NULL objects
    * passed to those setters is transferred to HYPREDRV, which will destroy them
    * at the appropriate time.
    *
    * @note This flag is a one-way latch: once set it cannot be unset for the
    * lifetime of the HYPREDRV_t object. Create a new object if you need to
    * switch back to owned mode.
    *
    * @param hypredrv The HYPREDRV_t object (passed by value - it is an opaque
    *                 pointer, not a pointer-to-pointer).
    *
    * @return Returns an error code with 0 indicating success. Any non-zero value
    * indicates a failure, and the error code can be further described using
    * HYPREDRV_ErrorCodeDescribe(error_code).
    *
    * Example Usage:
    * @code
    *    HYPREDRV_t hypredrv;
    *    // ... (hypredrv is created) ...
    *    HYPREDRV_SAFE_CALL(HYPREDRV_SetLibraryMode(hypredrv));
    * @endcode
    */

   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_SetLibraryMode(HYPREDRV_t hypredrv);

   /**
    * @brief Set or clear an optional display name for a HYPREDRV object.
    *
    * The name is used in human-readable statistics banners such as
    * ``STATISTICS SUMMARY for <name>:``. Passing ``NULL`` or an empty string
    * clears the current label.
    *
    * Embedded applications can use this to identify multiple concurrent
    * ``HYPREDRV_t`` objects without modifying an authoritative YAML file.
    * When ``general.name`` is present in the parsed YAML input, it populates
    * the same label automatically.
    *
    * @param hypredrv The HYPREDRV_t object to label.
    * @param name The optional display name, or ``NULL`` to clear it.
    *
    * @return Returns an error code with 0 indicating success.
    */

   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_ObjectSetName(HYPREDRV_t  hypredrv,
                                                          const char *name);

   /**
    * @brief Retrieve the warmup setting from a HYPREDRV object.
    *
    * @param hypredrv The HYPREDRV_t object to query.
    * @param[out] warmup Set to the warmup flag value (0 or 1) on success.
    *
    * @return Returns an error code with 0 indicating success.
    *
    * Example Usage:
    * @code
    *    int warmup = 0;
    *    HYPREDRV_SAFE_CALL(HYPREDRV_InputArgsGetWarmup(hypredrv, &warmup));
    * @endcode
    */

   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_InputArgsGetWarmup(HYPREDRV_t hypredrv,
                                                               int       *warmup);

   /**
    * @brief Retrieve the number of repetitions from a HYPREDRV object.
    *
    * @param hypredrv The HYPREDRV_t object to query.
    * @param[out] num_reps Set to the number of repetitions on success.
    *
    * @return Returns an error code with 0 indicating success.
    *
    * Example Usage:
    * @code
    *    int num_reps = 0;
    *    HYPREDRV_SAFE_CALL(HYPREDRV_InputArgsGetNumRepetitions(hypredrv, &num_reps));
    * @endcode
    */

   HYPREDRV_EXPORT_SYMBOL uint32_t
   HYPREDRV_InputArgsGetNumRepetitions(HYPREDRV_t hypredrv, int *num_reps);

   /**
    * @brief Retrieve the number of linear systems from a HYPREDRV object.
    *
    * @param hypredrv The HYPREDRV_t object to query.
    * @param[out] num_ls Set to the number of linear systems on success.
    *
    * @return Returns an error code with 0 indicating success.
    *
    * Example Usage:
    * @code
    *    int num_ls = 0;
    *    HYPREDRV_SAFE_CALL(HYPREDRV_InputArgsGetNumLinearSystems(hypredrv, &num_ls));
    * @endcode
    */

   HYPREDRV_EXPORT_SYMBOL uint32_t
   HYPREDRV_InputArgsGetNumLinearSystems(HYPREDRV_t hypredrv, int *num_ls);

   /**
    * @brief Retrieve the number of preconditioner variants from a HYPREDRV object.
    *
    * @param hypredrv The HYPREDRV_t object to query.
    * @param[out] num_variants Set to the number of preconditioner variants on success.
    *
    * @return Returns an error code with 0 indicating success.
    *
    * Example Usage:
    * @code
    *    int num_variants = 0;
    *    HYPREDRV_SAFE_CALL(HYPREDRV_InputArgsGetNumPreconVariants(hypredrv,
    * &num_variants));
    * @endcode
    */

   HYPREDRV_EXPORT_SYMBOL uint32_t
   HYPREDRV_InputArgsGetNumPreconVariants(HYPREDRV_t hypredrv, int *num_variants);

   /**
    * @brief Set the active preconditioner variant index.
    *
    * This function sets which preconditioner variant should be used for subsequent
    * solve operations. It updates the active variant and destroys any existing
    * solver/preconditioner objects to avoid stale configuration.
    *
    * @param hypredrv The HYPREDRV_t object for which to set the variant.
    * @param variant_idx The zero-based index of the variant to activate (must be in
    * range [0, num_variants-1]).
    *
    * @return Returns an error code with 0 indicating success. Any non-zero value
    * indicates a failure (e.g., invalid index, object not initialized).
    *
    * Example Usage:
    * @code
    *    HYPREDRV_t *hypredrv;
    *    // ... (hypredrv is created, and its components are set) ...
    *    HYPREDRV_SAFE_CALL(HYPREDRV_InputArgsSetPreconVariant(hypredrv, 1));
    * @endcode
    */

   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_InputArgsSetPreconVariant(HYPREDRV_t hypredrv,
                                                                      int variant_idx);

   /**
    * @brief Configure the active preconditioner variant using a predefined preset.
    *
    * This function allows library users to select a named preset (e.g., "poisson",
    * "elasticity-2D") without calling HYPREDRV_InputArgsParse. It ensures that the
    * input argument structure exists and that solver/preconditioner defaults are
    * initialized, then applies the preset to the active variant only.
    *
    * @param hypredrv The HYPREDRV_t object for which to set the preset.
    * @param preset The preset name string.
    *
    * @return Returns an error code with 0 indicating success. Any non-zero value
    * indicates a failure (e.g., invalid preset, object not initialized).
    *
    * Example Usage:
    * @code
    *    HYPREDRV_t *hypredrv;
    *    // ... (hypredrv is created) ...
    *    HYPREDRV_SAFE_CALL(HYPREDRV_InputArgsSetPreconPreset(hypredrv, "poisson"));
    * @endcode
    */

   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_InputArgsSetPreconPreset(HYPREDRV_t  hypredrv,
                                                                     const char *preset);

   /**
    * @brief Configure the solver using a predefined preset name.
    *
    * This function allows library users to select a named solver (e.g., "pcg",
    * "gmres", "fgmres", "bicgstab") without calling HYPREDRV_InputArgsParse. It
    * ensures that the input argument structure exists, then sets the solver method
    * and initializes its defaults.
    *
    * @param hypredrv The HYPREDRV_t object for which to set the solver.
    * @param preset The solver name string (e.g., "pcg", "gmres").
    *
    * @return Returns an error code with 0 indicating success. Any non-zero value
    * indicates a failure (e.g., invalid solver name, object not initialized).
    *
    * Example Usage:
    * @code
    *    HYPREDRV_t *hypredrv;
    *    // ... (hypredrv is created) ...
    *    HYPREDRV_SAFE_CALL(HYPREDRV_InputArgsSetSolverPreset(hypredrv, "pcg"));
    * @endcode
    */

   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_InputArgsSetSolverPreset(HYPREDRV_t  hypredrv,
                                                                     const char *preset);

   /**
    * @brief Register a custom preconditioner preset.
    *
    * Registers a named preset from a YAML text string, making it available to
    * HYPREDRV_InputArgsSetPreconPreset() and the YAML `preset:` key. The preset
    * text must be a valid YAML snippet that expands under `preconditioner:`.
    *
    * @param name      Unique preset name (case-insensitive; '-' and '_' are equivalent).
    * @param yaml_text YAML snippet string (e.g. "amg:\n  max_iter: 1").
    * @param help      Short description shown in preset listings (may be NULL).
    *
    * @return 0 on success, non-zero on failure (duplicate name, NULL args, alloc
    * failure).
    *
    * Example Usage:
    * @code
    *    HYPREDRV_SAFE_CALL(HYPREDRV_PreconPresetRegister(
    *       "my_amg", "amg:\n  max_iter: 1\n  tolerance: 1e-8", "Custom AMG preset"));
    * @endcode
    */

   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_PreconPresetRegister(const char *name,
                                                                 const char *yaml_text,
                                                                 const char *help);

   /**
    * @brief Build the linear system (matrix, RHS, and LHS) according to the YAML input
    * passed to the HYPREDRV object.
    *
    * The matrix is read from file. Vectors might be read from file or built according to
    * predetermined options passed via YAML.
    *
    * @param hypredrv The HYPREDRV_t object for which the linear system matrix is to be
    * built.
    *
    * @return Returns an error code with 0 indicating success. Any non-zero value
    * indicates a failure, and the error code can be further described using
    * HYPREDRV_ErrorCodeDescribe(error_code).
    *
    * @note It's the caller's responsibility to ensure that the hypredrv parameter is a
    * valid pointer to an initialized HYPREDRV_t object. Passing a NULL or uninitialized
    * object will result in an error.
    *
    * Example Usage:
    * @code
    *    HYPREDRV_t *hypredrv;
    *    // ... (hypredrv is created, and its components are initialized) ...
    *    HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemBuild(hypredrv));
    * @endcode
    */

   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_LinearSystemBuild(HYPREDRV_t hypredrv);

   /**
    * @brief Read the linear system matrix from file for a HYPREDRV object.
    *
    * This function is responsible for reading the matrix data of a linear system
    * associated with a HYPREDRV object. It performs the reading process given input
    * arguments related to the linear system, and uses a pointer to store the read matrix.
    * After reading the matrix, it retrieves and returns the error code generated during
    * the process.
    *
    * @param hypredrv The HYPREDRV_t object for which the linear system matrix is to be
    * read.
    *
    * @return Returns an error code with 0 indicating success. Any non-zero value
    * indicates a failure, and the error code can be further described using
    * HYPREDRV_ErrorCodeDescribe(error_code).
    *
    * @note It's the caller's responsibility to ensure that the hypredrv parameter is a
    * valid pointer to an initialized HYPREDRV_t object. Passing a NULL or uninitialized
    * object will result in an error.
    *
    * Example Usage:
    * @code
    *    HYPREDRV_t *hypredrv;
    *    // ... (hypredrv is created, and its components are initialized) ...
    *    HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemReadMatrix(hypredrv));
    * @endcode
    */

   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_LinearSystemReadMatrix(HYPREDRV_t hypredrv);

   /**
    * @brief Set the linear system matrix for a HYPREDRV object.
    *
    * This function is responsible for setting the matrix data of a linear system
    * associated with a HYPREDRV object.
    *
    * @param hypredrv The HYPREDRV_t object for which the linear system matrix is to be
    * associated.
    *
    * @param mat_A The HYPRE_Matrix object for the linear system matrix.
    *
    * @return Returns an error code with 0 indicating success. Any non-zero value
    * indicates a failure, and the error code can be further described using
    * HYPREDRV_ErrorCodeDescribe(error_code).
    *
    * @note It's the caller's responsibility to ensure that the hypredrv parameter is a
    * valid pointer to an initialized HYPREDRV_t object. Passing a NULL or uninitialized
    * object will result in an error.
    *
    * Example Usage:
    * @code
    *    HYPREDRV_t *hypredrv;
    *    // ... (hypredrv is created, and its components are initialized) ...
    *    HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetMatrix(hypredrv, mat_A));
    * @endcode
    */

   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_LinearSystemSetMatrix(HYPREDRV_t   hypredrv,
                                                                  HYPRE_Matrix mat_A);

   /**
    * @brief Set the linear system right-hand side (RHS) vector for a HYPREDRV object.
    *
    * @param hypredrv The HYPREDRV_t object for which the RHS vector of the linear system
    * is to be set.
    *
    * @param vec The HYPRE_Vector vector object representing the RHS of the linear system
    * (NULL if set from input file).
    *
    * @return Returns an error code with 0 indicating success. Any non-zero value
    * indicates a failure, and the error code can be further described using
    * HYPREDRV_ErrorCodeDescribe(error_code).
    *
    * @note It's the caller's responsibility to ensure that the hypredrv parameter is a
    * valid pointer to an initialized HYPREDRV_t object. Passing a NULL or uninitialized
    * object will result in an error.
    *
    * Example Usage:
    * @code
    *    HYPREDRV_t *hypredrv;
    *    // ... (hypredrv is created, and its components are initialized) ...
    *    HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetRHS(hypredrv, vec));
    * @endcode
    */

   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_LinearSystemSetRHS(HYPREDRV_t   hypredrv,
                                                               HYPRE_Vector vec);

   /**
    * @brief Set the initial guess for the solution vector of the linear system for a
    * HYPREDRV object.
    *
    * This function is responsible for setting the initial guess for the solution vector
    * of a linear system associated with a HYPREDRV object.
    *
    * @param hypredrv The HYPREDRV_t object for which the initial guess of the solution
    * vector of the linear system is to be set.
    *
    * @param vec Optional initial-guess vector. If NULL, the initial guess is built from
    * input/default behavior (`x0_filename`/`init_guess_mode`). If non-NULL, this vector
    * is used as `x0`.
    *
    * @return Returns an error code with 0 indicating success. Any non-zero value
    * indicates a failure, and the error code can be further described using
    * HYPREDRV_ErrorCodeDescribe(error_code).
    *
    * @note It's the caller's responsibility to ensure that the hypredrv parameter is a
    * valid pointer to an initialized HYPREDRV_t object. Passing a NULL or uninitialized
    * object will result in an error.
    *
    * @note Ownership for non-NULL \e vec:
    * - library mode ON (`HYPREDRV_SetLibraryMode`): borrowed (not destroyed by HYPREDRV).
    * - library mode OFF: ownership is transferred to HYPREDRV.
    *
    * Example Usage:
    * @code
    *    HYPREDRV_t *hypredrv;
    *    // ... (hypredrv is created, and its components are initialized) ...
    *    HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetInitialGuess(hypredrv, NULL));
    * @endcode
    */

   HYPREDRV_EXPORT_SYMBOL uint32_t
   HYPREDRV_LinearSystemSetInitialGuess(HYPREDRV_t hypredrv, HYPRE_Vector vec);

   /**
    * @brief Set the solution vector of the linear system for a HYPREDRV object.
    *
    * Allows the caller to supply a pre-built HYPRE_Vector to use as the working
    * solution buffer (the vector the linear solver writes its result into).
    *
    * @param hypredrv The HYPREDRV_t object.
    * @param vec      The HYPRE_Vector to use as the solution buffer. If NULL, HYPREDRV
    *                 allocates an internal buffer (vec_b must already be set). If
    *                 non-NULL, the vector is borrowed - HYPREDRV will never destroy it.
    *
    * @return 0 on success; nonzero error code otherwise. Returns an error when @p vec
    *         is NULL and the RHS vector has not been set yet.
    *
    * @note Typical use: provide your own HYPRE_ParVector allocation before calling
    *       HYPREDRV_LinearSolverApply() so the result is written directly into your
    *       buffer. You can then retrieve the same pointer via
    *       HYPREDRV_LinearSystemGetSolution().
    *
    * Example Usage:
    * @code
    *    HYPRE_IJVector my_x; // previously created, same size as b
    *    HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetSolution(hypredrv,
    *                                                        (HYPRE_Vector)my_x));
    *    HYPREDRV_SAFE_CALL(HYPREDRV_LinearSolverApply(hypredrv));
    *    // my_x now contains the solution.
    * @endcode
    */

   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_LinearSystemSetSolution(HYPREDRV_t   hypredrv,
                                                                    HYPRE_Vector vec);

   /**
    * @brief Set or refresh the reference solution vector used by GMRES tagged
    * residual/error reporting.
    *
    * This function updates the internal reference solution vector associated with the
    * current linear system. The reference solution is consumed by GMRES tagged output
    * modes (for example, when printing tagged residual or tagged error histories).
    *
    * @param hypredrv The HYPREDRV_t object for which the reference solution vector is to
    * be set or refreshed.
    *
    * @param vec Optional reference-solution vector. If NULL, the reference solution is
    * updated using the existing file/default rules (`xref_filename`/`xref_basename`).
    * If non-NULL, this vector is used directly as the reference solution.
    *
    * @return Returns an error code with 0 indicating success. Any non-zero value
    * indicates a failure, and the error code can be further described using
    * HYPREDRV_ErrorCodeDescribe(error_code).
    *
    * @note On hypre versions older than v3.0.0, this function is a no-op for GMRES tagged
    * residual/error internals.
    *
    * @note Ownership for non-NULL \e vec:
    * - library mode ON (`HYPREDRV_SetLibraryMode`): borrowed (not destroyed by HYPREDRV).
    * - library mode OFF: ownership is transferred to HYPREDRV.
    *
    * This vector can be tagged and scaled/unscaled internally during solve setup/apply.
    *
    * Example Usage:
    * @code
    *    HYPREDRV_t *hypredrv;
    *    // ... (hypredrv is created, and its components are initialized) ...
    *    HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetReferenceSolution(hypredrv, NULL));
    * @endcode
    */
   HYPREDRV_EXPORT_SYMBOL uint32_t
   HYPREDRV_LinearSystemSetReferenceSolution(HYPREDRV_t hypredrv, HYPRE_Vector vec);

   /**
    * @brief Reset the initial guess of the solution vector for a HYPREDRV object to its
    * original state as computed with \e HYPREDRV_LinearSystemSetInitialGuess
    *
    * @param hypredrv The HYPREDRV_t object for which the initial guess of the solution
    * vector is to be reset.
    *
    * @return Returns an error code with 0 indicating success. Any non-zero value
    * indicates a failure, and the error code can be further described using
    * HYPREDRV_ErrorCodeDescribe(error_code).
    *
    * @note It's the caller's responsibility to ensure that the hypredrv parameter is a
    * valid pointer to an initialized HYPREDRV_t object. Passing a NULL or uninitialized
    * object will result in an error.
    *
    * Example Usage:
    * @code
    *    HYPREDRV_t *hypredrv;
    *    // ... (hypredrv is created, and its components are initialized) ...
    *    HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemResetInitialGuess(hypredrv));
    * @endcode
    */

   HYPREDRV_EXPORT_SYMBOL uint32_t
   HYPREDRV_LinearSystemResetInitialGuess(HYPREDRV_t hypredrv);

   /**
    * @brief Set the matrix that is used to compute the preconditioner of a HYPREDRV
    * object.
    *
    * By default, the preconditioning matrix is the same as the linear system matrix.
    * However, it is also possible to use an arbitrary matrix set by the user, e.g., a
    * filtered version of the linear system matrix.
    *
    * @param hypredrv The HYPREDRV_t object for which the preconditioner matrix is to be
    * set.
    *
    * @param mat Optional preconditioner matrix. If NULL, the preconditioner matrix is
    * resolved by current file/default behavior (`precmat_filename`/`precmat_basename`
    * with fallback to the system matrix). If non-NULL, this matrix is used directly.
    *
    * @return Returns an error code with 0 indicating success. Any non-zero value
    * indicates a failure, and the error code can be further described using
    * HYPREDRV_ErrorCodeDescribe(error_code).
    *
    * @note It's the caller's responsibility to ensure that the hypredrv parameter is a
    * valid pointer to an initialized HYPREDRV_t object. Passing a NULL or uninitialized
    * object will result in an error.
    *
    * @note Ownership for non-NULL \e mat:
    * - library mode ON (`HYPREDRV_SetLibraryMode`): borrowed (not destroyed by HYPREDRV).
    * - library mode OFF: ownership is transferred to HYPREDRV.
    *
    * This matrix may be scaled/unscaled internally during solve setup/apply.
    *
    * Example Usage:
    * @code
    *    HYPREDRV_t *hypredrv;
    *    // ... (hypredrv is created, and its components are initialized) ...
    *    HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetPrecMatrix(hypredrv, NULL));
    * @endcode
    */

   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_LinearSystemSetPrecMatrix(HYPREDRV_t hypredrv,
                                                                      HYPRE_Matrix mat);

   /**
    * @brief Set the degree of freedom (DOF) map for the linear system of a HYPREDRV
    * object.
    *
    * @param hypredrv The HYPREDRV_t object for which the DOF map of the linear system is
    * to be set.
    *
    * @param size The local size (current rank) of the dofmap array.
    *
    * @param dofmap The array containing the mapping codes for the degrees of freedom
    *
    * @return Returns an error code with 0 indicating success. Any non-zero value
    * indicates a failure, and the error code can be further described using
    * HYPREDRV_ErrorCodeDescribe(error_code).
    *
    * @note It's the caller's responsibility to ensure that the hypredrv parameter is a
    * valid pointer to an initialized HYPREDRV_t object. Passing a NULL or uninitialized
    * object will result in an error.
    *
    * Example Usage:
    * @code
    *    HYPREDRV_t *hypredrv;
    *    int size = ... // Size of dofmap is set
    *    int *dofmap = ... // dofmap array is set
    *    // ... (hypredrv is created, and its components are initialized) ...
    *    HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetDofmap(hypredrv, size, dofmap));
    * @endcode
    */

   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_LinearSystemSetDofmap(HYPREDRV_t hypredrv,
                                                                  int        size,
                                                                  const int *dofmap);

   /**
    * @brief Set an interleaved degree of freedom (DOF) map for the linear system of a
    * HYPREDRV object.
    *
    * @param hypredrv The HYPREDRV_t object for which the DOF map of the linear system is
    * to be set.
    *
    * @param num_local_blocks The local (owned by the current rank) number of blocks
    * (cells/nodes) containing a set of num_dof_types degrees of freedom.
    *
    * @param num_dof_types The number of degree of freedom types.
    *
    * @return Returns an error code with 0 indicating success. Any non-zero value
    * indicates a failure, and the error code can be further described using
    * HYPREDRV_ErrorCodeDescribe(error_code).
    *
    * @note It's the caller's responsibility to ensure that the hypredrv parameter is a
    * valid pointer to an initialized HYPREDRV_t object. Passing a NULL or uninitialized
    * object will result in an error.
    *
    * Example Usage:
    * @code
    *    HYPREDRV_t *hypredrv;
    *    int num_dof_types = ... // Number of degree of freedom types
    *    int num_local_nodes = ... // Number of local (current MPI rank) nodes
    *    // ... (hypredrv is created, and its components are initialized) ...
    *    HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetInterleavedDofmap(hypredrv,
    * num_local_dofs, num_dof_types));
    * @endcode
    */

   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_LinearSystemSetInterleavedDofmap(
      HYPREDRV_t hypredrv, int num_local_blocks, int num_dof_types);

   /**
    * @brief Set a contiguous degree of freedom (DOF) map for the linear system of a
    * HYPREDRV object.
    *
    * @param hypredrv The HYPREDRV_t object for which the DOF map of the linear system is
    * to be set.
    *
    * @param num_local_blocks The local (owned by the current rank) number of blocks
    * (cells/nodes) containing a set of num_dof_types degrees of freedom.
    *
    * @param num_dof_types The number of degree of freedom types.
    *
    * @return Returns an error code with 0 indicating success. Any non-zero value
    * indicates a failure, and the error code can be further described using
    * HYPREDRV_ErrorCodeDescribe(error_code).
    *
    * @note It's the caller's responsibility to ensure that the hypredrv parameter is a
    * valid pointer to an initialized HYPREDRV_t object. Passing a NULL or uninitialized
    * object will result in an error.
    *
    * Example Usage:
    * @code
    *    HYPREDRV_t *hypredrv;
    *    int num_dof_types = ... // Number of degree of freedom types
    *    int num_local_nodes = ... // Number of local (current MPI rank) nodes
    *    // ... (hypredrv is created, and its components are initialized) ...
    *    HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetContiguousDofmap(hypredrv,
    * num_local_blocks, num_dof_types));
    * @endcode
    */

   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_LinearSystemSetContiguousDofmap(
      HYPREDRV_t hypredrv, int num_local_blocks, int num_dof_types);

   /**
    * @brief Read the degree of freedom (DOF) map for the linear system of a HYPREDRV
    * object.
    *
    * @param hypredrv The HYPREDRV_t object for which the DOF map of the linear system is
    * to be read.
    *
    * @return Returns an error code with 0 indicating success. Any non-zero value
    * indicates a failure, and the error code can be further described using
    * HYPREDRV_ErrorCodeDescribe(error_code).
    *
    * @note It's the caller's responsibility to ensure that the hypredrv parameter is a
    * valid pointer to an initialized HYPREDRV_t object. Passing a NULL or uninitialized
    * object will result in an error.
    *
    * Example Usage:
    * @code
    *    HYPREDRV_t *hypredrv;
    *    // ... (hypredrv is created, and its components are initialized) ...
    *    HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemReadDofmap(hypredrv));
    * @endcode
    */

   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_LinearSystemReadDofmap(HYPREDRV_t hypredrv);

   /**
    * @brief Print the current degree of freedom (DOF) map of the linear system to file.
    *
    * Writes the locally owned DOF map entries to a text file. In MPI runs with more than
    * one process, the rank id is appended to the provided filename as a numeric suffix:
    * ``filename.rank``, zero-padded to 5 digits. In single-process runs, the file is
    * written exactly to ``filename``.
    *
    * File format (ASCII):
    *   line 1: number of entries
    *   lines : dofmap entries (space-separated), arbitrary line breaks
    *
    * @param hypredrv A valid HYPREDRV_t object containing the DOF map.
    * @param filename The output filename (prefix). If running with multiple MPI ranks,
    *                 the function appends ".%05d" with the rank id.
    * @return 0 on success; nonzero error code otherwise.
    */

   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_LinearSystemPrintDofmap(HYPREDRV_t  hypredrv,
                                                                    const char *filename);

   /**
    * @brief Print the linear system (matrix and RHS) and the DOF map (if present).
    *
    * This convenience routine dumps:
    *  - Matrix A via HYPRE_IJMatrixPrint to an ASCII file (basename + ".out").
    *  - RHS b via HYPRE_IJVectorPrint to an ASCII file (basename + ".out").
    *  - DOF map via HYPREDRV_LinearSystemPrintDofmap if it exists (basename + ".out").
    *
    * Basenames are taken from the input arguments if set; otherwise, defaults are used:
    *  - Matrix: ls.matrix_basename or "IJ.out.A"
    *  - RHS:    ls.rhs_basename    or "IJ.out.b"
    *  - DOFMap: ls.dofmap_basename or "dofmap"
    *
    * @param hypredrv A valid HYPREDRV_t object with matrix/vector set.
    * @return 0 on success; nonzero error code otherwise.
    */

   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_LinearSystemPrint(HYPREDRV_t hypredrv);

   /**
    * @brief Set near-nullspace modes for the current linear system (e.g., elastic RBMs).
    *
    * The input array must contain num_components contiguous blocks (component-major/SoA),
    * each of length num_entries, stored back-to-back (not interleaved). The data is
    * copied into internal storage owned and freed by HYPREDRV; the caller retains
    * ownership of its input buffer and is responsible for freeing it after the call
    * returns.
    *
    * Typical use for linear elasticity: provide 6 rigid-body modes with num_entries = 3 *
    * num_local_nodes.
    *
    * @param hypredrv        The HYPREDRV_t object.
    * @param num_entries     Number of local entries.
    * @param num_components  Number of components per entry (modes).
    * @param values          Pointer to contiguous data (length = num_entries *
    * num_components).
    *
    * @return 0 on success; nonzero error code otherwise.
    */

   HYPREDRV_EXPORT_SYMBOL uint32_t
   HYPREDRV_LinearSystemSetNearNullSpace(HYPREDRV_t hypredrv, int num_entries,
                                         int num_components, const HYPRE_Complex *values);

   /**
    * @brief Retrieves the solution values from the linear system of a HYPREDRV object.
    *
    * This function provides access to the internal pointer where the solution vector of
    * the linear system associated with the given HYPREDRV_t object is stored. It does not
    * copy the solution values; instead, it assigns the internal pointer to the
    * user-provided pointer @p sol_data.
    *
    * @param hypredrv A valid HYPREDRV_t object from which the internal solution pointer
    * is to be retrieved.
    * @param sol_data A pointer to a HYPRE_Complex pointer, which will be set to point to
    * the internal solution data array. The user must not free or modify the internal
    * array.
    *
    * @return Returns an error code, with 0 indicating success. If the @p hypredrv
    * parameter is invalid (e.g., NULL or uninitialized), an error code is returned, and
    * the error can be further described using HYPREDRV_ErrorCodeDescribe(error_code).
    *
    * Example Usage:
    * @code
    *    HYPREDRV_t *hypredrv;
    *    HYPRE_Complex *sol_data = NULL;
    *    // Ensure hypredrv is initialized and the system is solved
    *    HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemGetSolutionValues(hypredrv, &sol_data));
    *    // Use sol_data but do not free or modify it.
    * @endcode
    */

   HYPREDRV_EXPORT_SYMBOL uint32_t
   HYPREDRV_LinearSystemGetSolutionValues(HYPREDRV_t hypredrv, HYPRE_Complex **sol_data);

   /**
    * @brief Computes a norm of the solution vector from the linear system.
    *
    * This function computes the specified norm of the solution vector. Valid norm types
    * are:
    * - "L1" or "l1": L1 norm (sum of absolute values)
    * - "L2" or "l2": L2 norm (Euclidean norm, sqrt of sum of squares)
    * - "inf", "Linf", or "linf": Infinity norm (maximum absolute value)
    *
    * @param hypredrv A valid HYPREDRV_t object with a solved linear system.
    * @param norm_type String specifying the norm type ("L1", "L2", or "inf"/"Linf").
    * @param norm Pointer to a double where the computed norm will be stored.
    *
    * @return Returns an error code, with 0 indicating success. If the norm_type is
    * invalid, the norm will be set to -1.0.
    *
    * Example Usage:
    * @code
    *    double norm_inf;
    *    HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemGetSolutionNorm(hypredrv, "inf",
    * &norm_inf));
    * @endcode
    */

   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_LinearSystemGetSolutionNorm(
      HYPREDRV_t hypredrv, const char *norm_type, double *norm);

   /**
    * @brief Retrieve the solution vector from a HYPREDRV object.
    *
    * Returns the HYPRE_Vector that the linear solver wrote its result into. This is
    * useful for passing the solution directly to another HYPRE routine without copying
    * the underlying data.
    *
    * @param hypredrv A valid HYPREDRV_t object with a solved linear system.
    * @param vec      A pointer to a HYPRE_Vector, set to the internal solution vector
    *                 on success. The returned pointer is owned by HYPREDRV (unless a
    *                 user-supplied buffer was injected via
    *                 HYPREDRV_LinearSystemSetSolution()); the caller must not free it.
    *
    * @return 0 on success; nonzero if @p hypredrv is invalid or the solution vector has
    *         not been allocated yet (i.e., before the first solve or SetSolution call).
    *
    * Example Usage:
    * @code
    *    HYPRE_Vector x = NULL;
    *    HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemGetSolution(hypredrv, &x));
    *    // Pass x to another HYPRE routine; do not free it.
    * @endcode
    */

   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_LinearSystemGetSolution(HYPREDRV_t hypredrv,
                                                                    HYPRE_Vector *vec);

   /**
    * @brief Retrieves the right-hand side values from the linear system of a HYPREDRV
    * object.
    *
    * This function provides access to the internal pointer where the right-hand side
    * vector of the linear system associated with the given HYPREDRV_t object is stored.
    * It does not copy the solution values; instead, it assigns the internal pointer to
    * the user-provided pointer @p rhs_data.
    *
    * @param hypredrv A valid HYPREDRV_t object from which the internal solution pointer
    * is to be retrieved.
    * @param rhs_data A pointer to a HYPRE_Complex pointer, which will be set to point to
    * the internal right-hand side data array. The user must not free or modify the
    * internal array.
    *
    * @return Returns an error code, with 0 indicating success. If the @p obj parameter is
    * invalid (e.g., NULL or uninitialized), an error code is returned, and the error can
    * be further described using HYPREDRV_ErrorCodeDescribe(error_code).
    *
    * Example Usage:
    * @code
    *    HYPREDRV_t *hypredrv;
    *    HYPRE_Complex *rhs_data = NULL;
    *    // Ensure hypredrv is initialized and the system is solved
    *    HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemGetRHSValues(hypredrv, &rhs_data));
    *    // Use rhs_data but do not free or modify it.
    * @endcode
    */

   HYPREDRV_EXPORT_SYMBOL uint32_t
   HYPREDRV_LinearSystemGetRHSValues(HYPREDRV_t hypredrv, HYPRE_Complex **rhs_data);

   /**
    * @brief Retrieve the right-hand side vector from a HYPREDRV object.
    *
    * Returns the HYPRE_Vector for the RHS (vec_b). This is useful for passing the
    * vector directly to another HYPRE routine without copying the underlying data.
    *
    * @param hypredrv A valid HYPREDRV_t object with the RHS set.
    * @param vec      A pointer to a HYPRE_Vector, set to the internal RHS vector on
    *                 success. The returned pointer is owned by HYPREDRV; the caller
    *                 must not free it.
    *
    * @return 0 on success; nonzero if @p hypredrv is invalid or the RHS has not been
    *         set yet.
    *
    * Example Usage:
    * @code
    *    HYPRE_Vector b = NULL;
    *    HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemGetRHS(hypredrv, &b));
    *    // Pass b to another HYPRE routine; do not free it.
    * @endcode
    */

   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_LinearSystemGetRHS(HYPREDRV_t    hypredrv,
                                                               HYPRE_Vector *vec);

   /**
    * @brief Retrieve the system matrix from a HYPREDRV object.
    *
    * This function provides access to the internal system matrix (mat_A) associated
    * with the given HYPREDRV_t object. The returned pointer is owned by HYPREDRV;
    * the caller must not free it.
    *
    * @param hypredrv A valid HYPREDRV_t object with the matrix set.
    * @param mat      A pointer to a HYPRE_Matrix, which will be set to the internal
    *                 system matrix. The user must not free or destroy this object.
    *
    * @return Returns an error code, with 0 indicating success. Returns a non-zero
    * code if @p hypredrv is invalid or the matrix has not been set yet.
    *
    * Example Usage:
    * @code
    *    HYPRE_Matrix A = NULL;
    *    HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemGetMatrix(hypredrv, &A));
    *    // Use A for inspection but do not free it.
    * @endcode
    */

   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_LinearSystemGetMatrix(HYPREDRV_t    hypredrv,
                                                                  HYPRE_Matrix *mat);

   /**
    * @brief Set state vectors for time-stepping or multi-state applications.
    *
    * This function initializes the state vector management system, which is useful for
    * time-stepping applications where multiple solution states need to be maintained
    * (e.g., previous time step, current time step). The state vectors are managed
    * internally using a circular indexing scheme.
    *
    * @param hypredrv The HYPREDRV_t object for which state vectors are to be set.
    * @param nstates The number of state vectors to manage.
    * @param vecs An array of HYPRE_Vector objects (HYPRE_IJVector) representing the
    *             state vectors. The array must contain at least nstates valid vectors.
    *
    * @return Returns an error code with 0 indicating success. Any non-zero value
    * indicates a failure, and the error code can be further described using
    * HYPREDRV_ErrorCodeDescribe(error_code).
    *
    * @note The state vectors must be created and initialized by the caller before
    * calling this function. Typically, these are HYPRE_IJVector objects created with
    * the same row range as the linear system.
    *
    * @note This function must be called before using other state vector functions
    * such as HYPREDRV_StateVectorGetValues, HYPREDRV_StateVectorCopy, etc.
    *
    * Example Usage:
    * @code
    *    HYPREDRV_t hypredrv;
    *    HYPRE_IJVector vec_s[2];
    *    // ... (create and initialize vec_s[0] and vec_s[1]) ...
    *    HYPREDRV_SAFE_CALL(HYPREDRV_StateVectorSet(hypredrv, 2, vec_s));
    * @endcode
    */

   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_StateVectorSet(HYPREDRV_t      hypredrv,
                                                           int             nstates,
                                                           HYPRE_IJVector *vecs);

   /**
    * @brief Retrieve a pointer to the data array of a state vector.
    *
    * This function provides direct access to the underlying data array of a state
    * vector, allowing efficient read/write access without copying. The returned
    * pointer points to the local portion of the distributed vector on the current
    * MPI rank.
    *
    * @param hypredrv The HYPREDRV_t object containing the state vectors.
    * @param index The logical index of the state vector (0-based, relative to the
    *              current state mapping).
    * @param data_ptr A pointer to a HYPRE_Complex pointer, which will be set to
    *                 point to the local data array. The user must not free this
    *                 pointer, but may read and write to the array.
    *
    * @return Returns an error code with 0 indicating success. Any non-zero value
    * indicates a failure, and the error code can be further described using
    * HYPREDRV_ErrorCodeDescribe(error_code).
    *
    * @note The returned pointer is valid only as long as the state vector exists
    * and has not been modified by operations that may reallocate the underlying
    * storage.
    *
    * @note The data layout is contiguous and follows the DOF ordering of the
    * linear system (e.g., interleaved u, v, p for Navier-Stokes).
    *
    * Example Usage:
    * @code
    *    HYPREDRV_t hypredrv;
    *    HYPRE_Complex *u_old = NULL;
    *    // Get pointer to state vector at index 1 (e.g., previous time step)
    *    HYPREDRV_SAFE_CALL(HYPREDRV_StateVectorGetValues(hypredrv, 1, &u_old));
    *    // Now u_old points to the local data array
    * @endcode
    */

   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_StateVectorGetValues(
      HYPREDRV_t hypredrv, int index, HYPRE_Complex **data_ptr);

   /**
    * @brief Copy one state vector to another.
    *
    * This function copies the contents of one state vector to another. It is
    * commonly used in time-stepping applications to initialize the new time step
    * with the solution from the previous time step.
    *
    * @param hypredrv The HYPREDRV_t object containing the state vectors.
    * @param index_in The logical index of the source state vector (0-based).
    * @param index_out The logical index of the destination state vector (0-based).
    *
    * @return Returns an error code with 0 indicating success. Any non-zero value
    * indicates a failure, and the error code can be further described using
    * HYPREDRV_ErrorCodeDescribe(error_code).
    *
    * @note Both state vectors must have been set via HYPREDRV_StateVectorSet and
    * must have compatible sizes (same row ranges).
    *
    * Example Usage:
    * @code
    *    HYPREDRV_t hypredrv;
    *    // Copy state 1 (previous time step) to state 0 (current time step)
    *    HYPREDRV_SAFE_CALL(HYPREDRV_StateVectorCopy(hypredrv, 1, 0));
    * @endcode
    */

   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_StateVectorCopy(HYPREDRV_t hypredrv,
                                                            int index_in, int index_out);

   /**
    * @brief Cycle through state vector indices (advance state mapping).
    *
    * This function advances the internal state mapping by one position in a
    * circular manner. After calling this function, the logical indices (0, 1, ...)
    * now refer to different physical state vectors. This is useful for time-stepping
    * applications where state vectors are reused in a circular buffer pattern.
    *
    * @param hypredrv The HYPREDRV_t object containing the state vectors.
    *
    * @return Returns an error code with 0 indicating success. Any non-zero value
    * indicates a failure, and the error code can be further described using
    * HYPREDRV_ErrorCodeDescribe(error_code).
    *
    * @note This function modifies the internal state mapping but does not copy
    * any data. It is typically called at the end of a time step to prepare for the
    * next iteration.
    *
    * @note After calling this function, the logical index 0 will refer to what was
    * previously index 1, index 1 will refer to what was previously index 2, etc.,
    * wrapping around for the last index.
    *
    * Example Usage:
    * @code
    *    HYPREDRV_t hypredrv;
    *    // At end of time step, cycle states so next iteration uses updated mapping
    *    HYPREDRV_SAFE_CALL(HYPREDRV_StateVectorUpdateAll(hypredrv));
    * @endcode
    */

   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_StateVectorUpdateAll(HYPREDRV_t hypredrv);

   /**
    * @brief Apply the linear solver correction to a state vector.
    *
    * This function adds the solution increment from the linear solver (stored in
    * vec_x) to the state vector at logical index @p state_idx. This implements
    * the Newton update: U^{k+1} = U^k + ΔU, where ΔU is the solution from the
    * linear system J ΔU = -R.
    *
    * Pass @p state_idx = 0 to apply the correction to the current (most recent)
    * state, which is the typical usage after HYPREDRV_StateVectorUpdateAll() has
    * cycled the indices so that index 0 refers to the new current state.
    *
    * @param hypredrv  The HYPREDRV_t object containing the state vectors and the
    *                  linear solver solution (vec_x).
    * @param state_idx Zero-based logical index of the state vector to update.
    *                  Must be in [0, nstates).
    *
    * @return Returns an error code with 0 indicating success. Any non-zero value
    * indicates a failure, and the error code can be further described using
    * HYPREDRV_ErrorCodeDescribe(error_code).
    *
    * @note This function should be called after HYPREDRV_LinearSolverApply has
    * completed successfully.
    *
    * @note The operation performed is: state[state_idx] += vec_x (in-place update).
    *
    * Example Usage:
    * @code
    *    HYPREDRV_t hypredrv;
    *    // ... (assemble linear system, solve) ...
    *    HYPREDRV_SAFE_CALL(HYPREDRV_LinearSolverApply(hypredrv));
    *    HYPREDRV_SAFE_CALL(HYPREDRV_StateVectorApplyCorrection(hypredrv, 0));
    * @endcode
    */

   HYPREDRV_EXPORT_SYMBOL uint32_t
   HYPREDRV_StateVectorApplyCorrection(HYPREDRV_t hypredrv, int state_idx);

   /**
    * @brief Create a preconditioner for the HYPREDRV object based on the specified
    * method.
    *
    * @param hypredrv The HYPREDRV_t object for which the preconditioner is to be created.
    *
    * @return Returns an error code with 0 indicating success. Any non-zero value
    * indicates a failure, and the error code can be further described using
    * HYPREDRV_ErrorCodeDescribe(error_code).
    *
    * @note It's the caller's responsibility to ensure that the hypredrv parameter is a
    * valid pointer to an initialized HYPREDRV_t object. Passing a NULL or uninitialized
    * object will result in an error. The function assumes that the preconditioning method
    * and other related settings are properly set in the input arguments.
    *
    * Example Usage:
    * @code
    *    HYPREDRV_t *hypredrv;
    *    // ... (hypredrv is created, and its components are initialized) ...
    *    HYPREDRV_SAFE_CALL(HYPREDRV_PreconCreate(hypredrv));
    * @endcode
    */

   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_PreconCreate(HYPREDRV_t hypredrv);

   /**
    * @brief Create a linear solver for the HYPREDRV object based on the specified method.
    *
    * @param hypredrv The HYPREDRV_t object for which the linear solver is to be created.
    *
    * @return Returns an error code with 0 indicating success. Any non-zero value
    * indicates a failure, and the error code can be further described using
    * HYPREDRV_ErrorCodeDescribe(error_code).
    *
    * @note It's the caller's responsibility to ensure that the hypredrv parameter is a
    * valid pointer to an initialized HYPREDRV_t object. Passing a NULL or uninitialized
    * object will result in an error. The function assumes that the solver method and
    * other related settings are properly set in the input arguments.
    *
    * Example Usage:
    * @code
    *    HYPREDRV_t *hypredrv;
    *    // ... (hypredrv is created, and its components are initialized) ...
    *    HYPREDRV_SAFE_CALL(HYPREDRV_LinearSolverCreate(hypredrv));
    * @endcode
    */

   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_LinearSolverCreate(HYPREDRV_t hypredrv);

   /**
    * @brief Set up the preconditioner for the HYPREDRV object.
    *
    * Sets up only the preconditioner, without setting up the enclosing linear solver.
    * Use this when applying the preconditioner standalone via HYPREDRV_PreconApply().
    *
    * @note HYPREDRV_LinearSolverSetup() sets up the preconditioner internally as part
    * of the combined solver setup. A prior call to this function is **not required**
    * before calling HYPREDRV_LinearSolverSetup().
    *
    * @param hypredrv The HYPREDRV_t object for which the preconditioner is to be set up.
    *
    * @return Returns an error code with 0 indicating success. Any non-zero value
    * indicates a failure, and the error code can be further described using
    * HYPREDRV_ErrorCodeDescribe(error_code).
    *
    * Example Usage:
    * @code
    *    HYPREDRV_t hypredrv;
    *    // ... (hypredrv is created, matrix is set) ...
    *    HYPREDRV_SAFE_CALL(HYPREDRV_PreconCreate(hypredrv));
    *    HYPREDRV_SAFE_CALL(HYPREDRV_PreconSetup(hypredrv));
    *    HYPREDRV_SAFE_CALL(HYPREDRV_PreconApply(hypredrv, vec_b, vec_x));
    * @endcode
    */

   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_PreconSetup(HYPREDRV_t hypredrv);

   /**
    * @brief Set up the linear solver for the HYPREDRV object.
    *
    * Configures the linear solver using the current solver and preconditioner methods,
    * matrix, RHS vector, and solution vector. This function sets up the preconditioner
    * internally - a prior call to HYPREDRV_PreconSetup() is **not required** for the
    * normal solver workflow.
    *
    * @note To use the preconditioner standalone (e.g., via HYPREDRV_PreconApply()),
    * call HYPREDRV_PreconSetup() explicitly instead.
    *
    * @param hypredrv The HYPREDRV_t object for which the linear solver is to be set up.
    *
    * @return Returns an error code with 0 indicating success. Any non-zero value
    * indicates a failure, and the error code can be further described using
    * HYPREDRV_ErrorCodeDescribe(error_code).
    *
    * Example Usage:
    * @code
    *    HYPREDRV_t hypredrv;
    *    // ... (hypredrv is created, and its components are initialized) ...
    *    HYPREDRV_SAFE_CALL(HYPREDRV_LinearSolverSetup(hypredrv));
    * @endcode
    */

   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_LinearSolverSetup(HYPREDRV_t hypredrv);

   /**
    * @brief Apply the linear solver to solve the linear system for the HYPREDRV object.
    *
    * @param hypredrv The HYPREDRV_t object for which the linear solver is to be applied.
    *
    * @return Returns an error code with 0 indicating success. Any non-zero value
    * indicates a failure, and the error code can be further described using
    * HYPREDRV_ErrorCodeDescribe(error_code).
    *
    * @note It's the caller's responsibility to ensure that the hypredrv parameter is a
    * valid pointer to an initialized HYPREDRV_t object. Passing a NULL or uninitialized
    * object will result in an error. The function assumes that the solver method, solver,
    * matrix, RHS vector, and solution vector are properly set in the input arguments.
    *
    * Example Usage:
    * @code
    *    HYPREDRV_t *hypredrv;
    *    // ... (hypredrv is created, and its components are initialized) ...
    *    HYPREDRV_SAFE_CALL(HYPREDRV_LinearSolverApply(hypredrv));
    * @endcode
    */

   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_LinearSolverApply(HYPREDRV_t hypredrv);

   /**
    * @brief Apply the preconditioner associated with a HYPREDRV_t object to an input
    * vector.
    *
    * @param hypredrv The HYPREDRV_t object defining the preconditioner to be applied.
    *
    * @param vec_b The HYPRE_Vector object defining the input vector.
    *
    * @param vec_x The HYPRE_Vector object defining the output vector.
    *
    * @return Returns an error code with 0 indicating success. Any non-zero value
    * indicates a failure, and the error code can be further described using
    * HYPREDRV_ErrorCodeDescribe(error_code).
    *
    * @note It's the caller's responsibility to ensure that the hypredrv parameter is a
    * valid pointer to an initialized HYPREDRV_t object. Passing a NULL or uninitialized
    * object will result in an error. The function assumes that the preconditioner method,
    * solver, matrix, RHS vector, and solution vector are properly set in the input
    * arguments.
    *
    * Example Usage:
    * @code
    *    HYPREDRV_t *hypredrv;
    *    // ... (hypredrv is created, and its components are initialized) ...
    *    HYPREDRV_SAFE_CALL(HYPREDRV_PreconApply(hypredrv, vec_b, vec_x));
    * @endcode
    */

   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_PreconApply(HYPREDRV_t   hypredrv,
                                                        HYPRE_Vector vec_b,
                                                        HYPRE_Vector vec_x);

   /**
    * @brief Destroy the preconditioner associated with the HYPREDRV object.
    *
    * @param hypredrv The HYPREDRV_t object whose preconditioner is to be destroyed.
    *
    * @return Returns an error code with 0 indicating success. Any non-zero value
    * indicates a failure, and the error code can be further described using
    * HYPREDRV_ErrorCodeDescribe(error_code).
    *
    * @note It's the caller's responsibility to ensure that the hypredrv parameter is a
    * valid pointer to an initialized HYPREDRV_t object. Passing a NULL or uninitialized
    * object will result in an error. The function assumes that the preconditioner method
    * and the preconditioner itself are properly set in the HYPREDRV_t object.
    *
    * Example Usage:
    * @code
    *    HYPREDRV_t *hypredrv;
    *    // ... (hypredrv is created, and its components are initialized) ...
    *    HYPREDRV_SAFE_CALL(HYPREDRV_PreconDestroy(hypredrv));
    * @endcode
    */

   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_PreconDestroy(HYPREDRV_t hypredrv);

   /**
    * @brief Destroy the linear solver associated with the HYPREDRV object.
    *
    * @param hypredrv The HYPREDRV_t object whose linear solver is to be destroyed.
    *
    * @return Returns an error code with 0 indicating success. Any non-zero value
    * indicates a failure, and the error code can be further described using
    * HYPREDRV_ErrorCodeDescribe(error_code).
    *
    * @note It's the caller's responsibility to ensure that the hypredrv parameter is a
    * valid pointer to an initialized HYPREDRV_t object. Passing a NULL or uninitialized
    * object will result in an error. The function assumes that the linear solver method
    * and the linear solver object itself are properly set in the HYPREDRV_t object.
    *
    * @note This function also destroys the associated preconditioner. There is no need
    * to call HYPREDRV_PreconDestroy separately before or after this call.
    *
    * Example Usage:
    * @code
    *    HYPREDRV_t *hypredrv;
    *    // ... (hypredrv is created, and its components are initialized) ...
    *    HYPREDRV_SAFE_CALL(HYPREDRV_LinearSolverDestroy(hypredrv));
    * @endcode
    */

   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_LinearSolverDestroy(HYPREDRV_t hypredrv);

   /**
    * @brief Print the statistics associated with the HYPREDRV object.
    *
    * This function is responsible for printing the statistics collected during the
    * operations performed by the HYPREDRV object.
    *
    * @param hypredrv The HYPREDRV_t object whose statistics are to be printed.
    *
    * @return Returns an error code with 0 indicating success. Any non-zero value
    * indicates a failure, and the error code can be further described using
    * HYPREDRV_ErrorCodeDescribe(error_code).
    *
    * @note It's the caller's responsibility to ensure that the hypredrv parameter is a
    * valid pointer to an initialized HYPREDRV_t object. Passing a NULL or uninitialized
    * object will result in an error.
    *
    * Example Usage:
    * @code
    *    HYPREDRV_t *hypredrv;
    *    // ... (hypredrv is created, and its components are initialized) ...
    *    HYPREDRV_SAFE_CALL(HYPREDRV_StatsPrint(hypredrv));
    * @endcode
    */

   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_StatsPrint(HYPREDRV_t hypredrv);

   /**
    * @brief Begin annotation of a code region for timing and Caliper instrumentation.
    *
    * This function marks the beginning of a code region for performance measurement.
    * It should be paired with HYPREDRV_AnnotateEnd to mark the end of the region.
    *
    * @param hypredrv The HYPREDRV_t object whose stats context should receive
    *                 the annotation.
    * @param name The name of the region to annotate. Available names include: "system",
    *             "matrix", "rhs", "dofmap", "prec", "solve", "reset_x0", "initialize",
    *             "finalize", or custom names.
    * @param id An integer identifier for the region. If id >= 0, the region name will
    *           be formatted as "name-id" (e.g., "system-1"). If id < 0, the name is
    *           used as-is (e.g., "system").
    *
    * @return Returns an error code with 0 indicating success.
    *
    * @note This function is **Caliper-only**: it does not record entries in the stats
    * context and thus does not feed HYPREDRV_StatsLevelGetEntry(). Use
    * HYPREDRV_AnnotateLevelBegin/End if you need both Caliper and stats recording.
    *
    * @note When Caliper is enabled (via HYPREDRV_ENABLE_CALIPER), this function also
    * creates Caliper regions that can be captured by Caliper profiling tools.
    *
    * Example Usage:
    * @code
    *    HYPREDRV_SAFE_CALL(HYPREDRV_AnnotateBegin(hypredrv, "system", -1));
    *    // ... code to measure ...
    *    HYPREDRV_SAFE_CALL(HYPREDRV_AnnotateEnd(hypredrv, "system", -1));
    * @endcode
    */
   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_AnnotateBegin(HYPREDRV_t  hypredrv,
                                                          const char *name, int id);

   /**
    * @brief End annotation of a code region for timing and Caliper instrumentation.
    *
    * This function marks the end of a code region for performance measurement.
    * It should be paired with HYPREDRV_AnnotateBegin to mark the beginning of the region.
    *
    * @param hypredrv The HYPREDRV_t object whose stats context should receive
    *                 the annotation end event.
    * @param name The name of the region to annotate. Must match the name used in the
    *             corresponding HYPREDRV_AnnotateBegin call.
    * @param id An integer identifier for the region. Must match the id used in the
    *           corresponding HYPREDRV_AnnotateBegin call.
    *
    * @return Returns an error code with 0 indicating success.
    */
   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_AnnotateEnd(HYPREDRV_t  hypredrv,
                                                        const char *name, int id);

   /**
    * @brief Begin hierarchical annotation of a code region with a specified level.
    *
    * This function marks the beginning of a hierarchical code region for performance
    * measurement. Hierarchical annotations allow tracking nested regions such as
    * time steps (level 0), non-linear iterations (level 1), etc.
    *
    * @param hypredrv The HYPREDRV_t object whose stats context should receive
    *                 the annotation.
    * @param level The hierarchical level (0-9). Lower levels represent outer loops,
    *              higher levels represent inner loops. For example:
    *              - Level 0: Time steps
    *              - Level 1: Non-linear iterations
    * @param name The name of the region to annotate (e.g., "timestep", "newton").
    * @param id An integer identifier for the region. If id >= 0, the region name
    *           will be formatted as "name-id" (e.g., "timestep-0", "newton-1").
    *           If id < 0, the name is used as-is.
    *
    * @return Returns an error code with 0 indicating success.
    *
    * @note Each level can only have one active annotation at a time. The annotation
    *       must be ended with HYPREDRV_AnnotateLevelEnd using the same level, name,
    *       and id.
    *
    * Example Usage:
    * @code
    *    // Time step loop (level 0)
    *    for (int t = 0; t < num_steps; t++)
    *    {
    *       HYPREDRV_SAFE_CALL(HYPREDRV_AnnotateLevelBegin(hypredrv, 0, "timestep", t));
    *
    *       // Non-linear iteration loop (level 1)
    *       for (int n = 0; n < max_nl_iters; n++)
    *       {
    *          HYPREDRV_SAFE_CALL(HYPREDRV_AnnotateLevelBegin(hypredrv, 1, "newton", n));
    *          // ... solve linear system ...
    *          HYPREDRV_SAFE_CALL(HYPREDRV_AnnotateLevelEnd(hypredrv, 1, "newton", n));
    *       }
    *
    *       HYPREDRV_SAFE_CALL(HYPREDRV_AnnotateLevelEnd(hypredrv, 0, "timestep", t));
    *    }
    * @endcode
    */
   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_AnnotateLevelBegin(HYPREDRV_t  hypredrv,
                                                               int         level,
                                                               const char *name, int id);

   /**
    * @brief End hierarchical annotation of a code region with a specified level.
    *
    * This function marks the end of a hierarchical code region. The level, name, and
    * id must match the corresponding HYPREDRV_AnnotateLevelBegin call.
    *
    * @param hypredrv The HYPREDRV_t object.
    * @param level The hierarchical level (0-9), must match the Begin call.
    * @param name The name of the region, must match the Begin call.
    * @param id An integer identifier for the region, must match the Begin call.
    *
    * @return Returns an error code with 0 indicating success.
    */
   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_AnnotateLevelEnd(HYPREDRV_t hypredrv,
                                                             int level, const char *name,
                                                             int id);

   /*--------------------------------------------------------------------------
    *--------------------------------------------------------------------------*/

   /**
    * @brief Compute the full eigenspectrum of the current linear system matrix
    * or the preconditioned linear system.
    *
    * @param hypredrv The HYPREDRV_t object for which the eigenspectrum is to be computed.
    *
    * @return Returns an error code with 0 indicating success
    *
    * @note When hypredrive is built **without** `-DHYPREDRV_ENABLE_EIGSPEC=ON`,
    * this function is a silent no-op: it prints a one-time warning to stderr on
    * rank 0 and returns success. It is therefore safe to leave the call
    * unconditionally in application code.
    *
    * @note When built **with** `-DHYPREDRV_ENABLE_EIGSPEC=ON`, this function
    * operates on a single MPI rank and writes eigenvalues (and optionally
    * eigenvectors) to files in the current directory as configured via the YAML
    * input under `linear_system.eigspec`.
    *
    * Example Usage:
    * @code
    *    HYPREDRV_t *hypredrv;
    *    // ... (hypredrv is created, and its components are initialized) ...
    *    HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemComputeEigenspectrum(hypredrv));
    * @endcode
    */
   HYPREDRV_EXPORT_SYMBOL uint32_t
   HYPREDRV_LinearSystemComputeEigenspectrum(HYPREDRV_t hypredrv);

   /**
    * @brief Get the iteration count from the last linear solve.
    *
    * @param hypredrv The HYPREDRV_t object.
    * @param iters    Pointer to store the iteration count.
    *
    * @return Returns an error code with 0 indicating success.
    *
    * Example Usage:
    * @code
    *    int iters;
    *    HYPREDRV_SAFE_CALL(HYPREDRV_LinearSolverGetNumIter(hypredrv, &iters));
    * @endcode
    */
   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_LinearSolverGetNumIter(HYPREDRV_t hypredrv,
                                                                   int       *iters);

   /**
    * @brief Get the preconditioner setup time from the last linear solve.
    *
    * @param hypredrv The HYPREDRV_t object.
    * @param seconds  Pointer to store the setup time in the currently configured
    *                 stats units (seconds by default, milliseconds when
    *                 ``general.use_millisec: on``).
    *
    * @return Returns an error code with 0 indicating success.
    *
    * Example Usage:
    * @code
    *    double setup_time;
    *    HYPREDRV_SAFE_CALL(HYPREDRV_LinearSolverGetSetupTime(hypredrv, &setup_time));
    * @endcode
    */
   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_LinearSolverGetSetupTime(HYPREDRV_t hypredrv,
                                                                     double    *seconds);

   /**
    * @brief Get the solver apply time from the last linear solve.
    *
    * @param hypredrv The HYPREDRV_t object.
    * @param seconds  Pointer to store the solve time in the currently configured
    *                 stats units (seconds by default, milliseconds when
    *                 ``general.use_millisec: on``).
    *
    * @return Returns an error code with 0 indicating success.
    *
    * Example Usage:
    * @code
    *    double solve_time;
    *    HYPREDRV_SAFE_CALL(HYPREDRV_LinearSolverGetSolveTime(hypredrv, &solve_time));
    * @endcode
    */
   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_LinearSolverGetSolveTime(HYPREDRV_t hypredrv,
                                                                     double    *seconds);

   /**
    * @brief Get the number of entries recorded at a specific level.
    *
    * Returns the count of entries recorded via level annotations
    * (HYPREDRV_AnnotateLevelBegin/End with the specified level).
    *
    * @param hypredrv The HYPREDRV_t object whose stats context should be queried.
    * @param level The annotation level (0 to STATS_MAX_LEVELS-1).
    * @param count Pointer to store the number of entries recorded.
    *
    * @return Returns an error code with 0 indicating success. Returns a non-zero
    * error code if level is out of range or no stats context is active.
    */
   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_StatsLevelGetCount(HYPREDRV_t hypredrv,
                                                               int level, int *count);

   /**
    * @brief Get statistics entry by level and index.
    *
    * Retrieves statistics for a specific entry at the given level. Statistics are
    * computed on-demand from the solve index range.
    *
    * @param hypredrv The HYPREDRV_t object whose stats context should be queried.
    * @param level The annotation level (0 to STATS_MAX_LEVELS-1).
    * @param index Zero-based index of the entry (0 to count-1).
    * @param entry_id Pointer to store the entry ID (1-based).
    * @param num_solves Pointer to store the number of solves in this entry.
    * @param linear_iters Pointer to store the total linear iterations.
    * @param setup_time Pointer to store the total setup time (seconds).
    * @param solve_time Pointer to store the total solve time (seconds).
    *
    * @return Returns an error code with 0 indicating success. Returns a non-zero
    * error code if the level or index is invalid, or if no stats context is active.
    *
    * @note Any pointer parameter can be NULL to skip retrieving that value.
    */
   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_StatsLevelGetEntry(
      HYPREDRV_t hypredrv, int level, int index, int *entry_id, int *num_solves,
      int *linear_iters, double *setup_time, double *solve_time);

   /**
    * @brief Print statistics summary for a specific level.
    *
    * Prints a table of all recorded entries at the specified level including
    * number of solves, linear iterations, setup time, and solve time per entry,
    * followed by totals and averages.
    *
    * @param hypredrv The HYPREDRV_t object whose stats context should be printed.
    * @param level The annotation level to print (0 to STATS_MAX_LEVELS-1).
    *
    * @return Returns an error code with 0 indicating success.
    */
   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_StatsLevelPrint(HYPREDRV_t hypredrv,
                                                            int        level);

   /*--------------------------------------------------------------------------
    *--------------------------------------------------------------------------*/

   /**@}*/

#ifdef __cplusplus
}
#endif

#endif /* ifndef HYPREDRV_HEADER */
