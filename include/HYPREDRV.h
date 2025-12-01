/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef HYPREDRV_HEADER
#define HYPREDRV_HEADER

#include <mpi.h>
#include <signal.h> // For: raise
#include <stdint.h> // For: uint
#include <string.h> // For: strcmp

#include <HYPRE.h>
#include <HYPRE_IJ_mv.h>
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

#ifdef __cplusplus
extern "C"
{
#endif

// Macro for safely calling HYPREDRV functions.
// Always uses MPI_COMM_WORLD for MPI_Abort as a safe fallback.
#ifndef HYPREDRV_SAFE_CALL
#define HYPREDRV_SAFE_CALL(call)                                                     \
   do                                                                                \
   {                                                                                 \
      uint32_t error_code = (call);                                                  \
      if (error_code != 0)                                                           \
      {                                                                              \
         (void)fprintf(stderr, "At %s:%d in %s():\n", __FILE__, __LINE__, __func__); \
         HYPREDRV_ErrorCodeDescribe(error_code);                                     \
         const char *debug_env = getenv("HYPREDRV_DEBUG");                           \
         if (debug_env && strcmp(debug_env, "1") == 0)                               \
         {                                                                           \
            raise(SIGTRAP); /* Breakpoint for gdb */                                 \
         }                                                                           \
         else                                                                        \
         {                                                                           \
            MPI_Abort(MPI_COMM_WORLD, error_code);                                   \
         }                                                                           \
      }                                                                              \
   } while (0)
#endif

// Visibility control macros
#define HYPREDRV_EXPORT_SYMBOL __attribute__((visibility("default")))

   /*--------------------------------------------------------------------------
    *--------------------------------------------------------------------------*/

   /**
    * @defgroup HYPREDRV
    *
    * @brief Public APIs to solve linear systems with hypre through hypredrive
    *
    * @{
    **/

   /*--------------------------------------------------------------------------
    *--------------------------------------------------------------------------*/

   /**
    * HYPREDRV_t
    *
    * The main object type for the HYPREDRV library that encapsulates all data and
    *functionality needed to solve linear systems using HYPRE. This includes:
    *
    * - Input parameters and configuration
    * - Linear system components (matrix, RHS vector, solution vector)
    * - Solver and preconditioner objects
    * - Performance statistics and timing data
    * - MPI communication context
    *
    * The object is created with HYPREDRV_Create() and must be destroyed with
    *HYPREDRV_Destroy() when no longer needed to prevent memory leaks.
    *
    * This is an opaque pointer type - the internal structure is not exposed to users of
    *the library. All interactions with HYPREDRV_t objects should be done through the
    *public API functions.
    **/

   struct hypredrv_struct;
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
    * function finalizes the HYPRE library and its device-specific components. It should
    * be the last HYPREDRV-related function called before the application exits.
    *
    * @return Returns an error code with 0 indicating success. Any non-zero value
    * indicates a failure, and the error code can be further described using
    * HYPREDRV_ErrorCodeDescribe(error_code).
    *
    * @note This function should be called before MPI_Finalize. It is safe to call this
    * function even if HYPREDRV_Initialize was not successfully called or was not called
    * at all; in such cases, the function will have no effect.
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
    * @note This function will not return if error_code is non-zero, as it calls MPI_Abort
    * @note For convenience, consider using the HYPREDRV_SAFE_CALL macro which
    * automatically handles error checking and description for HYPREDRV function calls. In
    * case a nonzero error occurs, the macro will also call the MPI program with the given
    * error code
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

   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_Create(MPI_Comm, HYPREDRV_t *hypredrv_ptr);

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
    * This function prints the current date and time, followed by version information
    * about the HYPRE library.
    *
    * @param comm The MPI communicator associated with the HYPREDRV object.
    *
    * @return Returns an error code with 0 indicating success. Any non-zero value
    * indicates a failure, and the error code can be further described using
    * HYPREDRV_ErrorCodeDescribe(error_code).
    *
    * @note This function uses conditional compilation to determine what information about
    * the HYPRE library to print.
    *
    * Example Usage:
    * @code
    *    MPI_Comm comm = MPI_COMM_WORLD; // or any other valid MPI_Comm
    *    HYPREDRV_SAFE_CALL(HYPREDRV_PrintLibInfo(comm));
    * @endcode
    */

   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_PrintLibInfo(MPI_Comm comm);

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
    * This function is responsible for parsing the command-line arguments provided to the
    * application.
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
    *    HYPREDRV_t *hypredrv;
    *    int argc = ...; // Number of arguments
    *    char **argv = ...; // Argument strings
    *    // ... (hypredrv is created) ...
    *    HYPREDRV_SAFE_CALL(HYPREDRV_InputArgsParse(argc, argv, &hypredrv));
    * @endcode
    */

   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_InputArgsParse(int, char **,
                                                           HYPREDRV_t hypredrv);

   /**
    * @brief Set library mode to HYPREDRV, in which matrices and vectors are not assumed
    * to be owned by the HYPREDRV_t object.
    *
    * @param hypredrv The HYPREDRV_t object.
    *
    * @return Returns an error code with 0 indicating success. Any non-zero value
    * indicates a failure, and the error code can be further described using
    * HYPREDRV_ErrorCodeDescribe(error_code).
    *
    * Example Usage:
    * @code
    *    HYPREDRV_t *hypredrv;
    *    // ... (hypredrv is created, and its components are initialized) ...
    *    HYPREDRV_SAFE_CALL(HYPREDRV_SetLibraryMode(hypredrv));
    * @endcode
    */

   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_SetLibraryMode(HYPREDRV_t hypredrv);

   /**
    * @brief Set HYPRE's global options according to the YAML input.
    *
    * @param hypredrv The HYPREDRV_t object from which the global options are to be
    * retrieved.
    *
    * @return Returns an error code with 0 indicating success. Any non-zero value
    * indicates a failure, and the error code can be further described using
    * HYPREDRV_ErrorCodeDescribe(error_code).
    *
    * Example Usage:
    * @code
    *    HYPREDRV_t *hypredrv;
    *    // ... (hypredrv is created, and its components are initialized) ...
    *    HYPREDRV_SAFE_CALL(HYPREDRV_SetGlobalOptions(hypredrv));
    * @endcode
    */

   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_SetGlobalOptions(HYPREDRV_t hypredrv);

   /**
    * @brief Retrieve the warmup setting from a HYPREDRV object.
    *
    * This function accesses the HYPREDRV object's input arguments structure to retrieve
    * the warmup setting. This setting indicates whether a warmup phase should be executed
    * before the main operations, often used to ensure accurate timing measurements by
    * eliminating any initialization overhead.
    *
    * @param hypredrv The HYPREDRV_t object from which the warmup setting is to be
    * retrieved.
    *
    * @return Returns the warmup setting as an integer. If the input object is NULL
    * or not properly initialized, the function returns -1 to indicate an error or invalid
    * state.
    *
    * Example Usage:
    * @code
    *    HYPREDRV_t *hypredrv;
    *    // ... (hypredrv is created, and its components are set) ...
    *    int warmup = HYPREDRV_InputArgsGetWarmup(hypredrv);
    * @endcode
    */

   HYPREDRV_EXPORT_SYMBOL int HYPREDRV_InputArgsGetWarmup(HYPREDRV_t hypredrv);

   /**
    * @brief Retrieve the number of repetitions from a HYPREDRV object.
    *
    * This function accesses the HYPREDRV object's input arguments structure to retrieve
    * the number of repetitions setting. This setting specifies how many times the linear
    * systems should be solved, potentially for benchmarking or testing purposes.
    *
    * @param hypredrv The HYPREDRV_t object from which the number of repetitions is to be
    * retrieved.
    *
    * @return Returns the number of repetitions as an integer. If the input object is NULL
    * or not properly initialized, the function returns -1 to indicate an error or invalid
    * state.
    *
    * Example Usage:
    * @code
    *    HYPREDRV_t *hypredrv;
    *    // ... (hypredrv is created, and its components are set) ...
    *    int num_reps = HYPREDRV_InputArgsGetNumRepetitions(hypredrv);
    * @endcode
    */

   HYPREDRV_EXPORT_SYMBOL int HYPREDRV_InputArgsGetNumRepetitions(HYPREDRV_t hypredrv);

   /**
    * @brief Retrieve the number of linear systems from a HYPREDRV object.
    *
    * This function accesses the HYPREDRV object's input arguments structure to retrieve
    * the number of linear systems being solved.
    *
    * @param hypredrv The HYPREDRV_t object from which the number of repetitions is to be
    * retrieved.
    *
    * @return Returns the number of linear systems as an integer. If the input object is
    * NULL or not properly initialized, the function returns -1 to indicate an error or
    * invalid state.
    *
    * Example Usage:
    * @code
    *    HYPREDRV_t *hypredrv;
    *    // ... (hypredrv is created, and its components are set) ...
    *    int num_ls = HYPREDRV_InputArgsGetNumLinearSystems(hypredrv);
    * @endcode
    */

   HYPREDRV_EXPORT_SYMBOL int HYPREDRV_InputArgsGetNumLinearSystems(HYPREDRV_t hypredrv);

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
    * @brief Set the linear system right-hand side (RHS) vector from file for a HYPREDRV
    * object.
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
    *    HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetInitialGuess(hypredrv));
    * @endcode
    */

   HYPREDRV_EXPORT_SYMBOL uint32_t
   HYPREDRV_LinearSystemSetInitialGuess(HYPREDRV_t hypredrv);

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
    *    HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetPrecMatrix(hypredrv));
    * @endcode
    */

   HYPREDRV_EXPORT_SYMBOL uint32_t
   HYPREDRV_LinearSystemSetPrecMatrix(HYPREDRV_t hypredrv);

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
    * "<filename>.<rank>", zero-padded to 5 digits. In single-process runs, the file is
    * written exactly to <filename>.
    *
    * File format (ASCII):
    *   line 1: <num_entries>
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

   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_StateVectorSet(HYPREDRV_t hypredrv,
                                                            int       nstates,
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

   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_StateVectorGetValues(HYPREDRV_t        hypredrv,
                                                                  int              index,
                                                                  HYPRE_Complex   **data_ptr);

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
                                                             int       index_in,
                                                             int       index_out);

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
    * @brief Apply the linear solver correction to the current state vector.
    *
    * This function adds the solution increment from the linear solver (stored in
    * vec_x) to the current state vector (state at logical index 0). This implements
    * the Newton update: U^{k+1} = U^k + ΔU, where ΔU is the solution from the
    * linear system J ΔU = -R.
    *
    * @param hypredrv The HYPREDRV_t object containing the state vectors and the
    *                 linear solver solution (vec_x).
    *
    * @return Returns an error code with 0 indicating success. Any non-zero value
    * indicates a failure, and the error code can be further described using
    * HYPREDRV_ErrorCodeDescribe(error_code).
    *
    * @note This function should be called after HYPREDRV_LinearSolverApply has
    * completed successfully. The linear solver solution (vec_x) must be set via
    * HYPREDRV_LinearSystemSetRHS or similar functions.
    *
    * @note The operation performed is: state[0] = state[0] + vec_x (in-place update).
    *
    * Example Usage:
    * @code
    *    HYPREDRV_t hypredrv;
    *    // ... (assemble linear system, solve) ...
    *    HYPREDRV_SAFE_CALL(HYPREDRV_LinearSolverApply(hypredrv));
    *    HYPREDRV_SAFE_CALL(HYPREDRV_StateVectorApplyCorrection(hypredrv));
    * @endcode
    */

   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_StateVectorApplyCorrection(HYPREDRV_t hypredrv);

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
    * @brief Set up the preconditioner for the HYPREDRV object based on the specified
    * preconditioner methods.
    *
    * @param hypredrv The HYPREDRV_t object for which the preconditioner is to be set up.
    *
    * @return Returns an error code with 0 indicating success. Any non-zero value
    * indicates a failure, and the error code can be further described using
    * HYPREDRV_ErrorCodeDescribe(error_code).
    *
    * @note It's the caller's responsibility to ensure that the hypredrv parameter is a
    * valid pointer to an initialized HYPREDRV_t object. Passing a NULL or uninitialized
    * object will result in an error. The function assumes that the preconditioner method
    * and the matrix are properly set in the input arguments.
    *
    * Example Usage:
    * @code
    *    HYPREDRV_t *hypredrv;
    *    // ... (hypredrv is created, and its components are initialized) ...
    *    HYPREDRV_SAFE_CALL(HYPREDRV_PreconSetup(hypredrv));
    * @endcode
    */

   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_PreconSetup(HYPREDRV_t hypredrv);

   /**
    * @brief Set up the linear solver for the HYPREDRV object based on the specified
    * solver and preconditioner methods.
    *
    * @param hypredrv The HYPREDRV_t object for which the linear solver is to be set up.
    *
    * @return Returns an error code with 0 indicating success. Any non-zero value
    * indicates a failure, and the error code can be further described using
    * HYPREDRV_ErrorCodeDescribe(error_code).
    *
    * @note It's the caller's responsibility to ensure that the hypredrv parameter is a
    * valid pointer to an initialized HYPREDRV_t object. Passing a NULL or uninitialized
    * object will result in an error. The function assumes that the solver and
    * preconditioner methods, as well as the matrix, RHS vector, and solution vector, are
    * properly set in the input arguments.
    *
    * Example Usage:
    * @code
    *    HYPREDRV_t *hypredrv;
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
    * @param name The name of the region to annotate (printf-style format string).
    *             Available names include: "system", "matrix", "rhs", "dofmap", "prec",
    *             "solve", "reset_x0", "initialize", "finalize", or custom names.
    * @param ... Additional arguments for the format string.
    *
    * @return Returns an error code with 0 indicating success. Any non-zero value
    * indicates a failure, and the error code can be further described using
    * HYPREDRV_ErrorCodeDescribe(error_code).
    *
    * @note When Caliper is enabled (via HYPREDRV_ENABLE_CALIPER), this function also
    * creates Caliper regions that can be captured by Caliper profiling tools.
    *
    * Example Usage:
    * @code
    *    HYPREDRV_SAFE_CALL(HYPREDRV_AnnotateBegin("system"));
    *    // ... code to measure ...
    *    HYPREDRV_SAFE_CALL(HYPREDRV_AnnotateEnd("system"));
    * @endcode
    */
   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_AnnotateBegin(const char *name, ...);

   /**
    * @brief End annotation of a code region for timing and Caliper instrumentation.
    *
    * This function marks the end of a code region for performance measurement.
    * It should be paired with HYPREDRV_AnnotateBegin to mark the beginning of the region.
    *
    * @param name The name of the region to annotate (printf-style format string).
    *             Must match the name used in the corresponding HYPREDRV_AnnotateBegin
    * call.
    * @param ... Additional arguments for the format string.
    *
    * @return Returns an error code with 0 indicating success. Any non-zero value
    * indicates a failure, and the error code can be further described using
    * HYPREDRV_ErrorCodeDescribe(error_code).
    *
    * @note When Caliper is enabled (via HYPREDRV_ENABLE_CALIPER), this function also
    * creates Caliper regions that can be captured by Caliper profiling tools.
    *
    * Example Usage:
    * @code
    *    HYPREDRV_SAFE_CALL(HYPREDRV_AnnotateBegin("Run-%d", iteration));
    *    // ... code to measure ...
    *    HYPREDRV_SAFE_CALL(HYPREDRV_AnnotateEnd("Run-%d", iteration));
    * @endcode
    */
   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_AnnotateEnd(const char *name, ...);

   /**
    * @brief Begin hierarchical annotation of a code region with a specified level.
    *
    * This function marks the beginning of a hierarchical code region for performance
    * measurement. Hierarchical annotations allow tracking nested regions such as
    * time steps (level 0), non-linear iterations (level 1), etc.
    *
    * @param level The hierarchical level (0-9). Lower levels represent outer loops,
    *              higher levels represent inner loops. For example:
    *              - Level 0: Time steps
    *              - Level 1: Non-linear iterations
    * @param name The name of the region to annotate (printf-style format string).
    * @param ... Additional arguments for the format string.
    *
    * @return Returns an error code with 0 indicating success.
    *
    * @note Each level can only have one active annotation at a time. The annotation
    *       must be ended with HYPREDRV_AnnotateLevelEnd using the same level and name.
    *
    * Example Usage:
    * @code
    *    // Time step loop (level 0)
    *    for (int t = 0; t < num_steps; t++)
    *    {
    *       HYPREDRV_SAFE_CALL(HYPREDRV_AnnotateLevelBegin(0, "timestep-%d", t));
    *
    *       // Non-linear iteration loop (level 1)
    *       for (int n = 0; n < max_nl_iters; n++)
    *       {
    *          HYPREDRV_SAFE_CALL(HYPREDRV_AnnotateLevelBegin(1, "newton-%d", n));
    *          // ... solve linear system ...
    *          HYPREDRV_SAFE_CALL(HYPREDRV_AnnotateLevelEnd(1, "newton-%d", n));
    *       }
    *
    *       HYPREDRV_SAFE_CALL(HYPREDRV_AnnotateLevelEnd(0, "timestep-%d", t));
    *    }
    * @endcode
    */
   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_AnnotateLevelBegin(int         level,
                                                               const char *name, ...);

   /**
    * @brief End hierarchical annotation of a code region with a specified level.
    *
    * This function marks the end of a hierarchical code region. The level and name
    * must match the corresponding HYPREDRV_AnnotateLevelBegin call.
    *
    * @param level The hierarchical level (0-9), must match the Begin call.
    * @param name The name of the region (printf-style format string), must match
    *             the Begin call.
    * @param ... Additional arguments for the format string.
    *
    * @return Returns an error code with 0 indicating success.
    */
   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_AnnotateLevelEnd(int level, const char *name,
                                                             ...);

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
    * @note Compiled and available only when hypredrive is built with
    * `-DHYPREDRV_ENABLE_EIGSPEC=ON`. This function operates on a single MPI rank
    * and writes eigenvalues (and optionally eigenvectors) to files in the current
    * directory as configured via the YAML input under `linear_system.eigspec`.
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
    * @brief Get a statistic from the last linear solve.
    *
    * @param hypredrv The HYPREDRV_t object.
    * @param name Name of the statistic ("iter", "setup", "solve").
    * @param value Pointer to store the value. Type depends on name:
    *              - "iter": int*
    *              - "setup": double* (seconds)
    *              - "solve": double* (seconds)
    *
    * @return Returns an error code with 0 indicating success.
    */
   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_GetLastStat(HYPREDRV_t  hypredrv,
                                                        const char *name, void *value);

   /**
    * @brief Get the number of entries recorded at a specific level.
    *
    * Returns the count of entries recorded via level annotations
    * (HYPREDRV_AnnotateLevelBegin/End with the specified level).
    *
    * @param level The annotation level (0 to STATS_MAX_LEVELS-1).
    *
    * @return The number of entries recorded, or 0 if no stats context is active.
    */
   HYPREDRV_EXPORT_SYMBOL int HYPREDRV_StatsLevelGetCount(int level);

   /**
    * @brief Get statistics entry by level and index.
    *
    * Retrieves statistics for a specific entry at the given level. Statistics are
    * computed on-demand from the solve index range.
    *
    * @param level The annotation level (0 to STATS_MAX_LEVELS-1).
    * @param index Zero-based index of the entry (0 to count-1).
    * @param entry_id Pointer to store the entry ID (1-based).
    * @param num_solves Pointer to store the number of solves in this entry.
    * @param linear_iters Pointer to store the total linear iterations.
    * @param setup_time Pointer to store the total setup time (seconds).
    * @param solve_time Pointer to store the total solve time (seconds).
    *
    * @return Returns 0 on success, -1 on error (invalid level/index or no stats).
    *
    * @note Any pointer parameter can be NULL to skip retrieving that value.
    */
   HYPREDRV_EXPORT_SYMBOL int HYPREDRV_StatsLevelGetEntry(int     level,
                                                          int     index,
                                                          int    *entry_id,
                                                          int    *num_solves,
                                                          int    *linear_iters,
                                                          double *setup_time,
                                                          double *solve_time);

   /**
    * @brief Print statistics summary for a specific level.
    *
    * Prints a table of all recorded entries at the specified level including
    * number of solves, linear iterations, setup time, and solve time per entry,
    * followed by totals and averages.
    *
    * @param level The annotation level to print (0 to STATS_MAX_LEVELS-1).
    *
    * @return Returns an error code with 0 indicating success.
    */
   HYPREDRV_EXPORT_SYMBOL uint32_t HYPREDRV_StatsLevelPrint(int level);

   /*--------------------------------------------------------------------------
    *--------------------------------------------------------------------------*/

   /**@}*/

#ifdef __cplusplus
}
#endif

#endif /* ifndef HYPREDRV_HEADER */
