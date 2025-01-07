/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef HYPREDRV_HEADER
#define HYPREDRV_HEADER

#include <stdint.h>
#include <mpi.h>

#include <HYPRE.h>
#include <HYPRE_utilities.h>

// Undefine autotools package macros from hypre
#undef PACKAGE_NAME
#undef PACKAGE_BUGREPORT
#undef PACKAGE_STRING
#undef PACKAGE_TARNAME
#undef PACKAGE_URL
#undef PACKAGE_VERSION

#ifdef __cplusplus
extern "C" {
#endif

// Macro for safely calling HYPREDRV functions
#ifndef HYPREDRV_SAFE_CALL
#define HYPREDRV_SAFE_CALL(call)                             \
   do {                                                      \
      uint32_t ierr = (call);                                \
      if (ierr != 0) {                                       \
         const char *msg = HYPREDRV_ErrorCodeDescribe(ierr); \
         fprintf(stderr, "HYPREDRV Error: %s\n", msg);       \
      }                                                      \
   } while(0)
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
 * Insert documentation
 **/

struct hypredrv_struct;
typedef struct hypredrv_struct* HYPREDRV_t;

/**
 * @brief Initializes the HYPREDRV library.
 *
 * Initializes the HYPREDRV library for use in the application. This function sets up the necessary
 * environment for the HYPREDRV library by initializing the HYPRE library and its device-specific
 * components. It ensures that initialization occurs only once even if called multiple times.
 *
 * @return Returns an error code with 0 indicating success. Any non-zero value indicates a failure,
 * and the error code can be further described using HYPREDRV_ErrorCodeDescribe(error_code).
 *
 * @note This function must be called after MPI_Init and before any other HYPREDRV functions are used.
 * It is safe to call this function multiple times; only the first call will perform the initialization.
 *
 * Example Usage:
 * @code
 *    HYPREDRV_SAFE_CALL(HYPREDRV_Initialize());
 * @endcode
 */

HYPREDRV_EXPORT_SYMBOL uint32_t
HYPREDRV_Initialize(void);

/**
 * @brief Finalizes the HYPREDRV library.
 *
 * Cleans up and releases any resources allocated by the HYPREDRV library. This function finalizes
 * the HYPRE library and its device-specific components. It should be the last HYPREDRV-related
 * function called before the application exits.
 *
 * @return Returns an error code with 0 indicating success. Any non-zero value indicates a failure,
 * and the error code can be further described using HYPREDRV_ErrorCodeDescribe(error_code).
 *
 * @note This function should be called before MPI_Finalize. It is safe to call this function
 * even if HYPREDRV_Initialize was not successfully called or was not called at all; in such cases,
 * the function will have no effect.
 *
 * Example Usage:
 * @code
 *    HYPREDRV_SAFE_CALL(HYPREDRV_Finalize());
 * @endcode
 */

HYPREDRV_EXPORT_SYMBOL uint32_t
HYPREDRV_Finalize(void);

/**
 * @brief Create a HYPREDRV object.
 *
 * This function allocates memory for a HYPREDRV object and initializes it with the
 * provided MPI communicator. The newly created object is returned through the hypredrv_ptr
 * parameter.
 *
 * @param comm The MPI communicator to be associated with the HYPREDRV object.
 * This communicator is used for parallel communications in the underlying HYPRE library calls.
 *
 * @param hypredrv_ptr A pointer to the HYPREDRV_t pointer where the address of the newly
 * allocated HYPREDRV object will be stored. After the function call, *hypredrv_ptr will
 * point to the allocated object.
 *
 * @return Returns an error code with 0 indicating success. Any non-zero value indicates a failure,
 * and the error code can be further described using HYPREDRV_ErrorCodeDescribe(error_code).
 *
 * @note It is the caller's responsibility to ensure that the MPI environment is properly
 * initialized before calling this function. The function does not initialize or finalize
 * the MPI environment.
 *
 * Example Usage:
 * @code
 *    HYPREDRV_t *hypredrv;
 *    MPI_Comm comm = MPI_COMM_WORLD; // or any other valid MPI_Comm
 *    HYPREDRV_SAFE_CALL(HYPREDRV_Create(comm, &hypredrv));
 * @endcode
 */

HYPREDRV_EXPORT_SYMBOL uint32_t
HYPREDRV_Create(MPI_Comm, HYPREDRV_t*);

/**
 * @brief Destroy a HYPREDRV object.
 *
 * This function deallocates the memory for a HYPREDRV object and performs necessary cleanup
 * for its associated resources. It destroys HYPRE matrices and vectors created and used by
 * the object, deallocates any input arguments, and finally frees the HYPREDRV object itself.
 * After this function is called, the pointer to the HYPREDRV object is set to NULL to prevent
 * any dangling references.
 *
 * @param hypredrv_ptr A pointer to the HYPREDRV_t object that is to be destroyed. This pointer
 * should have been initialized by HYPREDRV_Create function.
 *
 * @return Returns an error code with 0 indicating success. Any non-zero value indicates a failure,
 * and the error code can be further described using HYPREDRV_ErrorCodeDescribe(error_code).
 *
 * @note It is the caller's responsibility to ensure that hypredrv_ptr is a valid pointer to a
 * HYPREDRV_t object. Passing an invalid pointer or a pointer to an already destroyed object
 * can lead to undefined behavior.
 *
 * Example Usage:
 * @code
 *    HYPREDRV_t *hypredrv;
 *    // ... (hypredrv is created and used) ...
 *    HYPREDRV_SAFE_CALL(HYPREDRV_Destroy(&hypredrv));
 *    // hypredrv is now NULL.
 * @endcode
 */

HYPREDRV_EXPORT_SYMBOL uint32_t
HYPREDRV_Destroy(HYPREDRV_t*);

/**
 * @brief Print library information at entrance.
 *
 * This function prints the current date and time, followed by version information about the
 * HYPRE library.
 *
 * @param comm The MPI communicator associated with the HYPREDRV object.
 *
 * @return Returns an error code with 0 indicating success. Any non-zero value indicates a failure,
 * and the error code can be further described using HYPREDRV_ErrorCodeDescribe(error_code).
 *
 * @note This function uses conditional compilation to determine what information about the
 * HYPRE library to print.
 *
 * Example Usage:
 * @code
 *    MPI_Comm comm = MPI_COMM_WORLD; // or any other valid MPI_Comm
 *    HYPREDRV_SAFE_CALL(HYPREDRV_PrintLibInfo(comm));
 * @endcode
 */

HYPREDRV_EXPORT_SYMBOL uint32_t
HYPREDRV_PrintLibInfo(MPI_Comm comm);

/**
 * @brief Print system information.
 *
 * @param comm The MPI communicator associated with the HYPREDRV object.
 *
 * @return Returns an error code with 0 indicating success. Any non-zero value indicates a failure,
 * and the error code can be further described using HYPREDRV_ErrorCodeDescribe(error_code).
 *
 * @note This function uses conditional compilation to determine what information about the
 * system (machine) to print.
 *
 * Example Usage:
 * @code
 *    MPI_Comm comm = MPI_COMM_WORLD; // or any other valid MPI_Comm
 *    HYPREDRV_SAFE_CALL(HYPREDRV_PrintSystemInfo(comm));
 * @endcode
 */

HYPREDRV_EXPORT_SYMBOL uint32_t
HYPREDRV_PrintSystemInfo(MPI_Comm comm);

/**
 * @brief Print library information at exit.
 *
 * This function prints the driver name followed by the current date and time.
 *
 * @param argv[0] The application driver name.
 *
 * @return Returns an error code with 0 indicating success. Any non-zero value indicates a failure,
 * and the error code can be further described using HYPREDRV_ErrorCodeDescribe(error_code).

 * @note This function is intended to be used just before finishing the driver application
 *
 * Example Usage:
 * @code
 *    MPI_Comm comm = MPI_COMM_WORLD; // or any other valid MPI_Comm
 *    HYPREDRV_SAFE_CALL(HYPREDRV_PrintExitInfo(comm, argv[0]));
 * @endcode
 */

HYPREDRV_EXPORT_SYMBOL uint32_t
HYPREDRV_PrintExitInfo(MPI_Comm comm, const char*);

/**
 * @brief Parse input arguments for a HYPREDRV object.
 *
 * This function is responsible for parsing the command-line arguments provided to the application.
 *
 * @param argc The count of command-line arguments.
 * @param argv The array of command-line argument strings.
 * @param hypredrv The HYPREDRV_t object whose input arguments are to be parsed and set.
 *
 * @return Returns an error code with 0 indicating success. Any non-zero value indicates a failure,
 * and the error code can be further described using HYPREDRV_ErrorCodeDescribe(error_code).
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

HYPREDRV_EXPORT_SYMBOL uint32_t
HYPREDRV_InputArgsParse(int, char**, HYPREDRV_t);

/**
 * @brief Set library mode to HYPREDRV, in which matrices and vectors are not assumed
 * to be owned by the HYPREDRV_t object.
 *
 * @param hypredrv The HYPREDRV_t object.
 *
 * @return Returns an error code with 0 indicating success. Any non-zero value indicates a failure,
 * and the error code can be further described using HYPREDRV_ErrorCodeDescribe(error_code).
 *
 * Example Usage:
 * @code
 *    HYPREDRV_t *hypredrv;
 *    // ... (hypredrv is created, and its components are initialized) ...
 *    HYPREDRV_SAFE_CALL(HYPREDRV_SetLibraryMode(hypredrv));
 * @endcode
 */

HYPREDRV_EXPORT_SYMBOL uint32_t
HYPREDRV_SetLibraryMode(HYPREDRV_t);

/**
 * @brief Set HYPRE's global options according to the YAML input.
 *
 * @param hypredrv The HYPREDRV_t object from which the global options are to be retrieved.
 *
 * @return Returns an error code with 0 indicating success. Any non-zero value indicates a failure,
 * and the error code can be further described using HYPREDRV_ErrorCodeDescribe(error_code).
 *
 * Example Usage:
 * @code
 *    HYPREDRV_t *hypredrv;
 *    // ... (hypredrv is created, and its components are initialized) ...
 *    HYPREDRV_SAFE_CALL(HYPREDRV_SetGlobalOptions(hypredrv));
 * @endcode
 */

HYPREDRV_EXPORT_SYMBOL uint32_t
HYPREDRV_SetGlobalOptions(HYPREDRV_t);

/**
 * @brief Retrieve the warmup setting from a HYPREDRV object.
 *
 * This function accesses the HYPREDRV object's input arguments structure to retrieve
 * the warmup setting. This setting indicates whether a warmup phase should be executed
 * before the main operations, often used to ensure accurate timing measurements by
 * eliminating any initialization overhead.
 *
 * @param hypredrv The HYPREDRV_t object from which the warmup setting is to be retrieved.
 *
 * @return Returns the warmup setting as an integer. If the input object is NULL
 * or not properly initialized, the function returns -1 to indicate an error or invalid state.
 *
 * Example Usage:
 * @code
 *    HYPREDRV_t *hypredrv;
 *    // ... (hypredrv is created, and its components are set) ...
 *    int warmup = HYPREDRV_InputArgsGetWarmup(hypredrv);
 * @endcode
 */

HYPREDRV_EXPORT_SYMBOL int
HYPREDRV_InputArgsGetWarmup(HYPREDRV_t);

/**
 * @brief Retrieve the number of repetitions from a HYPREDRV object.
 *
 * This function accesses the HYPREDRV object's input arguments structure to retrieve
 * the number of repetitions setting. This setting specifies how many times the linear systems
 * should be solved, potentially for benchmarking or testing purposes.
 *
 * @param hypredrv The HYPREDRV_t object from which the number of repetitions is to be retrieved.
 *
 * @return Returns the number of repetitions as an integer. If the input object is NULL
 * or not properly initialized, the function returns -1 to indicate an error or invalid state.
 *
 * Example Usage:
 * @code
 *    HYPREDRV_t *hypredrv;
 *    // ... (hypredrv is created, and its components are set) ...
 *    int num_reps = HYPREDRV_InputArgsGetNumRepetitions(hypredrv);
 * @endcode
 */

HYPREDRV_EXPORT_SYMBOL int
HYPREDRV_InputArgsGetNumRepetitions(HYPREDRV_t);

/**
 * @brief Retrieve the number of linear systems from a HYPREDRV object.
 *
 * This function accesses the HYPREDRV object's input arguments structure to retrieve
 * the number of linear systems being solved.
 *
 * @param hypredrv The HYPREDRV_t object from which the number of repetitions is to be retrieved.
 *
 * @return Returns the number of linear systems as an integer. If the input object is NULL
 * or not properly initialized, the function returns -1 to indicate an error or invalid state.
 *
 * Example Usage:
 * @code
 *    HYPREDRV_t *hypredrv;
 *    // ... (hypredrv is created, and its components are set) ...
 *    int num_ls = HYPREDRV_InputArgsGetNumLinearSystems(hypredrv);
 * @endcode
 */

HYPREDRV_EXPORT_SYMBOL int
HYPREDRV_InputArgsGetNumLinearSystems(HYPREDRV_t);

/**
 * @brief Build the linear system (matrix, RHS, and LHS) according to the YAML input passed to
 * the HYPREDRV object.
 *
 * The matrix is read from file. Vectors might be read from file or built according to
 * predetermined options passed via YAML.
 *
 * @param hypredrv The HYPREDRV_t object for which the linear system matrix is to be built.
 *
 * @return Returns an error code with 0 indicating success. Any non-zero value indicates a failure,
 * and the error code can be further described using HYPREDRV_ErrorCodeDescribe(error_code).
 *
 * @note It's the caller's responsibility to ensure that the hypredrv parameter is a valid pointer to an
 * initialized HYPREDRV_t object. Passing a NULL or uninitialized object will result in an error.
 *
 * Example Usage:
 * @code
 *    HYPREDRV_t *hypredrv;
 *    // ... (hypredrv is created, and its components are initialized) ...
 *    HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemBuild(hypredrv));
 * @endcode
 */

HYPREDRV_EXPORT_SYMBOL uint32_t
HYPREDRV_LinearSystemBuild(HYPREDRV_t);

/**
 * @brief Read the linear system matrix from file for a HYPREDRV object.
 *
 * This function is responsible for reading the matrix data of a linear system associated
 * with a HYPREDRV object. It performs the reading process given input arguments related to
 * the linear system, and uses a pointer to store the read matrix. After reading the matrix,
 * it retrieves and returns the error code generated during the process.
 *
 * @param hypredrv The HYPREDRV_t object for which the linear system matrix is to be read.
 *
 * @return Returns an error code with 0 indicating success. Any non-zero value indicates a failure,
 * and the error code can be further described using HYPREDRV_ErrorCodeDescribe(error_code).
 *
 * @note It's the caller's responsibility to ensure that the hypredrv parameter is a valid pointer to an
 * initialized HYPREDRV_t object. Passing a NULL or uninitialized object will result in an error.
 *
 * Example Usage:
 * @code
 *    HYPREDRV_t *hypredrv;
 *    // ... (hypredrv is created, and its components are initialized) ...
 *    HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemReadMatrix(hypredrv));
 * @endcode
 */

HYPREDRV_EXPORT_SYMBOL uint32_t
HYPREDRV_LinearSystemReadMatrix(HYPREDRV_t);

/**
 * @brief Set the linear system matrix for a HYPREDRV object.
 *
 * This function is responsible for setting the matrix data of a linear system associated
 * with a HYPREDRV object.
 *
 * @param hypredrv The HYPREDRV_t object for which the linear system matrix is to be associated.
 *
 * @param mat_A The HYPRE_Matrix object for the linear system matrix.
 *
 * @return Returns an error code with 0 indicating success. Any non-zero value indicates a failure,
 * and the error code can be further described using HYPREDRV_ErrorCodeDescribe(error_code).
 *
 * @note It's the caller's responsibility to ensure that the hypredrv parameter is a valid pointer to an
 * initialized HYPREDRV_t object. Passing a NULL or uninitialized object will result in an error.
 *
 * Example Usage:
 * @code
 *    HYPREDRV_t *hypredrv;
 *    // ... (hypredrv is created, and its components are initialized) ...
 *    HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetMatrix(hypredrv));
 * @endcode
 */

HYPREDRV_EXPORT_SYMBOL uint32_t
HYPREDRV_LinearSystemSetMatrix(HYPREDRV_t, HYPRE_Matrix);

/**
 * @brief Set the linear system right-hand side (RHS) vector from file for a HYPREDRV object.
 *
 * @param hypredrv The HYPREDRV_t object for which the RHS vector of the linear system is to be set.
 *
 * @param vec The HYPRE_Vector vector object representing the RHS of the linear system (NULL if set from input file).
 *
 * @return Returns an error code with 0 indicating success. Any non-zero value indicates a failure,
 * and the error code can be further described using HYPREDRV_ErrorCodeDescribe(error_code).
 *
 * @note It's the caller's responsibility to ensure that the hypredrv parameter is a valid pointer to an
 * initialized HYPREDRV_t object. Passing a NULL or uninitialized object will result in an error.
 *
 * Example Usage:
 * @code
 *    HYPREDRV_t *hypredrv;
 *    // ... (hypredrv is created, and its components are initialized) ...
 *    HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetRHS(hypredrv));
 * @endcode
 */

HYPREDRV_EXPORT_SYMBOL uint32_t
HYPREDRV_LinearSystemSetRHS(HYPREDRV_t, HYPRE_Vector);

/**
 * @brief Set the initial guess for the solution vector of the linear system for a HYPREDRV object.
 *
 * This function is responsible for setting the initial guess for the solution vector of a
 * linear system associated with a HYPREDRV object.
 *
 * @param hypredrv The HYPREDRV_t object for which the initial guess of the solution vector of the
 * linear system is to be set.
 *
 * @return Returns an error code with 0 indicating success. Any non-zero value indicates a failure,
 * and the error code can be further described using HYPREDRV_ErrorCodeDescribe(error_code).
 *
 * @note It's the caller's responsibility to ensure that the hypredrv parameter is a valid pointer to an
 * initialized HYPREDRV_t object. Passing a NULL or uninitialized object will result in an error.
 *
 * Example Usage:
 * @code
 *    HYPREDRV_t *hypredrv;
 *    // ... (hypredrv is created, and its components are initialized) ...
 *    HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetInitialGuess(hypredrv));
 * @endcode
 */

HYPREDRV_EXPORT_SYMBOL uint32_t
HYPREDRV_LinearSystemSetInitialGuess(HYPREDRV_t);

/**
 * @brief Reset the initial guess of the solution vector for a HYPREDRV object to its original
 * state as computed with \e HYPREDRV_LinearSystemSetInitialGuess
 *
 * @param hypredrv The HYPREDRV_t object for which the initial guess of the solution vector is to be
 * reset.
 *
 * @return Returns an error code with 0 indicating success. Any non-zero value indicates a failure,
 * and the error code can be further described using HYPREDRV_ErrorCodeDescribe(error_code).
 *
 * @note It's the caller's responsibility to ensure that the hypredrv parameter is a valid pointer to an
 * initialized HYPREDRV_t object. Passing a NULL or uninitialized object will result in an error.
 *
 * Example Usage:
 * @code
 *    HYPREDRV_t *hypredrv;
 *    // ... (hypredrv is created, and its components are initialized) ...
 *    HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemResetInitialGuess(hypredrv));
 * @endcode
 */

HYPREDRV_EXPORT_SYMBOL uint32_t
HYPREDRV_LinearSystemResetInitialGuess(HYPREDRV_t);

/**
 * @brief Set the matrix that is used to compute the preconditioner of a HYPREDRV object.
 *
 * By default, the preconditioning matrix is the same as the linear system matrix. However, it is
 * also possible to use an arbitrary matrix set by the user, e.g., a filtered version of the linear
 * system matrix.
 *
 * @param hypredrv The HYPREDRV_t object for which the preconditioner matrix is to be set.
 *
 * @return Returns an error code with 0 indicating success. Any non-zero value indicates a failure,
 * and the error code can be further described using HYPREDRV_ErrorCodeDescribe(error_code).
 *
 * @note It's the caller's responsibility to ensure that the hypredrv parameter is a valid pointer to an
 * initialized HYPREDRV_t object. Passing a NULL or uninitialized object will result in an error.
 *
 * Example Usage:
 * @code
 *    HYPREDRV_t *hypredrv;
 *    // ... (hypredrv is created, and its components are initialized) ...
 *    HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetPrecMatrix(hypredrv));
 * @endcode
 */

HYPREDRV_EXPORT_SYMBOL uint32_t
HYPREDRV_LinearSystemSetPrecMatrix(HYPREDRV_t);

/**
 * @brief Set the degree of freedom (DOF) map for the linear system of a HYPREDRV object.
 *
 * @param hypredrv The HYPREDRV_t object for which the DOF map of the linear system is to be set.
 *
 * @param size The local size (current rank) of the dofmap array.
 *
 * @param dofmap The array containing the mapping codes for the degrees of freedom
 *
 * @return Returns an error code with 0 indicating success. Any non-zero value indicates a failure,
 * and the error code can be further described using HYPREDRV_ErrorCodeDescribe(error_code).
 *
 * @note It's the caller's responsibility to ensure that the hypredrv parameter is a valid pointer to an
 * initialized HYPREDRV_t object. Passing a NULL or uninitialized object will result in an error.
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

HYPREDRV_EXPORT_SYMBOL uint32_t
HYPREDRV_LinearSystemSetDofmap(HYPREDRV_t, int, int*);

/**
 * @brief Set an interleaved degree of freedom (DOF) map for the linear system of a HYPREDRV object.
 *
 * @param hypredrv The HYPREDRV_t object for which the DOF map of the linear system is to be set.
 *
 * @param num_local_blocks The local (owned by the current rank) number of blocks (cells/nodes)
 * containing a set of num_dof_types degrees of freedom.
 *
 * @param num_dof_types The number of degree of freedom types.
 *
 * @return Returns an error code with 0 indicating success. Any non-zero value indicates a failure,
 * and the error code can be further described using HYPREDRV_ErrorCodeDescribe(error_code).
 *
 * @note It's the caller's responsibility to ensure that the hypredrv parameter is a valid pointer to an
 * initialized HYPREDRV_t object. Passing a NULL or uninitialized object will result in an error.
 *
 * Example Usage:
 * @code
 *    HYPREDRV_t *hypredrv;
 *    int num_dof_types = ... // Number of degree of freedom types
 *    int num_local_nodes = ... // Number of local (current MPI rank) nodes
 *    // ... (hypredrv is created, and its components are initialized) ...
 *    HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetInterleavedDofmap(hypredrv, num_local_dofs, num_dof_types));
 * @endcode
 */

HYPREDRV_EXPORT_SYMBOL uint32_t
HYPREDRV_LinearSystemSetInterleavedDofmap(HYPREDRV_t, int, int);

/**
 * @brief Read the degree of freedom (DOF) map for the linear system of a HYPREDRV object.
 *
 * @param hypredrv The HYPREDRV_t object for which the DOF map of the linear system is to be read.
 *
 * @return Returns an error code with 0 indicating success. Any non-zero value indicates a failure,
 * and the error code can be further described using HYPREDRV_ErrorCodeDescribe(error_code).
 *
 * @note It's the caller's responsibility to ensure that the hypredrv parameter is a valid pointer to an
 * initialized HYPREDRV_t object. Passing a NULL or uninitialized object will result in an error.
 *
 * Example Usage:
 * @code
 *    HYPREDRV_t *hypredrv;
 *    // ... (hypredrv is created, and its components are initialized) ...
 *    HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemReadDofmap(hypredrv));
 * @endcode
 */

HYPREDRV_EXPORT_SYMBOL uint32_t
HYPREDRV_LinearSystemReadDofmap(HYPREDRV_t);

/**
 * @brief Retrieves the solution values from the linear system of a HYPREDRV object.
 *
 * This function provides access to the internal pointer where the solution vector of the linear system
 * associated with the given HYPREDRV_t object is stored. It does not copy the solution values; instead,
 * it assigns the internal pointer to the user-provided pointer @p sol_data.
 *
 * @param obj A valid HYPREDRV_t object from which the internal solution pointer is to be retrieved.
 * @param sol_data A pointer to a HYPRE_Complex pointer, which will be set to point to the internal
 *                 solution data array. The user must not free or modify the internal array.
 *
 * @return Returns an error code, with 0 indicating success. If the @p obj parameter is invalid
 *         (e.g., NULL or uninitialized), an error code is returned, and the error can be further
 *         described using HYPREDRV_ErrorCodeDescribe(error_code).
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
HYPREDRV_LinearSystemGetSolutionValues(HYPREDRV_t, HYPRE_Complex**);

/**
 * @brief Retrieves the right-hand side values from the linear system of a HYPREDRV object.
 *
 * This function provides access to the internal pointer where the right-hand side vector of the linear system
 * associated with the given HYPREDRV_t object is stored. It does not copy the solution values; instead,
 * it assigns the internal pointer to the user-provided pointer @p rhs_data.
 *
 * @param obj A valid HYPREDRV_t object from which the internal solution pointer is to be retrieved.
 * @param rhs_data A pointer to a HYPRE_Complex pointer, which will be set to point to the internal
 *                 right-hand side data array. The user must not free or modify the internal array.
 *
 * @return Returns an error code, with 0 indicating success. If the @p obj parameter is invalid
 *         (e.g., NULL or uninitialized), an error code is returned, and the error can be further
 *         described using HYPREDRV_ErrorCodeDescribe(error_code).
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
HYPREDRV_LinearSystemGetRHSValues(HYPREDRV_t, HYPRE_Complex**);

/**
 * @brief Create a preconditioner for the HYPREDRV object based on the specified method.
 *
 * @param hypredrv The HYPREDRV_t object for which the preconditioner is to be created.
 *
 * @return Returns an error code with 0 indicating success. Any non-zero value indicates a failure,
 * and the error code can be further described using HYPREDRV_ErrorCodeDescribe(error_code).
 *
 * @note It's the caller's responsibility to ensure that the hypredrv parameter is a valid pointer to an
 * initialized HYPREDRV_t object. Passing a NULL or uninitialized object will result in an error.
 * The function assumes that the preconditioning method and other related settings are properly
 * set in the input arguments.
 *
 * Example Usage:
 * @code
 *    HYPREDRV_t *hypredrv;
 *    // ... (hypredrv is created, and its components are initialized) ...
 *    HYPREDRV_SAFE_CALL(HYPREDRV_PreconCreate(hypredrv));
 * @endcode
 */

HYPREDRV_EXPORT_SYMBOL uint32_t
HYPREDRV_PreconCreate(HYPREDRV_t);

/**
 * @brief Create a linear solver for the HYPREDRV object based on the specified method.
 *
 * @param hypredrv The HYPREDRV_t object for which the linear solver is to be created.
 *
 * @return Returns an error code with 0 indicating success. Any non-zero value indicates a failure,
 * and the error code can be further described using HYPREDRV_ErrorCodeDescribe(error_code).
 *
 * @note It's the caller's responsibility to ensure that the hypredrv parameter is a valid pointer to an
 * initialized HYPREDRV_t object. Passing a NULL or uninitialized object will result in an error.
 * The function assumes that the solver method and other related settings are properly set in
 * the input arguments.
 *
 * Example Usage:
 * @code
 *    HYPREDRV_t *hypredrv;
 *    // ... (hypredrv is created, and its components are initialized) ...
 *    HYPREDRV_SAFE_CALL(HYPREDRV_LinearSolverCreate(hypredrv));
 * @endcode
 */

HYPREDRV_EXPORT_SYMBOL uint32_t
HYPREDRV_LinearSolverCreate(HYPREDRV_t);

/**
 * @brief Set up the preconditioner for the HYPREDRV object based on the specified preconditioner methods.
 *
 * @param hypredrv The HYPREDRV_t object for which the preconditioner is to be set up.
 *
 * @return Returns an error code with 0 indicating success. Any non-zero value indicates a failure,
 * and the error code can be further described using HYPREDRV_ErrorCodeDescribe(error_code).
 *
 * @note It's the caller's responsibility to ensure that the hypredrv parameter is a valid pointer to an
 * initialized HYPREDRV_t object. Passing a NULL or uninitialized object will result in an error.
 * The function assumes that the preconditioner method and the matrix are properly set in the
 * input arguments.
 *
 * Example Usage:
 * @code
 *    HYPREDRV_t *hypredrv;
 *    // ... (hypredrv is created, and its components are initialized) ...
 *    HYPREDRV_SAFE_CALL(HYPREDRV_PreconSetup(hypredrv));
 * @endcode
 */

HYPREDRV_EXPORT_SYMBOL uint32_t
HYPREDRV_PreconSetup(HYPREDRV_t);

/**
 * @brief Set up the linear solver for the HYPREDRV object based on the specified solver
 * and preconditioner methods.
 *
 * @param hypredrv The HYPREDRV_t object for which the linear solver is to be set up.
 *
 * @return Returns an error code with 0 indicating success. Any non-zero value indicates a failure,
 * and the error code can be further described using HYPREDRV_ErrorCodeDescribe(error_code).
 *
 * @note It's the caller's responsibility to ensure that the hypredrv parameter is a valid pointer to an
 * initialized HYPREDRV_t object. Passing a NULL or uninitialized object will result in an error.
 * The function assumes that the solver and preconditioner methods, as well as the matrix,
 * RHS vector, and solution vector, are properly set in the input arguments.
 *
 * Example Usage:
 * @code
 *    HYPREDRV_t *hypredrv;
 *    // ... (hypredrv is created, and its components are initialized) ...
 *    HYPREDRV_SAFE_CALL(HYPREDRV_LinearSolverSetup(hypredrv));
 * @endcode
 */

HYPREDRV_EXPORT_SYMBOL uint32_t
HYPREDRV_LinearSolverSetup(HYPREDRV_t);

/**
 * @brief Apply the linear solver to solve the linear system for the HYPREDRV object.
 *
 * @param hypredrv The HYPREDRV_t object for which the linear solver is to be applied.
 *
 * @return Returns an error code with 0 indicating success. Any non-zero value indicates a failure,
 * and the error code can be further described using HYPREDRV_ErrorCodeDescribe(error_code).
 *
 * @note It's the caller's responsibility to ensure that the hypredrv parameter is a valid pointer to an
 * initialized HYPREDRV_t object. Passing a NULL or uninitialized object will result in an error.
 * The function assumes that the solver method, solver, matrix, RHS vector, and solution vector
 * are properly set in the input arguments.
 *
 * Example Usage:
 * @code
 *    HYPREDRV_t *hypredrv;
 *    // ... (hypredrv is created, and its components are initialized) ...
 *    HYPREDRV_SAFE_CALL(HYPREDRV_LinearSolverApply(hypredrv));
 * @endcode
 */

HYPREDRV_EXPORT_SYMBOL uint32_t
HYPREDRV_LinearSolverApply(HYPREDRV_t);

/**
 * @brief Apply the preconditioner associated with a HYPREDRV_t object to an input vector.
 *
 * @param hypredrv The HYPREDRV_t object defining the preconditioner to be applied.
 *
 * @param vec_b The HYPRE_Vector object defining the input vector.
 *
 * @param vec_x The HYPRE_Vector object defining the output vector.
 *
 * @return Returns an error code with 0 indicating success. Any non-zero value indicates a failure,
 * and the error code can be further described using HYPREDRV_ErrorCodeDescribe(error_code).
 *
 * @note It's the caller's responsibility to ensure that the hypredrv parameter is a valid pointer to an
 * initialized HYPREDRV_t object. Passing a NULL or uninitialized object will result in an error.
 * The function assumes that the preconditioner method, solver, matrix, RHS vector, and solution vector
 * are properly set in the input arguments.
 *
 * Example Usage:
 * @code
 *    HYPREDRV_t *hypredrv;
 *    // ... (hypredrv is created, and its components are initialized) ...
 *    HYPREDRV_SAFE_CALL(HYPREDRV_PreconApply(hypredrv, vec_b, vec_x));
 * @endcode
 */

HYPREDRV_EXPORT_SYMBOL uint32_t
HYPREDRV_PreconApply(HYPREDRV_t, HYPRE_Vector, HYPRE_Vector);

/**
 * @brief Destroy the preconditioner associated with the HYPREDRV object.
 *
 * @param hypredrv The HYPREDRV_t object whose preconditioner is to be destroyed.
 *
 * @return Returns an error code with 0 indicating success. Any non-zero value indicates a failure,
 * and the error code can be further described using HYPREDRV_ErrorCodeDescribe(error_code).
 *
 * @note It's the caller's responsibility to ensure that the hypredrv parameter is a valid pointer to an
 * initialized HYPREDRV_t object. Passing a NULL or uninitialized object will result in an error.
 * The function assumes that the preconditioner method and the preconditioner itself are properly
 * set in the HYPREDRV_t object.
 *
 * Example Usage:
 * @code
 *    HYPREDRV_t *hypredrv;
 *    // ... (hypredrv is created, and its components are initialized) ...
 *    HYPREDRV_SAFE_CALL(HYPREDRV_PreconDestroy(hypredrv));
 * @endcode
 */

HYPREDRV_EXPORT_SYMBOL uint32_t
HYPREDRV_PreconDestroy(HYPREDRV_t);

/**
 * @brief Destroy the linear solver associated with the HYPREDRV object.
 *
 * @param hypredrv The HYPREDRV_t object whose linear solver is to be destroyed.
 *
 * @return Returns an error code with 0 indicating success. Any non-zero value indicates a failure,
 * and the error code can be further described using HYPREDRV_ErrorCodeDescribe(error_code).
 *
 * @note It's the caller's responsibility to ensure that the hypredrv parameter is a valid pointer to an
 * initialized HYPREDRV_t object. Passing a NULL or uninitialized object will result in an error.
 * The function assumes that the linear solver method and the linear solver object itself are
 * properly set in the HYPREDRV_t object.
 *
 * Example Usage:
 * @code
 *    HYPREDRV_t *hypredrv;
 *    // ... (hypredrv is created, and its components are initialized) ...
 *    HYPREDRV_SAFE_CALL(HYPREDRV_LinearSolverDestroy(hypredrv));
 * @endcode
 */

HYPREDRV_EXPORT_SYMBOL uint32_t
HYPREDRV_LinearSolverDestroy(HYPREDRV_t);

/**
 * @brief Print the statistics associated with the HYPREDRV object.
 *
 * This function is responsible for printing the statistics collected during the operations
 * performed by the HYPREDRV object.
 *
 * @param hypredrv The HYPREDRV_t object whose statistics are to be printed.
 *
 * @return Returns an error code with 0 indicating success. Any non-zero value indicates a failure,
 * and the error code can be further described using HYPREDRV_ErrorCodeDescribe(error_code).
 *
 * @note It's the caller's responsibility to ensure that the hypredrv parameter is a valid pointer to an
 * initialized HYPREDRV_t object. Passing a NULL or uninitialized object will result in an error.
 *
 * Example Usage:
 * @code
 *    HYPREDRV_t *hypredrv;
 *    // ... (hypredrv is created, and its components are initialized) ...
 *    HYPREDRV_SAFE_CALL(HYPREDRV_StatsPrint(hypredrv));
 * @endcode
 */

HYPREDRV_EXPORT_SYMBOL uint32_t
HYPREDRV_StatsPrint(HYPREDRV_t);

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**@}*/

#ifdef __cplusplus
}
#endif

#endif /* ifndef HYPREDRV_HEADER */
