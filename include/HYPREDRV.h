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
 * provided MPI communicator. The newly created object is returned through the obj_ptr
 * parameter.
 *
 * @param comm The MPI communicator to be associated with the HYPREDRV object.
 * This communicator is used for parallel communications in the underlying HYPRE library calls.
 *
 * @param obj_ptr A pointer to the HYPREDRV_t pointer where the address of the newly
 * allocated HYPREDRV object will be stored. After the function call, *obj_ptr will
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
 *    HYPREDRV_t *obj;
 *    MPI_Comm comm = MPI_COMM_WORLD; // or any other valid MPI_Comm
 *    HYPREDRV_SAFE_CALL(HYPREDRV_Create(comm, &obj));
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
 * @param obj_ptr A pointer to the HYPREDRV_t object that is to be destroyed. This pointer
 * should have been initialized by HYPREDRV_Create function.
 *
 * @return Returns an error code with 0 indicating success. Any non-zero value indicates a failure,
 * and the error code can be further described using HYPREDRV_ErrorCodeDescribe(error_code).
 *
 * @note It is the caller's responsibility to ensure that obj_ptr is a valid pointer to a
 * HYPREDRV_t object. Passing an invalid pointer or a pointer to an already destroyed object
 * can lead to undefined behavior.
 *
 * Example Usage:
 * @code
 *    HYPREDRV_t *obj;
 *    // ... (obj is created and used) ...
 *    HYPREDRV_SAFE_CALL(HYPREDRV_Destroy(&obj));
 *    // obj is now NULL.
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
 * @param obj The HYPREDRV_t object whose input arguments are to be parsed and set.
 *
 * @return Returns an error code with 0 indicating success. Any non-zero value indicates a failure,
 * and the error code can be further described using HYPREDRV_ErrorCodeDescribe(error_code).
 *
 * @note It is expected that argv[1] is the name of the input file in YAML format
 *
 * Example Usage:
 * @code
 *    HYPREDRV_t *obj;
 *    int argc = ...; // Number of arguments
 *    char **argv = ...; // Argument strings
 *    // ... (obj is created) ...
 *    HYPREDRV_SAFE_CALL(HYPREDRV_InputArgsParse(argc, argv, &obj));
 * @endcode
 */

HYPREDRV_EXPORT_SYMBOL uint32_t
HYPREDRV_InputArgsParse(int, char**, HYPREDRV_t);

/**
 * @brief Set HYPRE's global options according to the YAML input.
 *
 * @param obj The HYPREDRV_t object from which the global options are to be retrieved.
 *
 * @return Returns an error code with 0 indicating success. Any non-zero value indicates a failure,
 * and the error code can be further described using HYPREDRV_ErrorCodeDescribe(error_code).
 *
 * Example Usage:
 * @code
 *    HYPREDRV_t *obj;
 *    // ... (obj is created, and its components are initialized) ...
 *    HYPREDRV_SAFE_CALL(HYPREDRV_SetGlobalOptions(obj));
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
 * @param obj The HYPREDRV_t object from which the warmup setting is to be retrieved.
 *
 * @return Returns the warmup setting as an integer. If the input object is NULL
 * or not properly initialized, the function returns -1 to indicate an error or invalid state.
 *
 * Example Usage:
 * @code
 *    HYPREDRV_t *obj;
 *    // ... (obj is created, and its components are set) ...
 *    int warmup = HYPREDRV_InputArgsGetWarmup(obj);
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
 * @param obj The HYPREDRV_t object from which the number of repetitions is to be retrieved.
 *
 * @return Returns the number of repetitions as an integer. If the input object is NULL
 * or not properly initialized, the function returns -1 to indicate an error or invalid state.
 *
 * Example Usage:
 * @code
 *    HYPREDRV_t *obj;
 *    // ... (obj is created, and its components are set) ...
 *    int num_reps = HYPREDRV_InputArgsGetNumRepetitions(obj);
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
 * @param obj The HYPREDRV_t object from which the number of repetitions is to be retrieved.
 *
 * @return Returns the number of linear systems as an integer. If the input object is NULL
 * or not properly initialized, the function returns -1 to indicate an error or invalid state.
 *
 * Example Usage:
 * @code
 *    HYPREDRV_t *obj;
 *    // ... (obj is created, and its components are set) ...
 *    int num_ls = HYPREDRV_InputArgsGetNumLinearSystems(obj);
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
 * @param obj The HYPREDRV_t object for which the linear system matrix is to be built.
 *
 * @return Returns an error code with 0 indicating success. Any non-zero value indicates a failure,
 * and the error code can be further described using HYPREDRV_ErrorCodeDescribe(error_code).
 *
 * @note It's the caller's responsibility to ensure that the obj parameter is a valid pointer to an
 * initialized HYPREDRV_t object. Passing a NULL or uninitialized object will result in an error.
 *
 * Example Usage:
 * @code
 *    HYPREDRV_t *obj;
 *    // ... (obj is created, and its components are initialized) ...
 *    HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemBuild(obj));
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
 * @param obj The HYPREDRV_t object for which the linear system matrix is to be read.
 *
 * @return Returns an error code with 0 indicating success. Any non-zero value indicates a failure,
 * and the error code can be further described using HYPREDRV_ErrorCodeDescribe(error_code).
 *
 * @note It's the caller's responsibility to ensure that the obj parameter is a valid pointer to an
 * initialized HYPREDRV_t object. Passing a NULL or uninitialized object will result in an error.
 *
 * Example Usage:
 * @code
 *    HYPREDRV_t *obj;
 *    // ... (obj is created, and its components are initialized) ...
 *    HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemReadMatrix(obj));
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
 * @param obj The HYPREDRV_t object for which the linear system matrix is to be associated.
 *
 * @param mat_A The HYPRE_Matrix object for the linear system matrix.
 *
 * @return Returns an error code with 0 indicating success. Any non-zero value indicates a failure,
 * and the error code can be further described using HYPREDRV_ErrorCodeDescribe(error_code).
 *
 * @note It's the caller's responsibility to ensure that the obj parameter is a valid pointer to an
 * initialized HYPREDRV_t object. Passing a NULL or uninitialized object will result in an error.
 *
 * Example Usage:
 * @code
 *    HYPREDRV_t *obj;
 *    // ... (obj is created, and its components are initialized) ...
 *    HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetMatrix(obj));
 * @endcode
 */

HYPREDRV_EXPORT_SYMBOL uint32_t
HYPREDRV_LinearSystemSetMatrix(HYPREDRV_t, HYPRE_Matrix);

/**
 * @brief Set the linear system right-hand side (RHS) vector from file for a HYPREDRV object.
 *
 * @param obj The HYPREDRV_t object for which the RHS vector of the linear system is to be set.
 *
 * @param vec The HYPRE_Vector vector object representing the RHS of the linear system (NULL if set from input file).
 *
 * @return Returns an error code with 0 indicating success. Any non-zero value indicates a failure,
 * and the error code can be further described using HYPREDRV_ErrorCodeDescribe(error_code).
 *
 * @note It's the caller's responsibility to ensure that the obj parameter is a valid pointer to an
 * initialized HYPREDRV_t object. Passing a NULL or uninitialized object will result in an error.
 *
 * Example Usage:
 * @code
 *    HYPREDRV_t *obj;
 *    // ... (obj is created, and its components are initialized) ...
 *    HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetRHS(obj));
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
 * @param obj The HYPREDRV_t object for which the initial guess of the solution vector of the
 * linear system is to be set.
 *
 * @return Returns an error code with 0 indicating success. Any non-zero value indicates a failure,
 * and the error code can be further described using HYPREDRV_ErrorCodeDescribe(error_code).
 *
 * @note It's the caller's responsibility to ensure that the obj parameter is a valid pointer to an
 * initialized HYPREDRV_t object. Passing a NULL or uninitialized object will result in an error.
 *
 * Example Usage:
 * @code
 *    HYPREDRV_t *obj;
 *    // ... (obj is created, and its components are initialized) ...
 *    HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetInitialGuess(obj));
 * @endcode
 */

HYPREDRV_EXPORT_SYMBOL uint32_t
HYPREDRV_LinearSystemSetInitialGuess(HYPREDRV_t);

/**
 * @brief Reset the initial guess of the solution vector for a HYPREDRV object to its original
 * state as computed with \e HYPREDRV_LinearSystemSetInitialGuess
 *
 * @param obj The HYPREDRV_t object for which the initial guess of the solution vector is to be
 * reset.
 *
 * @return Returns an error code with 0 indicating success. Any non-zero value indicates a failure,
 * and the error code can be further described using HYPREDRV_ErrorCodeDescribe(error_code).
 *
 * @note It's the caller's responsibility to ensure that the obj parameter is a valid pointer to an
 * initialized HYPREDRV_t object. Passing a NULL or uninitialized object will result in an error.
 *
 * Example Usage:
 * @code
 *    HYPREDRV_t *obj;
 *    // ... (obj is created, and its components are initialized) ...
 *    HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemResetInitialGuess(obj));
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
 * @param obj The HYPREDRV_t object for which the preconditioner matrix is to be set.
 *
 * @return Returns an error code with 0 indicating success. Any non-zero value indicates a failure,
 * and the error code can be further described using HYPREDRV_ErrorCodeDescribe(error_code).
 *
 * @note It's the caller's responsibility to ensure that the obj parameter is a valid pointer to an
 * initialized HYPREDRV_t object. Passing a NULL or uninitialized object will result in an error.
 *
 * Example Usage:
 * @code
 *    HYPREDRV_t *obj;
 *    // ... (obj is created, and its components are initialized) ...
 *    HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetPrecMatrix(obj));
 * @endcode
 */

HYPREDRV_EXPORT_SYMBOL uint32_t
HYPREDRV_LinearSystemSetPrecMatrix(HYPREDRV_t);

/**
 * @brief Set the degree of freedom (DOF) map for the linear system of a HYPREDRV object.
 *
 * @param obj The HYPREDRV_t object for which the DOF map of the linear system is to be set.
 *
 * @param size The local size (current rank) of the dofmap array.
 *
 * @param dofmap The array containing the mapping codes for the degrees of freedom
 *
 * @return Returns an error code with 0 indicating success. Any non-zero value indicates a failure,
 * and the error code can be further described using HYPREDRV_ErrorCodeDescribe(error_code).
 *
 * @note It's the caller's responsibility to ensure that the obj parameter is a valid pointer to an
 * initialized HYPREDRV_t object. Passing a NULL or uninitialized object will result in an error.
 *
 * Example Usage:
 * @code
 *    HYPREDRV_t *obj;
 *    int size = ... // Size of dofmap is set
 *    int *dofmap = ... // dofmap array is set
 *    // ... (obj is created, and its components are initialized) ...
 *    HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetDofmap(obj, size, dofmap));
 * @endcode
 */

HYPREDRV_EXPORT_SYMBOL uint32_t
HYPREDRV_LinearSystemSetDofmap(HYPREDRV_t, int, int*);

/**
 * @brief Read the degree of freedom (DOF) map for the linear system of a HYPREDRV object.
 *
 * @param obj The HYPREDRV_t object for which the DOF map of the linear system is to be read.
 *
 * @return Returns an error code with 0 indicating success. Any non-zero value indicates a failure,
 * and the error code can be further described using HYPREDRV_ErrorCodeDescribe(error_code).
 *
 * @note It's the caller's responsibility to ensure that the obj parameter is a valid pointer to an
 * initialized HYPREDRV_t object. Passing a NULL or uninitialized object will result in an error.
 *
 * Example Usage:
 * @code
 *    HYPREDRV_t *obj;
 *    // ... (obj is created, and its components are initialized) ...
 *    HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemReadDofmap(obj));
 * @endcode
 */

HYPREDRV_EXPORT_SYMBOL uint32_t
HYPREDRV_LinearSystemReadDofmap(HYPREDRV_t);

/**
 * @brief Create a preconditioner for the HYPREDRV object based on the specified method.
 *
 * @param obj The HYPREDRV_t object for which the preconditioner is to be created.
 *
 * @return Returns an error code with 0 indicating success. Any non-zero value indicates a failure,
 * and the error code can be further described using HYPREDRV_ErrorCodeDescribe(error_code).
 *
 * @note It's the caller's responsibility to ensure that the obj parameter is a valid pointer to an
 * initialized HYPREDRV_t object. Passing a NULL or uninitialized object will result in an error.
 * The function assumes that the preconditioning method and other related settings are properly
 * set in the input arguments.
 *
 * Example Usage:
 * @code
 *    HYPREDRV_t *obj;
 *    // ... (obj is created, and its components are initialized) ...
 *    HYPREDRV_SAFE_CALL(HYPREDRV_PreconCreate(obj));
 * @endcode
 */

HYPREDRV_EXPORT_SYMBOL uint32_t
HYPREDRV_PreconCreate(HYPREDRV_t);

/**
 * @brief Create a linear solver for the HYPREDRV object based on the specified method.
 *
 * @param obj The HYPREDRV_t object for which the linear solver is to be created.
 *
 * @return Returns an error code with 0 indicating success. Any non-zero value indicates a failure,
 * and the error code can be further described using HYPREDRV_ErrorCodeDescribe(error_code).
 *
 * @note It's the caller's responsibility to ensure that the obj parameter is a valid pointer to an
 * initialized HYPREDRV_t object. Passing a NULL or uninitialized object will result in an error.
 * The function assumes that the solver method and other related settings are properly set in
 * the input arguments.
 *
 * Example Usage:
 * @code
 *    HYPREDRV_t *obj;
 *    // ... (obj is created, and its components are initialized) ...
 *    HYPREDRV_SAFE_CALL(HYPREDRV_LinearSolverCreate(obj));
 * @endcode
 */

HYPREDRV_EXPORT_SYMBOL uint32_t
HYPREDRV_LinearSolverCreate(HYPREDRV_t);

/**
 * @brief Set up the preconditioner for the HYPREDRV object based on the specified preconditioner methods.
 *
 * @param obj The HYPREDRV_t object for which the preconditioner is to be set up.
 *
 * @return Returns an error code with 0 indicating success. Any non-zero value indicates a failure,
 * and the error code can be further described using HYPREDRV_ErrorCodeDescribe(error_code).
 *
 * @note It's the caller's responsibility to ensure that the obj parameter is a valid pointer to an
 * initialized HYPREDRV_t object. Passing a NULL or uninitialized object will result in an error.
 * The function assumes that the preconditioner method and the matrix are properly set in the
 * input arguments.
 *
 * Example Usage:
 * @code
 *    HYPREDRV_t *obj;
 *    // ... (obj is created, and its components are initialized) ...
 *    HYPREDRV_SAFE_CALL(HYPREDRV_PreconSetup(obj));
 * @endcode
 */

HYPREDRV_EXPORT_SYMBOL uint32_t
HYPREDRV_PreconSetup(HYPREDRV_t);

/**
 * @brief Set up the linear solver for the HYPREDRV object based on the specified solver
 * and preconditioner methods.
 *
 * @param obj The HYPREDRV_t object for which the linear solver is to be set up.
 *
 * @return Returns an error code with 0 indicating success. Any non-zero value indicates a failure,
 * and the error code can be further described using HYPREDRV_ErrorCodeDescribe(error_code).
 *
 * @note It's the caller's responsibility to ensure that the obj parameter is a valid pointer to an
 * initialized HYPREDRV_t object. Passing a NULL or uninitialized object will result in an error.
 * The function assumes that the solver and preconditioner methods, as well as the matrix,
 * RHS vector, and solution vector, are properly set in the input arguments.
 *
 * Example Usage:
 * @code
 *    HYPREDRV_t *obj;
 *    // ... (obj is created, and its components are initialized) ...
 *    HYPREDRV_SAFE_CALL(HYPREDRV_LinearSolverSetup(obj));
 * @endcode
 */

HYPREDRV_EXPORT_SYMBOL uint32_t
HYPREDRV_LinearSolverSetup(HYPREDRV_t);

/**
 * @brief Apply the linear solver to solve the linear system for the HYPREDRV object.
 *
 * @param obj The HYPREDRV_t object for which the linear solver is to be applied.
 *
 * @return Returns an error code with 0 indicating success. Any non-zero value indicates a failure,
 * and the error code can be further described using HYPREDRV_ErrorCodeDescribe(error_code).
 *
 * @note It's the caller's responsibility to ensure that the obj parameter is a valid pointer to an
 * initialized HYPREDRV_t object. Passing a NULL or uninitialized object will result in an error.
 * The function assumes that the solver method, solver, matrix, RHS vector, and solution vector
 * are properly set in the input arguments.
 *
 * Example Usage:
 * @code
 *    HYPREDRV_t *obj;
 *    // ... (obj is created, and its components are initialized) ...
 *    HYPREDRV_SAFE_CALL(HYPREDRV_LinearSolverApply(obj));
 * @endcode
 */

HYPREDRV_EXPORT_SYMBOL uint32_t
HYPREDRV_LinearSolverApply(HYPREDRV_t);

/**
 * @brief Apply the preconditioner associated with a HYPREDRV_t object to an input vector.
 *
 * @param obj The HYPREDRV_t object defining the preconditioner to be applied.
 *
 * @param vec_b The HYPRE_Vector object defining the input vector.
 *
 * @param vec_x The HYPRE_Vector object defining the output vector.
 *
 * @return Returns an error code with 0 indicating success. Any non-zero value indicates a failure,
 * and the error code can be further described using HYPREDRV_ErrorCodeDescribe(error_code).
 *
 * @note It's the caller's responsibility to ensure that the obj parameter is a valid pointer to an
 * initialized HYPREDRV_t object. Passing a NULL or uninitialized object will result in an error.
 * The function assumes that the preconditioner method, solver, matrix, RHS vector, and solution vector
 * are properly set in the input arguments.
 *
 * Example Usage:
 * @code
 *    HYPREDRV_t *obj;
 *    // ... (obj is created, and its components are initialized) ...
 *    HYPREDRV_SAFE_CALL(HYPREDRV_PreconApply(obj));
 * @endcode
 */

HYPREDRV_EXPORT_SYMBOL uint32_t
HYPREDRV_PreconApply(HYPREDRV_t, HYPRE_Vector, HYPRE_Vector);

/**
 * @brief Destroy the preconditioner associated with the HYPREDRV object.
 *
 * @param obj The HYPREDRV_t object whose preconditioner is to be destroyed.
 *
 * @return Returns an error code with 0 indicating success. Any non-zero value indicates a failure,
 * and the error code can be further described using HYPREDRV_ErrorCodeDescribe(error_code).
 *
 * @note It's the caller's responsibility to ensure that the obj parameter is a valid pointer to an
 * initialized HYPREDRV_t object. Passing a NULL or uninitialized object will result in an error.
 * The function assumes that the preconditioner method and the preconditioner itself are properly
 * set in the HYPREDRV_t object.
 *
 * Example Usage:
 * @code
 *    HYPREDRV_t *obj;
 *    // ... (obj is created, and its components are initialized) ...
 *    HYPREDRV_SAFE_CALL(HYPREDRV_PreconDestroy(obj));
 * @endcode
 */

HYPREDRV_EXPORT_SYMBOL uint32_t
HYPREDRV_PreconDestroy(HYPREDRV_t);

/**
 * @brief Destroy the linear solver associated with the HYPREDRV object.
 *
 * @param obj The HYPREDRV_t object whose linear solver is to be destroyed.
 *
 * @return Returns an error code with 0 indicating success. Any non-zero value indicates a failure,
 * and the error code can be further described using HYPREDRV_ErrorCodeDescribe(error_code).
 *
 * @note It's the caller's responsibility to ensure that the obj parameter is a valid pointer to an
 * initialized HYPREDRV_t object. Passing a NULL or uninitialized object will result in an error.
 * The function assumes that the linear solver method and the linear solver object itself are
 * properly set in the HYPREDRV_t object.
 *
 * Example Usage:
 * @code
 *    HYPREDRV_t *obj;
 *    // ... (obj is created, and its components are initialized) ...
 *    HYPREDRV_SAFE_CALL(HYPREDRV_LinearSolverDestroy(obj));
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
 * @param obj The HYPREDRV_t object whose statistics are to be printed.
 *
 * @return Returns an error code with 0 indicating success. Any non-zero value indicates a failure,
 * and the error code can be further described using HYPREDRV_ErrorCodeDescribe(error_code).
 *
 * @note It's the caller's responsibility to ensure that the obj parameter is a valid pointer to an
 * initialized HYPREDRV_t object. Passing a NULL or uninitialized object will result in an error.
 *
 * Example Usage:
 * @code
 *    HYPREDRV_t *obj;
 *    // ... (obj is created, and its components are initialized) ...
 *    HYPREDRV_SAFE_CALL(HYPREDRV_StatsPrint(obj));
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
