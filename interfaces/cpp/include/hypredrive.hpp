/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef HYPREDRIVE_CXX_HPP
#define HYPREDRIVE_CXX_HPP

#include <HYPREDRV.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <istream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace hypredrive
{

using status  = std::uint32_t;
using bigint  = HYPRE_BigInt;
using real    = HYPRE_Real;
using complex = HYPRE_Complex;

using bigint_vector  = std::vector<bigint>;
using real_vector    = std::vector<real>;
using complex_vector = std::vector<complex>;

namespace detail
{

inline bool
looks_like_yaml_file_path(std::string_view value)
{
   return !value.empty() && value.find('\0') == std::string_view::npos &&
          value.find('\n') == std::string_view::npos &&
          value.find('\r') == std::string_view::npos &&
          value.find(':') == std::string_view::npos &&
          value.find('{') == std::string_view::npos &&
          value.find('[') == std::string_view::npos;
}

} // namespace detail

class error : public std::runtime_error
{
 public:
   error(status code, const std::string &where)
      : std::runtime_error(where + " failed with HYPREDRV status " +
                           std::to_string(code)),
        code_(code)
   {
   }

   status
   code() const noexcept
   {
      return code_;
   }

 private:
   status code_;
};

inline void
check(status code, const char *where)
{
   if (code != HYPREDRV_SUCCESS)
   {
      throw error(code, where ? where : "HYPREDRV call");
   }
}

// Intended for direct HYPREDRV_* calls; #call is used as exception context.
#define HYPREDRIVE_CXX_CHECK(call) ::hypredrive::check((call), #call)
/// @brief C++ wrapper for HYPREDRV_Initialize.
/// @see HYPREDRV_Initialize
inline void
initialize()
{
   HYPREDRIVE_CXX_CHECK(HYPREDRV_Initialize());
}
/// @brief C++ wrapper for HYPREDRV_Finalize.
/// @see HYPREDRV_Finalize
inline void
finalize()
{
   HYPREDRIVE_CXX_CHECK(HYPREDRV_Finalize());
}
/// @brief C++ wrapper for HYPREDRV_ErrorCodeDescribe.
/// @see HYPREDRV_ErrorCodeDescribe
inline void
describe_error(status code)
{
   HYPREDRV_ErrorCodeDescribe(code);
}
/// @brief C++ wrapper for HYPREDRV_ErrorInvalidValue.
/// @see HYPREDRV_ErrorInvalidValue
inline void
throw_invalid_value(const char *message)
{
   HYPREDRIVE_CXX_CHECK(HYPREDRV_ErrorInvalidValue(message));
}
/// @brief C++ wrapper for HYPREDRV_PrintLibInfo.
/// @see HYPREDRV_PrintLibInfo
inline void
print_lib_info(MPI_Comm comm, bool print_datetime = true)
{
   HYPREDRIVE_CXX_CHECK(HYPREDRV_PrintLibInfo(comm, print_datetime ? 1 : 0));
}
/// @brief C++ wrapper for HYPREDRV_PrintSystemInfo.
/// @see HYPREDRV_PrintSystemInfo
inline void
print_system_info(MPI_Comm comm)
{
   HYPREDRIVE_CXX_CHECK(HYPREDRV_PrintSystemInfo(comm));
}
/// @brief C++ wrapper for HYPREDRV_PrintExitInfo.
/// @see HYPREDRV_PrintExitInfo
inline void
print_exit_info(MPI_Comm comm, const char *argv0)
{
   HYPREDRIVE_CXX_CHECK(HYPREDRV_PrintExitInfo(comm, argv0));
}
/// @brief C++ wrapper for HYPREDRV_SolverPresetRegister.
/// @see HYPREDRV_SolverPresetRegister
inline void
register_solver_preset(const char *name, const char *yaml_text, const char *help)
{
   HYPREDRIVE_CXX_CHECK(HYPREDRV_SolverPresetRegister(name, yaml_text, help));
}
/// @brief C++ wrapper for HYPREDRV_PreconPresetRegister.
/// @see HYPREDRV_PreconPresetRegister
inline void
register_precon_preset(const char *name, const char *yaml_text, const char *help)
{
   HYPREDRIVE_CXX_CHECK(HYPREDRV_PreconPresetRegister(name, yaml_text, help));
}

class driver
{
 public:
   /// @brief C++ RAII constructor wrapping HYPREDRV_Create.
   /// @see HYPREDRV_Create
   explicit driver(MPI_Comm comm = MPI_COMM_SELF)
   {
      HYPREDRIVE_CXX_CHECK(HYPREDRV_Create(comm, &handle_));
   }
   /// @brief Best-effort RAII destructor wrapping HYPREDRV_Destroy.
   /// @see HYPREDRV_Destroy
   ~driver() noexcept
   {
      if (handle_)
      {
         // Destructors cannot throw; HYPREDRV_Destroy owns solver/system teardown.
         (void)HYPREDRV_Destroy(&handle_);
      }
   }
   driver(const driver &)            = delete;
   driver &operator=(const driver &) = delete;
   driver(driver &&other) noexcept : handle_(other.handle_)
   {
      other.handle_ = nullptr;
   }
   driver &
   operator=(driver &&other) noexcept
   {
      if (this != &other)
      {
         if (handle_)
         {
            // Move assignment is best-effort for the previously owned handle.
            (void)HYPREDRV_Destroy(&handle_);
         }
         handle_       = other.handle_;
         other.handle_ = nullptr;
      }
      return *this;
   }

   /// @brief Return the owned raw HYPREDRV_t without transferring ownership.
   HYPREDRV_t
   native_handle() const noexcept
   {
      return handle_;
   }
   /// @brief Release ownership of the raw HYPREDRV_t to the caller.
   HYPREDRV_t
   release() noexcept
   {
      HYPREDRV_t out = handle_;
      handle_        = nullptr;
      return out;
   }
   /// @brief C++ wrapper for HYPREDRV_Destroy.
   /// @see HYPREDRV_Destroy
   void
   destroy()
   {
      HYPREDRIVE_CXX_CHECK(HYPREDRV_Destroy(&handle_));
   }
   /// @brief C++ wrapper for HYPREDRV_InputArgsParse.
   /// @see HYPREDRV_InputArgsParse
   void
   parse_args(int argc, char **argv)
   {
      HYPREDRIVE_CXX_CHECK(HYPREDRV_InputArgsParse(argc, argv, handle_));
   }
   /// @brief Parse inline YAML options from a string view.
   /// @see HYPREDRV_InputArgsParse
   void
   parse_yaml(std::string_view yaml)
   {
      if (yaml.find('\0') != std::string_view::npos)
      {
         throw std::invalid_argument(
            "hypredrive YAML text must not contain embedded NUL bytes");
      }
      if (detail::looks_like_yaml_file_path(yaml))
      {
         parse_yaml(std::filesystem::path(std::string(yaml)));
         return;
      }
      std::vector<char> text(yaml.begin(), yaml.end());
      text.push_back('\0');
      // HYPREDRV_InputArgsParse accepts argc=1 with argv[0] containing inline YAML.
      // The payload is passed as a C string, so embedded NUL bytes are rejected above.
      char *argv[] = {text.data()};
      HYPREDRIVE_CXX_CHECK(HYPREDRV_InputArgsParse(1, argv, handle_));
   }
   /// @brief Parse YAML options from a string.
   /// @see HYPREDRV_InputArgsParse
   void
   parse_yaml(const std::string &yaml)
   {
      parse_yaml(std::string_view(yaml));
   }
   /// @brief Parse inline YAML options from a null-terminated C string.
   /// @see HYPREDRV_InputArgsParse
   void
   parse_yaml(const char *yaml)
   {
      if (!yaml)
      {
         throw std::invalid_argument("hypredrive YAML text must not be null");
      }
      parse_yaml(std::string_view(yaml));
   }
   /// @brief Parse inline YAML options from an input stream.
   /// @see HYPREDRV_InputArgsParse
   void
   parse_yaml(std::istream &yaml)
   {
      if (!yaml)
      {
         throw std::invalid_argument("hypredrive YAML input stream is not readable");
      }
      std::ostringstream text;
      text << yaml.rdbuf();
      if (yaml.bad())
      {
         throw std::runtime_error("failed to read hypredrive YAML input stream");
      }
      parse_yaml(text.str());
   }
   /// @brief Parse YAML options from a file path.
   /// @see HYPREDRV_InputArgsParse
   void
   parse_yaml(const std::filesystem::path &path)
   {
      std::ifstream yaml(path);
      if (!yaml)
      {
         throw std::runtime_error("failed to open hypredrive YAML file: " +
                                  path.string());
      }
      parse_yaml(yaml);
   }
   /// @brief C++ wrapper for HYPREDRV_SetLibraryMode.
   /// @see HYPREDRV_SetLibraryMode
   void
   set_library_mode()
   {
      HYPREDRIVE_CXX_CHECK(HYPREDRV_SetLibraryMode(handle_));
   }
   /// @brief C++ wrapper for HYPREDRV_ObjectSetName.
   /// @see HYPREDRV_ObjectSetName
   void
   set_object_name(const char *name)
   {
      HYPREDRIVE_CXX_CHECK(HYPREDRV_ObjectSetName(handle_, name));
   }
   /// @brief C++ wrapper for HYPREDRV_InputArgsGetWarmup.
   /// @see HYPREDRV_InputArgsGetWarmup
   int
   get_warmup()
   {
      int v = 0;
      HYPREDRIVE_CXX_CHECK(HYPREDRV_InputArgsGetWarmup(handle_, &v));
      return v;
   }
   /// @brief C++ wrapper for HYPREDRV_InputArgsGetNumRepetitions.
   /// @see HYPREDRV_InputArgsGetNumRepetitions
   int
   get_num_repetitions()
   {
      int v = 0;
      HYPREDRIVE_CXX_CHECK(HYPREDRV_InputArgsGetNumRepetitions(handle_, &v));
      return v;
   }
   /// @brief C++ wrapper for HYPREDRV_InputArgsGetNumLinearSystems.
   /// @see HYPREDRV_InputArgsGetNumLinearSystems
   int
   get_num_linear_systems()
   {
      int v = 0;
      HYPREDRIVE_CXX_CHECK(HYPREDRV_InputArgsGetNumLinearSystems(handle_, &v));
      return v;
   }
   /// @brief C++ wrapper for HYPREDRV_InputArgsGetNumPreconVariants.
   /// @see HYPREDRV_InputArgsGetNumPreconVariants
   int
   get_num_precon_variants()
   {
      int v = 0;
      HYPREDRIVE_CXX_CHECK(HYPREDRV_InputArgsGetNumPreconVariants(handle_, &v));
      return v;
   }
   /// @brief C++ wrapper for HYPREDRV_InputArgsSetPreconVariant.
   /// @see HYPREDRV_InputArgsSetPreconVariant
   void
   set_precon_variant(int i)
   {
      HYPREDRIVE_CXX_CHECK(HYPREDRV_InputArgsSetPreconVariant(handle_, i));
   }
   /// @brief C++ wrapper for HYPREDRV_InputArgsSetPreconPreset.
   /// @see HYPREDRV_InputArgsSetPreconPreset
   void
   set_precon_preset(const char *p)
   {
      HYPREDRIVE_CXX_CHECK(HYPREDRV_InputArgsSetPreconPreset(handle_, p));
   }
   /// @brief C++ wrapper for HYPREDRV_InputArgsSetSolverPreset.
   /// @see HYPREDRV_InputArgsSetSolverPreset
   void
   set_solver_preset(const char *p)
   {
      HYPREDRIVE_CXX_CHECK(HYPREDRV_InputArgsSetSolverPreset(handle_, p));
   }
   /// @brief C++ wrapper for HYPREDRV_LinearSystemBuild.
   /// @see HYPREDRV_LinearSystemBuild
   void
   build_system()
   {
      HYPREDRIVE_CXX_CHECK(HYPREDRV_LinearSystemBuild(handle_));
   }
   /// @brief C++ wrapper for HYPREDRV_LinearSystemReadMatrix.
   /// @see HYPREDRV_LinearSystemReadMatrix
   void
   read_matrix()
   {
      HYPREDRIVE_CXX_CHECK(HYPREDRV_LinearSystemReadMatrix(handle_));
   }
   /// @brief C++ wrapper for HYPREDRV_LinearSystemSetMatrix.
   /// @see HYPREDRV_LinearSystemSetMatrix
   void
   set_matrix(HYPRE_Matrix m)
   {
      HYPREDRIVE_CXX_CHECK(HYPREDRV_LinearSystemSetMatrix(handle_, m));
   }
   /// @brief C++ wrapper for HYPREDRV_LinearSystemSetRHS.
   /// @see HYPREDRV_LinearSystemSetRHS
   void
   set_rhs(HYPRE_Vector v)
   {
      HYPREDRIVE_CXX_CHECK(HYPREDRV_LinearSystemSetRHS(handle_, v));
   }
   /// @brief C++ wrapper for HYPREDRV_LinearSystemSetMatrixFromCSR.
   /// @see HYPREDRV_LinearSystemSetMatrixFromCSR
   void
   set_matrix_from_csr(bigint rs, bigint re, const bigint *indptr, const bigint *cols,
                       const real *data)
   {
      HYPREDRIVE_CXX_CHECK(
         HYPREDRV_LinearSystemSetMatrixFromCSR(handle_, rs, re, indptr, cols, data));
   }
   /// @brief C++ wrapper for HYPREDRV_LinearSystemSetRHSFromArray.
   /// @see HYPREDRV_LinearSystemSetRHSFromArray
   void
   set_rhs_from_array(bigint rs, bigint re, const real *values)
   {
      HYPREDRIVE_CXX_CHECK(HYPREDRV_LinearSystemSetRHSFromArray(handle_, rs, re, values));
   }
   /// @brief C++ wrapper for HYPREDRV_LinearSystemSetInitialGuess.
   /// @see HYPREDRV_LinearSystemSetInitialGuess
   void
   set_initial_guess(HYPRE_Vector v)
   {
      HYPREDRIVE_CXX_CHECK(HYPREDRV_LinearSystemSetInitialGuess(handle_, v));
   }
   /// @brief C++ wrapper for HYPREDRV_LinearSystemSetInitialGuess.
   /// @see HYPREDRV_LinearSystemSetInitialGuess
   void
   set_zero_initial_guess()
   {
      HYPREDRIVE_CXX_CHECK(HYPREDRV_LinearSystemSetInitialGuess(handle_, nullptr));
   }
   /// @brief C++ wrapper for HYPREDRV_LinearSystemSetSolution.
   /// @see HYPREDRV_LinearSystemSetSolution
   void
   set_solution(HYPRE_Vector v)
   {
      HYPREDRIVE_CXX_CHECK(HYPREDRV_LinearSystemSetSolution(handle_, v));
   }
   /// @brief C++ wrapper for HYPREDRV_LinearSystemSetReferenceSolution.
   /// @see HYPREDRV_LinearSystemSetReferenceSolution
   void
   set_reference_solution(HYPRE_Vector v)
   {
      HYPREDRIVE_CXX_CHECK(HYPREDRV_LinearSystemSetReferenceSolution(handle_, v));
   }
   /// @brief C++ wrapper for HYPREDRV_LinearSystemResetInitialGuess.
   /// @see HYPREDRV_LinearSystemResetInitialGuess
   void
   reset_initial_guess()
   {
      HYPREDRIVE_CXX_CHECK(HYPREDRV_LinearSystemResetInitialGuess(handle_));
   }
   /// @brief C++ wrapper for HYPREDRV_LinearSystemSetPrecMatrix.
   /// @see HYPREDRV_LinearSystemSetPrecMatrix
   void
   set_prec_matrix(HYPRE_Matrix m)
   {
      HYPREDRIVE_CXX_CHECK(HYPREDRV_LinearSystemSetPrecMatrix(handle_, m));
   }
   /// @brief C++ wrapper for HYPREDRV_LinearSystemSetDofmap.
   /// @see HYPREDRV_LinearSystemSetDofmap
   void
   set_dofmap(int size, const int *dofmap)
   {
      HYPREDRIVE_CXX_CHECK(HYPREDRV_LinearSystemSetDofmap(handle_, size, dofmap));
   }
   /// @brief C++ wrapper for HYPREDRV_LinearSystemSetInterleavedDofmap.
   /// @see HYPREDRV_LinearSystemSetInterleavedDofmap
   void
   set_interleaved_dofmap(int blocks, int types)
   {
      HYPREDRIVE_CXX_CHECK(
         HYPREDRV_LinearSystemSetInterleavedDofmap(handle_, blocks, types));
   }
   /// @brief C++ wrapper for HYPREDRV_LinearSystemSetContiguousDofmap.
   /// @see HYPREDRV_LinearSystemSetContiguousDofmap
   void
   set_contiguous_dofmap(int blocks, int types)
   {
      HYPREDRIVE_CXX_CHECK(
         HYPREDRV_LinearSystemSetContiguousDofmap(handle_, blocks, types));
   }
   /// @brief C++ wrapper for HYPREDRV_LinearSystemReadDofmap.
   /// @see HYPREDRV_LinearSystemReadDofmap
   void
   read_dofmap()
   {
      HYPREDRIVE_CXX_CHECK(HYPREDRV_LinearSystemReadDofmap(handle_));
   }
   /// @brief C++ wrapper for HYPREDRV_LinearSystemPrintDofmap.
   /// @see HYPREDRV_LinearSystemPrintDofmap
   void
   print_dofmap(const char *filename)
   {
      HYPREDRIVE_CXX_CHECK(HYPREDRV_LinearSystemPrintDofmap(handle_, filename));
   }
   /// @brief C++ wrapper for HYPREDRV_LinearSystemPrint.
   /// @see HYPREDRV_LinearSystemPrint
   void
   print_system()
   {
      HYPREDRIVE_CXX_CHECK(HYPREDRV_LinearSystemPrint(handle_));
   }
   /// @brief C++ wrapper for HYPREDRV_LinearSystemSetNearNullSpace.
   /// @see HYPREDRV_LinearSystemSetNearNullSpace
   void
   set_near_null_space(int n, int c, const complex *values)
   {
      HYPREDRIVE_CXX_CHECK(HYPREDRV_LinearSystemSetNearNullSpace(handle_, n, c, values));
   }
   /// @brief C++ wrapper for HYPREDRV_LinearSystemSetNullSpace.
   /// @see HYPREDRV_LinearSystemSetNullSpace
   void
   set_null_space(int n, int c, const complex *values)
   {
      HYPREDRIVE_CXX_CHECK(HYPREDRV_LinearSystemSetNullSpace(handle_, n, c, values));
   }
   /// @brief Return the HYPREDRV-owned raw solution buffer.
   /// @note The returned pointer is owned by this driver and remains valid until the
   /// next solve, linear-system rebuild, solution replacement, or driver destruction.
   /// @see HYPREDRV_LinearSystemGetSolutionValues
   complex *
   get_solution_values_raw()
   {
      complex *p = nullptr;
      HYPREDRIVE_CXX_CHECK(HYPREDRV_LinearSystemGetSolutionValues(handle_, &p));
      return p;
   }
   /// @brief C++ wrapper for HYPREDRV_LinearSystemGetSolutionLength.
   /// @see HYPREDRV_LinearSystemGetSolutionLength
   bigint
   get_solution_length()
   {
      bigint n = 0;
      HYPREDRIVE_CXX_CHECK(HYPREDRV_LinearSystemGetSolutionLength(handle_, &n));
      return n;
   }
   /// @brief Return a copied solution vector.
   /// @note Allocates and copies the full solution. Use get_solution_values_raw() and
   /// get_solution_length() to access the HYPREDRV-owned buffer in place.
   /// @see HYPREDRV_LinearSystemGetSolutionValues
   std::vector<complex>
   get_solution_values()
   {
      const auto n      = static_cast<std::size_t>(get_solution_length());
      const auto values = get_solution_values_raw();
      return std::vector<complex>(values, values + n);
   }
   /// @brief C++ wrapper for HYPREDRV_LinearSystemGetSolutionNorm.
   /// @see HYPREDRV_LinearSystemGetSolutionNorm
   double
   get_solution_norm(const char *norm_type)
   {
      double n = 0.0;
      HYPREDRIVE_CXX_CHECK(HYPREDRV_LinearSystemGetSolutionNorm(handle_, norm_type, &n));
      return n;
   }
   /// @brief Return the HYPREDRV-owned solution vector handle.
   /// @note The returned handle is owned by this driver and remains valid until the
   /// linear-system state is replaced or the driver is destroyed.
   /// @see HYPREDRV_LinearSystemGetSolution
   HYPRE_Vector
   get_solution()
   {
      HYPRE_Vector v = nullptr;
      HYPREDRIVE_CXX_CHECK(HYPREDRV_LinearSystemGetSolution(handle_, &v));
      return v;
   }
   /// @brief Return the HYPREDRV-owned raw right-hand-side buffer.
   /// @note The returned pointer is owned by this driver and remains valid until the
   /// linear-system state is replaced or the driver is destroyed.
   /// @see HYPREDRV_LinearSystemGetRHSValues
   complex *
   get_rhs_values()
   {
      complex *p = nullptr;
      HYPREDRIVE_CXX_CHECK(HYPREDRV_LinearSystemGetRHSValues(handle_, &p));
      return p;
   }
   /// @brief Return the HYPREDRV-owned right-hand-side vector handle.
   /// @note The returned handle is owned by this driver and remains valid until the
   /// linear-system state is replaced or the driver is destroyed.
   /// @see HYPREDRV_LinearSystemGetRHS
   HYPRE_Vector
   get_rhs()
   {
      HYPRE_Vector v = nullptr;
      HYPREDRIVE_CXX_CHECK(HYPREDRV_LinearSystemGetRHS(handle_, &v));
      return v;
   }
   /// @brief Return the HYPREDRV-owned matrix handle.
   /// @note The returned handle is owned by this driver and remains valid until the
   /// linear-system state is replaced or the driver is destroyed.
   /// @see HYPREDRV_LinearSystemGetMatrix
   HYPRE_Matrix
   get_matrix()
   {
      HYPRE_Matrix m = nullptr;
      HYPREDRIVE_CXX_CHECK(HYPREDRV_LinearSystemGetMatrix(handle_, &m));
      return m;
   }
   /// @brief C++ wrapper for HYPREDRV_StateVectorSet.
   /// @see HYPREDRV_StateVectorSet
   void
   set_state_vector(int n, HYPRE_IJVector *vecs)
   {
      HYPREDRIVE_CXX_CHECK(HYPREDRV_StateVectorSet(handle_, n, vecs));
   }
   /// @brief C++ wrapper for HYPREDRV_StateVectorGetValues.
   /// @see HYPREDRV_StateVectorGetValues
   complex *
   get_state_vector_values(int index)
   {
      complex *p = nullptr;
      HYPREDRIVE_CXX_CHECK(HYPREDRV_StateVectorGetValues(handle_, index, &p));
      return p;
   }
   /// @brief C++ wrapper for HYPREDRV_StateVectorCopy.
   /// @see HYPREDRV_StateVectorCopy
   void
   copy_state_vector(int in, int out)
   {
      HYPREDRIVE_CXX_CHECK(HYPREDRV_StateVectorCopy(handle_, in, out));
   }
   /// @brief C++ wrapper for HYPREDRV_StateVectorUpdateAll.
   /// @see HYPREDRV_StateVectorUpdateAll
   void
   update_all_state_vectors()
   {
      HYPREDRIVE_CXX_CHECK(HYPREDRV_StateVectorUpdateAll(handle_));
   }
   /// @brief C++ wrapper for HYPREDRV_StateVectorApplyCorrection.
   /// @see HYPREDRV_StateVectorApplyCorrection
   void
   apply_state_vector_correction(int i)
   {
      HYPREDRIVE_CXX_CHECK(HYPREDRV_StateVectorApplyCorrection(handle_, i));
   }
   /// @brief C++ wrapper for HYPREDRV_PreconCreate.
   /// @see HYPREDRV_PreconCreate
   void
   create_precon()
   {
      HYPREDRIVE_CXX_CHECK(HYPREDRV_PreconCreate(handle_));
   }
   /// @brief C++ wrapper for HYPREDRV_LinearSolverCreate.
   /// @see HYPREDRV_LinearSolverCreate
   void
   create_linear_solver()
   {
      HYPREDRIVE_CXX_CHECK(HYPREDRV_LinearSolverCreate(handle_));
   }
   /// @brief C++ wrapper for HYPREDRV_PreconSetup.
   /// @see HYPREDRV_PreconSetup
   void
   setup_precon()
   {
      HYPREDRIVE_CXX_CHECK(HYPREDRV_PreconSetup(handle_));
   }
   /// @brief C++ wrapper for HYPREDRV_LinearSolverSetup.
   /// @see HYPREDRV_LinearSolverSetup
   void
   setup_linear_solver()
   {
      HYPREDRIVE_CXX_CHECK(HYPREDRV_LinearSolverSetup(handle_));
   }
   /// @brief C++ wrapper for HYPREDRV_LinearSolverApply.
   /// @see HYPREDRV_LinearSolverApply
   void
   apply_linear_solver()
   {
      HYPREDRIVE_CXX_CHECK(HYPREDRV_LinearSolverApply(handle_));
   }
   /// @brief Convenience wrapper for create/setup/apply solve sequence.
   /// @see HYPREDRV_LinearSolverCreate
   /// @see HYPREDRV_LinearSolverSetup
   /// @see HYPREDRV_LinearSolverApply
   void
   solve()
   {
      create_linear_solver();
      setup_linear_solver();
      apply_linear_solver();
   }
   /// @brief C++ wrapper for HYPREDRV_PreconApply.
   /// @see HYPREDRV_PreconApply
   void
   apply_precon(HYPRE_Vector b, HYPRE_Vector x)
   {
      HYPREDRIVE_CXX_CHECK(HYPREDRV_PreconApply(handle_, b, x));
   }
   /// @brief C++ wrapper for HYPREDRV_PreconDestroy.
   /// @see HYPREDRV_PreconDestroy
   void
   destroy_precon()
   {
      HYPREDRIVE_CXX_CHECK(HYPREDRV_PreconDestroy(handle_));
   }
   /// @brief C++ wrapper for HYPREDRV_LinearSolverDestroy.
   /// @see HYPREDRV_LinearSolverDestroy
   void
   destroy_linear_solver()
   {
      HYPREDRIVE_CXX_CHECK(HYPREDRV_LinearSolverDestroy(handle_));
   }
   /// @brief C++ wrapper for HYPREDRV_StatsPrint.
   /// @see HYPREDRV_StatsPrint
   void
   print_stats()
   {
      HYPREDRIVE_CXX_CHECK(HYPREDRV_StatsPrint(handle_));
   }
   /// @brief C++ wrapper for HYPREDRV_AnnotateBegin.
   /// @see HYPREDRV_AnnotateBegin
   void
   begin_annotation(const char *name, int id = -1)
   {
      HYPREDRIVE_CXX_CHECK(HYPREDRV_AnnotateBegin(handle_, name, id));
   }
   /// @brief C++ wrapper for HYPREDRV_AnnotateEnd.
   /// @see HYPREDRV_AnnotateEnd
   void
   end_annotation(const char *name, int id = -1)
   {
      HYPREDRIVE_CXX_CHECK(HYPREDRV_AnnotateEnd(handle_, name, id));
   }
   /// @brief C++ wrapper for HYPREDRV_AnnotateLevelBegin.
   /// @see HYPREDRV_AnnotateLevelBegin
   void
   begin_level_annotation(int level, const char *name, int id = -1)
   {
      HYPREDRIVE_CXX_CHECK(HYPREDRV_AnnotateLevelBegin(handle_, level, name, id));
   }
   /// @brief C++ wrapper for HYPREDRV_AnnotateLevelEnd.
   /// @see HYPREDRV_AnnotateLevelEnd
   void
   end_level_annotation(int level, const char *name, int id = -1)
   {
      HYPREDRIVE_CXX_CHECK(HYPREDRV_AnnotateLevelEnd(handle_, level, name, id));
   }
   /// @brief C++ wrapper for HYPREDRV_LinearSystemComputeEigenspectrum.
   /// @see HYPREDRV_LinearSystemComputeEigenspectrum
   void
   compute_eigenspectrum()
   {
      HYPREDRIVE_CXX_CHECK(HYPREDRV_LinearSystemComputeEigenspectrum(handle_));
   }
   /// @brief C++ wrapper for HYPREDRV_LinearSolverGetNumIter.
   /// @see HYPREDRV_LinearSolverGetNumIter
   int
   get_linear_solver_num_iter()
   {
      int v = 0;
      HYPREDRIVE_CXX_CHECK(HYPREDRV_LinearSolverGetNumIter(handle_, &v));
      return v;
   }
   /// @brief C++ wrapper for HYPREDRV_LinearSolverGetConverged.
   /// @see HYPREDRV_LinearSolverGetConverged
   bool
   get_linear_solver_converged()
   {
      int v = 0;
      HYPREDRIVE_CXX_CHECK(HYPREDRV_LinearSolverGetConverged(handle_, &v));
      return v != 0;
   }
   /// @brief C++ wrapper for HYPREDRV_LinearSolverGetFinalRelativeResidualNorm.
   /// @see HYPREDRV_LinearSolverGetFinalRelativeResidualNorm
   double
   get_linear_solver_final_relative_residual_norm()
   {
      double v = 0.0;
      HYPREDRIVE_CXX_CHECK(
         HYPREDRV_LinearSolverGetFinalRelativeResidualNorm(handle_, &v));
      return v;
   }
   /// @brief C++ wrapper for HYPREDRV_LinearSolverGetSetupTime.
   /// @see HYPREDRV_LinearSolverGetSetupTime
   double
   get_linear_solver_setup_time()
   {
      double v = 0.0;
      HYPREDRIVE_CXX_CHECK(HYPREDRV_LinearSolverGetSetupTime(handle_, &v));
      return v;
   }
   /// @brief C++ wrapper for HYPREDRV_LinearSolverGetSolveTime.
   /// @see HYPREDRV_LinearSolverGetSolveTime
   double
   get_linear_solver_solve_time()
   {
      double v = 0.0;
      HYPREDRIVE_CXX_CHECK(HYPREDRV_LinearSolverGetSolveTime(handle_, &v));
      return v;
   }
   /// @brief C++ wrapper for HYPREDRV_StatsLevelGetCount.
   /// @see HYPREDRV_StatsLevelGetCount
   int
   get_stats_level_count(int level)
   {
      int v = 0;
      HYPREDRIVE_CXX_CHECK(HYPREDRV_StatsLevelGetCount(handle_, level, &v));
      return v;
   }
   /// @brief C++ wrapper for HYPREDRV_StatsLevelGetEntry.
   /// @see HYPREDRV_StatsLevelGetEntry
   void
   get_stats_level_entry(int level, int index, int *entry_id, int *num_solves,
                         int *linear_iters, double *setup_time, double *solve_time)
   {
      HYPREDRIVE_CXX_CHECK(HYPREDRV_StatsLevelGetEntry(handle_, level, index, entry_id,
                                                       num_solves, linear_iters,
                                                       setup_time, solve_time));
   }
   /// @brief C++ wrapper for HYPREDRV_StatsLevelPrint.
   /// @see HYPREDRV_StatsLevelPrint
   void
   print_stats_level(int level)
   {
      HYPREDRIVE_CXX_CHECK(HYPREDRV_StatsLevelPrint(handle_, level));
   }

 private:
   HYPREDRV_t handle_ = nullptr;
};

} // namespace hypredrive

#undef HYPREDRIVE_CXX_CHECK

#endif // HYPREDRIVE_CXX_HPP
