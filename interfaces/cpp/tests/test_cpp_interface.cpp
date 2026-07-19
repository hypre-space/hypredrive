/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include <hypredrive.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace hd = hypredrive;

namespace
{

constexpr const char *test_options_yaml = R"yaml(
general:
  statistics: 0
solver:
  pcg:
    max_iter: 100
    relative_tol: 1.0e-8
    print_level: 0
preconditioner:
  amg:
    print_level: 0
)yaml";

void
partition_rows(int n, int size, int rank, hd::bigint &first, hd::bigint &last)
{
   const int base  = n / size;
   const int rem   = n % size;
   const int local = base + (rank < rem ? 1 : 0);
   const int start = rank * base + (rank < rem ? rank : rem);
   first           = start;
   last            = start + local - 1;
}

void
build_laplacian(hd::bigint n, hd::bigint first, hd::bigint last,
                hd::bigint_vector &indptr, hd::bigint_vector &cols, hd::real_vector &vals,
                hd::real_vector &rhs)
{
   const hd::bigint rows = last - first + 1;
   indptr.assign(static_cast<std::size_t>(rows + 1), 0);
   rhs.assign(static_cast<std::size_t>(rows), 1.0);
   cols.clear();
   vals.clear();
   for (hd::bigint i = 0; i < rows; ++i)
   {
      const hd::bigint row = first + i;
      if (row > 0)
      {
         cols.push_back(row - 1);
         vals.push_back(-1.0);
      }
      cols.push_back(row);
      vals.push_back(2.0);
      if (row + 1 < n)
      {
         cols.push_back(row + 1);
         vals.push_back(-1.0);
      }
      indptr[static_cast<std::size_t>(i + 1)] = static_cast<hd::bigint>(cols.size());
   }
}

void
solve_on_comm(MPI_Comm comm)
{
   int rank = 0, size = 1;
   MPI_Comm_rank(comm, &rank);
   MPI_Comm_size(comm, &size);
   const int  n     = 32;
   hd::bigint first = 0, last = 0;
   partition_rows(n, size, rank, first, last);

   hd::bigint_vector indptr, cols;
   hd::real_vector   vals, rhs;
   build_laplacian(n, first, last, indptr, cols, vals, rhs);

   hd::driver drv(comm);
   drv.set_library_mode();
   drv.parse_yaml(test_options_yaml);
   drv.set_matrix_from_csr(first, last, indptr.data(), cols.data(), vals.data());
   drv.set_rhs_from_array(first, last, rhs.data());
   drv.set_zero_initial_guess();
   drv.solve();
   const double norm = drv.get_solution_norm("l2");
   if (!(norm > 0.0))
   {
      throw std::runtime_error("solution norm must be positive");
   }
}

void
test_driver_move()
{
   hd::driver first(MPI_COMM_SELF);
   HYPREDRV_t raw = first.native_handle();
   hd::driver second(std::move(first));
   if (first.native_handle() != nullptr || second.native_handle() != raw)
   {
      throw std::runtime_error("hd::driver move construction did not transfer ownership");
   }

   hd::driver third(MPI_COMM_SELF);
   third = std::move(second);
   if (second.native_handle() != nullptr || third.native_handle() != raw)
   {
      throw std::runtime_error("hd::driver move assignment did not transfer ownership");
   }
}

void
test_invalid_yaml_throws()
{
   hd::driver drv(MPI_COMM_SELF);
   bool       threw = false;
   try
   {
      drv.parse_yaml("solver: [\n");
   }
   catch (const hd::error &)
   {
      threw = true;
   }
   if (!threw)
   {
      throw std::runtime_error("invalid YAML should throw hd::error");
   }
}

void
test_parse_yaml_overloads()
{
   hd::driver drv(MPI_COMM_SELF);
   drv.parse_yaml(std::string_view(test_options_yaml));
   drv.parse_yaml(std::string(test_options_yaml));
   std::istringstream stream(test_options_yaml);
   drv.parse_yaml(stream);
   int rank = 0;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   const std::string path = "hypredrive_cpp_options_" + std::to_string(rank) + ".yml";
   {
      std::ofstream file(path);
      file << test_options_yaml;
   }
   drv.parse_yaml(path);
   std::remove(path.c_str());

   // Overrides overload: resolves a path the C-side filename heuristic would
   // reject (uppercase extension) and applies CLI-style override tokens.
   const std::string upper_path =
      "hypredrive_cpp_options_" + std::to_string(rank) + ".YML";
   {
      std::ofstream file(upper_path);
      file << test_options_yaml;
   }
   drv.parse_yaml(upper_path, {"--solver:pcg:max_iter", "7"});
   std::remove(upper_path.c_str());

   // Inline text plus an override value ending in .yml: the value must be
   // applied as-is, not rediscovered as a configuration file.
   drv.parse_yaml(std::string_view(test_options_yaml),
                  {"-a", "--general:name", "run.yml"});

   bool threw = false;
   try
   {
      drv.parse_yaml(static_cast<const char *>(nullptr));
   }
   catch (const std::invalid_argument &)
   {
      threw = true;
   }
   if (!threw)
   {
      throw std::runtime_error("parse_yaml(nullptr) should throw std::invalid_argument");
   }
}

} // namespace

int
main(int argc, char **argv)
{
   MPI_Init(&argc, &argv);
   try
   {
      hd::initialize();
      test_driver_move();
      test_invalid_yaml_throws();
      test_parse_yaml_overloads();
      solve_on_comm(MPI_COMM_SELF);
      int world_size = 1;
      MPI_Comm_size(MPI_COMM_WORLD, &world_size);
      if (world_size > 1)
      {
         solve_on_comm(MPI_COMM_WORLD);
      }
      try
      {
         hd::driver drv(MPI_COMM_SELF);
         drv.apply_linear_solver();
         throw std::runtime_error("expected apply_linear_solver to throw before setup");
      }
      catch (const hd::error &)
      {
      }
      hd::finalize();
   }
   catch (const std::exception &e)
   {
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if (rank == 0)
      {
         std::cerr << e.what() << std::endl;
      }
      MPI_Abort(MPI_COMM_WORLD, 1);
   }
   MPI_Finalize();
   return EXIT_SUCCESS;
}
