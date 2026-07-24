/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include <hypredrive.hpp>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace hd = hypredrive;

namespace
{

struct params
{
   std::array<int, 3> n{16, 16, 16};
   std::array<int, 3> p{1, 1, 1};
   int                stencil = 7;
   int                verbose = 1;
   std::string        input_file{};
   int                hypredrv_argc = 0;       /* override args (incl. -a) */
   char             **hypredrv_argv = nullptr; /* override args, starting at -a */
};

void
usage(const char *argv0)
{
   std::cerr << "Usage: mpiexec -n <np> " << argv0
             << " [-n NX NY NZ] [-P PX PY PZ] [-s 7] [-v LEVEL] [-i YAML]"
                " [-a --path:to:key value ...]\n";
}

params
parse_args(int argc, char **argv)
{
   params out;
   for (int i = 1; i < argc; ++i)
   {
      const std::string arg = argv[i];
      if (arg == "-h" || arg == "--help")
      {
         usage(argv[0]);
         std::exit(EXIT_SUCCESS);
      }
      else if (arg == "-n" && i + 3 < argc)
      {
         out.n = {std::atoi(argv[++i]), std::atoi(argv[++i]), std::atoi(argv[++i])};
      }
      else if (arg == "-P" && i + 3 < argc)
      {
         out.p = {std::atoi(argv[++i]), std::atoi(argv[++i]), std::atoi(argv[++i])};
      }
      else if (arg == "-s" && i + 1 < argc)
      {
         out.stencil = std::atoi(argv[++i]);
      }
      else if (arg == "-v" && i + 1 < argc)
      {
         out.verbose = std::atoi(argv[++i]);
      }
      else if ((arg == "-i" || arg == "--input") && i + 1 < argc)
      {
         out.input_file = argv[++i];
      }
      else if (arg == "-a" || arg == "--args")
      {
         out.hypredrv_argc = argc - i;
         out.hypredrv_argv = argv + i;
         break;
      }
      else
      {
         usage(argv[0]);
         throw std::runtime_error("invalid command-line arguments");
      }
   }
   if (out.stencil != 7)
   {
      throw std::runtime_error("the C++ Laplacian example currently supports only -s 7");
   }
   if (out.n[0] <= 0 || out.n[1] <= 0 || out.n[2] <= 0 || out.p[0] <= 0 ||
       out.p[1] <= 0 || out.p[2] <= 0)
   {
      throw std::runtime_error("grid and processor dimensions must be positive");
   }
   return out;
}

hd::bigint_vector
split_points(int n, int p)
{
   hd::bigint_vector points(static_cast<std::size_t>(p + 1));
   const int         base = n / p;
   const int         rem  = n % p;
   for (int i = 0; i <= p; ++i)
   {
      points[static_cast<std::size_t>(i)] = base * i + std::min(i, rem);
   }
   return points;
}

int
owner_coord(hd::bigint x, const hd::bigint_vector &starts)
{
   return static_cast<int>(std::upper_bound(starts.begin(), starts.end(), x) -
                           starts.begin()) -
          1;
}

hd::bigint
block_size(const std::array<hd::bigint_vector, 3> &starts,
           const std::array<int, 3>               &coords)
{
   hd::bigint total = 1;
   for (int d = 0; d < 3; ++d)
   {
      total *= starts[d][static_cast<std::size_t>(coords[d] + 1)] -
               starts[d][static_cast<std::size_t>(coords[d])];
   }
   return total;
}

hd::bigint
local_id(const std::array<hd::bigint_vector, 3> &starts, const std::array<int, 3> &block,
         hd::bigint x, hd::bigint y, hd::bigint z)
{
   const auto x0 = starts[0][static_cast<std::size_t>(block[0])];
   const auto y0 = starts[1][static_cast<std::size_t>(block[1])];
   const auto z0 = starts[2][static_cast<std::size_t>(block[2])];
   const auto nx = starts[0][static_cast<std::size_t>(block[0] + 1)] - x0;
   const auto ny = starts[1][static_cast<std::size_t>(block[1] + 1)] - y0;
   return ((z - z0) * ny + (y - y0)) * nx + (x - x0);
}

struct grid
{
   MPI_Comm                         cart = MPI_COMM_NULL;
   std::array<int, 3>               n{};
   std::array<int, 3>               p{};
   std::array<int, 3>               coords{};
   std::array<hd::bigint_vector, 3> starts{};
   hd::bigint_vector                rank_offsets{};

   ~grid()
   {
      if (cart != MPI_COMM_NULL) MPI_Comm_free(&cart);
   }

   hd::bigint
   block_offset(const std::array<int, 3> &block) const
   {
      int rank = 0;
      int c[3] = {block[0], block[1], block[2]};
      MPI_Cart_rank(cart, c, &rank);
      return rank_offsets[static_cast<std::size_t>(rank)];
   }

   hd::bigint
   global_id(hd::bigint x, hd::bigint y, hd::bigint z) const
   {
      const std::array<int, 3> block{owner_coord(x, starts[0]), owner_coord(y, starts[1]),
                                     owner_coord(z, starts[2])};
      return block_offset(block) + local_id(starts, block, x, y, z);
   }

   hd::bigint
   row_start() const
   {
      return block_offset(coords);
   }
   hd::bigint
   local_size() const
   {
      return block_size(starts, coords);
   }
   hd::bigint
   row_end() const
   {
      return row_start() + local_size() - 1;
   }
};

grid
make_grid(MPI_Comm comm, const params &par)
{
   int size = 1;
   MPI_Comm_size(comm, &size);
   if (par.p[0] * par.p[1] * par.p[2] != size)
   {
      throw std::runtime_error("-P product must match MPI size");
   }
   grid g;
   g.n            = par.n;
   g.p            = par.p;
   int dims[3]    = {par.p[0], par.p[1], par.p[2]};
   int periods[3] = {0, 0, 0};
   MPI_Cart_create(comm, 3, dims, periods, 0, &g.cart);
   int rank = 0;
   MPI_Comm_rank(g.cart, &rank);
   int coords[3] = {0, 0, 0};
   MPI_Cart_coords(g.cart, rank, 3, coords);
   g.coords = {coords[0], coords[1], coords[2]};
   for (int d = 0; d < 3; ++d) g.starts[d] = split_points(g.n[d], g.p[d]);
   g.rank_offsets.assign(static_cast<std::size_t>(size), 0);
   hd::bigint_vector rank_sizes(static_cast<std::size_t>(size), 0);
   for (int r = 0; r < size; ++r)
   {
      int rc[3] = {0, 0, 0};
      MPI_Cart_coords(g.cart, r, 3, rc);
      rank_sizes[static_cast<std::size_t>(r)] =
         block_size(g.starts, {rc[0], rc[1], rc[2]});
   }
   for (int r = 1; r < size; ++r)
   {
      g.rank_offsets[static_cast<std::size_t>(r)] =
         g.rank_offsets[static_cast<std::size_t>(r - 1)] +
         rank_sizes[static_cast<std::size_t>(r - 1)];
   }
   return g;
}

constexpr const char *default_options_yaml = R"yaml(
general:
  name: cpp-laplacian
  statistics: 1
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
build_local_csr(const grid &g, hd::bigint_vector &indptr, hd::bigint_vector &cols,
                hd::real_vector &vals, hd::real_vector &rhs)
{
   const auto xs = g.starts[0][static_cast<std::size_t>(g.coords[0])];
   const auto xe = g.starts[0][static_cast<std::size_t>(g.coords[0] + 1)];
   const auto ys = g.starts[1][static_cast<std::size_t>(g.coords[1])];
   const auto ye = g.starts[1][static_cast<std::size_t>(g.coords[1] + 1)];
   const auto zs = g.starts[2][static_cast<std::size_t>(g.coords[2])];
   const auto ze = g.starts[2][static_cast<std::size_t>(g.coords[2] + 1)];
   indptr.assign(static_cast<std::size_t>(g.local_size() + 1), 0);
   rhs.assign(static_cast<std::size_t>(g.local_size()), 1.0);
   cols.clear();
   vals.clear();
   hd::bigint row   = 0;
   const int  dx[6] = {-1, 1, 0, 0, 0, 0};
   const int  dy[6] = {0, 0, -1, 1, 0, 0};
   const int  dz[6] = {0, 0, 0, 0, -1, 1};
   for (auto z = zs; z < ze; ++z)
   {
      for (auto y = ys; y < ye; ++y)
      {
         for (auto x = xs; x < xe; ++x)
         {
            hd::real diag = 0.0;
            for (int q = 0; q < 6; ++q)
            {
               const auto xn = x + dx[q];
               const auto yn = y + dy[q];
               const auto zn = z + dz[q];
               if (xn >= 0 && xn < g.n[0] && yn >= 0 && yn < g.n[1] && zn >= 0 &&
                   zn < g.n[2])
               {
                  cols.push_back(g.global_id(xn, yn, zn));
                  vals.push_back(-1.0);
                  diag += 1.0;
               }
            }
            cols.push_back(g.global_id(x, y, z));
            vals.push_back(diag);
            indptr[static_cast<std::size_t>(++row)] =
               static_cast<hd::bigint>(cols.size());
         }
      }
   }
}

} // namespace

int
main(int argc, char **argv)
{
   int ierr = EXIT_SUCCESS;
   int rank = 0;
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   try
   {
      const params      par = parse_args(argc, argv);
      grid              g   = make_grid(MPI_COMM_WORLD, par);
      hd::bigint_vector indptr, cols;
      hd::real_vector   vals, rhs;
      build_local_csr(g, indptr, cols, vals, rhs);

      hd::initialize();
      hd::driver drv(g.cart);
      drv.set_library_mode();
      const auto solver_options =
         par.input_file.empty() ? std::string(default_options_yaml) : par.input_file;
      const std::vector<std::string> overrides(par.hypredrv_argv,
                                               par.hypredrv_argv + par.hypredrv_argc);
      drv.parse_yaml(solver_options, overrides);
      drv.set_matrix_from_csr(g.row_start(), g.row_end(), indptr.data(), cols.data(),
                              vals.data());
      drv.set_rhs_from_array(g.row_start(), g.row_end(), rhs.data());
      drv.set_zero_initial_guess();
      drv.solve();
      const auto global_unknowns =
         static_cast<hd::bigint>(par.n[0]) * par.n[1] * par.n[2];
      if (rank == 0 && par.verbose > 0)
      {
         std::cout << "ranks          : " << par.p[0] * par.p[1] * par.p[2] << "\n"
                   << "unknowns       : " << global_unknowns << "\n"
                   << "solution norm  : " << drv.get_solution_norm("l2") << "\n"
                   << "iterations     : " << drv.get_linear_solver_num_iter() << "\n";
      }
      if (rank == 0) drv.print_stats();
      hd::finalize();
   }
   catch (const std::exception &e)
   {
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if (rank == 0) std::cerr << e.what() << "\n";
      // Best-effort cleanup; ignore finalization errors while reporting the original
      // failure.
      try
      {
         hd::finalize();
      }
      catch (const std::exception &)
      {
      }
      ierr = EXIT_FAILURE;
   }
   MPI_Finalize();
   return ierr;
}
