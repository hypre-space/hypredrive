/* Bind against the same configured headers used by the native library. */
#include <HYPREDRV.h>

#if defined(HYPRE_COMPLEX) || defined(HYPRE_SINGLE) || defined(HYPRE_LONG_DOUBLE)
#error "The Rust interface requires real, double-precision HYPRE (disable HYPRE_COMPLEX, HYPRE_SINGLE, and HYPRE_LONG_DOUBLE)."
#endif

/* Also catch configurations that express scalar choices through typedefs. */
_Static_assert(sizeof(HYPRE_Real) == sizeof(double),
               "The Rust interface requires double-precision HYPRE_Real.");
_Static_assert(sizeof(HYPRE_Complex) == sizeof(double),
               "The Rust interface requires real, double-precision HYPRE_Complex.");

/* Describe the MPI header's ABI before replacing its MPI_Comm with rsmpi's. */
enum
{
   HYPREDRV_RUST_MPI_COMM_SIZE = sizeof(MPI_Comm),
   HYPREDRV_RUST_MPI_COMM_ALIGN = _Alignof(MPI_Comm),
   HYPREDRV_RUST_MPI_COMM_IS_INT = _Generic((MPI_Comm){0}, int: 1, unsigned int: 1, default: 0),
   HYPREDRV_RUST_MPI_VERSION = MPI_VERSION,
   HYPREDRV_RUST_MPI_SUBVERSION = MPI_SUBVERSION,
};
