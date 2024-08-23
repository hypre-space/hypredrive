/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/utsname.h>
#if defined(__APPLE__)
#include <sys/sysctl.h>
#include <mach/mach.h>
#include <mach-o/dyld.h>
#else
#define __USE_GNU
#include <sys/sysinfo.h>
#include <link.h>
#endif
#include "info.h"
#include "HYPRE_config.h"
#if defined(HYPRE_USING_OPENMP)
#include <omp.h>
#endif

#ifndef __APPLE__

/*--------------------------------------------------------------------------
 * dlpi_callback
 *
 * Linux: Use dl_iterate_phdr to list dynamic libraries
 *--------------------------------------------------------------------------*/

int
dlpi_callback(struct dl_phdr_info *info, size_t size, void *data)
{
   if (info->dlpi_name && info->dlpi_name[0])
   {
      const char* filename = strrchr(info->dlpi_name, '/');
      filename = filename ? filename + 1 : info->dlpi_name;
      printf("   %s => %s (0x%lx)\n", filename, info->dlpi_name, info->dlpi_addr);
   }
   return 0;
}

#endif

/*--------------------------------------------------------------------------
 * PrintSystemInfo
 *--------------------------------------------------------------------------*/

void
PrintSystemInfo(MPI_Comm comm)
{
   int    myid, nprocs;
   char   hostname[256];
   double bytes_to_GB = (double) (1 << 30);

   MPI_Comm_rank(comm, &myid);
   MPI_Comm_size(comm, &nprocs);

   /* Array to gather all hostnames */
   char allHostnames[nprocs * 256];

   /* Get the hostname for this process */
   gethostname(hostname, sizeof(hostname));

   /* Gather all hostnames to all processes */
   MPI_Allgather(hostname, 256, MPI_CHAR, allHostnames, 256, MPI_CHAR, MPI_COMM_WORLD);

   if (!myid)
   {
      printf("================================ System Information ================================\n\n");

      // 1. CPU cores and model
      int   numPhysicalCPUs = 0;
      int   physicalCPUSeen = 0;
      int   numCPU;
      char  cpuModels[8][256];
      char  gpuInfo[256] = "Unknown";

      /* Count unique hostnames */
      int numNodes = 0;
      for (int i = 0; i < nprocs; i++)
      {
         int isUnique = 1;
         for (int j = 0; j < i; j++)
         {
            if (strncmp(&allHostnames[i * 256], &allHostnames[j * 256], 256) == 0)
            {
               isUnique = 0;
               break;
            }
         }
         if (isUnique)
         {
            numNodes++;
         }
      }

#if defined(__APPLE__)
      size_t size = sizeof(numCPU);
      sysctlbyname("hw.ncpu", &numCPU, &size, NULL, 0);

      size_t size = sizeof(numPhysicalCPUs);
      sysctlbyname("hw.packages", &numPhysicalCPUs, &size, NULL, 0);

      for (int i = 0; i < numPhysicalCPUs; i++)
      {
         size = sizeof(cpuModels[i]);
         sysctlbyname("machdep.cpu.brand_string", &cpuModels[i], &size, NULL, 0);
      }
#else
      char buffer[256];
      FILE* fp = fopen("/proc/cpuinfo", "r");
      if (fp != NULL)
      {
         while (fgets(buffer, sizeof(buffer), fp))
         {
            if (strncmp(buffer, "physical id", 11) == 0)
            {
               int physicalID = atoi(strchr(buffer, ':') + 2);
               if (physicalID >= 0 && physicalID < 8)
               {
                  unsigned long long mask = 1ULL << physicalID;
                  if (!(physicalCPUSeen & mask))
                  {
                     physicalCPUSeen |= mask;
                     numPhysicalCPUs++;
                  }
               }
            }

            if (strncmp(buffer, "model name", 10) == 0)
            {
               int physicalID = numPhysicalCPUs - 1;
               if (physicalID >= 0 && physicalID < 8)
               {
                  char* model = strchr(buffer, ':') + 2;
                  strncpy(cpuModels[physicalID], model,
                          sizeof(cpuModels[physicalID]) - 1);
                  cpuModels[physicalID][strlen(cpuModels[physicalID]) - 1] = '\0';
               }
            }
         }
         fclose(fp);
      }

      numCPU = sysconf(_SC_NPROCESSORS_ONLN);
#endif
      if (strlen(gpuInfo) == 0)
      {
         strncpy(gpuInfo, "Unknown", sizeof(buffer));
      }

      printf("Processing Units\n");
      printf("-----------------\n");
      printf("Number of Nodes       : %d\n", numNodes);
      printf("Number of Processors  : %d\n", numPhysicalCPUs);
      printf("Number of CPU Cores   : %d\n", numCPU);
      for (int i = 0; i < numPhysicalCPUs; i++)
      {
         printf("CPU Model #%d          : %s\n", i, cpuModels[i]);
      }

#ifndef __APPLE__
      int gcount = 0;
      fp = popen("lspci | grep -Ei 'vga|3d|2d|display|accel'", "r");
      if (fp != NULL)
      {
         while (fgets(buffer, sizeof(buffer), fp) != NULL)
         {
            /* Skip onboard server graphics */
            if (strstr(buffer, "Matrox")  != NULL) { continue; }
            if (strstr(buffer, "ASPEED")  != NULL) { continue; }
            if (strstr(buffer, "Nuvoton") != NULL) { continue; }

            char *start = strstr(buffer, "VGA compatible controller");
            if (!start) start = strstr(buffer, "3D controller");
            if (!start) start = strstr(buffer, "2D controller");
            if (!start) start = strstr(buffer, "Display controller");
            if (!start) start = strstr(buffer, "Processing accelerators");

            if (start)
            {
               /* Adjust the strncpy depending on which controller type was found */
               const char *controller_type = "VGA compatible controller: ";
               if (strstr(buffer, "3D controller") != NULL)
               {
                  controller_type = "3D controller: ";
               }
               else if (strstr(buffer, "2D controller") != NULL)
               {
                  controller_type = "2D controller: ";
               }
               else if (strstr(buffer, "Display controller") != NULL)
               {
                  controller_type = "Display controller: ";
               }
               else if (strstr(buffer, "Processing accelerators") != NULL)
               {
                  controller_type = "Processing accelerators: ";
               }

               strncpy(gpuInfo, start + strlen(controller_type), sizeof(gpuInfo) - 1);
               gpuInfo[sizeof(gpuInfo) - 1] = '\0';

               /* Remove newline if present */
               size_t len = strlen(gpuInfo);
               if (len > 0 && gpuInfo[len - 1] == '\n')
               {
                  gpuInfo[len - 1] = '\0';
               }

               printf("GPU Model #%d          : %s\n", gcount++, gpuInfo);
            }
            else
            {
               strncpy(gpuInfo, buffer, sizeof(gpuInfo) - 1);
               gpuInfo[sizeof(gpuInfo) - 1] = '\0';
            }
         }
         pclose(fp);
      }
#endif
      printf("\n");

      // 2. Memory available and used
      printf("Memory Information\n");
      printf("-------------------\n");
#if defined(__APPLE__)
      int64_t memSize;
      size_t memSizeLen = sizeof(memSize);
      sysctlbyname("hw.memsize", &memSize, &memSizeLen, NULL, 0);

      mach_msg_type_number_t count = HOST_VM_INFO_COUNT;
      vm_statistics_data_t vmstat;
      if (host_statistics(mach_host_self(), HOST_VM_INFO, (host_info_t)&vmstat, &count) == KERN_SUCCESS)
      {
         int64_t freeMemory = (int64_t)vmstat.free_count * sysconf(_SC_PAGESIZE);
         int64_t usedMemory = memSize - freeMemory;

         printf("Total Memory          : %.3f GB\n", (double) memSize / bytes_to_GB);
         printf("Used Memory           : %.3f GB\n", (double) usedMemory / bytes_to_GB);
         printf("Free Memory           : %.3f GB\n\n", (double) freeMemory / bytes_to_GB);
      }
#else
      struct sysinfo info;
      if (sysinfo(&info) == 0)
      {
         printf("Total Memory          : %.3f GB\n", info.totalram * info.mem_unit / bytes_to_GB);
         printf("Used Memory           : %.3f GB\n", (info.totalram - info.freeram) * info.mem_unit / bytes_to_GB);
         printf("Free Memory           : %.3f GB\n\n", info.freeram * info.mem_unit / bytes_to_GB);
      }
#endif

      // 3. OS system info, release, version, machine
      printf("Operating System\n");
      printf("-----------------\n");
      struct utsname sysinfo;
      if (uname(&sysinfo) == 0)
      {
         printf("System Name           : %s\n", sysinfo.sysname);
         printf("Node Name             : %s\n", sysinfo.nodename);
         printf("Release               : %s\n", sysinfo.release);
         printf("Version               : %s\n", sysinfo.version);
         printf("Machine Architecture  : %s\n\n", sysinfo.machine);
      }

      // 4. Compilation Flags Information
      printf("Compilation Information\n");
      printf("------------------------\n");
      printf("Date                  : %s at %s\n", __DATE__, __TIME__);

#if defined(__OPTIMIZE__)
      printf("Optimization          : Enabled\n");
#else
      printf("Optimization          : Disabled\n");
#endif
#if defined(DEBUG)
      printf("Debugging             : Enabled\n");
#else
      printf("Debugging             : Disabled\n");
#endif
#if defined(__clang__)
      printf("Compiler              : Clang %d.%d.%d\n", __clang_major__, __clang_minor__, __clang_patchlevel__);
#elif defined(__INTEL_COMPILER)
      printf("Compiler              : Intel %d.%d\n", __INTEL_COMPILER / 100, (__INTEL_COMPILER % 100) / 10);
#elif defined(__GNUC__)
      printf("Compiler              : GCC %d.%d.%d\n", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#else
      printf("Compiler              : Unknown\n");
#endif
#if defined(HYPRE_USING_OPENMP) && defined(_OPENMP)
      printf("OpenMP                : Supported (Version: %d)\n", _OPENMP);
#else
      printf("OpenMP                : Not used\n");
#endif
      printf("MPI Implementation    : ");
#if defined(CRAY_MPICH_VERSION)
      printf("Cray MPI (Version: %s)\n", CRAY_MPICH_VERSION);
#elif defined(INTEL_MPI_VERSION)
      printf("Intel MPI (Version: %s)\n", INTEL_MPI_VERSION);
#elif defined(MPICH_NAME)
      printf("MPICH (Version: %s)\n", MPICH_VERSION);
#elif defined(OMPI_MAJOR_VERSION)
      printf("OpenMPI (Version: %d.%d.%d)\n", OMPI_MAJOR_VERSION, OMPI_MINOR_VERSION, OMPI_RELEASE_VERSION);
#elif defined(MVAPICH2_VERSION)
      printf("MVAPICH2 (Version: %s)\n", MVAPICH2_VERSION);
#elif defined(SGI_MPI)
      printf("SGI MPI\n");
#else
      printf("N/A\n");
#endif
#if defined(__x86_64__)
      printf("Target architecture   : x86_64\n");
#elif defined(__i386__)
      printf("Target architecture   : x86 (32-bit)\n");
#elif defined(__aarch64__)
      printf("Target architecture   : ARM64\n");
#elif defined(__arm__)
      printf("Target architecture   : ARM\n");
#else
      printf("Target architecture   : Unknown\n");
#endif
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
      printf("Endianness            : Little-endian\n");
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
      printf("Endianness            : Big-endian\n");
#else
      printf("Endianness            : Unknown\n");
#endif
      printf("\n");

      // 5. Current working directory
      printf("Current Working Directory\n");
      printf("--------------------------\n");
      char cwd[4096];
      if (getcwd(cwd, sizeof(cwd)) != NULL)
      {
         printf("%s\n\n", cwd);
      }

      // 6. Dynamic libraries used
      printf("Dynamic Libraries Loaded\n");
      printf("------------------------\n");
#if defined(__APPLE__)
      uint32_t dcount = _dyld_image_count();
      for (uint32_t i = 0; i < dcount; i++)
      {
         const char* name = _dyld_get_image_name(i);
         const struct mach_header* header = _dyld_get_image_header(i);
         const char* filename = strrchr(name, '/');

         filename = filename ? filename + 1 : name;
         printf("   %s => %s (0x%lx)\n", filename, name, (unsigned long)header);
      }
#else
      dl_iterate_phdr(dlpi_callback, NULL);
#endif

      printf("\n================================ System Information ================================\n\n");
      printf("Running on %d MPI rank%s\n", nprocs, nprocs > 1 ? "s" : "");

     /* Number of OpenMP threads per rank used in hypre */
#if defined(HYPRE_USING_OPENMP) && defined(_OPENMP)
      int num_threads = omp_get_max_threads();
      printf("Running on %d OpenMP thread%s per MPI rank\n", num_threads, num_threads > 1 ? "s" : "");
#else
      printf("HYPRE not using OpenMP\n");
#endif
   }
}

/*--------------------------------------------------------------------------
 * PrintLibInfo
 *--------------------------------------------------------------------------*/

void
PrintLibInfo(MPI_Comm comm)
{
   int         myid;
   time_t      t;
   struct tm  *tm_info;
   char        buffer[100];

   MPI_Comm_rank(comm, &myid);

   if (!myid)
   {
      /* Get current time */
      time(&t);
      tm_info = localtime(&t);

      /* Format and print the date and time */
      strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", tm_info);
      printf("Date and time: %s\n", buffer);

      /* Print hypre info */
#if defined(HYPRE_DEVELOP_STRING) && defined(HYPRE_BRANCH_NAME)
      printf("\nUsing HYPRE_DEVELOP_STRING: %s (%s)\n\n",
              HYPRE_DEVELOP_STRING, HYPRE_BRANCH_NAME);

#elif defined(HYPRE_DEVELOP_STRING) && !defined(HYPRE_BRANCH_NAME)
      printf("\nUsing HYPRE_DEVELOP_STRING: %s\n\n",
              HYPRE_DEVELOP_STRING);

#elif defined(HYPRE_RELEASE_VERSION)
      printf("\nUsing HYPRE_RELEASE_VERSION: %s\n\n",
              HYPRE_RELEASE_VERSION);
#endif
   }
}

/*--------------------------------------------------------------------------
 * PrintExitInfo
 *--------------------------------------------------------------------------*/

void
PrintExitInfo(MPI_Comm comm, const char *argv0)
{
   int    myid;
   time_t t;
   struct tm *tm_info;
   char buffer[100];

   MPI_Comm_rank(comm, &myid);

   if (!myid)
   {
      /* Get current time */
      time(&t);
      tm_info = localtime(&t);

      /* Format and print the date and time */
      strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", tm_info);
      printf("Date and time: %s\n%s done!\n", buffer, argv0);
   }
}
