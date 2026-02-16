/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef __APPLE__
#define _GNU_SOURCE 1
#endif
#include "info.h"
#include <ctype.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <sys/utsname.h>
#include <unistd.h>
#include "HYPREDRV_config.h"
#include "HYPRE_config.h"
#ifdef HYPRE_USING_OPENMP
#include <omp.h>
#endif

#ifdef HAVE_HWLOC
#include <hwloc.h>
#endif

#ifdef __APPLE__
#include <mach-o/dyld.h>
#include <mach/mach.h>
#include <sys/sysctl.h>
#else
#include <dirent.h>
#include <limits.h>
#include <link.h>
#include <sys/sysinfo.h>
#endif

#ifndef STRINGIFY
#define STRINGIFY(x) #x
#endif
#ifndef TOSTRING
#define TOSTRING(x) STRINGIFY(x)
#endif

#define HYPRE_MAX_HOSTNAME 256
#define HYPRE_MAX_GPU_BINDING 512

#ifndef __APPLE__
static int  ReadLineFromFile(const char *path, char *buffer, size_t len);
static int  ReadIntFromFile(const char *path, int *value);
static int  ReadUllFromProcMeminfo(const char *field, unsigned long long *value);
static int  ExtractBracketedToken(const char *line, char *token, size_t len);
static void PrintLinuxNumaInformation(double bytes_to_gib);
static void PrintNetworkInformation(void);
static void PrintAcceleratorRuntimeInformation(void);
static void PrintLinuxKernelTuningInformation(void);
#endif
static void BuildGpuBindingString(char *buffer, size_t len);
static void PrintMpiRuntimeInformation(MPI_Comm comm);
static void PrintThreadingEnvironmentInformation(void);
static void TrimTrailingWhitespace(char *s);
static void NormalizeWhitespace(char *s);

#ifdef HAVE_HWLOC
typedef struct
{
   char        name[64];
   char        vendor[32];
   char        model[128];
   char        uuid[64];
   char        pci_busid[20];
   int         smi_id;
   hwloc_obj_t obj;      // PCI device object
   hwloc_obj_t ancestor; // Non-I/O ancestor (has nodeset)
} GpuInfo;

static hwloc_topology_t topology = NULL;
static int              InitHwlocTopology(void);
static void             CleanupHwlocTopology(void);
static void             PrintSystemInfoHwloc(MPI_Comm comm);
static void             PrintCpuTopologyInfo(MPI_Comm comm);
static void             PrintCacheHierarchy(void);
static void             PrintGpuInfo(GpuInfo *gpus, int gpu_count);
static int              DiscoverGpus(GpuInfo **gpus, int *count);
static void             PrintNumaInfo(double bytes_to_gib, GpuInfo *gpus, int gpu_count);
static void             PrintNetworkInfoHwloc(void);
static void             PrintProcessBinding(void);
static void             PrintThreadAffinity(MPI_Comm comm, GpuInfo *gpus, int gpu_count);
static void             PrintGpuAffinity(MPI_Comm comm, GpuInfo *gpus, int gpu_count);
static void             PrintTopologyTree(void);
static void             PrintTopologyTreeRecursive(hwloc_obj_t obj, int depth);
static void             PrintMemoryInformation(double bytes_to_gib, double mib_to_gib);
static void             PrintOperatingSystemInfo(void);
static void             PrintCompilationInfo(void);
static void             PrintWorkingDirectory(void);
static void             PrintDynamicLibraries(void);
static void             PrintRunningInfo(MPI_Comm comm);
#endif
void PrintSystemInfoLegacy(MPI_Comm comm);

#ifndef __APPLE__

/*--------------------------------------------------------------------------
 * dlpi_callback
 *
 * Linux: Use dl_iterate_phdr to list dynamic libraries
 *--------------------------------------------------------------------------*/

int
dlpi_callback(struct dl_phdr_info *info, size_t size, void *data)
{
   (void)size;
   (void)data;
   if (info->dlpi_name && info->dlpi_name[0])
   {
      const char *filename = strrchr(info->dlpi_name, '/');
      filename             = filename ? filename + 1 : info->dlpi_name;
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
   int use_hwloc = 0;

#ifdef HAVE_HWLOC
   if (InitHwlocTopology() == 0)
   {
      use_hwloc = 1;
   }
#endif

   if (use_hwloc)
   {
#ifdef HAVE_HWLOC
      PrintSystemInfoHwloc(comm);
#endif
   }
   else
   {
      PrintSystemInfoLegacy(comm);
   }

#ifdef HAVE_HWLOC
   if (use_hwloc)
   {
      CleanupHwlocTopology();
   }
#endif
}

/*--------------------------------------------------------------------------
 * PrintSystemInfoLegacy
 *--------------------------------------------------------------------------*/

void
PrintSystemInfoLegacy(MPI_Comm comm)
{
   int    myid = 0, nprocs = 0;
   char   hostname[HYPRE_MAX_HOSTNAME];
   double bytes_to_gib = (double)(1 << 30);
   double mib_to_gib   = (double)(1 << 10);
   int    gcount;
   size_t total = 0, used = 0;
   FILE  *fp = NULL;
   char   buffer[32768];

   MPI_Comm_rank(comm, &myid);
   MPI_Comm_size(comm, &nprocs);

   /* Get the hostname for this process */
   gethostname(hostname, sizeof(hostname));

   /* Gather hostnames on all ranks (needed for unique counts, avoids huge stack VLAs) */
   char *allHostnames = NULL;
   if (nprocs > 0)
   {
      allHostnames = (char *)malloc((size_t)nprocs * HYPRE_MAX_HOSTNAME);
   }
   int host_alloc_ok = allHostnames ? 1 : 0;
   MPI_Allreduce(MPI_IN_PLACE, &host_alloc_ok, 1, MPI_INT, MPI_MIN, comm);
   if (!host_alloc_ok)
   {
      if (allHostnames)
      {
         free(allHostnames);
      }
      return;
   }
   MPI_Allgather(hostname, HYPRE_MAX_HOSTNAME, MPI_CHAR, allHostnames, HYPRE_MAX_HOSTNAME,
                 MPI_CHAR, comm);

   /* Gather per-rank GPU binding strings on rank 0 */
   char gpuBindingLocal[HYPRE_MAX_GPU_BINDING];
   BuildGpuBindingString(gpuBindingLocal, sizeof(gpuBindingLocal));
   char *gpuBindingAll = NULL;
   if (!myid && nprocs > 0)
   {
      gpuBindingAll = (char *)malloc((size_t)nprocs * HYPRE_MAX_GPU_BINDING);
   }
   MPI_Gather(gpuBindingLocal, HYPRE_MAX_GPU_BINDING, MPI_CHAR, gpuBindingAll,
              HYPRE_MAX_GPU_BINDING, MPI_CHAR, 0, comm);

   if (!myid)
   {
      printf("================================== System Information "
             "=================================\n\n");

      // 1. CPU cores and model
      int  numPhysicalCPUs = 0;
      int  numCPUs         = 0;
      char cpuModels[8][256];
      char gpuInfo[256] = "Unknown";

      /* Count unique hostnames */
      int numNodes = 0;
      for (int i = 0; i < nprocs; i++)
      {
         int isUnique = 1;
         for (int j = 0; j < i; j++)
         {
            if (strncmp(&allHostnames[i * HYPRE_MAX_HOSTNAME],
                        &allHostnames[j * HYPRE_MAX_HOSTNAME], HYPRE_MAX_HOSTNAME) == 0)
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

#ifdef __APPLE__
      size_t msize = sizeof(numCPUs);
      sysctlbyname("hw.ncpu", &numCPUs, &msize, NULL, 0);

      msize = sizeof(numPhysicalCPUs);
      sysctlbyname("hw.packages", &numPhysicalCPUs, &msize, NULL, 0);

      for (int i = 0; i < numPhysicalCPUs; i++)
      {
         msize = sizeof(cpuModels[i]);
         sysctlbyname("machdep.cpu.brand_string", &cpuModels[i], &msize, NULL, 0);
      }
#else
      int physicalCPUSeen = 0;
      fp                  = fopen("/proc/cpuinfo", "r");
      if (fp != NULL)
      {
         while (fgets(buffer, sizeof(buffer), fp))
         {
            if (strncmp(buffer, "physical id", 11) == 0)
            {
               int physicalID = atoi(strchr(buffer, ':') + 2);
               if (physicalID >= 0 && physicalID < 8)
               {
                  unsigned long long mask = 1ULL << (unsigned)physicalID;
                  if (!((unsigned long long)physicalCPUSeen & mask))
                  {
                     physicalCPUSeen = (int)((unsigned long long)physicalCPUSeen | mask);
                     numPhysicalCPUs++;
                  }
               }
            }

            if (strncmp(buffer, "model name", 10) == 0)
            {
               int physicalID = numPhysicalCPUs - 1;
               if (physicalID >= 0 && physicalID < 8)
               {
                  const char *model = strchr(buffer, ':') + 2;
                  strncpy(cpuModels[physicalID], model,
                          sizeof(cpuModels[physicalID]) - 1);
                  cpuModels[physicalID][sizeof(cpuModels[physicalID]) - 1] = '\0';
               }
            }
         }
         fclose(fp);

         if (numPhysicalCPUs == 0)
         {
            fp = popen("lscpu | grep 'Socket(s)' | awk '{print $2}'", "r");
            if (fp != NULL)
            {
               if (fgets(buffer, sizeof(buffer), fp) != NULL)
               {
                  numPhysicalCPUs = atoi(buffer);
               }
               pclose(fp);
            }

            fp = popen("lscpu | grep 'Model name:' | sed 's/Model name:\\s*//'", "r");
            if (fp != NULL)
            {
               if (fgets(buffer, sizeof(buffer), fp) != NULL)
               {
                  buffer[strcspn(buffer, "\n")] = '\0';
                  for (int i = 0; i < numPhysicalCPUs; i++)
                  {
                     strncpy(cpuModels[i], buffer, sizeof(cpuModels[i]) - 1);
                     cpuModels[i][sizeof(cpuModels[i]) - 1] = '\0';
                  }
               }
               pclose(fp);
            }
         }
      }

      numCPUs = (int)sysconf(_SC_NPROCESSORS_ONLN);
#endif
      if (strlen(gpuInfo) == 0)
      {
         strncpy(gpuInfo, "Unknown", sizeof(gpuInfo) - 1);
         gpuInfo[sizeof(gpuInfo) - 1] = '\0';
      }

      printf("Processing Units\n");
      printf("-----------------\n");
      printf("Number of Nodes       : %d\n", numNodes);
      printf("Number of Processors  : %d\n", numPhysicalCPUs);
      printf("Number of CPU threads : %d\n", numCPUs);
      printf("Tot. # of Processors  : %lld\n",
             (long long)numNodes * (long long)numPhysicalCPUs);
      printf("Tot. # of CPU threads : %lld\n", (long long)numNodes * (long long)numCPUs);
      for (int i = 0; i < numPhysicalCPUs; i++)
      {
         printf("CPU Model #%d          : %s\n", i, cpuModels[i]);
      }

#ifndef __APPLE__
      gcount = 0;
      fp     = NULL;
      if (system("command -v lspci > /dev/null 2>&1") == 0)
      {
         fp = popen("lspci | grep -Ei 'vga|3d|2d|display|accel'", "r");
      }
      if (fp != NULL)
      {
         while (fgets(buffer, sizeof(buffer), fp) != NULL)
         {
            /* Skip onboard server graphics */
            if (strstr(buffer, "Matrox") != NULL)
            {
               continue;
            }
            if (strstr(buffer, "ASPEED") != NULL)
            {
               continue;
            }
            if (strstr(buffer, "Nuvoton") != NULL)
            {
               continue;
            }

            const char *start = strstr(buffer, "VGA compatible controller");
            if (!start)
            {
               start = strstr(buffer, "3D controller");
            }
            if (!start)
            {
               start = strstr(buffer, "2D controller");
            }
            if (!start)
            {
               start = strstr(buffer, "Display controller");
            }
            if (!start)
            {
               start = strstr(buffer, "Processing accelerators");
            }

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
      else if (system("command -v nvidia-smi > /dev/null 2>&1") == 0)
      {
         fp = popen("nvidia-smi --query-gpu=name --format=csv,noheader", "r");
         if (fp != NULL)
         {
            while (fgets(buffer, sizeof(buffer), fp) != NULL)
            {
               printf("GPU Model #%d          : %s", gcount++, buffer);
            }
            pclose(fp);
         }
      }
      gcount = 0;
#endif
      printf("\n");

      // 2. Memory available and used
      printf("Memory Information (Used/Total)\n");
      printf("--------------------------------\n");
#ifdef __APPLE__
      size_t memSizeLen = sizeof(total);
      sysctlbyname("hw.memsize", &total, &memSizeLen, NULL, 0);

      mach_msg_type_number_t count  = HOST_VM_INFO_COUNT;
      vm_statistics_data_t   vmstat = {0};
      if (host_statistics(mach_host_self(), HOST_VM_INFO, (host_info_t)&vmstat, &count) ==
          KERN_SUCCESS)
      {
         used = total - (size_t)vmstat.free_count * sysconf(_SC_PAGESIZE);

         printf("CPU RAM               : %6.2f / %6.2f  (%5.2f %%) GiB\n",
                (double)used / bytes_to_gib, (double)total / bytes_to_gib,
                100.0 * (total - used) / (double)total);
      }
#else
      struct sysinfo info;
      if (sysinfo(&info) == 0)
      {
         printf("CPU RAM               : %6.2f / %6.2f  (%5.2f %%) GiB\n",
                (double)(info.totalram - info.freeram) * info.mem_unit / bytes_to_gib,
                (double)info.totalram * info.mem_unit / bytes_to_gib,
                100.0 * (double)(info.totalram - info.freeram) / (double)info.totalram);
      }
#endif

      /* NVIDIA GPU Memory Information */
      fp = NULL;
      if (system("command -v nvidia-smi > /dev/null 2>&1") == 0)
      {
         fp = popen("nvidia-smi --query-gpu=memory.total,memory.used "
                    "--format=csv,noheader,nounits",
                    "r");
      }
      if (fp != NULL)
      {
         gcount = 0;
         while (fgets(buffer, sizeof(buffer), fp) != NULL)
         {
            if (sscanf(buffer, "%zu, %zu", &total, &used) == 2)
            {
               printf("GPU RAM #%d            : %6.2f / %6.2f  (%5.2f %%) GiB\n",
                      gcount++, (double)used / mib_to_gib, (double)total / mib_to_gib,
                      100.0 * (double)used / (double)total);
            }
         }
         pclose(fp);
      }

      /* AMD GPU Memory Information */
      fp = NULL;
      if (system("command -v amd-smi > /dev/null 2>&1") == 0)
      {
         fp = popen("amd-smi metric -m --json 2>/dev/null", "r");
      }
      if (fp != NULL)
      {
         char   json_buffer[32768];
         size_t read       = fread(json_buffer, 1, sizeof(json_buffer) - 1, fp);
         json_buffer[read] = '\0';
         pclose(fp);

         // Parse amd-smi JSON format for all GPUs
         const char *total_vram_str = "\"total_vram\"";
         const char *used_vram_str  = "\"used_vram\"";
         const char *ptr            = json_buffer;

         while ((ptr = strstr(ptr, "\"gpu\"")) != NULL)
         {
            // Find GPU index
            ptr = strchr(ptr, ':');
            if (ptr)
            {
               ptr++;
               while (*ptr == ' ') ptr++;
               {
                  char *endptr;
                  (void)strtol(ptr, &endptr, 10); /* idx unused, advance past it */
                  ptr = endptr;
               }

               // Find total_vram for this GPU
               const char *gpu_start = ptr;
               const char *total_ptr = strstr(gpu_start, total_vram_str);
               const char *used_ptr  = strstr(gpu_start, used_vram_str);

               if (total_ptr && used_ptr)
               {
                  // Extract total_vram value
                  const char *val_ptr = strstr(total_ptr, "\"value\"");
                  if (val_ptr)
                  {
                     val_ptr = strchr(val_ptr, ':');
                     if (val_ptr)
                     {
                        val_ptr++;
                        while (*val_ptr == ' ') val_ptr++;
                        total = strtoull(val_ptr, NULL, 10) * 1024 * 1024; // MB to bytes
                     }
                  }

                  // Extract used_vram value
                  val_ptr = strstr(used_ptr, "\"value\"");
                  if (val_ptr)
                  {
                     val_ptr = strchr(val_ptr, ':');
                     if (val_ptr)
                     {
                        val_ptr++;
                        while (*val_ptr == ' ') val_ptr++;
                        used = strtoull(val_ptr, NULL, 10) * 1024 * 1024; // MB to bytes
                     }
                  }

                  if (total > 0)
                  {
                     printf("GPU RAM #%d            : %6.2f / %6.2f  (%5.2f %%) GiB\n",
                            gcount++, (double)used / bytes_to_gib,
                            (double)total / bytes_to_gib,
                            100.0 * (double)used / (double)total);
                  }
               }
            }

            // Move to next GPU entry
            ptr = strstr(ptr, "}");
            if (ptr) ptr++;
            else break;
         }
      }

#ifndef __APPLE__
      PrintLinuxNumaInformation(bytes_to_gib);
      PrintNetworkInformation();
      PrintAcceleratorRuntimeInformation();
#endif

      if (gpuBindingAll)
      {
         printf("Accelerator Binding (per rank)\n");
         printf("-------------------------------\n");
         for (int r = 0; r < nprocs; r++)
         {
            printf("Rank %-3d              : %s\n", r,
                   gpuBindingAll + (size_t)r * HYPRE_MAX_GPU_BINDING);
         }
      }

      // 3. OS system info, release, version, machine
      printf("\nOperating System\n");
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

      /* Check optimization level */
#if defined(__OPTIMIZE__)
      printf("Optimization          : Enabled\n");
#elif defined(__OPTIMIZE_SIZE__)
      printf("Optimization          : Enabled (size)\n");
#elif defined(_MSC_VER)
      /* MSVC doesn't define __OPTIMIZE__, optimization detection is complex */
      printf("Optimization          : Unknown (MSVC)\n");
#else
      printf("Optimization          : Disabled\n");
#endif
      /* Check debug symbols:
       * - NDEBUG means "not debug" (when defined, assertions are disabled, typically
       * release)
       * - _DEBUG is MSVC's debug macro
       * - DEBUG might be user-defined
       * Note: This detects debug macros, not necessarily debug symbols (-g flag) */
#if defined(HYPRE_DEBUG)
      printf("Debugging             : Enabled (HYPRE)\n");
#elif defined(_DEBUG) || defined(DEBUG)
      printf("Debugging             : Enabled\n");
#elif defined(NDEBUG)
      printf("Debugging             : Disabled\n");
#else
      /* NDEBUG not defined - could be debug or release, default to unknown */
      printf("Debugging             : Unknown\n");
#endif
#ifdef __clang_version__
      printf("Compiler              : Clang %s\n", (const char *)__clang_version__);
#elif defined(__clang__)
      printf("Compiler              : Clang %d.%d.%d\n", __clang_major__, __clang_minor__,
             __clang_patchlevel__);
#elif defined(__INTEL_COMPILER)
      printf("Compiler              : Intel %d.%d\n", __INTEL_COMPILER / 100,
             (__INTEL_COMPILER % 100) / 10);
#elif defined(__GNUC__)
      printf("Compiler              : GCC %d.%d.%d\n", __GNUC__, __GNUC_MINOR__,
             __GNUC_PATCHLEVEL__);
#else
      printf("Compiler              : Unknown\n");
#endif
#if defined(HYPRE_USING_OPENMP) && defined(_OPENMP)
      printf("OpenMP                : Supported (Version: %d)\n", _OPENMP);
#else
      printf("OpenMP                : Not used\n");
#endif
      printf("MPI library           : ");
#ifdef CRAY_MPICH_VERSION
      printf("Cray MPI (Version: %s)\n", TOSTRING(CRAY_MPICH_VERSION));
#elif defined(INTEL_MPI_VERSION)
      printf("Intel MPI (Version: %s)\n", (const char *)INTEL_MPI_VERSION);
#elif defined(__IBM_MPI__)
      printf("IBM Spectrum MPI (Version: %d.%d.%d)\n", __IBM_MPI_MAJOR_VERSION,
             __IBM_MPI_MINOR_VERSION, __IBM_MPI_RELEASE_VERSION);
#elif defined(MVAPICH2_VERSION)
      printf("MVAPICH2 (Version: %s)\n", (const char *)MVAPICH2_VERSION);
#elif defined(MPICH_NAME)
      printf("MPICH (Version: %s)\n", MPICH_VERSION);
#elif defined(OMPI_MAJOR_VERSION)
      printf("OpenMPI (Version: %d.%d.%d)\n", OMPI_MAJOR_VERSION, OMPI_MINOR_VERSION,
             OMPI_RELEASE_VERSION);
#elif defined(SGI_MPI)
      printf("SGI MPI\n");
#else
      printf("N/A\n");
#endif
#ifdef __x86_64__
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
#ifdef __APPLE__
      uint32_t dcount = _dyld_image_count();
      for (uint32_t i = 0; i < dcount; i++)
      {
         const char               *name     = _dyld_get_image_name(i);
         const struct mach_header *header   = _dyld_get_image_header(i);
         const char               *filename = strrchr(name, '/');

         filename = filename ? filename + 1 : name;
         printf("   %s => %s (0x%lx)\n", filename, name, (unsigned long)header);
      }
#else
      dl_iterate_phdr(dlpi_callback, NULL);
#endif

      PrintMpiRuntimeInformation(comm);
      PrintThreadingEnvironmentInformation();
#ifndef __APPLE__
      PrintLinuxKernelTuningInformation();
#endif

      printf("\n================================== System Information "
             "=================================\n\n");
      printf("Running on %d MPI rank%s\n", nprocs, nprocs > 1 ? "s" : "");

      /* Number of OpenMP threads per rank used in hypre */
#if defined(HYPRE_USING_OPENMP) && defined(_OPENMP)
      int num_threads = omp_get_max_threads();
      printf("Running on %d OpenMP thread%s per MPI rank\n", num_threads,
             num_threads > 1 ? "s" : "");
#endif
   }

   if (allHostnames)
   {
      free(allHostnames);
   }
   if (gpuBindingAll)
   {
      free(gpuBindingAll);
   }
}

static void
BuildGpuBindingString(char *buffer, size_t len)
{
   struct BindingVar
   {
      const char *label;
      const char *env;
   };
   static const struct BindingVar vars[] = {
      {"CUDA", "CUDA_VISIBLE_DEVICES"}, {"HIP", "HIP_VISIBLE_DEVICES"},
      {"ROCR", "ROCR_VISIBLE_DEVICES"}, {"ONEAPI", "ONEAPI_DEVICE_SELECTOR"},
      {"ZE", "ZE_AFFINITY_MASK"},       {"SYCL", "SYCL_DEVICE_FILTER"}};

   if (!buffer || len == 0)
   {
      return;
   }

   buffer[0] = '\0';
   for (size_t i = 0; i < sizeof(vars) / sizeof(vars[0]); i++)
   {
      const char *value = getenv(vars[i].env);
      if (!value || !value[0])
      {
         continue;
      }

      size_t used = strlen(buffer);
      if (used >= len - 1)
      {
         break;
      }
      int written = snprintf(buffer + used, len - used, "%s%s=%s", used ? ", " : "",
                             vars[i].label, value);
      if (written < 0)
      {
         break;
      }
   }

   if (buffer[0] == '\0')
   {
      strncpy(buffer, "unset", len - 1);
      buffer[len - 1] = '\0';
   }
}

static void
PrintMpiRuntimeInformation(MPI_Comm comm)
{
   int myid = 0;
   MPI_Comm_rank(comm, &myid);
   if (myid)
   {
      return;
   }

   printf("MPI Runtime Information\n");
   printf("------------------------\n");

   int mpi_major = 0;
   int mpi_minor = 0;
   if (MPI_Get_version(&mpi_major, &mpi_minor) == MPI_SUCCESS)
   {
      printf("MPI Standard          : %d.%d\n", mpi_major, mpi_minor);
   }

   char lib_version[MPI_MAX_LIBRARY_VERSION_STRING + 1];
   int  lib_len = 0;
   if (MPI_Get_library_version(lib_version, &lib_len) == MPI_SUCCESS)
   {
      if (lib_len < 0)
      {
         lib_len = 0;
      }
      if ((size_t)lib_len >= sizeof(lib_version))
      {
         lib_len = (int)(sizeof(lib_version) - 1);
      }
      lib_version[lib_len]                      = '\0';
      lib_version[strcspn(lib_version, "\r\n")] = '\0';
      NormalizeWhitespace(lib_version);
      if (lib_version[0])
      {
         printf("MPI Implementation    : %s\n", lib_version);
      }
   }

   char processor[MPI_MAX_PROCESSOR_NAME + 1];
   int  processor_len = 0;
   if (MPI_Get_processor_name(processor, &processor_len) == MPI_SUCCESS)
   {
      if (processor_len < 0)
      {
         processor_len = 0;
      }
      if ((size_t)processor_len >= sizeof(processor))
      {
         processor_len = (int)(sizeof(processor) - 1);
      }
      processor[processor_len] = '\0';
      printf("Rank 0 Processor Name : %s\n", processor);
   }
   printf("\n");
}

static void
TrimTrailingWhitespace(char *s)
{
   if (!s)
   {
      return;
   }

   size_t len = strlen(s);
   while (len > 0 && isspace((unsigned char)s[len - 1]))
   {
      s[--len] = '\0';
   }
}

static void
NormalizeWhitespace(char *s)
{
   if (!s)
   {
      return;
   }

   char *src       = s;
   char *dst       = s;
   int   saw_space = 0;

   while (*src)
   {
      unsigned char c = (unsigned char)*src++;
      if (isspace(c))
      {
         saw_space = 1;
         continue;
      }

      if (saw_space && dst != s)
      {
         *dst++ = ' ';
      }
      *dst++    = (char)c;
      saw_space = 0;
   }
   *dst = '\0';
}

static void
PrintThreadingEnvironmentInformation(void)
{
   struct ThreadEnvVar
   {
      const char *label;
      const char *name;
   };

   static const struct ThreadEnvVar vars[] = {
      {"OMP_NUM_THREADS", "OMP_NUM_THREADS"}, {"OMP_PROC_BIND", "OMP_PROC_BIND"},
      {"OMP_PLACES", "OMP_PLACES"},           {"OMP_SCHEDULE", "OMP_SCHEDULE"},
      {"KMP_AFFINITY", "KMP_AFFINITY"},       {"GOMP_CPU_AFFINITY", "GOMP_CPU_AFFINITY"},
   };

   printf("Threading Environment\n");
   printf("----------------------\n");
#if defined(HYPRE_USING_OPENMP) && defined(_OPENMP)
   printf("OpenMP max threads    : %d\n", omp_get_max_threads());
#else
   printf("OpenMP max threads    : not available (OpenMP disabled)\n");
#endif

   int printed_env = 0;
   for (size_t i = 0; i < sizeof(vars) / sizeof(vars[0]); i++)
   {
      const char *value = getenv(vars[i].name);
      if (value && value[0])
      {
         printf("%-22s : %s\n", vars[i].label, value);
         printed_env = 1;
      }
   }

   if (!printed_env)
   {
      printf("OpenMP environment    : unset\n");
   }
   printf("\n");
}

#ifdef HAVE_HWLOC
static int
InitHwlocTopology(void)
{
   if (topology != NULL)
   {
      return 0; // Already initialized
   }
   if (hwloc_topology_init(&topology) < 0)
   {
      return -1;
   }

   // Set flags for accurate binding information
   hwloc_topology_set_flags(topology, HWLOC_TOPOLOGY_FLAG_IS_THISSYSTEM);

   // Include PCI and OS devices
   hwloc_topology_set_type_filter(topology, HWLOC_OBJ_PCI_DEVICE,
                                  HWLOC_TYPE_FILTER_KEEP_IMPORTANT);
   hwloc_topology_set_type_filter(topology, HWLOC_OBJ_OS_DEVICE,
                                  HWLOC_TYPE_FILTER_KEEP_IMPORTANT);

   if (hwloc_topology_load(topology) < 0)
   {
      hwloc_topology_destroy(topology);
      topology = NULL;
      return -1;
   }

   return 0;
}

static void
CleanupHwlocTopology(void)
{
   if (topology != NULL)
   {
      hwloc_topology_destroy(topology);
      topology = NULL;
   }
}

static void
PrintCacheHierarchy(void)
{
   if (topology == NULL)
   {
      return;
   }

   printf("\nCache Hierarchy\n");
   printf("-----------------\n");

   // Group caches by level and show summary
   for (int level = 1; level <= 5; level++)
   {
      hwloc_obj_type_t cache_type;
      switch (level)
      {
         case 1:
            cache_type = HWLOC_OBJ_L1CACHE;
            break;
         case 2:
            cache_type = HWLOC_OBJ_L2CACHE;
            break;
         case 3:
            cache_type = HWLOC_OBJ_L3CACHE;
            break;
         case 4:
            cache_type = HWLOC_OBJ_L4CACHE;
            break;
         case 5:
            cache_type = HWLOC_OBJ_L5CACHE;
            break;
         default:
            continue;
      }

      int count = hwloc_get_nbobjs_by_type(topology, cache_type);
      if (count == 0)
      {
         continue;
      }

      // Get first cache to get size info
      hwloc_obj_t first_cache = hwloc_get_obj_by_type(topology, cache_type, 0);
      if (!first_cache)
      {
         continue;
      }

      char type_name[32];
      hwloc_obj_type_snprintf(type_name, sizeof(type_name), first_cache, 0);

      unsigned long long size_kb     = first_cache->attr->cache.size / 1024;
      int                sharing_pus = hwloc_bitmap_weight(first_cache->cpuset);

      // Check if all caches have the same size
      bool uniform_size = true;
      for (int i = 1; i < count; i++)
      {
         hwloc_obj_t cache = hwloc_get_obj_by_type(topology, cache_type, i);
         if (cache && cache->attr->cache.size != first_cache->attr->cache.size)
         {
            uniform_size = false;
            break;
         }
      }

      if (count == 1)
      {
         printf("  %s: %llu KB, shared by %d PU\n", type_name, size_kb, sharing_pus);
      }
      else if (uniform_size)
      {
         printf("  %s: %d x %llu KB, each shared by %d PU\n", type_name, count, size_kb,
                sharing_pus);
      }
      else
      {
         // Show range of sizes
         unsigned long long min_size = size_kb, max_size = size_kb;
         for (int i = 1; i < count; i++)
         {
            hwloc_obj_t cache = hwloc_get_obj_by_type(topology, cache_type, i);
            if (cache)
            {
               unsigned long long sz = cache->attr->cache.size / 1024;
               if (sz < min_size) min_size = sz;
               if (sz > max_size) max_size = sz;
            }
         }
         if (min_size == max_size)
         {
            printf("  %s: %d x %llu KB, each shared by %d PU\n", type_name, count,
                   min_size, sharing_pus);
         }
         else
         {
            printf("  %s: %d x %llu-%llu KB, each shared by %d PU\n", type_name, count,
                   min_size, max_size, sharing_pus);
         }
      }
   }
}

static void
PrintCpuTopologyInfo(MPI_Comm comm)
{
   if (topology == NULL)
   {
      return;
   }

   int myid = 0, nprocs = 0;
   MPI_Comm_rank(comm, &myid);
   MPI_Comm_size(comm, &nprocs);

   // Gather hostnames for multi-node information
   char hostname[HYPRE_MAX_HOSTNAME];
   gethostname(hostname, sizeof(hostname));
   char *allHostnames = NULL;
   if (nprocs > 0)
   {
      allHostnames = (char *)malloc((size_t)nprocs * HYPRE_MAX_HOSTNAME);
   }
   int host_alloc_ok = allHostnames ? 1 : 0;
   MPI_Allreduce(MPI_IN_PLACE, &host_alloc_ok, 1, MPI_INT, MPI_MIN, comm);
   if (!host_alloc_ok)
   {
      if (allHostnames)
      {
         free(allHostnames);
      }
      return;
   }
   MPI_Allgather(hostname, HYPRE_MAX_HOSTNAME, MPI_CHAR, allHostnames, HYPRE_MAX_HOSTNAME,
                 MPI_CHAR, comm);

   if (!myid)
   {
      // Count unique hostnames
      int numNodes = 0;
      for (int i = 0; i < nprocs; i++)
      {
         int isUnique = 1;
         for (int j = 0; j < i; j++)
         {
            if (strncmp(&allHostnames[i * HYPRE_MAX_HOSTNAME],
                        &allHostnames[j * HYPRE_MAX_HOSTNAME], HYPRE_MAX_HOSTNAME) == 0)
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

      int packages = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_PACKAGE);
      int cores    = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_CORE);
      int pus      = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_PU);
      int numas    = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_NUMANODE);

      printf("CPU Topology\n");
      printf("------------\n");
      printf("Number of Nodes       : %d\n", numNodes);
      printf("Packages (sockets)    : %d\n", packages);
      printf("Cores                : %d\n", cores);
      printf("Processing Units     : %d\n", pus);
      printf("NUMA domains         : %d\n", numas);
      if (cores > 0 && packages > 0)
      {
         printf("Cores per package    : %d\n", cores / packages);
      }
      if (pus > 0 && cores > 0)
      {
         printf("PUs per core (SMT)   : %d-way\n", pus / cores);
      }
      if (numNodes > 1)
      {
         printf("Tot. # of Processors : %lld\n",
                (long long)numNodes * (long long)packages);
         printf("Tot. # of CPU threads: %lld\n", (long long)numNodes * (long long)pus);
      }

      // Print CPU model for each package
      for (int i = 0; i < packages; i++)
      {
         hwloc_obj_t package = hwloc_get_obj_by_type(topology, HWLOC_OBJ_PACKAGE, i);
         if (package)
         {
            const char *cpuvendor = hwloc_obj_get_info_by_name(package, "CPUVendor");
            const char *cpumodel  = hwloc_obj_get_info_by_name(package, "CPUModel");
            if (cpuvendor || cpumodel)
            {
               char cpu_desc[256] = "";
               if (cpuvendor && cpumodel)
               {
                  snprintf(cpu_desc, sizeof(cpu_desc), "%s %s", cpuvendor, cpumodel);
               }
               else if (cpuvendor)
               {
                  snprintf(cpu_desc, sizeof(cpu_desc), "%s", cpuvendor);
               }
               else
               {
                  snprintf(cpu_desc, sizeof(cpu_desc), "%s", cpumodel);
               }
               TrimTrailingWhitespace(cpu_desc);

               if (packages > 1)
               {
                  printf("CPU Model #%d         : %s\n", i, cpu_desc);
               }
               else
               {
                  printf("CPU Model             : %s\n", cpu_desc);
               }
            }
         }
      }

      // Get CPU frequency if available
      hwloc_obj_t pu = hwloc_get_obj_by_type(topology, HWLOC_OBJ_PU, 0);
      if (pu)
      {
         const char *freq_str = hwloc_obj_get_info_by_name(pu, "CPUFrequency");
         if (freq_str)
         {
            double freq_mhz = atof(freq_str);
            if (freq_mhz > 0)
            {
               printf("CPU Frequency         : %.2f GHz\n", freq_mhz / 1000.0);
            }
         }
      }

      PrintCacheHierarchy();

      free(allHostnames);
   }
}

static hwloc_obj_t
GetPciAncestor(hwloc_obj_t obj)
{
   while (obj && obj->type != HWLOC_OBJ_PCI_DEVICE)
   {
      obj = obj->parent;
   }
   return obj;
}

static int
DiscoverGpus(GpuInfo **gpus, int *count)
{
   if (topology == NULL)
   {
      return -1;
   }

   int max_gpus = 16;
   *gpus        = (GpuInfo *)calloc(max_gpus, sizeof(GpuInfo));
   if (!*gpus)
   {
      *count = 0;
      return -1;
   }

   int gpu_count = 0;

   // First try: Use OS devices to find GPUs (like mpibind does)
   hwloc_obj_t os_dev = NULL;
   while ((os_dev = hwloc_get_next_osdev(topology, os_dev)) != NULL)
   {
      if (os_dev->attr->osdev.type != HWLOC_OBJ_OSDEV_GPU &&
          os_dev->attr->osdev.type != HWLOC_OBJ_OSDEV_COPROC)
      {
         continue;
      }

      // Skip CUDA/OpenCL coprocessors (use NVML/RSMI instead)
      if (os_dev->attr->osdev.type == HWLOC_OBJ_OSDEV_COPROC)
      {
         const char *subtype = os_dev->subtype;
         if (subtype && (strcmp(subtype, "CUDA") == 0 || strcmp(subtype, "OpenCL") == 0))
         {
            continue;
         }
      }

      if (gpu_count >= max_gpus)
      {
         break;
      }

      // Get PCI ancestor
      hwloc_obj_t pci_obj = GetPciAncestor(os_dev);
      if (!pci_obj || pci_obj->attr->pcidev.class_id >> 8 != 0x03)
      {
         continue;
      }

      GpuInfo *gpu  = &(*gpus)[gpu_count];
      gpu->obj      = pci_obj;
      gpu->ancestor = hwloc_get_non_io_ancestor_obj(topology, os_dev);
      gpu->smi_id   = -1; // Not set (would need OS device name parsing)

      snprintf(gpu->pci_busid, sizeof(gpu->pci_busid), "%04x:%02x:%02x.%d",
               pci_obj->attr->pcidev.domain, pci_obj->attr->pcidev.bus,
               pci_obj->attr->pcidev.dev, pci_obj->attr->pcidev.func);

      // Get UUID from OS device (this is the key fix!)
      const char *uuid = hwloc_obj_get_info_by_name(os_dev, "NVIDIAUUID");
      if (!uuid)
      {
         uuid = hwloc_obj_get_info_by_name(os_dev, "AMDUUID");
      }
      if (!uuid)
      {
         uuid = hwloc_obj_get_info_by_name(os_dev, "LevelZeroUUID");
      }
      if (uuid)
      {
         strncpy(gpu->uuid, uuid, sizeof(gpu->uuid) - 1);
         gpu->uuid[sizeof(gpu->uuid) - 1] = '\0';
      }
      else
      {
         strncpy(gpu->uuid, "N/A", sizeof(gpu->uuid) - 1);
      }

      // Vendor and Model from OS device if available, otherwise from PCI
      const char *vendor_name = hwloc_obj_get_info_by_name(os_dev, "GPUVendor");
      if (!vendor_name)
      {
         vendor_name = hwloc_obj_get_info_by_name(os_dev, "LevelZeroVendor");
      }
      if (!vendor_name)
      {
         vendor_name = hwloc_obj_get_info_by_name(pci_obj, "PCIVendor");
      }
      if (vendor_name)
      {
         // Get first word only
         sscanf(vendor_name, "%s", gpu->vendor);
      }
      else
      {
         snprintf(gpu->vendor, sizeof(gpu->vendor), "0x%04x",
                  pci_obj->attr->pcidev.vendor_id);
      }

      const char *model_name = hwloc_obj_get_info_by_name(os_dev, "GPUModel");
      if (!model_name)
      {
         model_name = hwloc_obj_get_info_by_name(os_dev, "LevelZeroModel");
      }
      if (!model_name)
      {
         model_name = hwloc_obj_get_info_by_name(pci_obj, "PCIDevice");
      }
      if (model_name)
      {
         strncpy(gpu->model, model_name, sizeof(gpu->model) - 1);
         gpu->model[sizeof(gpu->model) - 1] = '\0';
      }
      else
      {
         snprintf(gpu->model, sizeof(gpu->model), "0x%04x",
                  pci_obj->attr->pcidev.device_id);
      }

      gpu_count++;
   }

   // Fallback: If no GPUs found via OS devices, try PCI devices directly
   if (gpu_count == 0)
   {
      hwloc_obj_t pci_obj = NULL;
      while ((pci_obj = hwloc_get_next_obj_by_type(topology, HWLOC_OBJ_PCI_DEVICE,
                                                   pci_obj)) != NULL)
      {
         // Class 0x03xx is a display controller
         if (pci_obj->attr->pcidev.class_id >> 8 != 0x03)
         {
            continue;
         }

         if (gpu_count >= max_gpus)
         {
            break;
         }

         GpuInfo *gpu  = &(*gpus)[gpu_count];
         gpu->obj      = pci_obj;
         gpu->ancestor = hwloc_get_non_io_ancestor_obj(topology, pci_obj);
         gpu->smi_id   = -1;

         snprintf(gpu->pci_busid, sizeof(gpu->pci_busid), "%04x:%02x:%02x.%d",
                  pci_obj->attr->pcidev.domain, pci_obj->attr->pcidev.bus,
                  pci_obj->attr->pcidev.dev, pci_obj->attr->pcidev.func);

         // Try to find OS device for UUID
         hwloc_obj_t os_dev = NULL;
         while ((os_dev = hwloc_get_next_osdev(topology, os_dev)) != NULL)
         {
            hwloc_obj_t os_pci = GetPciAncestor(os_dev);
            if (os_pci == pci_obj)
            {
               const char *uuid = hwloc_obj_get_info_by_name(os_dev, "NVIDIAUUID");
               if (!uuid)
               {
                  uuid = hwloc_obj_get_info_by_name(os_dev, "AMDUUID");
               }
               if (!uuid)
               {
                  uuid = hwloc_obj_get_info_by_name(os_dev, "LevelZeroUUID");
               }
               if (uuid)
               {
                  strncpy(gpu->uuid, uuid, sizeof(gpu->uuid) - 1);
                  gpu->uuid[sizeof(gpu->uuid) - 1] = '\0';
                  break;
               }
            }
         }
         if (gpu->uuid[0] == '\0')
         {
            strncpy(gpu->uuid, "N/A", sizeof(gpu->uuid) - 1);
         }

         // Vendor
         const char *vendor_name = hwloc_obj_get_info_by_name(pci_obj, "PCIVendor");
         if (vendor_name)
         {
            sscanf(vendor_name, "%s", gpu->vendor);
         }
         else
         {
            snprintf(gpu->vendor, sizeof(gpu->vendor), "0x%04x",
                     pci_obj->attr->pcidev.vendor_id);
         }

         // Model
         const char *model_name = hwloc_obj_get_info_by_name(pci_obj, "PCIDevice");
         if (model_name)
         {
            strncpy(gpu->model, model_name, sizeof(gpu->model) - 1);
            gpu->model[sizeof(gpu->model) - 1] = '\0';
         }
         else
         {
            snprintf(gpu->model, sizeof(gpu->model), "0x%04x",
                     pci_obj->attr->pcidev.device_id);
         }

         gpu_count++;
      }
   }

   *count = gpu_count;
   if (gpu_count == 0)
   {
      free(*gpus);
      *gpus = NULL;
   }

   return 0;
}

static void
PrintGpuInfo(GpuInfo *gpus, int gpu_count)
{
   if (gpu_count == 0)
   {
      printf("\nGPU Information       : Not detected\n");
      return;
   }

   printf("\nGPU Information\n");
   printf("----------------\n");

   double mib_to_gib   = (double)(1 << 10);
   double bytes_to_gib = (double)(1 << 30);

   for (int i = 0; i < gpu_count; i++)
   {
      printf("GPU #%d\n", i);
      printf("  Model                : %s %s\n", gpus[i].vendor, gpus[i].model);
      printf("  PCI Bus ID           : %s\n", gpus[i].pci_busid);

      // Get PCIe link information
      if (gpus[i].obj)
      {
         double linkspeed = gpus[i].obj->attr->pcidev.linkspeed;
         if (linkspeed > 0)
         {
            printf("  PCIe Link Speed      : %.2f GT/s\n", linkspeed);
         }
      }

      if (strcmp(gpus[i].uuid, "N/A") != 0)
      {
         printf("  UUID                 : %s\n", gpus[i].uuid);
      }

      if (gpus[i].smi_id >= 0)
      {
         printf("  SMI ID               : %d\n", gpus[i].smi_id);
      }

      // Get GPU memory information from nvidia-smi or rocm-smi
      FILE  *fp = NULL;
      char   buffer[256];
      size_t total = 0, used = 0;
      bool   found_memory = false;

      // Try nvidia-smi first - query all GPUs and find the one matching our index
      if (system("command -v nvidia-smi > /dev/null 2>&1") == 0)
      {
         fp = popen("nvidia-smi --query-gpu=index,memory.total,memory.used "
                    "--format=csv,noheader,nounits 2>/dev/null",
                    "r");
         if (fp != NULL)
         {
            int line_idx = 0;
            while (fgets(buffer, sizeof(buffer), fp) != NULL && line_idx <= i)
            {
               if (line_idx == i)
               {
                  int idx;
                  if (sscanf(buffer, "%d, %zu, %zu", &idx, &total, &used) == 3)
                  {
                     printf("  Memory               : %6.2f / %6.2f GiB (%5.2f %%)\n",
                            used / mib_to_gib, total / mib_to_gib,
                            100.0 * used / (double)total);
                     found_memory = true;
                  }
                  break;
               }
               line_idx++;
            }
            pclose(fp);
         }
      }

      // Try amd-smi if nvidia-smi didn't work
      if (!found_memory && system("command -v amd-smi > /dev/null 2>&1") == 0)
      {
         char cmd[256];
         snprintf(cmd, sizeof(cmd), "amd-smi metric -g %d -m --json 2>/dev/null", i);
         fp = popen(cmd, "r");
         if (fp != NULL)
         {
            char   json_buffer[32768];
            size_t read       = fread(json_buffer, 1, sizeof(json_buffer) - 1, fp);
            json_buffer[read] = '\0';
            pclose(fp);

            // Parse amd-smi JSON format: "total_vram": {"value": 20464, "unit": "MB"}
            const char *total_vram_str = "\"total_vram\"";
            const char *used_vram_str  = "\"used_vram\"";
            const char *ptr            = json_buffer;

            // Find total_vram
            ptr = strstr(ptr, total_vram_str);
            if (ptr)
            {
               ptr = strstr(ptr, "\"value\"");
               if (ptr)
               {
                  ptr = strchr(ptr, ':');
                  if (ptr)
                  {
                     ptr++;
                     while (*ptr == ' ') ptr++;
                     total = strtoull(ptr, NULL, 10);
                     // Convert MB to bytes, then to GiB
                     total = total * 1024 * 1024;
                  }
               }
            }

            // Find used_vram
            ptr = json_buffer;
            ptr = strstr(ptr, used_vram_str);
            if (ptr)
            {
               ptr = strstr(ptr, "\"value\"");
               if (ptr)
               {
                  ptr = strchr(ptr, ':');
                  if (ptr)
                  {
                     ptr++;
                     while (*ptr == ' ') ptr++;
                     used = strtoull(ptr, NULL, 10);
                     // Convert MB to bytes, then to GiB
                     used = used * 1024 * 1024;
                  }
               }
            }

            if (total > 0)
            {
               printf("  Memory               : %6.2f / %6.2f GiB (%5.2f %%)\n",
                      used / bytes_to_gib, total / bytes_to_gib,
                      100.0 * used / (double)total);
               found_memory = true;
            }
         }
      }
   }
   printf("\n");
}

static void
PrintNumaInfo(double bytes_to_gib, GpuInfo *gpus, int gpu_count)
{
   if (topology == NULL)
   {
      return;
   }

   int num_numas = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_NUMANODE);
   if (num_numas <= 0)
   {
      printf("NUMA Information       : Not available\n\n");
      return;
   }

   printf("NUMA Information\n");
   printf("-----------------\n");

   hwloc_obj_t numa = NULL;
   while ((numa = hwloc_get_next_obj_by_type(topology, HWLOC_OBJ_NUMANODE, numa)) != NULL)
   {
      char cpuset_str[256];
      hwloc_bitmap_list_snprintf(cpuset_str, sizeof(cpuset_str), numa->cpuset);

      unsigned long long total_mem = numa->attr->numanode.local_memory;
      int                pu_count  = hwloc_bitmap_weight(numa->cpuset);

      printf("NUMA node %d\n", numa->os_index);
      printf("  CPUs                 : %s (%d PUs)\n", cpuset_str, pu_count);
      printf("  Memory (GiB)         : %.2f\n", total_mem / bytes_to_gib);

      // Count local GPUs
      int local_gpu_count = 0;
      if (gpus)
      {
         for (int i = 0; i < gpu_count; i++)
         {
            if (gpus[i].ancestor &&
                hwloc_bitmap_isset(gpus[i].ancestor->nodeset, numa->os_index))
            {
               local_gpu_count++;
            }
         }
      }
      if (local_gpu_count > 0)
      {
         printf("  Local GPUs           : %d", local_gpu_count);
         if (gpus)
         {
            bool first = true;
            for (int i = 0; i < gpu_count; i++)
            {
               if (gpus[i].ancestor &&
                   hwloc_bitmap_isset(gpus[i].ancestor->nodeset, numa->os_index))
               {
                  printf("%s%s", first ? " (" : ", ", gpus[i].pci_busid);
                  first = false;
               }
            }
            if (!first)
            {
               printf(")");
            }
         }
         printf("\n");
      }
   }
   printf("\n");
}

static void
PrintNetworkInfoHwloc(void)
{
   if (topology == NULL)
   {
      return;
   }

   printf("Network / Interconnect\n");
   printf("----------------------\n");

   int         found_ib  = 0;
   int         found_net = 0;
   hwloc_obj_t obj       = NULL;

   while ((obj = hwloc_get_next_osdev(topology, obj)) != NULL)
   {
      if (obj->attr->osdev.type == HWLOC_OBJ_OSDEV_OPENFABRICS)
      {
         found_ib            = 1;
         hwloc_obj_t pci_dev = obj;
         while (pci_dev && pci_dev->type != HWLOC_OBJ_PCI_DEVICE)
         {
            pci_dev = pci_dev->parent;
         }
         char pci_busid[20] = {0};
         if (pci_dev)
         {
            snprintf(pci_busid, sizeof(pci_busid), "%04x:%02x:%02x.%d",
                     pci_dev->attr->pcidev.domain, pci_dev->attr->pcidev.bus,
                     pci_dev->attr->pcidev.dev, pci_dev->attr->pcidev.func);
         }

         const char *nodeguid = hwloc_obj_get_info_by_name(obj, "NodeGUID");
         const char *address  = hwloc_obj_get_info_by_name(obj, "Address");

         printf("InfiniBand %-9s : %s\n", obj->name, obj->name);
         if (pci_busid[0])
         {
            printf("  PCI Bus ID           : %s\n", pci_busid);
         }
         if (nodeguid)
         {
            printf("  NodeGUID             : %s\n", nodeguid);
         }
         if (address)
         {
            printf("  Address              : %s\n", address);
         }
      }
      else if (obj->attr->osdev.type == HWLOC_OBJ_OSDEV_NETWORK)
      {
         found_net = 1;
         printf("Interface %-11s : detected\n", obj->name);
      }
   }

   if (!found_ib)
   {
      printf("InfiniBand            : Not detected\n");
   }

   if (!found_net)
   {
      // Fallback to legacy method
      PrintNetworkInformation();
   }
   printf("\n");
}

static void
PrintProcessBinding(void)
{
   if (topology == NULL)
   {
      return;
   }

   printf("Process Binding\n");
   printf("-----------------\n");

   hwloc_bitmap_t cpuset = hwloc_bitmap_alloc();
   hwloc_bitmap_t memset = hwloc_bitmap_alloc();

   if (hwloc_get_cpubind(topology, cpuset, HWLOC_CPUBIND_PROCESS) == 0)
   {
      char str[256];
      hwloc_bitmap_list_snprintf(str, sizeof(str), cpuset);
      printf("Process CPU binding    : %s\n", str);
   }

   if (hwloc_get_membind(topology, memset, NULL, HWLOC_MEMBIND_PROCESS) == 0)
   {
      char str[256];
      hwloc_bitmap_list_snprintf(str, sizeof(str), memset);
      printf("Process memory binding : %s\n", str);
   }

   hwloc_bitmap_free(cpuset);
   hwloc_bitmap_free(memset);
   printf("\n");
}

static void
PrintThreadAffinity(MPI_Comm comm, GpuInfo *gpus, int gpu_count)
{
#if defined(HYPRE_USING_OPENMP) && defined(_OPENMP)
   int myid = 0, nprocs = 0;
   MPI_Comm_rank(comm, &myid);
   MPI_Comm_size(comm, &nprocs);

   if (topology == NULL)
   {
      return;
   }

   int nthreads = omp_get_max_threads();
   if (nthreads <= 1)
   {
      return; // No threads to report
   }

   // Synchronize all ranks before printing
   MPI_Barrier(comm);

   if (myid == 0)
   {
      printf("Thread Affinity\n");
      printf("----------------\n");
      printf("OpenMP threads per rank: %d\n", nthreads);
      printf("\n");
   }

// Each thread reports its own affinity
#pragma omp parallel
   {
      int tid            = omp_get_thread_num();
      int nthreads_local = omp_get_num_threads();

      hwloc_bitmap_t cpuset = hwloc_bitmap_alloc();
      if (hwloc_get_cpubind(topology, cpuset, HWLOC_CPUBIND_THREAD) == 0)
      {
         char str[256];
         hwloc_bitmap_list_snprintf(str, sizeof(str), cpuset);
         printf("Rank %-3d Thread %-3d/%d: CPUs %s", myid, tid, nthreads_local, str);

         // Try to get GPU assignment for this thread
         if (gpus && gpu_count > 0)
         {
            // For now, just show which GPUs are visible to this process
            // Actual GPU assignment would require CUDA/HIP runtime calls
            printf(" (GPUs visible: %d)", gpu_count);
         }
         printf("\n");
      }
      hwloc_bitmap_free(cpuset);
   }

   // Synchronize all ranks after printing
   MPI_Barrier(comm);

   if (myid == 0)
   {
      printf("\n");
   }
#endif
}

static void
PrintGpuAffinity(MPI_Comm comm, GpuInfo *gpus, int gpu_count)
{
   int myid = 0, nprocs = 0;
   MPI_Comm_rank(comm, &myid);
   MPI_Comm_size(comm, &nprocs);

   if (gpu_count == 0)
   {
      return;
   }

   // Gather GPU visibility per rank
   int *gpu_counts = NULL;
   if (!myid)
   {
      gpu_counts = (int *)malloc(nprocs * sizeof(int));
   }
   MPI_Gather(&gpu_count, 1, MPI_INT, gpu_counts, 1, MPI_INT, 0, comm);

   if (!myid)
   {
      printf("GPU Affinity (per rank)\n");
      printf("------------------------\n");

      for (int r = 0; r < nprocs; r++)
      {
         printf("Rank %-3d              : %d GPU%s visible", r, gpu_counts[r],
                gpu_counts[r] != 1 ? "s" : "");

         // Show which GPUs are visible (based on environment variables)
         char gpuBindingLocal[HYPRE_MAX_GPU_BINDING];
         if (r == 0)
         {
            BuildGpuBindingString(gpuBindingLocal, sizeof(gpuBindingLocal));
            if (strcmp(gpuBindingLocal, "unset") != 0)
            {
               printf(" (%s)", gpuBindingLocal);
            }
         }
         printf("\n");
      }
      printf("\n");

      free(gpu_counts);
   }
}

static void
CountChildrenRecursive(hwloc_obj_t obj, hwloc_obj_type_t type, int *count, int *first_idx,
                       int *last_idx)
{
   // Check this object
   if (obj->type == type)
   {
      (*count)++;
      int idx = obj->logical_index;
      if (*first_idx < 0) *first_idx = idx;
      if (idx > *last_idx) *last_idx = idx;
   }

   // Recurse into ALL children (including through cache objects to find cores/PUs)
   for (unsigned i = 0; i < obj->arity; i++)
   {
      CountChildrenRecursive(obj->children[i], type, count, first_idx, last_idx);
   }
}

static void
PrintTopologyTreeCompact(hwloc_obj_t obj, int depth)
{
   char indent[32];
   memset(indent, ' ', depth * 2);
   indent[depth * 2] = '\0';

   char type_str[32];
   hwloc_obj_type_snprintf(type_str, sizeof(type_str), obj, 1);

   // Skip cache objects
   if (obj->type == HWLOC_OBJ_L1CACHE || obj->type == HWLOC_OBJ_L2CACHE ||
       obj->type == HWLOC_OBJ_L3CACHE || obj->type == HWLOC_OBJ_L4CACHE ||
       obj->type == HWLOC_OBJ_L5CACHE)
   {
      for (unsigned i = 0; i < obj->arity; i++)
      {
         PrintTopologyTreeCompact(obj->children[i], depth);
      }
      return;
   }

   // For Package: just show the name, no summary
   if (obj->type == HWLOC_OBJ_PACKAGE)
   {
      printf("%s%s[%d]\n", indent, type_str, obj->logical_index);
      // Recurse into Dies
      for (unsigned i = 0; i < obj->arity; i++)
      {
         hwloc_obj_t child = obj->children[i];
         if (child->type != HWLOC_OBJ_L1CACHE && child->type != HWLOC_OBJ_L2CACHE &&
             child->type != HWLOC_OBJ_L3CACHE && child->type != HWLOC_OBJ_L4CACHE &&
             child->type != HWLOC_OBJ_L5CACHE && child->type != HWLOC_OBJ_PU)
         {
            PrintTopologyTreeCompact(child, depth + 1);
         }
      }
   }
   // For Die: show summary with counts and ranges, don't recurse into cores
   else if (obj->type == HWLOC_OBJ_DIE)
   {
      int core_count = 0, core_first = -1, core_last = -1;
      int pu_count = 0, pu_first = -1, pu_last = -1;

      CountChildrenRecursive(obj, HWLOC_OBJ_CORE, &core_count, &core_first, &core_last);
      CountChildrenRecursive(obj, HWLOC_OBJ_PU, &pu_count, &pu_first, &pu_last);

      printf("%s%s[%d]:", indent, type_str, obj->logical_index);
      bool first = true;

      if (core_count > 0)
      {
         if (core_count == 1)
         {
            printf(" %d Core (Core[%d])", core_count, core_first);
         }
         else
         {
            printf(" %d Cores (Core[%d-%d])", core_count, core_first, core_last);
         }
         first = false;
      }

      if (pu_count > 0)
      {
         if (!first) printf(",");
         if (pu_count == 1)
         {
            printf(" %d PU (PU[%d])", pu_count, pu_first);
         }
         else
         {
            printf(" %d PUs (PU[%d-%d])", pu_count, pu_first, pu_last);
         }
      }

      printf("\n");
      // Don't recurse - we've shown the summary
   }
   else if (obj->type == HWLOC_OBJ_CORE || obj->type == HWLOC_OBJ_PU)
   {
      // Don't print cores and PUs individually - they're summarized at Die level
      return;
   }
   else
   {
      // For Machine and other types, just show the object
      printf("%s%s[%d]\n", indent, type_str, obj->logical_index);
      // Recurse
      for (unsigned i = 0; i < obj->arity; i++)
      {
         hwloc_obj_t child = obj->children[i];
         if (child->type != HWLOC_OBJ_L1CACHE && child->type != HWLOC_OBJ_L2CACHE &&
             child->type != HWLOC_OBJ_L3CACHE && child->type != HWLOC_OBJ_L4CACHE &&
             child->type != HWLOC_OBJ_L5CACHE)
         {
            PrintTopologyTreeCompact(child, depth + 1);
         }
      }
   }
}

static void
PrintTopologyTree(void)
{
   if (topology == NULL)
   {
      return;
   }

   printf("\nTopology Tree\n");
   printf("---------------\n");

   hwloc_obj_t root = hwloc_get_root_obj(topology);
   PrintTopologyTreeCompact(root, 0);
   printf("\n");
}

static void
PrintMemoryInformation(double bytes_to_gib, double mib_to_gib)
{
   printf("Memory Information (Used/Total)\n");
   printf("--------------------------------\n");

#ifdef __APPLE__
   size_t total = 0, used = 0;
   size_t memSizeLen = sizeof(total);
   sysctlbyname("hw.memsize", &total, &memSizeLen, NULL, 0);

   mach_msg_type_number_t count  = HOST_VM_INFO_COUNT;
   vm_statistics_data_t   vmstat = {0};
   if (host_statistics(mach_host_self(), HOST_VM_INFO, (host_info_t)&vmstat, &count) ==
       KERN_SUCCESS)
   {
      used = total - (size_t)vmstat.free_count * sysconf(_SC_PAGESIZE);

      printf("CPU RAM               : %6.2f / %6.2f  (%5.2f %%) GiB\n",
             (double)used / bytes_to_gib, (double)total / bytes_to_gib,
             100.0 * (total - used) / (double)total);
   }
#else
   struct sysinfo info;
   if (sysinfo(&info) == 0)
   {
      printf("CPU RAM               : %6.2f / %6.2f  (%5.2f %%) GiB\n",
             (double)(info.totalram - info.freeram) * info.mem_unit / bytes_to_gib,
             (double)info.totalram * info.mem_unit / bytes_to_gib,
             100.0 * (info.totalram - info.freeram) / (double)info.totalram);
   }
#endif
   printf("\n");
}

static void
PrintOperatingSystemInfo(void)
{
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
}

static void
PrintCompilationInfo(void)
{
   printf("Compilation Information\n");
   printf("------------------------\n");
   printf("Date                  : %s at %s\n", __DATE__, __TIME__);

   /* Check optimization level */
#if defined(__OPTIMIZE__)
   printf("Optimization          : Enabled\n");
#elif defined(__OPTIMIZE_SIZE__)
   printf("Optimization          : Enabled (size)\n");
#elif defined(_MSC_VER)
   printf("Optimization          : Unknown (MSVC)\n");
#else
   printf("Optimization          : Disabled\n");
#endif
   /* Check debug symbols */
#if defined(HYPRE_DEBUG)
   printf("Debugging             : Enabled (HYPRE)\n");
#elif defined(_DEBUG) || defined(DEBUG)
   printf("Debugging             : Enabled\n");
#elif defined(NDEBUG)
   printf("Debugging             : Disabled\n");
#else
   printf("Debugging             : Unknown\n");
#endif
#ifdef __clang_version__
   printf("Compiler              : Clang %s\n", (const char *)__clang_version__);
#elif defined(__clang__)
   printf("Compiler              : Clang %d.%d.%d\n", __clang_major__, __clang_minor__,
          __clang_patchlevel__);
#elif defined(__INTEL_COMPILER)
   printf("Compiler              : Intel %d.%d\n", __INTEL_COMPILER / 100,
          (__INTEL_COMPILER % 100) / 10);
#elif defined(__GNUC__)
   printf("Compiler              : GCC %d.%d.%d\n", __GNUC__, __GNUC_MINOR__,
          __GNUC_PATCHLEVEL__);
#else
   printf("Compiler              : Unknown\n");
#endif
#if defined(HYPRE_USING_OPENMP) && defined(_OPENMP)
   printf("OpenMP                : Supported (Version: %d)\n", _OPENMP);
#else
   printf("OpenMP                : Not used\n");
#endif
   printf("MPI library           : ");
#ifdef CRAY_MPICH_VERSION
   printf("Cray MPI (Version: %s)\n", TOSTRING(CRAY_MPICH_VERSION));
#elif defined(INTEL_MPI_VERSION)
   printf("Intel MPI (Version: %s)\n", (const char *)INTEL_MPI_VERSION);
#elif defined(__IBM_MPI__)
   printf("IBM Spectrum MPI (Version: %d.%d.%d)\n", __IBM_MPI_MAJOR_VERSION,
          __IBM_MPI_MINOR_VERSION, __IBM_MPI_RELEASE_VERSION);
#elif defined(MVAPICH2_VERSION)
   printf("MVAPICH2 (Version: %s)\n", (const char *)MVAPICH2_VERSION);
#elif defined(MPICH_NAME)
   printf("MPICH (Version: %s)\n", MPICH_VERSION);
#elif defined(OMPI_MAJOR_VERSION)
   printf("OpenMPI (Version: %d.%d.%d)\n", OMPI_MAJOR_VERSION, OMPI_MINOR_VERSION,
          OMPI_RELEASE_VERSION);
#elif defined(SGI_MPI)
   printf("SGI MPI\n");
#else
   printf("N/A\n");
#endif
#ifdef __x86_64__
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
}

static void
PrintWorkingDirectory(void)
{
   printf("Current Working Directory\n");
   printf("--------------------------\n");
   char cwd[4096];
   if (getcwd(cwd, sizeof(cwd)) != NULL)
   {
      printf("%s\n\n", cwd);
   }
}

static void
PrintDynamicLibraries(void)
{
   printf("Dynamic Libraries Loaded\n");
   printf("------------------------\n");
#ifdef __APPLE__
   uint32_t dcount = _dyld_image_count();
   for (uint32_t i = 0; i < dcount; i++)
   {
      const char               *name     = _dyld_get_image_name(i);
      const struct mach_header *header   = _dyld_get_image_header(i);
      const char               *filename = strrchr(name, '/');

      filename = filename ? filename + 1 : name;
      printf("   %s => %s (0x%lx)\n", filename, name, (unsigned long)header);
   }
#else
   dl_iterate_phdr(dlpi_callback, NULL);
#endif
   printf("\n");
}

static void
PrintRunningInfo(MPI_Comm comm)
{
   int myid = 0, nprocs = 0;
   MPI_Comm_rank(comm, &myid);
   MPI_Comm_size(comm, &nprocs);

   if (!myid)
   {
      printf("Running on %d MPI rank%s\n", nprocs, nprocs > 1 ? "s" : "");

      /* Number of OpenMP threads per rank used in hypre */
#if defined(HYPRE_USING_OPENMP) && defined(_OPENMP)
      int num_threads = omp_get_max_threads();
      printf("Running on %d OpenMP thread%s per MPI rank\n", num_threads,
             num_threads > 1 ? "s" : "");
#endif
   }
}

static void
PrintSystemInfoHwloc(MPI_Comm comm)
{
   int myid = 0, nprocs = 0;
   MPI_Comm_rank(comm, &myid);
   MPI_Comm_size(comm, &nprocs);

   double bytes_to_gib = (double)(1 << 30);
   double mib_to_gib   = (double)(1 << 10);

   // Discover GPUs on all ranks (needed for thread affinity)
   GpuInfo *gpus      = NULL;
   int      gpu_count = 0;
   DiscoverGpus(&gpus, &gpu_count);

   // Gather per-rank GPU binding strings on rank 0
   char gpuBindingLocal[HYPRE_MAX_GPU_BINDING];
   BuildGpuBindingString(gpuBindingLocal, sizeof(gpuBindingLocal));
   char *gpuBindingAll = NULL;
   if (!myid && nprocs > 0)
   {
      gpuBindingAll = (char *)malloc((size_t)nprocs * HYPRE_MAX_GPU_BINDING);
   }
   MPI_Gather(gpuBindingLocal, HYPRE_MAX_GPU_BINDING, MPI_CHAR, gpuBindingAll,
              HYPRE_MAX_GPU_BINDING, MPI_CHAR, 0, comm);

   if (!myid)
   {
      printf("================================ System Information (hwloc) "
             "================================\n\n");

      // 1. CPU Topology (includes multi-node summary and detailed info)
      PrintCpuTopologyInfo(comm);

      // 3. GPU Information
      PrintGpuInfo(gpus, gpu_count);

      // 4. Memory Information
      PrintMemoryInformation(bytes_to_gib, mib_to_gib);

      // 5. NUMA Information
      PrintNumaInfo(bytes_to_gib, gpus, gpu_count);

      // 6. Network Information
      PrintNetworkInfoHwloc();

      // 7. Accelerator Runtime Information
#ifndef __APPLE__
      PrintAcceleratorRuntimeInformation();
#endif

      // 8. Accelerator Binding (per rank)
      if (gpuBindingAll)
      {
         printf("Accelerator Binding (per rank)\n");
         printf("-------------------------------\n");
         for (int r = 0; r < nprocs; r++)
         {
            printf("Rank %-3d              : %s\n", r,
                   gpuBindingAll + (size_t)r * HYPRE_MAX_GPU_BINDING);
         }
         printf("\n");
      }

      // 9. Process Binding
      PrintProcessBinding();

      // 9a. GPU Affinity (per rank)
      PrintGpuAffinity(comm, gpus, gpu_count);

      // 9b. Thread Affinity (if OpenMP is enabled)
      PrintThreadAffinity(comm, gpus, gpu_count);

      // 10. Topology Tree
      PrintTopologyTree();

      // 11. Operating System
      PrintOperatingSystemInfo();

      // 12. Compilation Information
      PrintCompilationInfo();

      // 13. MPI Runtime Information
      PrintMpiRuntimeInformation(comm);

      // 14. Threading Environment
      PrintThreadingEnvironmentInformation();

#ifndef __APPLE__
      // 15. Linux Kernel Tuning
      PrintLinuxKernelTuningInformation();
#endif

      // 16. Current Working Directory
      PrintWorkingDirectory();

      // 17. Dynamic Libraries
      PrintDynamicLibraries();

      // 18. hwloc Information
      printf("hwloc Information\n");
      printf("-----------------\n");
      unsigned version = hwloc_get_api_version();
      printf("hwloc API version    : %d.%d.%d\n", (version >> 16) & 0xff,
             (version >> 8) & 0xff, version & 0xff);
      printf("\n");

      // 19. Running Information
      PrintRunningInfo(comm);

      if (gpuBindingAll)
      {
         free(gpuBindingAll);
      }

      printf("================================ System Information (hwloc) "
             "================================\n\n");
   }

   // Free GPUs on all ranks
   if (gpus)
   {
      free(gpus);
   }
}
#endif

#ifndef __APPLE__

static int
ReadLineFromFile(const char *path, char *out, size_t len)
{
   if (!path || !out || len == 0)
   {
      return 0;
   }

   FILE *fp = fopen(path, "r");
   if (!fp)
   {
      return 0;
   }

   if (!fgets(out, (int)len, fp))
   {
      fclose(fp);
      return 0;
   }
   fclose(fp);

   out[strcspn(out, "\n")] = '\0';
   return 1;
}

static int
ReadIntFromFile(const char *path, int *value)
{
   if (!value)
   {
      return 0;
   }

   char line[64];
   if (!ReadLineFromFile(path, line, sizeof(line)))
   {
      return 0;
   }

   *value = atoi(line);
   return 1;
}

static int
ReadUllFromProcMeminfo(const char *field, unsigned long long *value)
{
   if (!field || !value)
   {
      return 0;
   }

   FILE *fp = fopen("/proc/meminfo", "r");
   if (!fp)
   {
      return 0;
   }

   int  found = 0;
   char line[256];
   while (fgets(line, sizeof(line), fp))
   {
      char               key[128];
      unsigned long long parsed = 0;
      if (sscanf(line, "%127[^:]: %llu", key, &parsed) == 2 && strcmp(key, field) == 0)
      {
         *value = parsed;
         found  = 1;
         break;
      }
   }

   fclose(fp);
   return found;
}

static int
ExtractBracketedToken(const char *line, char *token, size_t len)
{
   if (!line || !token || len == 0)
   {
      return 0;
   }

   const char *start = strchr(line, '[');
   if (!start)
   {
      return 0;
   }
   start++;

   const char *end = strchr(start, ']');
   if (!end || end <= start)
   {
      return 0;
   }

   size_t n = (size_t)(end - start);
   if (n >= len)
   {
      n = len - 1;
   }
   memcpy(token, start, n);
   token[n] = '\0';
   return 1;
}

static void
PrintLinuxKernelTuningInformation(void)
{
   printf("Linux Kernel Tuning\n");
   printf("--------------------\n");

   struct sysinfo info;
   if (sysinfo(&info) == 0)
   {
      long uptime = info.uptime;
      long days   = uptime / 86400;
      uptime      = uptime % 86400;
      long hours  = uptime / 3600;
      uptime      = uptime % 3600;
      long mins   = uptime / 60;

      printf("System uptime         : %ldd %02ldh %02ldm\n", days, hours, mins);
      printf("Load average (1/5/15) : %.2f / %.2f / %.2f\n", info.loads[0] / 65536.0,
             info.loads[1] / 65536.0, info.loads[2] / 65536.0);
   }

   char line[256];
   char token[64];

   if (ReadLineFromFile("/sys/kernel/mm/transparent_hugepage/enabled", line,
                        sizeof(line)))
   {
      if (ExtractBracketedToken(line, token, sizeof(token)))
      {
         printf("THP policy            : %s\n", token);
      }
      else
      {
         printf("THP policy            : %s\n", line);
      }
   }

   if (ReadLineFromFile("/sys/kernel/mm/transparent_hugepage/defrag", line, sizeof(line)))
   {
      if (ExtractBracketedToken(line, token, sizeof(token)))
      {
         printf("THP defrag policy     : %s\n", token);
      }
      else
      {
         printf("THP defrag policy     : %s\n", line);
      }
   }

   int numa_balancing = -1;
   if (ReadIntFromFile("/proc/sys/kernel/numa_balancing", &numa_balancing))
   {
      printf("NUMA balancing        : %s\n", numa_balancing ? "enabled" : "disabled");
   }

   int overcommit = -1;
   if (ReadIntFromFile("/proc/sys/vm/overcommit_memory", &overcommit))
   {
      const char *mode = "unknown";
      if (overcommit == 0)
      {
         mode = "heuristic";
      }
      else if (overcommit == 1)
      {
         mode = "always";
      }
      else if (overcommit == 2)
      {
         mode = "never";
      }
      printf("Overcommit policy     : %d (%s)\n", overcommit, mode);
   }

   int zone_reclaim = -1;
   if (ReadIntFromFile("/proc/sys/vm/zone_reclaim_mode", &zone_reclaim))
   {
      printf("Zone reclaim mode     : %d\n", zone_reclaim);
   }

   unsigned long long hugepages_total = 0;
   unsigned long long hugepages_free  = 0;
   unsigned long long hugepage_size   = 0;
   if (ReadUllFromProcMeminfo("HugePages_Total", &hugepages_total))
   {
      printf("HugePages total/free  : %llu / ", hugepages_total);
      if (ReadUllFromProcMeminfo("HugePages_Free", &hugepages_free))
      {
         printf("%llu\n", hugepages_free);
      }
      else
      {
         printf("unknown\n");
      }
   }
   if (ReadUllFromProcMeminfo("Hugepagesize", &hugepage_size))
   {
      printf("HugePage size         : %llu kB\n", hugepage_size);
   }

   printf("\n");
}

static void
PrintLinuxNumaInformation(double bytes_to_gib)
{
   printf("\nNUMA Information\n");
   printf("-----------------\n");

   DIR *node_dir = opendir("/sys/devices/system/node");
   if (!node_dir)
   {
      printf("NUMA details unavailable (missing /sys/devices/system/node)\n\n");
      return;
   }

   struct dirent *entry = NULL;
   int            nodes = 0;
   while ((entry = readdir(node_dir)) != NULL)
   {
      if (strncmp(entry->d_name, "node", 4) != 0)
      {
         continue;
      }
      if (!isdigit((unsigned char)entry->d_name[4]))
      {
         continue;
      }

      char meminfo_path[PATH_MAX];
      snprintf(meminfo_path, sizeof(meminfo_path), "/sys/devices/system/node/%s/meminfo",
               entry->d_name);

      FILE *memfp = fopen(meminfo_path, "r");
      if (!memfp)
      {
         continue;
      }

      unsigned long long mem_total_kb = 0;
      unsigned long long mem_free_kb  = 0;
      char               line[256];
      while (fgets(line, sizeof(line), memfp))
      {
         if (strncmp(line, "MemTotal:", 9) == 0)
         {
            sscanf(line, "MemTotal: %llu kB", &mem_total_kb);
         }
         else if (strncmp(line, "MemFree:", 8) == 0)
         {
            sscanf(line, "MemFree: %llu kB", &mem_free_kb);
         }
      }
      fclose(memfp);

      double total_gib = (double)mem_total_kb * 1024.0 / bytes_to_gib;
      double free_gib  = (double)mem_free_kb * 1024.0 / bytes_to_gib;
      double used_gib  = total_gib - free_gib;
      double used_pct  = total_gib > 0.0 ? (used_gib / total_gib) * 100.0 : 0.0;

      int node_index = atoi(entry->d_name + 4);
      if (mem_total_kb > 0)
      {
         printf("NUMA node %-3d         : %6.2f / %6.2f  (%5.2f %%) GiB used\n",
                node_index, used_gib, total_gib, used_pct);
      }
      else
      {
         printf("NUMA node %-3d         : memory data unavailable\n", node_index);
      }
      nodes++;
   }
   closedir(node_dir);

   if (nodes == 0)
   {
      printf("No NUMA nodes detected.\n");
   }
   printf("\n");
}

static void
PrintNetworkInformation(void)
{
   printf("Network / Interconnect\n");
   printf("----------------------\n");

   DIR *ib_dir = opendir("/sys/class/infiniband");
   if (ib_dir)
   {
      struct dirent *entry = NULL;
      int            found = 0;
      char           line[256];
      while ((entry = readdir(ib_dir)) != NULL)
      {
         if (entry->d_name[0] == '.')
         {
            continue;
         }
         found = 1;
         char desc_path[PATH_MAX];
         snprintf(desc_path, sizeof(desc_path), "/sys/class/infiniband/%s/node_desc",
                  entry->d_name);
         if (ReadLineFromFile(desc_path, line, sizeof(line)))
         {
            printf("InfiniBand %-9s : %s\n", entry->d_name, line);
         }
         else
         {
            printf("InfiniBand %-9s : (description unavailable)\n", entry->d_name);
         }
      }
      closedir(ib_dir);
      if (!found)
      {
         printf("InfiniBand            : Not detected\n");
      }
   }
   else
   {
      printf("InfiniBand            : Not detected\n");
   }

   DIR *net_dir = opendir("/sys/class/net");
   if (net_dir)
   {
      struct dirent *entry = NULL;
      while ((entry = readdir(net_dir)) != NULL)
      {
         if (entry->d_name[0] == '.')
         {
            continue;
         }
         if (strcmp(entry->d_name, "lo") == 0)
         {
            continue;
         }

         char speed_path[PATH_MAX];
         snprintf(speed_path, sizeof(speed_path), "/sys/class/net/%s/speed",
                  entry->d_name);
         int speed_mbps = -1;
         ReadIntFromFile(speed_path, &speed_mbps);

         char state_path[PATH_MAX];
         snprintf(state_path, sizeof(state_path), "/sys/class/net/%s/operstate",
                  entry->d_name);
         char state[64];
         state[0] = '\0';
         ReadLineFromFile(state_path, state, sizeof(state));

         if (speed_mbps > 0)
         {
            printf("Interface %-11s : %4d Mbps (%s)\n", entry->d_name, speed_mbps,
                   state[0] ? state : "unknown");
         }
         else
         {
            printf("Interface %-11s : unknown speed (%s)\n", entry->d_name,
                   state[0] ? state : "unknown");
         }
      }
      closedir(net_dir);
   }
   else
   {
      printf("Network interfaces    : Unavailable\n");
   }
   printf("\n");
}

static void
PrintAcceleratorRuntimeInformation(void)
{
   printf("Accelerator Runtime Information\n");
   printf("--------------------------------\n");

   int   printed = 0;
   FILE *fp      = NULL;
   char  line[256];

   if (system("command -v nvidia-smi > /dev/null 2>&1") == 0)
   {
      fp = popen("nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1",
                 "r");
      if (fp)
      {
         if (fgets(line, sizeof(line), fp) != NULL)
         {
            line[strcspn(line, "\n")] = '\0';
            if (line[0] != '\0')
            {
               printf("NVIDIA driver         : %s\n", line);
               printed = 1;
            }
         }
         pclose(fp);
      }

      fp =
         popen("nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -u", "r");
      if (fp)
      {
         int cap_idx = 0;
         while (fgets(line, sizeof(line), fp) != NULL)
         {
            line[strcspn(line, "\n")] = '\0';
            if (cap_idx == 0)
            {
               printf("CUDA comp. capability : %s\n", line[0] ? line : "unknown");
            }
            else
            {
               printf("                        %s\n", line[0] ? line : "unknown");
            }
            printed = 1;
            cap_idx++;
         }
         pclose(fp);
      }
   }

   if (system("command -v amd-smi > /dev/null 2>&1") == 0)
   {
      fp = popen("amd-smi version 2>/dev/null", "r");
      if (fp)
      {
         // Read the first line which contains version info
         if (fgets(line, sizeof(line), fp) != NULL)
         {
            // Extract ROCm version from the output
            // Format: "AMDSMI Tool: 25.5.1+41065ee6 | AMDSMI Library version: 25.5.1 |
            // ROCm version: 6.4.3 | ..."
            const char *rocm_ver = strstr(line, "ROCm version:");
            if (rocm_ver)
            {
               rocm_ver += strlen("ROCm version:");
               while (*rocm_ver == ' ') rocm_ver++;
               // Extract version number (until next | or end of line)
               char version[64] = {0};
               int  i           = 0;
               while (*rocm_ver && *rocm_ver != '|' && i < (int)(sizeof(version) - 1))
               {
                  version[i++] = *rocm_ver++;
               }
               version[i] = '\0';
               // Trim trailing spaces
               while (i > 0 && version[i - 1] == ' ') version[--i] = '\0';
               printf("AMD driver            : ROCm %s\n", version);
            }
            else
            {
               // Fallback: try to extract any version info
               line[strcspn(line, "\n")] = '\0';
               printf("AMD driver            : %s\n", line);
            }
            printed = 1;
         }
         pclose(fp);
      }
   }

   const char *level_zero = getenv("ONEAPI_DEVICE_SELECTOR");
   if (level_zero && level_zero[0])
   {
      printf("oneAPI selector       : %s\n", level_zero);
      printed = 1;
   }

   if (!printed)
   {
      printf("No accelerator tooling detected.\n");
   }
   printf("\n");
}

#endif /* !__APPLE__ */

/*--------------------------------------------------------------------------
 * PrintLibInfo
 *--------------------------------------------------------------------------*/

void
PrintLibInfo(MPI_Comm comm, int print_datetime)
{
   int myid = 0;

   MPI_Comm_rank(comm, &myid);

   if (!myid)
   {
      if (print_datetime)
      {
         time_t           t       = 0;
         const struct tm *tm_info = NULL;
         char             buffer[100];

         /* Get current time */
         time(&t);
         tm_info = localtime(&t);

         /* Format and print the date and time */
         strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", tm_info);
         printf("Date and time: %s\n", buffer);
      }

#if defined(HYPREDRV_DEVELOP_STRING) && defined(HYPREDRV_BRANCH_NAME)
      printf("\nUsing HYPREDRV_DEVELOP_STRING: %s (%s)\n", HYPREDRV_DEVELOP_STRING,
             HYPREDRV_BRANCH_NAME);
#elif defined(HYPREDRV_DEVELOP_STRING)
      printf("\nUsing HYPREDRV_DEVELOP_STRING: %s\n", HYPREDRV_DEVELOP_STRING);
#elif defined(HYPREDRV_GIT_SHA)
      printf("\nUsing HYPREDRV_GIT_SHA: %s\n", HYPREDRV_GIT_SHA);
#elif defined(HYPREDRV_RELEASE_VERSION)
      printf("\nUsing HYPREDRV_RELEASE_VERSION: %s\n", HYPREDRV_RELEASE_VERSION);
#endif

#if defined(HYPRE_DEVELOP_STRING) && defined(HYPRE_BRANCH_NAME)
      printf("Using HYPRE_DEVELOP_STRING: %s (%s)\n", HYPRE_DEVELOP_STRING,
             HYPRE_BRANCH_NAME);
#elif defined(HYPRE_DEVELOP_STRING) && !defined(HYPRE_BRANCH_NAME)
      printf("Using HYPRE_DEVELOP_STRING: %s\n", HYPRE_DEVELOP_STRING);
#elif defined(HYPRE_RELEASE_VERSION)
      printf("Using HYPRE_RELEASE_VERSION: %s\n", HYPRE_RELEASE_VERSION);
#endif
   }
}

/*--------------------------------------------------------------------------
 * PrintExitInfo
 *--------------------------------------------------------------------------*/

void
PrintExitInfo(MPI_Comm comm, const char *argv0)
{
   int myid = 0;

   MPI_Comm_rank(comm, &myid);

   if (!myid)
   {
      char             buffer[100];
      const struct tm *tm_info = NULL;
      time_t           t       = 0;

      /* Get current time */
      time(&t);
      tm_info = localtime(&t);

      /* Format and print the date and time */
      strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", tm_info);
      printf("Date and time: %s\n%s done!\n", buffer, argv0 ? argv0 : "Driver");
   }
}
