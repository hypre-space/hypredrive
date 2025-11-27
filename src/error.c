/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "error.h"

#if defined(__linux__)

#include <execinfo.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>

/*-----------------------------------------------------------------------------
 * ErrorBacktraceGetBaseAddress
 *
 * For PIE (Position Independent Executable) binaries, we need to find the
 * base load address to convert runtime addresses to file offsets.
 *-----------------------------------------------------------------------------*/

static unsigned long
ErrorBacktraceGetBaseAddress(void)
{
   unsigned long base = 0;
   FILE         *fp   = fopen("/proc/self/maps", "r");
   if (fp)
   {
      char line[512];
      /* The first line typically contains the base address of the executable */
      if (fgets(line, sizeof(line), fp))
      {
         /* Parse the start address (format: "start-end perms offset ...") */
         char *dash = strchr(line, '-');
         if (dash)
         {
            *dash = '\0';
            base  = strtoul(line, NULL, 16);
         }
      }
      fclose(fp);
   }
   return base;
}

/*-----------------------------------------------------------------------------
 * ErrorBacktraceSymbolsPrint
 *-----------------------------------------------------------------------------*/

static void
ErrorBacktraceSymbolsPrint(void)
{
   void *trace[100];
   int   nptrs = backtrace(trace, sizeof(trace) / sizeof(trace[0]));

   /* Get executable path */
   char    exe_path[4096];
   ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
   if (len > 0)
   {
      exe_path[len] = '\0';
   }
   else
   {
      exe_path[0] = '\0';
   }

   /* Get base address for PIE executables */
   unsigned long base_addr = ErrorBacktraceGetBaseAddress();

   /* Try to use addr2line for each address individually */
   if (exe_path[0] != '\0' && nptrs > 0)
   {
      int saw_output = 0;
      for (int i = 0; i < nptrs; i++)
      {
         /* Compute offset from base for PIE binaries */
         unsigned long addr = (unsigned long)trace[i];
         unsigned long offset =
            (base_addr > 0 && addr > base_addr) ? (addr - base_addr) : addr;

         char cmd[4352];
         snprintf(cmd, sizeof(cmd), "addr2line -e %s -f -C -i 0x%lx 2>/dev/null",
                  exe_path, offset);
         FILE *fp = popen(cmd, "r");
         if (fp)
         {
            char func[256] = "";
            char file[256] = "";
            if (fgets(func, sizeof(func), fp))
            {
               /* Remove newline */
               char *nl = strchr(func, '\n');
               if (nl) *nl = '\0';
            }
            if (fgets(file, sizeof(file), fp))
            {
               /* Remove newline */
               char *nl = strchr(file, '\n');
               if (nl) *nl = '\0';
            }
            pclose(fp);

            if (func[0] != '\0' && file[0] != '\0' && strcmp(func, "??") != 0 &&
                strcmp(file, "??:0") != 0)
            {
               saw_output = 1;
               fprintf(stderr, "#%d %s at %s\n", i, func, file);
            }
         }
      }
      if (saw_output)
      {
         return;
      }
   }

   /* Fallback to backtrace_symbols_fd */
   backtrace_symbols_fd(trace, nptrs, STDERR_FILENO);
}

/*-----------------------------------------------------------------------------
 * ErrorBacktracePrint
 *-----------------------------------------------------------------------------*/

void
ErrorBacktracePrint(void)
{
   const char *HYPREDRV_NO_BACKTRACE = getenv("HYPREDRV_NO_BACKTRACE");
   if (HYPREDRV_NO_BACKTRACE)
   {
      return;
   }

   fprintf(stderr, "\nBacktrace:\n");

   const char *HYPREDRV_DEBUG = getenv("HYPREDRV_DEBUG");
   int         debug_mode =
      (HYPREDRV_DEBUG && HYPREDRV_DEBUG[0] == '1' && HYPREDRV_DEBUG[1] == '\0');

   if (!debug_mode)
   {
      /* Non-interactive mode: use backtrace_symbols for immediate output */
      ErrorBacktraceSymbolsPrint();
      return;
   }

   /* Interactive debug mode (HYPREDRV_DEBUG=1) */
   /* Check if already being debugged */
   FILE   *f      = fopen("/proc/self/status", "r");
   size_t  size   = 0;
   char   *line   = NULL;
   ssize_t length = 0;
   while ((length = getline(&line, &size, f)) > 0)
   {
      if (!strncmp(line, "TracerPid:", sizeof("TracerPid:") - 1) &&
          (length != sizeof("TracerPid:\t0\n") - 1 || line[length - 2] != '0'))
      {
         /* Already being debugged, and the breakpoint is the later abort() */
         free(line);
         fclose(f);
         return;
      }
   }
   free(line);
   fclose(f);

   int lock[2] = {-1, -1};
   (void)!pipe(lock); /* Don't start gdb until after PR_SET_PTRACER */

   const int parent_pid = getpid();
   const int child_pid  = fork();
   if (child_pid < 0)
   {
      /* Fork error */
      close(lock[1]);
      close(lock[0]);
      return;
   }
   else if (child_pid == 0)
   {
      /* Child process */
      char attach_cmd[64];
      char file_cmd[4200];
      char pid_str[16];

      snprintf(attach_cmd, sizeof(attach_cmd), "attach %d", parent_pid);
      snprintf(pid_str, sizeof(pid_str), "%d", parent_pid);

      close(lock[1]);
      (void)!read(lock[0], lock, 1);
      close(lock[0]);

      /* Get executable path from /proc/<parent_pid>/exe (not self, since we forked) */
      char exe_link[64];
      char exe_path[4096];
      snprintf(exe_link, sizeof(exe_link), "/proc/%d/exe", parent_pid);
      ssize_t len = readlink(exe_link, exe_path, sizeof(exe_path) - 1);
      if (len > 0)
      {
         exe_path[len] = '\0';
         /* Build "file <path>" as a single command string */
         snprintf(file_cmd, sizeof(file_cmd), "file %s", exe_path);
      }
      else
      {
         exe_path[0] = '\0';
         file_cmd[0] = '\0';
      }

      /* Try gdb - interactive mode with colored output
       * Order matters: load symbols first (file), then attach to process.
       * We continue the process briefly so SIGTRAP can be raised, then GDB catches it
       * giving the user an interactive session at the actual error location. */
      if (file_cmd[0] != '\0')
      {
         execlp("gdb", "gdb", "-nx", "-ex", "set style enabled on", "-ex",
                "set confirm off", "-ex", "set pagination off", "-ex", file_cmd, "-ex",
                attach_cmd, "-ex", "continue", (char *)NULL);
      }
      else
      {
         execlp("gdb", "gdb", "-nx", "-ex", "set style enabled on", "-ex",
                "set confirm off", "-ex", "set pagination off", "-ex", attach_cmd, "-ex",
                "continue", (char *)NULL);
      }
      /* Try lldb */
      execlp("lldb", "lldb", "-o", "bt", "-o", "quit", "-p", pid_str, (char *)NULL);
      /* gdb/lldb failed, fallback to backtrace_symbols */
      ErrorBacktraceSymbolsPrint();
      _Exit(0);
   }
   else
   {
      /* Parent process */
      prctl(PR_SET_PTRACER, child_pid);
      close(lock[1]);
      close(lock[0]);

      /* Give GDB time to attach before we do anything */
      usleep(100000); /* 100ms */

      /* Raise SIGTRAP so GDB catches us at a useful point.
       * This will stop the process and give the user an interactive GDB session
       * with the full stack visible. */
      raise(SIGTRAP);

      /* If GDB detaches or user continues, wait for the child to finish */
      waitpid(child_pid, NULL, 0);
   }
}

#else

/*-----------------------------------------------------------------------------
 * ErrorBacktracePrint
 *-----------------------------------------------------------------------------*/

void
ErrorBacktracePrint(void)
{
   /* Backtrace not supported on this platform */
}

#endif /* __linux__ */

enum
{
   ERROR_CODE_NUM_ENTRIES = 32
};

/* Struct for storing an error message in a linked list */
typedef struct ErrorMsgNode
{
   char                *message;
   struct ErrorMsgNode *next;
} ErrorMsgNode;

/* The head of the linked list of error messages */
static ErrorMsgNode *global_error_msg_head = NULL;

/* Global error code variable */
static uint32_t global_error_code;
static uint32_t global_error_count[ERROR_CODE_NUM_ENTRIES] = {0};

/*-----------------------------------------------------------------------------
 * ErrorCodeCountIncrement
 *-----------------------------------------------------------------------------*/

void
ErrorCodeCountIncrement(ErrorCode code)
{
   int index = 0;

   if (code == ERROR_NONE)
   {
      return;
   }

   while (code >>= 1)
   {
      index++;
   }
   if (index < ERROR_CODE_NUM_ENTRIES)
   {
      global_error_count[index]++;
   }
}

/*-----------------------------------------------------------------------------
 * ErrorCodeCountGet
 *-----------------------------------------------------------------------------*/

uint32_t
ErrorCodeCountGet(ErrorCode code)
{
   int index = 0;

   if (code == ERROR_NONE)
   {
      return 0;
   }

   while (code >>= 1)
   {
      index++;
   }
   return (index < ERROR_CODE_NUM_ENTRIES) ? global_error_count[index] : 0;
}

/*-----------------------------------------------------------------------------
 * ErrorCodeSet
 *-----------------------------------------------------------------------------*/

void
ErrorCodeSet(ErrorCode code)
{
   global_error_code |= (uint32_t)code;
   ErrorCodeCountIncrement(code);
}

/*-----------------------------------------------------------------------------
 * ErrorCodeGet
 *-----------------------------------------------------------------------------*/

uint32_t
ErrorCodeGet(void)
{
   return global_error_code;
}

/*-----------------------------------------------------------------------------
 * ErrorCodeActive
 *-----------------------------------------------------------------------------*/

bool
ErrorCodeActive(void)
{
   return (global_error_code == ERROR_NONE) ? false : true;
}

/*-----------------------------------------------------------------------------
 * DistributedErrorCodeActive
 *-----------------------------------------------------------------------------*/

bool
DistributedErrorCodeActive(MPI_Comm comm)
{
   uint32_t flag = 0;

   MPI_Allreduce(&global_error_code, &flag, 1, MPI_UINT32_T, MPI_BOR, comm);

   return (flag == ERROR_NONE) ? false : true;
}

/*-----------------------------------------------------------------------------
 * ErrorCodeDescribe
 *-----------------------------------------------------------------------------*/

void
ErrorCodeDescribe(uint32_t code)
{
   if (code & ERROR_YAML_INVALID_INDENT)
   {
      ErrorMsgAddCodeWithCount(ERROR_YAML_INVALID_INDENT, "invalid indendation");
   }

   if (code & ERROR_YAML_INVALID_DIVISOR)
   {
      ErrorMsgAddCodeWithCount(ERROR_YAML_INVALID_DIVISOR, "invalid divisor");
   }

   if (code & ERROR_INVALID_KEY)
   {
      ErrorMsgAddCodeWithCount(ERROR_INVALID_KEY, "invalid key");
   }

   if (code & ERROR_INVALID_VAL)
   {
      ErrorMsgAddCodeWithCount(ERROR_INVALID_VAL, "invalid value");
   }

   if (code & ERROR_UNEXPECTED_VAL)
   {
      ErrorMsgAddCodeWithCount(ERROR_UNEXPECTED_VAL, "unexpected value");
   }

   if (code & ERROR_MAYBE_INVALID_VAL)
   {
      ErrorMsgAddCodeWithCount(ERROR_MAYBE_INVALID_VAL, "possibly invalid value");
   }

   if (code & ERROR_MISSING_DOFMAP)
   {
      ErrorMsgAdd("Missing dofmap info needed by MGR!");
   }

   if (code & ERROR_UNKNOWN_HYPREDRV_OBJ)
   {
      ErrorMsgAdd("HYPREDRV object is not set properly!!");
   }

   if (code & ERROR_HYPREDRV_NOT_INITIALIZED)
   {
      ErrorMsgAdd("HYPREDRV is not initialized!!");
   }
}

/*-----------------------------------------------------------------------------
 * ErrorCodeReset
 *-----------------------------------------------------------------------------*/

void
ErrorCodeReset(uint32_t code)
{
   for (uint32_t i = 0; i < ERROR_CODE_NUM_ENTRIES; i++)
   {
      uint32_t bit = 1u << i;

      if ((bit & code) != 0)
      {
         global_error_code &= ~bit; /* Set n-th bit to zero */
         global_error_count[i] = 0; /* Reset counter */
      }
   }
}

/*-----------------------------------------------------------------------------
 * ErrorCodeResetAll
 *-----------------------------------------------------------------------------*/

void
ErrorCodeResetAll(void)
{
   ErrorCodeReset(0x7FFFFFFFu);
}

/*-----------------------------------------------------------------------------
 * ErrorMsgAdd
 *-----------------------------------------------------------------------------*/

void
ErrorMsgAdd(const char *format, ...)
{
   ErrorMsgNode *new = (ErrorMsgNode *)malloc(sizeof(ErrorMsgNode));
   va_list     args;
   int         length = 0;
   const char *fmt    = format;

   /* Ensure format is not NULL */
   if (fmt == NULL)
   {
      fmt = "(null format)";
   }

   /* Determine the length of the formatted message */
   va_start(args, format);
   length = vsnprintf(NULL, 0, fmt, args);
   va_end(args);

   /* Format the message */
   new->message = (char *)malloc(length + 1);
   va_start(args, format);
   vsnprintf(new->message, length + 1, fmt, args);
   va_end(args);

   /* Insert the new node at the head of the list */
   new->next             = global_error_msg_head;
   global_error_msg_head = new;
}

/*-----------------------------------------------------------------------------
 * ErrorMsgAddCodeWithCount
 *-----------------------------------------------------------------------------*/

void
ErrorMsgAddCodeWithCount(ErrorCode code, const char *suffix)
{
   char       *msg    = NULL;
   uint32_t    count  = ErrorCodeCountGet(code);
   const char *plural = (count > 1) ? "s" : "";
   int         length = strlen(suffix) + 24;

   msg = (char *)malloc(length);
   sprintf(msg, "Found %d %s%s!", (int)count, suffix, plural);
   ErrorMsgAdd(msg);
   free(msg);
}

/*-----------------------------------------------------------------------------
 * ErrorMsgAddMissingKey
 *-----------------------------------------------------------------------------*/

void
ErrorMsgAddMissingKey(const char *key)
{
   char *msg    = NULL;
   int   length = strlen(key) + 16;

   msg = (char *)malloc(length);
   sprintf(msg, "Missing key: %s", key);
   ErrorMsgAdd(msg);
   free(msg);
}

/*-----------------------------------------------------------------------------
 * ErrorMsgAddExtraKey
 *-----------------------------------------------------------------------------*/

void
ErrorMsgAddExtraKey(const char *key)
{
   char *msg    = NULL;
   int   length = strlen(key) + 24;

   msg = (char *)malloc(length);
   sprintf(msg, "Extra (unused) key: %s", key);
   ErrorMsgAdd(msg);
   free(msg);
}

/*-----------------------------------------------------------------------------
 * ErrorMsgAddUnexpectedVal
 *-----------------------------------------------------------------------------*/

void
ErrorMsgAddUnexpectedVal(const char *key)
{
   char *msg    = NULL;
   int   length = strlen(key) + 40;

   msg = (char *)malloc(length);
   sprintf(msg, "Unexpected value associated with %s key", key);
   ErrorMsgAdd(msg);
   free(msg);
}

/*-----------------------------------------------------------------------------
 * ErrorMsgAddInvalidFilename
 *-----------------------------------------------------------------------------*/

void
ErrorMsgAddInvalidFilename(const char *string)
{
   char msg[1024];

   sprintf(msg, "Invalid filename: %s", string);
   ErrorMsgAdd(msg);
}

/*-----------------------------------------------------------------------------
 * ErrorMsgPrint
 *-----------------------------------------------------------------------------*/

void
ErrorMsgPrint(void)
{
   ErrorMsgNode *current = global_error_msg_head;

   fprintf(stderr, "====================================================================="
                   "===============\n");
   fprintf(stderr, "                                HYPREDRIVE Failure!!!\n");
   fprintf(stderr, "====================================================================="
                   "===============\n");

   if (current)
   {
      fprintf(stderr, "\nError details:\n");
      while (current)
      {
         fprintf(stderr, "  --> %s\n", current->message);
         current = current->next;
      }
      fprintf(stderr, "\n");
      fprintf(stderr, "=================================================================="
                      "==================\n\n");
   }
}

/*-----------------------------------------------------------------------------
 * ErrorMsgClear
 *-----------------------------------------------------------------------------*/

void
ErrorMsgClear(void)
{
   ErrorMsgNode *current = global_error_msg_head;

   while (current)
   {
      ErrorMsgNode *temp = current;
      current            = current->next;
      free(temp->message);
      free(temp);
   }
   global_error_msg_head = NULL;
}
