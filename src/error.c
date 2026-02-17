/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "error.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __linux__

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
      char    line[512];
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

      /* Find the line that corresponds to the executable */
      while (fgets(line, sizeof(line), fp))
      {
         /* Check if this line contains the executable path */
         if (exe_path[0] != '\0' && strstr(line, exe_path) != NULL)
         {
            /* Parse the start address (format: "start-end perms offset ...") */
            char *dash = strchr(line, '-');
            if (dash)
            {
               *dash = '\0';
               base  = strtoul(line, NULL, 16);
               break;
            }
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

   /* Get symbol strings from backtrace_symbols */
   char **symbols = backtrace_symbols(trace, nptrs);
   if (!symbols)
   {
      /* Fallback if backtrace_symbols fails */
      backtrace_symbols_fd(trace, nptrs, STDERR_FILENO);
      return;
   }

   /* Get base address for PIE executables (used as fallback if offset parsing fails) */
   unsigned long base_addr = ErrorBacktraceGetBaseAddress();

   /* Try to use addr2line for each address */
   if (exe_path[0] != '\0' && nptrs > 0)
   {
      int saw_output = 0;
      for (int i = 0; i < nptrs; i++)
      {
         char          func[256] = "";
         char          file[256] = "";
         unsigned long offset    = 0;

         /* Parse offset from symbol string (format: "executable(+0xoffset)" or
          * "executable[+0xoffset]") */
         if (symbols[i])
         {
            const char *plus = strstr(symbols[i], "(+0x");
            if (!plus)
            {
               plus = strstr(symbols[i], "[+0x");
            }
            if (plus)
            {
               offset = strtoul(plus + 4, NULL, 16);
            }
         }

         /* Fallback: if we couldn't parse offset from symbol string, compute it from base
          * address */
         if (offset == 0 && base_addr > 0)
         {
            unsigned long addr = (unsigned long)trace[i];
            if (addr >= base_addr)
            {
               offset = addr - base_addr;
            }
         }

         /* If we found an offset, use addr2line with it */
         if (offset > 0)
         {
            char cmd[4352];
            /* Use absolute path and redirect stderr to avoid noise */
            /* Note: -p outputs on one line, so we use separate -f and -l flags */
            snprintf(cmd, sizeof(cmd), "addr2line -e %s -f -C -i 0x%lx 2>/dev/null",
                     exe_path, offset);
            FILE *fp = popen(cmd, "r");
            if (fp)
            {
               /* Read function name (first line) */
               if (fgets(func, sizeof(func), fp))
               {
                  /* Remove newline */
                  char *nl = strchr(func, '\n');
                  if (nl) *nl = '\0';
               }
               /* Read file:line (second line) */
               if (fgets(file, sizeof(file), fp))
               {
                  /* Remove newline */
                  char *nl = strchr(file, '\n');
                  if (nl) *nl = '\0';
               }
               int status = pclose(fp);
               /* If addr2line failed (non-zero exit), clear the output */
               if (status != 0)
               {
                  func[0] = '\0';
                  file[0] = '\0';
               }
            }
         }

         /* Print if we got valid output from addr2line */
         if (func[0] != '\0' && file[0] != '\0' && strcmp(func, "??") != 0 &&
             strcmp(file, "??:0") != 0)
         {
            saw_output = 1;
            fprintf(stderr, "#%d %s at %s\n", i, func, file);
         }
         /* For addresses in our executable that addr2line couldn't resolve, show the
          * symbol string */
         else if (symbols[i] && strstr(symbols[i], "hypredrive") != NULL)
         {
            saw_output = 1;
            fprintf(stderr, "#%d %s\n", i, symbols[i]);
         }
      }
      if (saw_output)
      {
         free(symbols);
         return;
      }
   }

   free(symbols);
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

   /* GCOVR/LCOV exclusion: the interactive debugger attach path is inherently
    * environment-dependent (procfs, fork/exec, gdb/lldb availability) and not
    * meaningfully testable in CI/unit tests without hanging or spawning debuggers. */
   /* GCOVR_EXCL_START */
   /* LCOV_EXCL_START */
   /* Interactive debug mode (HYPREDRV_DEBUG=1) */
   /* Check if already being debugged */
   FILE   *f      = fopen("/proc/self/status", "r");
   size_t  size   = 0;
   char   *line   = NULL;
   ssize_t length = 0;
   if (!f)
   {
      return;
   }
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
      ssize_t len = readlink(exe_path, exe_link, sizeof(exe_link) - 1);
      if (len > 0)
      {
         exe_path[len] = '\0';
         /* Build "file <path>" as a single command string */
         snprintf(file_cmd, sizeof(file_cmd), "file %s", exe_path);
      }
      else
      {
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
      // cppcheck-suppress unreachableCode
      waitpid(child_pid, NULL, 0);
   }
   /* LCOV_EXCL_STOP */
   /* GCOVR_EXCL_STOP */
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

/*-----------------------------------------------------------------------------
 * Error state container and helpers
 *-----------------------------------------------------------------------------*/

typedef struct ErrorMsgNode
{
   struct ErrorMsgNode *next;
   char                 message[];
} ErrorMsgNode;

typedef struct ErrorState
{
   uint32_t      code;
   uint32_t      code_count[ERROR_CODE_NUM_ENTRIES];
   ErrorMsgNode *msg_head;
   uint32_t      dropped_msg_count;
} ErrorState;

static ErrorState global_error_state = {0};

static ErrorState *
ErrorStateGet(void)
{
   return &global_error_state;
}

static int
ErrorCodeToIndex(ErrorCode code)
{
   int index = 0;

   if (code == ERROR_NONE)
   {
      return -1;
   }

   while (code >>= 1)
   {
      index++;
   }

   return (index < ERROR_CODE_NUM_ENTRIES) ? index : -1;
}

/*-----------------------------------------------------------------------------
 * ErrorCodeCountIncrement
 *-----------------------------------------------------------------------------*/

static void
ErrorCodeCountIncrement(ErrorState *state, ErrorCode code)
{
   int index = ErrorCodeToIndex(code);
   if (index >= 0)
   {
      state->code_count[index]++;
   }
}

/*-----------------------------------------------------------------------------
 * ErrorCodeCountGet
 *-----------------------------------------------------------------------------*/

static uint32_t
ErrorCodeCountGet(const ErrorState *state, ErrorCode code)
{
   int index = ErrorCodeToIndex(code);
   return (index >= 0) ? state->code_count[index] : 0;
}

static void
ErrorStateRecordMessageDrop(ErrorState *state)
{
   if (!state)
   {
      return;
   }

   state->code |= (uint32_t)ERROR_ALLOCATION;
   ErrorCodeCountIncrement(state, ERROR_ALLOCATION);
   state->dropped_msg_count++;
}

/*-----------------------------------------------------------------------------
 * ErrorCodeSet
 *-----------------------------------------------------------------------------*/

void
ErrorCodeSet(ErrorCode code)
{
   ErrorState *state = ErrorStateGet();

   state->code |= (uint32_t)code;
   ErrorCodeCountIncrement(state, code);
}

/*-----------------------------------------------------------------------------
 * ErrorCodeGet
 *-----------------------------------------------------------------------------*/

uint32_t
ErrorCodeGet(void)
{
   return ErrorStateGet()->code;
}

/*-----------------------------------------------------------------------------
 * ErrorCodeActive
 *-----------------------------------------------------------------------------*/

bool
ErrorCodeActive(void)
{
   return (ErrorStateGet()->code != ERROR_NONE);
}

/*-----------------------------------------------------------------------------
 * DistributedErrorCodeActive
 *-----------------------------------------------------------------------------*/

bool
DistributedErrorCodeActive(MPI_Comm comm)
{
   uint32_t flag = 0;
   uint32_t code = ErrorStateGet()->code;

   MPI_Allreduce(&code, &flag, 1, MPI_UINT32_T, MPI_BOR, comm);

   return (flag != ERROR_NONE);
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

   if (code & ERROR_HYPRE_INTERNAL)
   {
      ErrorMsgAddCodeWithCount(ERROR_HYPRE_INTERNAL, "HYPRE internal error");
   }
}

/*-----------------------------------------------------------------------------
 * ErrorCodeReset
 *-----------------------------------------------------------------------------*/

void
ErrorCodeReset(uint32_t code)
{
   ErrorState *state = ErrorStateGet();

   for (uint32_t i = 0; i < ERROR_CODE_NUM_ENTRIES; i++)
   {
      uint32_t bit = 1u << i;

      if ((bit & code) != 0)
      {
         state->code &= ~bit; /* Set n-th bit to zero */
         state->code_count[i] = 0;
      }
   }
}

/*-----------------------------------------------------------------------------
 * ErrorCodeResetAll
 *-----------------------------------------------------------------------------*/

void
ErrorCodeResetAll(void)
{
   /* Clear *all* bits, including ERROR_UNKNOWN (0x80000000). */
   ErrorCodeReset(0xFFFFFFFFu);
}

/*-----------------------------------------------------------------------------
 * ErrorStateReset
 *-----------------------------------------------------------------------------*/

void
ErrorStateReset(void)
{
   ErrorMsgClear();
   ErrorCodeResetAll();
}

/*-----------------------------------------------------------------------------
 * ErrorMsgAdd
 *-----------------------------------------------------------------------------*/

void
ErrorMsgAdd(const char *format, ...)
{
   ErrorState   *state        = ErrorStateGet();
   ErrorMsgNode *node         = NULL;
   const char   *fmt          = format ? format : "(null format)";
   const char   *fallback_msg = "(error formatting message)";
   va_list       args;
   va_list       args_copy;
   int           length     = 0;
   int           written    = 0;
   size_t        alloc_size = 0;

   va_start(args, format);
   va_copy(args_copy, args);
   length = vsnprintf(NULL, 0, fmt, args_copy);
   va_end(args_copy);

   if (length < 0)
   {
      va_end(args);
      length = (int)strlen(fallback_msg);

      node = (ErrorMsgNode *)malloc(sizeof(ErrorMsgNode) + (size_t)length + 1);
      if (!node)
      {
         ErrorStateRecordMessageDrop(state);
         return;
      }

      memcpy(node->message, fallback_msg, (size_t)length + 1);
      node->next      = state->msg_head;
      state->msg_head = node;
      return;
   }

   alloc_size = sizeof(ErrorMsgNode) + (size_t)length + 1;
   node       = (ErrorMsgNode *)malloc(alloc_size);
   if (!node)
   {
      va_end(args);
      ErrorStateRecordMessageDrop(state);
      return;
   }

   written = vsnprintf(node->message, (size_t)length + 1, fmt, args);
   va_end(args);

   if (written < 0)
   {
      snprintf(node->message, (size_t)length + 1, "%s", fallback_msg);
   }

   node->next      = state->msg_head;
   state->msg_head = node;
}

static void
ErrorMsgAddBoundedPieces(const char *prefix, const char *value, const char *suffix)
{
   char        msg[1024];
   const char *safe_prefix = prefix ? prefix : "";
   const char *safe_value  = value ? value : "(null)";
   const char *safe_suffix = suffix ? suffix : "";

   if (snprintf(msg, sizeof(msg), "%s%s%s", safe_prefix, safe_value, safe_suffix) < 0)
   {
      ErrorMsgAdd("%s", "(failed to format error message)");
      return;
   }

   ErrorMsgAdd("%s", msg);
}

/*-----------------------------------------------------------------------------
 * ErrorMsgAddCodeWithCount
 *-----------------------------------------------------------------------------*/

void
ErrorMsgAddCodeWithCount(ErrorCode code, const char *suffix)
{
   uint32_t    count       = ErrorCodeCountGet(ErrorStateGet(), code);
   const char *safe_suffix = suffix ? suffix : "(null)";
   const char *plural      = (count > 1) ? "s" : "";

   ErrorMsgAdd("Found %d %s%s!", (int)count, safe_suffix, plural);
}

/*-----------------------------------------------------------------------------
 * ErrorMsgAddMissingKey
 *-----------------------------------------------------------------------------*/

void
ErrorMsgAddMissingKey(const char *key)
{
   ErrorMsgAddBoundedPieces("Missing key: ", key, "");
}

/*-----------------------------------------------------------------------------
 * ErrorMsgAddExtraKey
 *-----------------------------------------------------------------------------*/

void
ErrorMsgAddExtraKey(const char *key)
{
   ErrorMsgAddBoundedPieces("Extra (unused) key: ", key, "");
}

/*-----------------------------------------------------------------------------
 * ErrorMsgAddUnexpectedVal
 *-----------------------------------------------------------------------------*/

void
ErrorMsgAddUnexpectedVal(const char *key)
{
   ErrorMsgAddBoundedPieces("Unexpected value associated with ", key, " key");
}

/*-----------------------------------------------------------------------------
 * ErrorMsgAddInvalidFilename
 *-----------------------------------------------------------------------------*/

void
ErrorMsgAddInvalidFilename(const char *string)
{
   ErrorMsgAddBoundedPieces("Invalid filename: ", string, "");
}

/*-----------------------------------------------------------------------------
 * ErrorMsgPrint
 *-----------------------------------------------------------------------------*/

void
ErrorMsgPrint(void)
{
   ErrorState   *state   = ErrorStateGet();
   ErrorMsgNode *current = state->msg_head;

   fprintf(stderr, "====================================================================="
                   "===============\n");
   fprintf(stderr, "                                HYPREDRIVE Failure!!!\n");
   fprintf(stderr, "====================================================================="
                   "===============\n");

   if (current || state->dropped_msg_count > 0)
   {
      fprintf(stderr, "\nError details:\n");

      if (state->dropped_msg_count > 0)
      {
         const char *plural = (state->dropped_msg_count > 1) ? "s" : "";
         fprintf(stderr, "  --> Dropped %u error message%s due to allocation failure.\n",
                 state->dropped_msg_count, plural);
      }

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
   ErrorState   *state   = ErrorStateGet();
   ErrorMsgNode *current = state->msg_head;

   while (current)
   {
      ErrorMsgNode *temp = current;
      current            = current->next;
      free(temp);
   }
   state->msg_head          = NULL;
   state->dropped_msg_count = 0;
}
