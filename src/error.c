/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "internal/error.h"
#include <limits.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __linux__

#include <errno.h>
#include <execinfo.h>
#include <fcntl.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>

/* addr2line / fork / pipe paths are fault-heavy and not meaningfully testable in CI. */
/* GCOVR_EXCL_START */
static void
ErrorTrimNewline(char *text)
{
   if (!text)
   {
      return;
   }

   char *nl = strchr(text, '\n');
   if (nl)
   {
      *nl = '\0';
   }
}

static int
ErrorWaitChild(pid_t pid)
{
   int status = 0;

   while (waitpid(pid, &status, 0) < 0)
   {
      if (errno == EINTR)
      {
         continue;
      }
      return 0;
   }

   return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

static int
ErrorAddr2lineResolve(const char *exe_path, unsigned long offset, char *func,
                      size_t func_size, char *file, size_t file_size)
{
   int   pipefd[2] = {-1, -1};
   pid_t child_pid = -1;
   char  offset_arg[32];
   FILE *fp = NULL;

   if (!exe_path || exe_path[0] == '\0' || !func || !file || func_size == 0 ||
       file_size == 0)
   {
      return 0;
   }

   if (snprintf(offset_arg, sizeof(offset_arg), "0x%lx", offset) < 0)
   {
      return 0;
   }

   if (pipe(pipefd) != 0)
   {
      return 0;
   }

   child_pid = fork();
   if (child_pid < 0)
   {
      close(pipefd[0]);
      close(pipefd[1]);
      return 0;
   }

   if (child_pid == 0)
   {
      int devnull = -1;
      close(pipefd[0]);

      if (dup2(pipefd[1], STDOUT_FILENO) < 0)
      {
         _Exit(127);
      }

      devnull = open("/dev/null", O_WRONLY);
      if (devnull >= 0)
      {
         (void)dup2(devnull, STDERR_FILENO);
         close(devnull);
      }
      close(pipefd[1]);

      {
         char *const argv[] = {
            "addr2line", "-e", (char *)exe_path, "-f", "-C", "-i", offset_arg, NULL,
         };
         execvp("addr2line", argv);
      }
      _Exit(127);
   }

   close(pipefd[1]);
   fp = fdopen(pipefd[0], "r");
   if (!fp)
   {
      close(pipefd[0]);
      (void)ErrorWaitChild(child_pid);
      return 0;
   }

   {
      int func_cap = (func_size > (size_t)INT_MAX) ? INT_MAX : (int)func_size;
      int file_cap = (file_size > (size_t)INT_MAX) ? INT_MAX : (int)file_size;
      if (!fgets(func, func_cap, fp) || !fgets(file, file_cap, fp))
      {
         fclose(fp);
         (void)ErrorWaitChild(child_pid);
         return 0;
      }
   }

   ErrorTrimNewline(func);
   ErrorTrimNewline(file);
   fclose(fp);

   return ErrorWaitChild(child_pid);
}

/* GCOVR_EXCL_STOP */

/*-----------------------------------------------------------------------------
 * ErrorBacktraceGetBaseAddress
 *
 * For PIE (Position Independent Executable) binaries, we need to find the
 * base load address to convert runtime addresses to file offsets.
 *-----------------------------------------------------------------------------*/

/* GCOVR_EXCL_START */
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

/* GCOVR_EXCL_STOP */

/*-----------------------------------------------------------------------------
 * ErrorBacktraceSymbolsPrint
 *-----------------------------------------------------------------------------*/

/* GCOVR_EXCL_START */
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
            if (!ErrorAddr2lineResolve(exe_path, offset, func, sizeof(func), file,
                                       sizeof(file)))
            {
               func[0] = '\0';
               file[0] = '\0';
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

/* GCOVR_EXCL_STOP */

/*-----------------------------------------------------------------------------
 * hypredrv_ErrorBacktracePrint
 *-----------------------------------------------------------------------------*/

void
hypredrv_ErrorBacktracePrint(void)
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
      char pid_str[16];

      snprintf(attach_cmd, sizeof(attach_cmd), "attach %d", parent_pid);
      snprintf(pid_str, sizeof(pid_str), "%d", parent_pid);

      close(lock[1]);
      (void)!read(lock[0], lock, 1);
      close(lock[0]);

      /* Get executable path from /proc/<parent_pid>/exe (not self, since we forked) */
      char proc_exe_path[64];
      char resolved_exe[4096];
      snprintf(proc_exe_path, sizeof(proc_exe_path), "/proc/%d/exe", parent_pid);
      ssize_t len = readlink(proc_exe_path, resolved_exe, sizeof(resolved_exe) - 1);
      if (len > 0)
      {
         resolved_exe[len] = '\0';
      }
      else
      {
         resolved_exe[0] = '\0';
      }

      /* Try gdb - interactive mode with colored output
       * Order matters: load symbols first (file), then attach to process.
       * We continue the process briefly so SIGTRAP can be raised, then GDB catches it
       * giving the user an interactive session at the actual error location. */
      if (resolved_exe[0] != '\0')
      {
         execlp("gdb", "gdb", "-nx", resolved_exe, "-ex", "set style enabled on", "-ex",
                "set confirm off", "-ex", "set pagination off", "-ex", attach_cmd, "-ex",
                "continue", (char *)NULL);
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
   /* GCOVR_EXCL_BR_STOP */
}

#else

/*-----------------------------------------------------------------------------
 * hypredrv_ErrorBacktracePrint
 *-----------------------------------------------------------------------------*/

void
hypredrv_ErrorBacktracePrint(void)
{
   /* Backtrace not supported on this platform */
}

#endif /* __linux__ */

enum
{
   ERROR_CODE_NUM_ENTRIES = 32,
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
   bool          collective_report;
} ErrorState;

static ErrorState global_error_state = {0};

static ErrorState *
ErrorStateGet(void)
{
   return &global_error_state;
}

static int
ErrorCodeToIndex(hypredrv_error_t code)
{
   int index = 0;

   if (code == ERROR_NONE)
   {
      return -1; /* GCOVR_EXCL_LINE */
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
ErrorCodeCountIncrement(ErrorState *state, hypredrv_error_t code)
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
ErrorCodeCountGet(const ErrorState *state, hypredrv_error_t code)
{
   int index = ErrorCodeToIndex(code);
   return (index >= 0) ? state->code_count[index] : 0;
}

/* GCOVR_EXCL_START */
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

/* GCOVR_EXCL_STOP */

/*-----------------------------------------------------------------------------
 * hypredrv_ErrorCodeSet
 *-----------------------------------------------------------------------------*/

void
hypredrv_ErrorCodeSet(hypredrv_error_t code)
{
   ErrorState *state = ErrorStateGet();

   state->code |= (uint32_t)code;
   ErrorCodeCountIncrement(state, code);
}

/*-----------------------------------------------------------------------------
 * hypredrv_ErrorCodeGet
 *-----------------------------------------------------------------------------*/

uint32_t
hypredrv_ErrorCodeGet(void)
{
   return ErrorStateGet()->code;
}

/*-----------------------------------------------------------------------------
 * hypredrv_ErrorCodeActive
 *-----------------------------------------------------------------------------*/

bool
hypredrv_ErrorCodeActive(void)
{
   return (ErrorStateGet()->code != ERROR_NONE);
}

/*-----------------------------------------------------------------------------
 * hypredrv_ErrorReportSetCollective
 *-----------------------------------------------------------------------------*/

void
hypredrv_ErrorReportSetCollective(bool collective)
{
   ErrorStateGet()->collective_report = collective;
}

/*-----------------------------------------------------------------------------
 * hypredrv_ErrorReportIsCollective
 *-----------------------------------------------------------------------------*/

bool
hypredrv_ErrorReportIsCollective(void)
{
   return ErrorStateGet()->collective_report;
}

/*-----------------------------------------------------------------------------
 * hypredrv_SafeCallHandleError
 *-----------------------------------------------------------------------------*/

void
hypredrv_SafeCallHandleError(uint32_t error_code, MPI_Comm comm, const char *file,
                             int line, const char *func)
{
   /* GCOVR_EXCL_BR_START */
   if (error_code != 0)
   {
      bool collective_report = hypredrv_ErrorReportIsCollective();
      bool describe_error    = true;
      if (collective_report)
      {
         int myid = 0;
         MPI_Comm_rank(comm, &myid);
         describe_error = (myid == 0);
      }

      if (describe_error)
      {
         (void)fprintf(stderr, "At %s:%d in %s():\n", file, line, func);
         hypredrv_ErrorCodeDescribe(error_code);
         hypredrv_ErrorMsgPrint();
         hypredrv_ErrorMsgClear();
         hypredrv_ErrorBacktracePrint();
      }
      if (collective_report)
      {
         MPI_Barrier(comm);
      }
      const char *debug_env = getenv("HYPREDRV_DEBUG");
      if (debug_env && strcmp(debug_env, "1") == 0)
      {
         raise(SIGTRAP); /* Breakpoint for gdb */
      }
      else
      {
         MPI_Abort(comm, (int)error_code);
         exit((int)error_code);
      }
   }
   /* GCOVR_EXCL_STOP */
}

/*-----------------------------------------------------------------------------
 * hypredrv_DistributedErrorCodeActive
 *-----------------------------------------------------------------------------*/

bool
hypredrv_DistributedErrorCodeActive(MPI_Comm comm)
{
   uint32_t flag = 0;
   uint32_t code = ErrorStateGet()->code;

   MPI_Allreduce(&code, &flag, 1, MPI_UINT32_T, MPI_BOR, comm);

   return (flag != ERROR_NONE);
}

static char *
ErrorStateSerializeMessages(const ErrorState *state, uint64_t *serialized_size)
{
   const char **messages = NULL;
   size_t       count    = 0;
   size_t       total    = 1;
   char        *text     = NULL;
   size_t       pos      = 0;

   if (serialized_size)
   {
      *serialized_size = 0;
   }
   if (!state)
   {
      return NULL;
   }

   for (const ErrorMsgNode *node = state->msg_head; node; node = node->next)
   {
      total += strlen(node->message);
      count++;
   }
   if (count > 1)
   {
      total += count - 1;
   }
   if (count == 0)
   {
      return NULL;
   }

   messages = (const char **)calloc(count, sizeof(*messages));
   if (!messages)
   {
      return NULL;
   }

   {
      size_t idx = 0;
      for (const ErrorMsgNode *node = state->msg_head; node; node = node->next)
      {
         messages[idx++] = node->message;
      }
   }

   text = (char *)malloc(total);
   if (!text)
   {
      free(messages);
      return NULL;
   }

   for (size_t idx = count; idx > 0; idx--)
   {
      const char *message = messages[idx - 1];
      size_t      len     = strlen(message);
      memcpy(text + pos, message, len);
      pos += len;
      if (idx > 1)
      {
         text[pos++] = '\n';
      }
   }
   text[pos] = '\0';

   if (serialized_size)
   {
      *serialized_size = (uint64_t)(pos + 1);
   }

   free(messages);
   return text;
}

static void
ErrorStateApplySynchronized(ErrorState *state, uint32_t code, uint32_t source_code,
                            const char *serialized)
{
   char *text = NULL;
   char *save = NULL;
   char *line = NULL;

   if (!state)
   {
      return;
   }

   hypredrv_ErrorStateReset();

   for (uint32_t i = 0; i < ERROR_CODE_NUM_ENTRIES; i++)
   {
      uint32_t bit = 1u << i;
      if ((code & bit) != 0)
      {
         state->code |= bit;
         state->code_count[i] = 1;
      }
   }
   if (serialized && serialized[0] != '\0')
   {
      text = strdup(serialized);
      if (!text)
      {
         state->code |= (uint32_t)ERROR_ALLOCATION;
         ErrorCodeCountIncrement(state, ERROR_ALLOCATION);
      }
   }

   if (text)
   {
      for (line = strtok_r(text, "\n", &save); line; line = strtok_r(NULL, "\n", &save))
      {
         hypredrv_ErrorMsgAdd("%s", line);
      }
      free(text);
   }

   if (!state->msg_head && code != ERROR_NONE)
   {
      hypredrv_ErrorCodeDescribe(code);
   }
   else if ((code & ~source_code) != ERROR_NONE)
   {
      hypredrv_ErrorCodeDescribe(code & ~source_code);
   }
}

bool
hypredrv_DistributedErrorStateSync(MPI_Comm comm)
{
   ErrorState *state       = ErrorStateGet();
   uint32_t    local_code  = state->code;
   uint32_t    global_code = ERROR_NONE;
   uint32_t    root_code   = ERROR_NONE;
   int         myid        = 0;
   int         root        = INT_MAX;
   int         candidate   = INT_MAX;
   uint64_t    text_size   = 0;
   char       *text        = NULL;

   MPI_Comm_rank(comm, &myid);
   if (local_code != ERROR_NONE)
   {
      candidate = myid;
   }

   MPI_Allreduce(&local_code, &global_code, 1, MPI_UINT32_T, MPI_BOR, comm);
   if (global_code == ERROR_NONE)
   {
      return false;
   }

   MPI_Allreduce(&candidate, &root, 1, MPI_INT, MPI_MIN, comm);
   if (root == INT_MAX)
   {
      root = 0;
   }

   if (myid == root)
   {
      root_code = local_code;
   }
   MPI_Bcast(&root_code, 1, MPI_UINT32_T, root, comm);

   if (myid == root)
   {
      text = ErrorStateSerializeMessages(state, &text_size);
      if (text_size > (uint64_t)INT_MAX)
      {
         /* Message too long to fit in MPI count; discard to avoid overflow. */
         free(text);
         text      = NULL;
         text_size = 0;
      }
   }

   MPI_Bcast(&text_size, 1, MPI_UINT64_T, root, comm);
   if (text_size > 0)
   {
      size_t offset = 0;
      char   sink[256];

      if (myid != root)
      {
         text = (char *)malloc((size_t)text_size);
      }

      while (offset < (size_t)text_size)
      {
         size_t rem   = (size_t)text_size - offset;
         int    chunk = (int)((rem > sizeof(sink)) ? sizeof(sink) : rem);
         char  *buf   = sink;

         if (myid == root && text)
         {
            buf = text + offset;
         }
         else if (text)
         {
            buf = text + offset;
         }

         MPI_Bcast(buf, chunk, MPI_CHAR, root, comm);
         offset += (size_t)chunk;
      }
   }

   ErrorStateApplySynchronized(state, global_code, root_code, text);
   free(text);

   return true;
}

/*-----------------------------------------------------------------------------
 * hypredrv_ErrorCodeDescribe
 *-----------------------------------------------------------------------------*/

void
hypredrv_ErrorCodeDescribe(uint32_t code)
{
   if (code & ERROR_YAML_INVALID_INDENT)
   {
      hypredrv_ErrorMsgAddCodeWithCount(ERROR_YAML_INVALID_INDENT, "invalid indendation");
   }

   if (code & ERROR_YAML_INVALID_DIVISOR)
   {
      hypredrv_ErrorMsgAddCodeWithCount(ERROR_YAML_INVALID_DIVISOR, "invalid divisor");
   }

   if (code & ERROR_INVALID_KEY)
   {
      hypredrv_ErrorMsgAddCodeWithCount(ERROR_INVALID_KEY, "invalid key");
   }

   if (code & ERROR_INVALID_VAL)
   {
      hypredrv_ErrorMsgAddCodeWithCount(ERROR_INVALID_VAL, "invalid value");
   }

   if (code & ERROR_UNEXPECTED_VAL)
   {
      hypredrv_ErrorMsgAddCodeWithCount(ERROR_UNEXPECTED_VAL, "unexpected value");
   }

   if (code & ERROR_MAYBE_INVALID_VAL)
   {
      hypredrv_ErrorMsgAddCodeWithCount(ERROR_MAYBE_INVALID_VAL,
                                        "possibly invalid value");
   }

   if (code & ERROR_MISSING_DOFMAP)
   {
      hypredrv_ErrorMsgAdd("Missing dofmap info needed by MGR!");
   }

   if (code & ERROR_UNKNOWN_HYPREDRV_OBJ)
   {
      hypredrv_ErrorMsgAdd("HYPREDRV object is not set properly!!");
   }

   if (code & ERROR_HYPREDRV_NOT_INITIALIZED)
   {
      hypredrv_ErrorMsgAdd("HYPREDRV is not initialized!!");
   }

   if (code & ERROR_HYPRE_INTERNAL)
   {
      hypredrv_ErrorMsgAddCodeWithCount(ERROR_HYPRE_INTERNAL, "HYPRE internal error");
   }
}

/*-----------------------------------------------------------------------------
 * hypredrv_ErrorCodeReset
 *-----------------------------------------------------------------------------*/

void
hypredrv_ErrorCodeReset(uint32_t code)
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
 * hypredrv_ErrorCodeResetAll
 *-----------------------------------------------------------------------------*/

void
hypredrv_ErrorCodeResetAll(void)
{
   /* Clear *all* bits, including ERROR_UNKNOWN (0x80000000). */
   hypredrv_ErrorCodeReset(0xFFFFFFFFu);
   ErrorStateGet()->collective_report = false;
}

/*-----------------------------------------------------------------------------
 * hypredrv_ErrorStateReset
 *-----------------------------------------------------------------------------*/

void
hypredrv_ErrorStateReset(void)
{
   hypredrv_ErrorMsgClear();
   hypredrv_ErrorCodeResetAll();
}

/*-----------------------------------------------------------------------------
 * hypredrv_ErrorMsgAdd
 *-----------------------------------------------------------------------------*/

void
hypredrv_ErrorMsgAdd(const char *format, ...)
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

   /* GCOVR_EXCL_START */
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

   /* GCOVR_EXCL_STOP */

   alloc_size = sizeof(ErrorMsgNode) + (size_t)length + 1;
   node       = (ErrorMsgNode *)malloc(alloc_size);
   if (!node)
   {
      va_end(args);                       /* GCOVR_EXCL_LINE */
      ErrorStateRecordMessageDrop(state); /* GCOVR_EXCL_LINE */
      return;                             /* GCOVR_EXCL_LINE */
   }

   written = vsnprintf(node->message, (size_t)length + 1, fmt, args);
   va_end(args);

   if (written < 0)
   {
      snprintf(node->message, (size_t)length + 1, "%s",
               fallback_msg); /* GCOVR_EXCL_LINE */
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
      hypredrv_ErrorMsgAdd("%s",
                           "(failed to format error message)"); /* GCOVR_EXCL_LINE */
      return;                                                   /* GCOVR_EXCL_LINE */
   }

   hypredrv_ErrorMsgAdd("%s", msg);
}

/*-----------------------------------------------------------------------------
 * hypredrv_ErrorMsgAddCodeWithCount
 *-----------------------------------------------------------------------------*/

void
hypredrv_ErrorMsgAddCodeWithCount(hypredrv_error_t code, const char *suffix)
{
   uint32_t    count       = ErrorCodeCountGet(ErrorStateGet(), code);
   const char *safe_suffix = suffix ? suffix : "(null)";
   const char *plural      = (count > 1) ? "s" : "";

   hypredrv_ErrorMsgAdd("Found %d %s%s!", (int)count, safe_suffix, plural);
}

/*-----------------------------------------------------------------------------
 * hypredrv_ErrorMsgAddMissingKey
 *-----------------------------------------------------------------------------*/

void
hypredrv_ErrorMsgAddMissingKey(const char *key)
{
   ErrorMsgAddBoundedPieces("Missing key: ", key, "");
}

/*-----------------------------------------------------------------------------
 * hypredrv_ErrorMsgAddExtraKey
 *-----------------------------------------------------------------------------*/

void
hypredrv_ErrorMsgAddExtraKey(const char *key)
{
   ErrorMsgAddBoundedPieces("Extra (unused) key: ", key, "");
}

/*-----------------------------------------------------------------------------
 * hypredrv_ErrorMsgAddUnexpectedVal
 *-----------------------------------------------------------------------------*/

void
hypredrv_ErrorMsgAddUnexpectedVal(const char *key)
{
   ErrorMsgAddBoundedPieces("Unexpected value associated with ", key, " key");
}

/*-----------------------------------------------------------------------------
 * hypredrv_ErrorMsgAddInvalidFilename
 *-----------------------------------------------------------------------------*/

void
hypredrv_ErrorMsgAddInvalidFilename(const char *string)
{
   ErrorMsgAddBoundedPieces("Invalid filename: ", string, "");
}

/*-----------------------------------------------------------------------------
 * hypredrv_ErrorMsgPrint
 *-----------------------------------------------------------------------------*/

void
hypredrv_ErrorMsgPrint(void)
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

      /* GCOVR_EXCL_START */
      if (state->dropped_msg_count > 0)
      {
         const char *plural = (state->dropped_msg_count > 1) ? "s" : "";
         fprintf(stderr, "  --> Dropped %u error message%s due to allocation failure.\n",
                 state->dropped_msg_count, plural);
      }

      /* GCOVR_EXCL_STOP */

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
 * hypredrv_ErrorMsgClear
 *-----------------------------------------------------------------------------*/

void
hypredrv_ErrorMsgClear(void)
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
