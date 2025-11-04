/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "error.h"

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
   while (code >>= 1)
   {
      index++;
   }
   if (index > 0 && index < ERROR_CODE_NUM_ENTRIES)
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
   while (code >>= 1)
   {
      index++;
   }
   return (index > 0 && index < ERROR_CODE_NUM_ENTRIES) ? global_error_count[index] : 0;
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
   for (uint32_t i = 1; i < ERROR_CODE_NUM_ENTRIES; i++)
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
   va_list args;
   int     length = 0;

   /* Determine the length of the formatted message */
   va_start(args, format);
   length = vsnprintf(NULL, 0, format, args);
   va_end(args);

   /* Format the message */
   new->message = (char *)malloc(length + 1);
   va_start(args, format);
   vsnprintf(new->message, length + 1, format, args);
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
   const char *plural = (ErrorCodeCountGet(code) > 1) ? "s" : "";
   int         length = strlen(suffix) + 24;

   msg = (char *)malloc(length);
   sprintf(msg, "Found %d %s%s!", (int)ErrorCodeCountGet(code), suffix, plural);
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
ErrorMsgPrint()
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
ErrorMsgClear()
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

/*-----------------------------------------------------------------------------
 * ErrorMsgPrintAndAbort
 *
 * TODO: remove MPI_Abort from internal HYPREDRV functions
 *-----------------------------------------------------------------------------*/

void
ErrorMsgPrintAndAbort(MPI_Comm comm)
{
   /* TODO: check error codes in other processes? */
   ErrorCodeDescribe(ErrorCodeGet());
   ErrorMsgPrint();
   ErrorMsgClear();
   MPI_Barrier(comm);
   MPI_Abort(comm, ErrorCodeGet());
}
