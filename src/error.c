/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC, HYPRE and GEOS
 * Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "error.h"

#define ERROR_CODE_NUM_ENTRIES 32

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
   int index = 1;

   while (code > 1)
   {
      code >>= 1;
      index++;
   }

   if (index > 0 && index < 32)
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
   int index = 1;

   while (code > 1)
   {
      code >>= 1;
      index++;
   }

   if (index > 0 && index < 32)
   {
      return global_error_count[index];
   }

   return -1;
}

/*-----------------------------------------------------------------------------
 * ErrorCodeSet
 *-----------------------------------------------------------------------------*/

void
ErrorCodeSet(ErrorCode code)
{
   global_error_code |= (uint32_t) code;
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
 * ErrorCodeDescribe
 *-----------------------------------------------------------------------------*/

void
ErrorCodeDescribe(void)
{
   if (global_error_code & ERROR_YAML_INVALID_INDENT)
   {
      ErrorMsgAddCodeWithCount(ERROR_YAML_INVALID_INDENT, "invalid indendation");
   }

   if (global_error_code & ERROR_YAML_INVALID_DIVISOR)
   {
      ErrorMsgAddCodeWithCount(ERROR_YAML_INVALID_DIVISOR, "invalid divisor");
   }

   if (global_error_code & ERROR_INVALID_KEY)
   {
      ErrorMsgAddCodeWithCount(ERROR_INVALID_KEY, "invalid key");
   }

   if (global_error_code & ERROR_INVALID_VAL)
   {
      ErrorMsgAddCodeWithCount(ERROR_INVALID_VAL, "invalid value");
   }

   if (global_error_code & ERROR_UNEXPECTED_VAL)
   {
      ErrorMsgAddCodeWithCount(ERROR_UNEXPECTED_VAL, "unexpected value");
   }

   if (global_error_code & ERROR_MAYBE_INVALID_VAL)
   {
      ErrorMsgAddCodeWithCount(ERROR_MAYBE_INVALID_VAL, "possible invalid value");
   }

   if (global_error_code & ERROR_MISSING_DOFMAP)
   {
      ErrorMsgAdd("Missing dofmap info needed by MGR!");
   }
}

/*-----------------------------------------------------------------------------
 * ErrorCodeReset
 *-----------------------------------------------------------------------------*/

void
ErrorCodeReset(uint32_t code)
{
   uint32_t i, bit;

   for (i = 1; i < ERROR_CODE_NUM_ENTRIES; i++)
   {
      bit = 1u << i;

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
ErrorMsgAdd(const char *message)
{
   ErrorMsgNode *new = (ErrorMsgNode *) malloc(sizeof(ErrorMsgNode));

   new->message = (char *) malloc(strlen(message) + 1);
   strcpy(new->message, message);
   new->next = global_error_msg_head;
   global_error_msg_head = new;
}

/*-----------------------------------------------------------------------------
 * ErrorMsgAddCodeWithCount
 *-----------------------------------------------------------------------------*/

void
ErrorMsgAddCodeWithCount(ErrorCode code, const char* suffix)
{
   char        *msg;
   const char  *plural = (ErrorCodeCountGet(code) > 1) ? "s" : "";
   int          length = strlen(suffix) + 24;

   msg = (char*) malloc(length);
   sprintf(msg, "Found %d %s%s!", ErrorCodeCountGet(code), suffix, plural);
   ErrorMsgAdd(msg);
   free(msg);
}

/*-----------------------------------------------------------------------------
 * ErrorMsgAddInvalidKeyValPair
 *-----------------------------------------------------------------------------*/

void
ErrorMsgAddInvalidKeyValPair(const char *key, const char* val)
{
   char *msg;
   int   length = strlen(key) + strlen(val) + 32;

   msg = (char*) malloc(length);
   sprintf(msg, "Invalid (key, val) pair: (%s, %s)", key, val);
   ErrorMsgAdd(msg);
   free(msg);
}

/*-----------------------------------------------------------------------------
 * ErrorMsgAddUnknownKey
 *-----------------------------------------------------------------------------*/

void
ErrorMsgAddUnknownKey(const char *key)
{
   char *msg;
   int   length = strlen(key) + 16;

   msg = (char*) malloc(length);
   sprintf(msg, "Unknown key: %s", key);
   ErrorMsgAdd(msg);
   free(msg);
}

/*-----------------------------------------------------------------------------
 * ErrorMsgAddUnknownVal
 *-----------------------------------------------------------------------------*/

void
ErrorMsgAddUnknownVal(const char *val)
{
   char *msg;
   int   length = strlen(val) + 16;

   msg = (char*) malloc(length);
   sprintf(msg, "Unknown val: %s", val);
   ErrorMsgAdd(msg);
   free(msg);
}

/*-----------------------------------------------------------------------------
 * ErrorMsgAddMissingKey
 *-----------------------------------------------------------------------------*/

void
ErrorMsgAddMissingKey(const char *key)
{
   char *msg;
   int   length = strlen(key) + 16;

   msg = (char*) malloc(length);
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
   char *msg;
   int   length = strlen(key) + 24;

   msg = (char*) malloc(length);
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
   char *msg;
   int   length = strlen(key) + 40;

   msg = (char*) malloc(length);
   sprintf(msg, "Unexpected value associated with %s key", key);
   ErrorMsgAdd(msg);
   free(msg);
}

/*-----------------------------------------------------------------------------
 * ErrorMsgAddInvalidSolverOption
 *-----------------------------------------------------------------------------*/

void
ErrorMsgAddInvalidSolverOption(int option)
{
   char  msg[64];

   sprintf(msg, "Invalid solver option: %d", option);
   ErrorMsgAdd(msg);
}

/*-----------------------------------------------------------------------------
 * ErrorMsgAddInvalidPreconOption
 *-----------------------------------------------------------------------------*/

void
ErrorMsgAddInvalidPreconOption(int option)
{
   char  msg[64];

   sprintf(msg, "Invalid preconditioner option: %d", option);
   ErrorMsgAdd(msg);
}

/*-----------------------------------------------------------------------------
 * ErrorMsgAddInvalidString
 *-----------------------------------------------------------------------------*/

void
ErrorMsgAddInvalidString(const char* string)
{
   char  msg[1024];

   sprintf(msg, "Invalid option: %s", string);
   ErrorMsgAdd(msg);
}

/*-----------------------------------------------------------------------------
 * ErrorMsgAddInvalidFilename
 *-----------------------------------------------------------------------------*/

void
ErrorMsgAddInvalidFilename(const char* string)
{
   char  msg[1024];

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

   printf("\n");
   while (current)
   {
      printf("--> %s\n", current->message);
      current = current->next;
   }
   printf("\n");
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
      current = current->next;
      free(temp->message);
      free(temp);
   }
   global_error_msg_head = NULL;
}

/*-----------------------------------------------------------------------------
 * ErrorMsgPrintAndAbort
 *-----------------------------------------------------------------------------*/

void
ErrorMsgPrintAndAbort(MPI_Comm comm)
{
   /* TODO: check error codes in other processes? */
   ErrorCodeDescribe();
   ErrorMsgPrint();
   ErrorMsgClear();
   MPI_Abort(comm, ErrorCodeGet());
}
