/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC, HYPRE and GEOS
 * Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "error.h"

/* Struct for storing an error message in a linked list */
typedef struct ErrorMsgNode {
    char *message;
    struct ErrorMsgNode *next;
} ErrorMsgNode;

/* The head of the linked list of error messages */
static ErrorMsgNode *errorMsgHead = NULL;

/*-----------------------------------------------------------------------------
 * ErrorMsgAdd
 *-----------------------------------------------------------------------------*/

void
ErrorMsgAdd(const char *message)
{
   ErrorMsgNode *new = (ErrorMsgNode *) malloc(sizeof(ErrorMsgNode));
   new->message = (char *) malloc(strlen(message) + 1);
   strcpy(new->message, message);
   new->next = errorMsgHead;
   errorMsgHead = new;
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
   ErrorMsgNode *current = errorMsgHead;
   while (current)
   {
      printf("%s\n", current->message);
      current = current->next;
   }
}

/*-----------------------------------------------------------------------------
 * ErrorMsgClear
 *-----------------------------------------------------------------------------*/

void
ErrorMsgClear()
{
   ErrorMsgNode *current = errorMsgHead;
   while (current)
   {
      ErrorMsgNode *temp = current;
      current = current->next;
      free(temp->message);
      free(temp);
   }
   errorMsgHead = NULL;
}
