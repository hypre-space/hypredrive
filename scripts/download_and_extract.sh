#!/bin/bash
# /******************************************************************************
# * Copyright (c) 2024 Lawrence Livermore National Security, LLC
# * SPDX-License-Identifier: MIT
# ******************************************************************************/
#
# Download and extract a file from a URL to a destination directory
# Usage: download_and_extract.sh <URL> <TARBALL> <DEST> <MD5>
# Example: download_and_extract.sh https://zenodo.org/records/123456789/files/dataset.tar.gz dataset.tar.gz data 1234567890abcdef
#
# Arguments:
# - URL: the URL of the file to download
# - TARBALL: the name of the tarball to download
# - DEST: the destination directory to extract the tarball to
# - MD5: the MD5 checksum of the file

# Exit on error
set -e

# Arguments
URL="$1"
TARBALL="$2"
DEST="$3"
MD5="$4"

# Check if all arguments are provided
if [ -z "$URL" ] || [ -z "$TARBALL" ] || [ -z "$DEST" ] || [ -z "$MD5" ]; then
    echo "Usage: $0 <URL> <TARBALL> <DEST> <MD5>"
    exit 1
fi

# Remove quotes from URL and MD5 if present
URL=$(echo "$URL" | sed 's/^"//;s/"$//')
MD5=$(echo "$MD5" | sed 's/^"//;s/"$//')

# Create output directory
mkdir -p "$(dirname "$TARBALL")"
mkdir -p "$DEST"

# Download with curl
curl -L -f -o "$TARBALL" "$URL"

# Verify file exists
if [ ! -f "$TARBALL" ]; then
    echo "Downloaded file does not exist: $TARBALL"
    exit 1
fi

# Verify MD5 checksum
ACTUAL_MD5=$(md5sum "$TARBALL" | cut -d' ' -f1)
if [ "$ACTUAL_MD5" != "$MD5" ]; then
    echo "Checksum mismatch for $TARBALL: $ACTUAL_MD5 != $MD5"
    rm -f "$TARBALL"
    exit 1
fi

# Extract the tarball
tar -xzf "$TARBALL" -C "$DEST"
