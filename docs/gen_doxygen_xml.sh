#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/.." && pwd)
tmp_doxyfile="${TMPDIR:-/tmp}/hypredrive-doxygen-rtd.conf"

version=$(
  sed -n 's/^project(hypredrive VERSION \([^ )]*\).*/\1/p' "${repo_root}/CMakeLists.txt"
)

if [[ -z "${version}" ]]; then
  echo "Failed to determine hypredrive version from CMakeLists.txt" >&2
  exit 1
fi

sed \
  -e "s|@PACKAGE_NAME@|hypredrive|g" \
  -e "s|@VERSION@|${version}|g" \
  -e "s|@OUTPUT_DIRECTORY@|docs|g" \
  -e "s|@GENERATE_HTML@|NO|g" \
  -e "s|@GENERATE_HTMLHELP@|NO|g" \
  -e "s|@GENERATE_CHI@|NO|g" \
  -e "s|@GENERATE_LATEX@|NO|g" \
  -e "s|@GENERATE_RTF@|NO|g" \
  -e "s|@GENERATE_MAN@|NO|g" \
  -e "s|@GENERATE_XML@|YES|g" \
  -e "s|@HAVE_DOT@|YES|g" \
  -e "s|@top_srcdir@|${repo_root}|g" \
  "${repo_root}/Doxyfile.in" > "${tmp_doxyfile}"

(
  cd "${repo_root}"
  doxygen "${tmp_doxyfile}"
)
