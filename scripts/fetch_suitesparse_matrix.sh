#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/fetch_suitesparse_matrix.sh [options] <matrix>

Fetch and unpack a SuiteSparse Matrix Collection Matrix Market archive.

Arguments:
  matrix                    Matrix id as a number or Group/Name, a
                            sparse.tamu.edu page URL, or a SuiteSparse
                            Matrix Market archive URL.

Options:
  -d, --dest DIR            Destination root (default: data/suitesparse)
  -f, --force               Re-download and re-extract even if files exist
  --no-extract              Download the archive but do not unpack it
  --dry-run                 Print resolved paths/URL without downloading
  -h, --help                Show this help

Examples:
  scripts/fetch_suitesparse_matrix.sh 1501
  scripts/fetch_suitesparse_matrix.sh Janna/Geo_1438
  scripts/fetch_suitesparse_matrix.sh https://sparse.tamu.edu/Janna/Geo_1438
  scripts/fetch_suitesparse_matrix.sh -d /tmp/ss Janna/Geo_1438

Output layout:
  <dest>/<Group>/<Name>.tar.gz
  <dest>/<Group>/<Name>/<Name>.mtx
EOF
}

die() {
    echo "Error: $*" >&2
    exit 1
}

require_tool() {
    command -v "$1" >/dev/null 2>&1 || die "$1 is required"
}

strip_trailing_slash() {
    local value="$1"
    while [[ "$value" == */ ]]; do
        value="${value%/}"
    done
    printf '%s\n' "$value"
}

resolve_matrix_id() {
    local id="$1"
    local page route found_id

    page="$(curl --fail --location --silent --show-error \
        "https://sparse.tamu.edu/?filterrific%5Bmin_id%5D=${id}&filterrific%5Bmax_id%5D=${id}")"

    found_id="$(printf '%s\n' "$page" |
        sed -n "s|.*<td class='column-id'>\\([0-9][0-9]*\\)</td>.*|\\1|p" |
        head -n 1)"
    [[ "$found_id" == "$id" ]] || die "SuiteSparse matrix ID not found: $id"

    route="$(printf '%s\n' "$page" |
        sed -n "s|.*<td class='column-name'><a href=\"/\\([^\"]*\\)\">.*|\\1|p" |
        head -n 1)"

    [[ "$route" =~ ^[^/]+/[^/]+$ ]] || die "could not resolve SuiteSparse matrix ID: $id"
    printf '%s\n' "$route"
}

dest="data/suitesparse"
force=0
extract=1
dry_run=0
matrix=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--dest)
            [[ $# -ge 2 ]] || die "$1 requires a directory"
            dest="$2"
            shift 2
            ;;
        -f|--force)
            force=1
            shift
            ;;
        --no-extract)
            extract=0
            shift
            ;;
        --dry-run)
            dry_run=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            die "unknown option: $1"
            ;;
        *)
            [[ -z "$matrix" ]] || die "only one matrix argument is supported"
            matrix="$1"
            shift
            ;;
    esac
done

[[ -n "$matrix" ]] || {
    usage >&2
    exit 1
}

require_tool curl
require_tool tar

matrix="$(strip_trailing_slash "$matrix")"
group=""
name=""
url=""

if [[ "$matrix" =~ ^https?:// ]]; then
    # Accept either the collection page form:
    #   https://sparse.tamu.edu/Janna/Geo_1438
    # or the Matrix Market archive form:
    #   https://sparse.tamu.edu/MM/Janna/Geo_1438.tar.gz
    path="${matrix#*://}"
    path="/${path#*/}"
    path="${path%%\?*}"
    path="$(strip_trailing_slash "$path")"

    if [[ "$path" =~ ^/MM/([^/]+)/([^/]+)\.tar\.gz$ ]]; then
        group="${BASH_REMATCH[1]}"
        name="${BASH_REMATCH[2]}"
        url="$matrix"
    elif [[ "$path" =~ ^/([^/]+)/([^/]+)$ ]]; then
        group="${BASH_REMATCH[1]}"
        name="${BASH_REMATCH[2]}"
    else
        die "could not parse SuiteSparse matrix URL: $matrix"
    fi
else
    matrix="${matrix#SuiteSparse Matrix Collection/}"
    matrix="${matrix#SSMC/}"
    matrix="$(strip_trailing_slash "$matrix")"
    if [[ "$matrix" =~ ^[0-9]+$ ]]; then
        matrix="$(resolve_matrix_id "$matrix")"
        group="${matrix%%/*}"
        name="${matrix#*/}"
    elif [[ "$matrix" =~ ^([^/]+)/([^/]+)$ ]]; then
        group="${BASH_REMATCH[1]}"
        name="${BASH_REMATCH[2]}"
    else
        die "matrix must be a numeric ID, Group/Name, or a SuiteSparse URL"
    fi
fi

[[ "$group" =~ ^[A-Za-z0-9_.+-]+$ ]] || die "invalid group: $group"
[[ "$name" =~ ^[A-Za-z0-9_.+-]+$ ]] || die "invalid matrix name: $name"

if [[ -z "$url" ]]; then
    url="https://sparse.tamu.edu/MM/${group}/${name}.tar.gz"
fi

matrix_dir="${dest}/${group}/${name}"
archive_dir="${dest}/${group}"
archive="${archive_dir}/${name}.tar.gz"
mtx="${matrix_dir}/${name}.mtx"

cat <<EOF
matrix:  ${group}/${name}
url:     ${url}
archive: ${archive}
extract: ${matrix_dir}
mtx:     ${mtx}
EOF

if [[ "$dry_run" -eq 1 ]]; then
    exit 0
fi

mkdir -p "$archive_dir"

if [[ "$force" -eq 1 || ! -s "$archive" ]]; then
    echo "Downloading ${url}"
    curl --fail --location --retry 3 --output "$archive" "$url"
else
    echo "Using existing archive ${archive}"
fi

[[ -s "$archive" ]] || die "archive is missing or empty: $archive"

if [[ "$extract" -eq 0 ]]; then
    echo "Archive ready: ${archive}"
    exit 0
fi

if [[ "$force" -eq 1 && -d "$matrix_dir" ]]; then
    rm -rf "$matrix_dir"
fi

if [[ -s "$mtx" ]]; then
    echo "Using existing Matrix Market file ${mtx}"
else
    echo "Extracting into ${archive_dir}"
    tar -xzf "$archive" -C "$archive_dir"
fi

[[ -s "$mtx" ]] || die "expected Matrix Market file not found: $mtx"

echo "Matrix Market file: ${mtx}"
