#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR=""
QUIET=0
FIX_MODE=0
FIX_DRY_RUN=0
BACKUP_EXT=".prefixfix.bak"

usage() {
  cat <<'EOF'
Usage: scripts/check_private_prefix.sh [--root <repo>] [--quiet] [--fix] [--fix-dry-run]

Checks that libHYPREDRV private callable functions use the `hypredrv_` prefix.

Validation rules:
  1) Internal header declarations (`include/*.h`, excluding `HYPREDRV.h`)
     must start with `hypredrv_`.
  2) Non-static global function definitions in `src/*.c`:
     - `src/HYPREDRV.c`: must start with `HYPREDRV_` (public API implementation)
     - `src/main.c`: `main` is allowed
     - all other files: must start with `hypredrv_`

Exit status:
  0 if no violations were found
  1 if violations were found (or in --fix-dry-run mode when changes are needed)
  2 for script usage / environment errors

Fix options:
  --fix          Rewrite private callables to use the hypredrv_ prefix
                 and create one backup per modified file (<file>.prefixfix.bak).
  --fix-dry-run  Show which files would be modified by --fix without editing files.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      [[ $# -ge 2 ]] || { echo "Missing value for --root" >&2; exit 2; }
      ROOT_DIR="$2"
      shift 2
      ;;
    --quiet)
      QUIET=1
      shift
      ;;
    --fix)
      FIX_MODE=1
      shift
      ;;
    --fix-dry-run)
      FIX_MODE=1
      FIX_DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$ROOT_DIR" ]]; then
  ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null || true)"
fi

if [[ -z "$ROOT_DIR" || ! -d "$ROOT_DIR" ]]; then
  echo "Could not determine repository root. Pass --root <repo>." >&2
  exit 2
fi

INCLUDE_DIR="${ROOT_DIR}/include"
SRC_DIR="${ROOT_DIR}/src"

if [[ ! -d "$INCLUDE_DIR" || ! -d "$SRC_DIR" ]]; then
  echo "Expected include/ and src/ under: ${ROOT_DIR}" >&2
  exit 2
fi

if [[ "$QUIET" -eq 0 ]]; then
  echo "Checking private function prefixes under: ${ROOT_DIR}"
fi

declare -a VIOLATIONS=()
declare -A FIX_CANDIDATES=()
header_checked=0
header_decl_count=0
source_checked=0
source_def_count=0

build_fix_file_list() {
  local root="$1"
  local d
  for d in include src tests utils examples; do
    if [[ -d "${root}/${d}" ]]; then
      find "${root}/${d}" -type f \( -name '*.c' -o -name '*.h' \)
    fi
  done | LC_ALL=C sort -u
}

apply_fixes() {
  local -a names=()
  local -a fix_files=()
  local -a changed_files=()
  local name file
  local changed_count=0
  local perl_prog

  mapfile -t names < <(printf '%s\n' "${!FIX_CANDIDATES[@]}" | LC_ALL=C sort)
  if [[ "${#names[@]}" -eq 0 ]]; then
    echo "No auto-fixable private names were discovered."
    return 1
  fi

  mapfile -t fix_files < <(build_fix_file_list "$ROOT_DIR")
  if [[ "${#fix_files[@]}" -eq 0 ]]; then
    echo "No source files found to apply fixes."
    return 1
  fi

  perl_prog="$(mktemp)"
  {
    echo 'use strict;'
    echo 'use warnings;'
    echo 'local $/;'
    echo 'my $s = <>;'
    for name in "${names[@]}"; do
      printf '$s =~ s/(?<![A-Za-z0-9_])%s(?![A-Za-z0-9_])/hypredrv_%s/g;' "$name" "$name"
      echo
    done
    echo 'print $s;'
  } > "$perl_prog"

  for file in "${fix_files[@]}"; do
    local tmp_out
    tmp_out="$(mktemp)"
    perl "$perl_prog" "$file" > "$tmp_out"

    if ! cmp -s "$file" "$tmp_out"; then
      ((changed_count += 1))
      changed_files+=("$file")

      if [[ "$FIX_DRY_RUN" -eq 0 ]]; then
        if [[ ! -e "${file}${BACKUP_EXT}" ]]; then
          cp -- "$file" "${file}${BACKUP_EXT}"
        fi
        mv -- "$tmp_out" "$file"
      else
        rm -f "$tmp_out"
      fi
    else
      rm -f "$tmp_out"
    fi
  done

  rm -f "$perl_prog"

  if [[ "$FIX_DRY_RUN" -eq 1 ]]; then
    echo
    echo "Fix dry-run: ${changed_count} files would be modified."
  else
    echo
    echo "Fix mode: modified ${changed_count} files."
    echo "Backups written as: *${BACKUP_EXT}"
  fi

  if [[ "${#changed_files[@]}" -gt 0 ]]; then
    echo "Changed files:"
    printf '  - %s\n' "${changed_files[@]}"
  fi

  return 0
}

extract_header_decls() {
  local file="$1"
  awk '
    function trim(s) {
      sub(/^[ \t\r\n]+/, "", s)
      sub(/[ \t\r\n]+$/, "", s)
      return s
    }
    function strip_comments(s,    p1, p2, left, right) {
      while (1) {
        if (in_block_comment) {
          p2 = index(s, "*/")
          if (p2 == 0) {
            return ""
          }
          s = substr(s, p2 + 2)
          in_block_comment = 0
        }

        p1 = index(s, "/*")
        if (p1 == 0) {
          break
        }

        p2 = index(substr(s, p1 + 2), "*/")
        if (p2 == 0) {
          s = substr(s, 1, p1 - 1)
          in_block_comment = 1
          break
        }

        left = substr(s, 1, p1 - 1)
        right = substr(s, p1 + 2 + p2 + 1)
        s = left right
      }
      sub(/\/\/.*/, "", s)
      return s
    }
    BEGIN {
      in_block_comment = 0
      in_macro_define = 0
      brace_depth = 0
      pending_decl = 0
      pending_name = ""
      pending_line = 0
    }
    {
      line = strip_comments($0)
      t = trim(line)

      if (in_macro_define) {
        if (t !~ /\\[ \t]*$/) {
          in_macro_define = 0
        }
        next
      }

      if (t ~ /^#define[ \t]/) {
        if (t ~ /\\[ \t]*$/) {
          in_macro_define = 1
        }
        next
      }

      if (t ~ /^#/) {
        next
      }

      if (t == "") {
        next
      }

      if (brace_depth > 0) {
        brace_depth += gsub(/\{/, "{", t) - gsub(/\}/, "}", t)
        if (brace_depth < 0) {
          brace_depth = 0
        }
        next
      }

      if (index(t, "{") > 0 || index(t, "}") > 0) {
        brace_depth += gsub(/\{/, "{", t) - gsub(/\}/, "}", t)
        if (brace_depth < 0) {
          brace_depth = 0
        }
        next
      }

      if (!pending_decl) {
        if (t !~ /^typedef[ \t]/ &&
            match(t, /^(extern[ \t]+)?[A-Za-z_][A-Za-z0-9_ \t\*]*[ \t]+([A-Za-z_][A-Za-z0-9_]*)[ \t]*\(/, m)) {
          pending_decl = 1
          pending_name = m[2]
          pending_line = NR

          if (t ~ /;/) {
            printf "%d:%s\n", pending_line, pending_name
            pending_decl = 0
            pending_name = ""
            pending_line = 0
          }
        }
      } else {
        if (index(t, "{") > 0) {
          pending_decl = 0
          pending_name = ""
          pending_line = 0
          next
        }
        if (t ~ /;/) {
          printf "%d:%s\n", pending_line, pending_name
          pending_decl = 0
          pending_name = ""
          pending_line = 0
        }
      }
    }
  ' "$file"
}

extract_source_global_defs() {
  local file="$1"
  awk '
    function trim(s) {
      sub(/^[ \t\r\n]+/, "", s)
      sub(/[ \t\r\n]+$/, "", s)
      return s
    }
    function strip_comments(s,    p1, p2, left, right) {
      while (1) {
        if (in_block_comment) {
          p2 = index(s, "*/")
          if (p2 == 0) {
            return ""
          }
          s = substr(s, p2 + 2)
          in_block_comment = 0
        }

        p1 = index(s, "/*")
        if (p1 == 0) {
          break
        }

        p2 = index(substr(s, p1 + 2), "*/")
        if (p2 == 0) {
          s = substr(s, 1, p1 - 1)
          in_block_comment = 1
          break
        }

        left = substr(s, 1, p1 - 1)
        right = substr(s, p1 + 2 + p2 + 1)
        s = left right
      }
      sub(/\/\/.*/, "", s)
      return s
    }
    BEGIN {
      in_block_comment = 0
      in_macro_define = 0
      brace_depth = 0
      prev_type_line = ""
      prev_type_static = 0
      pending_def = 0
      pending_name = ""
      pending_line = 0
      pending_static = 0
    }
    {
      line = strip_comments($0)
      t = trim(line)

      if (in_macro_define) {
        if (t !~ /\\[ \t]*$/) {
          in_macro_define = 0
        }
        next
      }

      if (t ~ /^#define[ \t]/) {
        if (t ~ /\\[ \t]*$/) {
          in_macro_define = 1
        }
        next
      }

      if (t ~ /^#/) {
        next
      }

      if (t == "") {
        next
      }

      if (pending_def) {
        if (index(t, "{") > 0) {
          if (!pending_static) {
            printf "%d:%s\n", pending_line, pending_name
          }
          pending_def = 0
          pending_name = ""
          pending_line = 0
          pending_static = 0
        } else if (t ~ /;/) {
          pending_def = 0
          pending_name = ""
          pending_line = 0
          pending_static = 0
        }
      }

      if (!pending_def && brace_depth == 0) {
        if (match(t, /^([A-Za-z_][A-Za-z0-9_]*)[ \t]*\(/, m) &&
            prev_type_line != "" &&
            m[1] !~ /^(if|for|while|switch|return|sizeof)$/) {
          pending_def = 1
          pending_name = m[1]
          pending_line = NR
          pending_static = prev_type_static

          if (index(t, "{") > 0) {
            if (!pending_static) {
              printf "%d:%s\n", pending_line, pending_name
            }
            pending_def = 0
            pending_name = ""
            pending_line = 0
            pending_static = 0
          }
        }
      }

      brace_depth += gsub(/\{/, "{", t) - gsub(/\}/, "}", t)
      if (brace_depth < 0) {
        brace_depth = 0
      }

      if (brace_depth == 0) {
        if (t !~ /[(){};]/ && t !~ /^typedef[ \t]/) {
          prev_type_line = t
          prev_type_static = (t ~ /(^|[ \t])static([ \t]|$)/) ? 1 : 0
        } else {
          prev_type_line = ""
          prev_type_static = 0
        }
      } else {
        prev_type_line = ""
        prev_type_static = 0
      }
    }
  ' "$file"
}

while IFS= read -r header; do
  base="$(basename "$header")"
  if [[ "$base" == "HYPREDRV.h" || "$base" == "compatibility.h" || "$base" == "gen_macros.h" ]]; then
    continue
  fi

  ((header_checked += 1))
  while IFS= read -r rec; do
    [[ -n "$rec" ]] || continue
    line="${rec%%:*}"
    name="${rec#*:}"
    ((header_decl_count += 1))

    if [[ "$name" != hypredrv_* ]]; then
      VIOLATIONS+=("header:${header}:${line}: '${name}' must start with hypredrv_")
      FIX_CANDIDATES["$name"]=1
    fi
  done < <(extract_header_decls "$header")
done < <(LC_ALL=C find "$INCLUDE_DIR" -maxdepth 1 -type f -name '*.h' | sort)

while IFS= read -r source; do
  base="$(basename "$source")"
  ((source_checked += 1))

  while IFS= read -r rec; do
    [[ -n "$rec" ]] || continue
    line="${rec%%:*}"
    name="${rec#*:}"
    ((source_def_count += 1))

    case "$base" in
      HYPREDRV.c)
        if [[ "$name" != HYPREDRV_* ]]; then
          VIOLATIONS+=("source:${source}:${line}: public implementation '${name}' must start with HYPREDRV_")
        fi
        ;;
      main.c)
        if [[ "$name" != "main" ]]; then
          VIOLATIONS+=("source:${source}:${line}: entrypoint file should only expose 'main', found '${name}'")
        fi
        ;;
      *)
        if [[ "$name" != hypredrv_* ]]; then
          VIOLATIONS+=("source:${source}:${line}: private function '${name}' must start with hypredrv_")
          FIX_CANDIDATES["$name"]=1
        fi
        ;;
    esac
  done < <(extract_source_global_defs "$source")
done < <(LC_ALL=C find "$SRC_DIR" -maxdepth 1 -type f -name '*.c' | sort)

if [[ "$QUIET" -eq 0 ]]; then
  echo "Scanned ${header_checked} internal headers (${header_decl_count} declarations)"
  echo "Scanned ${source_checked} source files (${source_def_count} global definitions)"
fi

if [[ "${#VIOLATIONS[@]}" -gt 0 ]]; then
  echo
  echo "Prefix violations found (${#VIOLATIONS[@]}):"
  for v in "${VIOLATIONS[@]}"; do
    echo "  - ${v}"
  done
  echo
  if [[ "$FIX_MODE" -eq 1 ]]; then
    echo "Attempting automatic fix for private callable prefixes..."
    apply_fixes

    if [[ "$FIX_DRY_RUN" -eq 1 ]]; then
      echo
      echo "Dry-run complete. Re-run with --fix to apply edits."
      exit 1
    fi

    echo
    echo "Re-running validation after fixes..."
    if [[ "$QUIET" -eq 1 ]]; then
      bash "$0" --root "$ROOT_DIR" --quiet
    else
      bash "$0" --root "$ROOT_DIR"
    fi
    exit $?
  fi

  echo "Failing check: private callable functions must use hypredrv_ prefix."
  echo "Tip: run with --fix to auto-rename private callables."
  exit 1
fi

if [[ "$QUIET" -eq 0 ]]; then
  echo "OK: all checked private callables use hypredrv_ prefix."
fi

