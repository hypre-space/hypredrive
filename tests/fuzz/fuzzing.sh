#!/usr/bin/env bash
set -euo pipefail

mode=${1:-}
duration=${2:-600}
engine=${3:-libfuzzer}

if [[ -z "$mode" || ! "$mode" =~ ^(parse|solve|lsseq|matrix|vector)$ ]]; then
    echo "usage: $0 <parse|solve|lsseq|matrix|vector> [duration_seconds=600] [libfuzzer|afl]" >&2
    exit 2
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd "$script_dir/../.." && pwd)
build_dir=${HYPREDRV_FUZZ_BUILD_DIR:-"$repo_dir/build-fuzz"}
target="$build_dir/hypredrv-fuzz-$mode"
if [[ ! -x "$target" ]]; then
    target="$build_dir/tests/fuzz/hypredrv-fuzz-$mode"
fi
if [[ ! -x "$target" ]]; then
    echo "missing fuzz target for mode '$mode' under $build_dir" >&2
    exit 1
fi

timestamp=$(date +%Y%m%d-%H%M%S)
run_dir="$build_dir/fuzz-run/$mode-$timestamp"
mkdir -p "$run_dir/corpus"
mkdir -p "$run_dir/tmp"
export TMPDIR="$run_dir/tmp"
shopt -s nullglob
corpus_inputs=("$script_dir/seeds/$mode"/* "$script_dir/regressions/$mode"/*)
case "$mode" in
    parse)
        corpus_inputs+=("$repo_dir/examples"/*.yml)
        ;;
    solve)
        corpus_inputs+=(
            "$repo_dir/examples/ex1.yml"
            "$repo_dir/examples/ex2.yml"
            "$repo_dir/examples/ex7.yml"
        )
        ;;
esac
shopt -u nullglob
if (( ${#corpus_inputs[@]} == 0 )); then
    echo "no seed or regression inputs found for mode '$mode'" >&2
    exit 1
fi
for input in "${corpus_inputs[@]}"; do
    base=$(basename "$input")
    dest="$run_dir/corpus/$base"
    if [[ -e "$dest" ]]; then
        stem=${base%.*}
        ext=
        if [[ "$base" == *.* ]]; then
            ext=".${base##*.}"
        else
            stem="$base"
        fi
        n=1
        while [[ -e "$run_dir/corpus/${stem}-$n${ext}" ]]; do
            n=$((n + 1))
        done
        dest="$run_dir/corpus/${stem}-$n${ext}"
    fi
    cp "$input" "$dest"
done

dict_args=()
afl_dict_args=()
case "$mode" in
    parse|solve)
        dict_args=(-dict="$script_dir/dicts/yaml.dict" -dict="$script_dir/dicts/cli.dict")
        cat "$script_dir/dicts/yaml.dict" "$script_dir/dicts/cli.dict" >"$run_dir/afl.dict"
        afl_dict_args=(-x "$run_dir/afl.dict")
        ;;
    matrix|lsseq|vector)
        dict_args=(-dict="$script_dir/dicts/ijmatrix.dict")
        afl_dict_args=(-x "$script_dir/dicts/ijmatrix.dict")
        ;;
esac

cleanup() {
    rm -rf "$run_dir/tmp"
}
trap cleanup EXIT

run_status=0
if [[ "$engine" == "libfuzzer" ]]; then
    stdout_log="$run_dir/stdout.log"
    "$target" \
        -max_total_time="$duration" \
        -max_len=65536 \
        -rss_limit_mb=2048 \
        -timeout=10 \
        -print_final_stats=1 \
        -artifact_prefix="$run_dir/" \
        "${dict_args[@]}" \
        "$run_dir/corpus" \
        >"$stdout_log" || run_status=$?
elif [[ "$engine" == "afl" ]]; then
    out_dir="$run_dir/afl"
    mkdir -p "$out_dir"
    afl_timeout_ms=${HYPREDRV_FUZZ_AFL_TIMEOUT_MS:-}
    if [[ -z "$afl_timeout_ms" ]]; then
        case "$mode" in
            solve) afl_timeout_ms=500 ;;
            *) afl_timeout_ms=100 ;;
        esac
    fi
    afl-fuzz -i "$run_dir/corpus" -o "$out_dir" -V "$duration" -t "$afl_timeout_ms" \
        "${afl_dict_args[@]}" -- "$target" || run_status=$?
else
    echo "unknown engine '$engine'" >&2
    exit 2
fi

findings_file="$run_dir/findings.txt"
{
    find "$run_dir" -maxdepth 1 \( -name 'crash-*' -o -name 'leak-*' -o -name 'oom-*' -o -name 'timeout-*' \) -print
    find "$run_dir" \( -path '*/crashes/id:*' -o -path '*/hangs/id:*' \) -print
} | tee "$findings_file"

if [[ -s "$findings_file" ]]; then
    echo "fuzz findings written under $run_dir" >&2
    exit 1
fi

exit "$run_status"
