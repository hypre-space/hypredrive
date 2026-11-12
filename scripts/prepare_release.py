#!/usr/bin/env python3
# Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: MIT
"""Bind and stage a release candidate, then validate its final commit."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time
import tomllib


def command(*args, **kwargs):
    return subprocess.check_output(args, text=True, **kwargs).strip()


def api(repo, endpoint, payload=None, missing_ok=False):
    args = ["gh", "api", "-H", "X-GitHub-Api-Version: 2026-03-10"]
    if payload is not None:
        args += ["--method", "POST", "--input", "-"]
    args.append(f"repos/{repo}/{endpoint}")
    result = subprocess.run(
        args, input=json.dumps(payload) if payload is not None else None,
        text=True, capture_output=True, check=False,
    )
    if result.returncode:
        if missing_ok and "HTTP 404" in result.stderr:
            return None
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return json.loads(result.stdout)


def dispatch(repo, workflow, branch, inputs=None):
    result = api(repo, f"actions/workflows/{workflow}/dispatches",
                 {"ref": branch, "inputs": inputs or {}})
    run_id = result["workflow_run_id"]
    print(f"Started {workflow}: {result['html_url']}", flush=True)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as summary:
            summary.write(f"- [{workflow}]({result['html_url']})\n")
    return run_id


def wait_for_runs(repo, run_ids, expected_sha, timeout=5400):
    pending = set(run_ids)
    deadline = time.monotonic() + timeout
    states = {}
    while pending:
        for run_id in sorted(pending):
            run = api(repo, f"actions/runs/{run_id}")
            if run["head_sha"] != expected_sha:
                raise RuntimeError(f"{run['html_url']}: unexpected source commit")
            state = (run["status"], run["conclusion"])
            if states.get(run_id) != state:
                print(f"{run['html_url']}: {state}", flush=True)
                states[run_id] = state
            if run["status"] == "completed":
                if run["conclusion"] != "success":
                    raise RuntimeError(f"{run['html_url']}: {run['conclusion']}")
                pending.remove(run_id)
        if pending:
            if time.monotonic() >= deadline:
                raise RuntimeError("Timed out waiting for release workflows")
            time.sleep(30)


def branch_sha(repo, branch):
    return api(repo, f"git/ref/heads/{branch}")["object"]["sha"]


def release_version(branch):
    match = re.search(r"^project\(hypredrive VERSION (\d+\.\d+\.\d+)\b",
                      Path("CMakeLists.txt").read_text(), re.MULTILINE)
    if not match or branch != f"release-{match[1]}":
        raise RuntimeError("Prepare Release requires a release-<package-version> branch")
    version = match[1]
    python = tomllib.loads(Path("interfaces/python/pyproject.toml").read_text())
    julia = tomllib.loads(Path("interfaces/julia/Project.toml").read_text())
    if python["project"]["version"] != version or julia["version"] != version:
        raise RuntimeError("Core, Python and Julia package versions must match")
    return version


def verify_bindings(tag, assets):
    bindings = tomllib.loads(Path("interfaces/julia/Artifacts.toml").read_text())
    entries = bindings["hypredrive_mpi_trampoline"]
    expected = {
        f"HYPREDRV.{tag}.{target}.tar.gz"
        for target in ("x86_64-linux-gnu", "aarch64-linux-gnu",
                       "x86_64-apple-darwin", "aarch64-apple-darwin",
                       "x86_64-w64-mingw32")
    }
    found = set()
    for entry in entries:
        for download in entry["download"]:
            url = download["url"]
            name = url.rsplit("/", 1)[-1]
            if f"/releases/download/{tag}/" not in url or name not in expected:
                raise RuntimeError(f"Unexpected Julia artifact URL: {url}")
            if assets[name].get("digest") != f"sha256:{download['sha256']}":
                raise RuntimeError(f"Draft asset does not match Julia binding: {name}")
            found.add(name)
    if found != expected:
        raise RuntimeError("Julia bindings must cover all five release platforms")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--sha", required=True)
    args = parser.parse_args()
    version = release_version(args.branch)
    tag = f"v{version}"
    if command("git", "rev-parse", "HEAD") != args.sha:
        raise RuntimeError("Checkout does not match the requested candidate")
    if branch_sha(args.repo, args.branch) != args.sha:
        raise RuntimeError("Release branch changed before preparation started")
    if api(args.repo, f"git/ref/tags/{tag}", missing_ok=True) is not None:
        raise RuntimeError(f"{tag} already exists; prepare the release before tagging")

    julia = dispatch(args.repo, "julia-artifacts.yml", args.branch,
                     {"release_tag": tag, "update_artifacts_toml": True})
    wait_for_runs(args.repo, [julia], args.sha)

    # The binding workflow intentionally commits using GITHUB_TOKEN. Such a
    # push does not trigger PR CI, so dispatch validation for that final SHA.
    final_sha = branch_sha(args.repo, args.branch)
    command("git", "fetch", "origin", args.branch)
    changed = command("git", "diff", "--name-only", args.sha, final_sha).splitlines()
    if set(changed) - {"interfaces/julia/Artifacts.toml"}:
        raise RuntimeError("Release source changed during the Julia build; rerun preparation")
    command("git", "checkout", "--detach", final_sha)
    workflows = ("ci.yml", "format.yml", "analysis.yml", "coverage.yml",
                 "docs.yml", "portability.yml", "fuzz.yml")
    runs = [dispatch(args.repo, workflow, args.branch) for workflow in workflows]
    runs.append(dispatch(args.repo, "python-wheels.yml", args.branch, {"release_tag": tag}))
    wait_for_runs(args.repo, runs, final_sha)
    if branch_sha(args.repo, args.branch) != final_sha:
        raise RuntimeError("Release branch changed during validation; rerun preparation")

    release = json.loads(command("gh", "release", "view", tag, "--repo", args.repo,
                                 "--json", "isDraft,databaseId,url"))
    if not release["isDraft"]:
        raise RuntimeError("Release preparation requires an unpublished draft")
    assets = api(args.repo, f"releases/{release['databaseId']}/assets?per_page=100")
    by_name = {asset["name"]: asset for asset in assets}
    verify_bindings(tag, by_name)
    wheels = [name for name in by_name if name.endswith(".whl")]
    if len(wheels) != 45 or any(f"-{version}-" not in name for name in wheels):
        raise RuntimeError("Draft release must contain all 45 matching Python wheels")
    if api(args.repo, f"git/ref/tags/{tag}", missing_ok=True) is not None:
        raise RuntimeError("Release was tagged during preparation; review its state manually")
    print(f"Validated {final_sha}; draft assets ready: {release['url']}", flush=True)


if __name__ == "__main__":
    main()
