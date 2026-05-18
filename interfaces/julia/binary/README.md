# Julia artifact maintenance

The Julia package can load HYPREDRV from three places: `HYPREDRV_LIBRARY`,
`HYPREDRV_PREFIX`/`HYPREDRV_DIR`, or a Julia artifact named
`hypredrive_mpi_trampoline`.

Release artifacts are intentionally bound before tagging:

1. Run the `Julia Artifacts` workflow manually on the release branch.
2. Set `update_artifacts_toml=true`.
3. Set `release_tag` to the tag that will be created.
4. Let the workflow commit `interfaces/julia/Artifacts.toml`.
5. Create the release tag from that commit.

The tag workflow verifies that `Artifacts.toml` already references the tag URL
before uploading assets. A tag pushed before the manual binding commit is
expected to fail rather than ship an empty artifact table. The verification is
tag-exact, so bindings for a different release tag are rejected.

`bind_artifact.jl` derives the artifact name from `src/artifacts.jl` so the
loader and maintainer tooling have a single source of truth.

If a platform binding already exists, `bind_artifact.jl` refuses to overwrite it
unless `HYPREDRV_ARTIFACT_REPLACE=1` is set. The release workflow sets this only
for the explicit maintainer update job.
