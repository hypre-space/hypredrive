#!/usr/bin/env python3
"""
Job launcher for hypredrive-cli on HPC systems.

Generates and optionally submits batch scripts from a command-line or YAML
configuration.  The script embeds the hypredrive YAML input for transparency,
tees the solver output to a predictable filename usable by analyze_statistics.py,
and supports several HPC machines.

.hypredriverc format (one entry per line):
    <machine>: /path/to/hypredrive-cli
"""

from datetime import datetime
from pathlib import Path
import yaml
import math
import argparse
import subprocess
import sys
import os
import re
import socket
import shutil

# (submit_cmd, run_cmd, ranks_per_node, default_account)
MACHINE_METADATA = {
    "matrix":     ("sbatch",      "srun",       4,  None),
    "perlmutter": ("sbatch",      "srun",       4,  None),
    "tioga":      ("sbatch",      "srun",       8,  None),
    "frontier":   ("sbatch",      "srun",       8,  None),
    "tuo-spx":    ("flux batch",  "flux run",   4,  None),
    "tuo-cpx":    ("flux batch",  "flux run",  24,  None),
    "dane":       ("sbatch",      "srun",      64,  None),
}
machine_choices = list(MACHINE_METADATA.keys())
today = datetime.now().strftime("%Y-%m-%d")

_VERBOSE = 0


def log(msg, level=1):
    """Conditional stderr logging controlled by global verbosity."""
    if _VERBOSE >= level:
        tag = "DEBUG" if level > 1 else "INFO"
        print(f"[{tag}] {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Machine
# ---------------------------------------------------------------------------

class Machine:
    """
    HPC machine with all parameters needed to generate and submit hypredrive jobs.

    Attributes:
        name (str): Machine identifier (key in MACHINE_METADATA).
        driver_path (str): Full path to the hypredrive-cli executable.
        submit_command (str): Scheduler front-end (e.g. 'sbatch', 'flux batch').
        run_command (str): Parallel launcher (e.g. 'srun', 'flux run').
        ranks_per_node (int): MPI slots per node (hardware-dependent).
        default_account (str | None): Billing account used when none is specified.
    """

    def __init__(self, name, driver_path,
                 submit_command, run_command, ranks_per_node, default_account):
        self.name            = name
        self.driver_path     = driver_path
        self.submit_command  = submit_command
        self.run_command     = run_command
        self.ranks_per_node  = ranks_per_node
        self.default_account = default_account

    def compute_nnodes(self, n_ranks):
        """Minimum node count for the requested rank count."""
        return math.ceil(n_ranks / self.ranks_per_node)

    def _auto_partition(self, time_limit, n_nodes):
        """Select the appropriate partition when the user did not specify one."""
        if self.name == "frontier":
            if (time_limit > 120 and n_nodes < 91) or \
               (time_limit > 360 and n_nodes < 183) or \
               (time_limit > 720 and n_nodes < 9408):
                return "extended"
            return "batch"
        if self.name == "tioga":
            return "pdebug"
        return "pbatch"

    def _machine_env(self, n_ranks, lines):
        """
        Append machine-specific #SBATCH directives and environment exports to *lines*.
        Returns the extra options to pass to srun/flux run (empty string if none).
        """
        run_opts = ""

        if self.name in ("dane", "frontier"):
            lines += [
                "#SBATCH --contiguous",
                "",
                "### Environment",
                "export OMP_NUM_THREADS=1",
                "export MKL_NUM_THREADS=1",
            ]
            if self.name == "frontier":
                gpn = min(8, n_ranks)
                lines += [
                    "",
                    "### Frontier modules",
                    "export MPICH_VERSION=8.1.32",
                    "export MPICH_GPU_SUPPORT_ENABLED=0",
                    "export LD_LIBRARY_PATH="
                    "/opt/cray/pe/mpich/${MPICH_VERSION}/gtl/lib/:"
                    "/opt/cray/pe/mpich/${MPICH_VERSION}/gtl/crayclang/17.0/lib"
                    ":${LD_LIBRARY_PATH}",
                    "module load cray-mpich/${MPICH_VERSION}",
                    "module load rocm/6.2.0",
                ]
                run_opts = f"--gpus-per-node={gpn} --ntasks-per-gpu=1"
                if gpn == 8:
                    run_opts += " --gpu-bind=closest"

        elif self.name == "matrix":
            lines += [
                "#SBATCH --mem=0",
                "",
                "### Affinity wrapper",
                "cat << 'EOF_AFF' > aff.sh",
                "#!/bin/bash",
                "export CUDA_VISIBLE_DEVICES=${SLURM_LOCAL_ID}",
                "$@",
                "EOF_AFF",
                "chmod +x aff.sh",
            ]
            # run_opts = "./aff.sh"   # uncomment to enable CUDA affinity

        elif "tuo" in self.name:
            run_opts = "--exclusive --setopt=mpibind=verbose:1"

        return run_opts

    @staticmethod
    def _yaml_dump_block(inputfile):
        """
        Return script lines that echo the YAML input to stdout inside the job.
        Embedding the config in the log makes runs fully reproducible and
        compatible with analyze_statistics.py, which reads solver output files.
        """
        sep = "=" * 72
        try:
            content = Path(inputfile).read_text().strip()
        except OSError:
            content = f"# Warning: could not read {inputfile} at script-generation time"
        return [
            f"echo '{sep}'",
            "echo 'HYPREDRIVE INPUT YAML START'",
            "cat << 'EOF_YAML'",
            content,
            "EOF_YAML",
            "echo 'HYPREDRIVE INPUT YAML END'",
            f"echo '{sep}'",
        ]

    def generate_job_script(self, n_nodes, n_ranks, inputfile, time_limit,
                             account, email, partition, extra_args,
                             log_level=0, yaml_dump=True):
        """
        Build a complete batch script as a string.

        The solver's stdout is teed to a predictable file
        ``{today}_{casename}_N{n_nodes:03}_np{n_ranks:05}.out`` in the
        submission directory so that analyze_statistics.py can find it
        without knowing the scheduler-assigned job ID.

        Parameters:
            n_nodes (int): Node count.
            n_ranks (int): MPI rank count.
            inputfile (str): Path to the hypredrive YAML config.
            time_limit (int): Walltime in minutes.
            account (str | None): Billing account.
            email (str | None): Address for job-status notifications.
            partition (str | None): Cluster partition; auto-selected if None.
            extra_args (str): Additional CLI arguments for hypredrive-cli.
            log_level (int): Value for HYPREDRV_LOG_LEVEL.
            yaml_dump (bool): Embed the YAML config in the job output.
        """
        casename = Path(inputfile).stem
        log_dir  = Path(f"logs/{self.name}")
        log_dir.mkdir(parents=True, exist_ok=True)
        prefix   = str(log_dir / f"{today}_{casename}_N{n_nodes:03}_np{n_ranks:05}")
        # Output file with a predictable name (no %J) for post-processing
        outfile  = f"{today}_{casename}_N{n_nodes:03}_np{n_ranks:05}.out"

        if not partition:
            partition = self._auto_partition(time_limit, n_nodes)

        L = ["#!/bin/bash"]

        # ── Scheduler headers ──────────────────────────────────────────────
        if self.submit_command == "sbatch":
            L += [
                f"#SBATCH -t {time_limit // 60:02}:{time_limit % 60:02}:00",
                f"#SBATCH -N {n_nodes}",
                f"#SBATCH -e {prefix}_%J.err",
                f"#SBATCH -o {prefix}_%J.out",
                f"#SBATCH -p {partition}",
                "#SBATCH -J hypredrive",
                "#SBATCH --mem=0",
                "#SBATCH --exclusive",
            ]
            if account: L.append(f"#SBATCH -A {account}")
            if email:   L += ["#SBATCH --mail-type=BEGIN,END",
                               f"#SBATCH --mail-user={email}"]

        elif self.submit_command == "flux batch":
            L += [
                "# flux: --exclusive",
                "# flux: --job-name=hypredrive",
                f"# flux: --time={time_limit}",
                f"# flux: --nodes={n_nodes}",
                f"# flux: --nslots={n_ranks}",
                f"# flux: --error={prefix}_{{{{id}}}}.err",
                f"# flux: --output={prefix}_{{{{id}}}}.out",
            ]
            if account: L.append(f"# flux: --bank={account}")
            if "tuo" in self.name:
                mode = "SPX" if "spx" in self.name else "CPX"
                L.append(f"# flux: --amd-gpumode={mode}")

        # ── Machine environment (may append more directives to L) ──────────
        run_opts = self._machine_env(n_ranks, L)

        # ── Embedded YAML ──────────────────────────────────────────────────
        if yaml_dump:
            L += ["", "### hypredrive input YAML (embedded for reproducibility)"]
            L += self._yaml_dump_block(inputfile)

        # ── Execution ──────────────────────────────────────────────────────
        cmd_tokens = [self.run_command, f"-N {n_nodes}", f"-n {n_ranks}"]
        if run_opts:
            cmd_tokens.append(run_opts)
        cmd_tokens += [f"${{DRIVER}}", str(inputfile)]
        if extra_args:
            cmd_tokens.append(extra_args)
        cmd = " ".join(cmd_tokens)

        L += [
            "",
            "### Launch",
            "date; hostname",
            "",
            f"export HYPREDRV_LOG_LEVEL={log_level}",
            "",
            f"DRIVER={self.driver_path}",
            "ls -asl ${DRIVER}",
            "ldd ${DRIVER}",
            f'CMD="{cmd}"',
            'echo "Running: ${CMD}"',
            # Tee to a predictable filename; use PIPESTATUS to propagate exit code
            f"eval ${{CMD}} 2>&1 | tee {outfile}",
            "STATUS=${PIPESTATUS[0]}",
            "",
            "echo Done",
            "date",
            'exit "${STATUS}"',
        ]

        if self.name == "matrix":
            L.insert(-1, "rm -f aff.sh")

        return "\n".join(L)


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------

def _submit(submit_cmd, script, n_nodes, n_ranks, inputfile):
    """Pipe *script* to the scheduler and print the result."""
    log(f"Submitting via '{submit_cmd}'", level=2)
    try:
        res = subprocess.run(
            submit_cmd.split(),   # "flux batch" → ["flux", "batch"]
            input=script, text=True,
            check=True, capture_output=True,
        )
        print(f"Submitted {inputfile} (N={n_nodes}, np={n_ranks}): {res.stdout.strip()}")
    except subprocess.CalledProcessError as e:
        print(f"Submission error (N={n_nodes}, np={n_ranks}): {e.stderr.strip()}",
              file=sys.stderr)


def process_with(args, machine):
    """Generate (and optionally submit) job scripts for all requested rank counts."""
    log(f"Input: {args.inputfile}")
    for np in args.nranks:
        nn = machine.compute_nnodes(np)
        log(f"  np={np} → {nn} nodes", level=2)

        script = machine.generate_job_script(
            nn, np,
            args.inputfile,
            args.maxwalltime,
            args.account,
            args.email,
            args.partition,
            args.extra_args,
            log_level=args.log_level,
            yaml_dump=args.yaml_dump,
        )

        if args.dryrun:
            print(script)
            print("=" * 90)
        else:
            _submit(machine.submit_command, script, nn, np, args.inputfile)

        if args.output:
            out_name = f"{args.machine}_N{nn:03}_np{np:05}.job"
            log(f"Saving script: {out_name}")
            Path(out_name).write_text(script)

    if args.ymloutput:
        _append_yaml_config(args)


# ---------------------------------------------------------------------------
# YAML job-config helpers
# ---------------------------------------------------------------------------

def _verify_config(config):
    """Raise ValueError when required keys are missing from the YAML config."""
    req = {"nranks", "max_walltime"}
    for machine, jobs in config.items():
        if machine in ("input", "log_level", "yaml_dump"):
            continue
        if not isinstance(jobs, list):
            raise ValueError(f"Config for '{machine}' must be a list of job dicts.")
        for idx, job in enumerate(jobs):
            missing = req - job.keys()
            if missing:
                raise ValueError(f"Job {idx} on '{machine}' is missing: {missing}")


def _jobs_from_yaml(args, config):
    """
    Yield one Namespace per job defined in the YAML config for *args.machine*.

    If ``--nranks`` was given on the CLI, only jobs matching those rank counts
    are yielded.  Top-level keys ``input``, ``log_level``, and ``yaml_dump``
    serve as file-wide defaults that individual job entries can override.
    """
    _verify_config(config)

    global_input    = config.get("input")
    global_loglevel = int(config.get("log_level", 0))
    global_yamldump = config.get("yaml_dump", True) not in (False, "false", "no", "off", "0", "n")

    for job in config.get(args.machine, []):
        # CLI --nranks acts as a filter when given alongside -yc
        if args.nranks and job["nranks"] not in args.nranks:
            continue

        new = argparse.Namespace(**vars(args))
        new.inputfile   = job.get("input") or global_input
        new.nranks      = [job["nranks"]]
        new.maxwalltime = job["max_walltime"]
        new.log_level   = int(job.get("log_level", global_loglevel))
        new.yaml_dump   = job.get("yaml_dump", global_yamldump)
        new.extra_args  = job.get("extra_args", args.extra_args or "")

        if not new.inputfile:
            raise ValueError(
                f"No input file for job {job} on '{args.machine}'. "
                "Set 'input' in the job entry or at the top level."
            )
        yield new


def _append_yaml_config(args, yaml_file="jobconf.yml"):
    """
    Append the current run's parameters to *yaml_file* without duplicates.
    Useful for building a reproducible YAML config incrementally.
    """
    try:
        existing = yaml.safe_load(Path(yaml_file).read_text()) or {}
    except FileNotFoundError:
        existing = {}

    jobs = existing.get(args.machine, [])
    for np in args.nranks:
        entry = {
            "nranks":       np,
            "max_walltime": args.maxwalltime,
        }
        if args.inputfile:   entry["input"]      = str(args.inputfile)
        if args.extra_args:  entry["extra_args"]  = args.extra_args
        if args.log_level:   entry["log_level"]   = args.log_level
        if entry not in jobs:
            jobs.append(entry)
    existing[args.machine] = jobs

    Path(yaml_file).write_text(
        yaml.dump(existing, default_flow_style=False, sort_keys=False)
    )
    print(f"Config saved to {yaml_file}")


# ---------------------------------------------------------------------------
# Machine registry
# ---------------------------------------------------------------------------

def get_machines(args):
    """
    Return a dict mapping machine name → Machine.

    Executable paths come from:
    1. ``--executable`` CLI flag (same binary for all machines), or
    2. ``.hypredriverc`` file in the same directory as this script.

    .hypredriverc format::

        dane:   /path/to/hypredrive-cli
        tioga:  /path/to/hypredrive-cli
    """
    if args.executable:
        paths = {m: args.executable for m in machine_choices}
    else:
        rc = Path(__file__).parent / ".hypredriverc"
        if not rc.exists():
            sys.exit(
                f"Error: {rc} not found. "
                "Use -e to specify the hypredrive-cli executable, "
                "or create a .hypredriverc file."
            )
        paths = {}
        for line in rc.read_text().splitlines():
            if ": " in line:
                k, v = line.split(": ", 1)
                paths[k.strip()] = v.strip()

    # Ensure all machines have an entry (None when not configured)
    paths = {**dict.fromkeys(machine_choices, None), **paths}
    return {m: Machine(m, paths[m], *MACHINE_METADATA[m]) for m in machine_choices}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    # Auto-detect the local machine by stripping trailing digits from the hostname
    hostname = re.sub(r"\d+$", "", socket.gethostname())

    p = argparse.ArgumentParser(
        description="Generate and submit hypredrive batch jobs on HPC systems.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry-run on dane with 64 ranks:
  %(prog)s -m dane -i solver.yml -n 64 -t 60 -d

  # Submit two rank counts, save scripts to disk:
  %(prog)s -m dane -i solver.yml -n 64 128 -t 120 -o

  # Run from a YAML job config (filter to 128 ranks only):
  %(prog)s -m dane -yc jobconf.yml -n 128

YAML job config format (jobconf.yml):
  input: solver.yml          # top-level default; overridable per job
  log_level: 0               # HYPREDRV_LOG_LEVEL default
  dane:
    - nranks: 64
      max_walltime: 60
    - nranks: 128
      max_walltime: 120
      input: solver_large.yml   # per-job override
""",
    )

    p.add_argument("-@", "--email",      metavar="ADDR",
                   help="Email address for job-status notifications")
    p.add_argument("-a", "--account",    metavar="PROJ",
                   help="Billing account (default: machine-specific)")
    p.add_argument("-d", "--dryrun",     action="store_true",
                   help="Print job scripts to stdout without submitting")
    p.add_argument("-o", "--output",     action="store_true",
                   help="Save generated job scripts to <machine>_N<nn>_np<np>.job")
    p.add_argument("-e", "--executable", metavar="PATH",
                   help="Path to hypredrive-cli (overrides .hypredriverc)")
    p.add_argument("-p", "--partition",  metavar="QUEUE",
                   help="Cluster partition/queue (default: auto-selected)")
    p.add_argument("-m", "--machine",    default=hostname, choices=machine_choices,
                   metavar="MACHINE",
                   help=f"Target machine (default: auto-detected as '{hostname}'). "
                        f"Choices: {machine_choices}")
    p.add_argument("-n", "--nranks",     nargs="+", type=int, metavar="N",
                   help="MPI rank counts; multiple values submit one job each")
    p.add_argument("-i", "--inputfile",  metavar="FILE",
                   help="hypredrive YAML input file")
    p.add_argument("-t", "--maxwalltime",default=60, type=int, metavar="MIN",
                   help="Maximum walltime in minutes (default: 60)")
    p.add_argument("-L", "--log-level",  default=0, type=int, dest="log_level",
                   metavar="N",
                   help="HYPREDRV_LOG_LEVEL value passed to the job (default: 0)")
    p.add_argument("--extra-args",       default="", dest="extra_args",
                   metavar="ARGS",
                   help="Extra arguments forwarded verbatim to hypredrive-cli")
    p.add_argument("-yc", "--ymlconf",   metavar="FILE",
                   help="YAML job configuration file (alternative to -i / -n)")
    p.add_argument("-yo", "--ymloutput", action="store_true",
                   help="Append current job parameters to jobconf.yml")
    p.add_argument("--yaml-dump",        action="store_true", default=True,
                   help="Embed the input YAML in the job log (default: on)")
    p.add_argument("--no-yaml-dump",     action="store_false", dest="yaml_dump",
                   help="Disable YAML embedding in job log")
    p.add_argument("--unsafe",           action="store_true",
                   help="Skip executable existence check")
    p.add_argument("-v", "--verbose",    action="count", default=0,
                   help="Increase verbosity (-v INFO, -vv DEBUG)")

    args = p.parse_args()

    global _VERBOSE
    _VERBOSE = args.verbose

    # ── Validation ────────────────────────────────────────────────────────
    log(f"Machine: {args.machine}")

    if args.machine not in machine_choices:
        p.error(f"Unknown machine '{args.machine}'. Choices: {machine_choices}")

    if not args.ymlconf and (not args.inputfile or not args.nranks):
        p.error("Provide either -yc <jobconf.yml> or both -i <input.yml> and -n <nranks>.")

    # ── Machine setup ─────────────────────────────────────────────────────
    machines = get_machines(args)
    machine  = machines[args.machine]

    if args.account is None:
        args.account = machine.default_account
        log(f"Using default account: {args.account}", level=2)

    # ── Safety checks ─────────────────────────────────────────────────────
    if not args.unsafe:
        if not machine.driver_path:
            sys.exit(
                f"Error: no executable configured for '{args.machine}'. "
                "Use -e or add an entry to .hypredriverc."
            )
        if not (os.path.isfile(machine.driver_path) or
                shutil.which(machine.driver_path)):
            sys.exit(
                f"Error: executable not found: '{machine.driver_path}'. "
                "Use --unsafe to bypass this check."
            )

    # ── Dispatch ──────────────────────────────────────────────────────────
    if args.ymlconf:
        log(f"Loading YAML config: {args.ymlconf}")
        config = yaml.safe_load(Path(args.ymlconf).read_text())
        if args.machine not in config:
            sys.exit(f"Error: no config for '{args.machine}' in {args.ymlconf}.")
        for job_args in _jobs_from_yaml(args, config):
            process_with(job_args, machine)
    else:
        process_with(args, machine)


if __name__ == "__main__":
    main()
