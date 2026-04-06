# koa-cli

`koa-cli` is a clean local-first CLI for working with the KOA HPC cluster. It focuses on the operational layer only: SSH, rsync, Slurm submission, project-scoped storage, auth token sync, and result retrieval.

The GitHub repository for this codebase is `codex-koa-cli`. The Python package and CLI command remain `koa-cli` and `koa`.

## What It Does

- Sync any local repository to KOA with sensible excludes
- Run the full "prepare, sync, submit" flow with one command
- Submit Slurm jobs with optional automatic GPU selection
- Inspect your jobs and the shared queue
- Stream logs from the real Slurm stdout path
- Review recent local submission history
- Check post-run efficiency with `seff`
- Create clean per-project code and scratch directories
- Sync a local `.env` file to the remote project checkout
- Export Weights & Biases metadata and scratch-backed W&B paths into KOA jobs
- List and download train or eval artifacts from scratch storage

## KOA-Specific Safety Notes

- KOA requires SSH plus DUO MFA on every connection.
- `koa_scratch` data is deleted automatically after 90 days of inactivity.
- On the `gpu` partition, jobs that leave GPUs idle for 3+ consecutive hours can be auto-cancelled.
- Home directories are not intended for active job output; use KOA scratch storage for results.

## Project Layout on KOA

Each local repository is mapped to its own remote project root:

- Code: `~/koa-jobs/<project>`
- Scratch data: `/mnt/lustre/koa/scratch/<user>/koa-jobs/<project>`

That prevents one project from overwriting another and keeps results isolated.

## Install

```bash
git clone https://github.com/michaelburiek/codex-koa-cli.git
cd codex-koa-cli
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install '.[dev]'
```

If editable installs are flaky on your machine, use the repo-local dev helpers
instead of relying on the installed `koa` entrypoint during development:

```bash
./scripts/dev-test -q
./scripts/dev-lint
./scripts/koa-dev --help
```

These commands run directly against `src/` via `PYTHONPATH`, so code changes are
picked up immediately without `pip install -e .`.

If you want the W&B client package available in the same environment as your
training scripts, install the optional W&B extra:

```bash
python3 -m pip install '.[dev,wandb]'
```

## Recommended SSH Setup

For KOA, the most reliable workflow is a dedicated SSH key plus connection
sharing. This keeps `koa-cli` non-interactive and avoids repeated Duo prompts
during a single work session.

If you want the full step-by-step setup from a clean machine, see
[`ssh.md`](ssh.md).

Create a KOA-specific key:

```bash
ssh-keygen -t ed25519 -C "your_uh_username@hawaii.edu" -f ~/.ssh/id_ed25519_koa
mkdir -p ~/.ssh/agent
chmod 700 ~/.ssh ~/.ssh/agent
```

Add a host alias to `~/.ssh/config`:

```sshconfig
Host koa
  HostName koa.its.hawaii.edu
  User your_koa_username
  IdentityFile ~/.ssh/id_ed25519_koa
  IdentitiesOnly yes
  AddKeysToAgent yes
  UseKeychain yes
  ServerAliveInterval 60
  ServerAliveCountMax 3
  ControlMaster auto
  ControlPersist 8h
  ControlPath ~/.ssh/agent/%r@%h:%p
```

Then open one manual SSH session and complete Duo:

```bash
ssh koa
```

After that, update `~/.config/koa-cli/config.yaml` to use `host: koa`, and
`koa-cli` commands can reuse the authenticated master connection for hours.

Note: if KOA requires Duo, there is usually no supported way to make login
"one time forever." The common pattern is one Duo approval per work session,
then `ControlMaster`/`ControlPersist` reuse that session.

## Configure

Run:

```bash
koa setup
```

Or create `~/.config/koa-cli/config.yaml` manually:

```yaml
user: your_koa_username
host: koa.its.hawaii.edu
remote_workdir: ~/koa-jobs
remote_data_dir: /mnt/lustre/koa/scratch/your_koa_username/koa-jobs
# identity_file: ~/.ssh/id_ed25519
# proxy_command: ssh -W %h:%p jump-host
```

Environment variables can override config values:

```bash
export KOA_USER=your_koa_username
export KOA_HOST=koa.its.hawaii.edu
export KOA_REMOTE_WORKDIR=~/koa-jobs
export KOA_REMOTE_DATA_DIR=/mnt/lustre/koa/scratch/$USER/koa-jobs
```

## Common Workflow

```bash
cd ~/code/my-project

koa check
koa run scripts/train.slurm --gpus 2 --time 4:00:00
koa jobs
koa logs
koa efficiency
koa results list --kind train
```

If you want each step separately instead of the one-shot flow:

```bash
koa storage setup --link
koa sync
koa submit scripts/train.slurm --gpus 2 --time 4:00:00
koa logs
koa efficiency
```

## Weights & Biases

`koa-cli` is designed to work with W&B in a split-responsibility setup:

- `koa-cli` handles KOA-specific concerns: SSH, sync, storage, Slurm, partition choice, and GPU selection
- W&B handles experiment dashboards, metrics, run comparison, logs, and artifacts

Recommended setup:

1. Add your key locally:

```bash
echo 'WANDB_API_KEY=your_key_here' >> .env
```

2. Upload it once to the remote project:

```bash
koa wandb sync
```

3. Verify readiness:

```bash
koa wandb check
```

4. Submit a W&B-aware job:

```bash
koa submit scripts/train.slurm \
  --wandb \
  --wandb-project eeg-decoder \
  --wandb-group baseline \
  --wandb-tags koa,eeg
```

When W&B integration is enabled, `koa-cli` exports W&B environment variables
into the remote `sbatch` invocation and stores W&B directories under your KOA
project/scratch layout.

## Commands

- `koa setup`: interactive config wizard
- `koa check`: verify SSH and basic Slurm connectivity
- `koa run <script>`: create storage, sync code, and submit in one command
- `koa sync`: rsync the current repo to the remote project directory
- `koa submit <script>`: upload and submit a Slurm script
- `koa jobs`: show your jobs
- `koa queue`: show the queue, optionally filtered by partition
- `koa cancel <job_id>`: cancel a job
- `koa logs [job_id]`: stream job output, or use the most recent local job
- `koa efficiency [job_id]`: run `seff` for a finished job
- `koa history`: show locally recorded submissions
- `koa storage setup|show|link`: manage remote project directories and symlinks
- `koa auth check|sync`: inspect or upload a `.env` file for the current project
- `koa wandb check|sync`: verify and upload W&B credentials for the current project
- `koa results list|pull`: inspect or download project artifacts

Useful flags:

- `--verbose` or `-v`: print the exact SSH/SCP/rsync command before running it
- `--json`: machine-readable output for `jobs`, `queue`, `history`, and `efficiency`
- `--config FILE`: point at a non-default config file

## Simple Job Example

Minimal Slurm script:

```bash
#!/bin/bash
#SBATCH --job-name=demo-train
#SBATCH --time=00:30:00

set -euo pipefail

RESULTS_DIR="${KOA_REMOTE_DATA_DIR}/train/results/${SLURM_JOB_ID}"
mkdir -p "${RESULTS_DIR}"

cd "${KOA_PROJECT_DIR}"
python train.py --output_dir "${RESULTS_DIR}"
```

Run it:

```bash
koa run scripts/train.slurm --gpus 1 --time 00:30:00
```

Then watch it:

```bash
koa jobs
koa logs
koa efficiency
koa results pull --latest
```

## Slurm Script Contract

`koa submit` exports these environment variables before `sbatch` runs:

- `KOA_PROJECT_DIR`
- `KOA_REMOTE_WORKDIR`
- `KOA_REMOTE_DATA_DIR`
- `WANDB_*` variables when you pass `--wandb` or related W&B flags

Example:

```bash
#!/bin/bash
#SBATCH --job-name=demo
#SBATCH --time=01:00:00

set -euo pipefail

RESULTS_DIR="${KOA_REMOTE_DATA_DIR}/train/results/${SLURM_JOB_ID}"
mkdir -p "${RESULTS_DIR}"

cd "${KOA_PROJECT_DIR}"
python train.py --output_dir "${RESULTS_DIR}"
```

This contract is what makes the CLI easy to integrate into ML projects: your
training script can stay cluster-agnostic and only rely on the exported KOA
paths.

## Development

```bash
./scripts/dev-lint
./scripts/dev-test -q
./scripts/koa-dev --help
```

If you specifically want to verify the packaged console entrypoint too, reinstall
the package into the venv and then run `koa`:

```bash
python3 -m pip --python .venv/bin/python install --force-reinstall --no-deps --no-build-isolation .
koa --help
```

## Resources

- [Koa HPC Documentation](https://www.hawaii.edu/its/ci/koa/)
- [SLURM Documentation](https://slurm.schedmd.com/)
- [GitHub Issues](https://github.com/michaelburiek/codex-koa-cli/issues)

## License

MIT License. See `LICENSE`.
