# KOA SSH Setup From Scratch

This guide walks through a clean SSH setup for `koa-cli` on a Mac if you were
starting from nothing.

It is designed for the KOA cluster at the University of Hawaii and for a local
repo at:

`/Users/michaelburiek/Documents/GitHub/koa-cli`

## Goal

Set up SSH so that:

- you can log into KOA with `ssh koa`
- `koa-cli` can reuse that connection cleanly
- you do not have to create a new SSH config every time
- Duo prompts happen at most once per work session, not on every single command

Important note:

- You usually cannot avoid Duo forever on a protected HPC system
- The practical goal is one interactive login per session, then connection reuse

## 1. Create a dedicated KOA SSH key on your Mac

Run this on your local machine, not on KOA:

```bash
mkdir -p ~/.ssh ~/.ssh/agent
chmod 700 ~/.ssh ~/.ssh/agent
ssh-keygen -t ed25519 -C "mburiek@hawaii.edu" -f ~/.ssh/id_ed25519_koa
chmod 600 ~/.ssh/id_ed25519_koa
chmod 644 ~/.ssh/id_ed25519_koa.pub
```

This creates:

- private key: `~/.ssh/id_ed25519_koa`
- public key: `~/.ssh/id_ed25519_koa.pub`

## 2. Create an SSH config alias on your Mac

Create or edit `~/.ssh/config`:

```sshconfig
Host koa
  HostName koa.its.hawaii.edu
  User mburiek
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

Then lock down permissions:

```bash
chmod 600 ~/.ssh/config
```

## 3. Log into KOA once and install your public key

First, connect from your Mac:

```bash
ssh koa
```

If the new key is not installed yet, KOA may still fall back to password + Duo.
That is okay for this first login.

Once you are on KOA, run:

```bash
mkdir -p ~/.ssh
chmod 700 ~/.ssh
cat >> ~/.ssh/authorized_keys
```

Then paste the contents of your local public key, which you can view on your Mac
with:

```bash
cat ~/.ssh/id_ed25519_koa.pub
```

After pasting the key into the KOA terminal, press `Control-D`, then run:

```bash
chmod 600 ~/.ssh/authorized_keys
```

## 4. Test the dedicated KOA key from your Mac

Exit KOA:

```bash
exit
```

Then reconnect from your Mac:

```bash
ssh koa
```

If KOA policy still requires Duo, you may still see Duo once here. That is
normal. The important part is that the dedicated key is now installed and the
`Host koa` alias is persistent.

Because `ControlMaster` and `ControlPersist` are enabled, later commands can
reuse this authenticated session for up to 8 hours.

## 5. Configure `koa-cli` to use the alias

Your local `koa-cli` config should look like this:

```yaml
user: mburiek
host: koa
remote_workdir: ~/koa-jobs
remote_data_dir: /mnt/lustre/koa/scratch/mburiek/koa-jobs
identity_file: ~/.ssh/id_ed25519_koa
```

That `host: koa` value is intentional. It tells `koa-cli` to use the SSH alias
from `~/.ssh/config`.

## 6. Test `koa-cli` locally from your Mac

Run these on your Mac inside the repo:

```bash
cd /Users/michaelburiek/Documents/GitHub/koa-cli
source .venv/bin/activate
./scripts/dev-test -q
./scripts/dev-lint
./scripts/koa-dev check
```

If that succeeds, continue with:

```bash
./scripts/koa-dev storage setup --link
./scripts/koa-dev sync --dry-run
```

## Local vs Remote Paths

This is the most common confusion point:

- `/Users/michaelburiek/Documents/GitHub/koa-cli` is a path on your Mac
- it does not exist on KOA

So these commands belong on your Mac:

```bash
cd /Users/michaelburiek/Documents/GitHub/koa-cli
source .venv/bin/activate
./scripts/koa-dev check
```

These commands belong on KOA:

```bash
mkdir -p ~/.ssh
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
```

## Daily Workflow

Typical start of a work session:

```bash
ssh koa
```

Complete Duo if prompted, then you can usually exit that shell.

After that, run `koa-cli` commands from your Mac:

```bash
cd /Users/michaelburiek/Documents/GitHub/koa-cli
source .venv/bin/activate
./scripts/koa-dev check
./scripts/koa-dev sync
./scripts/koa-dev submit scripts/smoke.slurm
```

## Troubleshooting

If `koa-cli` hangs or fails:

1. Test raw SSH first:

```bash
ssh koa
```

2. If needed, inspect SSH behavior:

```bash
ssh -v koa
```

3. Confirm your public key is installed on KOA:

```bash
cat ~/.ssh/authorized_keys
```

4. Confirm your local alias exists:

```bash
cat ~/.ssh/config
```

5. Confirm `koa-cli` is using the alias:

```bash
cat ~/.config/koa-cli/config.yaml
```

## Recommended Pattern

Use a dedicated KOA key and keep your older SSH keys unless you are sure nothing
else depends on them.

That gives you:

- a cleaner KOA setup
- less risk of breaking GitHub or other SSH tools
- a persistent SSH alias
- reusable authenticated connections for a work session
