# Project setup and working agreement

This repository uses a strict local-to-server workflow:

1. **Edit, review, and commit code on the local Mac.**
2. **Push the local commits to GitHub.**
3. **Pull those commits from GitHub on the GPU server.**
4. **Run experiments on the GPU server, where the datasets, Python environment,
   pretrained checkpoint cache, and GPUs are available.**

Do not develop separate versions of the code on the server. Git commit hashes are
the link between source code and experiment results.

## Locations

| Purpose | Location |
|---|---|
| Local working copy | `/Users/zhongyu/Desktop/ram` |
| GitHub repository | `git@github.com:ZHONGYUzu/ram.git` |
| Server working copy | `~/ram` |
| Server Python | `~/envs/ram/bin/python` |
| CINE data example | `/mnt/qdata/rawdata/CINE/2D_h5_compressed/Sub0008.h5` |
| Server result root | `~/ram-results` |
| Server scheduler logs | `~/ram/logs` |

Datasets, checkpoints, credentials, logs, and reconstruction outputs stay off
GitHub.

## 1. Edit and commit locally

Start by checking the local worktree. Preserve all existing and untracked work.

```bash
cd /Users/zhongyu/Desktop/ram
git status --short --branch
git pull --ff-only origin main
```

After editing, review and stage only the intended files. Do not use `git add .`.

```bash
cd /Users/zhongyu/Desktop/ram
git status --short
git diff -- path/to/changed_file
git add path/to/changed_file
git diff --cached
git commit -m "Describe the change"
```

## 2. Push local commits to GitHub

Pushing is an explicit user step:

```bash
cd /Users/zhongyu/Desktop/ram
git push origin main
```

Confirm the commit that the server should use:

```bash
git log -1 --oneline
```

## 3. Update the server from GitHub

Connect to the login node and update the server checkout. The checkout should be
clean before pulling; do not discard server changes automatically.

```bash
ssh login
cd ~/ram
git status --short --branch
git pull --ff-only origin main
git log -1 --oneline
```

If `git status` reports server-side changes, stop and inspect them. Bring any
intentional source change back through the local/GitHub workflow instead of
overwriting it.

Activate the existing server environment when needed:

```bash
conda activate ~/envs/ram
which python
python -c 'import torch, deepinv; print(torch.__version__); print(deepinv.__version__); print(torch.cuda.is_available())'
```

The Python executable should resolve to `~/envs/ram/bin/python`.

## 4. Run small tasks directly on the server

Use an interactive GPU node or an already allocated GPU shell for short
diagnostics and few-slice validation runs. Do not run GPU inference on the login
node.

```bash
ssh node-gpu-05
cd ~/ram
conda activate ~/envs/ram
nvidia-smi
python path/to/script.py --help
```

Run the exact command from the repository root and write outputs outside the
repository, normally below `~/ram-results/`.

For every direct run, record at least:

- the complete command;
- `git rev-parse HEAD`;
- `git status --short`;
- `python -m pip freeze`;
- `nvidia-smi`; and
- the console output and result paths.

## 5. Run larger tasks with Slurm

Use an `sbatch` script for longer runs, parameter sweeps, or work that should
survive a disconnected terminal. Keep job definitions under `scripts/` and
scheduler logs under `logs/`.

```bash
ssh login
cd ~/ram
mkdir -p logs
sbatch scripts/name_of_job.sbatch
squeue -u "$USER"
```

Start with one or a few samples. Inspect shapes, numerical diagnostics, metrics,
and saved images before submitting a larger dataset run.

## 6. Exchange results and findings

Source code moves through GitHub. Large results do not.

- Keep reconstructions, datasets, checkpoints, and logs on the server.
- Add small reproducibility records, summary tables, and exact commands to the
  repository only when they contain no private data or large artifacts.
- Identify every reported experiment by its Git commit hash.
- Copy a small plot or metrics file from the server only when it is needed for
  local inspection, and do not commit it unless explicitly intended.

Example for retrieving one small artifact:

```bash
scp login:~/ram-results/EXPERIMENT/metrics.json /Users/zhongyu/Desktop/ram/
```

## Additional experiment details

See [`EXPERIMENT_SETUP.md`](EXPERIMENT_SETUP.md) for the existing RAM environment,
CINE commands, result-directory structure, and Slurm examples. See
[`EXPERIMENT_RECORD.md`](EXPERIMENT_RECORD.md) for the evolving experiment record.
