# RAM experiment setup

## Project architecture

This project uses three distinct locations:

- **Local Mac:** edit, review, and commit code here.
- **GitHub (`ZHONGYUzu/ram`):** source of truth for code synchronization.
- **GPU server:** stores all datasets and runs every model experiment.

Do not commit datasets, model checkpoints, credentials, or experiment outputs.
The server should normally treat the checked-out repository as read-only: update it
from GitHub, then write outputs to a separate results directory.

The reproducibility unit for an experiment is:

1. the Git commit recorded in `git-commit.txt`;
2. the input and mask paths recorded in `command.txt`;
3. the Python packages recorded in `pip-freeze.txt`;
4. the GPU state recorded in `nvidia-smi.txt`; and
5. the full console output recorded in `run.log`.

The helper script `scripts/run_cine_experiment.sh` records all five.

## One-time server setup

Replace `USER`, `SERVER`, and data/results paths with real values. The current
server checkout is `~/ram` and the environment is a Conda prefix environment at
`~/envs/ram`.

```bash
ssh USER@SERVER

cd ~
git clone git@github.com:ZHONGYUzu/ram.git
cd ram

mkdir -p ~/envs
conda create --prefix ~/envs/ram python=3.10 pip -y
conda activate ~/envs/ram
python -m pip install --upgrade pip
python -m pip install \
  torch==2.5.1 \
  torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e .
```

The environment directory is independent of the repository and only needs to be
created once. In a new server session, reactivate it with:

```bash
conda activate ~/envs/ram
which python
```

The Python path printed by `which python` should end in `envs/ram/bin/python`.
Use `conda deactivate` when finished. Do not add a trailing dot to the environment
name unless you intentionally want the directory to be named `ram.`.

The tested server configuration is an NVIDIA Tesla V100 with driver 535.309.01.
That driver is too old for the default PyTorch 2.13.0 CUDA 13 build, so this setup
pins the compatible PyTorch 2.5.1 and Torchvision 0.20.1 CUDA 12.1 wheels before
installing RAM. Verify the environment on a GPU node before running RAM:

```bash
nvidia-smi
python -c 'import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
```

`torch.cuda.is_available()` must print `True`. RAM downloads pretrained weights
from `mterris/ram` on Hugging Face the first time it starts. Keep the Hugging Face
cache on persistent server storage if home directories are temporary:

```bash
export HF_HOME=/server/path/to/persistent-cache/huggingface
```

## Normal code synchronization

### 1. Before editing on the Mac

```bash
cd /Users/zhongyu/Desktop/ram
git status
git pull --ff-only origin main
```

If `git status` shows uncommitted changes, review or commit them before pulling.
Do not use a reset to discard them.

### 2. After editing on the Mac

Run the relevant checks, review the diff, and commit only intended files:

```bash
cd /Users/zhongyu/Desktop/ram
git status
git diff
git add path/to/changed_file.py
git diff --cached
git commit -m "Describe the code change"
git push origin main
```

Use a specific `git add` list rather than `git add .` when data or generated files
could be present.

### 3. Update the server before an experiment

```bash
ssh USER@SERVER
cd ~/ram
git status --short
git pull --ff-only origin main
git log -1 --oneline

conda activate ~/envs/ram
python -m pip install -e .
```

The server worktree should be clean before pulling. Re-running the editable install
ensures dependency metadata changes in `pyproject.toml` are applied.

### 4. If an emergency code fix is made on the server

The preferred workflow is to fix code locally. If a server-only hotfix is
unavoidable, do not leave it untracked:

```bash
cd ~/ram
git status
git add path/to/fixed_file.py
git commit -m "Fix server-discovered issue"
git push origin main
```

Then synchronize the Mac immediately:

```bash
cd /Users/zhongyu/Desktop/ram
git pull --ff-only origin main
```

## Run a recorded CINE experiment

The input H5 and optional mask remain on the server. The H5 file is expected to
contain `kSpace` and `dMap`, as documented by `scripts/infer_cine_h5.py`.

With a MAT mask:

```bash
cd ~/ram
conda activate ~/envs/ram

bash scripts/run_cine_experiment.sh \
  --experiment-id cine-baseline-001 \
  --input-h5 /server/path/to/data/input.h5 \
  --mask-mat /server/path/to/data/mask.mat \
  --results-root /server/path/to/results \
  --slice-index 6 \
  --batch-size 1
```

`--slice-index` and `--time-index` are zero-based and optional. Selecting one
slice while omitting `--time-index` reconstructs every CINE frame for that slice.
Selecting both is the smallest technical test. Replace every `/server/path/...`
placeholder with an existing server path before running the command.

For the established Sub0008 acceleration-8 test, use the assigned comma-delimited
VISTA mask directly. It has shape `(phase, time) = (132, 25)` and is expanded
across the frequency dimension by the inference adapter:

```bash
bash scripts/run_cine_experiment.sh \
  --experiment-id sub0008-acc8-slice06-time10 \
  --input-h5 /mnt/qdata/rawdata/CINE/2D_h5_compressed/Sub0008.h5 \
  --mask-txt /home/students/studxusiy1/mr_recon/masks/mask_VISTA_132x25_acc8_8.txt \
  --results-root ~/ram-results \
  --slice-index 6 \
  --time-index 10 \
  --batch-size 1 \
  --device cuda
```

Use only one of `--mask-mat` and `--mask-txt`. The TXT delimiter defaults to a
comma; pass `--mask-txt-delimiter whitespace` for a whitespace-delimited mask.

To run the same Sub0008 single-slice experiment as a Slurm GPU job, submit the
tracked job file from the repository root:

```bash
cd ~/ram
mkdir -p logs
sbatch scripts/slurm_sub0008_acc8_slice06.sbatch
```

Monitor it with `squeue -u "$USER"`. The scheduler log files are written under
`logs/`, while the inference artifacts remain under
`~/ram-results/sub0008-acc8-slice06-alltimes/`. The job requests one GPU, eight
CPU cores, 64 GB host memory, and three hours.

After the unnormalized acc8 baseline, isolate measurement scaling with a
single-frame p99.5-normalized diagnostic. This computes one scale from the masked
zero-filled reconstruction, divides k-space by it before RAM, restores the scale
on output, and records the mode and scale in the MAT file:

```bash
cd ~/ram
mkdir -p logs
sbatch scripts/slurm_sub0008_acc8_slice06_time10_p995.sbatch
```

After confirming p99.5 normalization, test optional post-RAM measurement data
consistency on the same diagnostic frame. The tracked Slurm array creates three
separate experiment directories for `--dc-gamma 0.1`, `1`, and `10`:

```bash
cd ~/ram
mkdir -p logs
sbatch scripts/slurm_sub0008_acc8_slice06_time10_p995_dc_sweep.sbatch
```

The post-processing step solves a multicoil least-squares proximal problem using
the same acquired k-space and mask. It is disabled by default (`--dc-gamma 0`),
so existing inference behavior is unchanged. Do not expand beyond the diagnostic
frame until a setting exceeds the zero-filled reference of 19.9336 dB PSNR and
improves on 0.341558 NMSE under the same global scale-fit evaluation.

Once a single-frame setting beats zero-filled, submit the tracked overnight
Sub0008 campaign. It contains 168 independent single-frame experiments: 12
gamma/CG refinement runs, 36 temporal runs on slice 6, and 120 runs spanning six
other slices and five times. The submitter packages these into one sequential
12-hour Slurm job so the campaign stays below the account's one-submitted-job
limit. The job skips already-existing experiment directories; if it reaches the
wall-time before finishing, resubmit the same command to resume safely:

```bash
cd ~/ram
bash scripts/submit_sub0008_overnight.sh
```

After the arrays finish, create a CSV containing scale-fitted reconstruction and
zero-filled metrics for every experiment:

```bash
~/envs/ram/bin/python scripts/evaluate_cine_experiments.py \
  --input-h5 /mnt/qdata/rawdata/CINE/2D_h5_compressed/Sub0008.h5 \
  --mask-txt /home/students/studxusiy1/mr_recon/masks/mask_VISTA_132x25_acc8_8.txt \
  --results-root ~/ram-results \
  --experiment-glob 'sub0008-*-s??-t??-*' \
  --output-csv ~/ram-results/sub0008-overnight-metrics.csv
```

If validation shows that post-DC improves RAM consistently but remains behind
zero-filled, run the bounded strong-DC diagnostic before expanding the spatial
grid. It evaluates gamma 100, 300, and 1000 with 8 and 32 post-DC CG iterations
on six representative frames (36 sequential experiments in one Slurm job):

```bash
cd ~/ram
mkdir -p logs
sbatch scripts/slurm_sub0008_strong_dc.sbatch
```

After completion, evaluate only this diagnostic family:

```bash
~/envs/ram/bin/python scripts/evaluate_cine_experiments.py \
  --input-h5 /mnt/qdata/rawdata/CINE/2D_h5_compressed/Sub0008.h5 \
  --mask-txt /home/students/studxusiy1/mr_recon/masks/mask_VISTA_132x25_acc8_8.txt \
  --results-root ~/ram-results \
  --experiment-glob 'sub0008-strongdc-*' \
  --output-csv ~/ram-results/sub0008-strongdc-metrics.csv
```

Without a separate mask, omit both mask options; the inference code derives the
mask from nonzero k-space samples:

```bash
bash scripts/run_cine_experiment.sh \
  --experiment-id cine-baseline-002 \
  --input-h5 /server/path/to/data/input.h5 \
  --results-root /server/path/to/results \
  --batch-size 4
```

Each experiment gets its own directory:

```text
/server/path/to/results/<experiment-id>/
  command.txt
  git-commit.txt
  git-status.txt
  nvidia-smi.txt
  pip-freeze.txt
  reconstruction.mat
  run.log
```

Experiment IDs must be unique. The script refuses to overwrite an existing run.
If GPU memory is insufficient, start with `--batch-size 1`.

## Updating from the original RAM repository

`origin` is the project fork and `upstream` is the original RAM repository. Normal
Mac/server synchronization uses only `origin`. Upstream updates should first be
reviewed and integrated on the Mac:

```bash
cd /Users/zhongyu/Desktop/ram
git fetch upstream
git log --oneline main..upstream/main
git diff main...upstream/main
git merge upstream/main
```

After resolving and testing any conflicts:

```bash
git push origin main
```

Then update the server from `origin` using the normal workflow. Never reset the
server directly to `upstream/main`, because this fork contains the custom CINE H5
inference adapter.
