# fastMRI dataset inventory and RAM validation plan

Last updated: 2026-07-23

## Experiment ledger

This file is the canonical human-readable fastMRI/RAM experiment record. Do not
rely on chat history for experiment state. The machine-readable high-level
ledger is `fastMRI_experiment_results.csv`; all 25 trial deltas from Slurm job
41963 are preserved in `fastMRI_sweep_job_41963_deltas.csv`. Per-slice metrics,
images, environment information, and reconstructed arrays remain in each
server-side result directory.

| # | Experiment | Dataset | Configurations / slice cases | Physics matrix | Metric matrix | Status | Main result |
|---:|---|---|---:|---:|---:|---|---|
| 1 | Knee single-coil acceleration 8 | `knee/data/singlecoil_val/file1000000.h5` | 1 / 3 | `640×368` | `320×320` | Completed | RAM and ZF tied; numerical checks passed; anatomy is out of distribution |
| 2 | Brain virtual-coil first smoke | Brain AXFLAIR volume below | 1 / 3 | `640×320` | `320×320` | Completed | RAM `+0.045 dB`; preprocessing still mismatched |
| 3 | Brain parameter sweep | Same brain AXFLAIR volume | 25 / 75 | `640×320` | `320×320` | Completed, job 41963 | Scale dominated; best divisor `0.0002`, RAM `+0.0645 dB` |
| 4 | Brain complex-crop smoke | Same brain AXFLAIR volume | 1 / 3 | `320×320` | `320×320` | Next | Crop complex image before FFT/measurement; use best scale from experiment 3 |

“Slice cases” counts reconstructed slices, so experiment 3 contains 25
configurations × 3 slices = 75 slice cases. None of experiments 1–4 is a large
dataset run.

## Server location

The fastMRI data is stored at:

```text
/mnt/qdata/rawdata/fastMRI
```

The inventory command found 10,261 H5 files. Approximate top-level sizes were:

| Directory | Size observed |
|---|---:|
| `brain/` | 2.4 TB |
| `knee/` | 4.7 TB |
| `knee_nyu/` | 573 GB |
| `breast/` | no H5 files found by the inventory command |
| `prostate/` | no H5 files found by the inventory command |

## Important selection result

The installed brain data is multicoil-only. No single-coil brain validation
folder was found. Therefore the most controlled RAM validation sequence is:

1. Validate the inference implementation on a few slices from fastMRI knee
   `singlecoil_val`, using acceleration 8 and the official DeepInverse `MRI`
   physics.
2. Once the single-coil path works, validate a few brain slices from
   `brain/multicoil_val`, using the official DeepInverse `MultiCoilMRI` physics
   and its required measurement shape `(B, 2, coils, H, W)`.
3. Use matching ESPIRiT sensitivity maps from `brain/multicoil_val_espirit` for
   the multicoil experiment.
4. Do not use test or challenge data for the initial quantitative comparison:
   those files contain masks but no reconstruction targets, so PSNR, NMSE, and
   SSIM cannot be evaluated against ground truth.

This order separates an inference/representation error from anatomy and
acquisition domain shift. A successful knee single-coil result validates the
two-real-channel MRI path before introducing multicoil sensitivity maps and the
brain domain.

## Recommended first dataset: knee single-coil validation

Path:

```text
/mnt/qdata/rawdata/fastMRI/knee/data/singlecoil_val
```

Observed file count: 199 volumes.

Representative files:

| File | K-space | Targets | Acquisition |
|---|---|---|---|
| `file1000000.h5` | `(35, 640, 368)`, `complex64` | `reconstruction_esc` and `reconstruction_rss`: `(35, 320, 320)`, `float32` | `CORPDFS_FBK` |
| `file1000007.h5` | `(38, 640, 368)`, `complex64` | `reconstruction_esc` and `reconstruction_rss`: `(38, 320, 320)`, `float32` | `CORPDFS_FBK` |

The dimensions are `(slices, H, W)` for single-coil k-space. For DeepInverse,
one selected complex slice must be converted to two real/imaginary channels:

```text
raw H5 slice:       (H, W), complex
DeepInverse input:  (B, 2, H, W), real-valued tensor
```

Use a fully sampled validation file, generate a retrospective Cartesian
acceleration-8 mask, and compare zero-filled and plain `model(y, physics)` output
against `reconstruction_esc`. Begin with only a few central slices from one or
two volumes.

Related folders:

| Folder | Files | Use |
|---|---:|---|
| `singlecoil_train` | 973 | Fully sampled training volumes; unnecessary for initial validation |
| `singlecoil_train_bak` | 973 | Apparent backup; do not use unless provenance is clarified |
| `singlecoil_val` | 199 | **Use first**: fully sampled k-space plus targets |
| `singlecoil_test` | 108 | Already masked, acceleration metadata present, no target |
| `singlecoil_challenge` | 92 | Already masked, no target |

The observed test examples were acceleration 8 with 15 low-frequency lines.
The observed challenge examples were acceleration 4 with 30 or 41
low-frequency lines. These are useful references for mask configuration, but
not for the first metric-bearing validation.

## Recommended brain dataset: multicoil validation

Raw validation path:

```text
/mnt/qdata/rawdata/fastMRI/brain/multicoil_val
```

ESPIRiT path:

```text
/mnt/qdata/rawdata/fastMRI/brain/multicoil_val_espirit
```

Observed raw validation count: 504 volumes. The ESPIRiT directory contained
1,365 H5 files, so files must be matched by filename and checked rather than
assuming that both directories have identical membership.

Representative matching volume: `file_brain_AXFLAIR_200_6002462.h5`.

| Dataset | Key | Shape and dtype |
|---|---|---|
| Raw validation | `kspace` | `(16, 16, 640, 320)`, `complex64` |
| Raw validation | `reconstruction_rss` | `(16, 320, 320)`, `float32` |
| ESPIRiT | `reference_acl15` | `(16, 2, 640, 320)`, `complex128` |
| ESPIRiT | `reference_acl30` | `(16, 2, 640, 320)`, `complex128` |
| ESPIRiT | `smaps_acl15` | `(16, 16, 2, 640, 320)`, `complex64` |
| ESPIRiT | `smaps_acl30` | `(16, 16, 2, 640, 320)`, `complex64` |

For raw k-space the dimensions are `(slices, coils, H, W)`. DeepInverse
`MultiCoilMRI` requires:

```text
raw H5 slice:       (coils, H, W), complex
DeepInverse input:  (B, 2, coils, H, W), real-valued tensor
```

The ESPIRiT sensitivity-map tensor contains an extra map dimension of size 2:
`(slices, coils, maps, H, W)`. The reconstruction script must select or correctly
combine the map set expected by DeepInverse; it must not silently interpret that
map dimension as real/imaginary channels. Start with the ACL15 product and one
central slice, while recording the exact selection.

Related brain folders:

| Folder | Files | Contents/use |
|---|---:|---|
| `multicoil_train` | 38 | Fully sampled k-space and RSS targets |
| `multicoil_val` | 504 | **Use for brain metrics**: fully sampled k-space and RSS targets |
| `multicoil_val_espirit` | 1,365 | Complex references and ESPIRiT maps; match by filename |
| `multicoil_test` | 558 | Already masked test data; no reconstruction target |

Observed brain test examples were acceleration 8, with 13 low-frequency lines,
and k-space shaped approximately `(slices, 20 coils, 640, 320)`.

## Other available multicoil data

The knee dataset also provides 199 fully sampled `multicoil_val` volumes and 199
matching `multicoil_val_espirit` files. Representative raw k-space has shape
`(35, 15, 640, 368)` and the target has shape `(35, 320, 320)`. This is a useful
intermediate test if knee single-coil succeeds but brain multicoil fails.

The `knee_nyu` folders contain explicit complex references and sensitivity maps:

| Folder | Files | Representative shapes |
|---|---:|---|
| `axial_t2` | 21 | k-space/smaps `(38, 15, 640, 484)`; reference `(38, 320, 320)` |
| `coronal_pd` | 21 | k-space/smaps `(42, 15, 640, 368)`; reference `(42, 320, 320)` |
| `coronal_pd_fs` | 21 | k-space/smaps approximately `(40, 15, 640, 368)`; reference `(40, 320, 320)` |

These files use `complex128` and are not the first choice for matching the
official DeepInverse fastMRI loader and RAM's training distribution.

## Validation requirements

The initial experiment must verify and record all of the following before any
larger run:

- official DeepInverse `MRI` or `MultiCoilMRI`, not a custom CINE operator;
- real/imaginary channel layout and exact tensor shapes;
- centered orthonormal FFT convention;
- Cartesian mask orientation and achieved acceleration;
- input/target normalization and restoration of scale;
- Gaussian noise sigma used by RAM conditioning;
- numerical adjointness test;
- estimated operator norm;
- zero-filled and RAM PSNR, NMSE, and SSIM;
- plain `model(y, physics)` with no post-RAM data consistency;
- saved reference, zero-filled, RAM, error, and mask images for a few slices;
- exact command, Git commit, package versions, GPU details, and result paths.

## Experiment record

### Experiment 1: knee single-coil acceleration-8 smoke test

| Field | Recorded value |
|---|---|
| Experiment ID | `fastmri-knee-sc-acc8-smoke-001` |
| Status | Completed successfully in `r3`; image inspection pending |
| Dataset | fastMRI knee single-coil validation |
| Input volume | `/mnt/qdata/rawdata/fastMRI/knee/data/singlecoil_val/file1000000.h5` |
| Slices | 16, 17, 18 (zero-based) |
| Physics | Official `deepinv.physics.MRI` |
| Measurement representation | `(B, 2, H, W)` real/imaginary channels |
| Mask | Official DeepInverse equispaced Cartesian mask |
| Acceleration / center fraction | 8 / 0.04 |
| Normalization | DeepInverse ACS RSS 99th percentile, 15 ACS lines |
| Noise sigma | `0.001` for RAM conditioning; no synthetic noise added |
| RAM call | Plain `model(y, physics)` |
| Post-RAM data consistency | None |
| Metrics | PSNR, NMSE, SSIM for zero-filled and RAM |
| First-attempt output | `~/ram-results/fastmri-knee-sc-acc8-smoke-001/` |
| Stale-code retry output | `~/ram-results/fastmri-knee-sc-acc8-smoke-001-r1/` |
| Second-attempt output | `~/ram-results/fastmri-knee-sc-acc8-smoke-001-r2/` |
| Current retry output | `~/ram-results/fastmri-knee-sc-acc8-smoke-001-r3/` |
| Git commit | `03152f8` expected; verify from the saved `git-commit.txt` |
| Result | `r3` completed; RAM was essentially tied with zero-filled on the three knee slices |

Run from an interactive GPU node:

```bash
cd ~/ram
conda activate ~/envs/ram
set -o pipefail

python scripts/validate_fastmri_ram.py \
  --input-h5 /mnt/qdata/rawdata/fastMRI/knee/data/singlecoil_val/file1000000.h5 \
  --output-dir ~/ram-results/fastmri-knee-sc-acc8-smoke-001-r3 \
  --slices 16 17 18 \
  --acceleration 8 \
  --center-fraction 0.04 \
  --acs 15 \
  --noise-sigma 0.001 \
  --seed 0 \
  --device cuda 2>&1 | tee ~/ram-results/fastmri-knee-sc-acc8-smoke-001-r3-console.log && \
  mv ~/ram-results/fastmri-knee-sc-acc8-smoke-001-r3-console.log \
    ~/ram-results/fastmri-knee-sc-acc8-smoke-001-r3/run.log
```

The output directory is intentionally required to be empty. Use a new experiment
ID rather than overwriting a previous run. The script writes environment details,
the exact command, Git status and commit, package versions, GPU information,
operator diagnostics, per-slice and aggregate metrics, PNG comparison panels,
and compressed reconstruction arrays.

#### Experiment 1 results

Successful runtime environment:

| Component | Value |
|---|---|
| Node/GPU | `node-gpu-01`; Tesla V100-SXM2 32 GB |
| PyTorch/CUDA | 2.5.1+cu121 / CUDA 12.1 |
| DeepInverse | 0.4.1 |
| h5py | 3.16.0 |
| Matplotlib | 3.10.9 |

Numerical validation:

| Check | Result |
|---|---:|
| Mask shape | `(1, 2, 640, 368)` |
| Sampled fraction / acceleration | 0.125 / 8.0 |
| Sampled columns / rows | 46 / 640 |
| Mask constant across rows | true |
| Adjoint relative error | `1.60e-7` |
| Estimated operator norm | `0.99999994` |
| DeepInverse versus explicit centered orthonormal IFFT error | `0.0` |
| Full-k-space physics reference versus H5 target error | approximately `1.8e-7` |

Aggregate reconstruction metrics across slices 16, 17, and 18:

| Reconstruction | PSNR (dB) | NMSE | SSIM |
|---|---:|---:|---:|
| Zero-filled | 24.27225 | 0.123079 | 0.464423 |
| RAM | 24.25602 | 0.123539 | 0.468138 |
| RAM minus zero-filled | -0.01623 | +0.000459 | +0.003715 |

The behavior was consistent across all three slices: RAM lost approximately
0.014--0.017 dB PSNR, slightly increased NMSE, and slightly improved SSIM. This
is a numerical tie rather than a meaningful reconstruction improvement.

The operator and representation checks are sufficiently accurate to validate
the basic single-coil inference implementation. In particular, this result is
not explained by FFT centering, FFT normalization, mask orientation,
acceleration, adjointness, operator scaling, target cropping, or a missing
real/imaginary channel.

However, knee single-coil is not the MRI distribution used to train RAM. The RAM
paper states that its MRI images were produced by virtual coil-combination of
the fastMRI **brain multicoil** dataset. Its in-distribution MRI task used
acceleration factors 4 and 8. Therefore this experiment is best interpreted as
an out-of-distribution knee smoke test, not a reproduction of the paper's
in-distribution result. See the RAM paper's
[MRI preprocessing and training configuration](https://arxiv.org/html/2503.08915#A1.SS2).

Before testing native multicoil inference, the next controlled experiment should
use the complex virtual-coil-combined brain validation references in
`brain/multicoil_val_espirit`, select the intended ESPIRiT reference map, simulate
single-coil acceleration 8 with official DeepInverse `MRI`, and run the same
checks. This more closely matches RAM's training distribution and cleanly tests
the pretrained checkpoint before introducing `MultiCoilMRI`.

Visual observations: pending inspection of `slice_0016.png`, `slice_0017.png`,
and `slice_0018.png`.

#### Dataset-shift finding

Experiment 1 exposed an important dataset mismatch that must be kept separate
from inference-code correctness:

| Property | RAM MRI training distribution | Experiment 1 | Cardiac CINE |
|---|---|---|---|
| Anatomy | Brain | Knee | Heart |
| Source | fastMRI brain multicoil, converted to complex virtual-coil images | fastMRI native knee single-coil | Custom dynamic CINE H5 |
| Temporal dimension | Static 2D slices | Static 2D slices | Dynamic 2D+t frames |
| Training status | In distribution | Out of distribution | Out of distribution |
| Sampling used here | Cartesian acceleration 4/8 in RAM training | Equispaced Cartesian acceleration 8 | VISTA k-t acceleration 8 |
| Measurement representation | Single complex image as `(B,2,H,W)` after virtual coil-combination | `(B,2,H,W)` | Previously passed as native complex multicoil data; RAM requires `(B,2,coils,H,W)` for multicoil physics |

The knee experiment therefore validates the single-coil DeepInverse/RAM
inference mechanics but cannot reproduce the paper's in-distribution MRI result.
Its near-tie between RAM and zero-filled may be caused by knee-to-brain dataset
shift and/or by preprocessing differences. It is not evidence by itself that the
RAM checkpoint is ineffective on its intended MRI distribution.

Experiment 2 uses virtual-coil-combined fastMRI brain validation references and
is the required in-distribution checkpoint test. Only after experiment 2 should
we interpret the CINE deficit: strong brain performance would point toward
cardiac/CINE domain shift or CINE-specific multicoil handling, whereas weak brain
performance would indicate a remaining checkpoint, mask, noise-conditioning, or
preprocessing mismatch in our inference pipeline.

### Experiment 2: brain virtual-coil acceleration-8 smoke test

This is the first test designed to match RAM's MRI training distribution.

| Field | Recorded value |
|---|---|
| Experiment ID | `fastmri-brain-vcc-acc8-smoke-002` |
| Status | Completed; retained as the first brain preprocessing diagnostic |
| First-run date | 2026-07-23 |
| ESPIRiT input | `/mnt/qdata/rawdata/fastMRI/brain/multicoil_val_espirit/file_brain_AXFLAIR_200_6002462.h5` |
| Matching raw input | `/mnt/qdata/rawdata/fastMRI/brain/multicoil_val/file_brain_AXFLAIR_200_6002462.h5` |
| Complex reference | `reference_acl15`, with map selected automatically against `reconstruction_rss` |
| Slices | 7, 8, 9 (zero-based central slices) |
| Physics | Official `deepinv.physics.MRI` |
| Measurement representation | `(B, 2, H, W)` real/imaginary channels |
| Acceleration / center fraction | 8 / 0.04 |
| First-run mask | Equispaced Cartesian |
| First-run normalization | Per-slice p99 of cropped virtual-coil reference magnitude |
| First-run noise | Sigma `0.001` for conditioning; no noise added |
| RAM call | Plain `model(y, physics)` |
| Post-RAM data consistency | None |
| Metrics | PSNR, NMSE, SSIM for zero-filled and RAM |
| First-run output | `~/ram-results/fastmri-brain-vcc-acc8-smoke-002/` |
| Result | RAM essentially tied with zero-filled; follow-up settings tested in experiment 3 |

Run from an interactive GPU node:

```bash
cd ~/ram
conda activate ~/envs/ram
set -o pipefail

python scripts/validate_fastmri_brain_ram.py \
  --input-h5 /mnt/qdata/rawdata/fastMRI/brain/multicoil_val_espirit/file_brain_AXFLAIR_200_6002462.h5 \
  --raw-h5 /mnt/qdata/rawdata/fastMRI/brain/multicoil_val/file_brain_AXFLAIR_200_6002462.h5 \
  --output-dir ~/ram-results/fastmri-brain-vcc-acc8-smoke-002-r1 \
  --reference-key reference_acl15 \
  --map-index auto \
  --slices 7 8 9 \
  --no-crop-before-physics \
  --acceleration 8 \
  --center-fraction 0.04 \
  --mask-type random \
  --normalization-scale 0.005 \
  --noise-sigma 0.0005 \
  --add-noise \
  --seed 0 \
  --device cuda 2>&1 | tee ~/ram-results/fastmri-brain-vcc-acc8-smoke-002-r1-console.log && \
  mv ~/ram-results/fastmri-brain-vcc-acc8-smoke-002-r1-console.log \
    ~/ram-results/fastmri-brain-vcc-acc8-smoke-002-r1/run.log
```

#### Experiment 2 first-run results

The first brain execution selected reference map 0; map 1 had zero usable
energy. All numerical operator checks passed: exact centered orthonormal FFT
agreement, adjoint relative error 0, operator norm approximately 1, and exact
acceleration 8 along the correct phase-encoding axis.

| Reconstruction | PSNR (dB) | NMSE | SSIM |
|---|---:|---:|---:|
| Zero-filled | 22.13886 | 0.053529 | 0.475689 |
| RAM | 22.18389 | 0.052969 | 0.473348 |
| RAM minus zero-filled | +0.04503 | -0.000560 | -0.002341 |

This is only a marginal PSNR/NMSE improvement and a small SSIM regression. It
does not reproduce the published in-distribution RAM result. The run used
per-slice p99 normalization, sigma `0.001` without synthetic noise, and an
equispaced mask. Experiment 3 tested these suspected differences and showed
that treating `0.005` as a divisor was not supported by the observed data.

### Experiment 3: overnight brain parameter sweep

Script: `scripts/slurm_fastmri_brain_overnight.sbatch`

Status: completed on 2026-07-23 as Slurm job `41963`, using Git commit
`b8db0fdb29d2f01accd9b932d8624065ed7a5f7d`.

The sweep runs 25 configurations sequentially in one resumable 12-hour Slurm
job. This respects the account's one-submitted-job limit and avoids an opaque
full Cartesian product. Every run uses brain slices 7, 8, and 9, plain
`model(y, physics)`, and no post-RAM data consistency.

| Family | Trials | Parameters |
|---|---:|---|
| Paper core | 6 | R=8/center 0.04 and R=4/center 0.08; random masks; seeds 0, 1, 2; scale 0.005; sigma 0.0005 with noise |
| Global scaling | 6 | Scale 0.0002, 0.0005, 0.001, 0.0025, 0.01, 0.02; otherwise paper R=8 setup |
| Noise | 8 | Four noisy sigma values and four noiseless conditioning sigma values |
| Mask/reference | 5 | Two equispaced seeds, center fractions 0.02/0.08, and ACL30 reference |

The job skips directories that already contain `metrics.json`, preserves failed
or incomplete directories, and writes the combined comparison to:

```text
~/ram-results/fastmri-brain-overnight-sweep.csv
```

Submit all families:

```bash
cd ~/ram
mkdir -p logs
sbatch --export=ALL,BUNDLE=all scripts/slurm_fastmri_brain_overnight.sbatch
```

Submit only one diagnostic family if needed:

```bash
sbatch --export=ALL,BUNDLE=core scripts/slurm_fastmri_brain_overnight.sbatch
sbatch --export=ALL,BUNDLE=scale scripts/slurm_fastmri_brain_overnight.sbatch
sbatch --export=ALL,BUNDLE=noise scripts/slurm_fastmri_brain_overnight.sbatch
sbatch --export=ALL,BUNDLE=mask scripts/slurm_fastmri_brain_overnight.sbatch
```

Do not submit these simultaneously under the one-job account limit. The `all`
bundle is the default and is the recommended overnight command.

#### Experiment 3 completed results

All 25 configurations completed successfully: 25 configurations × 3 central
slices = 75 reconstructed slice cases. Every case used the two-real-channel
representation `(B,2,640,320)`, centered orthonormal FFT, official DeepInverse
`MRI`, plain `model(y, physics)`, and no post-RAM data consistency. The complex
reference stayed at `640×320` through physics; magnitude images were
center-cropped to `320×320` only for metrics.

The full server summary is
`~/ram-results/fastmri-brain-overnight-sweep.csv`. The tracked trial-delta table
is `fastMRI_sweep_job_41963_deltas.csv`.

| Configuration | ZF PSNR | RAM PSNR | ΔPSNR | ZF NMSE | RAM NMSE | ΔNMSE | ZF SSIM | RAM SSIM | ΔSSIM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Best scale: divisor `0.0002`, R8, random, seed 0, σ `0.0005` added | 22.68076 | 22.74524 | +0.06448 | 0.047313 | 0.046614 | -0.000699 | 0.468203 | 0.465472 | -0.002731 |
| Divisor `0.0005`, otherwise same | 22.68060 | 22.71671 | +0.03611 | 0.047315 | 0.046920 | -0.000395 | 0.467883 | 0.444931 | -0.022952 |
| Divisor `0.001`, otherwise same | 22.68007 | 22.63933 | -0.04074 | 0.047321 | 0.047771 | +0.000450 | 0.466753 | 0.400666 | -0.066088 |
| Previous divisor `0.005`, R8 seed 0 | 22.66294 | 20.80290 | -1.86005 | 0.047509 | 0.072958 | +0.025449 | 0.436211 | 0.214890 | -0.221321 |

Interpretation:

- The fixed divisor `0.005` was incorrect for these raw ESPIRiT reference
  values and severely harmed RAM.
- The empirically best divisor was `0.0002`, equivalent to multiplying the raw
  complex reference by 5000.
- Mask type, seed, center fraction, noise, acceleration, and ACL15/ACL30 did not
  rescue the `0.005` baseline.
- Even the best scale gave only a marginal gain. The checkpoint therefore
  remains unvalidated, and cardiac underperformance cannot yet be attributed
  to domain shift.

The next isolated mismatch is spatial preprocessing: experiment 3 simulated MRI
at `640×320`, although its quantitative brain target is `320×320`.

### Experiment 4: crop complex brain reference before physics

Experiment ID: `fastmri-brain-crop320-acc8-smoke-004`

Status: prepared; run next on only slices 7, 8, and 9.

| Field | Recorded value |
|---|---|
| ESPIRiT input | `/mnt/qdata/rawdata/fastMRI/brain/multicoil_val_espirit/file_brain_AXFLAIR_200_6002462.h5` |
| Raw target input | `/mnt/qdata/rawdata/fastMRI/brain/multicoil_val/file_brain_AXFLAIR_200_6002462.h5` |
| Acquisition / reference | AXFLAIR / `reference_acl15`, map auto-selected |
| Cases | 3 slices: 7, 8, 9 |
| Source complex matrix | `640×320` |
| Complex preprocessing | Center-crop to `320×320` before normalization, FFT, mask generation, and MRI |
| Physics / measurement matrix | Official DeepInverse `MRI`; `(1,2,320,320)` |
| Mask | Random Cartesian, acceleration 8, center fraction 0.04, seed 0 |
| Normalization | Divide complex image by `0.0002` (multiply by 5000) |
| Noise | Gaussian σ `0.0005`, added through physics |
| Inference | Plain `model(y, physics)`; no post-RAM data consistency |
| Outputs | Per-slice panels, metrics JSON/CSV, reconstructions NPZ, environment and command record |
| Acceptance checks | Shapes, FFT error, adjoint error, operator norm, acceleration, mask orientation, PSNR, NMSE, SSIM, and images |
| Output path | `~/ram-results/fastmri-brain-crop320-acc8-smoke-004/` |

Run from an interactive GPU node:

```bash
cd ~/ram
git pull --ff-only
conda activate ~/envs/ram
set -o pipefail

python scripts/validate_fastmri_brain_ram.py \
  --input-h5 /mnt/qdata/rawdata/fastMRI/brain/multicoil_val_espirit/file_brain_AXFLAIR_200_6002462.h5 \
  --raw-h5 /mnt/qdata/rawdata/fastMRI/brain/multicoil_val/file_brain_AXFLAIR_200_6002462.h5 \
  --output-dir ~/ram-results/fastmri-brain-crop320-acc8-smoke-004 \
  --reference-key reference_acl15 \
  --map-index auto \
  --slices 7 8 9 \
  --crop-before-physics \
  --acceleration 8 \
  --center-fraction 0.04 \
  --mask-type random \
  --normalization-scale 0.0002 \
  --noise-sigma 0.0005 \
  --add-noise \
  --seed 0 \
  --device cuda 2>&1 | tee ~/ram-results/fastmri-brain-crop320-acc8-smoke-004-console.log && \
  mv ~/ram-results/fastmri-brain-crop320-acc8-smoke-004-console.log \
    ~/ram-results/fastmri-brain-crop320-acc8-smoke-004/run.log
```

Inspect before any larger run:

```bash
python -m json.tool \
  ~/ram-results/fastmri-brain-crop320-acc8-smoke-004/metrics.json

ls -lh ~/ram-results/fastmri-brain-crop320-acc8-smoke-004/*.png
```

Do not start a complete-volume or multivolume run until experiment 4 metrics and
images have been reviewed.

## Inventory command

The dataset details above were collected on the server with:

```bash
cd ~/ram
conda activate ~/envs/ram

python - <<'PY'
from pathlib import Path
import h5py

root = Path("/mnt/qdata/rawdata/fastMRI")
files = sorted(root.rglob("*.h5"))
print(f"Total H5 files: {len(files)}")

by_parent = {}
for path in files:
    by_parent.setdefault(path.parent, []).append(path)

for parent, paths in sorted(by_parent.items()):
    print(f"\nDATASET: {parent}")
    print(f"FILES:   {len(paths)}")
    for path in paths[:2]:
        print(f"  SAMPLE: {path.name}")
        with h5py.File(path, "r") as h5_file:
            print(f"    keys: {list(h5_file.keys())}")
            for key in h5_file.keys():
                obj = h5_file[key]
                if isinstance(obj, h5py.Dataset):
                    print(f"    {key}: shape={obj.shape}, dtype={obj.dtype}")
            print(f"    attrs: {dict(h5_file.attrs)}")
PY
```
