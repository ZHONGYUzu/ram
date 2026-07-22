# RAM CINE experiment record

Last updated: 2026-07-22

This document separates settings defined in the repository from facts verified by
run artifacts. Blank fields are intentionally left open. Fill them from the GPU
server's `~/ram/logs/` and `~/ram-results/` directories. A tracked job definition
does **not** by itself prove that the job was submitted or completed.

## Shared setup

| Field | Recorded value |
|---|---|
| Model/codebase | RAM |
| Dataset | CINE `Sub0008.h5` |
| Input path | `/mnt/qdata/rawdata/CINE/2D_h5_compressed/Sub0008.h5` |
| Sampling mask | `mask_VISTA_132x25_acc8_8.txt` |
| Mask shape / acceleration | `(phase, time) = (132, 25)` / acceleration 8 |
| Python | `/home/students/studxuzho1/envs/ram/bin/python` |
| Results root | `/home/students/studxuzho1/ram-results` |
| Device | CUDA |
| Batch size | 1 |
| GPU model |  |
| CUDA version |  |
| PyTorch version |  |
| Slurm partition |  |

## Run summary

| # | Experiment or family | Planned trials | Submitted | Completed | Job ID | Node | Result summary |
|---:|---|---:|---|---|---|---|---|
| 1 | Initial CINE inference | 1 | Yes | No (failed) |  |  | Failure recorded by Git commit `9e9466f`; error/details:  |
| 2 | `sub0008-acc8-slice06-alltimes` | 1 job / 25 frames |  |  |  |  |  |
| 3 | `sub0008-acc8-slice06-time10-p995` | 1 |  |  |  |  |  |
| 4 | `sub0008-acc8-slice06-time10-p995-dc*` | 3 |  |  |  |  |  |
| 5 | `sub0008-refine-*`, `sub0008-temporal-*`, `sub0008-grid-*` | 168 |  |  |  |  |  |
| 6 | `sub0008-strongdc-*` | 36 |  |  |  |  |  |
| 7 | `sub0008-projectdc-*` | 24 |  |  |  |  |  |

Known from the local repository: one initial inference failed. Completion of all
other rows is unverified until the corresponding server logs and result folders
are inspected.

## Detailed setups and results

### 1. Initial CINE inference

- Date/time:
- Experiment ID:
- Git commit:
- Slurm job ID:
- Node:
- Setup:
- Status: failed
- Error:
- Interpretation:
- Next action taken: added p99.5 normalization diagnostic
- Log path:

### 2. Acceleration-8, slice 6, all times

- Experiment ID: `sub0008-acc8-slice06-alltimes`
- Script: `scripts/slurm_sub0008_acc8_slice06.sbatch`
- Slice: 6 (zero-based)
- Times: all 25 frames
- Normalization: none/default
- Requested resources: 1 GPU, 8 CPUs, 64 GB RAM, 3 hours
- Date/time:
- Slurm job ID:
- Node:
- Git commit:
- Status:
- Runtime:
- Result quality:
- PSNR:
- NMSE:
- Visual observations:
- Error/notes:
- Log path:
- Result path: `/home/students/studxuzho1/ram-results/sub0008-acc8-slice06-alltimes/`

### 3. Single-frame p99.5 normalization diagnostic

- Experiment ID: `sub0008-acc8-slice06-time10-p995`
- Script: `scripts/slurm_sub0008_acc8_slice06_time10_p995.sbatch`
- Slice/time: 6/10 (zero-based)
- Normalization: p99.5 of masked zero-filled reconstruction; scale restored on output
- Requested resources: 1 GPU, 8 CPUs, 64 GB RAM, 30 minutes
- Date/time:
- Slurm job ID:
- Node:
- Git commit:
- Status:
- Runtime:
- Applied scale:
- RAM PSNR:
- RAM NMSE:
- Zero-filled PSNR: `19.9336 dB` (reference recorded in setup document)
- Zero-filled NMSE: `0.341558` (reference recorded in setup document)
- Visual observations:
- Error/notes:
- Log path:
- Result path: `/home/students/studxuzho1/ram-results/sub0008-acc8-slice06-time10-p995/`

### 4. Post-RAM data-consistency diagnostic sweep

- Experiment pattern: `sub0008-acc8-slice06-time10-p995-dc<gamma>`
- Script: `scripts/slurm_sub0008_acc8_slice06_time10_p995_dc_sweep.sbatch`
- Slice/time: 6/10
- Normalization: p99.5
- Gamma values: 0.1, 1, 10
- Planned trials: 3
- Requested resources per array task: 1 GPU, 8 CPUs, 64 GB RAM, 30 minutes
- Slurm array job ID:
- Nodes:

| Gamma | Status | Runtime | PSNR | NMSE | Notes |
|---:|---|---:|---:|---:|---|
| 0.1 |  |  |  |  |  |
| 1 |  |  |  |  |  |
| 10 |  |  |  |  |  |

- Best setting:
- Conclusion:
- Log paths:

### 5. Overnight proximal data-consistency campaign

- Script: `scripts/slurm_sub0008_overnight_dc.sbatch`
- Normalization: p99.5
- Planned trials: 168
- Requested resources: 1 GPU, 8 CPUs, 64 GB RAM, 12 hours
- Resume behavior: existing experiment directories are skipped
- Slurm job ID(s):
- Node(s):
- Submission/resubmission dates:
- Completed trial count:
- Failed trial count:

Subfamilies:

- Refinement: slice/time 6/10; gamma 3, 10, 30, 100; CG iterations 8, 16, 32; 12 trials.
- Temporal: slice 6; times 0, 3, 6, 9, 12, 15, 18, 21, 24; gamma 0, 10, 30, 100; CG 32; 36 trials.
- Spatial grid: slices 1, 3, 5, 7, 9, 11; times 2, 7, 12, 17, 22; gamma 0, 10, 30, 100; CG 32; 120 trials.

- Best gamma:
- Best CG iterations:
- Best PSNR/NMSE:
- Consistency across time:
- Consistency across slices:
- Conclusion:
- Metrics CSV:
- Log path(s):

### 6. Strong proximal data-consistency diagnostic

- Experiment pattern: `sub0008-strongdc-*`
- Script: `scripts/slurm_sub0008_strong_dc.sbatch`
- Frames `(slice:time)`: 6:10, 6:3, 6:6, 6:18, 1:7, 3:7
- Gamma values: 100, 300, 1000
- CG iterations: 8, 32
- Normalization: p99.5
- Planned trials: 36
- Requested resources: 1 GPU, 8 CPUs, 64 GB RAM, 4 hours
- Slurm job ID:
- Node:
- Completed trial count:
- Best setting:
- Best PSNR/NMSE:
- Conclusion:
- Metrics CSV:
- Log path:

### 7. Measured k-space projection diagnostic

- Experiment pattern: `sub0008-projectdc-*`
- Script: `scripts/slurm_sub0008_projection_dc.sbatch`
- Frames `(slice:time)`: 6:10, 6:3, 6:6, 6:18, 1:7, 3:7
- DC method: explicit measured-k-space projection
- Projection iterations: 1, 2, 4, 8
- Normalization: p99.5
- Planned trials: 24
- Requested resources: 1 GPU, 8 CPUs, 64 GB RAM, 3 hours
- Slurm job ID:
- Node:
- Completed trial count:
- Best iteration count:
- Residual before/after:
- Best PSNR/NMSE:
- Conclusion:
- Metrics CSV:
- Log path:

## Server audit commands

List scheduler logs:

```bash
find ~/ram/logs -type f \( -name '*.out' -o -name '*.err' -o -name '*.log' \) -print
```

Count experiment run logs:

```bash
find ~/ram-results -type f -name run.log | wc -l
```

List experiment IDs with run logs:

```bash
find ~/ram-results -type f -name run.log -print | sort
```

Locate completed and failed exits in scheduler logs:

```bash
rg -n 'Exit status:|Finished:|Error|Traceback|FAILED|Killed' ~/ram/logs ~/ram-results
```

## Overall conclusion

- Total submitted experiments:
- Total completed experiments:
- Total failed experiments:
- Best configuration:
- Best quantitative result:
- Main qualitative finding:
- Remaining uncertainty:
- Recommended next experiment:
