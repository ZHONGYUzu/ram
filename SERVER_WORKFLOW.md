# RAM + Slurm 服务器工作流

本文件只适用于 RAM 项目，尤其是 `feature/fastmri-brain-adapter`。
NV-Raw2Insights-MRI 使用其自己仓库内的工作流文件，两者的仓库、Python
环境、数据、运行脚本和结果目录不得混用。

## 当前环境

```text
SSH 入口:       ssh login
计算节点:       slurm_debug（交互检查）或 sbatch（作业）
本机仓库:       /Users/zhongyu/Desktop/ram
服务器主仓库:   /home/students/studxuzho1/ram
服务器 worktree:/home/students/studxuzho1/ram-fastmri-brain-adapter
Git 分支:       feature/fastmri-brain-adapter
Python:         /home/students/studxuzho1/envs/ram/bin/python
fastMRI brain:  /mnt/qdata/rawdata/fastMRI/brain/multicoil_val
服务器结果根:   /home/students/studxuzho1/ram-results
```

当前 RAM checkpoint 已存在于服务器缓存。实验必须使用已有的本地 checkpoint，
并保持 Hugging Face 离线模式；不要下载模型、数据或依赖。

## 职责划分

- 本机：修改、检查、commit 并 push RAM 代码。
- GitHub：同步 `feature/fastmri-brain-adapter` 分支的代码和 Slurm 脚本。
- Login 节点：Git 同步、只读路径/日志检查，以及 `sbatch`、`squeue`、
  `sacct`、`scancel` 等调度操作。
- 计算节点：Python、DeepInverse、数据检查、VCC/ESC、推理、评价和绘图。
- 服务器存储：fastMRI 数据、checkpoint、日志、指标和 preview；不上传 GitHub。

禁止在 Login 节点直接运行计算型 Python。

## 本机同步代码

```bash
cd /Users/zhongyu/Desktop/ram
git switch feature/fastmri-brain-adapter
git status --short --branch
git push origin feature/fastmri-brain-adapter
```

只提交与当前 feature 有关的文件。不要把其他未提交的 cine、报告或图片改动
带入 adapter commit。

## Login 节点同步 worktree

```bash
ssh login
cd /home/students/studxuzho1/ram-fastmri-brain-adapter
git pull --ff-only
git status --short --branch
git log -1 --oneline
```

服务器实验目录 `runs/` 是服务器本地产物，应通过 worktree 的
`.git/info/exclude` 忽略，不提交 GitHub：

```bash
exclude_file="$(git rev-parse --git-path info/exclude)"
grep -qxF '/runs/' "$exclude_file" || echo '/runs/' >> "$exclude_file"
```

## 计算节点检查

交互检查时先从 Login 节点申请计算节点：

```bash
slurm_debug
hostname
nvidia-smi
cd /home/students/studxuzho1/ram-fastmri-brain-adapter
source /home/students/studxuzho1/envs/ram/bin/activate
```

只有进入计算节点后，才可运行 Python 环境、DeepInverse、数据 shape 或模型检查。

## 当前 fastMRI brain debug

目标配置：

```text
模型实现:       deepinv.models.RAM
DeepInverse:    >= 0.4.1
数据:           fastMRI brain multicoil validation
线圈处理:       volume-level ESC/VCC -> complex single-coil
采样:           random Cartesian mask
加速:           4x
center fraction:0.08
scaling:        固定除以 0.005
noise sigma:    0.0005
debug 范围:     1 case、3 slices
下载:           禁止
```

提交 debug 作业：

```bash
cd /home/students/studxuzho1/ram-fastmri-brain-adapter
ACCELERATION=4 MAX_VOLUMES=1 \
  sbatch scripts/slurm_fastmri_brain_adapter.sbatch
```

## 当前服务器实验目录

```text
runs/ram-001/
├── config.yaml
├── cases.txt
├── output/
│   ├── slice_metrics.csv
│   ├── summary.json
│   └── previews/
└── logs/
    ├── run.log
    ├── slurm-<job_id>.out
    └── slurm-<job_id>.err
```

目前 Slurm 脚本的 debug 结果仍写到仓库外的：

```text
/home/students/studxuzho1/ram-results/
  fastmri-brain-adapter-deepinv041-r4-v1/
```

失败时可能留下：

```text
/home/students/studxuzho1/ram-results/
  fastmri-brain-adapter-deepinv041-r4-v1.run.log
```

不要把 `runs/ram-001/output/` 为空误判为没有运行；应同时核对 Slurm 状态、
Slurm 日志和上面的实际结果目录。正式实验前再统一输出路径。

## 查看任务与结果

在 Login 节点：

```bash
squeue -u "$USER"
sacct -S 2026-08-11 -u "$USER" \
  --name=ram_fmri_adapter \
  --format=JobID,JobName,State,ExitCode,Elapsed
```

查看指定作业（将数字替换为实际 Job ID，不输入尖括号）：

```bash
cat runs/ram-001/logs/slurm-42440.err
tail -n 150 runs/ram-001/logs/slurm-42440.out
```

成功标准：

```text
State:    COMPLETED
ExitCode: 0:0
stderr:   空
```

成功结果至少应包含：

```text
environment.json
slice_metrics.csv
summary.json
preview-*.png
run.log
```

## 与 NV-Raw2Insights-MRI 隔离

RAM 工作中不要使用以下 NV 项目路径：

```text
/Users/zhongyu/Documents/NV-Raw2insights-MRI
/home/students/studxuzho1/NV-Raw2insights-MRI
/home/students/studxuzho1/envs/nv-raw2insights-mri
```

同样，不要将 RAM 的 fastMRI 数据路径、RAM checkpoint、RAM Slurm 脚本或
`ram-results` 写入 NV 项目的配置和工作流文件。
