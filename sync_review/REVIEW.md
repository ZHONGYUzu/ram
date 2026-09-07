# 人工审核清单

这是原始代码归档，不是已合并或验证过的新版运行代码。所有不同版本均保留原路径，不自动选择或覆盖。

## 与仓库版本不同

| 服务器文件 | 归档文件 | 仓库候选文件 | diff | 状态 |
|---|---|---|---|---|
| `/home/students/studxuzho1/ram-fastmri-brain-adapter/runs/ram-022/code/scripts/validate_fastmri_multicoil_ram.py` | `runs/ram-022/code/scripts/validate_fastmri_multicoil_ram.py` | `scripts/validate_fastmri_multicoil_ram.py` | 已删除辅助 diff；原脚本保留 | 已审核：保留 runs 版本；根目录同名脚本暂缓退役：已发现早期实验引用 |
| `/home/students/studxuzho1/ram-fastmri-brain-adapter/runs/ram-023/code/scripts/validate_fastmri_multicoil_ram.py` | `runs/ram-023/code/scripts/validate_fastmri_multicoil_ram.py` | `scripts/validate_fastmri_multicoil_ram.py` | 已删除辅助 diff；原脚本保留 | 保留实验快照；不逐份重复审核，不自动合并 |
| `/home/students/studxuzho1/ram-fastmri-brain-adapter/runs/ram-024/code/scripts/validate_fastmri_multicoil_ram.py` | `runs/ram-024/code/scripts/validate_fastmri_multicoil_ram.py` | `scripts/validate_fastmri_multicoil_ram.py` | 已删除辅助 diff；原脚本保留 | 保留实验快照；不逐份重复审核，不自动合并 |
| `/home/students/studxuzho1/ram-fastmri-brain-adapter/runs/ram-025/code/scripts/validate_fastmri_multicoil_ram.py` | `runs/ram-025/code/scripts/validate_fastmri_multicoil_ram.py` | `scripts/validate_fastmri_multicoil_ram.py` | 已删除辅助 diff；原脚本保留 | 保留实验快照；不逐份重复审核，不自动合并 |
| `/home/students/studxuzho1/ram-fastmri-brain-adapter/runs/ram-026/code/scripts/validate_fastmri_multicoil_ram.py` | `runs/ram-026/code/scripts/validate_fastmri_multicoil_ram.py` | `scripts/validate_fastmri_multicoil_ram.py` | 已删除辅助 diff；原脚本保留 | 保留实验快照；不逐份重复审核，不自动合并 |
| `/home/students/studxuzho1/ram-fastmri-brain-adapter/runs/ram-029/code/scripts/validate_fastmri_multicoil_ram.py` | `runs/ram-029/code/scripts/validate_fastmri_multicoil_ram.py` | `scripts/validate_fastmri_multicoil_ram.py` | 已删除辅助 diff；原脚本保留 | 保留实验快照；不逐份重复审核，不自动合并 |
| `/home/students/studxuzho1/ram-fastmri-brain-adapter/runs/ram-028/code/scripts/validate_fastmri_multicoil_ram.py` | `runs/ram-028/code/scripts/validate_fastmri_multicoil_ram.py` | `scripts/validate_fastmri_multicoil_ram.py` | 已删除辅助 diff；原脚本保留 | 保留实验快照；不逐份重复审核，不自动合并 |
| `/home/students/studxuzho1/ram-fastmri-brain-adapter/runs/ram-028/code/scripts/validate_fastmri_ram.py` | `runs/ram-028/code/scripts/validate_fastmri_ram.py` | `scripts/validate_fastmri_ram.py` | 已删除辅助 diff；原脚本保留 | 保留实验快照；不逐份重复审核，不自动合并 |

## 本机与服务器不同

- 本机 `/Users/zhongyu/Desktop/ram/SERVER_WORKFLOW.md`；服务器 `/home/students/studxuzho1/ram-fastmri-brain-adapter/SERVER_WORKFLOW.md`。双方保留，待人工审核。
- 本机 `/Users/zhongyu/Desktop/ram/scripts/infer_cine_h5.py`；服务器 `/home/students/studxuzho1/ram-fastmri-brain-adapter/scripts/infer_cine_h5.py`。双方保留，待人工审核。
- 本机 `/Users/zhongyu/Desktop/ram/scripts/run_cine_experiment.sh`；服务器 `/home/students/studxuzho1/ram-fastmri-brain-adapter/scripts/run_cine_experiment.sh`。双方保留，待人工审核。
- 本机 `/Users/zhongyu/Desktop/ram/scripts/validate_fastmri_multicoil_ram.py`；服务器 `/home/students/studxuzho1/ram-fastmri-brain-adapter/scripts/validate_fastmri_multicoil_ram.py`。双方保留，待人工审核。
- 本机 `/Users/zhongyu/Desktop/ram/scripts/validate_fastmri_ram.py`；服务器 `/home/students/studxuzho1/ram-fastmri-brain-adapter/scripts/validate_fastmri_ram.py`。双方保留，待人工审核。

## 检查范围

仅执行 Python 语法解析和 bash -n；未运行模型、数据或 GPU 测试。语法问题见 syntax-issues.json，入口兼容性待查项见 launcher-review.json。归档不代表对科学正确性的认可。

## 实验对应关系

run-index.json 按目录和脚本中的实验编号建立配置、代码、启动脚本、服务器日志路径索引；未把静态关联当作实际执行证据。

## 同名代码的多个内容版本

不同内容不自动合并；相同哈希可合并审阅，但保留实验归属。

### `scripts/validate_fastmri_multicoil_ram.py`

| 归档路径 | SHA-256 前 16 位 |
|---|---|
| `scripts/validate_fastmri_multicoil_ram.py` | `ee875f4fe20bb2c5` |
| `runs/ram-022/code/scripts/validate_fastmri_multicoil_ram.py` | `cd1d561049f1d525` |
| `runs/ram-023/code/scripts/validate_fastmri_multicoil_ram.py` | `bb8d1e99a1036ed5` |
| `runs/ram-024/code/scripts/validate_fastmri_multicoil_ram.py` | `54737001f8a65dcb` |
| `runs/ram-025/code/scripts/validate_fastmri_multicoil_ram.py` | `54737001f8a65dcb` |
| `runs/ram-026/code/scripts/validate_fastmri_multicoil_ram.py` | `9ad696b6952aea2c` |
| `runs/ram-029/code/scripts/validate_fastmri_multicoil_ram.py` | `54737001f8a65dcb` |
| `runs/ram-028/code/scripts/validate_fastmri_multicoil_ram.py` | `286ac1b18b7d50ac` |

### `scripts/validate_fastmri_ram.py`

| 归档路径 | SHA-256 前 16 位 |
|---|---|
| `scripts/validate_fastmri_ram.py` | `d38dc6bbffb98638` |
| `runs/ram-022/code/scripts/validate_fastmri_ram.py` | `d38dc6bbffb98638` |
| `runs/ram-023/code/scripts/validate_fastmri_ram.py` | `d38dc6bbffb98638` |
| `runs/ram-024/code/scripts/validate_fastmri_ram.py` | `d38dc6bbffb98638` |
| `runs/ram-025/code/scripts/validate_fastmri_ram.py` | `d38dc6bbffb98638` |
| `runs/ram-026/code/scripts/validate_fastmri_ram.py` | `d38dc6bbffb98638` |
| `runs/ram-029/code/scripts/validate_fastmri_ram.py` | `d38dc6bbffb98638` |
| `runs/ram-028/code/scripts/validate_fastmri_ram.py` | `3fc035d5a1694759` |

### `scripts/validate_ram029_text.py`

| 归档路径 | SHA-256 前 16 位 |
|---|---|
| `runs/ram-029/code/scripts/validate_ram029_text.py` | `1bfd963d1e4eaf3a` |
| `runs/ram-029/attempts/attempt-1-bbox-270-246/design_snapshot/validate_ram029_text.py` | `0d01273bc8f0e464` |

### `scripts/aggregate_ram029.py`

| 归档路径 | SHA-256 前 16 位 |
|---|---|
| `runs/ram-029/code/scripts/aggregate_ram029.py` | `54fdcf66e36044d9` |
| `runs/ram-029/attempts/attempt-1-bbox-270-246/design_snapshot/aggregate_ram029.py` | `16bb301760bb0141` |

### `scripts/ram-029-15cases.sbatch`

| 归档路径 | SHA-256 前 16 位 |
|---|---|
| `runs/ram-029/attempts/attempt-1-bbox-270-246/design_snapshot/ram-029-15cases.sbatch` | `34dc553e99bb22bc` |
| `runs/ram-029/ram-029-15cases.sbatch` | `e76fb991827c01b9` |

## 本机差异的可读对比

- `SERVER_WORKFLOW.md`：[查看差异](local-diffs/SERVER_WORKFLOW.md.diff)。
- `scripts/infer_cine_h5.py`：[查看差异](local-diffs/scripts/infer_cine_h5.py.diff)。
- `scripts/run_cine_experiment.sh`：[查看差异](local-diffs/scripts/run_cine_experiment.sh.diff)。
- `scripts/validate_fastmri_multicoil_ram.py`：[查看差异](local-diffs/scripts/validate_fastmri_multicoil_ram.py.diff)。
- `scripts/validate_fastmri_ram.py`：[查看差异](local-diffs/scripts/validate_fastmri_ram.py.diff)。

本机 4 张 imgs/example_*.png 的删除未同步；它们仍保留在同步仓库的基线提交中。本机未跟踪代码与报告亦未混入本次服务器归档，备份清单留在本机 sync-staging/2026-09-07/local-preserved.json。

## ram-022 保留决定

用户确认保留 `runs/ram-022/code/scripts/validate_fastmri_multicoil_ram.py`；根目录 `scripts/validate_fastmri_multicoil_ram.py` 原标记待退役；复查发现历史引用，现暂缓退役。删除前必须核对其他实验和入口引用，本次不删除、不改运行代码，不扩大到整个 scripts/ 目录。后续实验快照按用户决定保留，不自动合并。

## 实验 diff 清理决定

用户决定保留已归档实验快照，删除本审核目录的实验对比 diff。代码正确性未因此视为已验证；原脚本、配置、历史版本和哈希清单保留，必要时可重新生成差异。本机与服务器尚未解决的冲突仍保留在 local-diffs/，初始 audit/ 调查资料未改动。


## 根目录多线圈脚本引用复查：暂缓退役

服务器 ram-results 中找到 9 个历史 command.txt 使用 `scripts/validate_fastmri_multicoil_ram.py`；错误栈另直接指向服务器 ram/scripts/ 下的此文件。此前未发现调用的搜索未覆盖这些结果目录，不能作为删除依据。两类文件都先保留，待历史复现依赖审核；未删除代码。


## 早期实验分组

已按指标与运行记录分开标记失败项，未把九个实验整体判为失败。见 [分组及指标依据](HISTORICAL_EXPERIMENTS.md)。此记录不授权删除或重跑。
