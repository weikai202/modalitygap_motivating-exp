# MMSD2.0 / MM-IMDb 完整实验交接

这是完整研究的可运行入口。正式训练使用 `symmetric-attention-v2`，不是历史 mean-noise 协议。已有小规模结果不能代替本次完整实验；仓库不包含原始数据、模型权重或训练断点。

## 实验规模

| 版本 | 训练数据 | 开发数据 | 矩阵 |
|---|---|---|---|
| 协议完整版 | 固定约6000条训练池，审计后抽样 | 全部保留的官方开发样本 | 2数据集 × 6配置 × 3 seed = 36个八轮作业 |
| 全部训练样本版 | 全部审计后训练样本 | 全部保留的官方开发样本 | 相同36作业，独立workspace |

先用 seed42 筛选12个作业，再追加 seed43/44 确认；也可首次直接启动36作业。每次训练从原始Qwen模型开始；每个作业评估epoch0和epoch1–8，每轮full/遮图/遮文三条件，共27次。36作业共972次开发集条件评估，不包括资源pilot和显式锁定test。

六配置为 L-low/L-high/M-low/M-high/LM-low/LM-high，学习率2e-5/1e-4；LLM LoRA、visual merger、两者联合。8轮schedule从开始即固定，早期checkpoint是同一训练的前缀。

## 1. 新机器安装

在仓库根目录执行。需要Linux、Python3.12、支持BF16的NVIDIA GPU。3090可用于代码/资源验证，更好的GPU可用于正式研究。每个workspace顺序使用一张GPU；多个独立workspace可通过`CUDA_VISIBLE_DEVICES`分配到不同GPU。

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-cu128.lock -r requirements-data.lock
.venv/bin/python -m pip install --no-deps -e .
nvidia-smi
```

`requirements-data.lock`只增加读取MMSD原图parquet所需的pyarrow；使用本地作者图片导出时不需要它。代码没有训练额外文本encoder。

## 2. 下载和完整导出

```bash
.venv/bin/python scripts/download_full_data.py --root data/full-inputs
.venv/bin/python scripts/prepare_full_experiment.py \
  --inputs data/full-inputs/inputs.json \
  --output data/full-export-v1 \
  --workspace artifacts/full-study-v1
```

下载脚本取得固定作者提交中的MMSD2.0修正标注，及固定公开镜像的原始图片字节；图片镜像标签始终不作为真值。MM-IMDb使用作者原始海报/JSON压缩包，必须通过发布的MD5 `38e154264b3771f70840c6cb5f170a93`。下载有缓存和分段恢复；已有不匹配文件不会被悄悄替换。

导出完整官方train/dev/test，不在导出时抽pilot。图片原始字节保留，JSONL中的图片路径相对其所在目录，便于整体搬迁。数据导出目录已存在时拒绝覆盖；中断的导出保留原目录，选择新版本重试。

- MMSD2.0完整标注：train19816/dev2410/test2409。原始输入必须是`text_json_final`，保留作者正文和0/1语义。
- MM-IMDb完整标注：train15552/dev2608/test7799。剧情取发布顺序中第一个非空条目，不拼电影标题/年份/ID。原始JSON中的额外标签投影到固定23类，并在provenance记录；全25959部电影的投影已在本地对照官方HDF5标签和划分验证。
- 导出数量不等于最后训练数量：缺图、坏图、空正文和重复组按下述规则审计，实际计数见manifest。

如果已有作者原始图片/标注，可绕过镜像下载，直接运行：

```bash
.venv/bin/python -m modalitygap.full_data mmsd2 \
  --annotations /path/to/MMSD2.0/data/text_json_final \
  --images /path/to/original/images --output data/manual-mmsd2-v1
.venv/bin/python -m modalitygap.full_data mmimdb \
  --archive /path/to/mmimdb.tar.gz --output data/manual-mmimdb-v1
```

手动导出时复制`configs/full-screen.example.json`和`full-confirm.example.json`，修改两者的`datasets.*.source`及`source_revision`（取导出provenance），保持其余科学设置和workspace一致。

## 3. 训练前审计和资源pilot

生成的`screen.json`与`confirm.json`含本机路径，迁移到另一台机器后应在新机器生成配置或更新路径；原图和JSONL可整体拷贝。配置路径相对配置文件解释。

```bash
bash scripts/run_experiment.sh --config data/full-export-v1/screen.json --dry-run
bash scripts/run_experiment.sh --config data/full-export-v1/screen.json --stage prepare
bash scripts/run_experiment.sh --config data/full-export-v1/screen.json --stage pilot --resume
```

审计会完整解码图像、核对ID/标签/划分，按decoded RGB及归一化正文建立重复连通组。完整配置使用预先固定的`prefer-heldout`规则：跨划分重复组只保留优先级最高的原划分（test > dev > train），不移动、不重标记，所有移除写入`overlap_excluded`。这是训练前内容重复审计，不使用test模型表现。若希望遇到重复就停止，将`overlap_policy`改为`error`并使用新workspace。

约6000训练池使用固定seed42单标签分层或多标签迭代分层；多标签算法允许近似数量。所有配置/seed共用同一冻结池。开发集不抽样，仅应用前述缺失/重复规则。

资源pilot每个数据集分别运行L/M/LM范围，只用至多64 train/32 dev；冻结均值从正式训练池中固定抽256例计算。检查pilot的显存、梯度、非法率、长度和OCR审核清单。单配置遇到OOM时不要单独降低分辨率，统一修订协议并换新workspace。

## 4. 运行12作业筛选，再扩成36作业

```bash
# 首次已经做过prepare/pilot，所以追加 --resume
bash scripts/run_experiment.sh --config data/full-export-v1/screen.json --resume

# 相同数据池/workspace，跳过seed42，补齐seed43和44
bash scripts/run_experiment.sh --config data/full-export-v1/confirm.json --resume
```

若尚未初始化workspace，直接使用`confirm.json`且去掉`--resume`，就是一次性启动36作业。输入数据和模型下载均与训练入口分开；缓存齐全后可设置`HF_HUB_OFFLINE=1`。

使用全部训练样本时，首次导出将`prepare_full_experiment.py`加上`--pool-size all`，并选择不同的output/workspace；或使用`configs/full-all-train.example.json`。运行后禁止把6000改成all来复用已有workspace。

正式完整开发集的生成评估成本很高，不能用32条pilot的总耗时直接推算。运行前准备充足磁盘；所有epoch0–8断点保留，中途step断点只保留最近2个。可检查计划、进度及pilot估算的断点空间：

```bash
.venv/bin/python scripts/full_experiment_status.py --config data/full-export-v1/confirm.json
```

## 5. SSH断连和精确恢复

```bash
tmux new -s modalitygap-full
bash scripts/run_experiment.sh --config data/full-export-v1/confirm.json --resume
# Ctrl-b d 脱离；重新SSH后：
tmux attach -t modalitygap-full
```

进程确实结束时，重新运行同一命令并加`--resume`，自动从最近完整optimizer边界续训。恢复参数、优化器、8轮scheduler、样本游标和Python/NumPy/Torch/CUDA随机状态，已完成作业核验后跳过。GPU/核心软件或源码改变时精确恢复会拒绝，不要绕过签名；历史pilot恢复应使用其保存的源码快照。

## 6. 单独锁定test，避免开发时提前使用

训练、筛选、确认和分析阶段都不自动运行test。选定比较后显式锁定，例如预先规定的fallback三配置：

```bash
.venv/bin/python scripts/lock_heldout_selection.py \
  --config data/full-export-v1/confirm.json \
  --configs L-low L-high LM-low --seeds 42 43 44 \
  --output selections/full-study-v1
```

该命令检查所选作业完整，写每数据集的锁定列表，只含epoch2/8，并打印两条可直接执行的test命令。检查锁定列表后执行打印的命令。选择目录拒绝覆盖，test输出也拒绝覆盖。test中断时保留部分输出，使用新的`--output .../heldout-v2`重新评估；无需重新训练。

## 7. 输出与验证

```text
artifacts/full-study-v1/
  pipeline.json                    # 科学设置、数据及源码签名
  manifests/<dataset>/             # train/dev/test、审计、排除及分组
  stats/<dataset>.pt               # 原模型训练均值；跨配置固定
  pilots/<dataset>/<scope>/...     # 独立资源预检
  runs/<dataset>/<config>/seed_N/
    config.json
    metrics.csv                    # epoch0..8 × full/image-mask/text-mask
    contributions.csv              # C_image、C_text
    predictions/                   # 原回答与生成token
    interventions/                 # 实际key屏蔽核验
    resources.json                 # loss、显存、训练耗时
    checkpoints/epoch_0..8/         # 参数、optimizer、scheduler、RNG等
    checkpoints/step_*/             # 最近两个中途完整断点
  figures/<dataset>/               # 贡献曲线、配对bootstrap、排名分析
  logs/                            # 每阶段日志；可恢复追加
  heldout-v1/                      # 仅在显式执行test命令后产生
```

当前变更的具体测试及真实全数据审计结果见[交接验证报告](full-experiment-validation.md)。GPU测试包含小Qwen工程模型跑通36作业及八轮调度、三seed分析、held-out加载，另有真实3B模型的既有pilot证据。工程测试不是36个正式研究结果。

```bash
# 新机器重跑回归，使用实际缓存processor路径
OMP_NUM_THREADS=1 MODALITYGAP_GPU_TESTS=1 \
MODALITYGAP_PROCESSOR=artifacts/hf-cache/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3 \
.venv/bin/python -m unittest discover -s tests -v
```
