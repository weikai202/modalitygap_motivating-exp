# Modality Gap：实验运行与断点续训

使用 **Qwen2.5-VL-3B-Instruct**，分别在 **MMSD2.0** 和 **MM-IMDb** 上研究早期/后期排名反转及模态依赖变化。训练协议见 [experiment-plan.md](docs/experiment-plan.md)。

本项目已在 RTX 3090 上验证训练、消融、检查点和精确续训。验证包含合成工程样例及真实数据小子集；完整全量数据集的研究结论仍需独立实验。

## 完整实验交接入口

完整数据导出、6000训练池/全部训练两种规模、12作业筛选与36作业确认、资源预检、断点恢复和锁定test命令，见 [完整实验运行手册](docs/full-experiment.md)。新实验统一使用 `symmetric-attention-v2`。下面的早期验证报告是历史小规模证据，不能替代完整实验结果。

## 1. 首次运行前准备什么

推荐环境：**Linux x86_64、Python 3.12、可用的 NVIDIA GPU**。锁定环境使用 PyTorch 2.8.0 / torchvision 0.23.0 / CUDA 12.8。先在实际启动训练的终端确认：

```bash
python3 --version
nvidia-smi
```

一键脚本会安装 Python 依赖、下载固定版本的模型、准备数据清单、估计消融统计量、运行 pilot、训练和绘图。**原始数据集和图片需要提前取得并导出；脚本不会自动下载作者数据或用假数据填充。**

### 输入数据格式

每个数据集准备一个 UTF-8 JSONL 文件，每行一个样本，必须包含作者发布的 `train`、`dev`、`test` 划分。例如 MMSD2.0：

```json
{"id":"post-001","split":"train","image":"images/post-001.jpg","text":"作者清洗后的原始正文","labels":["sarcastic"]}
{"id":"post-002","split":"dev","image":"images/post-002.jpg","text":"另一条正文","labels":["not_sarcastic"]}
{"id":"post-003","split":"test","image":"images/post-003.jpg","text":"测试正文","labels":["sarcastic"]}
```

MM-IMDb 的行格式相同，`text` 为剧情，`labels` 是多标签数组，例如 `["Comedy", "Drama"]`。

- `image` 可以是绝对路径，或**相对于这个 JSONL 文件**的路径。
- `id` 必须稳定且唯一；`split` 只能是 `train`、`dev`、`test`。
- MMSD2.0 必须使用作者修正后的 `data/text_json_final` 标注和对应原始图片，标签按语义转换成 `sarcastic` / `not_sarcastic`，保留官方清洗和划分。
- MM-IMDb 使用原始海报、剧情和作者划分。多个剧情时，固定取发布顺序中第一个非空剧情，并在 `source_revision` 中记录。不要把电影标题、年份、IMDb ID 或文件名额外拼进正文。
- 训练 JSONL 的 genre 名称需与 [protocol.py](src/modalitygap/protocol.py) 的 23 个标签完全一致。准备阶段会按固定顺序排列。
- 缺图、损坏图片、空正文会被排除并计数；重复 ID、跨划分的精确重复图片/文本会让准备过程停止，需要核对来源并记录处理决定。
- 默认从训练划分固定抽取约 6000 条；不足时使用实际完整样本。开发集保持完整，测试集不参与筛选。
- 原始 JSONL 和图片放在生成的 `workspace` **外部**，例如仓库的 `data/` 目录。运行后保留原文件，不要移动图片或改写冻结清单。

## 2. 修改一个配置文件

在仓库根目录执行：

```bash
cp configs/experiment.example.json configs/experiment.json
```

编辑 `configs/experiment.json`：

```json
{
  "workspace": "../artifacts/study-v1",
  "cache_dir": "../artifacts/hf-cache",
  "model_revision": "66285546d2b821cf421d4f5eb2576359d3770cd3",
  "datasets": {
    "mmsd2": {
      "source": "../data/mmsd2-source.jsonl",
      "source_revision": "实际作者提交版本，以及你的导出规则"
    },
    "mmimdb": {
      "source": "../data/mmimdb-source.jsonl",
      "source_revision": "实际作者发布版本；按发布顺序取第一个非空剧情"
    }
  },
  "configs": ["L-low", "L-high", "M-low", "M-high", "LM-low", "LM-high"],
  "seeds": [42],
  "masking_method": "symmetric-attention-v2",
  "pool_size": 6000,
  "overlap_policy": "error",
  "keep_step_checkpoints": 2,
  "stats_samples": 256,
  "pilot_steps": 4,
  "checkpoint_every_steps": 50
}
```

**配置里的所有路径都相对于配置文件所在目录解析。** 因此示例中的 `../data/` 指仓库根目录的 `data/`，不是相对于 shell 当前目录。

| 字段 | 含义 |
|---|---|
| `workspace` | 本次实验全部生成结果的根目录；首次运行使用空目录/新路径 |
| `cache_dir` | 固定模型的 Hugging Face 缓存，可供多个实验共享 |
| `model_revision` | 完整模型 commit SHA，模型、processor、tokenizer 使用同一版本 |
| `datasets` | 运行的数据集、原始导出文件和可追踪的来源版本；只跑一个可删除另一个条目 |
| `configs` | 训练候选；完整筛选保留全部六个，调试可显式选子集 |
| `seeds` | 首轮 `[42]`；完整确认可改为 `[42,43,44]` 后用 `--resume` 补跑 |
| `stats_samples` | 从训练池估计固定消融统计量的样本数；首次冻结后不能修改 |
| `pilot_steps` | 三种适配范围各自试跑的最大 optimizer step 数，默认 4，允许 4–100 |
| `checkpoint_every_steps` | 每 N 个 optimizer step 额外保存；设为 0 时只保留每个 epoch 的检查点 |

L / M / LM 分别表示语言 LoRA、视觉 merger、二者联合；low / high 学习率分别为 `2e-5` / `1e-4`。全部配置使用同一数据池、输入预算和八轮训练计划。

## 3. 一键启动

先只打印执行计划。此命令不下载、不访问 GPU、不写实验结果：

```bash
bash scripts/run_experiment.sh --config configs/experiment.json --dry-run
```

首次安装并运行：

```bash
bash scripts/run_experiment.sh --setup --config configs/experiment.json
```

如果默认 `python3` 不是 3.12，但机器装有 `python3.12`：

```bash
MODALITYGAP_BOOTSTRAP_PYTHON=python3.12 \
  bash scripts/run_experiment.sh --setup --config configs/experiment.json
```

`--setup` 创建/使用仓库根目录的 `.venv`，按 `requirements-cu128.lock` 安装依赖，再安装本地包。之后启动不必再加 `--setup`。安装需要网络和 Linux `flock`；`--setup --dry-run` 不会执行安装。

默认 `--stage all` 按顺序执行：

```text
检查配置、GPU和目录锁
  → 冻结数据清单
  → 下载/复用固定模型
  → 用原始冻结模型估计并保存消融统计量
  → 每个数据集分别运行 L-low、M-low、LM-low 的资源 pilot
  → 顺序运行所选数据集 × 配置 × seed，各完整训练八轮
  → 2000 次配对 bootstrap、排名反转分析和绘图
```

默认配置为 **12 个完整训练作业**：2 个数据集 × 6 个候选 × seed 42。确认三个 seed 时为 36 个作业。每个正式作业评估 epoch 0–8，包含 27 次主条件评估，当前对称协议不增加噪声敏感性评估；历史 mean-noise 协议另有18次。pilot 输出单独存放。

pilot 是接口/显存/梯度检查；请查看其结果及 `ocr_audit.jsonl` 后再使用正式结果。脚本不会按表现提前停止、自动降低某个配置的分辨率或改用量化。需要人工审查后再启动训练矩阵时，使用下一节的分阶段命令。

### 分阶段执行

| `--stage` | 执行内容 | 需要 GPU |
|---|---|---|
| `all`（默认） | 数据、统计量、pilot、训练矩阵、分析 | 是 |
| `prepare` | 仅冻结数据清单 | 否 |
| `stats` | 准备/复用清单和模型，冻结统计量 | 是 |
| `pilot` | 准备/复用前置产物，运行三种范围 pilot | 是 |
| `screen` | 验证已有 pilot，再训练矩阵；不自动绘图 | 是 |
| `analyze` | 仅分析已有完整作业、生成图表 | 否 |

例如先人工检查 pilot，再训练：

```bash
bash scripts/run_experiment.sh --setup --config configs/experiment.json --stage pilot
# 查看 pilot 的 resources.json、metrics.csv、干预日志，并审核 OCR 样本
bash scripts/run_experiment.sh --config configs/experiment.json --stage screen --resume
bash scripts/run_experiment.sh --config configs/experiment.json --stage analyze --resume
```

模型已完整缓存时，后续模型推理/训练使用离线缓存；不会每次重连都重新下载。如果主动设置 `HF_HUB_OFFLINE=1`，请确保模型已经下载完整。

## 4. SSH 断连、重新连接和断点续训

### 只是 SSH 断开，进程仍在运行

建议在 `tmux` 中启动（机器需已有 tmux）：

```bash
tmux new -s modalitygap
bash scripts/run_experiment.sh --config configs/experiment.json --resume
```

按 `Ctrl-b` 再按 `d` 脱离会话。重新 SSH 后连接：

```bash
tmux attach -t modalitygap
```

也可以使用 nohup：

```bash
mkdir -p logs
nohup bash scripts/run_experiment.sh --config configs/experiment.json --resume \
  > logs/study-v1-console.log 2>&1 &
echo $! > logs/study-v1-launcher.pid
tail -f logs/study-v1-console.log
```

这时 SSH 断连不等于训练停止；先查看日志和 `nvidia-smi`。**不要重复启动正在运行的作业，也不要删除 `.pipeline.lock` 强行解锁。** 锁会在所有持有它的进程退出后自动释放，训练子进程也持有该锁。

### 进程确实停止、机器重启或 CUDA/OOM 报错

保持同一个配置、workspace、数据、统计量、训练核心源代码及兼容的运行环境，重新执行：

```bash
bash scripts/run_experiment.sh --config configs/experiment.json --resume
```

脚本会：

1. 验证冻结的数据、统计量和模型版本，不重新抽训练池。
2. 跳过已验证完成的 pilot 和完整训练作业。
3. 对未完成作业调用 `--resume-from latest`，从最近完整的 optimizer step/epoch 检查点恢复。
4. 继续余下矩阵，所有作业完成后再绘图。

恢复包括可训练参数、AdamW 状态、八轮 scheduler、样本游标、Python/NumPy/Torch CPU/CUDA RNG 和训练历史。半个梯度累积组不会被保存；未保存的计算会从上个检查点重放。中断在评估期间时会重新完成该检查点的评估，不会重复追加指标行。

若要主动验证中断，可在第一次命令增加 `--stop-after-steps 3`；它会在第一个未完成正式作业的全局第 3 个 optimizer step 保存后退出。恢复时去掉此选项并加 `--resume`。pilot 不受此选项影响，八轮学习率计划也不改变。

源码、模型/数据/统计量、GPU 类型、可见 CUDA 数量或核心库等不兼容时，原生恢复检查会拒绝继续；不要修改签名绕过。v0.1 旧检查点不能通过这个启动器自动接管；需使用底层 CLI 的显式旧格式选项，且没有逐位一致保证，详见[续训说明](docs/implementation.md#resume-training-v02)。已有手工 CLI 作业没有 `pipeline.json` 时，也应使用原 CLI 恢复，不要把其目录冒充新 workspace。

## 5. 输出文件和存储位置

按示例配置，结果在 **`artifacts/study-v1/`**，模型缓存单独在 **`artifacts/hf-cache/`**。源码包不包含模型和训练输出。

```text
artifacts/
├── hf-cache/                         # 固定模型、tokenizer、processor 的下载缓存
└── study-v1/                         # 配置中的 workspace
    ├── pipeline.json                 # 冻结的核心源码/数据来源/统计量/pilot 设置
    ├── pipeline-status.json          # 启动器当前阶段、错误或停止位置
    ├── invocations.jsonl             # 每次启动的配置选择、seed、保存频率等
    ├── .pipeline.lock               # 单作业目录锁；不是完成标记
    ├── manifests/
    │   ├── mmsd2/
    │   │   ├── train.jsonl           # 固定训练池
    │   │   ├── dev.jsonl             # 完整开发样本
    │   │   ├── test.jsonl            # 保留测试样本
    │   │   ├── manifest.json         # 来源、标签顺序、计数、排除项、重复组、哈希
    │   │   └── ocr_audit.jsonl       # 固定人工 OCR 审核样本清单
    │   └── mmimdb/                  # 同样结构
    ├── stats/
    │   ├── mmsd2.pt                  # 固定图像/正文均值和标准差、采样ID和元数据
    │   ├── mmsd2.json                # 可读的统计量估计元数据
    │   ├── mmimdb.pt
    │   └── mmimdb.json
    ├── pilots/
    │   └── mmsd2/L-low/seed_42_pilot/
    │       ├── pilot-complete.json   # 启动器确认 pilot 已完成，含硬件和完整性信息
    │       └── ...                  # 与下面正式作业相同的训练输出；不混入筛选分析
    ├── runs/
    │   └── mmsd2/LM-low/seed_42/
    │       ├── config.json           # 超参数、实际可训练参数名/数量、软件/源码来源
    │       ├── progress.json         # 完成/主动停止时的状态；不是实时进度条
    │       ├── resources.json        # 每轮损失、梯度、长度、耗时、显存和吞吐
    │       ├── metrics.csv           # 每个 checkpoint/条件的任务指标和非法输出率
    │       ├── contributions.csv     # C_image、C_text、G，保留负值
    │       ├── checkpoints/
    │       │   ├── epoch_0/          # 训练前
    │       │   ├── epoch_1/          # 一直到 epoch_8
    │       │   ├── epoch_8/
    │       │   └── step_50/          # 如启用额外保存；编号是全局 optimizer step
    │       │       ├── trainable.pt  # LoRA/merger 参数；不重复保存冻结骨干
    │       │       ├── state.pt      # optimizer、scheduler、RNG、游标和历史
    │       │       ├── complete.json # 完整性标记和两个 .pt 文件的 SHA256
    │       │       └── processor/    # processor/tokenizer 配置
    │       ├── predictions/
    │       │   ├── epoch_2_full.jsonl
    │       │   ├── epoch_2_mask_image.jsonl
    │       │   ├── epoch_2_mask_text.jsonl
    │       │   └── ...              # epoch 0–8 主条件，及 0/4/8 的敏感性文件
    │       └── interventions/
    │           └── epoch_2.jsonl    # 实际替换位置、张量形状、范数、噪声种子
    ├── figures/
    │   └── mmsd2/
    │       ├── analysis.json
    │       ├── full_performance.png
    │       ├── image_contribution.png
    │       ├── text_contribution.png
    │       ├── gain_over_best_ablation.png
    │       ├── raw_condition_scores.png
    │       ├── invalid_output_rate.png
    │       └── performance_vs_gpu_time.png
    ├── logs/
    │   ├── download_model.log
    │   ├── prepare_mmsd2.log
    │   ├── stats_mmsd2.log
    │   ├── pilot_mmsd2_L-low_seed_42.log
    │   ├── train_mmsd2_LM-low_seed_42.log
    │   └── analyze_mmsd2.log
    └── recovery/                    # 若准备/初始化曾中断，保留未完成产物的备份
```

`epoch_*` 目录内部也都有 `trainable.pt`、`state.pt`、`complete.json`、`processor/`。若周期保存点恰好位于 epoch 结束，使用该 epoch 检查点即可，不一定另有同编号的 `step_*` 目录。

每条 prediction 记录含样本 ID、参考标签、原始输出、生成 token、解析结果、非法标记、训练 seed、epoch、step、条件及噪声设置。CSV 的 F1 使用 0–1 数值，贡献图以 F1 百分点显示。所有完整作业、所有 seed 都会进入 workspace 的分析。

查看运行日志：

```bash
tail -f artifacts/study-v1/logs/train_mmsd2_LM-low_seed_42.log
cat artifacts/study-v1/pipeline-status.json
ls artifacts/study-v1/runs/mmsd2/LM-low/seed_42/checkpoints/
```

日志文件仅在对应阶段实际执行时创建；完整缓存已存在时可能没有 `download_model.log`。指标和资源文件在每轮评估完成后更新。模型缓存和检查点都可能很大；step 检查点包含优化器状态，**不会自动删除旧检查点**。磁盘紧张时可增加 `checkpoint_every_steps`，或设为 0 只保留 epoch 检查点。不要删除正式分析所需的 epoch 检查点、统计量和清单。

## 6. 正式测试集评估是单独一步

一键默认不会自动选择比较对象或评估测试集。先根据开发集结果预先锁定 epoch 2/8 的比较，将 [heldout-selection.example.json](configs/heldout-selection.example.json) 中的 `run` 路径改为实际 `workspace/runs/...`，填写真实锁定时间，然后运行：

```bash
.venv/bin/modalitygap test \
  --data artifacts/study-v1/manifests/mmsd2 \
  --stats artifacts/study-v1/stats/mmsd2.pt \
  --cache-dir artifacts/hf-cache \
  --selection locked-mmsd2.json \
  --output artifacts/study-v1/heldout
```

结果位于 `heldout/<dataset>/<config>/seed_<seed>/epoch_2/` 和 `epoch_8/`，各包含 `metrics.csv`、`contributions.csv`、`selection.json` 及预测/干预日志。测试入口只接受 epoch 2/8，不能据测试结果再选择有利的检查点。

## 7. 故障处理与验证

- **目录正在运行**：查看 tmux/nohup 日志和 GPU 进程，不要再启动第二份。
- **输入或源码签名变了**：恢复原文件；确实要改协议时用新 workspace，不能覆盖旧实验。
- **准备阶段中断**：没有依赖这些产物的训练时，脚本保留不完整目录到 `recovery/` 再重建；已有训练使用的清单/统计量损坏时会停止，要求恢复原文件。若首次模型初始化就中断且还没有任何已发布检查点/历史，脚本会先备份该未开始作业，再从原始权重启动。
- **模型下载中断**：重跑 `--resume`，下载器复用缓存；若设置了离线模式而缓存不完整，先恢复网络或完整下载模型。
- **显存不足**：保留日志和检查点；先按协议完成统一预算的资源试跑，不能只为单个候选悄悄降低分辨率。

工程测试、真实 3B 验证和精确续训记录分别见 [GPU 验证](docs/validation.md)、[续训验证](docs/resume-validation.md)。启动器验证见 [launcher-validation.md](docs/launcher-validation.md)。完整实现细节见 [implementation.md](docs/implementation.md)。

运行测试：

```bash
# 已安装依赖后，下载真实 processor 配置（不下载权重）：
.venv/bin/python scripts/fetch_model.py --processor-only
export MODALITYGAP_PROCESSOR="$PWD/artifacts/hf-cache/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3"
export MODALITYGAP_GPU_TESTS=1
OMP_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -v
```

原始数据导出、人工 OCR/重复审核、完整筛选/确认的计算预算和研究结论仍需由真实数据实验完成。源码仓库与分发包不包含数据集、模型权重或正式研究结果。

真实数据小规模 pilot（MMSD2.0 / MM-IMDb）的 loss、F1、消融异常说明、来源核对和输出路径见 [真实数据 pilot 报告](docs/real-data-pilot.md)。

mean-only、多噪声 seed、打乱配对与首 token EOS/embedding 范数诊断见 [masking 诊断报告](docs/mask-diagnostics.md)。

对称“均值占位＋attention 屏蔽”的独立实现、真实数据诊断及无泄漏验证见 [v2 报告](docs/symmetric-mask-v2.md)，输出单独存于 `artifacts/real-data-pilot-v1/symmetric-attention-mask-v2/`。

## 对称 masking 的正式训练（新协议）

新实验可在配置 JSON 中设置：

```json
"masking_method": "symmetric-attention-v2"
```

图像和正文都用固定训练均值占位，并从第一层开始，在全部语言 decoder 层及缓存解码期间屏蔽该模态内容位置的 Key/Value。图片边界、任务指令、标签列表、assistant 边界、原始位置编码与序列长度保留。生成的 repetition penalty 只读取可见 prompt 和已生成 token。训练仍使用完整图文输入，无额外训练、无 modality dropout。

CLI 对应参数为 `modalitygap run --masking-method symmetric-attention-v2 ...`；省略参数保留 `mean-noise-v1` 行为。协议名保存在 run config、pipeline 和精确恢复签名中，跨协议续训会被拒绝。held-out 评估自动采用 checkpoint 对应协议。

新协议每个正式作业评估 epoch **0–8**，每轮 full / mask-image / mask-text，共 **27 次评估**。没有随机噪声，故不重复运行旧协议的额外噪声种子与 mean-only 条件；旧协议仍为45次。资源 pilot 仅覆盖所选训练配置涉及的适配范围。

### 本机已准备的真实数据子集：完整8轮

配置：[configs/real-symmetric-8epoch.json](configs/real-symmetric-8epoch.json)。使用真实 MMSD2.0 的64条、MM-IMDb的63条训练样本，各32条开发样本；L-low、seed42，从固定原始3B模型重新训练完整8轮。它用于检查完整轨迹，不是全量数据集结果，未使用 test。

```bash
# 一键：准备、统计量、资源pilot、完整8轮、分析绘图
HF_HUB_OFFLINE=1 bash scripts/run_experiment.sh --config configs/real-symmetric-8epoch.json

# 中断后：同一条命令追加 --resume
HF_HUB_OFFLINE=1 bash scripts/run_experiment.sh --config configs/real-symmetric-8epoch.json --resume

# 只看执行计划，不写文件
bash scripts/run_experiment.sh --config configs/real-symmetric-8epoch.json --dry-run
```

该配置写入独立的 `artifacts/real-symmetric-8epoch-v1/`。首次执行拒绝已有 workspace；恢复时核对源码、协议、数据和统计量签名，并跳过已完成作业。要进行另一项实验，先复制配置并更换 `workspace`；全量实验也需更换 `datasets.*.source` 为全量作者数据导出。

主要输出：

- `runs/<dataset>/L-low/seed_42/metrics.csv`：逐轮 full/mask F1 和非法率。
- 同目录 `contributions.csv`：`image_contribution = F1_full − F1_mask_image`、`text_contribution = F1_full − F1_mask_text`；数值单位为0–1，乘100为F1百分点。
- 同目录 `predictions/`、`interventions/`、`resources.json`、`checkpoints/epoch_0..8/`：原始回答、实际屏蔽日志、loss/显存及可恢复状态。
- `figures/<dataset>/`：标准分析与bootstrap输出；`logs/`：分阶段日志；`pipeline.json`：冻结身份信息。

贡献曲线表示本协议下的性能差，不是因果贡献；始终结合非法率阅读。小子集、单seed不支持正式排名反转或泛化结论。旧实验 workspace 与本次 workspace 分开保存，两个协议的曲线不混用。

逐轮真实子集结果与贡献曲线见 [完整8轮报告](docs/real-symmetric-8epoch.md)。

## NewsCLIPpings / CrisisMMD / Hateful Memes：真实数据 pilot

新增三个独立二分类任务，均使用 `symmetric-attention-v2`、L-low、seed42、64 train / 32 dev、完整8 epoch。epoch0额外保留训练前基线；每轮验证 full、mask_image、mask_text。三者的主诊断指标均为严格生成的 macro-F1；Hateful Memes 的官方 AUROC 不在本次 F1 pilot 中替代报告。

数据下载与核对脚本为 `scripts/prepare_expansion.py`：保留固定源版本、官方样本记录、图片字节及 RGB 哈希，按照模型输出无关的 seed42 规则抽样。数据位于 `data/real-expansion-v1/<dataset>/`，原标注和来源仓库位于 `data/downloads/expansion-v1/` 与 `data/sources/expansion-v1/`。这里只落盘 pilot 所需原图，不要求保存完整图片包。

- `newsclippings`：官方 `merged_balanced` 的 train/val，32/16个完整正负 caption 配对。caption取 VisualNews `id`，图片取 `image_id`，不可混用。所用完整 data.json 已与作者 TAR 内原文件核验 SHA256，逐字节一致。作者原图包通过 `scripts/fetch_visualnews_images_v2.py` 分段流式读取，只保留选定原始 JPEG；分段进度与图片偏移索引可用于续传和后续按需取图。
- `crisismmd`：作者 `task_informative_text_img_agreed_lab_{train,dev,test}.tsv` 的图文一致信息性标签。全部 agreed-label 记录已对照 QCRI 官方仓库核验；图片直接从固定 QCRI 版本取得，保留原始发布文字。
- `hatefulmemes`：原始 challenge 的 train / dev_seen；固定公开镜像提供 JSONL/PNG，两个镜像的原始标注逐字节一致，ID、正文、路径、划分已与作者后续细粒度发布核对。后续版本少量标签有差异，本次保留原始标签，详见数据目录 `author-crosscheck.json`；没有官方压缩包校验值可独立认证镜像 PNG 的字节身份。**PNG 内已有文字，text mask 只隐藏额外输入的正文，图片中的文字仍可被模型读取。** 不把这个差分解释成全部语言信息的贡献。

```bash
# 本次数据和 workspace 已准备／初始化后，依次运行或恢复三个实验
HF_HUB_OFFLINE=1 bash scripts/run_expansion.sh --resume

# 单独运行或恢复一个任务
HF_HUB_OFFLINE=1 bash scripts/run_experiment.sh \
  --config configs/expansion-crisismmd-8epoch-v1.json --resume

# 检查三个任务的命令，不加载 GPU、不写文件
bash scripts/run_expansion.sh --dry-run
```

全新空 workspace 首次运行去掉 `--resume`。再次实验先复制相应配置并修改 `workspace`；现有结果目录不会自动覆盖。续训须保持原始源文件、模型版本、统计量及源码签名一致；源码快照保存于本次汇总根目录，不能用修改后的代码静默恢复旧 checkpoint。

独立输出：`artifacts/real-expansion-symmetric-8epoch-v1/<dataset>/`。每个任务的 `runs/<dataset>/L-low/seed_42/` 保存逐轮 `metrics.csv`、`contributions.csv`、原始 token/回答 `predictions/`、实际屏蔽记录 `interventions/`、loss/显存 `resources.json`，以及包含参数、优化器、scheduler和所有随机状态的 `checkpoints/epoch_0..8/`。`manifests/`、`stats/`、`pilots/`、`logs/`、`figures/` 分别保存冻结划分、训练均值、资源预检、阶段日志、标准分析。

三个任务完成后生成汇总：

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/modalitygap-mpl .venv/bin/python scripts/report_expansion.py
```

根目录输出 `epoch-report.json` 和 `contribution-trajectories.png`，逐轮表格见 [三个数据集8轮报告](docs/expansion-symmetric-8epoch-v1.md)。旧的 `real-symmetric-8epoch-v1`、`real-data-pilot-v1` 等结果不参与新目录写入。

独立核验输出：每个 workspace 的 `prediction-audit.json` 记录864条正式预测重解码与指标复算；汇总根目录的 `data-audit.json` 核对原图哈希、官方映射／标签／划分和训练均值样本ID，`real-input-audit.json` 验证288条真实样本的推理tensor和正文span均不依赖参考标签。Hateful Memes 的 `label-version-sensitivity.json` 单独记录后续细粒度标签版本对同一批预测的重评分，主实验保持原始challenge标签。

## 负贡献／零贡献的原因诊断

[CrisisMMD 与 Hateful Memes 诊断报告](docs/contribution-cause-diagnostics-v1.md) 包括真实 epoch8 断点内容隔离测试、128条官方开发样本复核、逐样本错对翻转、固定占位长度与联合遮蔽控制，以及图片单独 OCR。新结果独立保存在 `artifacts/contribution-cause-diagnostics-v1/`；正式训练源码和旧结果保持原样。对应脚本为 `scripts/prepare_contribution_diagnostics.py`、`scripts/diagnose_contribution_causes.py`、`scripts/report_contribution_causes.py`。
