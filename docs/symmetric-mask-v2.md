# 对称均值占位与 attention 屏蔽：真实数据诊断

固定3B模型、RTX3090、L-low seed42，复用真实 MMSD2.0 / MM-IMDb 各32条开发样本及 epoch0/2 checkpoint，无额外训练。每个模态均使用原始训练统计的固定均值占位，并在全部 decoder 层、prefill 和缓存解码中排除对应内容位置的 Key/Value。旧干预、旧指标与旧报告不变。

| 数据 | epoch | 条件 | F1 | 非法率 | 首步 EOS 数 | 原始 EOS 概率均值 |
|---|---|---|---|---|---|---|
| mmsd2 | 0 | full | 0.4127 | 56.25% | 0/32 | 1e-08 |
| mmsd2 | 0 | mask_image | 0.2857 | 62.50% | 0/32 | 1.1864e-08 |
| mmsd2 | 0 | mask_text | 0.3235 | 43.75% | 0/32 | 7.4102e-09 |
| mmsd2 | 2 | full | 0.7729 | 6.25% | 0/32 | 1.0478e-08 |
| mmsd2 | 2 | mask_image | 0.5697 | 3.12% | 0/32 | 1.2291e-08 |
| mmsd2 | 2 | mask_text | 0.6184 | 6.25% | 0/32 | 7.4905e-09 |
| mmimdb | 0 | full | 0.4182 | 6.25% | 0/32 | 4.8463e-09 |
| mmimdb | 0 | mask_image | 0.4138 | 3.12% | 0/32 | 7.1833e-09 |
| mmimdb | 0 | mask_text | 0.4324 | 9.38% | 0/32 | 2.9797e-09 |
| mmimdb | 2 | full | 0.4182 | 6.25% | 0/32 | 4.2529e-09 |
| mmimdb | 2 | mask_image | 0.4167 | 3.12% | 0/32 | 6.8449e-09 |
| mmimdb | 2 | mask_text | 0.4182 | 12.50% | 0/32 | 2.5984e-09 |

新方法在两个数据集、两个 checkpoint 的全部384条生成中均没有首步 EOS。相较旧 text mask 的高 EOS 概率，提前终止现象在本诊断集上消失；但格式未完全正常。最终 MMSD2.0 的 text-mask 两条非法输出均为 `sarcasm`；MM-IMDb 有三条使用 `Science-Fiction`（包括一个多标签数组），另有一条 `["Rock"]`，固定词表要求 `Sci-Fi` 且不包含 `Rock`。没有放宽 parser 来提高 F1。

MMSD2.0 使用 macro-F1，MM-IMDb 使用 micro-F1；分数为固定小样本诊断，非正式全量 benchmark。非法率必须与 F1 一起解读，未增加标签同义词或强制合法标签生成。

## 无泄漏验证

每个数据集、每个 checkpoint 预先固定8条样本，分别扰动正文 token IDs（长度不变）或图像 pixel_values（网格不变）。比较干预后的整个词表 logits：prefill 原始 logits、应用 visible-history repetition penalty 后的分数、三个强制相同 continuation 的缓存步。共64组探针，所有5个比较点均 bitwise 相同，最大绝对差为0。该保证针对被屏蔽的输入通道：text-mask 时图片内文字仍然可见，原正文长度和图像网格元数据也仍保留。

注意力 hook 同时检查每层实际收到的4D causal mask，所有被遮位置的 key 列均被屏蔽。位置编码和 cache_position 在外层用原始完整结构构造，内层只改 key 可见性；不使用置零的2D mask重算 mRoPE。正文长度仍保留，因此这不是消除全部元数据。

生成时保留 repetition_penalty=1.05 的语义，但只依据可见 prompt 与已生成 token；被遮内容不参与历史惩罚。Full 条件128条 token 序列与原 pilot 精确一致。定向测试还检查了多层 attention、KV cache、mRoPE 一致性、无干预 logits 不变及处理器历史隔离。

全部34项回归测试在开启 CUDA 测试的3090环境中通过，无跳过；包括原有两任务六配置全流程、精确断点恢复，以及新增干预测试。日志：`artifacts/symmetric-mask-v2-gpu-regression-tests.log`。

mmsd2 未屏蔽的扰动阳性对照有 32/32 组改变首步 logits；逐条差异见 JSON。
mmimdb 未屏蔽的扰动阳性对照有 32/32 组改变首步 logits；逐条差异见 JSON。

## 实现与复现

- `scripts/symmetric_mask.py`：独立干预与可见历史处理器；旧 production masking 未修改。
- `scripts/diagnose_symmetric_mask.py`：冻结 checkpoint 的真实数据诊断与泄漏探针。
- `tests/test_symmetric_mask.py`：三层随机小型 Qwen 的 attention/cache/位置与生成处理器测试；真实数据分数均来自3090上的原始3B模型。
- `artifacts/real-data-pilot-v1/symmetric-attention-mask-v2/<dataset>/`：plan.json、epoch预测、summary.json、leakage.json、complete.json。
- 首次缓存探针发现 cache_position=None 时需从 KV cache 长度取得偏移；修正前的运行保留在 `mmsd2-initial-cache-audit-attempt/`，不计入本报告。
- `artifacts/{mmsd2,mmimdb}-symmetric-mask-v2.log`：真实运行日志；`artifacts/symmetric-mask-v2-tests.log`：定向测试日志。

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 .venv/bin/python scripts/diagnose_symmetric_mask.py --dataset mmsd2
PYTHONPATH=src OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 .venv/bin/python scripts/diagnose_symmetric_mask.py --dataset mmimdb
```

诊断程序拒绝覆盖已存在的数据集输出目录；复跑需显式更换 OUT 为新版本目录。汇总：`PYTHONPATH=src MPLCONFIGDIR=/tmp/modalitygap-mpl .venv/bin/python scripts/report_symmetric_mask.py`。

该方法改变原协议的 attention 可见性及 masking 历史处理，应作为独立协议报告；旧版和新版贡献曲线不可混用。即使生成稳定，也只支持此干预定义下的输入依赖，不构成因果模态贡献证明。
