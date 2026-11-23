# 真实数据 masking 诊断

RTX3090；固定真实开发子集，各32条；L-low seed42 的 epoch0/epoch2。主要条件使用全部32条；额外机制对照只使用预先固定 seed2026 的8条，下面分表报告，避免把不同样本量混在一起。未重新训练模型、未改变正式 masking 协议。

诊断结果支持：当前 mean+0.05std 噪声替换在正文通道形成均值主导、token间变化过小的输入，触发提前终止；范数缩小并非唯一因素。它在原始 epoch0 已出现，跨 seed 和 checkpoint 复现。恢复方差比单纯恢复范数有效，但不能完全恢复正常格式，尤其是 MM-IMDb。当前 text-mask F1 落差不能直接解释为可靠的文本贡献。

EOS 概率直接取实际 `generate` 的首步：同时保存处理前 logits softmax 和处理后 scores softmax。模型固定 generation config 含 repetition_penalty=1.05；实际首 token 取处理后 argmax。EOS 指 `<|im_end|>` (151645)，另行统计 `<|endoftext|>` (151643)。

## 最终 checkpoint：32条主要诊断

| 数据 | 条件 | F1 | 非法率 | 首步 EOS 概率（原始/处理后均值） | 首步 EOS 数 |
|---|---|---|---|---|---|
| mmsd2 | full | 0.7729 | 6.25% | 1.0478e-08 / 7.9111e-09 | 0/32 |
| mmsd2 | image_mean_only | 0.4895 | 3.12% | 5.0928e-09 / 3.9523e-09 | 0/32 |
| mmsd2 | image_noise_2026 | 0.5077 | 0.00% | 5.3078e-09 / 4.1028e-09 | 0/32 |
| mmsd2 | image_noise_2027 | 0.4818 | 0.00% | 5.379e-09 / 4.1536e-09 | 0/32 |
| mmsd2 | image_noise_2028 | 0.4895 | 3.12% | 5.1969e-09 / 4.0259e-09 | 0/32 |
| mmsd2 | text_mean_only | 0.0588 | 96.88% | 0.96013 / 0.9558 | 31/32 |
| mmsd2 | text_noise_2026 | 0.0588 | 96.88% | 0.93199 / 0.92708 | 30/32 |
| mmsd2 | text_noise_2027 | 0.0588 | 96.88% | 0.95753 / 0.95178 | 31/32 |
| mmsd2 | text_noise_2028 | 0.1176 | 93.75% | 0.93027 / 0.92622 | 30/32 |
| mmsd2 | shuffle_image | 0.4095 | 6.25% | 2.9859e-08 / 2.1957e-08 | 0/32 |
| mmsd2 | shuffle_text | 0.5233 | 3.12% | 3.9757e-08 / 2.7495e-08 | 0/32 |
| mmimdb | full | 0.4182 | 6.25% | 4.2529e-09 / 2.7258e-09 | 0/32 |
| mmimdb | image_mean_only | 0.3894 | 3.12% | 3.7077e-09 / 2.3239e-09 | 0/32 |
| mmimdb | image_noise_2026 | 0.3894 | 3.12% | 4.1285e-09 / 2.5779e-09 | 0/32 |
| mmimdb | image_noise_2027 | 0.3929 | 3.12% | 4.1665e-09 / 2.6028e-09 | 0/32 |
| mmimdb | image_noise_2028 | 0.3929 | 3.12% | 4.2613e-09 / 2.6585e-09 | 0/32 |
| mmimdb | text_mean_only | 0.0000 | 100.00% | 0.97383 / 0.9548 | 32/32 |
| mmimdb | text_noise_2026 | 0.0000 | 100.00% | 0.90182 / 0.87715 | 31/32 |
| mmimdb | text_noise_2027 | 0.0000 | 100.00% | 0.90671 / 0.88509 | 30/32 |
| mmimdb | text_noise_2028 | 0.0000 | 100.00% | 0.95253 / 0.92325 | 32/32 |
| mmimdb | shuffle_image | 0.4425 | 9.38% | 5.6104e-09 / 3.6537e-09 | 0/32 |
| mmimdb | shuffle_text | 0.1651 | 6.25% | 5.2342e-09 / 3.4354e-09 | 0/32 |

## 最终 checkpoint：相同8条机制对照

normmatched 用冻结训练统计量的 population RMS 为固定目标，只缩放替换向量，不读取当前样本原正文 embedding 或标签。std1 将系数从0.05改为1.0，恢复每维统计方差，但不恢复词汇或句子结构。这些仅用于定位机制，不能拿来挑选更好看的正式消融。

| 数据 | 条件 | F1 | 非法率 | EOS 概率（处理后均值） | 首步 EOS 数 | 正文原/替换 mean token norm | 原/替换 centered RMS |
|---|---|---|---|---|---|---|---|
| mmsd2 | full | 0.9000 | 12.50% | 7.0217e-09 | 0/8 | 0.8360 / 0.8360 | 0.6929 / 0.6929 |
| mmsd2 | text_mean_only | 0.1667 | 87.50% | 0.86023 | 7/8 | 0.8360 / 0.1770 | 0.6929 / 0.0000 |
| mmsd2 | text_noise_2026 | 0.1667 | 87.50% | 0.86787 | 7/8 | 0.8360 / 0.1821 | 0.6929 / 0.0354 |
| mmsd2 | text_noise_std1 | 0.6500 | 25.00% | 0.0014934 | 0/8 | 0.8360 / 0.8602 | 0.6929 / 0.7086 |
| mmsd2 | text_noise_normmatched | 0.1667 | 87.50% | 0.86367 | 7/8 | 0.8360 / 0.8635 | 0.6929 / 0.1681 |
| mmsd2 | text_mean_normmatched | 0.1667 | 87.50% | 0.86212 | 7/8 | 0.8360 / 0.8632 | 0.6929 / 0.0000 |
| mmsd2 | shuffle_text | 0.5636 | 0.00% | 5.792e-08 | 0/8 | 0.8299 / 0.8299 | 0.6737 / 0.6737 |
| mmimdb | full | 0.2308 | 12.50% | 1.8214e-09 | 0/8 | 0.8350 / 0.8350 | 0.8160 / 0.8160 |
| mmimdb | text_mean_only | 0.0000 | 100.00% | 0.99102 | 8/8 | 0.8350 / 0.1906 | 0.8160 / 0.0000 |
| mmimdb | text_noise_2026 | 0.0000 | 100.00% | 0.96148 | 8/8 | 0.8350 / 0.1948 | 0.8160 / 0.0403 |
| mmimdb | text_noise_std1 | 0.0952 | 75.00% | 0.2215 | 3/8 | 0.8350 / 0.8329 | 0.8160 / 0.8054 |
| mmimdb | text_noise_normmatched | 0.0000 | 100.00% | 0.98349 | 8/8 | 0.8350 / 0.8332 | 0.8160 / 0.1722 |
| mmimdb | text_mean_normmatched | 0.0000 | 100.00% | 0.99079 | 8/8 | 0.8350 / 0.8331 | 0.8160 / 0.0000 |
| mmimdb | shuffle_text | 0.0769 | 12.50% | 3.7874e-09 | 0/8 | 0.8402 / 0.8402 | 0.8208 / 0.8208 |

## 定位结论与解释边界

1. 崩溃在 epoch0 已存在；对照 epoch2 可检查其是否由本次 LoRA 更新引入。mean-only 与多个噪声 seed 的一致行为用于排除单个随机 seed 偶然性。
2. 原始 logits 与处理后 scores 分开保存，避免把 repetition penalty 的影响当作模型本身的 EOS 偏好。直接记录的首 token EOS 概率与实际生成长度用于区分提前终止和后处理格式失败。mask-text 的首步 EOS 不能由放宽标签 parser 或增加 max_new_tokens 修复；不改动正式 parser。
3. 仅恢复 RMS 范数和恢复每维噪声方差是两种不同的对照。原系数0.05使每维方差只有原统计方差的0.25%，token 会集中在共同均值附近；整体放大向量不能改变这种均值方向占主导的关系。std1 恢复每维方差，也改变向量方向分布；如果它解除 EOS 而范数匹配未解除，证据支持“均值主导、变化不足的替换分布”，而非仅仅范数小。它不能唯一识别模型内部的所有终止机制。
4. 打乱正文配对保留真实正文的边际分布，但改变单个样本的长度与语义配对；不是严格控制全部因素的干预。它主要检查模型是否仍能正常生成，不能把其 F1 当作可靠的单模态成绩。
5. 噪声系数1.0的对照还出现过不在固定词表中的标签、重复 `<|im_start|>`，个别回答达到生成长度上限；因此减少首步 EOS 并不等于恢复稳定的任务输出。对照解除 EOS 不等于得到中立的 missing-modality 分布，额外机制对照只有固定8条。不能据本诊断将系数1.0直接替换正式协议、宣称消融已修好或报告因果模态贡献。额外训练 noise encoder 也不是这些结果推出的必要步骤。

## 核对与文件

共56份逐条预测、1504条诊断记录。原 pilot 的 full、mask_image、mask_text 在两个 checkpoint、两个数据集上共384条生成 token 序列全部精确复现。所有条件受保护位置 embedding 完全相同，所有实际首 token 与生成处理后分数 argmax 一致，打乱配对没有自配对。

- 完整 epoch0/2、EOS 分布、embedding 均值/范数/中心化波动、非法原始回答和 probe 子集：`artifacts/real-data-pilot-v1/mask-diagnostics-v1/report.json`。
- 每条记录及预先固定的 donor/probe ID：`mask-diagnostics-v1/<dataset>/epoch_<epoch>_<condition>.jsonl` 和 `plan.json`。shuffle_image 保留原正文与原标签，替换图像；shuffle_text 保留原图与原标签，替换正文。均为开发集内部固定无自配对循环置换。
- 诊断运行：`PYTHONPATH=src OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 .venv/bin/python scripts/diagnose_real_masks.py --dataset mmsd2`（另跑 mmimdb）；已有完整条件可复用。
- 汇总重建：`PYTHONPATH=src MPLCONFIGDIR=/tmp/modalitygap-mpl .venv/bin/python scripts/report_mask_diagnostics.py`。
- 新增 instrumentation 测试检查与 production hook 完全一致、范数对照不修改 prompt/cache、完整28条件输出及恢复复用。
