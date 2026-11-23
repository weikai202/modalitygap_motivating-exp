"""Summarize completed real-data pilots without requiring nine-epoch screening."""
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from modalitygap.protocol import write_json

ROOT = Path('artifacts/real-data-pilot-v1')

def main():
    report = {}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for i, dataset in enumerate(('mmsd2', 'mmimdb')):
        directory = ROOT/'runs'/dataset/'L-low'/'seed_42_pilot'
        with (directory/'metrics.csv').open() as f:
            metrics = list(csv.DictReader(f))
        resources = json.loads((directory/'resources.json').read_text())
        purpose = json.loads((directory/'pilot-purpose.json').read_text())
        primary = 'macro_f1' if dataset == 'mmsd2' else 'micro_f1'
        epochs = sorted({int(r['epoch']) for r in metrics})
        if epochs != [0, 1, 2] or len(metrics) != 9 or resources[-1]['step'] != 8:
            raise AssertionError('Incomplete prescribed pilot')
        axes[i, 0].plot([r['step'] for r in resources], [r['mean_loss'] for r in resources], 'o-')
        axes[i, 0].set(xlabel='Optimizer step', ylabel='Mean training loss', title=dataset)
        diagnostics = {}
        for variant in ('full', 'mask_image', 'mask_text'):
            rows = sorted((r for r in metrics if r['variant'] == variant), key=lambda r:int(r['epoch']))
            axes[i, 1].plot(epochs, [float(r[primary]) for r in rows], 'o-', label=variant)
            axes[i, 2].plot(epochs, [float(r['invalid_rate']) for r in rows], 'o-', label=variant)
            predictions = [json.loads(s) for s in (directory/'predictions'/f'epoch_2_{variant}.jsonl').read_text().splitlines()]
            diagnostics[variant] = dict(
                final_score=float(rows[-1][primary]), final_invalid_rate=float(rows[-1]['invalid_rate']),
                final_invalid_outputs=dict(Counter(r['raw_output'] for r in predictions if r['invalid'])),
                generated_lengths=dict(Counter(r['generated_length'] for r in predictions)))
        axes[i, 1].set(xlabel='Epoch (4 optimizer steps each)', ylabel=primary, ylim=(-.02, 1.02))
        axes[i, 2].set(xlabel='Epoch', ylabel='Invalid output fraction', ylim=(-.02, 1.02))
        axes[i, 1].legend(fontsize=8)
        report[dataset] = dict(purpose=purpose, primary_metric=primary, metrics=metrics,
                               training=resources, final_diagnostics=diagnostics,
                               provenance=str(Path('data/real-pilot')/dataset/'provenance.json'))
    ROOT.mkdir(exist_ok=True)
    fig.savefig(ROOT/'loss-f1-invalid.png', dpi=160)
    write_json(ROOT/'report.json', report)
    write_json('docs/real-data-pilot-results.json', report)
    lines = [
        '# 真实数据小规模 pilot',
        '',
        '两组均在 RTX 3090 24 GB 上使用固定 Qwen2.5-VL-3B revision、L-low、seed 42、BF16 混合精度运行 8 个 optimizer step（2 个实际 epoch）。学习率仍按原定 8 epoch 计划计算；没有为了 pilot 缩短 scheduler。每个 epoch 评估 full、mask_image、mask_text，噪声 seed 2026、scale 0.05。',
        '',
        '这是训练、指标和干预的资源/数值检查；不是完整开发集结果，没有使用 test，没有完成六配置筛选、三 seed 确认或敏感性实验。所有抽样在读取模型结果前固定。',
        '',
        '| 数据 | train/dev | mean loss epoch1 → epoch2 | full F1 epoch0 → epoch2 | 最终 mask_image F1 | 最终 mask_text F1 |',
        '|---|---|---|---|---|---|',
    ]
    for ds, x in report.items():
        d = x['final_diagnostics']; loss = x['training']
        initial = next(float(r[x['primary_metric']]) for r in x['metrics'] if r['epoch']=='0' and r['variant']=='full')
        lines.append(f"| {ds} ({x['primary_metric']}) | {x['purpose']['train_count']}/{x['purpose']['dev_count']} | {loss[0]['mean_loss']:.6f} → {loss[-1]['mean_loss']:.6f} | {initial:.4f} → {d['full']['final_score']:.4f} | {d['mask_image']['final_score']:.4f} | {d['mask_text']['final_score']:.4f} |")
    lines += ['', '| 数据 | 最终 full 非法率 | mask_image 非法率 | mask_text 非法率 |', '|---|---|---|---|']
    for ds,x in report.items():
        d=x['final_diagnostics']
        lines.append(f"| {ds} | {d['full']['final_invalid_rate']:.2%} | {d['mask_image']['final_invalid_rate']:.2%} | {d['mask_text']['final_invalid_rate']:.2%} |")
    lines += [
        '',
        '## 如何解读',
        '',
        '训练 loss 和梯度均有限，GPU 训练与三条件验证完整结束。MMSD2.0 的 full 提升包含输出格式学习：初始 18/32 条非法，最终 2/32 条非法；非法例子为 `sarcasm`，协议只接受 `sarcastic` 或 `not_sarcastic`。严格 parser 没有为提高成绩增加同义词。',
        '',
        'MMSD2.0 的 mask_text 最终 31/32 条非法，主要是立即 EOS/空回答，生成长度远未触及上限。因此消融下降混入格式崩溃，不能作为可靠的模态贡献或融合增益证据。对 MM-IMDb 也必须结合上表非法率和逐条回答解读；微小样本、单 seed、仅两轮不足以证明模型优劣。',
        '',
        '人工查看 seed2026 的 16 张训练图：MMSD2.0 至少 9 张、MM-IMDb 15 张含可读文字；图像通道仍可能传递文字信息。原始数据还包含 MMSD2.0 的 223×21 图片和 MM-IMDb 的 7×5 海报，均按预先固定的数据规则保留，没有看结果后换样本。',
        '',
        'MM-IMDb 的 full micro-F1 在这 8 步保持 0.4182；训练 loss 只有小幅下降。epoch0 与 epoch2 的 288/288 个 LoRA 参数张量均发生变化且保持有限值，排除了完全没有更新参数的情况。最终 full 的两个非法回答是 `["Science-Fiction"]`，固定词表要求 `Sci-Fi`；mask_text 为31条空回答和1条 `Seven Seven`。这些结果不足以断言泛化性能已提升。',
        '',
        '## 数据来源与核对',
        '',
        '- MMSD2.0 使用[作者修正标注](https://github.com/JoeYing1019/MMSD2.0)，revision `1627811f7a1e0acedce5b9df8b58919d86dd638e`；[图片镜像](https://huggingface.co/datasets/coderchen01/MMSD2.0) revision `5b39e5b844accf2331d1738a3d91d4cb59005a50`。完整 train/dev 22226 条 ID、文本、标签逐项核对，零差异。train 每类32条、dev 每类16条，排序 ID 后 seed42 抽样。原始 JPEG bytes 直接写出。',
        '- MM-IMDb 使用[作者推荐](https://github.com/johnarevalo/gmu-mmimdb)的[原始归档](https://archive.org/download/mmimdb/mmimdb.tar.gz)，MD5 `38e154264b3771f70840c6cb5f170a93` 已验证。25959 对原始海报/JSON，官方划分15552/2608/7799，与发布 HDF5 的 ID 划分完全匹配；HDF5 仅用作 ID/标签/划分元数据核对，没有使用其图片或特征。',
        '- MM-IMDb 在官方 train/dev 顺序上以 seed42 做 iterative multilabel stratification，请求64/32，实际返回63/33；训练保留63，开发按官方索引顺序取前32，规则在模型评估前固定。选中样本 raw JSON genres 与 HDF5 全部一致；输入是发布列表中的第一条非空 plot，没有附加标题或其他 metadata。',
        '- prepare 完成真实图片解码、空文本检查、跨划分 ID/RGB/文本重复检查；两套 manifest 均无样本排除。完整选中 ID、哈希、源 revision 见 [provenance](real-data-pilot-provenance.json)。',
        '',
        '## 本机运行与恢复',
        '',
        '已准备的真实数据 manifest 位于 `data/real-pilot/{mmsd2,mmimdb}/manifest/`。在仓库根目录、可见 CUDA 的终端中执行：',
        '',
        '```bash',
        'PYTHONPATH=src OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_real_pilot.py --dataset mmsd2',
        'PYTHONPATH=src OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_real_pilot.py --dataset mmimdb',
        '# 仅对中断作业追加 --resume，复用最新完整 checkpoint；不要覆盖已完成的结果。',
        '```',
        '',
        '重新下载/准备真实数据的专用脚本需要额外 `h5py==3.14.0 pyarrow==20.0.0`，且使用本报告固定的数据源 revision 和选中 ID；原始数据不随 Git 发布。完整正式训练仍使用 README 的标准启动器。',
        '',
        '## 验证',
        '',
        '保存的18份预测文件共576条回答，逐条从 token IDs 重新解码、严格解析并重算汇总指标全部匹配。参数更新审计见 `audits/parameter-update-audit.json`；30 个单元及 CUDA 回归测试全部通过（无跳过），其中包含两个任务、三种适配范围的参数/优化器/scheduler/随机状态精确恢复；日志见 `artifacts/real-pilot-gpu-regression-tests.log`。CUDA 回归中的小型随机模型用于验证编排与精确恢复；本报告中的 F1 来自真实数据上的真实3B模型。',
        '',
        '## 输出位置',
        '',
        '- `artifacts/real-data-pilot-v1/runs/<dataset>/L-low/seed_42_pilot/`：metrics.csv、contributions.csv、resources.json、predictions/、interventions/、checkpoints/ 和冻结 config.json。',
        '- `artifacts/real-data-pilot-v1/stats/`：只用选中训练样本估计的原始模型 masking moments。',
        '- `artifacts/real-data-pilot-v1/report.json` 与 `loss-f1-invalid.png`：数值汇总和曲线；逐条回答可核查非法输出原因。',
        '- `artifacts/real-data-pilot-v1/audits/`：原图 contact sheets、尺寸/文字人工检查及 strict-decoding-audit.json。',
        '- `artifacts/real-mmsd2-pilot.log`、`artifacts/real-mmimdb-pilot.log`：运行日志。',
        '- [results JSON](real-data-pilot-results.json) 与 [provenance JSON](real-data-pilot-provenance.json)：保留在 docs 的可审阅快照。',
        '',
        '汇总重建：`PYTHONPATH=src .venv/bin/python scripts/report_real_pilots.py`。',
    ]
    Path('docs/real-data-pilot.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({ds:dict(loss=[r['mean_loss'] for r in x['training']],
                             final=x['final_diagnostics']) for ds,x in report.items()},indent=2))

if __name__ == '__main__':
    main()
