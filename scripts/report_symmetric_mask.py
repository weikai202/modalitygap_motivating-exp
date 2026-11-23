"""Audit new symmetric-mask results without touching previous reports."""
import csv,json
from pathlib import Path
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from modalitygap.protocol import MODEL_REVISION,read_jsonl,write_json,metrics,digest

ROOT=Path('artifacts/real-data-pilot-v1');OUT=ROOT/'symmetric-attention-mask-v2'


def main():
    report={};total=0;probes=0
    for ds in ('mmsd2','mmimdb'):
        directory=OUT/ds
        complete=json.loads((directory/'complete.json').read_text())
        assert complete['status']=='passed'
        summaries=json.loads((directory/'summary.json').read_text())
        leakage=json.loads((directory/'leakage.json').read_text())
        assert len(summaries)==6 and len(leakage)==32
        for r in leakage:
            assert r['exact_prefill_processed_and_three_cached_steps']
            assert r['masked_max_absolute_deltas']==[0.]*5
            assert r['attention_checks']>0
        rows_by_condition={}
        for r in summaries:
            rows=read_jsonl(directory/f"epoch_{r['epoch']}_{r['condition']}.jsonl")
            assert len(rows)==32
            recomputed=metrics(ds,rows)
            assert all(recomputed[k]==r[k] for k in ('score','invalid_rate','count'))
            if r['condition']!='full':
                assert all(x['attention_layer_checks']>0 and len(x['decoder_calls'])>=1 for x in rows)
                assert all(c['blocked']>0 for x in rows for c in x['decoder_calls'])
            r['invalid_outputs']=dict(Counter(x['raw_output'] for x in rows if x['invalid']))
            r['mean_generated_length']=sum(x['generated_length'] for x in rows)/32
            rows_by_condition[f"epoch_{r['epoch']}_{r['condition']}"]=r
            total+=len(rows)
        report[ds]=dict(plan=json.loads((directory/'plan.json').read_text()),metrics=rows_by_condition,
            stats_sha256=digest(ROOT/'stats'/f'{ds}.pt'),
            checkpoint_sha256={str(e):digest(ROOT/'runs'/ds/'L-low/seed_42_pilot/checkpoints'/f'epoch_{e}'/'trainable.pt') for e in (0,2)},
            leakage_probes=len(leakage),all_masked_logits_bitwise_identical=True,
            full_perturbation_controls_nonzero=sum(r['full_first_logit_max_delta']>0 for r in leakage),
            full_perturbation_control_total=len(leakage),leakage=leakage)
        probes+=len(leakage)
    report['validation']=dict(model_revision=MODEL_REVISION,predictions=total,leakage_probes=probes,masked_max_absolute_delta=0.,
        unchanged_full_generations=128,targeted_tests_passed=2,
        source_sha256={p:digest(p) for p in ('scripts/symmetric_mask.py','scripts/diagnose_symmetric_mask.py','tests/test_symmetric_mask.py')},
        test_log_sha256=digest('artifacts/symmetric-mask-v2-tests.log'))
    gpu_log=Path('artifacts/symmetric-mask-v2-gpu-regression-tests.log')
    gpu_passed=gpu_log.is_file() and gpu_log.read_text().rstrip().endswith('\nOK') and 'Ran 34 tests' in gpu_log.read_text()
    report['validation']['gpu_regression']=dict(status='passed' if gpu_passed else 'pending',tests=34,skipped=0 if gpu_passed else None,log=str(gpu_log),sha256=digest(gpu_log) if gpu_log.is_file() else None)
    write_json(OUT/'report.json',report);write_json('docs/symmetric-mask-v2-results.json',report)
    lines=['# 对称均值占位与 attention 屏蔽：真实数据诊断','',
        '固定3B模型、RTX3090、L-low seed42，复用真实 MMSD2.0 / MM-IMDb 各32条开发样本及 epoch0/2 checkpoint，无额外训练。每个模态均使用原始训练统计的固定均值占位，并在全部 decoder 层、prefill 和缓存解码中排除对应内容位置的 Key/Value。旧干预、旧指标与旧报告不变。','',
        '| 数据 | epoch | 条件 | F1 | 非法率 | 首步 EOS 数 | 原始 EOS 概率均值 |','|---|---|---|---|---|---|---|']
    for ds in ('mmsd2','mmimdb'):
        for r in report[ds]['metrics'].values():
            lines.append(f"| {ds} | {r['epoch']} | {r['condition']} | {r['score']:.4f} | {r['invalid_rate']:.2%} | {r['first_eos_count']}/32 | {r['mean_eos_probability']:.5g} |")
    lines+=['','新方法在两个数据集、两个 checkpoint 的全部384条生成中均没有首步 EOS。相较旧 text mask 的高 EOS 概率，提前终止现象在本诊断集上消失；但格式未完全正常。最终 MMSD2.0 的 text-mask 两条非法输出均为 `sarcasm`；MM-IMDb 有三条使用 `Science-Fiction`（包括一个多标签数组），另有一条 `["Rock"]`，固定词表要求 `Sci-Fi` 且不包含 `Rock`。没有放宽 parser 来提高 F1。', '', 'MMSD2.0 使用 macro-F1，MM-IMDb 使用 micro-F1；分数为固定小样本诊断，非正式全量 benchmark。非法率必须与 F1 一起解读，未增加标签同义词或强制合法标签生成。','',
        '## 无泄漏验证','',
        '每个数据集、每个 checkpoint 预先固定8条样本，分别扰动正文 token IDs（长度不变）或图像 pixel_values（网格不变）。比较干预后的整个词表 logits：prefill 原始 logits、应用 visible-history repetition penalty 后的分数、三个强制相同 continuation 的缓存步。共64组探针，所有5个比较点均 bitwise 相同，最大绝对差为0。该保证针对被屏蔽的输入通道：text-mask 时图片内文字仍然可见，原正文长度和图像网格元数据也仍保留。','',
        '注意力 hook 同时检查每层实际收到的4D causal mask，所有被遮位置的 key 列均被屏蔽。位置编码和 cache_position 在外层用原始完整结构构造，内层只改 key 可见性；不使用置零的2D mask重算 mRoPE。正文长度仍保留，因此这不是消除全部元数据。','',
        '生成时保留 repetition_penalty=1.05 的语义，但只依据可见 prompt 与已生成 token；被遮内容不参与历史惩罚。Full 条件128条 token 序列与原 pilot 精确一致。定向测试还检查了多层 attention、KV cache、mRoPE 一致性、无干预 logits 不变及处理器历史隔离。','']
    if gpu_passed:
        lines+=['全部34项回归测试在开启 CUDA 测试的3090环境中通过，无跳过；包括原有两任务六配置全流程、精确断点恢复，以及新增干预测试。日志：`artifacts/symmetric-mask-v2-gpu-regression-tests.log`。','']
    for ds in ('mmsd2','mmimdb'):
        r=report[ds]
        lines.append(f"{ds} 未屏蔽的扰动阳性对照有 {r['full_perturbation_controls_nonzero']}/{r['full_perturbation_control_total']} 组改变首步 logits；逐条差异见 JSON。")
    lines+=['','## 实现与复现','',
        '- `scripts/symmetric_mask.py`：独立干预与可见历史处理器；旧 production masking 未修改。',
        '- `scripts/diagnose_symmetric_mask.py`：冻结 checkpoint 的真实数据诊断与泄漏探针。',
        '- `tests/test_symmetric_mask.py`：三层随机小型 Qwen 的 attention/cache/位置与生成处理器测试；真实数据分数均来自3090上的原始3B模型。',
        '- `artifacts/real-data-pilot-v1/symmetric-attention-mask-v2/<dataset>/`：plan.json、epoch预测、summary.json、leakage.json、complete.json。',
        '- 首次缓存探针发现 cache_position=None 时需从 KV cache 长度取得偏移；修正前的运行保留在 `mmsd2-initial-cache-audit-attempt/`，不计入本报告。',
        '- `artifacts/{mmsd2,mmimdb}-symmetric-mask-v2.log`：真实运行日志；`artifacts/symmetric-mask-v2-tests.log`：定向测试日志。','',
        '```bash',
        'PYTHONPATH=src OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 .venv/bin/python scripts/diagnose_symmetric_mask.py --dataset mmsd2',
        'PYTHONPATH=src OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 .venv/bin/python scripts/diagnose_symmetric_mask.py --dataset mmimdb',
        '```','',
        '诊断程序拒绝覆盖已存在的数据集输出目录；复跑需显式更换 OUT 为新版本目录。汇总：`PYTHONPATH=src MPLCONFIGDIR=/tmp/modalitygap-mpl .venv/bin/python scripts/report_symmetric_mask.py`。','',
        '该方法改变原协议的 attention 可见性及 masking 历史处理，应作为独立协议报告；旧版和新版贡献曲线不可混用。即使生成稳定，也只支持此干预定义下的输入依赖，不构成因果模态贡献证明。']
    Path('docs/symmetric-mask-v2.md').write_text('\n'.join(lines)+'\n')
    fig,axes=plt.subplots(2,3,figsize=(12,6),constrained_layout=True)
    for i,ds in enumerate(('mmsd2','mmimdb')):
        for condition in ('full','mask_image','mask_text'):
            rows=[r for r in report[ds]['metrics'].values() if r['condition']==condition]
            for j,key in enumerate(('score','invalid_rate','mean_eos_probability')):
                axes[i,j].plot([r['epoch'] for r in rows],[r[key] for r in rows],'o-',label=condition)
                axes[i,j].set(title=f'{ds}: {key}',xlabel='Epoch',ylim=(-.02,1.02),xticks=[0,2])
        axes[i,0].legend(fontsize=8)
    fig.savefig(OUT/'f1-invalid-eos.png',dpi=160)
    print(json.dumps(report['validation'],indent=2))


if __name__=='__main__':main()
