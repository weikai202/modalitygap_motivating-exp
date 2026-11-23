"""Paired, sample-size-aware summaries of frozen real-data mask diagnostics."""
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from modalitygap.protocol import digest, metrics, read_jsonl, write_json

ROOT=Path('artifacts/real-data-pilot-v1')
OUT=ROOT/'mask-diagnostics-v1'


def summarize(rows,dataset):
    result=metrics(dataset,rows)
    for kind in ('eos_probability','processed_eos_probability'):
        values=np.array([r['first_token'][kind] for r in rows])
        result[kind]=dict(mean=float(values.mean()),median=float(np.median(values)),
                          min=float(values.min()),max=float(values.max()))
    result['raw_first_im_end_count']=sum(r['first_token']['raw_argmax_id']==151645 for r in rows)
    result['first_im_end_count']=sum(r['first_token']['argmax_id']==151645 for r in rows)
    result['first_endoftext_count']=sum(r['first_token']['argmax_id']==151643 for r in rows)
    result['invalid_outputs']=dict(Counter(r['raw_output'] for r in rows if r['invalid']))
    result['generated_lengths']=dict(Counter(r['generated_length'] for r in rows))
    result['embeddings']={m:{phase:{key:float(np.mean([r['embeddings'][m][phase][key] for r in rows]))
        for key in ('mean_token_norm','rms_token_norm','mean_vector_norm','centered_rms')}
        for phase in ('original','replacement')} for m in ('text','image')}
    return result


def main():
    report={};baseline_audit={};records=0
    for dataset in ('mmsd2','mmimdb'):
        directory=OUT/dataset
        plan=json.loads((directory/'plan.json').read_text())
        probe=set(plan['probe_ids'])
        result={};paired={}
        for epoch in (0,2):
            result[str(epoch)]={};paired[str(epoch)]={}
            for c in plan['conditions']:
                rows=read_jsonl(directory/f"epoch_{epoch}_{c['name']}.jsonl")
                expected=probe if c.get('probe') else set(plan['dev_ids'])
                assert {r['id'] for r in rows}==expected and len(rows)==len(expected)
                assert all(r['embeddings']['protected_tokens_identical'] for r in rows)
                assert all(r['first_token']['argmax_id']==r['generated_token_ids'][0] for r in rows)
                assert all(r['donor_id']!=r['id'] for r in rows if r['donor_id'])
                result[str(epoch)][c['name']]=summarize(rows,dataset)
                paired[str(epoch)][c['name']]=summarize([r for r in rows if r['id'] in probe],dataset)
                records+=len(rows)
                if c['name'] in ('full','image_noise_2026','text_noise_2026'):
                    variant={'full':'full','image_noise_2026':'mask_image','text_noise_2026':'mask_text'}[c['name']]
                    path=ROOT/'runs'/dataset/'L-low'/'seed_42_pilot'/'predictions'/f'epoch_{epoch}_{variant}.jsonl'
                    original={r['id']:r for r in read_jsonl(path)}
                    assert all(r['generated_token_ids']==original[r['id']]['generated_token_ids'] for r in rows)
                    baseline_audit[f'{dataset}/epoch{epoch}/{variant}']=dict(count=len(rows),all_tokens_exact=True)
        report[dataset]=dict(plan=plan,main=result,paired_eight=paired)
    write_json(OUT/'report.json',report)
    write_json('docs/mask-diagnostics-results.json',report)
    audit=dict(status='passed',examples=records,files=56,
        original_pilot_reproduction=baseline_audit,all_protected_embeddings_identical=True,
        all_first_generated_tokens_match_processed_argmax=True,no_self_shuffle_pairs=True,
        diagnostic_source_sha256=digest('scripts/diagnose_real_masks.py'),
        instrumentation_tests=dict(passed=2,log='artifacts/mask-diagnostic-instrumentation-tests.log',log_sha256=digest('artifacts/mask-diagnostic-instrumentation-tests.log')))
    write_json(OUT/'validation.json',audit)
    write_json('docs/mask-diagnostics-validation.json',audit)
    names=['full','image_noise_2026','text_mean_only','text_noise_2026','text_noise_2027','text_noise_2028','shuffle_image','shuffle_text']
    fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    for i,ds in enumerate(('mmsd2','mmimdb')):
        r=report[ds]['main']['2'];y=np.arange(len(names))
        for j,(key,title) in enumerate((('processed_eos_probability','First-token EOS probability (processed)'),('invalid_rate','Invalid output fraction'),('score','Macro F1' if ds=='mmsd2' else 'Micro F1'))):
            values=[r[n][key]['mean'] if key.endswith('probability') else r[n][key] for n in names]
            axes[i,j].barh(y,values)
            axes[i,j].set(yticks=y,yticklabels=names if j==0 else [],xlim=(0,1.02),title=f'{ds}: {title}')
            axes[i,j].invert_yaxis()
    fig.savefig(OUT/'eos-invalid-f1.png',dpi=160)
    fig2,axs=plt.subplots(1,2,figsize=(12,5.5),constrained_layout=True)
    for ax,ds in zip(axs,('mmsd2','mmimdb')):
        for name,marker in zip(('full','text_mean_only','text_noise_2026','text_noise_std1','text_noise_normmatched','text_mean_normmatched'),('o','s','^','v','D','x')):
            r=report[ds]['paired_eight']['2'][name]
            x=r['embeddings']['text']['replacement']['rms_token_norm']
            y=r['processed_eos_probability']['mean']
            ax.scatter(x,y,marker=marker,label=name.replace('text_',''))
        ax.set(xlabel='Text replacement RMS token norm',ylabel='Mean first-token EOS probability',title=f'{ds}: paired eight examples',ylim=(-.03,1.05),xlim=(.1,1.0))
        ax.legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=2,fontsize=8)
    fig2.savefig(OUT/'norm-eos-mechanism.png',dpi=160)
    lines=['# 真实数据 masking 诊断','',
        'RTX3090；固定真实开发子集，各32条；L-low seed42 的 epoch0/epoch2。主要条件使用全部32条；额外机制对照只使用预先固定 seed2026 的8条，下面分表报告，避免把不同样本量混在一起。未重新训练模型、未改变正式 masking 协议。',
        '',
        '诊断结果支持：当前 mean+0.05std 噪声替换在正文通道形成均值主导、token间变化过小的输入，触发提前终止；范数缩小并非唯一因素。它在原始 epoch0 已出现，跨 seed 和 checkpoint 复现。恢复方差比单纯恢复范数有效，但不能完全恢复正常格式，尤其是 MM-IMDb。当前 text-mask F1 落差不能直接解释为可靠的文本贡献。',
        '',
        'EOS 概率直接取实际 `generate` 的首步：同时保存处理前 logits softmax 和处理后 scores softmax。模型固定 generation config 含 repetition_penalty=1.05；实际首 token 取处理后 argmax。EOS 指 `<|im_end|>` (151645)，另行统计 `<|endoftext|>` (151643)。',
        '',
        '## 最终 checkpoint：32条主要诊断','',
        '| 数据 | 条件 | F1 | 非法率 | 首步 EOS 概率（原始/处理后均值） | 首步 EOS 数 |',
        '|---|---|---|---|---|---|']
    for ds in ('mmsd2','mmimdb'):
        for c in report[ds]['plan']['conditions']:
            if c.get('probe'):continue
            n=c['name'];r=report[ds]['main']['2'][n]
            lines.append(f"| {ds} | {n} | {r['score']:.4f} | {r['invalid_rate']:.2%} | {r['eos_probability']['mean']:.5g} / {r['processed_eos_probability']['mean']:.5g} | {r['first_im_end_count']}/32 |")
    lines+=['','## 最终 checkpoint：相同8条机制对照','',
        'normmatched 用冻结训练统计量的 population RMS 为固定目标，只缩放替换向量，不读取当前样本原正文 embedding 或标签。std1 将系数从0.05改为1.0，恢复每维统计方差，但不恢复词汇或句子结构。这些仅用于定位机制，不能拿来挑选更好看的正式消融。','',
        '| 数据 | 条件 | F1 | 非法率 | EOS 概率（处理后均值） | 首步 EOS 数 | 正文原/替换 mean token norm | 原/替换 centered RMS |',
        '|---|---|---|---|---|---|---|---|']
    probe_names=['full','text_mean_only','text_noise_2026','text_noise_std1','text_noise_normmatched','text_mean_normmatched','shuffle_text']
    for ds in ('mmsd2','mmimdb'):
        for n in probe_names:
            r=report[ds]['paired_eight']['2'][n];e=r['embeddings']['text']
            lines.append(f"| {ds} | {n} | {r['score']:.4f} | {r['invalid_rate']:.2%} | {r['processed_eos_probability']['mean']:.5g} | {r['first_im_end_count']}/8 | {e['original']['mean_token_norm']:.4f} / {e['replacement']['mean_token_norm']:.4f} | {e['original']['centered_rms']:.4f} / {e['replacement']['centered_rms']:.4f} |")
    lines+=['','## 定位结论与解释边界','',
        '1. 崩溃在 epoch0 已存在；对照 epoch2 可检查其是否由本次 LoRA 更新引入。mean-only 与多个噪声 seed 的一致行为用于排除单个随机 seed 偶然性。',
        '2. 原始 logits 与处理后 scores 分开保存，避免把 repetition penalty 的影响当作模型本身的 EOS 偏好。直接记录的首 token EOS 概率与实际生成长度用于区分提前终止和后处理格式失败。mask-text 的首步 EOS 不能由放宽标签 parser 或增加 max_new_tokens 修复；不改动正式 parser。',
        '3. 仅恢复 RMS 范数和恢复每维噪声方差是两种不同的对照。原系数0.05使每维方差只有原统计方差的0.25%，token 会集中在共同均值附近；整体放大向量不能改变这种均值方向占主导的关系。std1 恢复每维方差，也改变向量方向分布；如果它解除 EOS 而范数匹配未解除，证据支持“均值主导、变化不足的替换分布”，而非仅仅范数小。它不能唯一识别模型内部的所有终止机制。',
        '4. 打乱正文配对保留真实正文的边际分布，但改变单个样本的长度与语义配对；不是严格控制全部因素的干预。它主要检查模型是否仍能正常生成，不能把其 F1 当作可靠的单模态成绩。',
        '5. 噪声系数1.0的对照还出现过不在固定词表中的标签、重复 `<|im_start|>`，个别回答达到生成长度上限；因此减少首步 EOS 并不等于恢复稳定的任务输出。对照解除 EOS 不等于得到中立的 missing-modality 分布，额外机制对照只有固定8条。不能据本诊断将系数1.0直接替换正式协议、宣称消融已修好或报告因果模态贡献。额外训练 noise encoder 也不是这些结果推出的必要步骤。',
        '',
        '## 核对与文件','',
        f'共56份逐条预测、{records}条诊断记录。原 pilot 的 full、mask_image、mask_text 在两个 checkpoint、两个数据集上共384条生成 token 序列全部精确复现。所有条件受保护位置 embedding 完全相同，所有实际首 token 与生成处理后分数 argmax 一致，打乱配对没有自配对。',
        '',
        '- 完整 epoch0/2、EOS 分布、embedding 均值/范数/中心化波动、非法原始回答和 probe 子集：`artifacts/real-data-pilot-v1/mask-diagnostics-v1/report.json`。',
        '- 每条记录及预先固定的 donor/probe ID：`mask-diagnostics-v1/<dataset>/epoch_<epoch>_<condition>.jsonl` 和 `plan.json`。shuffle_image 保留原正文与原标签，替换图像；shuffle_text 保留原图与原标签，替换正文。均为开发集内部固定无自配对循环置换。',
        '- 诊断运行：`PYTHONPATH=src OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 .venv/bin/python scripts/diagnose_real_masks.py --dataset mmsd2`（另跑 mmimdb）；已有完整条件可复用。',
        '- 汇总重建：`PYTHONPATH=src MPLCONFIGDIR=/tmp/modalitygap-mpl .venv/bin/python scripts/report_mask_diagnostics.py`。',
        '- 新增 instrumentation 测试检查与 production hook 完全一致、范数对照不修改 prompt/cache、完整28条件输出及恢复复用。']
    Path('docs/mask-diagnostics.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
