"""Recompute all scores and attention audits; report the three isolated eight-epoch pilots."""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from modalitygap.protocol import CONDITIONS, contributions, digest, metrics, read_jsonl, write_json

TASKS=('newsclippings','crisismmd','hatefulmemes')


def report(root):
    result={}
    fig,axes=plt.subplots(3,4,figsize=(17,10),constrained_layout=True)
    for i,task in enumerate(TASKS):
        workspace=root/task;run=workspace/'runs'/task/'L-low/seed_42'
        status=json.loads((run/'progress.json').read_text());assert status['status']=='complete' and status['epoch']==8 and status['step']==32
        config=json.loads((run/'config.json').read_text());assert config['masking_method']=='symmetric-attention-v2'
        with (run/'metrics.csv').open() as f:saved={(int(r['epoch']),r['condition']):r for r in csv.DictReader(f)}
        assert len(saved)==27
        with (run/'contributions.csv').open() as f:derived_saved={int(r['epoch']):r for r in csv.DictReader(f)}
        resources=json.loads((run/'resources.json').read_text());assert len(resources)==8
        assert all(math.isfinite(r['mean_loss']) for r in resources)
        table=[];first_eos=0;invalid_outputs={};attention_checks=0
        expected_ids=None
        for e in range(9):
            scores={};invalid={};length={};predictions={}
            for condition in CONDITIONS:
                rows=read_jsonl(run/'predictions'/f'epoch_{e}_{condition}.jsonl');assert len(rows)==32
                ids=[r['id'] for r in rows];predictions[condition]={r['id']:r['prediction'] for r in rows}
                if expected_ids is None:expected_ids=ids
                assert ids==expected_ids
                score=metrics(task,rows)
                for field in ('score','macro_f1','invalid_rate','accuracy'):
                    assert math.isclose(score[field],float(saved[e,condition][field]),abs_tol=1e-12)
                scores[condition]=score['score'];invalid[condition]=score['invalid_rate'];length[condition]=score['output_length']
                first_eos+=sum(bool(r['generated_token_ids']) and r['generated_token_ids'][0] in (151645,151643) for r in rows)
                invalid_outputs[f'{e}/{condition}']=dict(Counter(r['raw_output'] for r in rows if r['invalid']))
            calculated=contributions(scores)
            for key,value in calculated.items():assert math.isclose(value,float(derived_saved[e][key]),abs_tol=1e-12)
            logs=read_jsonl(run/'interventions'/f'epoch_{e}.jsonl')
            # Production run performs one full-condition sanity forward before epoch0.
            expected=dict(full=33 if e==0 else 32,mask_image=32,mask_text=32)
            assert len(logs)==sum(expected.values())
            assert Counter(r['condition'] for r in logs)==expected
            for item in logs:
                if item['condition']=='full':assert item['hidden_positions']==[]
                else:
                    assert item['hidden_positions'] and item['decoder_calls'] and item['attention_layer_checks']>0
                    assert all(call['blocked']==len(item['hidden_positions']) for call in item['decoder_calls'])
                attention_checks+=item['attention_layer_checks']
            agreement={c:sum(predictions['full'][sid]==predictions[c][sid] for sid in expected_ids)/len(expected_ids) for c in CONDITIONS[1:]}
            table.append(dict(epoch=e,**scores,**calculated,invalid_rate=invalid,mean_output_length=length,full_prediction_agreement=agreement,
                         mean_loss=None if e==0 else resources[e-1]['mean_loss']))
        manifest=json.loads((workspace/'manifests'/task/'manifest.json').read_text())
        assert manifest['counts']==dict(train=64,dev=32,test=0)
        assert not manifest['excluded']
        result[task]=dict(epochs=table,first_token_eos_count=first_eos,predictions=864,attention_layer_checks=attention_checks,
             invalid_outputs=invalid_outputs,resources=resources,manifest_sha256=digest(workspace/'manifests'/task/'manifest.json'),
             config=config,source_provenance=json.loads((workspace/'pipeline.json').read_text()))
        if task=='hatefulmemes':
            result[task]['label_version_sensitivity']=json.loads((workspace/'label-version-sensitivity.json').read_text())
        for c in CONDITIONS:
            axes[i,0].plot(range(9),[r[c] for r in table],'o-',label=c)
            axes[i,2].plot(range(9),[100*r['invalid_rate'][c] for r in table],'o-',label=c)
        for modality in ('image','text'):axes[i,1].plot(range(9),[100*r[modality+'_contribution'] for r in table],'o-',label=modality)
        axes[i,1].axhline(0,color='gray',lw=.7)
        axes[i,3].plot(range(1,9),[r['mean_loss'] for r in resources],'o-',color='black')
        for j,title in enumerate(('Macro-F1','Contribution (F1 pp)','Invalid outputs (%)','Training loss')):
            axes[i,j].set_title(f'{task}: {title}');axes[i,j].set_xlabel('Epoch');axes[i,j].grid(alpha=.2)
            if j<3:axes[i,j].legend(fontsize=8)
    fig.savefig(root/'contribution-trajectories.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4),sharey=True,constrained_layout=True)
    for ax,task in zip(axes,TASKS):
        for modality in ('image','text'):
            ax.plot(range(9),[100*r[modality+'_contribution'] for r in result[task]['epochs']],'o-',label=modality)
        ax.axhline(0,color='gray',lw=.8);ax.set_title(task);ax.set_xlabel('Epoch');ax.set_xticks(range(9));ax.grid(alpha=.2);ax.legend()
    axes[0].set_ylabel('Contribution (Macro-F1 percentage points)')
    fig.savefig(root/'contribution-comparison.png',dpi=180);plt.close(fig)
    write_json(root/'epoch-report.json',result)
    lines=['# 三个新增数据集：对称 masking 8 epoch pilot','',
       'RTX 3090；Qwen2.5-VL-3B 固定 revision；L-low、seed42；每个任务64 train / 32 dev，完整8轮、每轮4次优化器更新。epoch0是训练前基线。训练用完整图文；对称均值占位＋attention屏蔽只用于验证。两种mask都保留原序列长度、图像网格和位置，因此贡献是在原结构条件下测得的性能差。没有 test 训练或评估。','',
       '全部任务使用严格类别标记生成与 macro-F1。image贡献 = full F1 − mask-image F1；text贡献 = full F1 − mask-text F1。表中贡献单位为F1百分点，不是归一化占比或因果归因。负数保留。','',
       '图片均为发布文件：NewsCLIPpings 从 VisualNews 作者 TAR 按原始字节提取；CrisisMMD 来自 QCRI 官方仓库；Hateful Memes 来自固定公开镜像，标注已与作者后续发布核对，但无官方压缩包校验值可独立认证 PNG 字节。所有保留图片均完整解码，选中 train/dev 无相同 RGB 或归一化正文跨划分重复。','',
       'NewsCLIPpings 使用 merged_balanced，保留32/16个完整正负caption对；bootstrap按配对正文分组。CrisisMMD 使用作者 informativeness 的 text/image-agreed 标签及划分，未把图像单独标签冒充共同标签。Hateful Memes 使用原始二分类标签及 dev_seen；选中的 dev ID 85761 与后续细粒度版本标签不同，明确沿用原始challenge标签。','',
       '**Hateful Memes 的 PNG 自带正文：mask_text 只移除独立正文输入，图片内的文字仍可被 OCR 读取。该任务的 text contribution 是额外文本输入的边际性能差，不能解释成全部语言信息贡献。** 原始图像保持不变。Hateful Memes 官方主指标是 AUROC，此处 macro-F1 用于统一探索性诊断，不作为官方排行榜结果。','']
    lines += ['## 跨任务比较','',
        '| 数据集 | full F1：epoch0 → 8 | image贡献(pp)：epoch0 → 8 | text贡献(pp)：epoch0 → 8 |',
        '|---|---|---|---|']
    for task in TASKS:
        first,last=result[task]['epochs'][0],result[task]['epochs'][-1]
        lines.append(f"| {task} | {first['full']:.4f} → {last['full']:.4f} | {100*first['image_contribution']:.2f} → {100*last['image_contribution']:.2f} | {100*first['text_contribution']:.2f} → {100*last['text_contribution']:.2f} |")
    lines += ['', '共同纵轴的贡献对比图：`artifacts/real-expansion-symmetric-8epoch-v1/contribution-comparison.png`；含F1、非法率和loss的完整图：同目录 `contribution-trajectories.png`。','']
    for task in TASKS:
        r=result[task];lines += [f'## {task}','',
         '| epoch | loss | full F1 | mask-image F1 | mask-text F1 | image贡献(pp) | text贡献(pp) | 非法率 full/image/text |',
         '|---|---|---|---|---|---|---|---|']
        for row in r['epochs']:
            loss='—' if row['mean_loss'] is None else f"{row['mean_loss']:.5f}";inv=row['invalid_rate']
            lines.append(f"| {row['epoch']} | {loss} | {row['full']:.4f} | {row['mask_image']:.4f} | {row['mask_text']:.4f} | {100*row['image_contribution']:.2f} | {100*row['text_contribution']:.2f} | {inv['full']:.1%}/{inv['mask_image']:.1%}/{inv['mask_text']:.1%} |")
        lines += ['',f"864条预测；首token EOS：{r['first_token_eos_count']}；实际attention层屏蔽核验：{r['attention_layer_checks']}次。",'']
    h=result['hatefulmemes']
    alt=h['label_version_sensitivity']['epochs'][-1]
    agreement=h['epochs'][-1]['full_prediction_agreement']['mask_text']
    lines += ['## Hateful Memes 标签与预测敏感性','',
        f"最终 full 与 mask_text 的 F1 相同，但只有 {round(agreement*32)}/32 条类别预测相同；text contribution 为0不代表模型完全不使用独立文本。",
        f"仅按作者后续细粒度版本把 dev ID85761 重评分为 not_hateful，最终 full F1 为 {alt['full']:.4f}，image/text贡献为 {100*alt['image_contribution']:.2f}/{100*alt['text_contribution']:.2f} 个百分点。主表、训练标签和所有预测保持原始challenge版本；该检查只改变这一条开发参考标签。",'']
    lines += ['## 验证与复现','',
       '每条生成token通过 `scripts/audit_predictions.py` 独立重解码，重新计算所有CSV指标；本报告再次重算各轮贡献并核验实际attention屏蔽日志。40项CUDA回归通过且无跳过，另有1项TAR分段边界测试通过。真实输入审计验证288条样本的推理tensor不依赖参考标签；数据审计验证每条图文映射及哈希、划分、训练均值样本ID。完整CUDA回归日志：`artifacts/expansion-v1-gpu-tests-threads1.log`。','',
       '运行：`HF_HUB_OFFLINE=1 bash scripts/run_expansion.sh --resume`。首次使用空workspace时去掉 `--resume`；本次workspace已初始化，使用 `--resume` 会校验签名并跳过完成作业。中断恢复同命令。数据在 `data/real-expansion-v1/<dataset>/`，原始标注、下载进度和TAR索引在 `data/downloads/expansion-v1/`。','',
       '输出在 `artifacts/real-expansion-symmetric-8epoch-v1/<dataset>/`；训练目录 `runs/<dataset>/L-low/seed_42/` 保存metrics.csv、contributions.csv、predictions/、interventions/、resources.json及checkpoints/epoch_0..8。每个数据集的 figures/ 是标准分析，logs/ 是阶段日志。根目录另有 epoch-report.json、contribution-trajectories.png及源码快照。旧结果目录保持原样。','',
       '这是小样本、单配置、单seed诊断；32条开发数据和News配对减少了独立样本数。F1变化受格式合规率及离散类别翻转影响，不能据此确认泛化趋势或排名反转。']
    Path('docs/expansion-symmetric-8epoch-v1.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({task:r['epochs'][-1] for task,r in result.items()},indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('artifacts/real-expansion-symmetric-8epoch-v1'));a=ap.parse_args();report(a.root)
