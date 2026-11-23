"""Epoch-by-epoch contribution report for the complete real-data subset run."""
import csv,json,math
from pathlib import Path
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from modalitygap.protocol import read_jsonl,write_json,metrics,contributions,digest

ROOT=Path('artifacts/real-symmetric-8epoch-v1')


def main():
    report={};lines=['# 真实数据子集：对称 masking 完整8轮','',
        'RTX3090；Qwen2.5-VL-3B 固定 revision；L-low、seed42，从原始模型重新训练完整8轮。MMSD2.0 train64/dev32，MM-IMDb train63/dev32；无 test 评估。该小规模轨迹不是全量数据实验，也没有多配置/多seed确认。训练使用完整图文，对称干预只用于验证。','',
        'image contribution = full F1 − mask-image F1；text contribution = full F1 − mask-text F1。以下贡献乘100，以F1百分点展示，可为负；不是归一化百分比或因果贡献。每轮均保留非法率，不增加标签同义词。','']
    fig,axes=plt.subplots(2,4,figsize=(17,7),constrained_layout=True)
    for i,ds in enumerate(('mmsd2','mmimdb')):
        run=ROOT/'runs'/ds/'L-low/seed_42'
        config=json.loads((run/'config.json').read_text())
        status=json.loads((run/'progress.json').read_text())
        assert status['status']=='complete' and status['epoch']==8 and status['step']==32
        assert config['masking_method']=='symmetric-attention-v2'
        with (run/'metrics.csv').open() as f:raw=list(csv.DictReader(f))
        assert len(raw)==27
        lookup={(int(r['epoch']),r['condition']):r for r in raw}
        assert set(lookup)=={(e,c) for e in range(9) for c in ('full','mask_image','mask_text')}
        resources=json.loads((run/'resources.json').read_text());assert len(resources)==8
        assert all(math.isfinite(r['mean_loss']) for r in resources)
        with (run/'contributions.csv').open() as f:saved_contrib={int(r['epoch']):r for r in csv.DictReader(f)}
        table=[];first_eos=0;invalid_outputs={}
        for epoch in range(9):
            scores={};invalid={}
            for condition in ('full','mask_image','mask_text'):
                rows=read_jsonl(run/'predictions'/f'epoch_{epoch}_{condition}.jsonl');assert len(rows)==32
                score=metrics(ds,rows)
                assert math.isclose(score['score'],float(lookup[epoch,condition]['score']),abs_tol=1e-12)
                scores[condition]=score['score'];invalid[condition]=score['invalid_rate']
                first_eos+=sum(r['generated_token_ids'][0] in (151645,151643) for r in rows)
                invalid_outputs[f'{epoch}/{condition}']=dict(Counter(r['raw_output'] for r in rows if r['invalid']))
            derived=contributions(scores)
            for k,v in derived.items():assert math.isclose(v,float(saved_contrib[epoch][k]),abs_tol=1e-12)
            table.append(dict(epoch=epoch,**scores,**derived,invalid_rate=invalid,
                              loss=None if epoch==0 else resources[epoch-1]['mean_loss']))
        reproduction=[]
        old=Path('artifacts/real-data-pilot-v1/symmetric-attention-mask-v2')/ds
        for epoch in (0,2):
            for condition in ('full','mask_image','mask_text'):
                a={r['id']:r for r in read_jsonl(old/f'epoch_{epoch}_{condition}.jsonl')}
                b=read_jsonl(run/'predictions'/f'epoch_{epoch}_{condition}.jsonl')
                exact=sum(r['generated_token_ids']==a[r['id']]['generated_token_ids'] for r in b)
                reproduction.append(dict(epoch=epoch,condition=condition,count=len(b),exact_tokens=exact))
        report[ds]=dict(integration_reproduction=reproduction,config=dict(masking_method=config['masking_method'],device=config['device'],revision=config['revision'],
            seed=config['seed'],data_manifest=config['data_manifest'],stats_sha256=config['stats_sha256'],source_sha256=config['source_sha256']),
            epochs=table,resources=resources,first_eos_count=first_eos,predictions=864,
            invalid_outputs=invalid_outputs,completed_steps=32)
        lines += [f'## {ds}', '', '| epoch | loss | full F1 | mask-image F1 | mask-text F1 | image贡献(pp) | text贡献(pp) | 非法率 full / image-mask / text-mask |',
                  '|---|---|---|---|---|---|---|---|']
        for r in table:
            loss='—' if r['loss'] is None else f"{r['loss']:.5f}"
            inv=r['invalid_rate']
            lines.append(f"| {r['epoch']} | {loss} | {r['full']:.4f} | {r['mask_image']:.4f} | {r['mask_text']:.4f} | {100*r['image_contribution']:.2f} | {100*r['text_contribution']:.2f} | {inv['full']:.2%} / {inv['mask_image']:.2%} / {inv['mask_text']:.2%} |")
        lines+=['',f'共864条预测，首token EOS计数：{first_eos}。','']
        for c in ('full','mask_image','mask_text'):
            axes[i,0].plot(range(9),[r[c] for r in table],'o-',label=c)
            axes[i,2].plot(range(9),[r['invalid_rate'][c] for r in table],'o-',label=c)
        for c in ('image_contribution','text_contribution'):
            axes[i,1].plot(range(9),[100*r[c] for r in table],'o-',label=c)
        axes[i,1].axhline(0,color='gray',lw=.7)
        axes[i,3].plot(range(1,9),[r['mean_loss'] for r in resources],'o-')
        for j,name in enumerate(('F1','Contribution (F1 pp)','Invalid fraction','Train loss')):
            axes[i,j].set(title=f'{ds}: {name}',xlabel='Epoch',xticks=range(9))
            if j<3:axes[i,j].legend(fontsize=7)
        axes[i,0].set_ylim(-.02,1.02);axes[i,2].set_ylim(-.02,1.02)
    report['validation']=dict(predictions=1728,epochs=list(range(9)),conditions_per_epoch=3,
        independent_workspace=True,config_sha256=digest('configs/real-symmetric-8epoch.json'))
    test_paths=['artifacts/symmetric-training-full-gpu-tests.log','artifacts/symmetric-training-targeted-gpu-tests.log']
    audit=json.loads((ROOT/'prediction-audit.json').read_text())
    assert audit['status']=='passed' and audit['examples']==1728
    report['validation']['prediction_audit']=audit
    report['validation']['source_snapshot']='source-snapshot/'
    report['validation']['tests']={p:dict(passed=Path(p).read_text().rstrip().endswith('\nOK'),sha256=digest(p)) for p in test_paths}
    assert all(r['passed'] for r in report['validation']['tests'].values())
    write_json(ROOT/'epoch-report.json',report);write_json('docs/real-symmetric-8epoch-results.json',report)
    fig.savefig(ROOT/'contribution-trajectories.png',dpi=160)
    lines+=['## 观察与限制','', 'MMSD2.0：image贡献在第2轮后大多约13–20个百分点，非单调增长；最终20.40个百分点。text贡献第2轮达到15.45个百分点，后期多为5.85–9.27个百分点。前两轮非法率很高，早期差分明显混入格式学习。Full F1的最高值出现在第2轮（0.7729），第8轮为0.7117，不能说训练越久整体效果越好。', '', 'MM-IMDb：image贡献在约−1.84到+1.48个百分点间跨零波动，第8轮接近零（+0.15）；text贡献由接近零转为后期约+2.38到+3.73个百分点。没有观察到稳定的图像差分增长。text-mask后期仍有12.5%非法输出，因此不能将差分解释成纯语义贡献。', '', '本次只有一个配置和seed，标准analysis中的单配置排名/negative_result字段不构成已检验的排名反转结论。小样本差分也不足以证明模型没有利用某一模态。', '', '## 验证','', '全部36项CUDA回归测试通过，无跳过；新增扩展测试还验证新协议两个任务的精确恢复、跨协议拒绝，以及held-out自动沿用checkpoint协议。逐条预测重新计算的F1及贡献与CSV完全一致。日志和哈希见结果JSON的validation。', '', '## 复现与文件','',
        '运行：`HF_HUB_OFFLINE=1 bash scripts/run_experiment.sh --config configs/real-symmetric-8epoch.json`；中断后追加 `--resume`。已完成作业经完整性核对后跳过；新实验需更换配置中的workspace。','',
        '本次完整一键 workspace：`artifacts/real-symmetric-8epoch-v1/`。每轮指标和贡献位于 `runs/<dataset>/L-low/seed_42/{metrics,contributions}.csv`，预测、干预日志及epoch0–8断点在同目录。loss和显存记录在resources.json。标准分析位于figures/；此汇总位于epoch-report.json和contribution-trajectories.png。','',
        '重新生成本报告：`PYTHONPATH=src MPLCONFIGDIR=/tmp/modalitygap-mpl .venv/bin/python scripts/report_symmetric_training.py`。旧结果仍在原workspace，未混入此分析。','',
        '注意：text-mask保留图片内文字，序列长度/图像网格仍保留；F1差分同时受格式合规率影响。若后期非法率下降，贡献变化也可能包含格式学习，而非纯粹模态语义利用变化。']
    Path('docs/real-symmetric-8epoch.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({ds:report[ds]['epochs'] for ds in ('mmsd2','mmimdb')},indent=2))


if __name__=='__main__':main()
