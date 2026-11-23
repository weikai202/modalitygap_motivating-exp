"""Independent re-scoring, example transitions and bounded root-cause report."""
from collections import Counter
import json,math,re,shutil
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from modalitygap.protocol import LABELS,read_jsonl,write_json,metrics,contributions,digest
from modalitygap.runner import decode_prediction

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/contribution-cause-diagnostics-v1'
OLD=ROOT/'artifacts/real-expansion-symmetric-8epoch-v1'
TASKS=('crisismmd','hatefulmemes')


def confusion(rows,task):
    return {label:dict(Counter(r['prediction'][0] if r['prediction'] else 'INVALID' for r in rows if r['reference']==[label])) for label in LABELS[task]}


def transitions(full,masked):
    a={r['id']:r for r in full};b={r['id']:r for r in masked};assert set(a)==set(b)
    counts=Counter();changed=[]
    for k,r in a.items():
        left=r['prediction']==r['reference'];right=b[k]['prediction']==r['reference']
        counts[f'{"correct" if left else "wrong"}_to_{"correct" if right else "wrong"}']+=1
        if r['prediction']!=b[k]['prediction']:
            changed.append(dict(id=k,reference=r['reference'],full=r['prediction'],mask_text=b[k]['prediction'],full_correct=left,mask_text_correct=right))
    return dict(counts=dict(counts),changed=changed)


def bootstrap(full,masked,task):
    n=len(full);labels=LABELS[task];lookup={r['id']:r for r in masked};ordered=[lookup[r['id']] for r in full]
    features=[]
    for rows in (full,ordered):
        features.append(np.array([[[int(l in r['reference'] and l in r['prediction']),int(l not in r['reference'] and l in r['prediction']),int(l in r['reference'] and l not in r['prediction'])] for l in labels] for r in rows]))
    weights=np.random.default_rng(2026).multinomial(n,np.ones(n)/n,size=10000)
    scores=[]
    for x in features:
        counts=(weights@x.reshape(n,-1)).reshape(10000,len(labels),3);den=2*counts[:,:,0]+counts[:,:,1]+counts[:,:,2]
        scores.append(np.divide(2*counts[:,:,0],den,out=np.zeros_like(den,dtype=float),where=den!=0).mean(1))
    d=scores[0]-scores[1]
    return dict(percentile95=[float(x) for x in np.quantile(d,[.025,.975])],samples=10000,
                caveat='Paired empirical bootstrap conditional on observed examples; exploratory interval, not a proof of a population effect.')


def main():
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'artifacts/hf-cache/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3',use_fast=False,local_files_only=True)
    report={};decoded=0;probes_count=0
    for task in TASKS:
        folder=OUT/task;result=folder/'results';complete=json.loads((folder/'complete.json').read_text());assert complete['status']=='passed'
        rows=read_jsonl(folder/'dev128.jsonl');source={r['id']:r for r in rows}
        assert len(rows)==128 and len(source)==128
        for r in rows:
            a=r['original_record']
            if task=='crisismmd':assert r['labels']==[a['label']]==[a['label_text']]==[a['label_image']]
            else:assert r['labels']==[('not_hateful','hateful')[a['label']]]
        for path in result.glob('*.jsonl'):
            for r in read_jsonl(path):
                raw,pred,invalid,error=decode_prediction(tokenizer,task,r['generated_token_ids'])
                assert (raw,pred,invalid,error)==(r['raw_output'],r['prediction'],r['invalid'],r['decode_error'])
                decoded+=1
        records={c:read_jsonl(result/f'dev128_{c}.jsonl') for c in ('full','mask_image','mask_text')}
        summaries={}
        saved=json.loads((result/'summary.json').read_text())
        for subset,sl in [('original32',slice(0,32)),('additional96',slice(32,None)),('combined128',slice(None))]:
            scores={c:metrics(task,r[sl])['score'] for c,r in records.items()}
            for c,score in scores.items():assert math.isclose(score,saved[c][subset]['score'],abs_tol=1e-12)
            summaries[subset]=dict(**scores,**contributions(scores),transitions=transitions(records['full'][sl],records['mask_text'][sl]),
                full_confusion=confusion(records['full'][sl],task),mask_text_confusion=confusion(records['mask_text'][sl],task),
                text_contribution_bootstrap=bootstrap(records['full'][sl],records['mask_text'][sl],task))
        for c,rr in records.items():
            previous={r['id']:r for r in read_jsonl(OLD/task/'runs'/task/'L-low/seed_42/predictions'/f'epoch_8_{c}.jsonl')}
            assert all(r['generated_token_ids']==previous[r['id']]['generated_token_ids'] for r in rr[:32])
        probes=json.loads((result/'leakage-probes.json').read_text());assert len(probes)==16
        assert all(all(x==0 for x in r['prefill_and_three_cached_steps_raw_and_processed_max_deltas']) and r['full_control_max_logit_delta']>0 and r['attention_layer_checks']>0 for r in probes)
        probes_count+=len(probes)
        controls={}
        for length in (1,64):
            fixed=read_jsonl(result/f'fixed_body_{length}_mask_text.jsonl');assert {r['body_length'] for r in fixed}=={length}
            controls[f'fixed_body_{length}']=dict(score=metrics(task,fixed)['score'],transitions=transitions(records['mask_text'][:32],fixed))
        joint=read_jsonl(result/'both_modalities_blocked.jsonl');controls['mask_both']=metrics(task,joint)
        flipped=[]
        index={c:{r['id']:r for r in rr} for c,rr in records.items()}
        for r in summaries['original32']['transitions']['changed']:
            sid=r['id'];flipped.append(dict(r,text=source[sid]['text'],mask_image=index['mask_image'][sid]['prediction'],image=source[sid]['image']))
        report[task]=dict(subsets=summaries,controls=controls,flip_cases=flipped,probe_count=len(probes),mask_max_logit_delta=0.,original32_three_conditions_exact=True)
        if task=='crisismmd':
            cleaned=read_jsonl(result/'url_mentions_removed_full.jsonl');controls['url_mentions_removed']=dict(score=metrics(task,cleaned)['score'],transitions=transitions(records['full'][:32],cleaned))
            report[task]['mojibake_markers_original32']=sum(any(m in r['text'] for m in ('Â','Ã','â€','‚Ä')) for r in rows[:32])
        else:
            ocr=json.loads((result/'image-only-ocr.json').read_text());assert len(ocr)==8
            for r in ocr:assert r['reference_body']==source[r['id']]['text']
            report[task]['ocr']=dict(count=8,mean_reference_word_recall=sum(r['reference_word_recall'] for r in ocr)/8,exact_normalized_words=sum(r['exact_normalized_words'] for r in ocr))
            alt={}
            for subset,sl in [('original32',slice(0,32)),('combined128',slice(None))]:
                ss={}
                for c,rr in records.items():
                    changed=[dict(r,reference=['not_hateful']) if r['id']=='85761' else dict(r) for r in rr[sl]];ss[c]=metrics(task,changed)['score']
                alt[subset]=dict(**ss,**contributions(ss))
            report[task]['later_label_version_sensitivity']=alt
    c=report['crisismmd']['subsets']['original32']['transitions']['counts']
    assert c.get('wrong_to_correct',0)==4 and c.get('correct_to_wrong',0)==0
    h=report['hatefulmemes']['subsets']['original32']
    assert h['full_confusion']==h['mask_text_confusion']
    assert h['transitions']['counts'].get('wrong_to_correct',0)==h['transitions']['counts'].get('correct_to_wrong',0)==1
    old_validation=json.loads((OLD/'validation.json').read_text())
    assert old_validation['current_core_sha256']=={p.name:digest(p) for p in (ROOT/'src/modalitygap').glob('*.py')}
    assert digest(OLD/'epoch-report.json')==old_validation['report_sha256']
    old_hashes=json.loads((OLD/'preserved-old-results.json').read_text())
    assert all(digest(ROOT/path)==h for path,h in old_hashes.items())
    tests=ROOT/'artifacts/contribution-cause-joint-control-test.log';assert 'Ran 1 test' in tests.read_text() and tests.read_text().rstrip().endswith('OK')
    write_json(OUT/'diagnosis.json',report)
    write_json(OUT/'validation.json',dict(status='passed',classification_predictions_redecoded=decoded,real_checkpoint_content_probes=probes_count,
        exact_original_predictions_reproduced=192,production_core_unchanged=True,previous_expansion_report_unchanged=True,earlier_result_files_unchanged=len(old_hashes),
        joint_control_test_sha256=digest(tests),diagnostic_script_sha256=digest(ROOT/'scripts/diagnose_contribution_causes.py')))
    lines=['# CrisisMMD 负 text contribution / Hateful Memes 零贡献诊断','',
        '使用原64条训练样本得到的同一 epoch8 断点；没有训练或改动生产 masking。官方开发子集从原32条扩大到128条（保留原32＋新增96，每类新增48，seed2026，排除训练ID、归一化正文和RGB重复），没有使用test。新结果独立保存于 `artifacts/contribution-cause-diagnostics-v1/`。','',
        '当前证据未发现图文导出错位、标签反转或被遮内容进入decoder的实现错误。CrisisMMD 的负值来自该模型在部分样本上的图文融合干扰，且原32条子集放大了幅度；Hateful Memes 的0首先是两条预测变化抵消，图片里的正文又使单独文本输入具有冗余。这里的贡献是条件消融F1差，保留原序列长度、图像网格与位置。','',
        '## 统一复算','', '| 数据集 | 子集 | full F1 | 遮图F1 | 遮文F1 | image贡献(pp) | text贡献(pp) |','|---|---|---|---|---|---|---|']
    for task in TASKS:
        for subset,r in report[task]['subsets'].items():lines.append(f"| {task} | {subset} | {r['full']:.4f} | {r['mask_image']:.4f} | {r['mask_text']:.4f} | {100*r['image_contribution']:.2f} | {100*r['text_contribution']:.2f} |")
    lines+=['','## 数据与实现核对','',
        '扩展开发数据的标签逐条对照原发布记录；CrisisMMD 逐条断言 label == label_text == label_image。Hateful Memes 沿用原challenge二分类标签，已知ID85761与后续细粒度版本的分歧仍保留，并单独重评分。原32条×3条件×2任务共192条生成token逐条完全复现。','',
        '32个真实epoch8内容隔离probe（每任务8条×遮文/遮图）：替换全部被遮token或同形状像素，prefill及后续3个cache step的原始/处理后logits最大差均为0；full条件的相同替换均引起非零变化。每个probe都核验实际attention层隐藏keys不可达。说明在已测样本、原结构条件下内容屏蔽生效；正文长度和图像网格保留是协议定义的一部分。','',
        '新增joint blocking控制通过独立小Qwen测试，prefill/cache/repetition penalty均同时隔离两路内容；控制只用于本次诊断，不改变正式训练实现。','',
        '## CrisisMMD：四条错误与文本干扰','',
        '原32条full为28/32正确，遮文为32/32：4条均为错→对，无对→错。两条在遮图、遮文时都正确，full却错误；另外两条full延续遮图条件的错误。这是观察到的融合行为，不能进一步据此定位某个神经元或断言一般性的文本有害。','',
        '| ID | 参考 | full | 遮图（保留文本） | 遮文（保留图像） | 原图人工查看 |','|---|---|---|---|---|---|']
    visual={'910111529290301445_0':'飓风路径/强度预报图','920320104839237632_0':'人像照片','922558317049864192_0':'Tesla产品/住宅照片','930072830032666626_0':'CNN标志画面'}
    for r in report['crisismmd']['flip_cases']:lines.append(f"| {r['id']} | {r['reference'][0]} | {r['full'][0]} | {r['mask_image'][0]} | {r['mask_text'][0]} | {visual.get(r['id'],'')} |")
    c=report['crisismmd']['controls']
    lines += ['', f"将被遮正文长度统一为1或64 token，遮文F1均为 {c['fixed_body_1']['score']:.4f}/{c['fixed_body_64']['score']:.4f}；同时遮图遮文F1为 {c['mask_both']['score']:.4f}。原32条的满分不需要正文长度差异，且去掉图像内容后消失。",'',
        f"原32条有 {report['crisismmd']['mojibake_markers_original32']} 条含明显乱码标记，原发布记录本身已有这些字符。移除URL/@mention（未修复正文乱码）后full F1为 {c['url_mentions_removed']['score']:.4f}，改善2条、损害1条，仍未达到遮文F1。该控制同时改变正文内容和长度，不能把变化完全归因为URL语义。部分错误正文为正常ASCII，所以乱码也不能独立解释负贡献。",'',
        '扩大开发集后遮文不再满分，但额外96条及合计128条上text contribution仍为负；这是该checkpoint/该开发子集上的文本干扰证据，不能视为全量benchmark结论。','',
        '## Hateful Memes：精确抵消与图片OCR','',
        '原32条：full与遮文都是20/32正确，且混淆矩阵均为 TN=11、FP=5、FN=7、TP=9。30条类别预测一致；ID63987 对→错，ID67208 错→对，两次翻转都发生在 not_hateful 类，所以两个条件的macro-F1精确相同。0不是四舍五入，也不表示模型忽略了独立文本。','']
    h=report['hatefulmemes'];o=h['ocr'];hc=h['controls']
    lines += [f"原图单独OCR（没有提供正文或标签）8条，正文词召回平均 {o['mean_reference_word_recall']:.2%}，归一化词序列完全匹配 {o['exact_normalized_words']}/8。证明被遮的独立正文仍有一条可从图片恢复的通道；这支持冗余解释，但不是分类内部一定依赖OCR的因果证明。",'',
        f"固定被遮正文长度1/64 token后的F1为 {hc['fixed_body_1']['score']:.4f}/{hc['fixed_body_64']['score']:.4f}；联合遮蔽F1为 {hc['mask_both']['score']:.4f}。固定64 token使F1比原结构下降约3.02个百分点，显示长度/位置会影响该诊断数值；这属于保留结构的消融定义，不是隐藏正文语义仍可被读取的证据。",'',
        f"仅把ID85761按后续细粒度版本重评分，原32条text contribution仍为 {100*h['later_label_version_sensitivity']['original32']['text_contribution']:.2f} 个百分点；这一个标签版本差异不能解释原先的零值。",'',
        '## 复现与边界','',
        '准备数据：`.venv/bin/python scripts/prepare_contribution_diagnostics.py`（目录已存在时拒绝覆盖）。GPU运行：`OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 .venv/bin/python scripts/diagnose_contribution_causes.py --dataset crisismmd`，另一个任务将参数改为hatefulmemes；已完成目录拒绝覆盖。报告复算：`OMP_NUM_THREADS=1 .venv/bin/python scripts/report_contribution_causes.py`。','',
        '所有新分类预测再次从生成token重解码、复算分组F1与贡献。完整混淆矩阵、错对转移、配对bootstrap探索区间和敏感性结果见新目录diagnosis.json；判定依据和哈希见validation.json。生产源码与先前汇总报告SHA不变，204份更早结果文件SHA不变。','',
        '结论的范围是同一单seed、小规模训练得到的checkpoint及选中开发样本；没有把开发子集扩充当作新的训练或正式test，也没有通过调mask、改标签或换样本使贡献变正。']
    (ROOT/'docs/contribution-cause-diagnostics-v1.md').write_text('\n'.join(lines)+'\n')
    snapshot=OUT/'source-snapshot'
    for name in ('scripts/prepare_contribution_diagnostics.py','scripts/diagnose_contribution_causes.py','scripts/report_contribution_causes.py','tests/test_contribution_diagnostics.py'):
        target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    print(json.dumps({task:{s:r['text_contribution'] for s,r in v['subsets'].items()} for task,v in report.items()},indent=2));print('VALIDATED',decoded,'classification predictions;',probes_count,'real content probes')


if __name__=='__main__':main()
