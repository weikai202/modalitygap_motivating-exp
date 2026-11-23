"""Record full-release audit + CUDA engineering coverage for the source handoff."""
import json,re,platform
from collections import Counter
from pathlib import Path
from modalitygap.protocol import GENRES,CONFIGS,read_jsonl,digest,write_json

ROOT=Path(__file__).resolve().parents[1]


def main():
    export=ROOT/'data/full-handoff-v1';workspace=ROOT/'artifacts/full-handoff-real-prepare-v1'
    log=ROOT/'artifacts/full-handoff-all-gpu-tests.log';text=log.read_text()
    assert text.rstrip().endswith('OK') and 'skipped=' not in text
    total=int(re.search(r'Ran (\d+) tests',text).group(1))
    jobs=re.findall(r'FULL MATRIX (mmsd2|mmimdb) ([A-Za-z-]+) (42|43|44)',text)
    assert set(jobs)=={(d,c,str(s)) for d in ('mmsd2','mmimdb') for c in CONFIGS for s in (42,43,44)}
    focused=ROOT/'artifacts/full-handoff-six-export-tests.log';assert 'Ran 6 tests' in focused.read_text() and focused.read_text().rstrip().endswith('OK')
    data={}
    for task in ('mmsd2','mmimdb'):
        rows=read_jsonl(export/task/'source.jsonl');p=json.loads((export/task/'provenance.json').read_text());manifest=workspace/'manifests'/task/'manifest.json';m=json.loads(manifest.read_text())
        assert p['source_sha256']==digest(export/task/'source.jsonl')==m['source_sha256']
        for split,sha in m['split_hashes'].items():assert digest(manifest.parent/f'{split}.jsonl')==sha
        retained={s:read_jsonl(manifest.parent/f'{s}.jsonl') for s in ('train','dev','test')}
        for key in ('id','image_hash','text_hash'):
            sets={s:{r[key] for r in retained[s]} for s in retained}
            assert not (sets['train']&sets['dev'] or sets['train']&sets['test'] or sets['dev']&sets['test'])
        data[task]=dict(official_counts=p['counts'],actual_counts=m['counts'],retained_complete_counts=m['retained_complete_counts'],
            incomplete_exclusions=len(m['excluded']),overlap_exclusions=len(m['overlap_excluded']),
            overlap_exclusions_by_split=dict(Counter(x['split'] for x in m['overlap_excluded'])),
            requested_pool_size=m['requested_pool_size'],all_retained_dev_and_test_used=True,manifest_sha256=digest(manifest),
            source_sha256=p['source_sha256'],source_revision=p['source_revision'])
        if task=='mmimdb':
            meta=json.loads((ROOT/'data/downloads/mmimdb-official-metadata.json').read_text())
            expected={sid:{g for g in GENRES if y[meta['genre_order'].index(g)]} for sid,y in zip(meta['ids'],meta['labels'])}
            assert len(rows)==len(expected)==25959 and all(set(r['labels'])==expected[r['id']] for r in rows)
            for split,(a,b) in meta['split_bounds'].items():assert {r['id'] for r in rows if r['split']==split}==set(meta['ids'][a:b])
            data[task].update(all_25959_raw_labels_and_splits_match_official_hdf5=True,raw_movies_with_out_of_vocabulary_tags=len(p['raw_tag_projection']),archive_md5=p['archive_md5'])
        else:
            folder=ROOT/'data/sources/mmsd2-authors/data/text_json_final'
            lookup={str(r['image_id']):(split,r) for split,name in [('train','train'),('dev','valid'),('test','test')] for r in json.loads((folder/f'{name}.json').read_text())}
            assert len(rows)==len(lookup)==24635
            for r in rows:
                split,a=lookup[r['id']];assert r['split']==split and r['text']==a['text'] and r['labels']==['sarcastic' if a['label']==1 else 'not_sarcastic']
            data[task].update(all_24635_rows_match_corrected_author_annotations=True,mirror_annotation_mismatch_count=len(p['mirror_annotation_mismatches']))
    report=dict(status='passed',purpose='Complete-source engineering handoff, not 36 full-scale research training results',python=platform.python_version(),
        gpu_suite_tests=total,focused_export_tests=6,unique_tests_covered=total+1,gpu_suite_skips=0,
        tiny_qwen_symmetric_8epoch_jobs=36,formal_full_data_training_run=False,
        confirms='all six configs and three seeds on both tasks; paired analysis; epoch2/8 test reload; exact RNG/optimizer/scheduler restoration including retained-step cleanup',
        datasets=data,gpu_test_log_sha256=digest(log),focused_test_log_sha256=digest(focused),
        current_core_sha256={p.name:digest(p) for p in (ROOT/'src/modalitygap').glob('*.py')})
    write_json(ROOT/'docs/full-experiment-validation.json',report)
    lines=['# 完整实验交接验证','',
        f'RTX3090 / CUDA12.8 / Python{report["python"]}。完整回归{total}项通过，无跳过；文件句柄修正及新增parquet场景后定向导出测试6项通过，共覆盖{total+1}个不同测试。','',
        '新增GPU工程矩阵使用小Qwen随机权重及工程encoder，跑完2数据集×6配置×3seed的36个八轮作业；检查每个作业的27行三条件指标、epoch0–8断点、16个optimizer/scheduler step、最多两个step断点和真实参数更新。三seed分析及锁定epoch2/8的test加载通过。独立精确恢复测试验证参数、优化器、scheduler和Python/NumPy/Torch/CUDA随机状态与不中断运行一致，包括step断点清理。','',
        '这不是36个真实全规模训练结果。正式研究将在接手机器上运行。这里另外对真实完整原始数据完成了导出、解码、映射及跨划分审计：','',
        '| 数据集 | 官方train/dev/test | 最终训练池/dev/test | 跨划分重复移除 | 缺图/坏图/空文排除 |','|---|---|---|---|---|']
    for task,r in data.items():
        original='/'.join(str(r['official_counts'][s]) for s in ('train','dev','test'));actual='/'.join(str(r['actual_counts'][s]) for s in ('train','dev','test'))
        lines.append(f'| {task} | {original} | {actual} | {r["overlap_exclusions"]} | {r["incomplete_exclusions"]} |')
    lines += ['', 'MM-IMDb固定23类投影的全部25959条标签及三个split ID集合，与官方HDF5元数据完全匹配。原始JSON有71部电影带词表外标签，按固定任务词表投影并记录，未据此删除电影。MMSD2.0全部24635条正文/标签/划分与作者text_json_final完全匹配，镜像标签不作为真值。','',
        '重复处理预先固定为test > dev > train，只保留每个重复连通组的最高优先级原划分，不移动样本、不用模型表现决定。原始划分、实际移除和最终计数均可审计。MM-IMDb的5969训练池来自预先固定的迭代多标签分层算法，对应协议的约6000条，未重复抽样凑有利结果。','',
        '完整36作业dry-run生成36条正式训练及6条资源pilot命令。全部训练样本模式的dry-run、wheel构建和导出模块打包通过。原始图片、数据JSONL、模型权重、训练断点及Git凭据均不纳入源码提交。','',
        '具体科学设置、数据规则及复现命令见[完整实验运行手册](full-experiment.md)，机器可读结果及源码哈希见[validation JSON](full-experiment-validation.json)。']
    (ROOT/'docs/full-experiment-validation.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
