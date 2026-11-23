"""Per-seed derived curves and paired, group-aware endpoint bootstrap."""
import csv
import itertools
import math
import json
import random
from collections import defaultdict
from pathlib import Path

from .protocol import CONFIGS, CONDITIONS, LABELS, contributions, metrics, read_jsonl, write_json


def percentile(values, q):
    values = sorted(values)
    p = (len(values)-1)*q
    lo = int(p)
    return values[lo]+(values[min(lo+1,len(values)-1)]-values[lo])*(p-lo)


def paired_bootstrap(dataset, predictions, samples=2000, seed=2026):
    """One resampling plan reused across all configs, epochs and conditions."""
    import numpy as np
    keys = list(predictions)
    base = predictions[keys[0]]
    ids = [r['id'] for r in base]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate evaluation IDs')
    groups = defaultdict(list)
    for r in base:
        groups[r.get('group',r['id'])].append(r['id'])
    lookup = {}
    for key, rows in predictions.items():
        indexed = {r['id']:r for r in rows}
        if len(indexed)!=len(rows) or set(indexed)!=set(ids):
            raise ValueError('Paired bootstrap requires identical development IDs')
        if any(indexed[r['id']]['reference']!=r['reference'] or
               indexed[r['id']].get('group',r['id'])!=r.get('group',r['id']) for r in base):
            raise ValueError('Reference/group mismatch')
        lookup[key] = indexed
    rng, group_keys = random.Random(seed), sorted(groups)
    distributions = {key:[] for key in keys}
    gains = defaultdict(list)
    pairs = list(itertools.combinations(sorted({k[0] for k in keys}),2))
    differences = defaultdict(list)
    # Precompute group confusion counts; repeated bootstrap scores remain exact F1.
    matrices = {}
    for key in keys:
        counts = []
        for group in group_keys:
            rows = [lookup[key][sid] for sid in groups[group]]
            counts.append([[sum(label in r['reference'] and label in r['prediction'] for r in rows),
                            sum(label not in r['reference'] and label in r['prediction'] for r in rows),
                            sum(label in r['reference'] and label not in r['prediction'] for r in rows)]
                           for label in LABELS[dataset]])
        matrices[key] = np.asarray(counts, dtype=float).reshape(len(group_keys), -1)
    for _ in range(samples):
        choices = rng.choices(range(len(group_keys)), k=len(group_keys))
        weights = np.bincount(choices, minlength=len(group_keys))
        scores = {}
        for key in keys:
            counts = (weights @ matrices[key]).reshape(len(LABELS[dataset]), 3)
            if dataset == 'mmimdb':
                tp, fp, fn = counts.sum(axis=0)
                scores[key] = float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.
            else:
                denom = 2*counts[:,0]+counts[:,1]+counts[:,2]
                scores[key] = float(np.divide(2*counts[:,0], denom, out=np.zeros_like(denom), where=denom!=0).mean())
        for key, score in scores.items():
            distributions[key].append(score)
        for config,epoch in {(k[0],k[1]) for k in keys}:
            if all((config,epoch,c) in scores for c in CONDITIONS):
                gains[(config,epoch)].append(contributions({c:scores[config,epoch,c] for c in CONDITIONS})['gain_over_best_ablation'])
        for a,b in pairs:
            for epoch in (2,8):
                if (a,epoch,'full') in scores and (b,epoch,'full') in scores:
                    differences[(a,b,epoch)].append(scores[a,epoch,'full']-scores[b,epoch,'full'])
    interval = lambda values: dict(low=percentile(values,.025), high=percentile(values,.975))
    return dict(score_intervals={str(k):interval(v) for k,v in distributions.items()},
                gain_intervals={str(k):interval(v) for k,v in gains.items()},
                pair_difference_intervals={str(k):interval(v) for k,v in differences.items()},
                samples=samples, seed=seed, unit='shared image/text group (singleton if unique)',
                interpretation='Evaluation-sample uncertainty conditional on trained checkpoints; exploratory, no multiplicity correction')


def analyze(root, dataset, output, bootstrap_samples=2000):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)/dataset
    output.mkdir(parents=True,exist_ok=True)
    runs, raw = {}, {}
    evaluation_count = 0
    for path in sorted((Path(root)/dataset).glob('*/seed_*/metrics.csv')):
        with path.open() as f:
            rows = list(csv.DictReader(f))
        config,seed = path.parent.parent.name,int(path.parent.name.split('_')[1])
        if config not in CONFIGS:
            continue
        evaluation_count += len(rows)
        main = [r for r in rows if r['variant'] in CONDITIONS]
        indexed = {(int(r['epoch']),r['condition']):r for r in main}
        if len(indexed) != len(main):
            raise ValueError(f'Duplicate checkpoint-condition rows: {path}')
        if any(not math.isfinite(float(r['score'])) or not 0 <= float(r['score']) <= 1 for r in main):
            raise ValueError(f'Invalid task score: {path}')
        if set(indexed) != {(e,c) for e in range(9) for c in CONDITIONS}:
            raise ValueError(f'Incomplete main-condition trajectory: {path}')
        runs[config,seed] = indexed
        raw[config,seed] = path.parent
    if not runs:
        raise ValueError('No complete full-experiment runs found')
    seeds = sorted({s for _,s in runs})
    pair_reports, bootstrap = [], {}
    for seed in seeds:
        configs = sorted(c for c,s in runs if s==seed)
        for a,b in itertools.permutations(configs,2):
            d = {e:float(runs[a,seed][e,'full']['score'])-float(runs[b,seed][e,'full']['score']) for e in (1,2,6,7,8)}
            pair_reports.append(dict(seed=seed,a=a,b=b,early_difference=d[2],final_difference=d[8],
                candidate=d[2]<=-.01+1e-12 and d[8]>=.01-1e-12,
                late_differences={e:d[e] for e in (6,7,8)}, epoch1_difference=d[1]))
        predictions = {}
        for config in configs:
            for epoch in (2,8):
                for condition in CONDITIONS:
                    predictions[config,epoch,condition] = read_jsonl(raw[config,seed]/'predictions'/f'epoch_{epoch}_{condition}.jsonl')
        bootstrap[str(seed)] = paired_bootstrap(dataset,predictions,bootstrap_samples)
    confirmations = []
    for a,b in sorted({(r['a'],r['b']) for r in pair_reports}):
        selected = [r for r in pair_reports if r['a']==a and r['b']==b]
        confirmations.append(dict(a=a,b=b,seeds=[r['seed'] for r in selected],
            reversal_seeds=[r['seed'] for r in selected if r['candidate']],
            confirmed_two_of_three=len(selected)==3 and sum(r['candidate'] for r in selected)>=2))
    ranking = []
    for seed in seeds:
        configs = sorted(c for c,s in runs if s==seed)
        early = sorted(configs,key=lambda c:float(runs[c,seed][2,'full']['score']),reverse=True)
        final = sorted(configs,key=lambda c:float(runs[c,seed][8,'full']['score']),reverse=True)
        # Pairwise rank agreement excludes tied pairs; top-k retains all ties at the cutoff.
        concordant, comparable = 0,0
        for a,b in itertools.combinations(configs,2):
            d2 = float(runs[a,seed][2,'full']['score'])-float(runs[b,seed][2,'full']['score'])
            d8 = float(runs[a,seed][8,'full']['score'])-float(runs[b,seed][8,'full']['score'])
            if d2 and d8:
                comparable += 1
                concordant += d2*d8>0
        best = float(runs[final[0],seed][8,'full']['score'])
        winners = [c for c in final if float(runs[c,seed][8,'full']['score'])==best]
        retained = {k:any(float(runs[w,seed][2,'full']['score'])>=float(runs[early[k-1],seed][2,'full']['score']) for w in winners)
                    for k in range(1,len(configs)+1)}
        ranking.append(dict(seed=seed,early_ranking=early,final_ranking=final,final_winners=winners,
                            pairwise_rank_agreement=concordant/comparable if comparable else None,
                            early_top_k_retains_final_winner=retained))
    write_json(output/'analysis.json',dict(pairs=pair_reports,confirmation=confirmations,rankings=ranking,
        bootstrap=bootstrap,negative_result=not any(r['candidate'] for r in pair_reports),
        actual_main_evaluations=sum(len(v) for v in runs.values()),
        actual_total_evaluations=evaluation_count))
    colors = {'L':'#2878b5','M':'#d95132','LM':'#30945a'}
    styles = {'low':'-','high':'--'}
    names = ('full_performance','image_contribution','text_contribution','gain_over_best_ablation',
             'raw_condition_scores','invalid_output_rate','performance_vs_gpu_time')
    for name in names:
        fig,ax = plt.subplots(figsize=(9,5))
        for config in sorted({c for c,_ in runs}):
            scope,rate = config.split('-')
            conditions = CONDITIONS if name in ('raw_condition_scores','invalid_output_rate') else ('full',)
            for condition in conditions:
                ys,xs = [],[]
                for seed in seeds:
                    if (config,seed) not in runs:
                        continue
                    indexed = runs[config,seed]
                    scores = [{c:float(indexed[e,c]['score']) for c in CONDITIONS} for e in range(9)]
                    if name in ('image_contribution','text_contribution','gain_over_best_ablation'):
                        y = [contributions(s)[name] for s in scores]
                    else:
                        field = 'invalid_rate' if name=='invalid_output_rate' else 'score'
                        y = [float(indexed[e,condition][field]) for e in range(9)]
                    x = [float(indexed[e,condition]['training_seconds'])/3600 for e in range(9)] if name=='performance_vs_gpu_time' else list(range(9))
                    ys.append(np.array(y)*100)
                    xs.append(x)
                    ax.plot(x,ys[-1],color=colors[scope],linestyle=styles[rate],alpha=.22,linewidth=1)
                if ys:
                    marker = {'full':'o','mask_image':'s','mask_text':'^'}[condition]
                    ax.plot(np.mean(xs,axis=0),np.mean(ys,axis=0),color=colors[scope],linestyle=styles[rate],
                            marker=marker,markersize=3,label=config+(f' / {condition}' if len(conditions)>1 else ''))
        ax.axhline(0,color='gray',linewidth=.7)
        ax.set(xlabel='Cumulative training GPU hours (validation excluded)' if name=='performance_vs_gpu_time' else 'Epoch',
               ylabel='F1 percentage-point difference' if 'contribution' in name or name=='gain_over_best_ablation' else 'Invalid outputs (%)' if name=='invalid_output_rate' else 'F1 (%)',
               title=f'{dataset}: {name.replace("_"," ")} (faint: individual seeds; bold: mean)')
        ax.legend(fontsize=7,ncol=3)
        fig.tight_layout()
        fig.savefig(output/f'{name}.png',dpi=180)
        plt.close(fig)
