"""Plot the five original fixed-dev, eight-epoch symmetric-masking trajectories."""
import csv,json,math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from modalitygap.protocol import digest,write_json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/five-dataset-modality-gap-v1'
SOURCES=[('mmsd2','MMSD2.0','Macro-F1','real-symmetric-8epoch-v1'),
         ('mmimdb','MM-IMDb','Micro-F1','real-symmetric-8epoch-v1'),
         ('newsclippings','NewsCLIPpings','Macro-F1','real-expansion-symmetric-8epoch-v1/newsclippings'),
         ('crisismmd','CrisisMMD','Macro-F1','real-expansion-symmetric-8epoch-v1/crisismmd'),
         ('hatefulmemes','Hateful Memes','Macro-F1','real-expansion-symmetric-8epoch-v1/hatefulmemes')]


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'data.json').exists():raise ValueError('Use a fresh version; existing plot data will not be overwritten')
    data={};files={}
    for task,label,metric,folder in SOURCES:
        run=ROOT/'artifacts'/folder/'runs'/task/'L-low/seed_42'
        config=json.loads((run/'config.json').read_text());assert config['masking_method']=='symmetric-attention-v2'
        assert json.loads((run/'progress.json').read_text())['status']=='complete'
        with (run/'metrics.csv').open() as f:scores={(int(r['epoch']),r['condition']):float(r['score']) for r in csv.DictReader(f)}
        with (run/'contributions.csv').open() as f:stored={int(r['epoch']):r for r in csv.DictReader(f)}
        assert set(stored)==set(range(9));rows=[]
        for e in range(9):
            image=scores[e,'full']-scores[e,'mask_image'];text=scores[e,'full']-scores[e,'mask_text']
            assert math.isclose(image,float(stored[e]['image_contribution']),abs_tol=1e-12)
            assert math.isclose(text,float(stored[e]['text_contribution']),abs_tol=1e-12)
            rows.append(dict(epoch=e,image_pp=100*image,text_pp=100*text,gap_text_minus_image_pp=100*(text-image)))
        data[task]=dict(label=label,metric=metric,rows=rows,run=str(run))
        files[str(run/'metrics.csv')]=digest(run/'metrics.csv');files[str(run/'contributions.csv')]=digest(run/'contributions.csv')
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(15,8),sharex=True,sharey=True,constrained_layout=True)
    for ax,(task,entry) in zip(axes.flat,data.items()):
        rows=entry['rows']
        ax.plot(range(9),[r['image_pp'] for r in rows],'-o',color='#2369ad',label='Image contribution',lw=2,ms=5)
        ax.plot(range(9),[r['text_pp'] for r in rows],'-s',color='#e47b21',label='Text contribution',lw=2,ms=5)
        ax.axhline(0,color='#777777',lw=.9);ax.grid(alpha=.2);ax.set_title(entry['label']+'  |  '+entry['metric'])
        ax.set_xticks(range(9));ax.set_xlabel('Epoch');ax.set_ylabel('Contribution (F1 percentage points)')
    axes.flat[5].axis('off')
    axes.flat[5].legend(handles=[Line2D([0],[0],color='#2369ad',marker='o',lw=2,label='Image: F1(full) - F1(mask image)'),Line2D([0],[0],color='#e47b21',marker='s',lw=2,label='Text: F1(full) - F1(mask text)')],loc='upper left',frameon=False,fontsize=11)
    axes.flat[5].text(.04,.65,'Qwen2.5-VL-3B | L-low | seed 42\nSymmetric mean + attention masking\nEpoch 0: before training\nFixed original dev set: 32 per dataset\nTrain: 64 (MM-IMDb: 63)\nMM-IMDb uses micro-F1; others macro-F1\nExpanded-dev diagnostics are excluded.',transform=axes.flat[5].transAxes,va='top',linespacing=1.7)
    fig.suptitle('Five datasets: image / text ablation contributions across training',fontsize=17)
    for suffix in ('png','svg','pdf'):fig.savefig(OUT/f'five-dataset-contributions.{suffix}',dpi=200)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,6),constrained_layout=True)
    colors=['#0072b2','#009e73','#d55e00','#cc79a7','#80582c'];markers=['o','s','^','D','v']
    for (task,entry),color,marker in zip(data.items(),colors,markers):
        ax.plot(range(9),[r['gap_text_minus_image_pp'] for r in entry['rows']],label=entry['label'],color=color,marker=marker,lw=2,ms=6)
    ax.axhline(0,color='#555555',lw=1);ax.grid(alpha=.2);ax.set_xticks(range(9));ax.set_xlabel('Epoch')
    ax.set_ylabel('Gap = text contribution - image contribution (F1 pp)')
    ax.set_title('Five datasets: performance modality gap',fontsize=16);ax.legend(ncol=2,frameon=False)
    fig.supxlabel('Positive: larger text ablation contribution. Negative: larger image ablation contribution.',fontsize=10)
    for suffix in ('png','svg','pdf'):fig.savefig(OUT/f'five-dataset-performance-gap.{suffix}',dpi=200)
    plt.close(fig)
    write_json(OUT/'data.json',dict(datasets=data,source_sha256=files,masking_method='symmetric-attention-v2',gap_definition='C_text - C_image = F1(mask_image) - F1(mask_text)',units='F1 percentage points',original_dev32_only=True))
    with (OUT/'trajectories.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['dataset','metric','epoch','image_pp','text_pp','gap_text_minus_image_pp']);writer.writeheader()
        for task,entry in data.items():
            for row in entry['rows']:writer.writerow(dict(dataset=task,metric=entry['metric'],**row))
    print(OUT)


if __name__=='__main__':main()
