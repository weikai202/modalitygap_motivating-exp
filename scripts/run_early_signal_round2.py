"""Score early checkpoints and compare the locked two-slot policies."""
import csv
import json
import math
from pathlib import Path

from modalitygap.protocol import digest, write_json
from score_early_signals import EPOCHS, OUTPUT, evaluate, run_path

ROOT = Path('artifacts/early-signals-round2')


def read_json(path):
    return json.loads(path.read_text())


def path_for(dataset, setting, seed):
    if setting == 'L-low' and seed == 42:
        return run_path(dataset)
    return ROOT / 'runs' / dataset / setting / f'seed_{seed}'


def full_f1(run, epoch):
    with (run / 'metrics.csv').open() as stream:
        scores = [float(row['score']) for row in csv.DictReader(stream)
                  if row['condition'] == row['variant'] == 'full'
                  and int(row['epoch']) == epoch]
    if len(scores) != 1:
        raise ValueError(f'Expected one Full score for {run}, epoch {epoch}')
    return scores[0]


def score_files_ready(dataset, run, output):
    for epoch in EPOCHS:
        file = output / f'{dataset}-epoch-{epoch}.json'
        if not file.exists():
            return False
        payload = read_json(file)
        weights = run / 'checkpoints' / f'epoch_{epoch}' / 'trainable.pt'
        if payload['device'] != 'cuda' or payload['checkpoint_sha256'] != digest(weights):
            return False
    return True


def early_signal(dataset, setting, seed, decision_epoch):
    run = path_for(dataset, setting, seed)
    output = OUTPUT if setting == 'L-low' and seed == 42 else ROOT / 'scores' / dataset / setting
    if not score_files_ready(dataset, run, output):
        evaluate(dataset, 'cuda', run=run, output=output)
    curve = {}
    reference_counts = None
    for epoch in EPOCHS:
        rows = read_json(output / f'{dataset}-epoch-{epoch}.json')['scores']
        with (run / 'predictions' / f'epoch_{epoch}_full.jsonl').open() as stream:
            ids = {json.loads(line)['id'] for line in stream}
        curve[epoch] = {}
        for condition in ('full', 'mask_image', 'mask_text'):
            subset = [row for row in rows if row['condition'] == condition]
            counts = {row['id']: row['label_tokens'] for row in subset}
            if set(counts) != ids or len(subset) != len(ids):
                raise ValueError(f'Scored sample IDs differ for {dataset}/{setting}/{epoch}/{condition}')
            if reference_counts is None:
                reference_counts = counts
            if counts != reference_counts:
                raise ValueError('Label token counts changed across conditions or checkpoints')
            loss = sum(row['label_nll_sum'] for row in subset) / sum(counts.values())
            if not math.isfinite(loss):
                raise ValueError('Non-finite label loss')
            curve[epoch][condition] = loss
    first, last = curve[0], curve[decision_epoch]
    resources = {row['epoch']: row['mean_loss'] for row in read_json(run / 'resources.json')}
    decline = (first['full'] - last['full']) / decision_epoch
    return dict(early_f1=full_f1(run, decision_epoch), label_log_ppl=curve,
                full_label_decline=decline, training_loss_decline=resources[1] - resources[2],
                text_gap_growth=decline + (last['mask_text'] - first['mask_text']) / decision_epoch)


def picks_for(dataset, settings, rows):
    order = {setting: index for index, setting in enumerate(settings)}
    by_f1 = sorted(settings, key=lambda setting: (-rows[setting]['early_f1'], order[setting]))
    picks = {'early_f1': by_f1[:2]}
    policies = ['full_label_decline', 'training_loss_decline']
    if dataset == 'mmsd2':
        policies.append('text_gap_growth')
    for policy in policies:
        ranked = sorted(by_f1[1:], key=lambda setting: (-rows[setting][policy], by_f1.index(setting)))
        picks[policy] = [by_f1[0], ranked[0]]
    return picks


def main():
    protocol = read_json(ROOT / 'protocol.json')
    settings, seed = protocol['settings'], protocol['seed']
    decision_epoch, endpoint_epoch = protocol['decision_epoch'], protocol['endpoint_epoch']
    assert decision_epoch == 2 and protocol['budget_per_policy'] == 2
    early, picks = {}, {}
    for dataset in protocol['datasets']:
        early[dataset] = {setting: early_signal(dataset, setting, seed, decision_epoch)
                          for setting in settings}
        picks[dataset] = picks_for(dataset, settings, early[dataset])
        write_json(ROOT / 'early-signals.json', early)
        write_json(ROOT / 'picks.json', picks)

    results = {}
    for dataset in protocol['datasets']:
        scores = {setting: full_f1(path_for(dataset, setting, seed), endpoint_epoch)
                  for setting in settings}
        continuation_seconds = {
            setting: sum(row['seconds'] for row in read_json(
                path_for(dataset, setting, seed) / 'resources.json')
                if decision_epoch < row['epoch'] <= endpoint_epoch)
            for setting in settings
        }
        final_rank = sorted(settings, key=lambda setting: -scores[setting])
        top_two = final_rank[:2]
        tied = scores[final_rank[1]] == scores[final_rank[2]]
        top_two_with_ties = [setting for setting in final_rank
                             if scores[setting] >= scores[top_two[-1]]]
        targets = set(top_two_with_ties) - set(picks[dataset]['early_f1'])
        results[dataset] = dict(epoch_8_f1=scores, top_two=top_two, top_two_boundary_tie=tied,
                                top_two_with_ties=top_two_with_ties,
                                missed_by_baseline=sorted(targets), policies={})
        baseline_mean = sum(scores[setting] for setting in picks[dataset]['early_f1']) / 2
        for policy, chosen in picks[dataset].items():
            mean = sum(scores[setting] for setting in chosen) / 2
            results[dataset]['policies'][policy] = dict(
                picks=chosen, mean_f1=mean, gain_vs_f1=mean - baseline_mean,
                continuation_training_seconds=sum(continuation_seconds[setting] for setting in chosen),
                recovered=sorted(targets.intersection(chosen)),
                recall=len(targets.intersection(chosen)) / len(targets) if targets and not tied else None)
    write_json(ROOT / 'results.json', results)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
