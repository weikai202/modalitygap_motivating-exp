"""Reproduce/verify the selection frozen before real-data model evaluation."""
import json
from pathlib import Path

import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from modalitygap.protocol import GENRES, write_json


def main():
    metadata = json.loads(Path('data/downloads/mmimdb-official-metadata.json').read_text())
    selected = {}
    for split, size in (('train', 64), ('dev', 32)):
        a, b = metadata['split_bounds'][split]
        y = np.array(metadata['labels'][a:b])
        indices, _ = next(MultilabelStratifiedShuffleSplit(
            n_splits=1, train_size=size, test_size=len(y)-size, random_state=42).split(y,y))
        # Stratification can return an approximate size. No performance-based resampling.
        selected[split] = [dict(id=metadata['ids'][a+int(i)], labels=[
            g for g in GENRES if y[i,metadata['genre_order'].index(g)]])
            for i in sorted(indices)[:size]]
    path = Path('data/downloads/mmimdb-pilot-selection.json')
    result = dict(sampling='seed42 iterative multilabel stratification on official train/dev IDs, no model outputs used',
                  official_counts={s:b-a for s,(a,b) in metadata['split_bounds'].items()}, selected=selected)
    if path.exists():
        if json.loads(path.read_text()) != result:
            raise ValueError('Existing frozen selection differs; never overwrite it')
        print('PASS: selected IDs and labels exactly reproduce the frozen pre-evaluation selection')
    else:
        write_json(path,result)
        print('Selection frozen:',{s:len(v) for s,v in selected.items()})


if __name__ == '__main__':
    main()
