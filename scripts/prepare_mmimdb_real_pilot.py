"""Extract a model-blind fixed pilot from checksum-verified original MM-IMDb."""
import json
import tarfile
from collections import Counter
from pathlib import Path

from modalitygap.data import prepare
from modalitygap.protocol import GENRES, digest, write_json, write_jsonl


def main():
    root = Path('data/real-pilot/mmimdb')
    if root.exists():
        raise ValueError('Use a fresh immutable pilot directory')
    selection = json.loads(Path('data/downloads/mmimdb-pilot-selection.json').read_text())
    metadata = json.loads(Path('data/downloads/mmimdb-official-metadata.json').read_text())
    expected = {r['id']: (s, r) for s, rows in selection['selected'].items() for r in rows}
    if len(expected) != sum(map(len, selection['selected'].values())):
        raise ValueError('Selected ID overlap')
    ids = metadata['ids']
    if len(ids) != len(set(ids)):
        raise ValueError('Official duplicate IDs')
    official = {s: set(ids[a:b]) for s, (a, b) in metadata['split_bounds'].items()}
    root.mkdir(parents=True)
    (root/'images').mkdir(); (root/'raw-json').mkdir()
    found = {}; inventory = Counter(); archive_split = None
    with tarfile.open('data/downloads/mmimdb.tar.gz', 'r|gz') as archive:
        for member in archive:
            if not member.isfile():
                continue
            path = Path(member.name)
            inventory[path.suffix] += 1
            if path.name == 'split.json':
                archive_split = json.load(archive.extractfile(member))
            sid = path.stem
            if sid not in expected or path.suffix not in ('.json', '.jpeg'):
                continue
            key = (sid, path.suffix)
            if key in found:
                raise ValueError(f'Duplicate raw member {key}')
            raw = archive.extractfile(member).read()
            destination = root/('raw-json' if path.suffix == '.json' else 'images')/path.name
            destination.write_bytes(raw)
            found[key] = destination
    if archive_split is None:
        raise ValueError('Missing official raw split')
    # Raw release calls the development partition "dev" (or "val" in some exports).
    raw_sets = {('dev' if s in ('val', 'valid', 'validation') else s): set(v) for s,v in archive_split.items()}
    if raw_sets != official:
        raise ValueError('Raw split.json differs from official HDF5 split')
    rows = []; hashes = {}
    for sid, (split, selected) in expected.items():
        if sid not in official[split]:
            raise ValueError('Pilot ID outside official split')
        data = json.loads(found[(sid, '.json')].read_text())
        genres = [g for g in GENRES if g in data['genres']]
        if set(genres) != set(selected['labels']):
            raise ValueError(f'Raw/HDF5 genre mismatch: {sid}')
        plots = data.get('plot', [])
        if isinstance(plots, str):
            plots = [plots]
        text = next((p for p in plots if isinstance(p, str) and p.strip()), '')
        if not text:
            raise ValueError(f'No released plot: {sid}')
        image = found[(sid, '.jpeg')]
        rows.append(dict(id=sid, split=split, image=str(image.resolve()), text=text, labels=genres))
        hashes[sid] = dict(image=digest(image), raw_json=digest(found[(sid, '.json')]))
    write_jsonl(root/'source.jsonl', rows)
    prepare(root/'source.jsonl', root/'manifest', 'mmimdb',
            'MM-IMDb author raw archive MD5 38e154264b3771f70840c6cb5f170a93; official raw split.json crosschecked with HDF5; seed42 fixed pilot; first nonempty released plot')
    write_json(root/'provenance.json', dict(
        dataset='MM-IMDb', official_repo='https://github.com/johnarevalo/gmu-mmimdb',
        official_revision='95226b0d265743e7a9330549bc35b95fcb60e1dd',
        archive='https://archive.org/download/mmimdb/mmimdb.tar.gz',
        archive_verified_md5='38e154264b3771f70840c6cb5f170a93',
        official_counts=selection['official_counts'], selected=selection['selected'],
        sampling=selection['sampling'],
        size_rule='Request train64/dev32 using iterative multilabel split seed42; retain undersized train63; trim oversized dev33 to first32 official-order indices; fixed before model evaluation',
        plot_rule='First nonempty released plot in list order; no title or additional metadata in prompt',
        raw_hdf5_split_match=True, selected_raw_hdf5_label_match=True,
        raw_inventory=dict(inventory), selected_sha256=hashes, test_used=False))
    print(json.dumps(dict(counts=Counter(r['split'] for r in rows), inventory=inventory,
                          raw_hdf5_split_match=True, selected_raw_hdf5_label_match=True), indent=2))


if __name__ == '__main__':
    main()
