"""Final integrity checks for completed campaigns; preserve previous experiment hashes."""
import json
import math
from pathlib import Path
import shutil

from modalitygap.protocol import digest,write_json

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'artifacts/real-expansion-symmetric-8epoch-v1'
TASKS=('newsclippings','crisismmd','hatefulmemes')


def main():
    current={p.name:digest(p) for p in (ROOT/'src/modalitygap').glob('*.py')}
    data=json.loads((OUTPUT/'data-audit.json').read_text())
    inputs=json.loads((OUTPUT/'real-input-audit.json').read_text())
    summary=json.loads((OUTPUT/'epoch-report.json').read_text())
    checkpoints=0;predictions=0
    for task in TASKS:
        workspace=OUTPUT/task;run=workspace/'runs'/task/'L-low/seed_42'
        pipeline=json.loads((workspace/'pipeline.json').read_text());config=json.loads((run/'config.json').read_text())
        assert pipeline['source_sha256']==current==config['source_sha256']
        assert config['epochs']==8 and config['seed']==42 and config['dataset']==task
        assert config['masking_method']=='symmetric-attention-v2' and '3090' in str(config['device'])
        assert json.loads((workspace/'pipeline-status.json').read_text())['status']=='complete'
        assert json.loads((run/'progress.json').read_text())['status']=='complete'
        assert data[task]['status']==inputs[task]['status']=='passed'
        audit=json.loads((workspace/'prediction-audit.json').read_text())
        assert audit['status']=='passed' and audit['examples']==864 and audit['prediction_files']==27
        predictions+=audit['examples']
        assert len(summary[task]['epochs'])==9 and summary[task]['predictions']==864
        assert len(summary[task]['resources'])==8
        assert all(math.isfinite(r['mean_loss']) and math.isfinite(r['last_grad_norm']) and r['last_grad_norm']>0 for r in summary[task]['resources'])
        for epoch in range(9):
            folder=run/'checkpoints'/f'epoch_{epoch}'
            marker=json.loads((folder/'complete.json').read_text())
            assert marker['epoch']==epoch and marker['step']==4*epoch
            for kind in ('trainable','state'):assert digest(folder/f'{kind}.pt')==marker[f'{kind}_sha256']
            checkpoints+=1
    old=json.loads((OUTPUT/'preserved-old-results.json').read_text())
    for path,value in old.items():assert digest(ROOT/path)==value,('Old result modified',path)
    gpu_log=ROOT/'artifacts/expansion-v1-gpu-tests-threads1.log';gpu_text=gpu_log.read_text()
    assert 'Ran 40 tests' in gpu_text and gpu_text.rstrip().endswith('OK') and 'skipped=' not in gpu_text
    boundary_log=ROOT/'artifacts/expansion-v1-download-boundary-test.log'
    assert 'Ran 1 test' in boundary_log.read_text() and boundary_log.read_text().rstrip().endswith('OK')
    snapshot=OUTPUT/'source-snapshot-final'
    files=list((ROOT/'src/modalitygap').glob('*.py'))+list((ROOT/'tests').glob('test_*.py'))
    files+=list((ROOT/'configs').glob('expansion-*-8epoch-v1.json'))
    files+=[ROOT/'scripts'/name for name in ('prepare_expansion.py','fetch_visualnews_images.py','fetch_visualnews_images_v2.py','run_experiment.sh','run_experiment.py','run_expansion.sh','audit_predictions.py','audit_expansion_data.py','audit_expansion_inputs.py','report_expansion.py','validate_expansion.py')]
    files += [ROOT/'README.md',ROOT/'pyproject.toml',ROOT/'requirements-cu128.lock']
    file_hashes={}
    for file in files:
        relative=file.relative_to(ROOT);dest=snapshot/relative
        if dest.exists() and digest(dest)!=digest(file):raise ValueError('Final snapshot already differs')
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,dest);file_hashes[str(relative)]=digest(file)
    write_json(snapshot/'sha256.json',file_hashes)
    validation=dict(status='passed',datasets=list(TASKS),train_per_dataset=64,dev_per_dataset=32,
        epochs=8,checkpoints_sha256_verified=checkpoints,formal_predictions_independently_redecoded=predictions,
        real_examples_audited=288,reference_independent_inference_examples=288,
        old_result_files_verified_unchanged=len(old),current_core_sha256=current,
        tests=dict(cuda_enabled_regression=dict(count=40,skipped=0,log_sha256=digest(gpu_log)),
                   tar_range_boundary=dict(count=1,log_sha256=digest(boundary_log))),
        report_sha256=digest(OUTPUT/'epoch-report.json'),figure_sha256=digest(OUTPUT/'contribution-trajectories.png'),comparison_figure_sha256=digest(OUTPUT/'contribution-comparison.png'),
        data_audit_sha256=digest(OUTPUT/'data-audit.json'),real_input_audit_sha256=digest(OUTPUT/'real-input-audit.json'))
    write_json(OUTPUT/'validation.json',validation)
    write_json(ROOT/'docs/expansion-symmetric-8epoch-v1-validation.json',validation)
    print(json.dumps(validation,indent=2))


if __name__=='__main__':main()
