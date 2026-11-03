"""Package code and the completed accuracy experiment, excluding raw/cloud data."""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submission',action='store_true',help='Include the combined-method design report and evidence')
    args=parser.parse_args()
    destination=root.parent/('dist/occupancy-perception-pipeline-submission.zip' if args.submission else 'dist/occupancy-perception-pipeline-accuracy-update.zip')
    files=set()
    for folder in ('src','tests','configs','scripts'):
        files.update(p for p in (root/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    for name in ('README.md','pyproject.toml','Dockerfile','.dockerignore','.gitignore','SNAPSHOT_PROVENANCE.json',
                 'docs/IMPLEMENTATION_UPDATES.md','docs/ACCURACY_RESULTS.md','docs/SUBMISSION.md','docs/ASSISTANCE.md'):
        files.add(root/name)
    benchmark=root/'artifacts/accuracy-benchmark-v2'
    required=('plan.json','results.json','tests.txt','execution.log','accuracy-comparison.png',
              'recommended-checkpoint.pt','recommended-model.json','recommended-evaluation.json',
              'recommended-evaluation.log','visualization.log')
    files.update(benchmark/name for name in required)
    files.update((benchmark/'figures').glob('*.png'))
    for split in ('train','dev','test'):
        manifest=benchmark/split/'manifest.json';files.add(manifest)
        files.update(manifest.parent/name for name in json.loads(manifest.read_text())['files'])
    plan=json.loads((benchmark/'plan.json').read_text())
    for model in plan['models']:
        for seed in plan['seeds']:
            folder=benchmark/f'{model}-seed-{seed}'
            files.update(folder/name for name in ('training-summary.json','decision-selection.json','checkpoint_best.pt'))
    poisson=root/'artifacts/poisson-benchmark'
    files.update(p for p in poisson.glob('*.json'))
    files.add(poisson/'execution.log')
    if args.submission:
        files.add(root/'output/pdf/Occupancy_Pipeline_Design_Report.pdf')
        combined=root/'artifacts/combined-validation'
        files.update(combined/name for name in ('comparison.json','execution.log','tests.txt','generation-check.json'))
        manifest=combined/'targets/manifest.json'
        files.add(manifest)
        files.update(manifest.parent/name for name in json.loads(manifest.read_text())['files'])
        walkthrough=root/'artifacts/final-walkthrough'
        video=json.loads((walkthrough/'video-manifest.json').read_text())
        transcript=json.loads((walkthrough/'transcript.json').read_text())
        assert video['full_decode_passed'] and len(transcript)==7 and all(r['exit_code']==0 for r in transcript)
        files.update(walkthrough/name for name in ('walkthrough.mp4','captions.srt','narration.txt','video-manifest.json',
                     'transcript.json','transcript.txt','demo-summary.json','checkpoint-evaluation.json','combined-gt.png'))
        files.update((walkthrough/'predictions').glob('*.png'))
    checksums={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    destination.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):z.write(p,'occupancy-perception-pipeline/'+str(p.relative_to(root)))
        z.writestr('occupancy-perception-pipeline/SHA256SUMS.json',json.dumps(checksums,indent=2))
    with zipfile.ZipFile(destination) as z:
        assert z.testzip() is None
        assert all(hashlib.sha256(z.read('occupancy-perception-pipeline/'+name)).hexdigest()==digest for name,digest in checksums.items())
        assert not any('/data/' in n.replace('/src/occ_pipeline/data/','/source/') or '/.claude/' in n for n in z.namelist())
    print(json.dumps({'archive':str(destination),'files':len(files)+1,'bytes':destination.stat().st_size,'checksums_verified':True},indent=2))


if __name__=='__main__':main()
