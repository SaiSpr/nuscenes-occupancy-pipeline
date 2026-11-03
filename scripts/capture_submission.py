"""Capture actual Docker execution for an edited, narrated command walkthrough."""
import json
import platform
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/final-walkthrough'
IMAGE='occupancy-pipeline:submission'
PLATFORM='linux/arm64' if platform.machine() in ('arm64','aarch64') else 'linux/amd64'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data=ROOT.parent/'data/nuscenes'
    evidence=ROOT/'artifacts/accuracy-benchmark-v2'
    comparison=json.loads((ROOT/'artifacts/combined-validation/comparison.json').read_text())
    tokens=','.join(r['sample_token'] for r in comparison['samples'] if r['status']=='completed')
    run=['docker','run','--rm','--platform',PLATFORM]
    stages=[
      ('Build reproducible environment',['docker','build','--platform',PLATFORM,'--target','reconstruction','--progress','plain','-t',IMAGE,'.']),
      ('Run regression tests',run+[IMAGE,'python','-m','pytest']),
      ('Generate combined GT and compute model loss',run+['-v',f'{data}:/data/nuscenes:ro','-v',f'{OUT}:/outputs',IMAGE,'python','scripts/demo_submission.py','--data-root','/data/nuscenes','--output','/outputs','--sample-tokens',tokens]),
      ('Evaluate the trained checkpoint',run+['-v',f'{evidence}:/evidence:ro','-v',f'{OUT}:/outputs',IMAGE,'python','scripts/evaluate.py','--checkpoint','/evidence/recommended-checkpoint.pt','--gt-dir','/evidence/test','--output','/outputs/checkpoint-evaluation.json']),
      ('Visualize actual model predictions',run+['-v',f'{evidence}:/evidence:ro','-v',f'{OUT}:/outputs',IMAGE,'python','scripts/visualize.py','--checkpoint','/evidence/recommended-checkpoint.pt','--gt-dir','/evidence/test','--n-samples','2','--png-dir','/outputs/predictions']),
      ('Record installed dependencies',run+[IMAGE,'python','-m','pip','freeze']),
      ('Record Docker image identity',['docker','image','inspect',IMAGE,'--format','{{.Id}}'])]
    recordings=[]
    for title,command in stages:
        print(title,flush=True);start=time.monotonic()
        process=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        events=[]
        for line in process.stdout:
            events.append({'seconds':round(time.monotonic()-start,3),'text':line.rstrip()})
            print(line.rstrip(),flush=True)
        status=process.wait()
        recordings.append({'title':title,'command':command,'elapsed_seconds':round(time.monotonic()-start,3),'exit_code':status,'events':events})
        (OUT/'transcript.json').write_text(json.dumps(recordings,indent=2))
        (OUT/'transcript.txt').write_text('\n\n'.join(r['title']+'\n$ '+' '.join(r['command'])+'\n'+'\n'.join(e['text'] for e in r['events']) for r in recordings))
        print(json.dumps({'stage':title,'status':status,'seconds':round(time.monotonic()-start,2)}),flush=True)
        if status:
            print('\n'.join(e['text'] for e in events[-35:]),flush=True)
            raise SystemExit(status)


if __name__=='__main__':main()
