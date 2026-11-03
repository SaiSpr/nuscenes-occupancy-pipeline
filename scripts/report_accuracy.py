"""Summarize a completed benchmark and export a development-selected checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from occ_pipeline.utils.io import write_json


def plot_results(report,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    keys=['tiny_cnn/raw','unet/raw','tiny_cnn/development_selected','unet/development_selected']
    labels=['CNN\nraw','U-Net\nraw','CNN\ndev-selected','U-Net\ndev-selected']
    rows=[report['aggregate'][k] for k in keys]
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
    colors=['#5276a5','#dc8a3b','#5276a5','#dc8a3b']
    axes[0].bar(labels,[r['occupied_iou_mean'] for r in rows],
                yerr=[r['occupied_iou_std'] for r in rows],color=colors,capsize=4)
    axes[0].axhline(report['copy_input_baseline']['iou_occupied'],ls='--',color='#56616a',label='Copy input only')
    axes[0].set_title('Occupied IoU (mean ± SD across 3 seeds)');axes[0].legend(fontsize=8)
    axes[1].bar(labels,[r['hidden_occupied_iou_mean'] for r in rows],color=colors)
    axes[1].set_title('Occupied IoU on cells hidden from input')
    for ax in axes:
        ax.set_ylim(0,1);ax.set_ylabel('IoU');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('GT mask completion | 16 held-out frames from 2 scenes',fontsize=13)
    fig.savefig(output,dpi=180);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark',default='artifacts/accuracy-benchmark-v2')
    p.add_argument('--poisson',default='artifacts/poisson-benchmark')
    p.add_argument('--report',default='docs/ACCURACY_RESULTS.md')
    a=p.parse_args();root=Path(a.benchmark)
    r=json.loads((root/'results.json').read_text());pdecision=json.loads((Path(a.poisson)/'decision.json').read_text())
    # Model and seed selection uses only recorded development metrics, never test scores.
    selected=max(r['runs'],key=lambda row:row['decision']['development_metrics']['iou_occupied'])
    origin=root/f"{selected['model']}-seed-{selected['seed']}"/'checkpoint_decision.pt'
    destination=root/'recommended-checkpoint.pt';shutil.copyfile(origin,destination)
    write_json(root/'recommended-model.json',{'model':selected['model'],'seed':selected['seed'],
              'selected_on':'Development occupied IoU only','inference_decision':selected['decision'],
              'checkpoint':destination.name,'sha256':hashlib.sha256(destination.read_bytes()).hexdigest()})
    lines=['# Measured occupancy-completion results','',
           'This is a small real nuScenes mini **GT mask-completion** experiment. It does not establish camera/raw-LiDAR perception performance or production safety.','',
           '## Protocol','',
           '- 42 training frames from 7 scenes; 6 development frames from one different scene.',
           '- 16 held-out frames from the two official mini validation scenes.',
           '- Frames selected at evenly spaced timestamps by a fixed rule, before training.',
           '- Three training seeds (42, 43, 44), 30 epochs each; identical scene split, target artifacts, loss weights and validation mask seed (42).',
           '- Both architectures use the same CE objective, Adam learning rate, cosine schedule, augmentation, training data and epoch budget.',
           '- Checkpoints selected by development loss. Occupied decision bias selected on development occupied IoU using the same fixed grid for both models.',
           '- Test was scored after all six training runs and development selection completed.',
           '- Raw, evidence-preserving and development-selected inference are reported separately. Hidden-cell IoU excludes occupied cells already visible in the input.',
           '- Earlier smoke runs inspected a few official validation frames; this is not a pristine external test dataset.',
           '- No Poisson proposals were fed into training or prediction.','',
           '## Held-out results','',
           '| Model / inference | Occupied IoU mean | SD across seeds | Hidden occupied IoU mean |',
           '|---|---:|---:|---:|']
    for name,row in r['aggregate'].items():
        lines.append(f"| {name} | {row['occupied_iou_mean']:.4f} | {row['occupied_iou_std']:.4f} | {row['hidden_occupied_iou_mean']:.4f} |")
    lines+=['',f"Copying only visible input occupancy scores {r['copy_input_baseline']['iou_occupied']:.4f} occupied IoU; its hidden-cell IoU is zero.",'',
            'The spread is across three optimizer seeds, not a confidence interval across independent driving environments. Only two test scenes were evaluated.','',
            '## Paired model differences','',
            '| Seed | Raw U-Net minus raw CNN | Development-selected U-Net minus development-selected CNN |',
            '|---|---:|---:|']
    for seed in r['plan']['seeds']:
        cnn=next(x for x in r['runs'] if x['seed']==seed and x['model']=='tiny_cnn')
        unet=next(x for x in r['runs'] if x['seed']==seed and x['model']=='unet')
        lines.append(f"| {seed} | {unet['raw']['iou_occupied']-cnn['raw']['iou_occupied']:+.4f} | {unet['development_selected']['iou_occupied']-cnn['development_selected']['iou_occupied']:+.4f} |")
    agg=r['aggregate'];raw_gain=agg['unet/raw']['occupied_iou_mean']-agg['tiny_cnn/raw']['occupied_iou_mean']
    final_gain=agg['unet/development_selected']['occupied_iou_mean']-agg['tiny_cnn/development_selected']['occupied_iou_mean']
    lines+=['',f'Raw architecture comparison: U-Net mean occupied-IoU difference {raw_gain:+.4f}.',
            f'With the same development-selection procedure applied to both: difference {final_gain:+.4f}.',
            'These differences describe this fixed subset and training budget. They do not prove universal gains or isolate every architectural factor.','',
            '## Usable checkpoint','',
            f"The exported model is **{selected['model']}, seed {selected['seed']}**, selected using development occupied IoU only. The test score was not used to choose its architecture, seed, checkpoint epoch or decision setting.",'',
            '```sh',
            f"python scripts/evaluate.py --checkpoint {destination} --gt-dir {root/'test'} --output {root/'recommended-evaluation.json'}",
            f"python scripts/visualize.py --checkpoint {destination} --gt-dir {root/'test'} --n-samples 3 --png-dir {root/'figures'}",
            '```','',
            'Decision bias is an IoU-oriented classification setting, not calibrated uncertainty. Input preservation assumes input occupied cells are reliable; false positive inputs would also be retained.','',
            '## Poisson decision','',
            'Six support-distance/density-trim combinations were compared using development scans only. A gate was fixed before evaluation: at least three paired development scans, a positive pooled IoU difference and no negative per-scan difference.']
    best=pdecision['selected_by_development_only']
    if best:
        lines += [f"Best development candidate: support={best['support_m']} m, density trim={best['trim_quantile']}, paired scans={best['paired_samples']}, IoU difference={best['iou_delta']:+.6f}."]
        held=pdecision.get('heldout_paired_aggregate',{})
        if held:
            b=held['box_proposals'];q=held['poisson_proposals']
            lines += [f"On {q['samples_scored']} eligible held-out scans: annotation-proposal IoU={b['occupied_iou']:.6f}, Poisson-proposal IoU={q['occupied_iou']:.6f}; recall {b['recall']:.6f} -> {q['recall']:.6f}, precision {b['precision']:.6f} -> {q['precision']:.6f}."]
    lines += [f"**Development promotion heuristic: {pdecision['enable_by_default']}.** This was an engineering heuristic, not an assignment rule or significance test. The selected submission profile keeps both motion correction and Poisson enabled, with proposals isolated from measured supervision. See the combined validation results and design report.",
              'These are sparse geometric agreement scores, separate from the learned completion-model scores. Poisson cannot improve the model when its proposals are intentionally excluded from model inputs and supervision.','',
              '## Evidence and reproducibility','',
              f'- `{root}/plan.json`: predeclared selections, models, seeds and decision grid.',
              f'- `{root}/results.json`: pooled, per-scene and hidden-cell metrics for every run.',
              f'- `{root}/*-seed-*/training-summary.json`: training histories and actual scene/file lists.',
              f'- `{root}/*-seed-*/decision-selection.json`: complete development-only bias search.',
              f'- `{a.poisson}/decision.json`: Poisson gate and paired geometric results.',
              '- The initial driver run retained a five-sample loader limit. It was invalidated, excluded from these results, and corrected with exact sample-ID assertions before training.',
              '- CPU macOS execution was verified. Consult `artifacts/final-walkthrough/transcript.json` for the final Docker verification status; a Dockerfile alone is not proof of execution. Full-dataset benchmarks remain unverified.',
              '- See `IMPLEMENTATION_UPDATES.md` for the inherited pipeline and assistance disclosure.','']
    plot_results(r,root/'accuracy-comparison.png')
    lines += [f"![Held-out completion comparison](../{root}/accuracy-comparison.png)",'']
    path=Path(a.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text('\n'.join(lines))
    print(json.dumps({'report':str(path),'recommended_model':selected['model'],'seed':selected['seed'],'raw_gain':raw_gain,'paired_selected_gain':final_gain},indent=2))


if __name__=='__main__':main()
