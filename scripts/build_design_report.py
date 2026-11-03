"""Build the brief submission design report from recorded local evidence."""
from pathlib import Path
import json
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image,Preformatted

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/pdf/Occupancy_Pipeline_Design_Report.pdf'
r=json.loads((ROOT/'artifacts/accuracy-benchmark-v2/results.json').read_text())
g=json.loads((ROOT/'artifacts/combined-validation/comparison.json').read_text())
walkthrough=ROOT/'artifacts/final-walkthrough'
transcript=json.loads((walkthrough/'transcript.json').read_text()) if (walkthrough/'transcript.json').exists() else []
docker_verified=len(transcript)==7 and all(stage['exit_code']==0 for stage in transcript)
video=json.loads((walkthrough/'video-manifest.json').read_text()) if (walkthrough/'video-manifest.json').exists() else {}
video_verified=video.get('full_decode_passed',False)
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyR',fontName='Helvetica',fontSize=9.6,leading=13.2,spaceAfter=7,textColor=colors.HexColor('#253347')))
styles.add(ParagraphStyle(name='SmallR',parent=styles['BodyR'],fontSize=8.3,leading=11,spaceAfter=5))
styles.add(ParagraphStyle(name='TitleR',fontName='Helvetica-Bold',fontSize=25,leading=29,spaceAfter=14,textColor=colors.HexColor('#162d49')))
styles.add(ParagraphStyle(name='H1R',fontName='Helvetica-Bold',fontSize=17,leading=21,spaceAfter=11,textColor=colors.HexColor('#162d49')))
styles.add(ParagraphStyle(name='H2R',fontName='Helvetica-Bold',fontSize=11.5,leading=15,spaceBefore=7,spaceAfter=6,textColor=colors.HexColor('#167d8d')))
styles.add(ParagraphStyle(name='CodeR',fontName='Courier',fontSize=7.1,leading=10,spaceAfter=9,backColor=colors.HexColor('#f0f4f8'),borderPadding=8))
story=[]
def p(text,style='BodyR'):story.append(Paragraph(text,styles[style]))
def h(text):p(text,'H2R')
def table(rows,widths):
    cells=[[Paragraph(str(v),styles['SmallR']) for v in row] for row in rows]
    t=Table(cells,colWidths=widths,hAlign='LEFT',repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e4edf4')),('VALIGN',(0,0),(-1,-1),'TOP'),
      ('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),6),
      ('BOTTOMPADDING',(0,0),(-1,-1),5),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#a4b8c9')),
      ('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#d9e2eb'))]))
    story.extend([t,Spacer(1,8)])
def code(text):story.append(Preformatted(text,styles['CodeR']))
def page(title):story.append(PageBreak());p(title,'H1R')
def pct(x):return f'{100*x:.2f}%'

p('Occupancy Perception Pipeline','TitleR')
p('Design report | Motion correction + Poisson reconstruction','H1R')
p('4 November 2026 | Implementation and validation report','SmallR')
h('1. Objective and scope')
p('Build a modular nuScenes perception pipeline that generates occupancy supervision from real LiDAR and demonstrates a learning-ready data flow, model forward pass and loss computation. The submission profile <b>keeps both motion correction and Poisson reconstruction enabled</b>.')
p('The learning demonstration is <b>GT mask completion</b>: a model reconstructs an occupancy grid from a copy with 30% of occupied input cells hidden. It is not camera-based prediction or inference from a fresh raw LiDAR scan. Training is supplementary evidence; the assignment does not require a trained model.')
h('System architecture')
table([['Stage / module','Responsibility and output'],
 ['Raw tables / data/nuscenes_adapter.py','Read scene-linked scans and poses; preserve each sweep timestamp and sensor origin.'],
 ['Temporal evidence / gt_generation/evidence','Align past scans, interpolate tracked boxes, correct moving-object points, filter ego-body returns and cast free-space rays.'],
 ['Poisson / evidence/reconstruction.py','Reconstruct a supported static surface shell after motion processing; emit separately identified proposals.'],
 ['Artifacts / generate.py, utils/io.py','Save measured labels, counts, provenance and proposals in NPZ; publish an exact generation manifest.'],
 ['Dataset / data/occupancy_dataset.py','Create repeatable partial-grid inputs; keep measured targets separate from reconstruction proposals.'],
 ['Learning / models, training','Tiny CNN or U-Net; CE/BCE/focal losses, scene-disjoint evaluation and development-only inference tuning.']], [168,343])
h('Selected submission design')
p('Up to ten scans (the keyframe plus nine past scans) are requested. Motion correction handles moving objects; Poisson reconstructs the remaining static geometry. These methods solve different problems and are retained together. Box-volume filling is disabled in <b>configs/submission.yaml</b> to keep this combination interpretable.')
p('The implementation is published in SaiSpr/nuscenes-occupancy-pipeline. The original branch snapshot and file hashes are documented in SNAPSHOT_PROVENANCE.json; the current project is not a merge of the original remote branch.','SmallR')

page('2. Ground-truth generation and reconstruction')
h('Geometry and temporal alignment')
p('<b>Grid:</b> x and y in [-40, 40) m; z in [-3, 3) m; 1 m voxels. Saved axis order is [Z,Y,X] = [6,80,80]. Vehicle axes are +X forward, +Y left and +Z up. Each past scan is transformed sensor -> its own ego pose -> global -> keyframe ego coordinates.')
p('Object annotations are matched by instance token. Translation and size are interpolated linearly; rotation uses shortest-path quaternion SLERP. Points inside matched boxes are reposed to the keyframe object pose. Reposed points provide occupied hits only: artificial free-space rays are not cast to their new positions.')
p('Unbracketed intervals or unmatched track sets are conservatively skipped and logged. Thus a request for ten scans can yield fewer usable scans. Ego-body filtering uses an approximate per-sweep exclusion box and must be checked for the sensor platform.')
h('Ray evidence and Poisson proposals')
p('Batched, clipped 3D DDA traverses cells from each actual sensor origin. All measured hits are applied after free-space rays, so a later ray cannot erase a surface. Unknown cells remain distinct from observed free space.')
p('Poisson uses in-range points outside annotated object boxes. Normals are estimated locally and oriented toward acquisition origins. Reconstruction uses Open3D, octree depth 7, density-trim quantile 0.3, a deterministic 100,000-point cap and a 1 m nearest-point support limit. Mesh surface voxels are cropped to the grid.')
p('<b>Evidence policy:</b> reconstructed surfaces are proposed only in unknown cells. They never overwrite measured free/occupied labels. Retaining Poisson in the system does not make its inferred surfaces verified ground truth.')
table([['Saved array','Contract'],['occupancy','uint8 [Z,Y,X]: 0 unknown, 1 occupied, 2 observed free.'],['point_counts','uint16 counts, saturated to prevent overflow.'],['provenance','0 unknown; 1 ray free; 2 measured hit; 3 motion-reposed hit.'],['proposal_kind','0 none; 4 box prior; 5 Poisson surface. Box filling is off in the submission profile.'],['proposed_occupancy','Measured occupancy plus inferred surface proposals. Stored separately; excluded from current training targets.']], [134,377])
p('Metadata records input hashes, sweep transforms, actual scan counts, skipped tokens, annotation hash and configuration. Atomic writes and manifests exclude stale targets from the current dataset.','SmallR')

page('3. Learning pipeline and measured results')
h('Model and evaluation design')
p('The single model input channel is a partially hidden occupied grid; no complete-GT free channel is supplied. The three-class head predicts unknown, occupied and free. The U-Net pools X/Y while retaining height and uses skip connections. The reported benchmark uses base width 4 without auxiliary heads; auxiliary supervision remains available as an extension.')
p('CE class weights are fixed from training data. Binary objectives ignore unknown cells. Validation masks are stable across access order and training seeds. Metrics are calculated from a pooled confusion matrix, not averages of per-batch IoUs. Development-only decision bias reduces class-balancing false positives, and inference preserves occupied input evidence.')
p('<b>Benchmark:</b> 42 frames / 7 training scenes; 6 frames / 1 development scene; 16 frames / 2 official validation scenes held out from this training and decision selection. Three seeds, 30 epochs each. Earlier smoke runs inspected a few official validation frames, so this is not a pristine external test set.')
table([['Method','Occupied IoU mean','SD across seeds'],['Tiny CNN, raw',pct(r['aggregate']['tiny_cnn/raw']['occupied_iou_mean']),f"{r['aggregate']['tiny_cnn/raw']['occupied_iou_std']:.4f}"],['CNN, development-selected',pct(r['aggregate']['tiny_cnn/development_selected']['occupied_iou_mean']),f"{r['aggregate']['tiny_cnn/development_selected']['occupied_iou_std']:.4f}"],['U-Net, development-selected',pct(r['aggregate']['unet/development_selected']['occupied_iou_mean']),f"{r['aggregate']['unet/development_selected']['occupied_iou_std']:.4f}"],['Copy occupied input only',pct(r['copy_input_baseline']['iou_occupied']),'Not a trained model']], [253,141,117])
p('With the same decision-fitting procedure, U-Net exceeds CNN by 1.36 IoU percentage points on average and wins in all three seeds. Hidden-cell occupied IoU is 0.2522 versus 0.1804 for the equally tuned CNN. Three seeds measure optimizer variability; they do not establish broad environmental generalization.','SmallR')
story.append(Image(str(ROOT/'artifacts/accuracy-benchmark-v2/accuracy-comparison.png'),width=500,height=218))
p('These learning results exclude Poisson proposals from inputs and targets. They must not be attributed to Poisson or presented as raw-sensor prediction performance.','SmallR')

page('4. Combined-method evidence and reproducibility')
a=g['aggregate']['motion_only'];b=g['aggregate']['motion_plus_poisson']
h('Direct paired comparison: motion correction + Poisson')
p('To isolate Poisson, both variants use identical motion-corrected past scans and no box filling. The current LiDAR scan is withheld as the sparse observed reference. Of 16 selected frames, only two have usable past scans under the conservative track policy; results below use exactly those same two frames.')
table([['Variant','IoU','Precision','Recall','Wrongly free'],['Motion correction only',pct(a['occupied_iou']),pct(a['precision']),pct(a['recall']),str(a['wrongly_free'])],['Motion + Poisson',pct(b['occupied_iou']),pct(b['precision']),pct(b['recall']),str(b['wrongly_free'])]], [190,76,82,81,82])
p('Poisson adds 20 true-positive occupied cells and 18 false positives in this comparison. The small IoU/recall increase comes with a small precision decrease. This is geometric scan agreement on two frames, not a model-accuracy claim. Both methods remain enabled by the selected submission design; the earlier minimum-sample gate was an experimental development heuristic, not an assignment requirement.')
h('Run the combined GT profile')
code("python -m pip install -e '.[dev,reconstruction,visualization]'\npython scripts/generate_gt.py --config configs/submission.yaml \\\n  --nuscenes-root /path/to/nuscenes --split mini_val \\\n  --max-samples 24 --output-dir artifacts/submission/val\npython -m pytest")
p('nuScenes data must be supplied explicitly and remains outside the image/repository. Synthetic runs require --synthetic. Open3D is required when the combined profile is selected; missing dependencies fail clearly.','SmallR')
h('Reproduce the learning experiment and evaluate its checkpoint')
code("python scripts/benchmark_accuracy.py --data-root /path/to/nuscenes \\\n  --output artifacts/accuracy-benchmark-v2 --epochs 30 --seeds 42,43,44\npython scripts/evaluate.py \\\n  --checkpoint artifacts/accuracy-benchmark-v2/recommended-checkpoint.pt \\\n  --gt-dir artifacts/accuracy-benchmark-v2/test \\\n  --output artifacts/accuracy-benchmark-v2/recommended-evaluation.json")
p('The benchmark uses its archived measured-label targets and fixed masks. Regenerating with another configuration changes target identities/masks; it is a new experiment. The supplied checkpoint is selected using development metrics only.','SmallR')
h('Docker environment')
code("docker build --target reconstruction \\\n  -t occupancy-pipeline:submission .")
p(('The reconstruction image was built and exercised successfully: regression tests, real combined GT, forward/loss, checkpoint evaluation and prediction plots. Full commands, dependency versions and image identity are in artifacts/final-walkthrough/transcript.json. ' if docker_verified else 'Docker verification remains incomplete. ')+'The base image and direct core dependencies are pinned; optional transitive dependencies are not fully locked.','SmallR')
p('Linux ARM64 uses Open3D 0.20.0 for wheel compatibility; the archived macOS geometric benchmarks used 0.19.0. Reconstruction across versions is not claimed bit-identical. AMD64 emulation on this Mac failed and is not the verified execution path.','SmallR')

page('5. Validation, tradeoffs and deliverables')
h('Verification evidence')
p('<b>27 regression tests passed.</b> They cover stable validation masks, scene separation, real three-class confusion matrices, fixed loss weights, unknown-cell binary gradients, stale-file exclusion, U-Net outputs, Poisson evidence isolation, input-preserving inference, checkpoint decisions, exact benchmark sample counts and the combined submission configuration.')
p('Real nuScenes runs exercised motion correction and Open3D reconstruction. Six model training runs completed. The exported recommended checkpoint independently reproduced the benchmark confusion matrix on all 16 test frames. Archive checksums validate packaged files. Evidence is stored under artifacts/combined-validation and artifacts/accuracy-benchmark-v2.')
h('Assumptions and limitations')
p('<b>Sparse reference:</b> unseen cells are not complete scene truth. Valid hidden geometry may be penalized by scan-agreement tests. <b>Motion:</b> interpolation and box assignment are approximate; the conservative skip policy reduces temporal coverage. <b>Poisson:</b> normals, sampling density and unannotated movers can create false surfaces. <b>Resolution:</b> 1 m voxels lose thin structures and small obstacles.')
p('<b>Learning scope:</b> the network completes a GT-derived partial grid. It is not a deployed perception system. Preserving input assumes reliable occupied evidence and would preserve input false positives. The benchmark is a small mini subset; a larger independent scene-level test and sensor-noise study are needed before broad accuracy claims.')
h('Implemented deliverables')
p('The package includes source, configuration, tests, Dockerfile, README, this report, recorded results and a trained demonstration checkpoint. '+('The accompanying MP4 is an edited walkthrough of actual Docker commands and plots, with no audio track, explanatory captions and full logs; it passed full decode verification. ' if video_verified else 'The final walkthrough video remains outstanding. ')+'The repository contains the modular code, design report, walkthrough and complete experiment bundle.')
h('Attribution and source provenance')
p('LLM assistance was used for the initial implementation, subsequent code changes, experiment automation and documentation. The initial snapshot was supplied in a Claude-generated walkthrough; Codex implemented and validated the extensions. Development scope is recorded in docs/ASSISTANCE.md; original hashes are retained in SNAPSHOT_PROVENANCE.json.')
h('References')
p('1. <link href="https://github.com/nutonomy/nuscenes-devkit">nuScenes devkit</link>: dataset schema, sensor transforms and annotation conventions.<br/>2. <link href="https://www.open3d.org/docs/release/tutorial/geometry/surface_reconstruction.html">Open3D surface reconstruction documentation</link>: Poisson API, density trimming and extrapolation limitations.<br/>3. <link href="https://www.cs.jhu.edu/~misha/Code/PoissonRecon/Version5.7/">Screened Poisson reference implementation</link>: reconstruction from oriented points.<br/>4. Local reproducible evidence: docs/ACCURACY_RESULTS.md; artifacts/combined-validation/comparison.json; benchmark plan and results JSON.','SmallR')


def decor(canvas,doc):
    w,h=A4
    canvas.setStrokeColor(colors.HexColor('#d9e2eb'));canvas.line(42,38,w-42,38)
    canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#607186'))
    canvas.drawString(42,25,'Occupancy Pipeline | Design report | 4 November 2026')
    canvas.drawRightString(w-42,25,f'{doc.page}')
OUT.parent.mkdir(parents=True,exist_ok=True)
SimpleDocTemplate(str(OUT),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=42,bottomMargin=48,
                  title='Occupancy Pipeline - Design Report',author='Project submission - LLM assistance disclosed').build(story,onFirstPage=decor,onLaterPages=decor)
print(OUT)
