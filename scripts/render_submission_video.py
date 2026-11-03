"""Silent, edited walkthrough from actual Docker logs and result images.

This is a programmatically rendered command walkthrough, not a live screen capture.
No audio is generated. Explanatory captions and raw command logs are retained.
Requires Pillow and imageio-ffmpeg; media generation is separate from the runtime.
"""
import json
from pathlib import Path
import re
import subprocess
import textwrap
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/final-walkthrough'
W,H=1280,720
BG=(13,23,38);FG=(225,232,242);TEAL=(88,211,192);MUTED=(169,188,207)
FFMPEG=imageio_ffmpeg.get_ffmpeg_exe()


def font(size,bold=False,mono=False):
    path='/System/Library/Fonts/Menlo.ttc' if mono else '/System/Library/Fonts/Supplemental/Arial Bold.ttf' if bold else '/System/Library/Fonts/Supplemental/Arial.ttf'
    return ImageFont.truetype(path,size)


def wrap(text,f,width):
    lines=[];line=''
    for word in text.split():
        trial=(line+' '+word).strip()
        if f.getlength(trial)>width and line:lines.append(line);line=word
        else:line=trial
    return lines+[line]


def frame(title,subtitle):
    im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im)
    d.text((40,22),'OCCUPANCY PIPELINE / SUBMISSION WALKTHROUGH',font=font(17,True),fill=TEAL)
    d.text((40,58),title,font=font(33,True),fill=FG)
    for i,line in enumerate(wrap(subtitle,font(19),1190)):d.text((40,109+i*25),line,font=font(19),fill=MUTED)
    d.line((40,668,1240,668),fill=(56,74,93),width=2)
    d.text((40,684),'Actual command logs and artifacts | Edited walkthrough | No audio',font=font(16),fill=TEAL)
    return im


def bullet_frame(title,subtitle,bullets):
    im=frame(title,subtitle);d=ImageDraw.Draw(im);y=190
    for heading,body in bullets:
        d.text((48,y),heading,font=font(25,True),fill=TEAL);y+=38
        for line in wrap(body,font(22),1160):d.text((48,y),line,font=font(22),fill=FG);y+=29
        y+=24
    if y>650:raise ValueError(f'Frame text overflows: {title}')
    return im


def image_frame(title,subtitle,path):
    im=frame(title,subtitle)
    with Image.open(path) as plot:
        plot=plot.convert('RGB');plot.thumbnail((1180,470));im.paste(plot,((W-plot.width)//2,180+(470-plot.height)//2))
    return im


def terminal_frame(stage,subtitle,extra=None):
    im=frame(stage['title'],subtitle);d=ImageDraw.Draw(im)
    command='$ '+' '.join(stage['command'])
    # Hide long machine-specific mount prefixes only in the readable display.
    command=command.replace(str(ROOT),'<project>').replace(str(ROOT.parent/'data/nuscenes'),'<nuScenes>')
    lines=textwrap.wrap(command,width=117,break_long_words=True)
    events=[re.sub(r'\x1b\[[0-9;]*m','',e['text']) for e in stage['events'] if e['text'].strip()]
    if extra is not None:events=extra
    else:events=events[-14:]
    lines+=['']+[line[:132] for line in events]
    y=181
    for index,line in enumerate(lines[:24]):
        d.text((44,y),line,font=font(15,mono=True),fill=TEAL if index<len(textwrap.wrap(command,width=117)) else FG);y+=18
    d.text((44,632),f"Exit {stage['exit_code']} | Actual elapsed {stage['elapsed_seconds']:.1f}s | Readable excerpt; full transcript included",font=font(17),fill=MUTED)
    return im


def timestamp(seconds):
    ms=round(seconds*1000);h,ms=divmod(ms,3600000);m,ms=divmod(ms,60000);s,ms=divmod(ms,1000)
    return f'{h:02}:{m:02}:{s:02},{ms:03}'


def main():
    transcript=json.loads((OUT/'transcript.json').read_text())
    if len(transcript)!=7 or any(r['exit_code'] for r in transcript):raise ValueError('Complete successful Docker recording required')
    demo=json.loads((OUT/'demo-summary.json').read_text())
    evaluation=json.loads((OUT/'checkpoint-evaluation.json').read_text())
    learning=demo['learning'];score=evaluation['metrics']['iou_occupied']
    stages={r['title']:r for r in transcript}
    chapters=[]
    def add(im,text):chapters.append((im,text))
    add(bullet_frame('A reproducible occupancy pipeline','Motion correction and Poisson reconstruction are both enabled.',[
      ('Ground truth','Real nuScenes LiDAR, time-aware scan alignment, ray evidence and separate surface proposals.'),
      ('Learning','A three-class U-Net completes a partially hidden occupancy grid.'),
      ('Video format','Silent walkthrough of actual Docker command logs and result images, with explanatory captions.')]),
      'This walkthrough demonstrates the final occupancy pipeline. Motion correction and Poisson reconstruction are both enabled. We start by building the Docker environment, run the tests, generate ground truth from real nuScenes scans, and compute a model forward pass and loss. We then evaluate a trained checkpoint. This is a silent edited walkthrough generated from actual recorded execution. Explanatory captions and complete command logs are included. It is not a live screen capture.')
    build=stages['Build reproducible environment']
    add(terminal_frame(build,'Clean declared environment; CPU PyTorch and Open3D are installed inside the image.'),
      'The first recorded command builds the reconstruction target from the Dockerfile. The Python base image is pinned by digest. CPU PyTorch, the project, regression tools and Open3D are installed in the image, so the host Python environment is not required to run the pipeline. On this ARM Mac we use native Linux ARM64 wheels. The AMD64 emulation attempt exposed an illegal-instruction failure and was not accepted. Build waiting time is edited out of the video; the transcript retains the complete output and actual duration.')
    add(terminal_frame(stages['Run regression tests'],'Tests execute in a fresh container built from the declared environment.'),
      'Next, the regression suite runs in a fresh container. The tests check repeatable masking, scene separation, genuine three-class metrics, loss behavior, U-Net outputs, proposal isolation, input-preserving decisions and the selected submission configuration. They also execute an actual Poisson reconstruction. These checks establish software behavior. They do not establish that every reconstructed surface is correct or that the system is safe for navigation.')
    add(bullet_frame('How the geometry flows','Files: data/nuscenes_adapter.py and gt_generation/evidence.',[
      ('One coordinate frame','Sensor -> sweep ego -> global -> keyframe ego; each scan retains its own origin.'),
      ('Moving objects','Instance tracks and interpolated boxes align object points to keyframe poses.'),
      ('Safe ray evidence','Reposed points contribute occupied hits only. Ambiguous tracks are skipped and logged.')]),
      'Every scan is transformed through its own sensor calibration and ego pose into the keyframe ego frame. This avoids treating measurements from a moving vehicle as if the sensor had stayed still. Tracked object boxes are interpolated in time, and points on moving objects are reposed to their keyframe position. Those shifted points contribute hits only. We never cast an artificial free ray toward the shifted position. Unbracketed or unmatched annotation tracks are conservatively skipped and recorded.')
    add(terminal_frame(stages['Generate combined GT and compute model loss'],'Real-data execution: both stages active; no synthetic fallback.',[
      'profile: configs/submission.yaml','motion_compensation: true','poisson_enabled: true','box_fill: false',
      *[f"sample {s['sample_token']}: sweeps={s['stats']['sweeps_used']}, occupied={s['stats']['occupied_voxels']}, free={s['stats']['free_voxels']}, Poisson proposals={s['stats']['poisson_proposal_voxels']}" for s in demo['samples']],
      f"input_shape: {learning['input_shape']}",f"logit_shape: {learning['logit_shape']}",f"forward CE loss: {learning['loss']:.6f}"]),
      'This real-data command uses two explicitly selected nuScenes keyframes and mounts the dataset read only. Both motion correction and Poisson are active. Box-volume filling is disabled so that the two selected methods remain interpretable. The recorded summary shows the actual number of usable scans and generated labels. It also shows the dataset input shape, three-class output shape and a finite cross-entropy loss. This forward demonstration uses a newly initialized model; the trained checkpoint is evaluated separately later.')
    add(image_frame('Measured labels and inferred surfaces','Poisson output is retained, but never overwrites measured occupied or free evidence.',OUT/'combined-gt.png'),
      'The left image shows motion-corrected sensor labels. The right adds inferred Poisson surface proposals. Unknown, occupied and observed free space are distinct labels. Poisson estimates normals, reconstructs static geometry, trims low-density regions and rejects surfaces without nearby point support. Its proposals are saved separately and are restricted to unknown cells. This allows us to retain the reconstructed map without silently treating inferred geometry as verified training truth. It can fill gaps, but it can also add false surfaces.')
    add(bullet_frame('The learning interface','Files: occupancy_dataset.py, occupancy_unet.py and training/losses.py.',[
      ('Input and target','One partial occupied-grid channel; 30% of occupied input cells hidden. Targets retain all three classes.'),
      ('Model','U-Net encoder/decoder, XY pooling, retained height and skip connections.'),
      ('Loss and validation','Fixed training-set class weights; pooled confusion matrices; scenes stay separate.')]),
      'The learning task is occupancy mask completion. The dataset hides thirty percent of occupied cells in the input grid while retaining the complete three-class target. No full ground-truth free channel is passed to the model. The U-Net pools the horizontal axes while retaining height and uses skip connections for spatial detail. Cross-entropy weights are computed from training data once. Validation masks are stable, and whole-scene splits prevent adjacent training and validation frames from being mixed. This is not camera or fresh raw-LiDAR prediction.')
    add(terminal_frame(stages['Evaluate the trained checkpoint'],'A separate trained checkpoint is evaluated on every supplied held-out frame.',[
      f"samples_evaluated: {evaluation['samples_evaluated']}",'scene_disjoint: true',
      f"occupied_iou: {score:.6f}",f"mean_iou: {evaluation['metrics']['mean_iou']:.6f}",
      f"mask_seed: {evaluation['mask_seed']}",f"inference_decision: {evaluation['inference_decision']}",
      f"confusion_matrix: {evaluation['metrics']['confusion_matrix']}"]),
      f'Now we load the trained checkpoint and evaluate all {evaluation["samples_evaluated"]} held-out frames. Evaluation rejects overlap with training scenes. The occupied score setting was chosen on development scenes, not on these test results. Inference preserves occupied cells already supplied in the input. The Docker run measures occupied intersection over union of {score*100:.2f} percent for this single checkpoint. The recorded three-seed benchmark averages seventy four point nine nine percent. Those are different summaries of the same completion task.')
    add(image_frame('Actual trained-model predictions','Unknown, occupied and free classes are model outputs; no free labels are copied from GT.',OUT/'predictions/sample_000.png'),
      'This plot is generated by the checkpoint visualization command, using the actual held-out input and model output. All three classes come from the model decision. We do not copy free-space labels from ground truth into the prediction. The checkpoint stores its architecture, development decision setting and mask seed, so evaluation and visualization use matching inputs. The raw source labels and inferred Poisson proposals remain distinct artifacts.')
    add(image_frame('Measured completion results','42 training / 6 development / 16 test frames; 3 optimizer seeds.',ROOT/'artifacts/accuracy-benchmark-v2/accuracy-comparison.png'),
      'The controlled comparison uses forty two training frames from seven scenes, six development frames from one different scene, and sixteen test frames from two official mini validation scenes. Both architectures use the same optimization budget and decision search. With development selection, the CNN averages seventy three point six three percent occupied IoU, and the U-Net averages seventy four point nine nine percent. The U-Net wins in all three seeds. Hidden-cell scores are reported separately, so copying known input cells is not mistaken for learned completion.')
    add(bullet_frame('What the results do and do not establish','The report separates geometry quality from model completion accuracy.',[
      ('Combined geometry','Motion + Poisson: 67.47% geometric IoU versus 67.28% for motion alone on two eligible scans.'),
      ('Evidence limitations','Sparse reference scans, approximate box tracks, limited scenes and possible Poisson false surfaces.'),
      ('Reproducibility','Dockerfile, full logs, dependency versions, tests, manifests, checkpoint and report are included.')]),
      'The direct geometric comparison, with no box filling, measures sixty seven point four seven percent IoU for motion correction plus Poisson, versus sixty seven point two eight percent for motion correction alone. Only two scans are eligible, so this small gain should not be generalized. The model completion benchmark is separate and excludes Poisson proposals from its inputs and targets. Important remaining limitations include sparse reference geometry, approximate object tracks and only two test scenes. The source, report, configuration, checksums, dependency records and full execution transcripts are supplied for review.')
    segments=[];captions=[];notes=[];elapsed=0.;counter=1
    for i,(im,text) in enumerate(chapters):
        prefix=OUT/f'chapter-{i:02d}';im.save(prefix.with_suffix('.png'))
        prefix.with_suffix('.txt').write_text(text)
        duration=max(18.,len(text.split())/155.*60.+2.)
        segment=prefix.with_suffix('.mp4')
        subprocess.run([FFMPEG,'-hide_banner','-loglevel','error','-y','-loop','1','-framerate','15','-i',str(prefix.with_suffix('.png')),
                        '-t',str(duration),'-an','-c:v','libx264','-preset','veryfast','-tune','stillimage','-pix_fmt','yuv420p',str(segment)],check=True)
        probe=subprocess.run([FFMPEG,'-hide_banner','-i',str(segment)],capture_output=True,text=True)
        h,m,s=map(float,re.search(r'Duration:\s*(\d+):(\d+):([\d.]+)',probe.stderr).groups())
        segment_duration=h*3600+m*60+s
        segments.append(segment)
        chunks=textwrap.wrap(text,width=142,break_long_words=False)
        for j,chunk in enumerate(chunks):
            start=elapsed+(duration-1)*j/len(chunks);end=elapsed+(duration-1)*(j+1)/len(chunks)
            captions.append(f'{counter}\n{timestamp(start)} --> {timestamp(end)}\n'+textwrap.fill(chunk,width=74)+'\n');counter+=1
        notes.append(f'Chapter {i+1}\n{text}\n');elapsed+=segment_duration
        print(f'Rendered chapter {i+1}/{len(chapters)}',flush=True)
    listing=OUT/'segments.txt';listing.write_text('\n'.join("file '"+str(p)+"'" for p in segments)+'\n')
    srt=OUT/'captions.srt';srt.write_text('\n'.join(captions))
    (OUT/'walkthrough-notes.txt').write_text('\n'.join(notes))
    joined=OUT/'joined.mp4'
    subprocess.run([FFMPEG,'-hide_banner','-loglevel','error','-y','-f','concat','-safe','0','-i',str(listing),'-c','copy',str(joined)],check=True)
    final=OUT/'walkthrough.mp4'
    subprocess.run([FFMPEG,'-hide_banner','-loglevel','error','-y','-i',str(joined),'-i',str(srt),'-map','0:v:0','-map','1:0','-an','-c:v','copy','-c:s','mov_text',
                    '-metadata:s:s:0','language=eng','-movflags','+faststart',str(final)],check=True)
    decoded=subprocess.run([FFMPEG,'-hide_banner','-loglevel','error','-i',str(final),'-f','null','-'],capture_output=True,text=True)
    if decoded.returncode:raise RuntimeError(decoded.stderr)
    probe=subprocess.run([FFMPEG,'-hide_banner','-i',str(final)],capture_output=True,text=True)
    if re.search(r'Stream #.*Audio:',probe.stderr):raise RuntimeError('Silent walkthrough unexpectedly contains audio')
    manifest={'video':final.name,'chapters':len(chapters),'estimated_duration_seconds':round(elapsed,2),
              'narration':'none','audio_streams':0,'source':'Edited from actual recorded Docker commands, not a live screen capture',
              'full_decode_passed':True,'subtitles':'captions.srt','width':W,'height':H}
    (OUT/'video-manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':main()
