"""Withhold each current scan; compare past-sweep evidence on identical samples."""
from dataclasses import replace
from occ_pipeline.gt_generation.evidence.temporal_gt import generate_temporal_occupancy
from occ_pipeline.utils.io import write_json

def counts_to_metrics(tp, fp, fn):
    return {'tp': tp, 'fp': fp, 'fn': fn,
            'occupied_iou': tp/(tp+fp+fn) if tp+fp+fn else None,
            'precision': tp/(tp+fp) if tp+fp else None,
            'recall': tp/(tp+fn) if tp+fn else None}

def compare_labels(predicted, reference):
    """Score only independently observed reference cells; unknown predictions abstain.

    Abstaining on a reference hit still counts as a missed occupied cell.
    Predictions outside reference coverage are not scored as false positives.
    """
    known = reference != -1
    occupied, truth = predicted == 1, reference == 1
    tp = int((occupied & truth & known).sum())
    fp = int((occupied & ~truth & known).sum())
    fn = int((~occupied & truth & known).sum())
    return {**counts_to_metrics(tp, fp, fn),
            'reference_observed': int(known.sum()),
            'prediction_known_on_reference': int(((predicted != -1) & known).sum()),
            'wrongly_free': int(((predicted == 0) & truth).sum())}

def compare_ground_truth(source, config, output, split='val', limit=2, include_poisson=False):
    """Withhold the current LiDAR scan; build candidates using only older sweeps.

    Keyframe annotations remain permitted priors. These scores measure agreement
    with a sparse scan, not complete-scene ground-truth accuracy.
    """
    variants = [('single_past', dict(num_sweeps=2, motion_compensation=False, box_fill=False)),
                ('multi_past', dict(motion_compensation=False, box_fill=False)),
                ('motion_corrected', dict(motion_compensation=True, box_fill=False)),
                ('box_proposals', dict(motion_compensation=True, box_fill=True))]
    if include_poisson:
        variants.append(('poisson_proposals', dict(motion_compensation=True, box_fill=True, poisson_enabled=True)))
    rows = []
    for sample in source.samples(split, limit):
        token = sample['token']
        scan = {'lidar_sample_data_token': source.channels[token]['LIDAR_TOP']['token']}
        # Reference retains every valid measured hit, irrespective of candidate denoising.
        ref_config = replace(config, num_sweeps=1, motion_compensation=False, box_fill=False,
                             poisson_enabled=False, min_hits_per_voxel=1)
        ref_sweeps, _, _ = source.load_sweeps(token, ref_config)
        reference, _ = generate_temporal_occupancy(ref_sweeps, [], ref_config)
        row = {'sample_token': token, 'scene_token': sample['scene_token'], 'variants': {}}
        for name, overrides in variants:
            options = dict(poisson_enabled=False, num_sweeps=max(2, config.num_sweeps))
            options.update(overrides)
            settings = replace(config, **options)
            sweeps, boxes, skipped = source.load_sweeps(token, settings)
            past = [s for s in sweeps if s['sample_data_token'] != scan['lidar_sample_data_token']]
            if not past:
                row['variants'][name] = {'status': 'skipped_no_usable_past_sweeps', 'skipped_sweeps': skipped}
                continue
            result, evidence = generate_temporal_occupancy(past, boxes, settings)
            prediction = evidence['proposed_labels'] if name.endswith('proposals') else result.labels
            row['variants'][name] = {'status': 'completed', 'past_sweeps': len(past),
                                     'held_out_token': scan['lidar_sample_data_token'],
                                     'input_tokens': [s['sample_data_token'] for s in past],
                                     'metrics': compare_labels(prediction, reference.labels), 'stats': result.stats,
                                     'skipped_sweeps': skipped}
        rows.append(row)
    aggregate = {}
    paired = {}
    common = [r for r in rows if all(v['status'] == 'completed' for v in r['variants'].values())]
    for name, _ in variants:
        values = [r['variants'][name]['metrics'] for r in rows if r['variants'][name]['status'] == 'completed']
        sums = {key: sum(v[key] for v in values) for key in ('tp','fp','fn','reference_observed','prediction_known_on_reference','wrongly_free')}
        aggregate[name] = {**counts_to_metrics(sums.pop('tp'), sums.pop('fp'), sums.pop('fn')), **sums, 'samples_scored': len(values)}
        values = [r['variants'][name]['metrics'] for r in common]
        sums = {key: sum(v[key] for v in values) for key in ('tp','fp','fn','reference_observed','prediction_known_on_reference','wrongly_free')}
        paired[name] = {**counts_to_metrics(sums.pop('tp'), sums.pop('fp'), sums.pop('fn')), **sums, 'samples_scored': len(values)}
    report = {'source_kind': source.source_kind, 'split': split, 'config': config.to_dict(),
              'protocol': 'current scan withheld; past LiDAR only; keyframe annotation priors allowed',
              'limitation': 'Sparse held-out scan agreement, not complete-scene accuracy. Box/Poisson are inferred proposals.',
              'samples': rows, 'aggregate': aggregate, 'paired_aggregate': paired,
              'paired_sample_tokens': [r['sample_token'] for r in common],
              'comparison_note': 'Use paired_aggregate for like-for-like comparisons. Aggregate sample counts vary when sweeps are skipped.'}
    write_json(output, report)
    return report
