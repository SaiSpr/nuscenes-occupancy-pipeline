"""Fit an occupied decision bias on a checkpoint's recorded development set."""
import argparse
import importlib.util
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from occ_pipeline.data.occupancy_dataset import OccupancyDataset
from occ_pipeline.training.pipeline import build_model
from occ_pipeline.utils.io import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--dev-dir',required=True);p.add_argument('--output',required=True)
    a=p.parse_args()
    if Path(a.output).resolve()==Path(a.checkpoint).resolve():raise ValueError('Keep the original raw checkpoint; choose a different output')
    payload=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    protocol=payload.get('protocol',{})
    model_config=payload['config']['model']
    if model_config.get('out_channels',1)!=3:raise ValueError('Decision fitting currently expects a three-class model')
    dataset=OccupancyDataset(a.dev_dir,3,seed=protocol.get('validation_seed',protocol.get('seed',42)))
    if any(g is None for g in dataset.groups) or set(dataset.groups)!=set(protocol.get('val_scenes',[])):
        raise ValueError('Use exactly the development scenes recorded in the checkpoint')
    if sorted(p.name for p in dataset.files)!=sorted(protocol.get('val_files',[])):
        raise ValueError('Use exactly the development samples recorded in the checkpoint')
    spec=importlib.util.spec_from_file_location('accuracy_benchmark',Path(__file__).with_name('benchmark_accuracy.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    _,model=build_model(model_config);model.load_state_dict(payload['model_state_dict']);model.eval()
    selected,grid=module.select_decision_bias(model,dataset)
    payload['inference_decision']={'occupied_bias':selected['occupied_bias'],'preserve_input':True,'selected_on':'recorded development scenes only'}
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(payload,output)
    write_json(output.with_suffix('.json'),{'selected':selected,'grid':grid,'checkpoint':str(output)})
    print(selected)


if __name__=='__main__':main()
