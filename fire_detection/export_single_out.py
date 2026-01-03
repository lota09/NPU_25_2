
import sys
import torch
import torch.nn as nn
from pathlib import Path
from copy import deepcopy
import argparse

# YOLOv7 경로 추가
sys.path.insert(0, str(Path(__file__).parent / 'yolov7'))

from models.experimental import attempt_load

def replace_activations(model, target_act=nn.SiLU, new_act_cls=nn.LeakyReLU, verbose=True):
    replaced_count = 0
    for name, module in model.named_modules():
        if isinstance(module, target_act):
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = model
            for part in parent_name.split('.'):
                if part:
                    parent = getattr(parent, part)
            new_activation = new_act_cls(negative_slope=0.1, inplace=True)
            setattr(parent, child_name, new_activation)
            replaced_count += 1
    if verbose:
        print(f"Replaced {replaced_count} activations.")
    return replaced_count

def export_model(
    weights_path,
    output_name,
    use_leaky=False,
    img_size=(640, 640),
    device='cpu',
    opset_version=12
):
    print(f"Exporting {output_name} | LeakyReLU={use_leaky} | Single Output")
    weights_path = Path(weights_path)
    output_path = weights_path.parent / output_name
    
    model = attempt_load(str(weights_path), map_location=device)
    
    if use_leaky:
        replace_activations(model, target_act=nn.SiLU, new_act_cls=nn.LeakyReLU)
        
    # Force Single Output (export=False -> returns 1 tensor, usually [1, 25200, 85])
    # Check model Detect output.
    if hasattr(model.model[-1], 'export'):
        model.model[-1].export = False 
        print("Set Detect.export = False (Single Concatenated Output)")
    
    model.eval()
    # for p in model.parameters():
    #     p.requires_grad = False
    
    dummy_input = torch.zeros(1, 3, img_size[0], img_size[1], device=device)
    
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=['images'],
        output_names=['output'], # Single Output Name
        dynamic_axes=None,
        verbose=False,
        opset_version=opset_version,
        do_constant_folding=True,
    )
    print(f"Saved to {output_path}")

if __name__ == '__main__':
    # usage: python export_single_out.py --leaky --output v7_leaky_single.onnx
    # usage: python export_single_out.py --output v7_silu_single.onnx
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='models/v7_merged_100epoch_16batch.pt')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--leaky', action='store_true')
    args = parser.parse_args()
    
    export_model(args.weights, args.output, use_leaky=args.leaky)
