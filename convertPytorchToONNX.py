#!/usr/bin/env python3
# -*- coding:utf-8 -*-

#================================================
# Author: jason-li-831202
# @File: convertPytorchToONNX.py
# @Software: Visual Stidio Code
#================================================

import argparse
import json
import time
import warnings

import torch
import torch.nn as nn

from io import BytesIO
from pathlib import Path
from onnxconverter_common import float16
from torchsummary import summary

warnings.filterwarnings("ignore")

# python3 convertPytorchToONNX.py -m yolo_nas_s -o ./models -imgsz 640 640 

ROOT = Path(__file__).resolve().parent

yolo_nas = [
    "yolo_nas_s",
    "yolo_nas_m",
    "yolo_nas_l",
]

class DetectNAS(nn.Module):
    """YOLO-NAS Detect head for detection models"""

    def __init__(self, old_detect, input_size):
        super().__init__()
        self.num_classes = old_detect.num_classes  # number of classes
        self.reg_max = old_detect.reg_max
        self.num_heads = old_detect.num_heads
        self.proj_conv = old_detect.proj_conv

        self.grid_cell_offset = old_detect.grid_cell_offset # 0.5
        self.fpn_strides = old_detect.fpn_strides # 8, 16, 32

        self.anchor_points, self.stride_tensor = self._generate_anchors(input_size[0], input_size[1]) 

        for i in range(self.num_heads):
            setattr(self, f"head{i + 1}", getattr(old_detect, f"head{i + 1}"))

    def _generate_anchors(self, target_height, target_width, dtype=torch.float):
        anchor_points = []
        stride_tensor = []
        for i, stride in enumerate(self.fpn_strides):
            num_grid_w = int(target_width / stride)
            num_grid_h = int(target_height / stride)

            shift_x = torch.arange(end=num_grid_w) + self.grid_cell_offset
            shift_y = torch.arange(end=num_grid_h) + self.grid_cell_offset

            shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
            anchor_point = torch.stack([shift_x, shift_y], dim=-1).to(dtype=dtype)

            anchor_points.append(anchor_point.reshape([-1, 2]))
            stride_tensor.append(torch.full([num_grid_h * num_grid_w, 1], stride, dtype=dtype))
        return anchor_points, stride_tensor

    def _batch_distance2bbox(self, points, distance) :
        lt, rb = torch.split(distance, 2, dim=-1)
        # while tensor add parameters, parameters should be better placed on the second place
        x1y1 = -lt + points
        x2y2 = rb + points
        wh = x2y2 - x1y1
        cxcy = x1y1 + wh/2
        return torch.cat([cxcy, wh], dim=-1)

    def forward(self, feats):
        output = []
        for i, feat in enumerate(feats):
            b, _, h, w = feat.shape
            height_mul_width = h * w
            reg_distri, cls_logit = getattr(self, f"head{i + 1}")(feat)

            reg_dist_reduced = torch.permute(reg_distri.reshape([-1, 4, self.reg_max + 1, height_mul_width]), [0, 2, 3, 1])
            reg_dist_reduced = nn.functional.conv2d(nn.functional.softmax(reg_dist_reduced, dim=1), weight=self.proj_conv).squeeze(1)  # (b, ny*nx, x1y1x2y2)
            reg_dist_reduced = self._batch_distance2bbox(self.anchor_points[i], reg_dist_reduced) * self.stride_tensor[i] # (b, ny*nx, cxcywh)

            # cls and reg
            pred_scores = cls_logit.sigmoid() # (b, class_num, ny, nx)
            pred_conf, _ = pred_scores.max(1, keepdim=True) # (b, 1, ny, nx)
            pred_bboxes = torch.permute(reg_dist_reduced, [0, 2, 1]).reshape([-1, 4, h, w]) #(b, cxcywh, ny*nx) -> (b, cxcywh, ny, nx)

            pred_output = torch.cat([ pred_bboxes , pred_conf, pred_scores], dim=1)
            bs, na, ny, nx = pred_output.shape
            
            pred_output = pred_output.view(bs, na, -1).permute(0, 2, 1).contiguous()  # (b, ny*nx, na=class_num+5) for NCNN
            output.append(pred_output)

        cat_output = torch.cat(output, dim=1) 
        return cat_output

def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {
        'black': '\033[30m',  # basic colors
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',  # bright colors
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',  # misc
        'bold': '\033[1m',
        'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']

def file_size(path: str):
    # Return file/dir size (MB)
    mb = 1 << 20  # bytes to MiB (1024 ** 2)
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / mb
    elif path.is_dir():
        return sum(f.stat().st_size for f in path.glob('**/*') if f.is_file()) / mb
    else:
        return 0.0

def parse_args():
    parser = argparse.ArgumentParser(
        description="Tool for converting Yolov8 models to the blob format used by OAK",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m",
        "-i",
        "--input_model",
        type=str,
        help="model name ",
        default="yolo_nas_s",
        choices=yolo_nas,
    )
    parser.add_argument(
        "-imgsz",
        "--img-size",
        nargs="+",
        type=int,
        default=[640, 640],
        help="image size",
    )  # height, width

    parser.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        help="Directory for saving files, none means using the same path as the input model",
    )

    parser.add_argument(
        "-w",
        "--checkpoint_path",
        type=Path,
        help="The path with save the trained model parameters",
    )

    parser.add_argument(
        "-c",
        "--class_names",
        type=Path,
        help="The path to class names file.",
    )
    parser.add_argument("--half", action="store_true", help="conver to fp16")
    parser.add_argument("-op", "--opset", type=int, default=12, help="opset version")
    parser.add_argument(
        "-s",
        "--spatial_detection",
        action="store_true",
        help="Inference with depth information",
    )
    parser.add_argument(
        "-sh",
        "--shaves",
        type=int,
        help="Inference with depth information",
    )

    parse_arg = parser.parse_args()
    
    if parse_arg.output_dir is None:
        parse_arg.output_dir = ROOT.joinpath(parse_arg.input_model)

    parse_arg.output_dir = parse_arg.output_dir.resolve().absolute()

    parse_arg.output_dir.mkdir(parents=True, exist_ok=True)

    parse_arg.img_size *= 2 if len(parse_arg.img_size) == 1 else 1  # expand

    if parse_arg.shaves is None:
        parse_arg.shaves = 5 if parse_arg.spatial_detection else 6

    if parse_arg.checkpoint_path != None:
        assert (parse_arg.checkpoint_path.is_file()), "Can't find the model file."
        assert (parse_arg.checkpoint_path.suffix in {".pth", ".pt"}), "The model file extension is not in the PyTorch format."
        assert (parse_arg.class_names.is_file()), "Can't find the class names file."

        with open(parse_arg.class_names, 'rt') as f:
            classes = f.read().rstrip("\n").split("\n")
        parse_arg.class_names = classes
    return parse_arg

def export(input_model, img_size, output_model, checkpoint_path, opset, class_names, half, **kwargs):
    t = time.time()
    from super_gradients.training import models

    # Load PyTorch model
    print(colorstr("Loading pytorch path [%s] with torch %s..." %(input_model, torch.__version__) ))
    if checkpoint_path is None:
        model = models.get(input_model, pretrained_weights="coco")
        labels = model._class_names  # get class names
    else :
        model = models.get(input_model, checkpoint_path=str(checkpoint_path), num_classes=len(class_names))
        labels = class_names  # get class names
    # model.predict("./models/demo.png", conf=0.5).show()

    labels = labels if isinstance(labels, list) else list(labels.values())

    # check num classes and labels
    assert model.num_classes == len(labels), f"Model class count {model.num_classes} != len(names) {len(labels)}"

    # Replace with the custom Detection Head
    model.heads = DetectNAS(model.heads, img_size)

    num_branches = model.heads.num_heads

    # Input
    img = torch.zeros(1, 3, *img_size)
    model.eval()
    model.prep_model_for_conversion(input_size=[1, 3, *img_size])

    # ONNX export
    try:
        import onnx

        print(colorstr("Starting export with onnx %s..." % onnx.__version__))
        with BytesIO() as f:
            torch.onnx.export(
                model,
                img,
                f,
                verbose=False,
                opset_version=opset,
                input_names=["images"],
                output_names=["outputs"],
            )

            # Checks
            onnx_model_fp32 = onnx.load_from_string(f.getvalue())  # load onnx model
            onnx.checker.check_model(onnx_model_fp32)  # check onnx model

        try:
            import onnxsim

            print(colorstr("Starting to simplify onnx with %s..." % onnxsim.__version__))
            onnx_model_fp32, check = onnxsim.simplify(onnx_model_fp32)
            assert check, "assert check failed"
        except ImportError:
            print(
                "onnxsim is not found, if you want to simplify the onnx, "
                + "you should install it:\n\t"
                + "pip install -U onnxsim onnxruntime\n"
                + "then use:\n\t"
                + f'python -m onnxsim "{output_model}" "{output_model}"'
            )
        except Exception as e:
            print(colorstr('red', f'Eexport failure ❌ : {e}'))
            exit()

        # Convert to float16
        if half:
            try:
                import onnxconverter_common
                print(colorstr(f'Starting to convert float16 with onnxconverter_common {onnxconverter_common.__version__}...'))
                onnx_model_fp16 = float16.convert_float_to_float16(onnx_model_fp32, op_block_list=["Softmax", "ReduceMax", "ConvTranspose"])
            except ImportError:
                print(
                    "onnxconverter_common is not found, if you want to quant the onnx, "
                    + "you should install it:\n\t"
                    + "pip install -U onnxconverter_common\n"
                )
            except Exception as e:
                print(colorstr('red', f'Eexport failure ❌ : {e}'))
                exit()
        
        onnx.save(onnx_model_fp16 if half else onnx_model_fp32, output_model)
        print(colorstr('bright_magenta', "ONNX export success ✅ , saved as:\n\t%s" % output_model))

    except Exception as e:
        print(colorstr('red', f'Eexport failure ❌ : {e}'))
        exit()

    # generate anchors and sides
    anchors = []

    # generate masks
    masks = dict()

    print(colorstr('bright_magenta', "anchors:\n\t%s" % anchors))
    print(colorstr('bright_magenta', "anchor_masks:\n\t%s" % masks))
    print(colorstr('bright_magenta', "num_classes:\n\t%s" % model.num_classes))
    export_json = output_model.with_suffix(".json")
    export_json.write_text(
        json.dumps(
            {
                "nn_config": {
                    "output_format": "detection",
                    "NN_family": "YOLO",
                    "input_size": f"{img_size[0]}x{img_size[1]}",
                    "NN_specific_metadata": {
                        "classes": model.num_classes,
                        "coordinates": 4,
                        "anchors": anchors,
                        "anchor_masks": masks,
                        "iou_threshold": 0.3,
                        "confidence_threshold": 0.5,
                    },
                },
                "mappings": {"labels": labels},
            },
            indent=4,
        )
    )
    print(colorstr('bright_magenta', "Anchors data export success, saved as:\n\t%s" % export_json))

    # Finish
    print(colorstr('bright_magenta', "Export complete (%.2fs).\n" % (time.time() - t)))


if __name__ == "__main__":
    args = parse_args()
    print(colorstr(args))

    output_model = Path.joinpath(args.output_dir, args.input_model + ("_fp16" if args.half else "_fp32") + ".onnx")
    export(output_model=output_model, **vars(args))


