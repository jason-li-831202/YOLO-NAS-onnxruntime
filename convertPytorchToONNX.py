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
from io import BytesIO
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn

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
        anchor_points = torch.cat(anchor_points)
        stride_tensor = torch.cat(stride_tensor)
        return anchor_points, stride_tensor

    def _batch_distance2bbox(self, points, distance) :
        lt, rb = torch.split(distance, 2, dim=-1)
        # while tensor add parameters, parameters should be better placed on the second place
        x1y1 = -lt + points
        x2y2 = rb + points
        return torch.cat([x1y1, x2y2], dim=-1)

    def forward(self, feats):
        output = []
        for i, feat in enumerate(feats):
            b, _, h, w = feat.shape
            height_mul_width = h * w
            reg_distri, cls_logit = getattr(self, f"head{i + 1}")(feat)

            reg_dist_reduced = torch.permute(reg_distri.reshape([-1, 4, self.reg_max + 1, height_mul_width]), [0, 2, 3, 1])
            reg_dist_reduced = nn.functional.conv2d(nn.functional.softmax(reg_dist_reduced, dim=1), weight=self.proj_conv).squeeze(1)

            # cls and reg
            pred_scores = cls_logit.sigmoid()
            pred_conf, _ = pred_scores.max(1, keepdim=True)
            pred_bboxes = torch.permute(reg_dist_reduced, [0, 2, 1]).reshape([-1, 4, h, w])
            
            pred_output = torch.cat([ pred_bboxes , pred_conf, pred_scores], dim=1)
            bs, na, ny, nx = pred_output.shape
            
            pred_output = pred_output.view(bs, 1, na, -1).permute(0, 1, 3, 2).contiguous()  # (b, na,20x20,85) for NCNN
            pred_output = pred_output.view(bs, -1, na)

            output.append(pred_output)

        cat_output = torch.cat(output, dim=1) 
        cat_output[:, :, :4] = self._batch_distance2bbox(self.anchor_points, cat_output[:, :, :4]) * self.stride_tensor

        return cat_output


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
        "-n",
        "--name",
        type=str,
        help="The name of the model to be saved, none means using the same name as the input model",
    )
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

    if parse_arg.name is None:
        parse_arg.name = parse_arg.input_model

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


def export(input_model, img_size, output_model, checkpoint_path, opset, class_names, **kwargs):
    t = time.time()
    from super_gradients.training import models

    # Load PyTorch model
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

        print()
        print("Starting ONNX export with onnx %s..." % onnx.__version__)
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
            onnx_model = onnx.load_from_string(f.getvalue())  # load onnx model
            onnx.checker.check_model(onnx_model)  # check onnx model

        try:
            import onnxsim

            print("Starting to simplify ONNX...")
            onnx_model, check = onnxsim.simplify(onnx_model)
            assert check, "assert check failed"

        except ImportError:
            print(
                "onnxsim is not found, if you want to simplify the onnx, "
                + "you should install it:\n\t"
                + "pip install -U onnxsim onnxruntime\n"
                + "then use:\n\t"
                + f'python -m onnxsim "{output_model}" "{output_model}"'
            )
        except Exception:
            print("Simplifier failure")

        onnx.save(onnx_model, output_model)
        print("ONNX export success, saved as:\n\t%s" % output_model)

    except Exception:
        print("ONNX export failure")

    # generate anchors and sides
    anchors = []

    # generate masks
    masks = dict()

    print("anchors:\n\t%s" % anchors)
    print("anchor_masks:\n\t%s" % masks)
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
    print("Anchors data export success, saved as:\n\t%s" % export_json)

    # Finish
    print("Export complete (%.2fs).\n" % (time.time() - t))


if __name__ == "__main__":
    args = parse_args()
    print(args)
    output_model = args.output_dir / (args.name + ".onnx")

    export(output_model=output_model, **vars(args))


