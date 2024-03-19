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
import re, os

import torch
import torch.nn as nn
import cv2 as cv2
import numpy as np
import onnx
import onnxruntime as ort

from io import BytesIO
from pathlib import Path
from onnxconverter_common import float16
from onnxruntime.quantization import quantize_dynamic, quantize_static, CalibrationDataReader, QuantFormat, QuantType

warnings.filterwarnings("ignore")

# python3 convertPytorchToONNX.py -m yolo_nas_s -o ./models -imgsz 640 640 

ROOT = Path(__file__).resolve().parent

yolo_nas = [
	"yolo_nas_s",
	"yolo_nas_m",
	"yolo_nas_l",
]


class DataReader(CalibrationDataReader):
	def __init__(self, calibration_image_folder, augmented_model_path=None):
		self.image_folder = calibration_image_folder
		self.augmented_model_path = augmented_model_path
		self.preprocess_flag = True
		self.enum_data_dicts = []
		self.datasize = 0
		self.providers = []
		if  ort.get_device() == 'GPU' and 'CUDAExecutionProvider' in  ort.get_available_providers():  # gpu 
			self.providers.append('CUDAExecutionProvider')
		self.providers.append('CPUExecutionProvider')

	def get_next(self):
		if self.preprocess_flag:
			self.preprocess_flag = False
			session = ort.InferenceSession(self.augmented_model_path, providers=self.providers)
			self.input_shapes = session.get_inputs()[0].shape
			nhwc_data_list = self.proprocess_func(self.image_folder, self.input_shapes)
			input_name = session.get_inputs()[0].name
			self.datasize = len(nhwc_data_list)
			self.enum_data_dicts = iter([{input_name: nhwc_data} for nhwc_data in nhwc_data_list])
		return next(self.enum_data_dicts, None)
	
	@staticmethod
	def resize_image_format(srcimg , frame_resize):
		padh, padw, newh, neww = 0, 0, frame_resize[0], frame_resize[1]
		if srcimg.shape[0] != srcimg.shape[1]:
			hw_scale = srcimg.shape[0] / srcimg.shape[1]
			if hw_scale > 1:
				newh, neww = frame_resize[0], int(frame_resize[1] / hw_scale)
				img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_CUBIC)
				padw = int((frame_resize[1] - neww) * 0.5)
				img = cv2.copyMakeBorder(img, 0, 0, padw, frame_resize[1] - neww - padw, cv2.BORDER_CONSTANT,
										value=0)  # add border
			else:
				newh, neww = int(frame_resize[0] * hw_scale) + 1, frame_resize[1]
				img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_CUBIC)
				padh = int((frame_resize[0] - newh) * 0.5)
				img = cv2.copyMakeBorder(img, padh, frame_resize[0] - newh - padh, 0, 0, cv2.BORDER_CONSTANT, value=0)
		else:
			img = cv2.resize(srcimg, (frame_resize[1], frame_resize[0]), interpolation=cv2.INTER_CUBIC)
		ratioh, ratiow = srcimg.shape[0] / newh, srcimg.shape[1] / neww
		return img, newh, neww, ratioh, ratiow, padh, padw

	def proprocess_func(self, images_folder, input_shapes):
		batch_filenames = [ str(name) for name in Path(images_folder).iterdir()]
		unconcatenated_batch_data = []
		for image_filepath in batch_filenames:
			img = cv2.imread(image_filepath, cv2.IMREAD_COLOR)
			image, newh, neww, ratioh, ratiow, padh, padw = self.resize_image_format(img, input_shapes[-2:])
			input_data = np.array(image, dtype=np.float32)[np.newaxis, :, :]/255.0
			unconcatenated_batch_data.append(input_data.transpose(0,3,1,2))
		batch_data = np.concatenate(np.expand_dims(unconcatenated_batch_data, axis=0), axis=0)
		print(colorstr('bright_black', "Loading calibration sample count = %s"% str(batch_data.shape)))
		return batch_data

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
		x1y1 = points - lt
		x2y2 = points + rb
		wh = x2y2 - x1y1
		cxcy = (x2y2 + x1y1) / 2
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
			
			pred_output = pred_output.view(bs, na, -1) # (b, na, ny*nx)
			output.append(pred_output)

		cat_output = torch.cat(output, dim=2).permute(0, 2, 1).contiguous(), # (b, ny*nx, na=class_num+5) for NCNN
		return cat_output

def benchmark(model_path, runs=10):
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
	providers = []
	if  ort.get_device() == 'GPU' and 'CUDAExecutionProvider' in  ort.get_available_providers():  # gpu 
		providers.append('CUDAExecutionProvider')
	providers.append('CPUExecutionProvider')

	print(colorstr('bright_cyan', f"👉 Model Name [{model_path:s}] "))
	print(colorstr('bright_cyan', f"    Model Size {file_size(model_path):.1f} MB"))
	for provider in providers :
		session = ort.InferenceSession(model_path, providers=[provider])
		input_name = session.get_inputs()[0].name
		input_shapes = session.get_inputs()[0].shape
		input_types = np.float16 if 'float16' in session.get_inputs()[0].type else np.float32

		total = 0.0
		input_data = np.zeros(input_shapes, input_types)  # 随便输入一个假数据
		# warming up
		_ = session.run([], {input_name: input_data})
		for i in range(runs+1):
			start = time.perf_counter()
			_ = session.run([], {input_name: input_data})
			end = (time.perf_counter() - start) * 1000
			if (i>0) :
				total += end
		total /= runs
		print(colorstr('bright_cyan', f"    Device: {provider:s}, Avg Infer Times: {total:.2f}ms"))

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
	parser.add_argument(
		"--calib_image_dir",
		type=Path,
		help="The calibrate data required for conversion to int8, if None will use dynamic quantization",
	)
	parser.add_argument("--int8", action="store_true", help="Conver to int8")
	parser.add_argument("--half", action="store_true", help="Conver to fp16")
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

	if parse_arg.calib_image_dir is not None:
		parse_arg.calib_image_dir = parse_arg.calib_image_dir.resolve().absolute()

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


class OnnxBuilder:
	def __init__(self, model_type, checkpoint_path=None, class_names=[]):
		from super_gradients.training import models
		# Load PyTorch model
		print(colorstr("Loading model version [%s] with torch %s..." %(model_type, torch.__version__) ))
		if checkpoint_path is None:
			self.torch_model = models.get(model_type, pretrained_weights="coco")
			self.labels = self.torch_model._class_names  # get class names
		else :
			self.torch_model = models.get(model_type, checkpoint_path=str(checkpoint_path), num_classes=len(class_names))
			self.labels = class_names  # get class names
			
		self.labels = self.labels if isinstance(self.labels, list) else list(self.labels.values())
		# check num classes and labels
		assert self.torch_model.num_classes == len(self.labels), f"Model class count {self.model.num_classes} != len(names) {len(self.labels)}"
		# self.torch_model.predict("./models/demo.png", conf=0.5).show()

		self.replace_head = False

	def create_network(self, img_size, output_base):
		self.output_base = output_base
		# Replace with the custom Detection Head
		self.torch_model.heads = DetectNAS(self.torch_model.heads, img_size)
		num_branches = self.torch_model.heads.num_heads

		# Input
		self.img = torch.zeros(1, 3, *img_size)
		self.torch_model.eval()
		self.torch_model.prep_model_for_conversion(input_size=list(self.img.shape))
		self.replace_head = True

	def _export_fp16(self, onnx_model_fp32):
		# --------------------- Convert to float16 --------------------- 
		try:
			import onnxconverter_common
			print(colorstr(f'Starting to convert float16 with onnxconverter_common {onnxconverter_common.__version__}...'))

			onnx_model_fp16 = float16.convert_float_to_float16(onnx_model_fp32, op_block_list=["Softmax", "ReduceMax"])
			onnx.save(onnx_model_fp16, self.output_base+"_fp16.onnx")
		except ImportError:
			print(
				"onnxconverter_common is not found, if you want to quant the onnx, "
				+ "you should install it:\n\t"
				+ "pip install -U onnxconverter_common\n"
			)
		except Exception as e:
			print(colorstr('red', f'Half onnx export failure ❌ : {e}'))
			exit()

	def _export_int8(self, onnx_model_fp32, calibration_dataset_path=None):
		# --------------------- Convert to int8 --------------------- 
		try:
			print(colorstr(f'Starting to convert int8 with quantize_static...'))

			nodes = [n.name for n in onnx_model_fp32.graph.node]
			exclude_nodes = []
			for n in nodes :
				if re.findall("ReduceMax|Sofmax|Concat", n) :
					exclude_nodes.append(n)

			if calibration_dataset_path != None and calibration_dataset_path.is_dir():
				dr = DataReader(str(calibration_dataset_path), self.output_base+"_fp32.onnx")
				onnx_model_int8 = quantize_static(self.output_base+"_fp32.onnx",
													self.output_base+"_int8.onnx",
													dr,
													nodes_to_exclude=exclude_nodes,
													quant_format=QuantFormat.QDQ,
													activation_type=QuantType.QUInt8,
													weight_type=QuantType.QUInt8)
			else :
				print(colorstr('yellow', f"Calibration dataset path is not exist. can't use static quantization."))
				onnx_model_int8 = quantize_dynamic(self.output_base+"_fp32.onnx",
													self.output_base+"_int8.onnx",
													nodes_to_exclude=exclude_nodes,
													weight_type=QuantType.QUInt8)
		except Exception as e:
			print(colorstr('red', f'Int8 onnx export failure ❌ : {e}'))
			exit()

	def export_network(self, opset=12, half=False, int8=False, calibration_dataset_path=None, simplify=True):
		if (not self.replace_head) :
			print(colorstr('yellow', f"Please use 'create_network' before export!"))
			return 
		
		t = time.time()
		try:
			print(colorstr("Starting export with onnx %s..." % onnx.__version__))
			with BytesIO() as f:
				torch.onnx.export(
					self.torch_model,
					self.img,
					f,
					verbose=False,
					opset_version=opset,
					input_names=["images"],
					output_names=["outputs"],
				)

				# Checks
				onnx_model_fp32 = onnx.load_from_string(f.getvalue())  # load onnx model
				onnx.checker.check_model(onnx_model_fp32)  # check onnx model

			# Simplify
			if simplify :
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
					)
				except Exception as e:
					print(colorstr('red', f'Simplify onnx export failure ❌ : {e}'))

			onnx.save(onnx_model_fp32, self.output_base+"_fp32.onnx")
		except Exception as e:
			print(colorstr('red', f'Export failure ❌ : {e}'))
			exit()

		if half: self._export_fp16(onnx_model_fp32)
		if int8: self._export_int8(onnx_model_fp32, calibration_dataset_path)
		self.export_json_info()

		# Testing inference speed
		times = 10
		print(colorstr('bright_cyan', "*"*40))
		print(colorstr('bright_cyan', 'underline', f'❄️  Inference speed Testing ... ❄️'))
		print(colorstr('bright_cyan', f'Total Counts: {times}'))
		benchmark(self.output_base+"_fp32.onnx", times)
		if half or int8 :
			if half :
				benchmark(self.output_base+"_fp16.onnx", times)
				print(colorstr('bright_magenta', "ONNX export Float16 success ✅ , saved as:\n\t%s" % self.output_base+"_fp16.onnx"))
			if int8 :
				benchmark(self.output_base+"_int8.onnx", times)
				print(colorstr('bright_magenta', "ONNX export Int8 success ✅ , saved as:\n\t%s" % self.output_base+"_int8.onnx"))
			os.remove(self.output_base+"_fp32.onnx")
		else :
			print(colorstr('bright_magenta', "ONNX export Float32 success ✅ , saved as:\n\t%s" % self.output_base+"_fp32.onnx"))
		print(colorstr('bright_cyan', "*"*40))

		# Finish
		print(colorstr( "Export complete (%.2fs).\n" % (time.time() - t)))

	def export_json_info(self):
		print(colorstr('bright_magenta', "fpn_strides:\n\t%s" % str(self.torch_model.heads.fpn_strides)))
		print(colorstr('bright_magenta', "num_classes:\n\t%s" % self.torch_model.num_classes))
		_,  _, h, w = list(self.img.shape)
		export_json = Path(self.output_base + ".json")
		export_json.write_text(
			json.dumps(
				{
					"nn_config": {
						"output_format": "detection",
						"NN_family": "YOLO",
						"input_size": f"{h}x{w}",
						"NN_specific_metadata": {
							"classes": self.torch_model.num_classes,
							"coordinates": 4,
							"fpn_strides":  str(self.torch_model.heads.fpn_strides),
							"iou_threshold": 0.3,
							"confidence_threshold": 0.5,
						},
					},
					"mappings": {"labels": self.labels},
				},
				indent=4,
			)
		)
		print(colorstr('bright_magenta', "Anchors data export success, saved as:\n\t%s" % export_json))


if __name__ == "__main__":
	args = parse_args()
	print(colorstr(args))

	output_base_path = str(Path.joinpath(args.output_dir, args.input_model))

	builder = OnnxBuilder(args.input_model, args.checkpoint_path, args.class_names)
	builder.create_network(args.img_size, output_base_path)
	builder.export_network(args.opset, half=args.half, int8=args.int8, calibration_dataset_path=args.calib_image_dir, simplify=True)

