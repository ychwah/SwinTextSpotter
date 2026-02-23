#
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
SparseRCNN Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

import os
import itertools
import time
import copy
from typing import Any, Dict, List, Set

import torch
import torch.nn as nn
import torch.nn.functional as F

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import AutogradProfiler, DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results, TextEvaluator
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.structures import Instances, pairwise_iou
from detectron2.layers import nms

from swints import SWINTSDatasetMapper, add_SWINTS_config


class ProgressiveMultiScaleInference(nn.Module):
    def __init__(self, model, scales=None):
        super().__init__()
        self.model = model
        # Default scales if not provided
        self.scales = scales if scales is not None else [1.0, 1.25, 1.5]

    def forward(self, batched_inputs):
        if self.training:
            return self.model(batched_inputs)

        all_results = []
        for input_dict in batched_inputs:
            image = input_dict["image"]
            c, h, w = image.shape
            max_dim = max(h, w)

            # Use provided scales, but apply adaptive filtering for speed
            curr_scales = []
            for s in self.scales:
                # Heuristic to skip large scales for already large images
                if s > 1.0 and max_dim > 1500:
                    continue
                if s > 1.25 and max_dim > 1000:
                    continue
                curr_scales.append(s)

            if not curr_scales:
                curr_scales = [1.0]

            multi_scale_instances = []
            # 1. First pass (Base Scale)
            with torch.no_grad():
                output = self.model([input_dict])[0]
                base_instances = output["instances"]
                multi_scale_instances.append(base_instances)

            # 2. Progressive check: run additional scales only if needed
            # e.g., if few detections found or image is small
            needs_more = len(base_instances) < 5 or max_dim < 1000

            if needs_more:
                additional_inputs = []
                for scale in curr_scales:
                    if scale == 1.0:
                        continue

                    # Cap maximum resolution to prevent extreme slowness/OOM
                    max_res = 2240
                    new_h, new_w = int(h * scale), int(w * scale)
                    if max(new_h, new_w) > max_res:
                        scale_factor = max_res / max(new_h, new_w)
                        new_h, new_w = int(new_h * scale_factor), int(new_w * scale_factor)

                    curr_image = F.interpolate(
                        image.unsqueeze(0).float(),
                        size=(new_h, new_w),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0).to(image.dtype)

                    curr_input = copy.copy(input_dict)
                    curr_input["image"] = curr_image
                    additional_inputs.append(curr_input)

                if additional_inputs:
                    with torch.no_grad():
                        # Run additional scales in parallel via batching
                        additional_outputs = self.model(additional_inputs)
                        for out in additional_outputs:
                            multi_scale_instances.append(out["instances"])

            # Concatenate all instances for this image
            merged_instances = Instances.cat(multi_scale_instances)

            # Apply NMS and simple score boosting for cross-scale consistency
            if len(merged_instances) > 0:
                # 0. Filter out boxes with zero or near-zero area to prevent evaluation issues
                areas = merged_instances.pred_boxes.area()
                valid_area = areas > 0.1
                merged_instances = merged_instances[valid_area]

            if len(merged_instances) > 0:
                # 3. Simple score boost for boxes detected at multiple scales
                if len(merged_instances) > 1:
                    # pairwise_iou expects Boxes objects
                    ious = pairwise_iou(merged_instances.pred_boxes, merged_instances.pred_boxes)
                    # For each box, find how many other boxes overlap significantly (> 0.8 IoU)
                    num_overlaps = (ious > 0.8).sum(dim=1).float()
                    # Boost score if detected in multiple scales (max 0.1 boost)
                    boost = torch.clamp((num_overlaps - 1) * 0.05, max=0.1)

                    if merged_instances.scores.dim() > 1:
                        # multi-class scores
                        merged_instances.scores = merged_instances.scores + boost.unsqueeze(1)
                    else:
                        merged_instances.scores = merged_instances.scores + boost
                    merged_instances.scores = torch.clamp(merged_instances.scores, max=1.0)

                # 4. Use box NMS with slightly higher threshold to keep candidates
                scores = merged_instances.scores
                if scores.dim() > 1:
                    max_scores, _ = scores.max(dim=1)
                else:
                    max_scores = scores

                keep = nms(
                    merged_instances.pred_boxes.tensor,
                    max_scores,
                    iou_threshold=0.6 # Slightly tighter NMS to improve speed
                )
                # Move keep to cpu to avoid device mismatch with CPU-based fields like pred_rec
                merged_instances = merged_instances[keep.to("cpu")]

            all_results.append({"instances": merged_instances})

        return all_results


class Trainer(DefaultTrainer):
#     """
#     Extension of the Trainer class adapted to SparseRCNN.
#     """

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        # Wrap the model for multi-scale inference during evaluation if TTA is enabled
        if cfg.TEST.AUG.ENABLED and not isinstance(model, ProgressiveMultiScaleInference):
            scales = cfg.TEST.AUG.get("SCALES", [1.0, 1.25, 1.5])
            model = ProgressiveMultiScaleInference(model, scales=scales)
        return super().test(cfg, model, evaluators)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return TextEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = SWINTSDatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_optimizer(cls, cfg, model):
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for key, value in model.named_parameters(recurse=True):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "backbone" in key:
                lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_SWINTS_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)

        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
