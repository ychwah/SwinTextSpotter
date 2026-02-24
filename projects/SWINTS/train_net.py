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
import logging
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


logger = logging.getLogger(__name__)


def _average_inference_time(times):
    """Return average inference time in seconds for a list of timings."""
    return sum(times) / len(times) if times else 0.0


def _safe_ratio(numerator, denominator):
    """Return percentage ratio in [0, 100] with safe zero-denominator handling."""
    return (100.0 * numerator / denominator) if denominator else 0.0


def _rotate_image_180(image):
    """Rotate CHW tensor by 180 degrees."""
    return torch.rot90(image, k=2, dims=(1, 2))


def _map_boxes_from_rot180(pred_boxes, height, width):
    """Map XYXY boxes from 180-rotated image coordinates back to original image."""
    boxes = pred_boxes.tensor.clone()
    x1 = width - boxes[:, 2]
    y1 = height - boxes[:, 3]
    x2 = width - boxes[:, 0]
    y2 = height - boxes[:, 1]
    boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3] = x1, y1, x2, y2
    pred_boxes.tensor = boxes


class InferenceTimingWrapper(nn.Module):
    """Simple eval-time wrapper to report baseline average per-image inference time."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batched_inputs):
        if self.training:
            return self.model(batched_inputs)

        all_results = []
        timings = []
        total_images = 0
        triggered_images = 0
        multiscale_images = 0
        total_extra_scales = 0

        for input_dict in batched_inputs:
            total_images += 1
            start = time.perf_counter()
            with torch.no_grad():
                output = self.model([input_dict])[0]
            timings.append(time.perf_counter() - start)
            all_results.append(output)

        if comm.is_main_process() and timings:
            avg_ms = _average_inference_time(timings) * 1000.0
            logger.info(f"[Baseline] Average per-image inference time: {avg_ms:.2f} ms")

        return all_results


class ProgressiveMultiScaleInference(nn.Module):
    def __init__(self, model, scales=None, cfg=None):
        super().__init__()
        self.model = model
        # Default scales if not provided
        self.scales = scales if scales is not None else [1.0, 1.2, 1.35]

        # Tunable PMSI heuristics for cross-dataset balancing.
        pmsi_cfg = getattr(getattr(cfg, "TEST", None), "PMSI", None) if cfg is not None else None
        self.max_resolution = getattr(pmsi_cfg, "MAX_RESOLUTION", 1920)
        self.rotations = list(getattr(pmsi_cfg, "ROTATIONS", [0]))
        self.skip_large_gt = getattr(pmsi_cfg, "SKIP_LARGE_GT", 1400)
        self.skip_very_large_gt = getattr(pmsi_cfg, "SKIP_VERY_LARGE_GT", 1000)
        self.min_confident_count = getattr(pmsi_cfg, "MIN_CONFIDENT_COUNT", 6)
        self.conf_thresh = getattr(pmsi_cfg, "CONF_THRESH", 0.45)
        self.small_image_trigger = getattr(pmsi_cfg, "SMALL_IMAGE_TRIGGER", 1100)
        self.cross_iou_thresh = getattr(pmsi_cfg, "CROSS_IOU_THRESH", 0.85)
        self.boost_per_match = getattr(pmsi_cfg, "BOOST_PER_MATCH", 0.02)
        self.max_boost = getattr(pmsi_cfg, "MAX_BOOST", 0.06)
        self.enable_score_boost = getattr(pmsi_cfg, "ENABLE_SCORE_BOOST", True)
        self.merge_nms_thresh = getattr(pmsi_cfg, "MERGE_NMS_THRESH", 0.6)

    def forward(self, batched_inputs):
        if self.training:
            return self.model(batched_inputs)

        all_results = []
        timings = []
        total_images = 0
        triggered_images = 0
        multiscale_images = 0
        total_extra_scales = 0

        for input_dict in batched_inputs:
            total_images += 1
            image = input_dict["image"]
            c, h, w = image.shape
            max_dim = max(h, w)

            start = time.perf_counter()

            # Use provided scales, but apply adaptive filtering for speed
            curr_scales = []
            for s in self.scales:
                # Heuristic to skip large scales for already large images
                if s > 1.0 and max_dim > self.skip_large_gt:
                    continue
                if s > 1.25 and max_dim > self.skip_very_large_gt:
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
            # For SWINTS, base_instances always has TEST_NUM_PROPOSALS (usually 100).
            # We run more scales if the model is not confident or the image is small.
            scores = base_instances.scores
            if scores.dim() > 1:
                scores = scores.max(dim=1)[0]

            num_confident = (scores > self.conf_thresh).sum().item()
            # Trigger extra scales if predictions are weak or image long-side is small.
            needs_more = num_confident < self.min_confident_count or max_dim < self.small_image_trigger

            if needs_more:
                triggered_images += 1
                additional_inputs = []
                aug_meta = []
                for scale in curr_scales:
                    for rotation in self.rotations:
                        if scale == 1.0 and rotation == 0:
                            continue
                        if rotation not in (0, 180):
                            continue

                        # Cap maximum resolution to prevent extreme slowness/OOM
                        max_res = self.max_resolution
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

                        if rotation == 180:
                            curr_image = _rotate_image_180(curr_image)

                        curr_input = copy.copy(input_dict)
                        curr_input["image"] = curr_image
                        additional_inputs.append(curr_input)
                        aug_meta.append((new_h, new_w, rotation))

                total_extra_scales += len(additional_inputs)
                if additional_inputs:
                    with torch.no_grad():
                        # Run additional augmentations in parallel via batching
                        additional_outputs = self.model(additional_inputs)
                        for out, (ah, aw, rotation) in zip(additional_outputs, aug_meta):
                            inst = out["instances"]
                            if rotation == 180 and len(inst) > 0:
                                _map_boxes_from_rot180(inst.pred_boxes, ah, aw)
                            multi_scale_instances.append(inst)

            # 3. Merge results
            if len(multi_scale_instances) > 1:
                multiscale_images += 1

            if len(multi_scale_instances) == 1:
                # If only one scale, return as is (baseline behavior)
                merged_instances = multi_scale_instances[0]
            else:
                # Multiple scales: apply merging and cross-scale boosting
                # Add scale_id to each instance to avoid intra-scale boosting
                for idx, inst in enumerate(multi_scale_instances):
                    inst.scale_id = torch.full((len(inst),), idx, device=inst.scores.device)

                merged_instances = Instances.cat(multi_scale_instances)

                if len(merged_instances) > 0:
                    # Filter out boxes with zero or near-zero area
                    areas = merged_instances.pred_boxes.area()
                    merged_instances = merged_instances[areas > 0.1]

                if len(merged_instances) > 0:
                    if self.enable_score_boost:
                        # Cross-scale score boost
                        ious = pairwise_iou(merged_instances.pred_boxes, merged_instances.pred_boxes)
                        scale_ids = merged_instances.scale_id
                        # Different scales mask: True where scale_ids are different
                        diff_scales = scale_ids.unsqueeze(0) != scale_ids.unsqueeze(1)

                        # Overlap from different scales
                        cross_overlaps = (ious > self.cross_iou_thresh) & diff_scales
                        num_cross_overlaps = cross_overlaps.sum(dim=1).float()

                        # Boost score (max 0.15 boost for multi-scale consistency)
                        boost = torch.clamp(num_cross_overlaps * self.boost_per_match, max=self.max_boost)

                        if merged_instances.scores.dim() > 1:
                            merged_instances.scores = merged_instances.scores + boost.unsqueeze(1)
                        else:
                            merged_instances.scores = merged_instances.scores + boost
                        merged_instances.scores = torch.clamp(merged_instances.scores, max=1.0)

                    # NMS to remove redundancies across scales
                    scores = merged_instances.scores
                    max_scores = scores.max(dim=1)[0] if scores.dim() > 1 else scores

                    # Keep a bit more to let TextEvaluator's polygon NMS do the final work
                    keep = nms(merged_instances.pred_boxes.tensor, max_scores, iou_threshold=self.merge_nms_thresh)
                    merged_instances = merged_instances[keep.to("cpu")]

            # Ensure all tensors are on CPU for the evaluator
            merged_instances = merged_instances.to(torch.device("cpu"))
            all_results.append({"instances": merged_instances})
            timings.append(time.perf_counter() - start)

        if comm.is_main_process() and timings:
            avg_ms = _average_inference_time(timings) * 1000.0
            logger.info(f"[PMSI] Average per-image inference time: {avg_ms:.2f} ms")

            # Scale Activation Analysis: quantify how often PMSI trigger/scales are activated.
            trigger_rate = _safe_ratio(triggered_images, total_images)
            multiscale_rate = _safe_ratio(multiscale_images, total_images)
            avg_extra_scales = (total_extra_scales / total_images) if total_images else 0.0
            logger.info(
                "[PMSI] Scale Activation Analysis | "
                f"trigger_rate={trigger_rate:.2f}% ({triggered_images}/{total_images}), "
                f"multiscale_rate={multiscale_rate:.2f}% ({multiscale_images}/{total_images}), "
                f"avg_extra_scales_per_image={avg_extra_scales:.3f}"
            )

        return all_results


class Trainer(DefaultTrainer):
#     """
#     Extension of the Trainer class adapted to SparseRCNN.
#     """

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        # Wrap with explicit timing helpers for transparent baseline vs PMSI speed comparison.
        # Baseline/pipeline separation is still preserved via TEST.PMSI.ENABLED.
        if cfg.TEST.PMSI.ENABLED and not isinstance(model, ProgressiveMultiScaleInference):
            scales = cfg.TEST.PMSI.SCALES
            model = ProgressiveMultiScaleInference(model, scales=scales, cfg=cfg)
        elif not cfg.TEST.PMSI.ENABLED and not isinstance(model, InferenceTimingWrapper):
            model = InferenceTimingWrapper(model)
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
