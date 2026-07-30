#!/usr/bin/env python3
"""
AP-Bench Experiment Runner
==========================
Runs all experiments described in the ECCV paper:
  - AP-Difficulty computation (Eq. 1-5)
  - Difficulty-stratified evaluation (Tables 2, 3)
  - Active learning experiments (Tables 4, 5)
  - Weight validation (Table 6)
  - Model scaling analysis (Table 7)
  - Cross-domain generalisation (Table 9)

Requirements:
  pip install ultralytics pycocotools numpy scipy pandas pyyaml
  pip install open_clip_torch torch torchvision scikit-learn

Usage:
  python run_experiments.py \
      --dataset_root /path/to/ap_bench \
      --output_dir /path/to/results \
      [--skip_training] [--skip_clip] [--gpu 0]
"""

import os
import sys
import json
import math
import glob
import shutil
import random
import logging
import argparse
import time
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
import pandas as pd
from scipy import stats

# ============================================================================
# Configuration
# ============================================================================

RANDOM_SEED = 42
IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

# AP-Difficulty constants (from paper Sec. 3)
A_REF = 0.04          # Reference area fraction (Eq. 1)
N_MAX = 8             # Max neighbour normalisation (Eq. 3)
ALPHA = 0.40          # Size weight (Eq. 4)
BETA  = 0.35          # Occlusion weight (Eq. 4)
GAMMA = 0.25          # Density weight (Eq. 4)
TAU_1 = 0.30          # Easy/Medium boundary (Eq. 5)
TAU_2 = 0.58          # Medium/Hard boundary (Eq. 5)

# Training hyperparameters (from paper Sec. 5.1)
TRAIN_CONFIG = dict(
    optimizer='AdamW',
    lr0=1e-3,
    lrf=1e-3,      # min LR ratio for cosine schedule → effectively 1e-6
    cos_lr=True,
    batch=8,
    imgsz=640,
    epochs=150,
    patience=15,
    hsv_h=0.015,   # photometric: brightness ±30%, contrast ±20%
    hsv_s=0.3,
    hsv_v=0.3,
    fliplr=0.5,
    mosaic=0.5,
    erasing=0.0,
    scale=0.5,
    translate=0.1,
    workers=4,
    exist_ok=True,
    cache=True,
)

# Active learning sample counts
AL_SAMPLE_COUNTS = [100, 200, 300, 400]

# Model variants for scaling experiment
MODEL_VARIANTS = {
    's': 'yolo11s-seg.pt',
    'm': 'yolo11m-seg.pt',
    'l': 'yolo11l-seg.pt',
}

logger = logging.getLogger('ap_bench')


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class BBox:
    """Bounding box in YOLO normalised format: x_center, y_center, width, height."""
    x_center: float
    y_center: float
    width: float
    height: float
    class_id: int = 0

    @property
    def x1(self): return self.x_center - self.width / 2
    @property
    def y1(self): return self.y_center - self.height / 2
    @property
    def x2(self): return self.x_center + self.width / 2
    @property
    def y2(self): return self.y_center + self.height / 2
    @property
    def area(self): return self.width * self.height
    @property
    def diagonal(self): return math.sqrt(self.width**2 + self.height**2)


@dataclass
class Instance:
    """A single annotated instance with difficulty metadata."""
    bbox: BBox
    image_id: str
    instance_id: int
    # Segmentation points (YOLO polygon format) — empty for box-only
    seg_points: List[float] = field(default_factory=list)
    # Difficulty factors (computed later)
    size_factor: float = 0.0
    occlusion_factor: float = 0.0
    density_factor: float = 0.0
    difficulty_score: float = 0.0
    difficulty_tier: str = 'Easy'


@dataclass
class ImageAnnotation:
    """All annotations for a single image."""
    image_path: str
    image_id: str
    width: int = 0
    height: int = 0
    instances: List[Instance] = field(default_factory=list)
    split: str = 'train'  # train/val/test


# ============================================================================
# Phase 0: AP-Difficulty Computation
# ============================================================================

def compute_iou(b1: BBox, b2: BBox) -> float:
    """Compute IoU between two normalised bounding boxes."""
    x1 = max(b1.x1, b2.x1)
    y1 = max(b1.y1, b2.y1)
    x2 = min(b1.x2, b2.x2)
    y2 = min(b1.y2, b2.y2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = b1.area + b2.area - inter
    return inter / union if union > 0 else 0.0


def compute_center_distance(b1: BBox, b2: BBox) -> float:
    """Euclidean distance between bbox centres (normalised coords)."""
    return math.sqrt((b1.x_center - b2.x_center)**2 + (b1.y_center - b2.y_center)**2)


def compute_difficulty_factors(instances: List[Instance],
                                img_width: int, img_height: int,
                                alpha: float = ALPHA,
                                beta: float = BETA,
                                gamma: float = GAMMA) -> None:
    """Compute AP-Difficulty factors for all instances in an image (Eq. 1-4).
    Modifies instances in place."""
    n = len(instances)
    bboxes = [inst.bbox for inst in instances]

    for i, inst in enumerate(instances):
        bi = bboxes[i]

        # Size factor (Eq. 1) — using normalised coords, so area is already relative
        # The paper uses pixel area / (A_ref * W * H), but with normalised boxes:
        # pixel_area = (w_norm * W) * (h_norm * H) = w_norm * h_norm * W * H
        # S_i = 1 - min(w_norm * h_norm * W * H / (A_ref * W * H), 1)
        #     = 1 - min(w_norm * h_norm / A_ref, 1)
        inst.size_factor = 1.0 - min(bi.area / A_REF, 1.0)

        # Occlusion factor (Eq. 2) — max pairwise IoU
        max_iou = 0.0
        for j in range(n):
            if i != j:
                max_iou = max(max_iou, compute_iou(bi, bboxes[j]))
        inst.occlusion_factor = max_iou

        # Density factor (Eq. 3) — count neighbours within 2 * diagonal
        search_radius = 2.0 * bi.diagonal
        neighbour_count = sum(
            1 for j in range(n) if i != j and
            compute_center_distance(bi, bboxes[j]) < search_radius
        )
        inst.density_factor = min(neighbour_count / N_MAX, 1.0)

        # Combined difficulty (Eq. 4)
        inst.difficulty_score = (
            alpha * inst.size_factor +
            beta * inst.occlusion_factor +
            gamma * inst.density_factor
        )


def assign_difficulty_tiers(instances: List[Instance],
                             tau1: float = TAU_1,
                             tau2: float = TAU_2) -> None:
    """Assign Easy/Medium/Hard tiers (Eq. 5). Modifies in place."""
    for inst in instances:
        if inst.difficulty_score < tau1:
            inst.difficulty_tier = 'Easy'
        elif inst.difficulty_score < tau2:
            inst.difficulty_tier = 'Medium'
        else:
            inst.difficulty_tier = 'Hard'


def compute_difficulty_with_weights(instances: List[Instance],
                                     img_width: int, img_height: int,
                                     alpha: float, beta: float, gamma: float,
                                     tau1: float = TAU_1, tau2: float = TAU_2):
    """Recompute difficulty under alternative weights (for Table 6 validation)."""
    compute_difficulty_factors(instances, img_width, img_height, alpha, beta, gamma)
    assign_difficulty_tiers(instances, tau1, tau2)


# ============================================================================
# Phase 0b: Dataset I/O (YOLO format)
# ============================================================================

def load_yolo_dataset(dataset_root: str) -> Tuple[Dict[str, List[ImageAnnotation]], List[str]]:
    """Load a YOLO-format dataset.

    Expected structure:
        dataset_root/
            data.yaml
            train/images/  train/labels/
            valid/images/  valid/labels/  (or val/)
            test/images/   test/labels/

    Returns:
        splits: dict mapping split name → list of ImageAnnotation
        class_names: list of class name strings
    """
    yaml_path = None
    for name in ['data.yaml', 'dataset.yaml']:
        p = os.path.join(dataset_root, name)
        if os.path.exists(p):
            yaml_path = p
            break
    if yaml_path is None:
        raise FileNotFoundError(f"No data.yaml found in {dataset_root}")

    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    class_names = config.get('names', [])
    if isinstance(class_names, dict):
        class_names = [class_names[k] for k in sorted(class_names.keys())]

    split_map = {'train': 'train', 'val': 'val', 'valid': 'val', 'test': 'test'}
    splits = {}

    for key, split_name in split_map.items():
        if key not in config:
            continue
        img_dir = config[key]
        if not os.path.isabs(img_dir):
            img_dir = os.path.join(dataset_root, img_dir)
        if not os.path.isdir(img_dir):
            continue

        label_dir = img_dir.replace('/images', '/labels')
        annotations = []

        for img_file in sorted(os.listdir(img_dir)):
            ext = os.path.splitext(img_file)[1].lower()
            if ext not in IMG_EXTENSIONS:
                continue

            img_path = os.path.join(img_dir, img_file)
            img_id = os.path.splitext(img_file)[0]
            label_path = os.path.join(label_dir, img_id + '.txt')

            ann = ImageAnnotation(
                image_path=img_path,
                image_id=img_id,
                split=split_name,
            )

            if os.path.exists(label_path):
                with open(label_path) as f:
                    for idx, line in enumerate(f):
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        cls_id = int(parts[0])
                        coords = [float(x) for x in parts[1:]]

                        if len(coords) == 4:
                            # Detection: x_c, y_c, w, h
                            bbox = BBox(*coords, class_id=cls_id)
                            inst = Instance(bbox=bbox, image_id=img_id,
                                            instance_id=idx)
                        elif len(coords) >= 6 and len(coords) % 2 == 0:
                            # Segmentation polygon → derive bbox
                            xs = coords[0::2]
                            ys = coords[1::2]
                            x_min, x_max = min(xs), max(xs)
                            y_min, y_max = min(ys), max(ys)
                            w = x_max - x_min
                            h = y_max - y_min
                            xc = x_min + w / 2
                            yc = y_min + h / 2
                            bbox = BBox(xc, yc, w, h, class_id=cls_id)
                            inst = Instance(bbox=bbox, image_id=img_id,
                                            instance_id=idx, seg_points=coords)
                        else:
                            continue

                        ann.instances.append(inst)
            annotations.append(ann)

        splits[split_name] = annotations
        logger.info(f"Loaded split '{split_name}': {len(annotations)} images, "
                     f"{sum(len(a.instances) for a in annotations)} instances")

    return splits, class_names


def get_image_dimensions(img_path: str) -> Tuple[int, int]:
    """Get image width, height without loading full image."""
    try:
        from PIL import Image
        with Image.open(img_path) as img:
            return img.size  # (width, height)
    except Exception:
        import cv2
        img = cv2.imread(img_path)
        if img is not None:
            h, w = img.shape[:2]
            return w, h
        return 640, 640  # fallback


def populate_image_dimensions(annotations: List[ImageAnnotation]) -> None:
    """Fill in width/height for all images."""
    for ann in annotations:
        if ann.width == 0 or ann.height == 0:
            ann.width, ann.height = get_image_dimensions(ann.image_path)


# ============================================================================
# Phase 0c: CLIP Diversity Sampling
# ============================================================================

def compute_clip_embeddings(image_paths: List[str], device: str = 'cuda') -> np.ndarray:
    """Compute CLIP ViT-B/32 embeddings for a list of images.

    Returns: (N, D) numpy array of L2-normalised embeddings.
    """
    import torch
    import open_clip
    from PIL import Image

    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='openai'
    )
    model = model.to(device).eval()

    embeddings = []
    batch_size = 32

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        images = []
        for p in batch_paths:
            try:
                img = preprocess(Image.open(p).convert('RGB'))
                images.append(img)
            except Exception as e:
                logger.warning(f"Failed to load {p}: {e}")
                images.append(torch.zeros(3, 224, 224))

        batch = torch.stack(images).to(device)
        with torch.no_grad():
            feats = model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        embeddings.append(feats.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    logger.info(f"Computed CLIP embeddings: {embeddings.shape}")
    return embeddings


def clip_uniqueness_ranking(embeddings: np.ndarray, k: int = 5) -> np.ndarray:
    """Compute CLIP uniqueness score = mean k-NN L2 distance.

    Returns: array of uniqueness scores (higher = more unique).
    """
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k+1, metric='euclidean')
    nn.fit(embeddings)
    distances, _ = nn.kneighbors(embeddings)
    # Skip self (distance=0), take mean of k nearest
    uniqueness = distances[:, 1:k+1].mean(axis=1)
    return uniqueness


def select_clip_subsets(annotations: List[ImageAnnotation],
                        embeddings: np.ndarray,
                        sample_counts: List[int]) -> Dict[int, List[int]]:
    """Select CLIP-diversity subsets at various sizes.

    Returns: dict mapping sample_count → list of indices into annotations.
    """
    uniqueness = clip_uniqueness_ranking(embeddings)
    # Sort by uniqueness descending (most unique first)
    ranked_indices = np.argsort(-uniqueness)

    subsets = {}
    for count in sample_counts:
        subsets[count] = ranked_indices[:count].tolist()
        logger.info(f"CLIP subset {count}: uniqueness range "
                     f"[{uniqueness[ranked_indices[count-1]]:.4f}, "
                     f"{uniqueness[ranked_indices[0]]:.4f}]")
    # Full dataset
    subsets[len(annotations)] = list(range(len(annotations)))
    return subsets


def select_random_subsets(n_total: int,
                          sample_counts: List[int],
                          seed: int = RANDOM_SEED) -> Dict[int, List[int]]:
    """Select random subsets at various sizes.

    Returns: dict mapping sample_count → list of indices.
    """
    rng = random.Random(seed)
    all_indices = list(range(n_total))
    rng.shuffle(all_indices)

    subsets = {}
    for count in sample_counts:
        subsets[count] = sorted(all_indices[:count])
    subsets[n_total] = list(range(n_total))
    return subsets


# ============================================================================
# Phase 1: Dataset Subset Creation
# ============================================================================

def create_yolo_subset(annotations: List[ImageAnnotation],
                       indices: List[int],
                       output_dir: str,
                       class_names: List[str],
                       box_only: bool = False,
                       val_annotations: Optional[List[ImageAnnotation]] = None,
                       test_annotations: Optional[List[ImageAnnotation]] = None) -> str:
    """Create a YOLO-format dataset directory from a subset of annotations.

    Args:
        annotations: source training annotations
        indices: which training annotations to include
        output_dir: where to write the dataset
        class_names: list of class names
        box_only: if True, convert segmentation polygons to bounding boxes
        val_annotations: validation set (always included in full)
        test_annotations: test set (always included in full)

    Returns: path to data.yaml
    """
    os.makedirs(output_dir, exist_ok=True)

    split_data = {'train': [annotations[i] for i in indices]}
    if val_annotations:
        split_data['val'] = val_annotations
    if test_annotations:
        split_data['test'] = test_annotations

    split_dirs = {'train': 'train', 'val': 'valid', 'test': 'test'}

    for split_name, anns in split_data.items():
        img_dir = os.path.join(output_dir, split_dirs[split_name], 'images')
        lbl_dir = os.path.join(output_dir, split_dirs[split_name], 'labels')
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        for ann in anns:
            # Symlink image
            dst_img = os.path.join(img_dir, os.path.basename(ann.image_path))
            if not os.path.exists(dst_img):
                os.symlink(os.path.abspath(ann.image_path), dst_img)

            # Write label
            lbl_file = os.path.join(lbl_dir,
                                     os.path.splitext(os.path.basename(ann.image_path))[0] + '.txt')
            with open(lbl_file, 'w') as f:
                for inst in ann.instances:
                    if box_only or not inst.seg_points:
                        # YOLO detection format: class x_c y_c w h
                        f.write(f"{inst.bbox.class_id} "
                                f"{inst.bbox.x_center:.6f} {inst.bbox.y_center:.6f} "
                                f"{inst.bbox.width:.6f} {inst.bbox.height:.6f}\n")
                    else:
                        # YOLO segmentation format: class x1 y1 x2 y2 ...
                        pts_str = ' '.join(f"{v:.6f}" for v in inst.seg_points)
                        f.write(f"{inst.bbox.class_id} {pts_str}\n")

    # Write data.yaml
    yaml_path = os.path.join(output_dir, 'data.yaml')
    yaml_data = {
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': len(class_names),
        'names': class_names,
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, sort_keys=False)

    return yaml_path


# ============================================================================
# Phase 2: Training
# ============================================================================

def train_yolo_model(data_yaml: str,
                     model_name: str,
                     run_name: str,
                     output_base: str,
                     copy_paste: float = 0.0,
                     extra_config: Optional[dict] = None) -> str:
    """Train a YOLO model and return path to best weights.

    Args:
        data_yaml: path to data.yaml
        model_name: e.g. 'yolo11s-seg.pt'
        run_name: unique name for this run
        output_base: base directory for runs
        copy_paste: Copy-Paste augmentation probability
        extra_config: override training config

    Returns: path to best.pt weights
    """
    from ultralytics import YOLO

    config = {**TRAIN_CONFIG}
    if extra_config:
        config.update(extra_config)

    config['copy_paste'] = copy_paste
    config['copy_paste_mode'] = 'mixup'

    model = YOLO(model_name)

    logger.info(f"Training {run_name}: model={model_name}, data={data_yaml}")
    logger.info(f"  Config: {config}")

    model.train(
        data=data_yaml,
        task='segment',
        name=run_name,
        project=output_base,
        **config,
    )

    weights_path = os.path.join(output_base, run_name, 'weights', 'best.pt')
    if not os.path.exists(weights_path):
        # Fallback to last.pt
        weights_path = os.path.join(output_base, run_name, 'weights', 'last.pt')

    logger.info(f"Training complete: {weights_path}")
    return weights_path


# ============================================================================
# Phase 3: Difficulty-Stratified Evaluation
# ============================================================================

def run_yolo_val(weights_path: str,
                 data_yaml: str,
                 split: str = 'test',
                 imgsz: int = 640) -> dict:
    """Run YOLO validation and return raw results dict."""
    from ultralytics import YOLO

    model = YOLO(weights_path, task='segment')
    results = model.val(
        data=data_yaml,
        split=split,
        imgsz=imgsz,
        batch=1,
        verbose=False,
        save_json=True,  # saves COCO-format predictions
    )

    return results


def load_coco_annotations_from_yolo(annotations: List[ImageAnnotation],
                                     class_names: List[str]) -> dict:
    """Convert our annotations to COCO JSON format for pycocotools eval."""
    coco_data = {
        'images': [],
        'annotations': [],
        'categories': [{'id': i, 'name': name} for i, name in enumerate(class_names)]
    }
    ann_id = 1

    for img_idx, ann in enumerate(annotations):
        img_id = img_idx + 1
        coco_data['images'].append({
            'id': img_id,
            'file_name': os.path.basename(ann.image_path),
            'width': ann.width,
            'height': ann.height,
        })

        for inst in ann.instances:
            # Convert normalised to pixel coords
            x = (inst.bbox.x_center - inst.bbox.width / 2) * ann.width
            y = (inst.bbox.y_center - inst.bbox.height / 2) * ann.height
            w = inst.bbox.width * ann.width
            h = inst.bbox.height * ann.height

            coco_ann = {
                'id': ann_id,
                'image_id': img_id,
                'category_id': inst.bbox.class_id,
                'bbox': [x, y, w, h],
                'area': w * h,
                'iscrowd': 0,
            }

            # Add segmentation if available
            if inst.seg_points:
                # Convert normalised polygon to pixel coords
                seg_px = []
                for j in range(0, len(inst.seg_points), 2):
                    seg_px.append(inst.seg_points[j] * ann.width)
                    seg_px.append(inst.seg_points[j+1] * ann.height)
                coco_ann['segmentation'] = [seg_px]
            else:
                # Box as polygon
                coco_ann['segmentation'] = [[x, y, x+w, y, x+w, y+h, x, y+h]]

            coco_data['annotations'].append(coco_ann)
            ann_id += 1

    return coco_data


def stratified_evaluation(predictions_json: str,
                          test_annotations: List[ImageAnnotation],
                          class_names: List[str],
                          alpha: float = ALPHA,
                          beta: float = BETA,
                          gamma: float = GAMMA,
                          tau1: float = TAU_1,
                          tau2: float = TAU_2) -> dict:
    """Run COCO evaluation stratified by AP-Difficulty tiers.

    Args:
        predictions_json: path to COCO-format predictions JSON
        test_annotations: ground truth with difficulty scores computed
        class_names: class name list

    Returns: dict with keys 'aggregate', 'Easy', 'Medium', 'Hard', each containing metrics
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import tempfile

    # Build GT COCO annotation
    gt_coco_data = load_coco_annotations_from_yolo(test_annotations, class_names)

    # Recompute difficulty if needed and store tier per annotation
    all_instances_flat = []
    ann_id_to_tier = {}
    ann_counter = 1
    for ann in test_annotations:
        if ann.instances:
            compute_difficulty_factors(ann.instances, ann.width, ann.height, alpha, beta, gamma)
            assign_difficulty_tiers(ann.instances, tau1, tau2)
        for inst in ann.instances:
            ann_id_to_tier[ann_counter] = inst.difficulty_tier
            all_instances_flat.append(inst)
            ann_counter += 1

    # Save GT to temp file
    gt_tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(gt_coco_data, gt_tmp)
    gt_tmp.close()

    coco_gt = COCO(gt_tmp.name)

    # Load predictions
    coco_dt = coco_gt.loadRes(predictions_json)

    results = {}

    # Aggregate evaluation
    for tier_name in ['aggregate', 'Easy', 'Medium', 'Hard']:
        if tier_name == 'aggregate':
            eval_ann_ids = list(coco_gt.getAnnIds())
        else:
            eval_ann_ids = [aid for aid, tier in ann_id_to_tier.items()
                           if tier == tier_name]

        if not eval_ann_ids:
            logger.warning(f"No annotations for tier '{tier_name}'")
            results[tier_name] = {'mAP50': 0, 'mAP75': 0, 'mAP50_95': 0, 'recall': 0}
            continue

        # Filter GT to only this tier's annotations
        # We create a subset COCO object
        tier_gt_data = {
            'images': gt_coco_data['images'],
            'categories': gt_coco_data['categories'],
            'annotations': [a for a in gt_coco_data['annotations'] if a['id'] in set(eval_ann_ids)]
        }

        tier_gt_tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(tier_gt_data, tier_gt_tmp)
        tier_gt_tmp.close()

        tier_coco_gt = COCO(tier_gt_tmp.name)
        tier_coco_dt = tier_coco_gt.loadRes(predictions_json)

        # Segmentation eval
        coco_eval = COCOeval(tier_coco_gt, tier_coco_dt, 'segm')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        results[tier_name] = {
            'mAP50_95': float(coco_eval.stats[0]),    # mAP @ IoU=0.50:0.95
            'mAP50': float(coco_eval.stats[1]),        # mAP @ IoU=0.50
            'mAP75': float(coco_eval.stats[2]),        # mAP @ IoU=0.75
            'recall': float(coco_eval.stats[8]),       # AR maxDet=100
            'instances': len(eval_ann_ids),
        }

        os.unlink(tier_gt_tmp.name)

    os.unlink(gt_tmp.name)

    # Compute delta
    if 'Easy' in results and 'Hard' in results:
        results['delta'] = {
            k: results['Easy'].get(k, 0) - results['Hard'].get(k, 0)
            for k in ['mAP50', 'mAP75', 'mAP50_95', 'recall']
        }

    return results


def compute_per_instance_failure(predictions_json: str,
                                  test_annotations: List[ImageAnnotation],
                                  class_names: List[str],
                                  iou_threshold: float = 0.5) -> List[Tuple[Instance, bool]]:
    """For each GT instance, determine if it was successfully detected.

    Returns list of (instance, detected_bool) tuples.
    Used for weight validation (Table 6).
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import tempfile

    gt_coco_data = load_coco_annotations_from_yolo(test_annotations, class_names)
    gt_tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(gt_coco_data, gt_tmp)
    gt_tmp.close()

    coco_gt = COCO(gt_tmp.name)
    coco_dt = coco_gt.loadRes(predictions_json)

    coco_eval = COCOeval(coco_gt, coco_dt, 'segm')
    coco_eval.params.iouThrs = [iou_threshold]
    coco_eval.evaluate()

    # Extract per-instance match info
    all_instances = []
    ann_counter = 0
    for ann in test_annotations:
        for inst in ann.instances:
            # Check if this instance was matched
            detected = False
            img_id = test_annotations.index(ann) + 1

            for eval_img in coco_eval.evalImgs:
                if eval_img is None:
                    continue
                if eval_img.get('image_id') == img_id and eval_img.get('category_id') == inst.bbox.class_id:
                    gt_matches = eval_img.get('gtMatches', np.array([]))
                    if gt_matches.size > 0:
                        # gtMatches shape: (T, G) where T=IoU thresholds, G=GT instances
                        gt_idx_in_img = [i for i, g in enumerate(
                            [a for a in gt_coco_data['annotations']
                             if a['image_id'] == img_id and a['category_id'] == inst.bbox.class_id]
                        ) if g['id'] == ann_counter + 1]
                        if gt_idx_in_img:
                            gidx = gt_idx_in_img[0]
                            if gidx < gt_matches.shape[1]:
                                detected = gt_matches[0, gidx] > 0
                    break

            all_instances.append((inst, detected))
            ann_counter += 1

    os.unlink(gt_tmp.name)
    return all_instances


# ============================================================================
# Phase 4: Weight Validation (Table 6)
# ============================================================================

WEIGHT_SCHEMES = {
    'Equal (0.33, 0.33, 0.33)': (0.33, 0.33, 0.33),
    'Size-only (1.0, 0.0, 0.0)': (1.0, 0.0, 0.0),
    'Occlusion-only (0.0, 1.0, 0.0)': (0.0, 1.0, 0.0),
    'Density-only (0.0, 0.0, 1.0)': (0.0, 0.0, 1.0),
    'Proposed (0.40, 0.35, 0.25)': (0.40, 0.35, 0.25),
}


def validate_weights(instance_results: List[Tuple[Instance, bool]],
                     test_annotations: List[ImageAnnotation],
                     class_names: List[str],
                     predictions_json: str) -> pd.DataFrame:
    """Run Table 6: weight sensitivity analysis.

    For each weight scheme, recompute difficulty → compute correlation with failure rate.
    """
    rows = []

    for scheme_name, (a, b, g) in WEIGHT_SCHEMES.items():
        # Recompute difficulty under this weight scheme
        for ann in test_annotations:
            compute_difficulty_with_weights(
                ann.instances, ann.width, ann.height, a, b, g
            )

        # Collect difficulty scores and failure indicators
        difficulties = []
        failures = []  # 1 = failed, 0 = detected
        for inst, detected in instance_results:
            difficulties.append(inst.difficulty_score)
            failures.append(0.0 if detected else 1.0)

        difficulties = np.array(difficulties)
        failures = np.array(failures)

        # Pearson and Spearman correlation
        pearson_r, _ = stats.pearsonr(difficulties, failures)
        spearman_rho, _ = stats.spearmanr(difficulties, failures)

        # Compute Easy-Hard delta under this scheme
        strat_results = stratified_evaluation(
            predictions_json, test_annotations, class_names,
            alpha=a, beta=b, gamma=g
        )
        delta_map50 = strat_results.get('delta', {}).get('mAP50', 0) * 100

        rows.append({
            'Weighting': scheme_name,
            'Pearson r': round(pearson_r, 2),
            'Spearman ρ': round(spearman_rho, 2),
            'Δ (%)': round(delta_map50, 1),
        })

    # Restore proposed weights
    for ann in test_annotations:
        compute_difficulty_with_weights(
            ann.instances, ann.width, ann.height, ALPHA, BETA, GAMMA
        )

    return pd.DataFrame(rows)


# ============================================================================
# Phase 5: Measure Inference Time
# ============================================================================

def measure_inference_time(weights_path: str, imgsz: int = 640,
                           n_warmup: int = 10, n_measure: int = 100,
                           device: str = 'cuda') -> float:
    """Measure average inference time in ms per image."""
    import torch
    from ultralytics import YOLO

    model = YOLO(weights_path, task='segment')

    # Create dummy input
    dummy = torch.randn(1, 3, imgsz, imgsz).to(device)

    # Warmup
    for _ in range(n_warmup):
        model.predict(dummy, verbose=False, imgsz=imgsz)

    # Measure
    if device == 'cuda':
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_measure):
        model.predict(dummy, verbose=False, imgsz=imgsz)
    if device == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    ms_per_img = (elapsed / n_measure) * 1000
    return ms_per_img


# ============================================================================
# Phase 6: Compute Dataset Statistics (Table 2)
# ============================================================================

def compute_dataset_statistics(all_annotations: Dict[str, List[ImageAnnotation]]) -> pd.DataFrame:
    """Compute Table 2: difficulty-tier distribution statistics."""
    all_instances = []
    for split_name, anns in all_annotations.items():
        for ann in anns:
            all_instances.extend(ann.instances)

    tiers = defaultdict(list)
    for inst in all_instances:
        tiers[inst.difficulty_tier].append(inst)

    rows = []
    for tier_name in ['Easy', 'Medium', 'Hard']:
        tier_insts = tiers[tier_name]
        n = len(tier_insts)
        if n == 0:
            continue
        rows.append({
            'Tier': tier_name,
            'Instances': n,
            'Fraction': f"{n/len(all_instances)*100:.1f}%",
            'Mean S_i': np.mean([i.size_factor for i in tier_insts]),
            'Mean O_i': np.mean([i.occlusion_factor for i in tier_insts]),
            'Mean D_i': np.mean([i.density_factor for i in tier_insts]),
        })

    return pd.DataFrame(rows)


# ============================================================================
# Main Orchestrator
# ============================================================================

@dataclass
class ExperimentRun:
    """Defines a single training + evaluation experiment."""
    name: str
    selection: str           # 'clip' or 'random'
    mask_type: str           # 'sam2' or 'box_only'
    n_samples: int
    model_variant: str       # 's', 'm', 'l'
    copy_paste: float        # 0.0 or 0.5
    eval_sets: List[str]     # ['test', 'boulders', 'stony_pt']
    tables: List[str]        # which paper tables this contributes to
    weights_path: str = ''   # filled after training


def define_all_experiments(n_train_pool: int) -> List[ExperimentRun]:
    """Define all experiments from the paper."""
    experiments = []

    # Table 4 + 5: Active learning (400 samples)
    experiments.append(ExperimentRun(
        name='random_boxonly_400', selection='random', mask_type='box_only',
        n_samples=400, model_variant='s', copy_paste=0.0,
        eval_sets=['test'], tables=['tab4', 'tab5']
    ))
    experiments.append(ExperimentRun(
        name='random_sam2_400', selection='random', mask_type='sam2',
        n_samples=400, model_variant='s', copy_paste=0.0,
        eval_sets=['test'], tables=['tab4', 'tab5']
    ))
    experiments.append(ExperimentRun(
        name='clip_boxonly_400', selection='clip', mask_type='box_only',
        n_samples=400, model_variant='s', copy_paste=0.0,
        eval_sets=['test'], tables=['tab4', 'tab5']
    ))
    experiments.append(ExperimentRun(
        name='clip_sam2_400', selection='clip', mask_type='sam2',
        n_samples=400, model_variant='s', copy_paste=0.0,
        eval_sets=['test'], tables=['tab3', 'tab4', 'tab5']
    ))

    # Table 4: Active learning (full 664)
    experiments.append(ExperimentRun(
        name='random_boxonly_full', selection='random', mask_type='box_only',
        n_samples=n_train_pool, model_variant='s', copy_paste=0.0,
        eval_sets=['test'], tables=['tab4']
    ))
    experiments.append(ExperimentRun(
        name='random_sam2_full', selection='random', mask_type='sam2',
        n_samples=n_train_pool, model_variant='s', copy_paste=0.0,
        eval_sets=['test'], tables=['tab4']
    ))
    experiments.append(ExperimentRun(
        name='clip_sam2_full', selection='clip', mask_type='sam2',
        n_samples=n_train_pool, model_variant='s', copy_paste=0.0,
        eval_sets=['test'], tables=['tab3', 'tab4', 'tab7']
    ))

    # Table 7: Model scaling (full dataset, clip+sam2)
    for variant in ['m', 'l']:
        experiments.append(ExperimentRun(
            name=f'clip_sam2_full_{variant}', selection='clip', mask_type='sam2',
            n_samples=n_train_pool, model_variant=variant, copy_paste=0.0,
            eval_sets=['test'], tables=['tab7']
        ))

    # Table 9: Cross-domain (Random 664, with and without Copy-Paste)
    experiments.append(ExperimentRun(
        name='random_sam2_full_xdomain', selection='random', mask_type='sam2',
        n_samples=n_train_pool, model_variant='s', copy_paste=0.0,
        eval_sets=['boulders', 'stony_pt'], tables=['tab9']
    ))
    experiments.append(ExperimentRun(
        name='random_sam2_full_cp_xdomain', selection='random', mask_type='sam2',
        n_samples=n_train_pool, model_variant='s', copy_paste=0.5,
        eval_sets=['boulders', 'stony_pt'], tables=['tab9']
    ))

    # Table 9: Cross-domain (CLIP 400, with and without Copy-Paste)
    experiments.append(ExperimentRun(
        name='clip_sam2_400_xdomain', selection='clip', mask_type='sam2',
        n_samples=400, model_variant='s', copy_paste=0.0,
        eval_sets=['boulders', 'stony_pt'], tables=['tab9']
    ))
    experiments.append(ExperimentRun(
        name='clip_sam2_400_cp_xdomain', selection='clip', mask_type='sam2',
        n_samples=400, model_variant='s', copy_paste=0.5,
        eval_sets=['boulders', 'stony_pt'], tables=['tab9']
    ))

    return experiments


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='AP-Bench Experiment Runner — runs all experiments from the ECCV paper'
    )
    parser.add_argument('--dataset_root', required=True,
                        help='Root dir of the AP-Bench YOLO dataset')
    parser.add_argument('--boulders_root', default=None,
                        help='Root dir of Boulders field YOLO dataset (cross-domain)')
    parser.add_argument('--stony_pt_root', default=None,
                        help='Root dir of Stony Point field YOLO dataset (cross-domain)')
    parser.add_argument('--output_dir', default='./ap_bench_results',
                        help='Base directory for all outputs')
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip training, use existing weights')
    parser.add_argument('--skip_clip', action='store_true',
                        help='Skip CLIP embedding computation (use cached)')
    parser.add_argument('--only_tables', nargs='+', default=None,
                        help='Only run experiments for specific tables (e.g. tab3 tab4)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Setup
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    output_dir = args.output_dir
    datasets_dir = os.path.join(output_dir, 'datasets')
    runs_dir = os.path.join(output_dir, 'runs')
    results_dir = os.path.join(output_dir, 'results')
    for d in [output_dir, datasets_dir, runs_dir, results_dir]:
        os.makedirs(d, exist_ok=True)

    # ---- Phase 0: Load dataset ----
    logger.info("=" * 80)
    logger.info("PHASE 0: Loading dataset")
    logger.info("=" * 80)

    splits, class_names = load_yolo_dataset(args.dataset_root)
    train_anns = splits.get('train', [])
    val_anns = splits.get('val', [])
    test_anns = splits.get('test', [])

    logger.info(f"Dataset: {len(train_anns)} train, {len(val_anns)} val, "
                f"{len(test_anns)} test images")
    logger.info(f"Classes: {class_names}")

    # Load cross-domain datasets if provided
    cross_domain = {}
    if args.boulders_root and os.path.isdir(args.boulders_root):
        bd_splits, _ = load_yolo_dataset(args.boulders_root)
        # Merge all splits into one eval set
        cross_domain['boulders'] = sum(bd_splits.values(), [])
        logger.info(f"Boulders: {len(cross_domain['boulders'])} images")

    if args.stony_pt_root and os.path.isdir(args.stony_pt_root):
        sp_splits, _ = load_yolo_dataset(args.stony_pt_root)
        cross_domain['stony_pt'] = sum(sp_splits.values(), [])
        logger.info(f"Stony Point: {len(cross_domain['stony_pt'])} images")

    # Populate image dimensions
    logger.info("Computing image dimensions...")
    for anns_list in [train_anns, val_anns, test_anns] + list(cross_domain.values()):
        populate_image_dimensions(anns_list)

    # ---- Phase 1: Compute AP-Difficulty for all instances ----
    logger.info("=" * 80)
    logger.info("PHASE 1: Computing AP-Difficulty scores")
    logger.info("=" * 80)

    all_annotations = {'train': train_anns, 'val': val_anns, 'test': test_anns}
    all_annotations.update(cross_domain)

    total_instances = 0
    for split_name, anns in all_annotations.items():
        for ann in anns:
            compute_difficulty_factors(ann.instances, ann.width, ann.height)
            assign_difficulty_tiers(ann.instances)
            total_instances += len(ann.instances)
    logger.info(f"Computed difficulty for {total_instances} instances")

    # Table 2: Difficulty distribution
    table2 = compute_dataset_statistics(all_annotations)
    logger.info(f"\n{'='*60}\nTable 2: Difficulty Distribution\n{'='*60}")
    logger.info(f"\n{table2.to_string(index=False)}")
    table2.to_csv(os.path.join(results_dir, 'table2_difficulty_distribution.csv'), index=False)

    # ---- Phase 2: CLIP embeddings + subset selection ----
    logger.info("=" * 80)
    logger.info("PHASE 2: CLIP diversity sampling")
    logger.info("=" * 80)

    clip_cache = os.path.join(output_dir, 'clip_embeddings.npy')
    train_paths = [ann.image_path for ann in train_anns]

    if args.skip_clip and os.path.exists(clip_cache):
        embeddings = np.load(clip_cache)
        logger.info(f"Loaded cached CLIP embeddings: {embeddings.shape}")
    else:
        embeddings = compute_clip_embeddings(train_paths, device=args.device)
        np.save(clip_cache, embeddings)
        logger.info(f"Saved CLIP embeddings to {clip_cache}")

    n_pool = len(train_anns)
    clip_subsets = select_clip_subsets(train_anns, embeddings, AL_SAMPLE_COUNTS)
    random_subsets = select_random_subsets(n_pool, AL_SAMPLE_COUNTS)

    # ---- Phase 3: Create dataset variants ----
    logger.info("=" * 80)
    logger.info("PHASE 3: Creating dataset variants")
    logger.info("=" * 80)

    experiments = define_all_experiments(n_pool)

    # Filter to requested tables
    if args.only_tables:
        experiments = [e for e in experiments
                       if any(t in e.tables for t in args.only_tables)]
        logger.info(f"Filtered to {len(experiments)} experiments for tables: {args.only_tables}")

    # Deduplicate training configs (some experiments can share weights)
    # Key: (selection, mask_type, n_samples, model_variant, copy_paste)
    unique_training_configs = {}
    for exp in experiments:
        key = (exp.selection, exp.mask_type, exp.n_samples, exp.model_variant, exp.copy_paste)
        if key not in unique_training_configs:
            unique_training_configs[key] = exp.name

    logger.info(f"Total experiments: {len(experiments)}, "
                f"unique training runs needed: {len(unique_training_configs)}")

    # Create YOLO datasets for each unique config
    dataset_paths = {}  # config_key → data.yaml path

    for config_key, run_name in unique_training_configs.items():
        selection, mask_type, n_samples, model_variant, copy_paste = config_key
        box_only = (mask_type == 'box_only')

        # Get subset indices
        if selection == 'clip':
            indices = clip_subsets[n_samples]
        else:
            indices = random_subsets[n_samples]

        ds_dir = os.path.join(datasets_dir, run_name)
        yaml_path = create_yolo_subset(
            train_anns, indices, ds_dir, class_names,
            box_only=box_only,
            val_annotations=val_anns,
            test_annotations=test_anns,
        )
        dataset_paths[config_key] = yaml_path
        logger.info(f"Created dataset: {run_name} ({n_samples} train images, "
                     f"{'box-only' if box_only else 'masks'})")

    # ---- Phase 4: Training ----
    logger.info("=" * 80)
    logger.info("PHASE 4: Training models")
    logger.info("=" * 80)

    weights_registry = {}  # config_key → weights_path

    for config_key, run_name in unique_training_configs.items():
        selection, mask_type, n_samples, model_variant, copy_paste = config_key
        data_yaml = dataset_paths[config_key]
        model_name = MODEL_VARIANTS[model_variant]

        expected_weights = os.path.join(runs_dir, run_name, 'weights', 'best.pt')

        if args.skip_training and os.path.exists(expected_weights):
            logger.info(f"Skipping training for {run_name} (weights exist)")
            weights_registry[config_key] = expected_weights
            continue

        if args.skip_training:
            logger.warning(f"Skip training requested but no weights found at {expected_weights}")
            continue

        weights_path = train_yolo_model(
            data_yaml=data_yaml,
            model_name=model_name,
            run_name=run_name,
            output_base=runs_dir,
            copy_paste=copy_paste,
        )
        weights_registry[config_key] = weights_path

    # Map weights back to experiments
    for exp in experiments:
        key = (exp.selection, exp.mask_type, exp.n_samples, exp.model_variant, exp.copy_paste)
        exp.weights_path = weights_registry.get(key, '')

    # ---- Phase 5: Evaluation ----
    logger.info("=" * 80)
    logger.info("PHASE 5: Running evaluations")
    logger.info("=" * 80)

    all_results = {}  # experiment_name → {eval_set → stratified_results}

    for exp in experiments:
        if not exp.weights_path or not os.path.exists(exp.weights_path):
            logger.warning(f"No weights for {exp.name}, skipping evaluation")
            continue

        logger.info(f"\nEvaluating: {exp.name}")
        exp_results = {}

        for eval_set in exp.eval_sets:
            if eval_set == 'test':
                # Evaluate on OIV7 test set
                config_key = (exp.selection, exp.mask_type, exp.n_samples,
                              exp.model_variant, exp.copy_paste)
                data_yaml = dataset_paths[config_key]

                val_results = run_yolo_val(exp.weights_path, data_yaml, split='test')

                # Find predictions JSON (saved by ultralytics)
                pred_json = None
                for root, dirs, files in os.walk(runs_dir):
                    for f in files:
                        if f.endswith('predictions.json') and exp.name in root:
                            pred_json = os.path.join(root, f)
                            break

                if pred_json:
                    strat = stratified_evaluation(
                        pred_json, test_anns, class_names
                    )
                    exp_results['test'] = strat
                    logger.info(f"  test: mAP50={strat['aggregate']['mAP50']:.3f}, "
                                f"Easy={strat.get('Easy', {}).get('mAP50', 0):.3f}, "
                                f"Hard={strat.get('Hard', {}).get('mAP50', 0):.3f}")
                else:
                    # Use ultralytics results directly (aggregate only)
                    metrics = val_results.results_dict if hasattr(val_results, 'results_dict') else {}
                    exp_results['test'] = {
                        'aggregate': {
                            'mAP50': metrics.get('metrics/mAP50(M)', 0),
                            'mAP75': metrics.get('metrics/mAP75(M)', 0),
                            'mAP50_95': metrics.get('metrics/mAP50-95(M)', 0),
                        }
                    }

            elif eval_set in cross_domain:
                # Cross-domain evaluation
                xd_anns = cross_domain[eval_set]

                # Create a temporary dataset for cross-domain eval
                xd_dir = os.path.join(datasets_dir, f'xdomain_{eval_set}')
                xd_yaml = create_yolo_subset(
                    [], [], xd_dir, class_names,
                    test_annotations=xd_anns,
                )

                xd_val_results = run_yolo_val(
                    exp.weights_path, xd_yaml, split='test'
                )

                metrics = xd_val_results.results_dict if hasattr(xd_val_results, 'results_dict') else {}
                exp_results[eval_set] = {
                    'aggregate': {
                        'mAP50': metrics.get('metrics/mAP50(M)', 0),
                        'mAP75': metrics.get('metrics/mAP75(M)', 0),
                        'mAP50_95': metrics.get('metrics/mAP50-95(M)', 0),
                    }
                }

        all_results[exp.name] = exp_results

    # ---- Phase 6: Compile Tables ----
    logger.info("=" * 80)
    logger.info("PHASE 6: Compiling result tables")
    logger.info("=" * 80)

    # --- Table 3: Difficulty-stratified performance (primary model) ---
    primary_exp = 'clip_sam2_full'  # CLIP+SAM2, full dataset, yolo11s-seg
    if primary_exp in all_results and 'test' in all_results[primary_exp]:
        strat = all_results[primary_exp]['test']
        table3_rows = []
        for tier in ['Easy', 'Medium', 'Hard']:
            if tier in strat:
                table3_rows.append({
                    'Tier': tier,
                    'Instances': strat[tier].get('instances', ''),
                    'mAP50': f"{strat[tier]['mAP50']*100:.1f}",
                    'mAP75': f"{strat[tier]['mAP75']*100:.1f}",
                    'mAP50_95': f"{strat[tier]['mAP50_95']*100:.1f}",
                    'Recall': f"{strat[tier]['recall']:.2f}",
                })
        if 'aggregate' in strat:
            table3_rows.append({
                'Tier': 'Aggregate',
                'Instances': strat['aggregate'].get('instances', ''),
                'mAP50': f"{strat['aggregate']['mAP50']*100:.1f}",
                'mAP75': f"{strat['aggregate']['mAP75']*100:.1f}",
                'mAP50_95': f"{strat['aggregate']['mAP50_95']*100:.1f}",
                'Recall': f"{strat['aggregate']['recall']:.2f}",
            })
        if 'delta' in strat:
            table3_rows.append({
                'Tier': 'Δ (Easy-Hard)',
                'Instances': '',
                'mAP50': f"{strat['delta']['mAP50']*100:.1f}",
                'mAP75': f"{strat['delta']['mAP75']*100:.1f}",
                'mAP50_95': f"{strat['delta']['mAP50_95']*100:.1f}",
                'Recall': f"{strat['delta']['recall']:.2f}",
            })

        table3 = pd.DataFrame(table3_rows)
        logger.info(f"\n{'='*60}\nTable 3: Stratified Performance\n{'='*60}\n{table3.to_string(index=False)}")
        table3.to_csv(os.path.join(results_dir, 'table3_stratified.csv'), index=False)

    # --- Table 4: Active learning comparison ---
    tab4_mapping = {
        ('Random', 'Box-only', 400): 'random_boxonly_400',
        ('Random', 'SAM2', 400): 'random_sam2_400',
        ('CLIP', 'Box-only', 400): 'clip_boxonly_400',
        ('CLIP', 'SAM2', 400): 'clip_sam2_400',
        ('Random', 'Box-only', n_pool): 'random_boxonly_full',
        ('Random', 'SAM2', n_pool): 'random_sam2_full',
        ('CLIP', 'SAM2', n_pool): 'clip_sam2_full',
    }
    table4_rows = []
    for (sel, masks, n), exp_name in tab4_mapping.items():
        if exp_name in all_results and 'test' in all_results[exp_name]:
            agg = all_results[exp_name]['test'].get('aggregate', {})
            table4_rows.append({
                'Selection': sel,
                'Masks': masks,
                'Samples': n,
                'mAP50': f"{agg.get('mAP50', 0):.2f}",
                'mAP75': f"{agg.get('mAP75', 0):.2f}",
                'mAP50_95': f"{agg.get('mAP50_95', 0):.2f}",
            })

    if table4_rows:
        table4 = pd.DataFrame(table4_rows)
        logger.info(f"\n{'='*60}\nTable 4: Active Learning\n{'='*60}\n{table4.to_string(index=False)}")
        table4.to_csv(os.path.join(results_dir, 'table4_active_learning.csv'), index=False)

    # --- Table 5: Stratified AL (per-tier mAP50 at 400 samples) ---
    tab5_mapping = {
        'Random, Box-only': 'random_boxonly_400',
        'Random, SAM2': 'random_sam2_400',
        'CLIP, Box-only': 'clip_boxonly_400',
        'CLIP, SAM2': 'clip_sam2_400',
    }
    table5_rows = []
    for config_label, exp_name in tab5_mapping.items():
        if exp_name in all_results and 'test' in all_results[exp_name]:
            strat = all_results[exp_name]['test']
            easy = strat.get('Easy', {}).get('mAP50', 0) * 100
            medium = strat.get('Medium', {}).get('mAP50', 0) * 100
            hard = strat.get('Hard', {}).get('mAP50', 0) * 100
            table5_rows.append({
                'Configuration': config_label,
                'Easy': f"{easy:.1f}",
                'Medium': f"{medium:.1f}",
                'Hard': f"{hard:.1f}",
                'Δ': f"{easy - hard:.1f}",
            })

    if table5_rows:
        table5 = pd.DataFrame(table5_rows)
        logger.info(f"\n{'='*60}\nTable 5: Stratified AL\n{'='*60}\n{table5.to_string(index=False)}")
        table5.to_csv(os.path.join(results_dir, 'table5_stratified_al.csv'), index=False)

    # --- Table 6: Weight validation ---
    # Requires per-instance failure data from val set
    if primary_exp in all_results and not args.skip_training:
        logger.info("Computing weight validation (Table 6)...")
        primary_weights = None
        for exp in experiments:
            if exp.name == primary_exp:
                primary_weights = exp.weights_path
                break

        if primary_weights:
            # Run on val set for weight validation
            primary_config_key = ('clip', 'sam2', n_pool, 's', 0.0)
            val_data_yaml = dataset_paths.get(primary_config_key)

            if val_data_yaml:
                val_results = run_yolo_val(primary_weights, val_data_yaml, split='val')

                # Find val predictions JSON
                val_pred_json = None
                for root, dirs, files in os.walk(runs_dir):
                    for f in files:
                        if f.endswith('predictions.json'):
                            val_pred_json = os.path.join(root, f)
                            break

                if val_pred_json:
                    instance_results = compute_per_instance_failure(
                        val_pred_json, val_anns, class_names
                    )
                    table6 = validate_weights(
                        instance_results, val_anns, class_names, val_pred_json
                    )
                    logger.info(f"\n{'='*60}\nTable 6: Weight Validation\n{'='*60}\n{table6.to_string(index=False)}")
                    table6.to_csv(os.path.join(results_dir, 'table6_weight_validation.csv'), index=False)

    # --- Table 7: Model scaling ---
    table7_rows = []
    for variant in ['s', 'm', 'l']:
        exp_name = f'clip_sam2_full' if variant == 's' else f'clip_sam2_full_{variant}'
        if exp_name in all_results and 'test' in all_results[exp_name]:
            strat = all_results[exp_name]['test']
            agg_map50 = strat.get('aggregate', {}).get('mAP50', 0) * 100
            easy_map50 = strat.get('Easy', {}).get('mAP50', 0) * 100
            hard_map50 = strat.get('Hard', {}).get('mAP50', 0) * 100

            # Measure inference time
            weights = None
            for exp in experiments:
                if exp.name == exp_name:
                    weights = exp.weights_path
                    break
            ms_per_img = measure_inference_time(weights) if weights and os.path.exists(weights) else 0

            model_params = {'s': '11.2M', 'm': '20.1M', 'l': '25.3M'}
            table7_rows.append({
                'Model': f'YOLOv11{variant}-seg',
                'Params': model_params[variant],
                'mAP50': f"{agg_map50:.1f}",
                'Easy': f"{easy_map50:.1f}",
                'Hard': f"{hard_map50:.1f}",
                'Δ': f"{easy_map50 - hard_map50:.1f}",
                'ms/img': f"{ms_per_img:.1f}",
            })

    if table7_rows:
        table7 = pd.DataFrame(table7_rows)
        logger.info(f"\n{'='*60}\nTable 7: Model Scaling\n{'='*60}\n{table7.to_string(index=False)}")
        table7.to_csv(os.path.join(results_dir, 'table7_scaling.csv'), index=False)

    # --- Table 9: Cross-domain ---
    tab9_mapping = [
        ('Random (664)', '', 'random_sam2_full_xdomain'),
        ('+ Copy-Paste', '', 'random_sam2_full_cp_xdomain'),
        ('CLIP+SAM2 (400)', '', 'clip_sam2_400_xdomain'),
        ('+ Copy-Paste', '', 'clip_sam2_400_cp_xdomain'),
    ]
    table9_rows = []
    for method, _, exp_name in tab9_mapping:
        if exp_name not in all_results:
            continue
        for site in ['boulders', 'stony_pt']:
            site_label = 'Boulders' if site == 'boulders' else 'Stony Pt.'
            if site in all_results[exp_name]:
                agg = all_results[exp_name][site].get('aggregate', {})
                table9_rows.append({
                    'Method': method,
                    'Target Site': site_label,
                    'mAP50': f"{agg.get('mAP50', 0):.2f}",
                    'mAP75': f"{agg.get('mAP75', 0):.2f}",
                    'mAP50_95': f"{agg.get('mAP50_95', 0):.2f}",
                })

    if table9_rows:
        table9 = pd.DataFrame(table9_rows)
        logger.info(f"\n{'='*60}\nTable 9: Cross-Domain\n{'='*60}\n{table9.to_string(index=False)}")
        table9.to_csv(os.path.join(results_dir, 'table9_cross_domain.csv'), index=False)

    # ---- Save all raw results ----
    with open(os.path.join(results_dir, 'all_results_raw.json'), 'w') as f:
        # Convert numpy types for JSON serialisation
        def convert(obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return obj
        json.dump(all_results, f, indent=2, default=convert)

    logger.info(f"\n{'='*80}")
    logger.info(f"ALL DONE — Results saved to {results_dir}")
    logger.info(f"{'='*80}")


if __name__ == '__main__':
    main()
