"""COCO-style dataset and dataloader utilities for RT-DETR migration."""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

import jittor as jt


COCO_80_CATEGORY_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
]


def _xyxy_to_cxcywh_normalized(
    xyxy_boxes: np.ndarray,
    width: float,
    height: float,
) -> np.ndarray:
    if xyxy_boxes.size == 0:
        return np.zeros((0, 4), dtype=np.float32)

    x1 = xyxy_boxes[:, 0] / width
    y1 = xyxy_boxes[:, 1] / height
    x2 = xyxy_boxes[:, 2] / width
    y2 = xyxy_boxes[:, 3] / height

    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    bw = x2 - x1
    bh = y2 - y1
    out = np.stack([cx, cy, bw, bh], axis=1).astype(np.float32)
    return np.clip(out, 0.0, 1.0)


def _image_to_tensor(image: Image.Image, image_size: int) -> jt.Var:
    resized = image.resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return jt.array(arr).float32()


@dataclass
class DetectionSample:
    image: jt.Var
    target: Dict[str, jt.Var]


class CocoDetectionDataset:
    """COCO-style detection dataset with lightweight augmentation."""

    def __init__(
        self,
        image_dir: str,
        ann_file: str,
        image_size: int = 640,
        is_train: bool = False,
        hflip_prob: float = 0.5,
        max_samples: int = -1,
    ) -> None:
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        if not os.path.isfile(ann_file):
            raise FileNotFoundError(f"Annotation file not found: {ann_file}")

        self.image_dir = image_dir
        self.ann_file = ann_file
        self.image_size = int(image_size)
        self.is_train = bool(is_train)
        self.hflip_prob = float(hflip_prob)

        with open(ann_file, "r", encoding="utf-8") as f:
            coco = json.load(f)

        categories = sorted(coco.get("categories", []), key=lambda x: x["id"])
        if categories:
            category_ids = [int(c["id"]) for c in categories]
        else:
            category_ids = COCO_80_CATEGORY_IDS

        self.category_ids = category_ids
        self.cat_id_to_label = {cat_id: i for i, cat_id in enumerate(category_ids)}
        self.label_to_cat_id = {i: cat_id for cat_id, i in self.cat_id_to_label.items()}

        images = sorted(coco.get("images", []), key=lambda x: x["id"])
        if max_samples and max_samples > 0:
            images = images[: int(max_samples)]
        self.images = images

        anns_by_image: Dict[int, List[Dict]] = {}
        for ann in coco.get("annotations", []):
            image_id = int(ann.get("image_id", -1))
            if int(ann.get("iscrowd", 0)) == 1:
                continue
            cat_id = int(ann.get("category_id", -1))
            if cat_id not in self.cat_id_to_label:
                continue
            anns_by_image.setdefault(image_id, []).append(ann)
        self.anns_by_image = anns_by_image

    @property
    def num_classes(self) -> int:
        return len(self.category_ids)

    def __len__(self) -> int:
        return len(self.images)

    def label_to_category(self, label: int) -> int:
        return int(self.label_to_cat_id[int(label)])

    def _prepare_target(self, image_record: Dict, do_flip: bool) -> Dict[str, jt.Var]:
        image_id = int(image_record["id"])
        width = float(image_record["width"])
        height = float(image_record["height"])

        boxes_xyxy: List[List[float]] = []
        labels: List[int] = []
        for ann in self.anns_by_image.get(image_id, []):
            x, y, w, h = [float(v) for v in ann["bbox"]]
            if w <= 1e-5 or h <= 1e-5:
                continue

            if do_flip:
                x = width - x - w

            x1 = max(0.0, min(width, x))
            y1 = max(0.0, min(height, y))
            x2 = max(0.0, min(width, x + w))
            y2 = max(0.0, min(height, y + h))
            if x2 <= x1 or y2 <= y1:
                continue

            boxes_xyxy.append([x1, y1, x2, y2])
            labels.append(self.cat_id_to_label[int(ann["category_id"])])

        boxes_xyxy_np = np.asarray(boxes_xyxy, dtype=np.float32)
        boxes_cxcywh = _xyxy_to_cxcywh_normalized(boxes_xyxy_np, width=width, height=height)

        if boxes_cxcywh.size == 0:
            boxes_var = jt.zeros((0, 4), dtype=jt.float32)
            labels_var = jt.zeros((0,), dtype=jt.int64)
        else:
            boxes_var = jt.array(boxes_cxcywh).float32()
            labels_var = jt.array(np.asarray(labels, dtype=np.int64)).int64()

        target = {
            "boxes": boxes_var,
            "labels": labels_var,
            "orig_size": jt.array(np.asarray([height, width], dtype=np.float32)),
            "image_id": jt.array(np.asarray([image_id], dtype=np.int64)),
        }
        return target

    def __getitem__(self, index: int) -> Tuple[jt.Var, Dict[str, jt.Var]]:
        record = self.images[index]
        image_path = os.path.join(self.image_dir, record["file_name"])
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        do_flip = self.is_train and (random.random() < self.hflip_prob)
        if do_flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        image_tensor = _image_to_tensor(image, self.image_size)
        target = self._prepare_target(record, do_flip=do_flip)
        return image_tensor, target


class DummyDetectionDataset:
    """Deterministic synthetic dataset for migration smoke tests."""

    def __init__(
        self,
        num_samples: int = 8,
        image_size: int = 640,
        num_classes: int = 80,
        max_objects: int = 8,
        seed: int = 42,
    ) -> None:
        self.num_samples = int(num_samples)
        self.image_size = int(image_size)
        self.num_classes = int(num_classes)
        self.max_objects = max(1, int(max_objects))
        self.seed = int(seed)
        self.category_ids = COCO_80_CATEGORY_IDS[: self.num_classes]

    @property
    def ann_file(self) -> str:
        return ""

    def __len__(self) -> int:
        return self.num_samples

    def label_to_category(self, label: int) -> int:
        idx = int(label)
        if 0 <= idx < len(self.category_ids):
            return int(self.category_ids[idx])
        return idx + 1

    def __getitem__(self, index: int) -> Tuple[jt.Var, Dict[str, jt.Var]]:
        rng = np.random.RandomState(self.seed + int(index))
        image = rng.uniform(0.0, 1.0, (3, self.image_size, self.image_size)).astype(np.float32)

        num_objects = int(rng.randint(1, self.max_objects + 1))
        centers = rng.uniform(0.1, 0.9, (num_objects, 2)).astype(np.float32)
        sizes = rng.uniform(0.05, 0.4, (num_objects, 2)).astype(np.float32)
        boxes = np.concatenate([centers, sizes], axis=1)
        boxes = np.clip(boxes, 0.0, 1.0).astype(np.float32)
        labels = rng.randint(0, self.num_classes, (num_objects,), dtype=np.int64)

        target = {
            "boxes": jt.array(boxes).float32(),
            "labels": jt.array(labels).int64(),
            "orig_size": jt.array(np.asarray([self.image_size, self.image_size], dtype=np.float32)),
            "image_id": jt.array(np.asarray([index], dtype=np.int64)),
        }
        return jt.array(image).float32(), target


def collate_detection_batch(batch: List[Tuple[jt.Var, Dict[str, jt.Var]]]):
    images = jt.stack([sample[0] for sample in batch], dim=0)
    targets = [sample[1] for sample in batch]
    return images, targets


class SimpleDetectionDataLoader:
    """A small deterministic dataloader for Jittor migration loops."""

    def __init__(
        self,
        dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
    ) -> None:
        self.dataset = dataset
        self.batch_size = max(1, int(batch_size))
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)

    def __len__(self) -> int:
        total = len(self.dataset)
        if self.drop_last:
            return total // self.batch_size
        return (total + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = np.arange(len(self.dataset), dtype=np.int64)
        if self.shuffle:
            np.random.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start:start + self.batch_size]
            if self.drop_last and len(batch_indices) < self.batch_size:
                continue
            batch = [self.dataset[int(i)] for i in batch_indices]
            yield collate_detection_batch(batch)
