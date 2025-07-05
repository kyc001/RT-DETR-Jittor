# dataset.py

import jittor as jt
from PIL import Image
import os
import json
import numpy as np


def xyxy_to_cxcywh_and_normalize(boxes, w, h):
    if boxes.shape[0] == 0:
        return boxes
    boxes_scaled = boxes / jt.array([w, h, w, h])
    x0, y0, x1, y1 = boxes_scaled.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return jt.stack(b, dim=-1)


class COCODataset(jt.dataset.Dataset):
    def __init__(self, img_dir, ann_file, transforms=None):
        super().__init__()
        self.img_dir = img_dir
        self.transforms = transforms

        with open(ann_file, 'r') as f:
            ann_data = json.load(f)

        self.images = {img['id']: img for img in ann_data['images']}
        self.annotations = ann_data['annotations']

        self.imgid2anns = {}
        for a in self.annotations:
            self.imgid2anns.setdefault(a['image_id'], []).append(a)

        self.ids = list(sorted(self.images.keys()))

        original_count = len(self.ids)
        self.ids = [img_id for img_id in self.ids if len(
            self.imgid2anns.get(img_id, [])) > 0]

        cat_ids = sorted({a['category_id'] for a in self.annotations})
        self.catid2contiguous = {cat_id: i for i, cat_id in enumerate(cat_ids)}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.images[img_id]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')

        w, h = img.size

        anns = self.imgid2anns.get(img_id, [])

        boxes = []
        labels = []
        for a in anns:
            bbox = a['bbox']
            x, y, bw, bh = bbox
            boxes.append([x, y, x + bw, y + h])
            labels.append(self.catid2contiguous[a['category_id']])

        boxes = jt.array(boxes, dtype='float32') if boxes else jt.zeros(
            (0, 4), dtype='float32')
        labels = jt.array(labels, dtype='int64') if labels else jt.zeros(
            (0,), dtype='int64')

        boxes_cxcywh = xyxy_to_cxcywh_and_normalize(boxes, w, h)

        if self.transforms:
            img = self.transforms(img)

        return img, boxes_cxcywh, labels

    @staticmethod
    def collate_batch(batch):
        imgs = [s[0] for s in batch]
        boxes = [s[1] for s in batch]
        labels = [s[2] for s in batch]
        return jt.stack(imgs, 0), boxes, labels
