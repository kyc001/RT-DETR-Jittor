import jittor as jt
from PIL import Image
import os
import json
import numpy as np


class COCODataset(jt.dataset.Dataset):
    def __init__(self, img_dir, ann_file, transforms=None):
        super().__init__()
        self.img_dir = img_dir
        self.transforms = transforms

        with open(ann_file, 'r') as f:
            ann_data = json.load(f)

        self.images = ann_data['images']
        self.annotations = ann_data['annotations']

        self.imgid2anns = {}
        for a in self.annotations:
            self.imgid2anns.setdefault(a['image_id'], []).append(a)

        self.id2name = {img['id']: img['file_name'] for img in self.images}
        self.ids = [img['id'] for img in self.images]

        cat_ids = sorted({a['category_id'] for a in self.annotations})
        self.catid2contiguous = {cat_id: i for i, cat_id in enumerate(cat_ids)}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = os.path.join(self.img_dir, self.id2name[img_id])
        img = Image.open(img_path).convert('RGB')

        anns = self.imgid2anns.get(img_id, [])

        boxes = []
        labels = []

        for a in anns:
            bbox = a['bbox']
            x, y, w, h = bbox
            boxes.append([x, y, x + w, y + h])
            labels.append(self.catid2contiguous[a['category_id']])

        if not boxes:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int32)
        else:
            boxes = np.array(boxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int32)

        boxes = jt.array(boxes)
        labels = jt.array(labels)

        if self.transforms:
            img = self.transforms(img)

        return img, boxes, labels

    # ## <<< 关键修正：将打包逻辑作为 collate_batch 方法添加到 Dataset 类中 >>>
    @staticmethod
    def collate_batch(batch):
        """
        这个方法会被 DataLoader 自动调用。
        """
        imgs = [s[0] for s in batch]
        boxes = [s[1] for s in batch]
        labels = [s[2] for s in batch]
        return jt.stack(imgs, 0), boxes, labels
