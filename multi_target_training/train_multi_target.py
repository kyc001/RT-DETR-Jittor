#!/usr/bin/env python3
"""
多目标检测单图训练脚本
专用于训练RT-DETR模型识别000000000785.jpg中的人和滑雪板
"""

from src.nn.model import RTDETR
from jittor import nn
import jittor as jt
import os
import sys
import argparse
import numpy as np
from PIL import Image
import json

# 添加项目路径
sys.path.insert(0, '/home/kyc/project/RT-DETR/jittor_rt_detr')


# 简单的数据变换

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        return image.resize(self.size, Image.LANCZOS)


class ToTensor:
    def __call__(self, image):
        image_array = np.array(image, dtype=np.float32) / 255.0
        return jt.array(image_array.transpose(2, 0, 1))


class Normalize:
    def __init__(self, mean, std):
        self.mean = jt.array(mean).view(-1, 1, 1)
        self.std = jt.array(std).view(-1, 1, 1)

    def __call__(self, tensor):
        return (tensor - self.mean) / self.std


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image):
        if np.random.rand() < self.p:
            return image.transpose(Image.FLIP_LEFT_RIGHT)
        return image

# 数据加载


def load_multi_image_data(img_path, ann_file, target_image_name="000000000785.jpg"):
    print(f"=== 加载多目标图片数据: {target_image_name} ===")
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    target_image = None
    for img in coco_data['images']:
        if img['file_name'] == target_image_name:
            target_image = img
            break
    if target_image is None:
        raise ValueError(f"找不到图片: {target_image_name}")
    print(f"✅ 找到图片: {target_image['file_name']}")
    print(f"   尺寸: {target_image['width']} x {target_image['height']}")
    print(f"   ID: {target_image['id']}")
    image_annotations = []
    for ann in coco_data['annotations']:
        if ann['image_id'] == target_image['id']:
            image_annotations.append(ann)
    print(f"✅ 找到 {len(image_annotations)} 个标注")
    category_counts = {}
    for ann in image_annotations:
        cat_id = ann['category_id']
        if cat_id not in category_counts:
            category_counts[cat_id] = 0
        category_counts[cat_id] += 1
    cat_id_to_name = {}
    for cat in coco_data['categories']:
        cat_id_to_name[cat['id']] = cat['name']
    print("📊 图片中的类别分布:")
    for cat_id, count in category_counts.items():
        cat_name = cat_id_to_name.get(cat_id, f"Unknown_{cat_id}")
        print(f"   {cat_name}: {count} 个实例")
    return target_image, image_annotations, cat_id_to_name


class MultiImageDataset(jt.dataset.Dataset):
    """
    多目标单图数据集
    """

    def __init__(self, img_path, image_info, annotations, cat_id_to_name, transforms=None):
        super().__init__()
        self.img_path = img_path
        self.image_info = image_info
        self.annotations = annotations
        self.cat_id_to_name = cat_id_to_name
        self.transforms = transforms
        unique_cat_ids = sorted(set(ann['category_id'] for ann in annotations))
        self.cat_id_to_idx = {cat_id: idx for idx,
                              cat_id in enumerate(unique_cat_ids)}
        self.idx_to_cat_id = {idx: cat_id for cat_id,
                              idx in self.cat_id_to_idx.items()}
        print(f"📋 类别映射:")
        for cat_id, idx in self.cat_id_to_idx.items():
            cat_name = self.cat_id_to_name.get(cat_id, f"Unknown_{cat_id}")
            print(f"   {cat_name} (ID:{cat_id}) -> 索引:{idx}")

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        image = Image.open(self.img_path).convert('RGB')
        original_size = image.size
        boxes = []
        labels = []
        for ann in self.annotations:
            x, y, w, h = ann['bbox']
            cx = (x + w / 2) / original_size[0]
            cy = (y + h / 2) / original_size[1]
            w_norm = w / original_size[0]
            h_norm = h / original_size[1]
            boxes.append([cx, cy, w_norm, h_norm])
            cat_id = ann['category_id']
            label_idx = self.cat_id_to_idx[cat_id]
            labels.append(label_idx)
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        if self.transforms:
            image = self.transforms(image)
        target = {
            'boxes': jt.array(boxes),
            'labels': jt.array(labels),
            'image_id': jt.array(np.array([int(self.image_info['id'])], dtype=np.int64)),
            'orig_size': jt.array(np.array([int(original_size[1]), int(original_size[0])], dtype=np.int64)),
            'size': jt.array(np.array([640, 640], dtype=np.int64))
        }
        return image, target


def main():
    parser = argparse.ArgumentParser(description="多目标单图训练 - 人和滑雪板")
    parser.add_argument('--img_path', type=str,
                        default='data/coco/val2017/000000000785.jpg', help='图片路径')
    parser.add_argument('--ann_file', type=str,
                        default='data/coco/annotations/instances_val2017.json', help='标注文件')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--output_dir', type=str,
                        default='multi_target_training/checkpoints', help='模型保存目录')
    args = parser.parse_args()
    print("=== 多目标单图训练：人和滑雪板检测 ===")
    print(f"配置参数:")
    print(f"  - 图片路径: {args.img_path}")
    print(f"  - 标注文件: {args.ann_file}")
    print(f"  - 训练轮数: {args.epochs}")
    print(f"  - 学习率: {args.lr}")
    print(f"  - 输出目录: {args.output_dir}")
    if not os.path.exists(args.img_path):
        raise FileNotFoundError(f"图片文件不存在: {args.img_path}")
    if not os.path.exists(args.ann_file):
        raise FileNotFoundError(f"标注文件不存在: {args.ann_file}")
    os.makedirs(args.output_dir, exist_ok=True)
    image_info, annotations, cat_id_to_name = load_multi_image_data(
        args.img_path, args.ann_file, "000000000785.jpg"
    )
    transform = Compose([
        Resize((640, 640)),
        RandomHorizontalFlip(p=0.5),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = MultiImageDataset(
        args.img_path, image_info, annotations, cat_id_to_name, transforms=transform
    )
    dataloader = jt.dataset.DataLoader(dataset, batch_size=1, shuffle=False)
    num_classes = len(dataset.cat_id_to_idx)
    print(f"📊 类别数量: {num_classes}")
    model = RTDETR(num_classes=num_classes)
    # 损失函数和优化器
    from scipy.optimize import linear_sum_assignment

    def box_cxcywh_to_xyxy(x):
        # x: (N, 4)
        x_c = x[..., 0]
        y_c = x[..., 1]
        w = x[..., 2]
        h = x[..., 3]
        b1 = jt.subtract(x_c, 0.5 * w)
        b2 = jt.subtract(y_c, 0.5 * h)
        b3 = jt.add(x_c, 0.5 * w)
        b4 = jt.add(y_c, 0.5 * h)
        return jt.stack([b1, b2, b3, b4], dim=-1)

    def box_iou(boxes1, boxes2):
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        lt = jt.maximum(boxes1[:, None, :2], boxes2[:, :2])
        rb = jt.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
        wh = (rb - lt).clamp(min_v=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        union = area1[:, None] + area2 - inter
        iou = inter / (union + 1e-6)
        return iou, union

    def generalized_box_iou(boxes1, boxes2):
        iou, union = box_iou(boxes1, boxes2)
        lt = jt.minimum(boxes1[:, None, :2], boxes2[:, :2])
        rb = jt.maximum(boxes1[:, None, 2:], boxes2[:, 2:])
        wh = (rb - lt).clamp(min_v=0)
        closure = wh[:, :, 0] * wh[:, :, 1]
        return iou - (closure - union) / (closure + 1e-6)

    class HungarianMatcher(nn.Module):
        def __init__(self, cost_class=1, cost_bbox=1, cost_giou=1):
            super().__init__()
            self.cost_class = cost_class
            self.cost_bbox = cost_bbox
            self.cost_giou = cost_giou

        def execute(self, outputs, targets):
            out_logits = outputs["pred_logits"][-1]
            out_bbox = outputs["pred_boxes"][-1]
            out_prob = jt.sigmoid(out_logits[0])
            out_bbox = out_bbox[0]
            tgt_ids = targets["labels"][0]
            tgt_bbox = targets["boxes"][0]
            # jt.transpose返回的axes参数应为单个int
            # 直接用jt.transpose(out_prob, (1, 0))会报错，改为out_prob.transpose(1, 0)
            cost_class = -out_prob.transpose(1, 0)[tgt_ids].transpose(1, 0)
            # jt没有cdist，改用nn.l1_loss的广播方式
            cost_bbox = jt.abs(out_bbox.unsqueeze(
                1) - tgt_bbox.unsqueeze(0)).sum(-1)
            cost_giou = - \
                generalized_box_iou(box_cxcywh_to_xyxy(
                    out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
            C = jt.add(jt.add(self.cost_bbox * cost_bbox,
                       self.cost_class * cost_class), self.cost_giou * cost_giou)
            indices = linear_sum_assignment(C.numpy())
            return [(jt.array(i, dtype=jt.int64), jt.array(j, dtype=jt.int64)) for i, j in zip(indices[0], indices[1])]

    class SetCriterion(nn.Module):
        def __init__(self, num_classes, matcher, weight_dict, eos_coef):
            super().__init__()
            self.num_classes = num_classes
            self.matcher = matcher
            self.weight_dict = weight_dict
            self.eos_coef = eos_coef
            empty_weight = jt.ones(self.num_classes + 1)
            empty_weight[-1] = self.eos_coef
            self.register_buffer('empty_weight', empty_weight)

        def loss_labels(self, outputs, targets, indices):
            idx = self._get_src_permutation_idx(indices)
            target_classes_o = jt.concat(
                [t["labels"][J] for t, (_, J) in zip(targets, indices)])
            target_classes = jt.full(
                outputs['pred_logits'].shape[:2], self.num_classes, dtype=jt.int32)
            target_classes[idx] = target_classes_o
            loss_ce = nn.cross_entropy_loss(outputs['pred_logits'].transpose(
                1, 2), target_classes, weight=self.empty_weight)
            return {'loss_ce': loss_ce}

        def loss_boxes(self, outputs, targets, indices):
            idx = self._get_src_permutation_idx(indices)
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = jt.concat(
                [t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            loss_bbox = nn.l1_loss(src_boxes, target_boxes, reduction='none')
            loss_giou = 1 - jt.diag(generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(target_boxes)))
            return {'loss_bbox': loss_bbox.sum() / len(target_boxes), 'loss_giou': loss_giou.sum() / len(target_boxes)}

        def _get_src_permutation_idx(self, indices):
            batch_idx = jt.concat([jt.full_like(src, i)
                                  for i, (src, _) in enumerate(indices)])
            src_idx = jt.concat([src for (src, _) in indices])
            return batch_idx, src_idx

        def execute(self, outputs, targets):
            outputs_without_aux = {k: v[-1] for k, v in outputs.items()}
            indices = self.matcher(outputs_without_aux, {
                                   "labels": targets["labels"], "boxes": targets["boxes"]})
            losses = {}
            losses.update(self.loss_labels(outputs_without_aux, [
                          {"labels": targets["labels"][0], "boxes": targets["boxes"][0]}], indices))
            losses.update(self.loss_boxes(outputs_without_aux, [
                          {"labels": targets["labels"][0], "boxes": targets["boxes"][0]}], indices))
            return losses
    weight_dict = {'loss_ce': 2, 'loss_bbox': 5, 'loss_giou': 2}
    matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)
    criterion = SetCriterion(
        num_classes=num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=0.1)
    optimizer = jt.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = jt.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.5)
    print(f"✅ 开始训练...")
    for epoch in range(args.epochs):
        model.train()
        criterion.train()
        for batch_idx, (images, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            logits, boxes, enc_logits, enc_boxes = model(images)
            outputs = {
                'pred_logits': logits,
                'pred_boxes': boxes,
            }
            losses = criterion(outputs, targets)
            total_loss = sum(losses[k] * weight_dict[k]
                             for k in losses.keys() if k in weight_dict)
            optimizer.backward(total_loss)
            optimizer.step()
            scheduler.step()
            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {total_loss.item():.4f}, "
                  f"CE: {losses['loss_ce'].item():.4f}, L1: {losses['loss_bbox'].item():.4f}, GIoU: {losses['loss_giou'].item():.4f}")
        if (epoch + 1) % 20 == 0:
            save_path = f'{args.output_dir}/multi_target_model_epoch_{epoch+1}.pkl'
            jt.save(model.state_dict(), save_path)
            print(f"✅ 模型已保存: {save_path}")
    final_save_path = f'{args.output_dir}/multi_target_model_final.pkl'
    jt.save(model.state_dict(), final_save_path)
    print(f"🎉 最终模型已保存: {final_save_path}")
    # 保存类别映射
    cat_id_to_idx_path = f'{args.output_dir}/cat_id_to_idx.json'
    with open(cat_id_to_idx_path, 'w') as f:
        json.dump(dataset.cat_id_to_idx, f)
    print(f"✅ 类别映射已保存: {cat_id_to_idx_path}")
    print("🎉 训练完成！")


if __name__ == "__main__":
    main()
