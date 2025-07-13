#!/usr/bin/env python3
"""
多目标训练脚本 - 000000000785.jpg
基于标准的DETR训练方式，训练RT-DETR模型识别人和滑雪板
使用匈牙利匹配和GIoU损失，确保训练稳定和准确
"""

import jittor as jt
import os
import sys
import argparse
import numpy as np
from PIL import Image
import json

# 新增import
from scipy.optimize import linear_sum_assignment
from collections import Counter

import importlib.util
spec = importlib.util.spec_from_file_location(
    "model", "/home/kyc/project/RT-DETR/jittor_rt_detr/src/nn/model.py")
if spec is None or spec.loader is None:
    raise ImportError("无法找到 RTDETR 模型文件，请检查路径！")
model_module = importlib.util.module_from_spec(spec)
sys.modules["model"] = model_module
spec.loader.exec_module(model_module)
RTDETR = model_module.RTDETR

# --------------------------------- -------------------
# 数据变换部分 (无改动)
# ----------------------------------------------------


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
        self.mean = jt.array(mean).reshape(3, 1, 1)
        self.std = jt.array(std).reshape(3, 1, 1)

    def __call__(self, tensor):
        return (tensor - self.mean) / self.std

# ----------------------------------------------------
# 数据加载部分 (无改动)
# ----------------------------------------------------


def load_multi_target_data(img_path, ann_file, target_image_name="000000000785.jpg"):
    """加载多目标图片的数据和标注"""
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

    image_annotations = []
    for ann in coco_data['annotations']:
        if ann['image_id'] == target_image['id']:
            image_annotations.append(ann)

    cat_id_to_name = {cat['id']: cat['name']
                      for cat in coco_data['categories']}

    return target_image, image_annotations, cat_id_to_name


class MultiTargetDataset(jt.dataset.Dataset):
    """多目标数据集 - 支持人和滑雪板"""

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
            labels.append(self.cat_id_to_idx[ann['category_id']])

        if self.transforms:
            image = self.transforms(image)

        target = {
            'boxes': jt.array(np.array(boxes, dtype=np.float32)),
            'labels': jt.array(np.array(labels, dtype=np.int64)),
            'image_id': jt.array([self.image_info['id']]),
            'orig_size': jt.array([original_size[1], original_size[0]]),
            'size': jt.array([640, 640])
        }

        return image, target

# ==============================================================================
# 新增模块：标准的DETR损失计算方式
# ==============================================================================


def box_cxcywh_to_xyxy(x):
    """将 [cx, cy, w, h] 格式的box转换为 [x1, y1, x2, y2] 格式"""
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return jt.stack(b, dim=-1)


def box_iou(boxes1, boxes2):
    """计算两组box的iou"""
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
    """计算Generalized IoU (GIoU)"""
    iou, union = box_iou(boxes1, boxes2)
    lt = jt.minimum(boxes1[:, None, :2], boxes2[:, :2])
    rb = jt.maximum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min_v=0)
    closure = wh[:, :, 0] * wh[:, :, 1]
    return iou - (closure - union) / (closure + 1e-6)


class HungarianMatcher(jt.nn.Module):
    """
    标准的匈牙利匹配器
    在预测和真实目标之间执行最优的二分匹配
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    def execute(self, outputs, targets):
        # outputs: 模型输出
        # targets: 真实目标

        # 我们只关心最后一层解码器的输出
        # [batch_size, num_queries, num_classes]
        out_logits = outputs["pred_logits"][-1]
        out_bbox = outputs["pred_boxes"][-1]   # [batch_size, num_queries, 4]

        # 取出batch中的第一个元素（因为我们是单一图片训练）
        out_prob = jt.sigmoid(out_logits[0])
        out_bbox = out_bbox[0]

        tgt_ids = targets["labels"][0]
        tgt_bbox = targets["boxes"][0]

        # 1. 计算分类代价
        #  --- 这是被修改的行 ---
        #  将 .transpose(1, 0) 修改为 .transpose((1, 0))
        num_queries = int(out_bbox.shape[0])
        num_targets = int(tgt_bbox.shape[0])
        cost_class = -jt.index_select(out_prob, 1, tgt_ids)
        if len(cost_class.shape) == 3 and cost_class.shape[-1] == 1:
            cost_class = jt.squeeze(cost_class, dim=-1)

        # 2. 计算L1代价
        cost_bbox_rows = []
        for i in range(num_queries):
            row = []
            for j in range(num_targets):
                row.append(jt.abs(out_bbox[i] - tgt_bbox[j]).sum())
            cost_bbox_rows.append(jt.stack(row))
        cost_bbox = jt.stack(cost_bbox_rows)
        if len(cost_bbox.shape) == 3 and cost_bbox.shape[-1] == 1:
            cost_bbox = jt.squeeze(cost_bbox, dim=-1)

        # 3. 计算GIoU代价
        cost_giou = - \
            generalized_box_iou(box_cxcywh_to_xyxy(
                out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # 最终代价矩阵
        cost_bbox_w = jt.ones_like(cost_bbox) * self.cost_bbox
        cost_class_w = jt.ones_like(cost_class) * self.cost_class
        cost_giou_w = jt.ones_like(cost_giou) * self.cost_giou
        C = cost_bbox * cost_bbox_w + cost_class * \
            cost_class_w + cost_giou * cost_giou_w

        print('C shape:', C.shape)
        # 使用scipy进行线性指派
        indices = linear_sum_assignment(C.numpy())
        # 返回匹配上的 (预测索引, 真实目标索引)
        return [(jt.array(i, dtype=jt.int64), jt.array(j, dtype=jt.int64)) for i, j in zip(indices[0], indices[1])]


class SetCriterion(jt.nn.Module):
    """标准的DETR损失函数"""

    def __init__(self, num_classes, matcher, weight_dict, eos_coef):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        empty_weight = jt.ones(self.num_classes + 1)
        # === 修改：前景类别权重为1，背景权重为10 ===
        empty_weight[:-1] = 1.0  # 前景类别
        empty_weight[-1] = 10.0  # 背景
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices):
        """分类损失"""
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = jt.concat([t["labels"][J]
                                     for t, (_, J) in zip(targets, indices)])

        target_classes = jt.full(
            outputs['pred_logits'].shape[:2], self.num_classes, 'int64')
        target_classes[idx] = target_classes_o

        loss_ce = jt.nn.cross_entropy_loss(outputs['pred_logits'].transpose(
            1, 2), target_classes, weight=self.empty_weight)
        return {'loss_ce': loss_ce}

    def loss_boxes(self, outputs, targets, indices):
        """边界框损失 (L1 + GIoU)"""
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = jt.concat([t['boxes'][i]
                                 for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = jt.nn.l1_loss(src_boxes, target_boxes)

        loss_giou = 1 - jt.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)))

        tb_len = target_boxes.shape[0]
        return {'loss_bbox': loss_bbox.sum() / tb_len, 'loss_giou': loss_giou.sum() / tb_len}

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

# ==============================================================================
# 脚本主逻辑
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="多目标训练 - 000000000785.jpg (标准版)")
    parser.add_argument('--img_path', type=str,
                        default='data/coco/val2017/000000000785.jpg', help='图片路径')
    parser.add_argument('--ann_file', type=str,
                        default='data/coco/annotations/instances_val2017.json', help='标注文件')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--output_dir', type=str,
                        default='multi_target_training/checkpoints', help='模型保存目录')
    args = parser.parse_args()

    print("=== 标准化多目标训练：人和滑雪板检测 ===")
    print(f"配置参数: {args}")

    if not os.path.exists(args.img_path):
        raise FileNotFoundError(
            f"图片文件不存在: {args.img_path}")
    if not os.path.exists(args.ann_file):
        raise FileNotFoundError(
            f"标注文件不存在: {args.ann_file}")
    os.makedirs(args.output_dir, exist_ok=True)

    image_info, annotations, cat_id_to_name = load_multi_target_data(
        args.img_path, args.ann_file, "000000000785.jpg"
    )

    # === 新增：统计每个类别的标注数量 ===
    label_counter = Counter([ann['category_id'] for ann in annotations])
    print("\n【标签分布统计】")
    for cat_id, count in label_counter.items():
        print(f"类别 {cat_id_to_name[cat_id]} (ID:{cat_id}): {count} 个标注")
    print("")

    # 修正缩进
    class RandomHorizontalFlip:
        def __init__(self, p=0.5):
            self.p = p

        def __call__(self, image):
            if np.random.rand() < self.p:
                return image.transpose(Image.FLIP_LEFT_RIGHT)
            return image

    transform = Compose([
        Resize((640, 640)),
        RandomHorizontalFlip(p=0.5),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = MultiTargetDataset(
        args.img_path, image_info, annotations, cat_id_to_name, transforms=transform
    )
    dataloader = jt.dataset.DataLoader(dataset, batch_size=1, shuffle=False)

    num_classes = len(dataset.cat_id_to_idx)
    print(f"📊 类别数量: {num_classes}")

    model = RTDETR(num_classes=num_classes)

    # --- 核心修改：使用标准匹配器和损失函数 ---
    weight_dict = {'loss_ce': 2, 'loss_bbox': 5, 'loss_giou': 2}
    matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)
    criterion = SetCriterion(
        num_classes=num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=0.1)
    # ----------------------------------------

    optimizer = jt.optim.Adam(model.parameters(), lr=args.lr)

    print(f"✅ 开始标准训练...")
    for epoch in range(args.epochs):
        model.train()
        criterion.train()

        # === 新增：统计每个类别的平均预测分数 ===
        all_logits = []
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

            # 收集logits用于统计
            all_logits.append(logits[-1][0].stop_grad().numpy())

            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {total_loss.item():.4f}, "
                  f"CE: {losses['loss_ce'].item():.4f}, L1: {losses['loss_bbox'].item():.4f}, GIoU: {losses['loss_giou'].item():.4f}")

        # === 新增：每轮结束后统计各类别平均分数 ===
        # (num_queries, num_classes)
        all_logits_np = np.concatenate(all_logits, axis=0)
        avg_scores = 1.0 / (1.0 + np.exp(-all_logits_np))  # sigmoid
        print("\n【每类别平均预测分数】")
        for idx, cat_name in enumerate([cat_id_to_name[cid] for cid in sorted(label_counter.keys())]):
            mean_score = np.mean(avg_scores[:, idx])
            print(f"类别 {cat_name}: 平均分数={mean_score:.4f}")
        print("")

        if (epoch + 1) % 20 == 0:  # 调整保存频率
            save_path = f'{args.output_dir}/standard_model_epoch_{epoch+1}.pkl'
            jt.save(model.state_dict(), save_path)
            print(f"✅ 模型已保存: {save_path}")

    final_save_path = f'{args.output_dir}/standard_model_final.pkl'
    jt.save(model.state_dict(), final_save_path)
    print(f"🎉 最终模型已保存: {final_save_path}")
    print("🎉 训练完成！")


if __name__ == "__main__":
    main()
