#!/usr/bin/env python3
"""
调试类别映射问题
"""

# COCO类别名称映射
COCO_CLASSES = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
    21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe',
    27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
    34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
    39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard',
    43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup',
    48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana',
    53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot',
    58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair',
    63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table',
    70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote',
    76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
    80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book',
    85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear',
    89: 'hair drier', 90: 'toothbrush'
}

def convert_to_coco_class(class_idx):
    """将模型类别索引转换为COCO类别ID和名称"""
    # 基于训练时的映射
    if class_idx == 0:
        coco_id = 1  # person
    elif class_idx == 2:
        coco_id = 3  # car
    elif class_idx == 26:
        coco_id = 27  # backpack
    elif class_idx == 32:
        coco_id = 33  # suitcase
    elif class_idx == 83:
        coco_id = 84  # book
    else:
        coco_id = class_idx + 1
    
    class_name = COCO_CLASSES.get(coco_id, f'class_{coco_id}')
    return coco_id, class_name

def main():
    print("🔍 类别映射调试")
    print("=" * 50)
    
    # 测试模型预测的类别0
    print("📊 测试类别映射:")
    
    test_classes = [0, 1, 2, 26, 32, 66, 67, 83]
    
    for class_idx in test_classes:
        coco_id, class_name = convert_to_coco_class(class_idx)
        print(f"   类别{class_idx} -> COCO ID {coco_id} -> {class_name}")
    
    print(f"\n🎯 关键发现:")
    print(f"   模型预测类别0 -> COCO ID 1 -> {COCO_CLASSES[1]}")
    print(f"   但是如果类别66 -> COCO ID 67 -> {COCO_CLASSES[67]}")
    
    # 检查为什么会显示dining table
    print(f"\n🔍 检查dining table:")
    print(f"   COCO ID 67 = {COCO_CLASSES[67]}")
    print(f"   对应模型类别索引 = 67 - 1 = 66")

if __name__ == "__main__":
    main()
