import json

# 请确保这个路径是正确的
ann_file = 'data/coco/annotations/instances_val2017.json'

with open(ann_file, 'r') as f:
    data = json.load(f)

annotations = data['annotations']
images = data['images']

# 创建一个从 image_id 到其标注数量的映射
img_id_to_ann_count = {}
for ann in annotations:
    img_id = ann['image_id']
    img_id_to_ann_count[img_id] = img_id_to_ann_count.get(img_id, 0) + 1

# 按标注数量降序排序
sorted_img_ids = sorted(img_id_to_ann_count.items(),
                        key=lambda item: item[1], reverse=True)

# 打印出标注最多的前5张图片的信息
id2name = {img['id']: img['file_name'] for img in images}
print("以下是一些包含较多物体的图片，请选择一两个用于测试：")
for img_id, count in sorted_img_ids[:5]:
    print(
        f"  - 图片ID (Image ID): {img_id}, 文件名 (File Name): {id2name[img_id]}, 物体数量: {count}")
