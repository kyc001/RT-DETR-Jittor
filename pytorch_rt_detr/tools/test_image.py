import os
import torch
from torchvision import transforms
from PIL import Image, ImageDraw
import glob

from src.core import YAMLConfig
from src.solver import TASKS


def load_latest_checkpoint(output_dir):
    # 支持递归查找所有常见后缀和命名
    patterns = [
        '**/model_*.pth', '**/model_*.pt', '**/model_*.pkl',
        '**/checkpoint*.pth', '**/checkpoint*.pt', '**/checkpoint*.pkl',
        '**/epoch*.pth', '**/epoch*.pt', '**/epoch*.pkl'
    ]
    ckpts = []
    for pat in patterns:
        ckpts.extend(glob.glob(os.path.join(output_dir, pat), recursive=True))
    if not ckpts:
        raise FileNotFoundError('未找到权重文件')
    ckpts = sorted(ckpts, key=os.path.getmtime)
    return ckpts[-1]


def main():
    config_path = 'pytorch_rt_detr/configs/rtdetr/rtdetr_r18vd_6x_coco.yml'
    output_dir = 'pytorch_rt_detr/output'
    image_path = '@test.png'
    result_path = 'pytorch_rt_detr/output/test_result.png'

    cfg = YAMLConfig(config_path)
    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    ckpt = load_latest_checkpoint(output_dir)
    print(f'加载权重: {ckpt}')
    solver.model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    solver.model.eval()

    img = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        outputs = solver.model(input_tensor)
        boxes = outputs['boxes'] if 'boxes' in outputs else outputs[0]['boxes']
        scores = outputs['scores'] if 'scores' in outputs else outputs[0]['scores']
        labels = outputs['labels'] if 'labels' in outputs else outputs[0]['labels']

    draw = ImageDraw.Draw(img)
    for box, score in zip(boxes, scores):
        if score < 0.3:
            continue
        box = [float(x) for x in box]
        draw.rectangle(box, outline='red', width=2)
    img.save(result_path)
    print(f'检测结果已保存到: {result_path}')


if __name__ == '__main__':
    main()
