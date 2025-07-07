import os
import torch
from torchvision import transforms
from PIL import Image, ImageDraw
import glob

# 假设模型加载和推理接口如下（需根据实际项目适配）
from src.core import YAMLConfig
from src.solver import TASKS


def load_latest_checkpoint(output_dir):
    # 优先查找eval/latest.pth软链接
    eval_latest = os.path.join(
        output_dir, 'rtdetr_r18vd_6x_coco', 'rtdetr_r18vd_6x_coco', 'eval', 'latest.pth')
    if os.path.exists(eval_latest):
        return eval_latest
    # 再查找上一级latest.pth
    latest = os.path.join(output_dir, 'rtdetr_r18vd_6x_coco',
                          'rtdetr_r18vd_6x_coco', 'latest.pth')
    if os.path.exists(latest):
        return latest
    # 否则递归查找所有常见后缀和命名
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


def get_model_from_solver(solver):
    if hasattr(solver, 'model') and solver.model is not None:
        return solver.model
    if hasattr(solver, '_model') and solver._model is not None:
        return solver._model
    if hasattr(solver, 'get_model') and callable(solver.get_model):
        m = solver.get_model()
        if m is not None:
            return m
    if hasattr(solver, 'cfg') and hasattr(solver.cfg, 'model') and solver.cfg.model is not None:
        return solver.cfg.model
    raise AttributeError('无法在DetSolver中找到模型对象')


def main():
    # 配置和权重路径
    config_path = 'pytorch_rt_detr/configs/rtdetr/rtdetr_r18vd_6x_coco.yml'
    output_dir = 'pytorch_rt_detr/output'
    image_path = '@test.png'
    result_path = 'pytorch_rt_detr/output/test_result.png'

    # 加载配置和模型
    cfg = YAMLConfig(config_path)
    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    # 加载最新权重
    ckpt = load_latest_checkpoint(output_dir)
    print(f'加载权重: {ckpt}')
    model = get_model_from_solver(solver)
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    model.eval()

    # 读取图片
    img = Image.open(image_path).convert('RGB')
    # 预处理（需与训练一致）
    preprocess = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(img).unsqueeze(0)

    # 推理
    with torch.no_grad():
        outputs = model(input_tensor)
        # 假设输出格式为dict，包含boxes、labels、scores
        # 需根据实际模型输出适配
        boxes = outputs['boxes'] if 'boxes' in outputs else outputs[0]['boxes']
        scores = outputs['scores'] if 'scores' in outputs else outputs[0]['scores']
        labels = outputs['labels'] if 'labels' in outputs else outputs[0]['labels']

    # 可视化
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
