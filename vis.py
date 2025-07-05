import jittor as jt
from model import RTDETR
from PIL import Image, ImageDraw
import numpy as np
import os


def preprocess(img_path):
    img = Image.open(img_path).convert('RGB').resize((320, 320))
    arr = np.array(img).astype(np.float32) / 255.
    arr = (arr - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    arr = arr.transpose(2, 0, 1)
    return jt.array(arr).unsqueeze(0)


def main():
    model = RTDETR()
    model.load_parameters('your_trained_model.pkl')  # 替换为实际模型文件
    model.eval()
    img_path = 'test.jpg'  # 替换为实际图片
    img_tensor = preprocess(img_path)
    logits, boxes = model(img_tensor)
    boxes = boxes[0].numpy() * 320  # 恢复到原图尺寸
    logits = logits[0].numpy()
    scores = logits.max(axis=1)
    labels = logits.argmax(axis=1)
    img = Image.open(img_path).convert('RGB').resize((320, 320))
    draw = ImageDraw.Draw(img)
    for i in range(len(boxes)):
        if scores[i] > 0.5:
            x1, y1, x2, y2 = boxes[i]
            draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
            draw.text((x1, y1), str(labels[i]), fill='red')
    img.save('vis_result.jpg')
    print('可视化结果已保存为 vis_result.jpg')


if __name__ == '__main__':
    main()
