#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COCO 数据集准备脚本
自动下载和准备 COCO 2017 数据集用于 RT-DETR 训练
"""

import os
import zipfile
import requests
import argparse
from tqdm import tqdm
import json


class COCODataPreparer:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.coco_dir = os.path.join(data_dir, "coco")
        self.annotations_dir = os.path.join(self.coco_dir, "annotations")
        self.train_dir = os.path.join(self.coco_dir, "train2017")
        self.val_dir = os.path.join(self.coco_dir, "val2017")

        # COCO 数据集下载链接
        self.download_urls = {
            'train2017': 'http://images.cocodataset.org/zips/train2017.zip',
            'val2017': 'http://images.cocodataset.org/zips/val2017.zip',
            'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
        }

        # 文件大小 (用于进度条)
        self.file_sizes = {
            'train2017': 18 * 1024 * 1024 * 1024,  # ~18GB
            'val2017': 1 * 1024 * 1024 * 1024,     # ~1GB
            'annotations': 241 * 1024 * 1024        # ~241MB
        }

    def create_directories(self):
        """创建必要的目录结构"""
        directories = [
            self.data_dir,
            self.coco_dir,
            self.annotations_dir,
            self.train_dir,
            self.val_dir
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"创建目录: {directory}")

    def download_file(self, url, filename, expected_size=None):
        """下载文件并显示进度"""
        filepath = os.path.join(self.data_dir, filename)

        # 检查文件是否已存在
        if os.path.exists(filepath):
            print(f"文件已存在: {filepath}")
            return filepath

        print(f"开始下载: {filename}")
        print(f"下载地址: {url}")

        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            if expected_size:
                total_size = expected_size

            with open(filepath, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            print(f"下载完成: {filepath}")
            return filepath

        except Exception as e:
            print(f"下载失败: {e}")
            if os.path.exists(filepath):
                os.remove(filepath)
            return None

    def extract_zip(self, zip_path, extract_dir):
        """解压 ZIP 文件"""
        print(f"解压文件: {zip_path}")
        print(f"解压到: {extract_dir}")

        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                total_files = len(zip_ref.namelist())
                with tqdm(total=total_files, desc="解压进度") as pbar:
                    for file in zip_ref.namelist():
                        zip_ref.extract(file, extract_dir)
                        pbar.update(1)

            print(f"解压完成: {extract_dir}")
            return True

        except Exception as e:
            print(f"解压失败: {e}")
            return False

    def verify_dataset(self):
        """验证数据集完整性"""
        print("验证数据集完整性...")

        # 检查必要的文件
        required_files = [
            os.path.join(self.annotations_dir, "instances_train2017.json"),
            os.path.join(self.annotations_dir, "instances_val2017.json")
        ]

        for file_path in required_files:
            if not os.path.exists(file_path):
                print(f"缺少文件: {file_path}")
                return False

        # 检查图片目录
        train_count = len([f for f in os.listdir(
            self.train_dir) if f.endswith('.jpg')])
        val_count = len([f for f in os.listdir(
            self.val_dir) if f.endswith('.jpg')])

        print(f"训练图片数量: {train_count}")
        print(f"验证图片数量: {val_count}")

        # 检查标注文件
        with open(os.path.join(self.annotations_dir, "instances_train2017.json"), 'r') as f:
            train_ann = json.load(f)

        with open(os.path.join(self.annotations_dir, "instances_val2017.json"), 'r') as f:
            val_ann = json.load(f)

        print(f"训练标注数量: {len(train_ann['annotations'])}")
        print(f"验证标注数量: {len(val_ann['annotations'])}")
        print(f"类别数量: {len(train_ann['categories'])}")

        return True

    def create_subset(self, subset_size=100):
        """创建数据集子集用于快速实验"""
        print(f"创建数据集子集 (大小: {subset_size})...")

        # 读取验证集标注
        val_ann_file = os.path.join(
            self.annotations_dir, "instances_val2017.json")
        with open(val_ann_file, 'r') as f:
            val_ann = json.load(f)

        # 选择前 subset_size 张图片
        subset_images = val_ann['images'][:subset_size]
        subset_image_ids = {img['id'] for img in subset_images}

        # 筛选对应的标注
        subset_annotations = [
            ann for ann in val_ann['annotations']
            if ann['image_id'] in subset_image_ids
        ]

        # 创建子集标注文件
        subset_ann = {
            'images': subset_images,
            'annotations': subset_annotations,
            'categories': val_ann['categories']
        }

        subset_ann_file = os.path.join(
            self.annotations_dir, f"instances_val2017_subset_{subset_size}.json")
        with open(subset_ann_file, 'w') as f:
            json.dump(subset_ann, f, indent=2)

        print(f"子集标注文件已创建: {subset_ann_file}")
        print(f"子集图片数量: {len(subset_images)}")
        print(f"子集标注数量: {len(subset_annotations)}")

        return subset_ann_file

    def prepare_full_dataset(self, download=True):
        """准备完整数据集"""
        print("开始准备 COCO 数据集...")

        # 创建目录
        self.create_directories()

        if download:
            # 下载文件
            downloaded_files = {}
            for name, url in self.download_urls.items():
                filename = f"{name}.zip"
                expected_size = self.file_sizes.get(name)
                filepath = self.download_file(url, filename, expected_size)
                if filepath:
                    downloaded_files[name] = filepath
                else:
                    print(f"下载失败: {name}")
                    return False

            # 解压文件
            for name, filepath in downloaded_files.items():
                if name == 'train2017':
                    extract_dir = self.data_dir
                elif name == 'val2017':
                    extract_dir = self.data_dir
                elif name == 'annotations':
                    extract_dir = self.data_dir

                if not self.extract_zip(filepath, extract_dir):
                    print(f"解压失败: {name}")
                    return False

                # 删除压缩文件以节省空间
                os.remove(filepath)
                print(f"已删除压缩文件: {filepath}")

        # 验证数据集
        if not self.verify_dataset():
            print("数据集验证失败")
            return False

        print("数据集准备完成!")
        return True

    def prepare_subset_dataset(self, subset_size=100):
        """准备数据集子集"""
        print(f"准备数据集子集 (大小: {subset_size})...")

        # 检查完整数据集是否存在
        if not self.verify_dataset():
            print("完整数据集不存在，请先运行完整数据集准备")
            return False

        # 创建子集
        subset_ann_file = self.create_subset(subset_size)

        print("数据集子集准备完成!")
        print(f"子集标注文件: {subset_ann_file}")
        return True


def main():
    parser = argparse.ArgumentParser(description="COCO 数据集准备工具")
    parser.add_argument('--data_dir', type=str, default='data',
                        help='数据目录路径')
    parser.add_argument('--download', action='store_true',
                        help='是否下载数据集文件')
    parser.add_argument('--subset_size', type=int, default=100,
                        help='子集大小 (用于快速实验)')
    parser.add_argument('--subset_only', action='store_true',
                        help='仅创建子集，不下载完整数据集')

    args = parser.parse_args()

    preparer = COCODataPreparer(args.data_dir)

    if args.subset_only:
        # 仅创建子集
        success = preparer.prepare_subset_dataset(args.subset_size)
    else:
        # 准备完整数据集
        success = preparer.prepare_full_dataset(args.download)

        if success and args.subset_size > 0:
            # 创建子集
            preparer.prepare_subset_dataset(args.subset_size)

    if success:
        print("\n数据集准备完成!")
        print(f"数据目录: {args.data_dir}")
        print("可以开始训练了!")
    else:
        print("\n数据集准备失败!")
        print("请检查网络连接和磁盘空间")


if __name__ == '__main__':
    main()
