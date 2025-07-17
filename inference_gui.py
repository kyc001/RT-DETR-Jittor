#!/usr/bin/env python3
"""
RT-DETR Model Inference & Visualization GUI - Final Corrected Version
Author: Gemini
Features:
- Load a pre-trained Jittor model.
- Select an image for object detection.
- Compare predictions with COCO ground truth annotations using a unified and robust class mapping.
"""
import os
import sys
import json
import numpy as np
from PIL import Image, ImageTk
import matplotlib
matplotlib.use('TkAgg') # Specify Matplotlib backend for Tkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- GUI Libraries ---
import tkinter as tk
from tkinter import filedialog, messagebox

# --- Jittor and Project Imports ---
# Add project path (please ensure this is correct)
sys.path.insert(0, '/home/kyc/project/RT-DETR')
import jittor as jt
import jittor.nn as nn

# Import model definitions from your project
try:
    from jittor_rt_detr.src.nn.backbone.resnet import ResNet50
    from jittor_rt_detr.src.zoo.rtdetr.rtdetr_decoder import RTDETRTransformer
except ImportError:
    print("❌ Error: Could not import model definition from jittor_rt_detr.src.")
    print("   Please ensure the script is in the project root or sys.path is set correctly.")
    sys.exit(1)

# --- Jittor Settings ---
jt.flags.use_cuda = 1
jt.set_global_seed(42)

# --- Core Logic ---

class RTDETRModel(nn.Module):
    """ RT-DETR Model Wrapper """
    def __init__(self, num_classes, pretrained_backbone=False):
        super().__init__()
        self.backbone = ResNet50(pretrained=pretrained_backbone)
        self.transformer = RTDETRTransformer(
            num_classes=num_classes,
            hidden_dim=256,
            num_queries=300,
            feat_channels=[256, 512, 1024, 2048]
        )
    def execute(self, x):
        features = self.backbone(x)
        return self.transformer(features)

def nms_filter(boxes, scores, classes, iou_threshold=0.5, score_threshold=0.3):
    """ Simple NMS filter to remove overlapping detections. """
    if len(boxes) == 0:
        return [], [], []
    sorted_indices = np.argsort(scores)[::-1]
    keep_indices = []
    for i in sorted_indices:
        if scores[i] < score_threshold:
            continue
        keep = True
        for j in keep_indices:
            box1, box2 = boxes[i], boxes[j]
            x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
            x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
            if x2 > x1 and y2 > y1:
                intersection = (x2 - x1) * (y2 - y1)
                area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
                union = area1 + area2 - intersection
                iou = intersection / union if union > 0 else 0
                if iou > iou_threshold:
                    keep = False
                    break
        if keep:
            keep_indices.append(i)
    return [boxes[i] for i in keep_indices], [scores[i] for i in keep_indices], [classes[i] for i in keep_indices]


class InferenceApp(tk.Tk):
    """Main application GUI class"""
    def __init__(self):
        super().__init__()
        self.title("RT-DETR Inference & Visualization Tool")
        self.geometry("1200x800")

        # --- Data Path Variables ---
        self.model_path = tk.StringVar()
        self.image_path = tk.StringVar()
        self.ann_path = tk.StringVar()
        self.coco_data = None
        
        # --- Unified Class Mapping Dictionaries ---
        self.cat_id_to_contiguous_id = {}
        self.contiguous_id_to_name = {}
        
        # --- Matplotlib Settings ---
        matplotlib.rcParams['axes.unicode_minus'] = False
        # Set a default font that is likely to exist.
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']

        # --- UI Layout ---
        main_frame = tk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        control_frame = tk.Frame(main_frame)
        control_frame.pack(fill=tk.X)
        self.result_frame = tk.Frame(main_frame, relief=tk.SUNKEN, borderwidth=1)
        self.result_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        self.canvas = None
        self.status_var = tk.StringVar()
        self.status_var.set("Welcome! Please select a model, an image, and an annotation file.")
        status_bar = tk.Label(self, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # --- Control Widgets ---
        tk.Button(control_frame, text="1. Select Model (.pkl)", command=self.select_model).pack(side=tk.LEFT, padx=5)
        tk.Label(control_frame, textvariable=self.model_path, fg="blue").pack(side=tk.LEFT)
        tk.Button(control_frame, text="2. Select Image", command=self.select_image).pack(side=tk.LEFT, padx=5)
        tk.Label(control_frame, textvariable=self.image_path, fg="blue").pack(side=tk.LEFT)
        tk.Button(control_frame, text="3. Select Annotation (.json)", command=self.select_annotations).pack(side=tk.LEFT, padx=5)
        tk.Label(control_frame, textvariable=self.ann_path, fg="blue").pack(side=tk.LEFT)
        tk.Button(control_frame, text="Run Inference", command=self.run_inference, bg="green", fg="white", font=('Helvetica', 10, 'bold')).pack(side=tk.RIGHT, padx=20)
        
    def select_model(self):
        path = filedialog.askopenfilename(title="Select Model File", filetypes=[("Jittor Models", "*.pkl")])
        if path:
            self.model_path.set(os.path.basename(path))
            self.model_full_path = path
            self.status_var.set("Model selected.")

    def select_image(self):
        path = filedialog.askopenfilename(title="Select Image File", filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            self.image_path.set(os.path.basename(path))
            self.image_full_path = path
            self.status_var.set("Image selected.")

    def select_annotations(self):
        path = filedialog.askopenfilename(title="Select COCO Annotation File", filetypes=[("JSON files", "*.json")])
        if path:
            self.ann_path.set(os.path.basename(path))
            self.ann_full_path = path
            self.status_var.set("Loading annotations...")
            try:
                with open(path, 'r') as f:
                    self.coco_data = json.load(f)
                
                # --- 🔧 FIX: Unified and robust class mapping generation ---
                # This logic is now the single source of truth for class mapping.
                categories = sorted(self.coco_data.get('categories', []), key=lambda x: x['id'])
                
                self.cat_id_to_contiguous_id = {cat['id']: i for i, cat in enumerate(categories)}
                self.contiguous_id_to_name = {i: cat['name'] for i, cat in enumerate(categories)}
                # --- End of fix ---

                self.status_var.set("Annotations loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load annotation file: {e}")
                self.status_var.set("Failed to load annotations!")

    def run_inference(self):
        if not all([self.model_path.get(), self.image_path.get(), self.ann_path.get()]):
            messagebox.showwarning("Input Incomplete", "Please ensure a model, an image, and an annotation file are selected.")
            return
            
        self.status_var.set("Preparing model...")
        self.update_idletasks()

        try:
            checkpoint = jt.load(self.model_full_path)
            num_classes = checkpoint.get('num_classes', 80)
            model = RTDETRModel(num_classes=num_classes)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            self.status_var.set("Model loaded. Running inference...")
            self.update_idletasks()

            image = Image.open(self.image_full_path).convert('RGB')
            original_width, original_height = image.size
            image_resized = image.resize((640, 640), Image.LANCZOS)
            img_array = np.array(image_resized).astype(np.float32) / 255.0
            img_tensor = jt.array(img_array.transpose(2, 0, 1)).float32().unsqueeze(0)

            with jt.no_grad():
                outputs = model(img_tensor)

            pred_logits = outputs['pred_logits'][0]
            pred_boxes = outputs['pred_boxes'][0]
            pred_scores = jt.nn.softmax(pred_logits, dim=-1)

            max_scores_result = jt.max(pred_scores[:, :-1], dim=-1)
            pred_scores_no_bg = max_scores_result[0] if isinstance(max_scores_result, tuple) else max_scores_result

            pred_classes_result = jt.argmax(pred_scores[:, :-1], dim=-1)
            pred_classes = pred_classes_result[0] if isinstance(pred_classes_result, tuple) else pred_classes_result
            
            gt_boxes, gt_classes = self.get_ground_truth(os.path.basename(self.image_full_path), original_width, original_height)
            
            self.display_results(
                image,
                pred_scores_no_bg.numpy(),
                pred_classes.numpy(),
                pred_boxes.numpy(),
                gt_boxes,
                gt_classes
            )
            self.status_var.set("Inference complete!")

        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Inference Failed", f"An error occurred: {e}")
            self.status_var.set(f"Error: {e}")
    
    def get_ground_truth(self, image_filename, width, height):
        img_id = None
        for img_info in self.coco_data['images']:
            if img_info['file_name'] == image_filename:
                img_id = img_info['id']
                break
        
        if img_id is None:
            return [], []

        gt_boxes, gt_classes = [], []
        for ann in self.coco_data.get('annotations', []):
            if ann['image_id'] == img_id:
                x, y, w, h = ann['bbox']
                x1, y1 = x / width, y / height
                x2, y2 = (x + w) / width, (y + h) / height
                if x2 > x1 and y2 > y1:
                    gt_boxes.append([x1, y1, x2, y2])
                    # --- 🔧 FIX: Use the unified mapping dictionary ---
                    contiguous_id = self.cat_id_to_contiguous_id.get(ann['category_id'])
                    if contiguous_id is not None:
                        gt_classes.append(contiguous_id)

        return gt_boxes, gt_classes

    def display_results(self, original_image, pred_scores, pred_classes, pred_boxes_norm, gt_boxes_norm, gt_classes):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        original_width, original_height = original_image.size

        # --- Left Panel: Ground Truth ---
        ax1.imshow(original_image)
        ax1.set_title('Ground Truth', fontsize=14, fontweight='bold')
        ax1.axis('off')
        if not gt_boxes_norm:
             ax1.text(0.5, 0.5, 'No Ground Truth Annotations', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, fontsize=15, color='gray')
        for box, cls_idx in zip(gt_boxes_norm, gt_classes):
            x1, y1, x2, y2 = [coord * dim for coord, dim in zip(box, [original_width, original_height, original_width, original_height])]
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='g', facecolor='none')
            ax1.add_patch(rect)
            # --- 🔧 FIX: Use the unified mapping dictionary ---
            class_name = self.contiguous_id_to_name.get(cls_idx, f'ID:{cls_idx}')
            ax1.text(x1, y1 - 2, class_name, bbox=dict(facecolor='green', alpha=0.7), color='white', fontsize=9)

        # --- Right Panel: Model Predictions ---
        ax2.imshow(original_image)
        ax2.set_title('Model Predictions', fontsize=14, fontweight='bold')
        ax2.axis('off')

        pred_boxes_pixel = [[coord * dim for coord, dim in zip(box, [original_width, original_height, original_width, original_height])] for box in pred_boxes_norm]
        
        filtered_boxes, filtered_scores, filtered_classes = nms_filter(pred_boxes_pixel, pred_scores, pred_classes)
        
        if not filtered_boxes:
            ax2.text(0.5, 0.5, 'No objects detected', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes, fontsize=15, color='gray')
            
        color_map = plt.cm.get_cmap('tab20', max(20, len(self.contiguous_id_to_name)))
        
        for box, score, cls_idx in zip(filtered_boxes, filtered_scores, filtered_classes):
            x1, y1, x2, y2 = box
            color = color_map(cls_idx % 20)
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor='none')
            ax2.add_patch(rect)
            # --- 🔧 FIX: Use the unified mapping dictionary ---
            class_name = self.contiguous_id_to_name.get(cls_idx, f'ID:{cls_idx}')
            label = f'{class_name}: {score:.2f}'
            ax2.text(x1, y1 - 2, label, bbox=dict(facecolor=color, alpha=0.7), color='white', fontsize=9)

        fig.tight_layout()
        
        self.canvas = FigureCanvasTkAgg(fig, master=self.result_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    app = InferenceApp()
    app.mainloop()