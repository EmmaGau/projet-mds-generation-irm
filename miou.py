import numpy as np
from PIL import Image
import os
import cv2
import argparse

class_colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 152, 0], [255, 152, 0], [0, 255, 237]]

def calculate_iou(gt_mask, pred_mask, num_classes):
    iou_sum = 0.0
    for class_color in class_colors:  # Class IDs start from 1
        gt_class_mask = (gt_mask == class_color).astype(np.uint8)
        pred_class_mask = (pred_mask == class_color).astype(np.uint8)
        # resize the mask to 64xx64
        gt_class_mask = cv2.resize(gt_class_mask, (64, 64))

        intersection = np.logical_and(gt_class_mask, pred_class_mask)
        union = np.logical_or(gt_class_mask, pred_class_mask)
        
        iou = np.sum(intersection) / (np.sum(union) + 1e-10)  # Add a small epsilon to avoid division by zero
        iou_sum += iou
    return iou_sum / num_classes

def calculate_miou(gt_masks, pred_masks, num_classes):
    total_iou = 0.0
    num_samples = len(gt_masks)

    for i in range(num_samples):
        gt_mask = np.array(Image.open(gt_masks[i]))
        pred_mask = np.array(Image.open(pred_masks[i]))

        iou = calculate_iou(gt_mask, pred_mask, num_classes)
        total_iou += iou

    return total_iou / num_samples

parser = argparse.ArgumentParser()
parser.add_argument('gt_folder')
parser.add_argument('results_folder')

args = parser.parse_args()
gt_mask_paths = [os.path.join(args.gt_folder, file_name) for file_name in os.listdir(args.gt_folder)]
pred_mask_paths = [os.path.join(args.results_folder, file_name) for file_name in os.listdir(args.results_folder)]
num_classes = 6

miou = calculate_miou(gt_mask_paths, pred_mask_paths, num_classes)
print(f"mIoU: {miou}")
