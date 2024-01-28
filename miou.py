import numpy as np
from PIL import Image
import os
import cv2
import argparse

categories = {
    'Email': ([255, 0, 0], 1),
    'Os': ([0, 255, 0], 2),
    'Dentine': ([0, 0, 255], 3),
    'Autre': ([255, 152, 0], 4),
    'Carie': ([255, 152, 0], 5),
    'Pulpe': ([0, 255, 237], 6)
}

def calculate_iou(gt_mask, pred_mask):
    intersection = np.logical_and(gt_mask, pred_mask)
    union = np.logical_or(gt_mask, pred_mask)
    iou = np.sum(intersection) / (np.sum(union) + 1e-10)  # Add a small epsilon to avoid division by zero
    return iou

def calculate_miou(gt_masks, pred_masks, categories):
    num_classes = len(categories)
    total_iou = 0.0
    class_iou = np.zeros(num_classes)
    num_samples = len(gt_masks)

    for i in range(num_samples):
        gt_mask = np.array(Image.open(gt_masks[i]))
        pred_mask = np.array(Image.open(pred_masks[i]))

        sample_iou = 0.0
        for class_name, (class_color, class_id) in categories.items():
            gt_class_mask = (gt_mask == class_color).astype(np.uint8)
            pred_class_mask = (pred_mask == class_color).astype(np.uint8)
            gt_class_mask = cv2.resize(gt_class_mask.astype(np.uint8), (64, 64))
            pred_class_mask = cv2.resize(pred_class_mask.astype(np.uint8), (64, 64))

            iou = calculate_iou(gt_class_mask, pred_class_mask)
            sample_iou += iou
            class_iou[class_id - 1] += iou

        total_iou += sample_iou / num_classes

    mIoU = total_iou / num_samples
    class_iou /= num_samples

    return mIoU, class_iou

parser = argparse.ArgumentParser()
parser.add_argument('gt_folder')
parser.add_argument('results_folder')

args = parser.parse_args()
gt_mask_paths = [os.path.join(args.gt_folder, file_name) for file_name in os.listdir(args.gt_folder)]
pred_mask_paths = [os.path.join(args.results_folder, file_name) for file_name in os.listdir(args.results_folder)]

miou, class_iou = calculate_miou(gt_mask_paths, pred_mask_paths, categories)
print(f"mIoU: {miou}")
print("mIoU per class:")
for class_name, (_, class_id) in categories.items():
    print(f"{class_name}: {class_iou[class_id - 1]}")
