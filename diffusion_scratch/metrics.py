import numpy as np
from PIL import Image
import os
import cv2
import argparse

categories = {
'Email' : ([255,0,0], 1),
'Os' : ([0,255,0], 2),
'Dentine': ([0,0,255], 3),
'Autre' : ([255, 0, 254], 4),
'Carie': ([255,152,0], 5),
'Pulpe': ([0, 255, 237], 6)
}

def calculate_iou(gt_mask, pred_mask):
    intersection = np.logical_and(gt_mask, pred_mask)
    union = np.logical_or(gt_mask, pred_mask)
    iou = np.sum(intersection) / (np.sum(union) + 1e-10)  # Add a small epsilon to avoid division by zero
    return iou


def calculate_recall(gt_mask, pred_mask):
    intersection = np.logical_and(gt_mask, pred_mask)
    recall = np.sum(intersection) / (np.sum(gt_mask) + 1e-10)  # Add a small epsilon to avoid division by zero
    return recall


def calculate_precision(gt_mask, pred_mask):
    intersection = np.logical_and(gt_mask, pred_mask)
    precision = np.sum(intersection) / (np.sum(pred_mask) + 1e-10)  # Add a small epsilon to avoid division by zero
    return precision


def calculate_metrics(gt_masks, pred_masks, categories):
    num_classes = len(categories)
    num_samples = len(gt_masks)
    total_iou = 0.0
    class_iou = np.zeros(num_classes)
    total_recall = 0.0
    total_precision = 0.0
    class_recall = np.zeros(num_classes)
    class_precision = np.zeros(num_classes)

    for i in range(num_samples):
        gt_mask = np.array(Image.open(gt_masks[i]))
        pred_mask = np.array(Image.open(pred_masks[i]))

        image_size = pred_mask.shape[0]

        sample_iou = 0.0
        sample_recall = 0.0
        sample_precision = 0.0

        for class_name, (class_color, class_id) in categories.items():
            gt_class_mask = (gt_mask == class_color).astype(np.uint8)
            pred_class_mask = (pred_mask == class_color).astype(np.uint8)
            gt_class_mask = cv2.resize(gt_class_mask.astype(np.uint8), (image_size, image_size))
            pred_class_mask = cv2.resize(pred_class_mask.astype(np.uint8), (image_size, image_size))

            iou = calculate_iou(gt_class_mask, pred_class_mask)
            sample_iou += iou
            class_iou[class_id - 1] += iou

            recall = calculate_recall(gt_class_mask, pred_class_mask)
            sample_recall += recall
            class_recall[class_id - 1] += recall

            precision = calculate_precision(gt_class_mask, pred_class_mask)
            sample_precision += precision
            class_precision[class_id - 1] += precision

        total_iou += sample_iou / num_classes
        total_recall += sample_recall / num_classes
        total_precision += sample_precision / num_classes

    mIoU = total_iou / num_samples
    class_iou /= num_samples
    total_recall /= num_samples
    class_recall /= num_samples
    total_precision /= num_samples
    class_precision /= num_samples

    return mIoU, class_iou, total_recall, class_recall, total_precision, class_precision


parser = argparse.ArgumentParser()
parser.add_argument('gt_folder')
parser.add_argument('results_folder')

args = parser.parse_args()
gt_mask_paths = [os.path.join(args.gt_folder, file_name) for file_name in os.listdir(args.gt_folder)]
pred_mask_paths = [os.path.join(args.results_folder, file_name) for file_name in os.listdir(args.results_folder)]

miou, class_iou, recall, class_recall, precision, class_precision = calculate_metrics(gt_mask_paths, pred_mask_paths, categories)

# Print a table with 3 digits of precision
print('        |  mIoU  | Recall | Precision')
print('Total   |  {:.2f}  |  {:.2f}  |  {:.2f}'.format(miou, recall, precision))
for class_name, (_, class_id) in categories.items():
    print('{:<8}|  {:.2f}  |  {:.2f}  |  {:.2f}'.format(class_name, class_iou[class_id - 1], class_recall[class_id - 1], class_precision[class_id - 1]))