import cv2
import numpy as np
import os
import labelme
from pre_detection import *
from pre_segmentation import *
import matplotlib.pyplot as plt

class MetricsRaiza:
    def __init__(self):
        # ------------ Initialization ------------ #
        self.seg = SegmModel()
        self.json_dir = os.path.join('./dataset/Sorted Data/', 'Json_files')
        self.json_listdir = []
        self.actual_json_idx = 0

    # ------------ PREPROCESSING ------------ #
    def json_idx(self):
        """Generate a list with the JSON files paths."""
        self.json_files_listdir = os.listdir(self.json_dir)
        for filename in self.json_files_listdir:
            file_path = os.path.join(self.json_dir, filename)
            self.json_listdir.append(file_path)

    def json_prev(self):
        """Iterate to the previous JSON file."""
        try:
            if self.json_listdir:
                self.actual_json_idx = (self.actual_json_idx - 1) % len(self.json_listdir)
        except Exception as e:
            print(f"Error: {e}")

    def json_next(self):
        """Iterate to the next JSON file."""
        try:
            if self.json_listdir:
                self.actual_json_idx = (self.actual_json_idx + 1) % len(self.json_listdir)
        except Exception as e:
            print(f"Error: {e}")

    # ------------ IOU ------------ #
    def GT_xyxybox(self):
        """Obtain bounding box coordinates from the current JSON file."""
        class_names = ('left normal', 'right normal')
        filename = self.json_listdir[self.actual_json_idx]
        label_file = labelme.LabelFile(filename=filename)
        labelme.utils.img_data_to_arr(label_file.imageData)

        self.bboxes = []
        for shape in label_file.shapes:
            if shape["label"] in class_names:
                class_id = class_names.index(shape["label"])
                sortingForObj = np.asarray(shape["points"])
                xmin, xmax = sorted([np.min(sortingForObj[:, 0]), np.max(sortingForObj[:, 0])])
                ymin, ymax = sorted([np.min(sortingForObj[:, 1]), np.max(sortingForObj[:, 1])])
                self.bboxes.append([xmin, ymin, xmax, ymax])
        return self.bboxes

    def calculate_iou(self, bbox_gt, bbox_pred):
        """Calculate Intersection over Union (IoU)."""
        x1_inter = max(bbox_gt[0], bbox_pred[0])
        y1_inter = max(bbox_gt[1], bbox_pred[1])
        x2_inter = min(bbox_gt[2], bbox_pred[2])
        y2_inter = min(bbox_gt[3], bbox_pred[3])

        area_inter = abs(x2_inter - x1_inter) * abs(y2_inter - y1_inter)
        area_gt = abs(bbox_gt[2] - bbox_gt[0]) * abs(bbox_gt[3] - bbox_gt[1])
        area_pred = abs(bbox_pred[2] - bbox_pred[0]) * abs(bbox_pred[3] - bbox_pred[1])

        iou = area_inter / float(area_gt + area_pred - area_inter)
        return iou

    def IOU(self, bboxes_gt, bboxes_pred):
        """Calculate the average IoU for multiple bounding boxes."""
        iou = iou_2 = 0.0
        if len(bboxes_gt) > 0 and len(bboxes_pred) > 0:
            iou = self.calculate_iou(bboxes_gt[0], bboxes_pred[0])
        if len(bboxes_gt) > 1 and len(bboxes_pred) > 1:
            iou_2 = self.calculate_iou(bboxes_gt[1], bboxes_pred[1])
        return round(abs((iou + iou_2) / 2), 6)

    # ------------ DICE COEFFICIENT ------------ #
    def round_yolo_coors(self, pred_coors):
        """Round YOLO coordinates and generate binary masks."""
        rounded_yolov8_coordinates = [
            [[int(round(x)), int(round(y))] for x, y in polygon] for polygon in pred_coors
        ]
        zero_mask = np.zeros((512, 512), dtype=np.uint8)
        self.yolo_binary_masks = [
            cv2.fillPoly(zero_mask.copy(), [np.array(coordinates)], 1)
            for coordinates in rounded_yolov8_coordinates
        ]
        return self.yolo_binary_masks

    def dice_coefficient(self, mask1, mask2):
        """Calculate the Dice coefficient between two masks."""
        intersection = np.logical_and(mask1, mask2)
        dice = (2. * intersection.sum()) / (mask1.sum() + mask2.sum())
        return dice

    def Run_dice(self, mask1, mask2):
        """Calculate the average Dice coefficient for multiple masks."""
        dice_scores = [self.dice_coefficient(m1, m2) for m1, m2 in zip(mask1, mask2)]
        return round(abs(np.mean(dice_scores)), 6)

    # ------------ ACCURACY, PRECISION, RECALL ------------ #
    def calculate_metrics(self, pred_mask, GT_mask):
        """Calculate accuracy, precision, and recall from masks."""
        TP = np.sum(np.logical_and(pred_mask, GT_mask))
        TN = np.sum(np.logical_and(~pred_mask, ~GT_mask))
        FP = np.sum(np.logical_and(pred_mask, ~GT_mask))
        FN = np.sum(np.logical_and(~pred_mask, GT_mask))

        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = 0.0 if TP + FP == 0 else TP / (TP + FP)
        recall = 0.0 if TP + FN == 0 else TP / (TP + FN)

        return round(abs(accuracy), 6), round(abs(precision), 6), round(abs(recall), 6)

    # ------------ VISUALIZATION ------------ #
    def visualize_Bboxes_masks(self, masks_list, titles):
        """Visualize bounding box masks and their histograms."""
        num_masks = len(masks_list[0])
        plt.figure(figsize=(12, 4 * num_masks))
        plt.suptitle(f'{titles[0]} vs {titles[1]}')

        for i in range(num_masks):
            plt.subplot(num_masks, 3, 3 * i + 1)
            plt.imshow(masks_list[0][i], cmap='gray')
            plt.title(f'Mask {i + 1} - {titles[0]}')

            plt.subplot(num_masks, 3, 3 * i + 2)
            plt.imshow(masks_list[1][i], cmap='gray')
            plt.title(f'Mask {i + 1} - {titles[1]}')

            plt.subplot(num_masks, 3, 3 * i + 3)
            plt.hist(masks_list[0][i].ravel(), bins=[-0.5, 0.5, 1.5], color='black', alpha=0.7)
            plt.title(f'Mask {i + 1} Histogram - {titles[0]}')

        plt.show()

if __name__ == "__main__":
    obj = MetricsRaiza()
