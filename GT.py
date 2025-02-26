from __future__ import print_function
import glob
import os
import os.path as osp
import sys
import shutil
import numpy as np
import imgviz
import labelme

try:
    import lxml.builder
    import lxml.etree
except ImportError:
    print("Please install lxml:\n\n    pip install lxml\n")
    sys.exit(1)

class GtGenerator:
    def __init__(self):
        """Initialize directories for detection and segmentation outputs."""
        self.detect_dir = os.path.join('./dataset/Sorted Data/GT/', 'detection/')  # Detection Output
        self.segmetation_dir = os.path.join('./dataset/Sorted Data/GT/', 'segmentation/')  # Segmentation Output

    def detect_labelme(self, input_dir):
        """Generate detection ground truth images from JSON annotations."""
        shutil.rmtree(self.detect_dir, ignore_errors=True)
        os.makedirs(self.detect_dir)
        print(f"Folder: {self.detect_dir} overwritten")

        class_names = ('left normal', 'right normal')
        class_colors = {'left normal': (255, 255, 0), 'right normal': (255, 0, 255)}
        self.gt_list_dir = []

        for filename in glob.glob(osp.join(input_dir, "*.json")):
            label_file = labelme.LabelFile(filename=filename)
            base = osp.splitext(osp.basename(filename))[0]
            out_viz_file = osp.join(self.detect_dir, base + ".png")
            self.gt_list_dir.append(out_viz_file)

            img = labelme.utils.img_data_to_arr(label_file.imageData)
            bboxes, labels, colors = [], [], []

            for shape in label_file.shapes:
                class_name_sp = shape["label"]
                if class_name_sp not in class_names:
                    continue
                class_id = class_names.index(class_name_sp)
                sortingForObj = np.asarray(shape["points"])

                # Calculate bounding box coordinates
                xmin, xmax = sorted([np.min(sortingForObj[:, 0]), np.max(sortingForObj[:, 0])])
                ymin, ymax = sorted([np.min(sortingForObj[:, 1]), np.max(sortingForObj[:, 1])])

                bboxes.append([ymin, xmin, ymax, xmax])
                labels.append(class_id)
                colors.append(class_colors[class_name_sp])

            captions = [class_names[label] for label in labels]
            viz = imgviz.instances2rgb(
                image=img, labels=labels, bboxes=bboxes, captions=captions, font_size=15, colormap=colors
            )
            imgviz.io.imsave(out_viz_file, viz)

        print('Ground truth images generated:', self.gt_list_dir)

    def segmentation_labelme(self, input_dir):
        """Generate segmentation ground truth images from JSON annotations."""
        shutil.rmtree(self.segmetation_dir, ignore_errors=True)
        os.makedirs(self.segmetation_dir)
        print(f"Folder: {self.segmetation_dir} overwritten")

        class_names = {'background': 0, 'cancer': 1, 'mix': 2, 'warthin': 3}
        classes = tuple(class_names.keys())
        class_colors = np.array([(0, 0, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)])
        self.seg_Gt_listdir = []

        for filename in glob.glob(osp.join(input_dir, "*.json")):
            label_file = labelme.LabelFile(filename=filename)
            base = osp.splitext(osp.basename(filename))[0]
            out_viz_file = osp.join(self.segmetation_dir, base + ".png")
            self.seg_Gt_listdir.append(out_viz_file)

            img = labelme.utils.img_data_to_arr(label_file.imageData)
            shapes = [shape for shape in label_file.shapes if shape["label"] in class_names]
            cls, _ = labelme.utils.shapes_to_label(
                img_shape=img.shape, shapes=shapes, label_name_to_value=class_names
            )

            cls[cls == -1] = 0  # Set background pixels
            clsv = imgviz.label2rgb(
                cls, img, label_names=classes, font_size=15, loc="rb", colormap=class_colors
            )
            imgviz.io.imsave(out_viz_file, clsv)

        print('Segmentation ground truth images generated:', self.seg_Gt_listdir)

if __name__ == "__main__":
    obj = GtGenerator()
    input_dir = os.path.join('./dataset/', 'demo_test9/')
    # Uncomment the following lines to run the generation methods
    # obj.detect_labelme(input_dir)
    # obj.segmentation_labelme(input_dir)
