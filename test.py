from ultralytics import YOLO
from PIL import Image
import cv2 as cv
import numpy as np


class SegmModel:
    """Segmentation model using YOLO for image segmentation prediction."""

    def __init__(self):
        """Initialize the YOLO segmentation model with the specified weights."""
        self.model = YOLO('best_segmen_final.pt')  # Load a custom model

    def predict(self, img_path):
        """Predict segmentation masks for the given image path.

        Args:
            img_path (str): Path to the input image.
        """
        results = self.model(source=img_path, show=False, conf=0.5, save=False)

        for r in results:
            if r.masks is not None and r.masks.xy is not None:
                self.pred_xyxy = r.masks.xy
                print('Predicted masks:', self.pred_xyxy)
            else:
                self.pred_xyxy = np.array([[[0, 0], [0, 0]]], dtype=np.float32)
                print('No masks found. Default mask returned:', self.pred_xyxy)

            # Visualize predictions
            im_array = r.plot()
            im_resize = cv.resize(im_array, (600, 600))
            im_rgb = cv.cvtColor(im_resize, cv.COLOR_BGR2RGB)
            cv.imshow('Predicted', im_rgb)
            cv.waitKey(0)
            cv.destroyAllWindows()


if __name__ == "__main__":
    # Example execution if script is run directly
    import sys
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        segmenter = SegmModel()
        segmenter.predict(img_path)
    else:
        print("Usage: python script_name.py <image_path>")
