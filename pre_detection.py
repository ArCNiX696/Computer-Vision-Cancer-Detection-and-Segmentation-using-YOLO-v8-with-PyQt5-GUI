from ultralytics import YOLO
from PIL import Image
import cv2 as cv


class DetModel:
    """Detection model using YOLO for image prediction."""

    def __init__(self):
        """Initialize the YOLO model with the specified weights."""
        self.model = YOLO('detection_best_final.pt')  # Load a custom model

    def predict(self, img_path):
        """Predict bounding boxes for the given image path.

        Args:
            img_path (str): Path to the input image.
        """
        results = self.model(source=img_path, show=False, conf=0.6, save=False)
        
        for r in results:
            self.pred_xyxy = r.boxes.xyxy.tolist()
            print('Predicted bounding boxes:', self.pred_xyxy)

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
        detector = DetModel()
        detector.predict(img_path)
    else:
        print("Usage: python script_name.py <image_path>")



