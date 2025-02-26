from ultralytics import YOLO


def main():
    model = YOLO('yolov8n.yaml').load('yolov8m.pt')

    # Train the model
    results = model.train(data='data.yaml', epochs=1000, imgsz=512,batch=8)


if __name__=='__main__':
    main()
