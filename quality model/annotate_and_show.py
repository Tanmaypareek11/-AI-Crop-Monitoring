import cv2
import torch

# Load YOLOv5 model and set confidence threshold
model = torch.hub.load('ultralytics/yolov5', 'custom', path='D:/SIH-2025/Quality model/yolov5/runs/train/exp2/weights/best.pt', force_reload=True)
model.conf = 0.5

# Get image file path from user input and strip any surrounding quotes
image_path = input("Enter the path to your image: ").strip('"')
img = cv2.imread(image_path)

if img is None:
    print(f"Could not load image at {image_path}")
    exit()

# Run inference and render bounding boxes on image
results = model(img)
results.render()

# Display the annotated image in a window
cv2.imshow('Predicted Wheat Quality', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
