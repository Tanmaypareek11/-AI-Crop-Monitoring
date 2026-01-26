import torch
from torchvision import transforms, models
from PIL import Image
import sys

def main(image_path):
    # Remove surrounding quotes if present
    image_path = image_path.strip('"').strip("'")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = models.resnet50(pretrained=False)
    num_features = model.fc.in_features

    # Number of classes should match training model
    num_classes = 15  # Update if necessary
    model.fc = torch.nn.Linear(num_features, num_classes)
    model.load_state_dict(torch.load('models/disease_model.pth', map_location=device))
    model = model.to(device)
    model.eval()

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    img_t = transform(image)
    batch_t = torch.unsqueeze(img_t, 0).to(device)

    # Prediction
    with torch.no_grad():
        out = model(batch_t)
        _, pred = torch.max(out, 1)

    # Class names: update if different
    class_names = ['Aphid', 'Black Rust', 'Blast', 'Brown Rust', 'Common Root Rot', 'Fusarium Head Blight',
                   'Healthy', 'Leaf Blight', 'Mildew', 'Mite', 'Septoria', 'Smut', 'Stem fly', 'Tan spot', 'Yellow Rust']

    print(f"Predicted disease: {class_names[pred.item()]}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        image_path = input("Please enter the image path: ")
    else:
        image_path = sys.argv[1]
    main(image_path)

